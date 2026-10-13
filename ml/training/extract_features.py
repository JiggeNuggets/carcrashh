"""Extract production-compatible tracked-motion features with pretrained YOLO."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cameras.ai_detection import Detection, TRAFFIC_CLASS_IDS, TRAFFIC_CLASS_NAMES
from cameras.crash_analysis import CrashAnalyzer
from cameras.temporal_features import FEATURE_NAMES, scene_motion_features
from cameras.tracking import TrackManager


def iter_frames(path, source_fps, sample_fps):
    stride = max(1, round(source_fps / sample_fps))
    if path.is_dir():
        files = sorted(path.glob("images/*.jpg")) or sorted(path.glob("*.jpg"))
        for index, image_path in enumerate(files):
            if index % stride == 0:
                frame = cv2.imread(str(image_path))
                if frame is not None:
                    yield index, frame
        return
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open {path}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index % stride == 0:
                yield index, frame
            index += 1
    finally:
        capture.release()


def parse_detections(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names
    return [
        Detection(
            class_id=int(class_id),
            label=str(names.get(int(class_id), TRAFFIC_CLASS_NAMES.get(int(class_id), class_id))),
            confidence=float(confidence),
            bbox=tuple(float(value) for value in coordinates),
        )
        for coordinates, confidence, class_id in zip(
            boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()
        )
    ]


def process_row(row, model, output_dir, sample_fps, image_size, confidence):
    tracker = TrackManager(max_lost_frames=max(2, sample_fps * 2))
    analyzer = CrashAnalyzer(processing_fps=sample_fps)
    path = Path(row["media_path"])
    source_fps = float(row["fps"])
    event_start = int(row["event_start_frame"])
    event_end = int(row["event_end_frame"])
    features, labels, timestamps, heuristic_scores = [], [], [], []
    for frame_index, frame in iter_frames(path, source_fps, sample_fps):
        result = model.predict(
            source=frame, conf=confidence, imgsz=image_size,
            classes=list(TRAFFIC_CLASS_IDS), device=0 if torch.cuda.is_available() else "cpu",
            verbose=False,
        )[0]
        timestamp = frame_index / source_fps
        tracks = tracker.update(parse_detections(result), timestamp)
        analyzer.analyze(tracks, timestamp)
        features.append(scene_motion_features(tracks, frame.shape))
        labels.append(int(event_start >= 0 and event_start <= frame_index <= event_end))
        timestamps.append(timestamp)
        heuristic_scores.append(analyzer.latest_score)
    if not features:
        raise ValueError("No readable frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{row['video_id']}.npz"
    np.savez_compressed(
        destination,
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        timestamps=np.asarray(timestamps, dtype=np.float32),
        heuristic_scores=np.asarray(heuristic_scores, dtype=np.float32),
        video_id=row["video_id"], split=row["split"], group_id=row["group_id"],
        feature_names=np.asarray(FEATURE_NAMES),
    )
    return len(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--sample-fps", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]
    model = YOLO(args.model)
    failures = []
    for index, row in enumerate(rows, 1):
        destination = args.output / f"{row['video_id']}.npz"
        if destination.is_file():
            continue
        try:
            count = process_row(
                row, model, args.output, args.sample_fps, args.image_size, args.confidence
            )
            print(f"[{index}/{len(rows)}] {row['video_id']}: {count} samples")
        except Exception as exc:
            failures.append({"video_id": row["video_id"], "error": str(exc)})
            print(f"[{index}/{len(rows)}] FAILED {row['video_id']}: {exc}")
    (args.output / "extraction_report.json").write_text(
        json.dumps({"requested": len(rows), "failures": failures}, indent=2), encoding="utf-8"
    )
    if failures:
        raise SystemExit(f"Feature extraction had {len(failures)} failure(s); inspect the report.")


if __name__ == "__main__":
    main()
