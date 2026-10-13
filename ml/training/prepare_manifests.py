"""Build video-disjoint CCD or DoTA manifests without inventing labels."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path


def split_for_group(group_id, seed="crashguard-v1"):
    value = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "val" if value < 90 else "test"


def write_manifest(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "video_id", "source", "media_path", "group_id", "split", "fps",
        "event_start_frame", "event_end_frame", "is_positive",
    )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        key = (row["split"], row["is_positive"])
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"rows": len(rows), "counts": {str(k): v for k, v in counts.items()}}, indent=2))


def parse_ccd_annotation(line):
    match = re.match(r"^\s*([^,\s]+)\s*,?\s*(\[[01,\s]+\])\s*,?\s*(.*)$", line)
    if not match:
        raise ValueError("Unrecognized CCD annotation line")
    video_id, labels_text, remainder = match.groups()
    labels = [int(value) for value in ast.literal_eval(labels_text)]
    if len(labels) != 50 or any(value not in (0, 1) for value in labels):
        raise ValueError(f"CCD {video_id} does not contain 50 binary frame labels")
    fields = [value.strip() for value in re.split(r"[,\s]+", remainder) if value.strip()]
    youtube_id = fields[1] if len(fields) > 1 else video_id
    positive = [index for index, value in enumerate(labels) if value]
    return video_id, youtube_id, min(positive), max(positive)


def build_ccd(root):
    videos = root / "videos"
    annotation_path = videos / "Crash-1500.txt"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing official CCD annotations: {annotation_path}")
    rows = []
    for number, line in enumerate(annotation_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            video_id, youtube_id, event_start, event_end = parse_ccd_annotation(line)
        except ValueError as exc:
            raise ValueError(f"{annotation_path}:{number}: {exc}") from exc
        media = videos / "Crash-1500" / f"{video_id}.mp4"
        if not media.is_file():
            raise FileNotFoundError(f"Missing CCD video referenced by annotations: {media}")
        rows.append({
            "video_id": f"ccd_crash_{video_id}", "source": "ccd", "media_path": str(media),
            "group_id": f"youtube_{youtube_id}", "split": split_for_group(youtube_id), "fps": 10,
            "event_start_frame": event_start, "event_end_frame": event_end, "is_positive": 1,
        })
    for media in sorted((videos / "Normal").glob("*.mp4")):
        group = f"normal_{media.stem}"
        rows.append({
            "video_id": f"ccd_normal_{media.stem}", "source": "ccd", "media_path": str(media),
            "group_id": group, "split": split_for_group(group), "fps": 10,
            "event_start_frame": -1, "event_end_frame": -1, "is_positive": 0,
        })
    return rows


def build_dota(root):
    annotation_root = root / "annotations"
    frame_root = root / "frames"
    rows = []
    for annotation_path in sorted(annotation_root.glob("*.json")):
        item = json.loads(annotation_path.read_text(encoding="utf-8"))
        if item.get("ignore"):
            continue
        video_id = item["video_name"]
        positive_frames = [
            int(label["frame_id"])
            for label in item.get("labels", [])
            if str(label.get("accident_id", "0")) != "0"
        ]
        if not positive_frames:
            raise ValueError(f"DoTA annotation has no positive frame labels: {annotation_path}")
        anomaly_start, anomaly_end = min(positive_frames), max(positive_frames)
        if positive_frames != list(range(anomaly_start, anomaly_end + 1)):
            raise ValueError(f"DoTA positive frame labels are not contiguous: {annotation_path}")
        if not 0 <= anomaly_start <= anomaly_end < int(item["num_frames"]):
            raise ValueError(f"DoTA anomaly boundaries are invalid: {annotation_path}")
        media = frame_root / video_id
        if not media.is_dir():
            raise FileNotFoundError(f"Missing DoTA frame directory: {media}")
        group = video_id.rsplit("_", 1)[0]
        rows.append({
            "video_id": f"dota_{video_id}", "source": "dota", "media_path": str(media),
            "group_id": f"youtube_{group}", "split": split_for_group(group), "fps": 10,
            "event_start_frame": anomaly_start,
            "event_end_frame": anomaly_end, "is_positive": 1,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("ccd", "dota"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_ccd(args.root) if args.source == "ccd" else build_dota(args.root)
    write_manifest(rows, args.output)


if __name__ == "__main__":
    main()
