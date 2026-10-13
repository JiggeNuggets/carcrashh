"""Fast synthetic end-to-end validation for the temporal training toolchain.

Synthetic data verifies mechanics only. Its metrics must never be reported as
real crash-detection performance.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cameras.temporal_features import FEATURE_NAMES
from ml.training.train_temporal import TemporalGRU


def create_features(root: Path, sequence_length: int, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for split_index, split in enumerate(("train", "val", "test")):
        for video_index, positive in enumerate((0, 1, 0, 1)):
            sample_count = sequence_length + 8
            features = rng.normal(0.05, 0.02, (sample_count, len(FEATURE_NAMES))).astype(np.float32)
            labels = np.zeros(sample_count, dtype=np.int64)
            if positive:
                labels[-6:] = 1
                features[-6:, 2:6] += 0.8
                features[-6:, 9:13] += 0.6
            video_id = f"synthetic_{split}_{video_index}"
            np.savez_compressed(
                root / f"{video_id}.npz",
                features=features,
                labels=labels,
                timestamps=np.arange(sample_count, dtype=np.float32) / 2,
                heuristic_scores=np.where(labels, 0.8, 0.1).astype(np.float32),
                video_id=video_id,
                split=split,
                group_id=f"synthetic_group_{split_index}_{video_index}",
                feature_names=np.asarray(FEATURE_NAMES),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "ml" / "smoke_output")
    parser.add_argument("--sequence-length", type=int, default=4)
    args = parser.parse_args()
    feature_root = args.output / "features"
    run_root = args.output / "run"
    create_features(feature_root, args.sequence_length)

    command = [
        sys.executable, str(PROJECT_ROOT / "ml" / "training" / "train_temporal.py"),
        "--features", str(feature_root), "--output", str(run_root),
        "--sequence-length", str(args.sequence_length), "--stride", "2",
        "--epochs", "2", "--batch-size", "8", "--num-workers", "0",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    resume_command = list(command)
    resume_command[resume_command.index("--epochs") + 1] = "3"
    resume_command.append("--resume")
    subprocess.run(resume_command, cwd=PROJECT_ROOT, check=True)
    recovered = torch.load(run_root / "last.pt", map_location="cpu", weights_only=True)
    if int(recovered["epoch"]) != 3:
        raise AssertionError("Checkpoint recovery did not resume at the next epoch.")

    checkpoint = torch.load(run_root / "best.pt", map_location="cpu", weights_only=True)
    model = TemporalGRU()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    sample = np.zeros((1, args.sequence_length, len(FEATURE_NAMES)), dtype=np.float32)
    with torch.no_grad():
        torch_logit = float(model(torch.from_numpy(sample)).item())
    session = ort.InferenceSession(str(run_root / "crash_temporal.onnx"), providers=["CPUExecutionProvider"])
    onnx_logit = float(session.run(None, {session.get_inputs()[0].name: sample})[0].reshape(-1)[0])
    difference = abs(torch_logit - onnx_logit)
    if difference > 1e-5:
        raise AssertionError(f"PyTorch/ONNX output mismatch: {difference}")

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "crash_monitor.settings")
    import django
    django.setup()
    from django.test import override_settings
    from cameras.temporal_classifier import TemporalCrashClassifier, create_temporal_classifier

    classifier = TemporalCrashClassifier(run_root / "crash_temporal.onnx")
    score = None
    for _ in range(args.sequence_length):
        score = classifier.update([], (64, 64, 3))
    classifier.close()
    if score is None or not 0.0 <= score <= 1.0:
        raise AssertionError("Django temporal classifier did not return a valid score.")
    with override_settings(CAMERA_AI_TEMPORAL_MODEL_PATH=""):
        if create_temporal_classifier() is not None:
            raise AssertionError("Disabled-model fallback did not return None.")

    result = {
        "scope": "synthetic functional validation only",
        "checkpoint_saved_and_loaded": True,
        "checkpoint_recovery_epoch": int(recovered["epoch"]),
        "validation_loop_completed": True,
        "onnx_runtime_loaded": True,
        "pytorch_logit": torch_logit,
        "onnx_logit": onnx_logit,
        "absolute_difference": difference,
        "django_score": score,
        "missing_model_fallback": True,
        "artifacts": str(run_root),
    }
    (args.output / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
