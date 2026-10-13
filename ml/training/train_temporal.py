"""Train, evaluate, checkpoint, and export the tracked-motion GRU classifier."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cameras.temporal_features import FEATURE_NAMES


class TemporalGRU(nn.Module):
    def __init__(self, input_size=len(FEATURE_NAMES), hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(0.2), nn.Linear(hidden_size, 1))

    def forward(self, features):
        output, _ = self.gru(features)
        return self.head(output[:, -1]).squeeze(-1)


@dataclass
class Window:
    features: np.ndarray
    label: int
    video_id: str
    timestamp: float
    heuristic_score: float


class WindowDataset(Dataset):
    def __init__(self, windows): self.windows = windows
    def __len__(self): return len(self.windows)
    def __getitem__(self, index):
        item = self.windows[index]
        return torch.from_numpy(item.features), torch.tensor(item.label, dtype=torch.float32), index


def load_windows(root, split, sequence_length, stride):
    windows = []
    for path in sorted(root.glob("*.npz")):
        item = np.load(path, allow_pickle=False)
        if str(item["split"]) != split:
            continue
        if list(item["feature_names"]) != list(FEATURE_NAMES):
            raise ValueError(f"Feature schema mismatch: {path}")
        features, labels = item["features"], item["labels"]
        times, heuristics = item["timestamps"], item["heuristic_scores"]
        video_id = str(item["video_id"])
        for end in range(sequence_length, len(features) + 1, stride):
            start = end - sequence_length
            windows.append(Window(
                features[start:end], int(labels[start:end].max()), video_id,
                float(times[end - 1]), float(heuristics[end - 1]),
            ))
    if not windows:
        raise ValueError(f"No {split} windows found in {root}")
    return windows


def predict(model, loader, device):
    model.eval(); scores, labels, indices = [], [], []
    with torch.no_grad():
        for features, target, index in loader:
            scores.extend(torch.sigmoid(model(features.to(device))).cpu().numpy().tolist())
            labels.extend(target.numpy().astype(int).tolist()); indices.extend(index.numpy().tolist())
    return np.asarray(scores), np.asarray(labels), np.asarray(indices)


def choose_threshold(scores, labels):
    best = (0.5, -1.0)
    for threshold in np.linspace(0.2, 0.9, 71):
        prediction = scores >= threshold
        _, _, f1, _ = precision_recall_fscore_support(labels, prediction, average="binary", zero_division=0)
        if f1 > best[1]: best = (float(threshold), float(f1))
    return best[0]


def metrics(scores, labels, threshold):
    prediction = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, average="binary", zero_division=0
    )
    total = max(1, len(labels))
    return {"threshold": threshold, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "accuracy": float((tp + tn) / total),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
            "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def operational_metrics(windows, scores, threshold):
    by_video = {}
    for window, score in zip(windows, scores): by_video.setdefault(window.video_id, []).append((window, score))
    delays, false_alarms, normal_minutes = [], 0, 0.0
    for values in by_video.values():
        positives = [item.timestamp for item, _ in values if item.label]
        detections = [item.timestamp for item, score in values if score >= threshold]
        if positives:
            if detections: delays.append(detections[0] - positives[0])
        else:
            false_alarms += sum(score >= threshold for _, score in values)
            normal_minutes += max(item.timestamp for item, _ in values) / 60.0
    return {"mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
            "detected_positive_videos": len(delays), "false_alarm_windows": int(false_alarms),
            "false_alarms_per_normal_minute": false_alarms / normal_minutes if normal_minutes else None}


def main():
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume model and optimizer state from OUTPUT/last.pt when present.",
    )
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = {name: load_windows(args.features, name, args.sequence_length, args.stride)
            for name in ("train", "val", "test")}
    for split, windows in sets.items():
        split_labels = {item.label for item in windows}
        if split_labels != {0, 1}:
            raise ValueError(f"{split} split must contain both crash and non-crash windows")
    loaders = {name: DataLoader(WindowDataset(items), batch_size=args.batch_size,
                                shuffle=name == "train", num_workers=args.num_workers,
                                pin_memory=device.type == "cuda")
               for name, items in sets.items()}
    positives = sum(item.label for item in sets["train"]); negatives = len(sets["train"]) - positives
    model = TemporalGRU().to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / max(1, positives)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )
    best_f1, patience, start_epoch = -1.0, 0, 1
    history = []
    last_path = args.output / "last.pt"
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint.get("best_f1", -1.0))
        patience = int(checkpoint.get("patience", 0))
        history = list(checkpoint.get("history", []))
        print(json.dumps({"resumed_from_epoch": start_epoch - 1, "device": str(device)}))
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); running = 0.0
        for features, target, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features.to(device)), target.to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            running += loss.item() * len(target)
        val_scores, val_labels, _ = predict(model, loaders["val"], device)
        threshold = choose_threshold(val_scores, val_labels)
        val_metrics = metrics(val_scores, val_labels, threshold)
        scheduler.step(val_metrics["f1"])
        record = {"epoch": epoch, "train_loss": running / len(sets["train"]),
                  "learning_rate": optimizer.param_groups[0]["lr"], **val_metrics}
        history.append(record); print(json.dumps(record))
        if val_metrics["f1"] > best_f1:
            best_f1, patience = val_metrics["f1"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "threshold": threshold}, args.output / "best.pt")
        else:
            patience += 1
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "record": record, "best_f1": best_f1,
            "patience": patience, "history": history,
        }, last_path)
        if patience >= 7: break
    if not (args.output / "best.pt").is_file():
        raise RuntimeError("Training produced no best checkpoint.")
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"]); threshold = float(checkpoint["threshold"])
    test_scores, test_labels, _ = predict(model, loaders["test"], device)
    report = {"device": str(device), "epochs_completed": int(history[-1]["epoch"]),
              "run_duration_seconds": time.perf_counter() - run_started,
              "window_counts": {k: len(v) for k, v in sets.items()},
              "window_class_counts": {
                  k: {"crash": int(sum(item.label for item in v)),
                      "non_crash": int(len(v) - sum(item.label for item in v))}
                  for k, v in sets.items()
              },
              "learned_model": metrics(test_scores, test_labels, threshold),
              "operational": operational_metrics(sets["test"], test_scores, threshold)}
    baseline = np.asarray([item.heuristic_score for item in sets["test"]])
    report["milestone3_heuristic_baseline"] = metrics(baseline, test_labels, 0.75)
    with (args.output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "video_id", "timestamp", "label", "model_score", "model_prediction",
            "heuristic_score", "heuristic_prediction",
        ))
        writer.writeheader()
        for window, model_score in zip(sets["test"], test_scores):
            writer.writerow({
                "video_id": window.video_id, "timestamp": window.timestamp,
                "label": window.label, "model_score": float(model_score),
                "model_prediction": int(model_score >= threshold),
                "heuristic_score": window.heuristic_score,
                "heuristic_prediction": int(window.heuristic_score >= 0.75),
            })
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    model.eval().cpu(); dummy = torch.zeros(1, args.sequence_length, len(FEATURE_NAMES))
    torch.onnx.export(model, dummy, args.output / "crash_temporal.onnx",
                      input_names=["features"], output_names=["logit"],
                      opset_version=18, dynamo=False)
    metadata = {"sequence_length": args.sequence_length, "feature_names": list(FEATURE_NAMES),
                "decision_threshold": threshold, "output": "logit", "sample_fps": 2}
    (args.output / "crash_temporal.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
