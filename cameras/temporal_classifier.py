"""Optional ONNX classifier for temporal crash evidence."""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np
from django.conf import settings

from .temporal_features import FEATURE_NAMES, scene_motion_features


class TemporalClassifierUnavailable(RuntimeError):
    pass


class TemporalCrashClassifier:
    def __init__(self, model_path, *, sequence_length=12):
        path = Path(model_path)
        if not path.is_file():
            raise TemporalClassifierUnavailable("Configured temporal crash model was not found.")
        metadata_path = path.with_suffix(".json")
        metadata = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError) as exc:
                raise TemporalClassifierUnavailable("Temporal model metadata is invalid.") from exc
        expected_features = metadata.get("feature_names", list(FEATURE_NAMES))
        if expected_features != list(FEATURE_NAMES):
            raise TemporalClassifierUnavailable("Temporal model feature schema does not match this application.")
        self.sequence_length = int(metadata.get("sequence_length", sequence_length))
        self.decision_threshold = float(metadata.get("decision_threshold", 0.75))
        self.output_kind = metadata.get("output", "score")
        if self.sequence_length < 2 or self.sequence_length > 120:
            raise TemporalClassifierUnavailable("Temporal model sequence length is invalid.")
        if not 0.0 < self.decision_threshold < 1.0:
            raise TemporalClassifierUnavailable("Temporal model decision threshold is invalid.")
        if self.output_kind not in {"score", "logit"}:
            raise TemporalClassifierUnavailable("Temporal model output type is invalid.")
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                str(path),
                providers=["CPUExecutionProvider"],
            )
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            raise TemporalClassifierUnavailable("ONNX temporal crash model could not be loaded.") from exc
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.features = collections.deque(maxlen=self.sequence_length)

    def update(self, tracks, frame_shape):
        self.features.append(scene_motion_features(tracks, frame_shape))
        if len(self.features) < self.sequence_length:
            return None
        batch = np.asarray(self.features, dtype=np.float32)[None, :, :]
        try:
            output = self.session.run([self.output_name], {self.input_name: batch})[0]
            value = float(np.asarray(output).reshape(-1)[0])
        except Exception as exc:
            raise TemporalClassifierUnavailable("Temporal classifier inference failed.") from exc
        if not math.isfinite(value):
            raise TemporalClassifierUnavailable("Temporal classifier produced a non-finite score.")
        if self.output_kind == "logit" or value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))
        return round(value, 4)

    def reset(self):
        self.features.clear()

    def close(self):
        self.features.clear()
        self.session = None


def create_temporal_classifier():
    configured = str(getattr(settings, "CAMERA_AI_TEMPORAL_MODEL_PATH", "")).strip()
    if not configured:
        return None
    return TemporalCrashClassifier(configured)
