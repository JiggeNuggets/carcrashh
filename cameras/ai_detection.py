"""Optional Ultralytics YOLO/ByteTrack adapter.

The module imports Ultralytics lazily so camera streaming remains available when
the optional AI dependency or model weights are not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from django.conf import settings


TRAFFIC_CLASS_IDS = (0, 2, 3, 5, 7)
TRAFFIC_CLASS_NAMES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASSES = frozenset({"car", "motorcycle", "bus", "truck"})


class DetectorUnavailable(RuntimeError):
    """Raised when the optional inference runtime or weights are unavailable."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int | None = None

    def as_dict(self):
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": [round(value, 1) for value in self.bbox],
            "track_id": self.track_id,
        }


class UltralyticsDetector:
    """One model and persistent ByteTrack state owned by one AI worker."""

    def __init__(self, model_path=None, device=None):
        configured_path = model_path or getattr(
            settings,
            "CAMERA_AI_MODEL_PATH",
            Path(settings.BASE_DIR) / "models" / "yolo11n.pt",
        )
        path = Path(configured_path)
        allow_download = bool(getattr(settings, "CAMERA_AI_ALLOW_MODEL_DOWNLOAD", False))
        if not path.exists() and not allow_download:
            raise DetectorUnavailable(
                "YOLO model weights are not installed. Place yolo11n.pt in the configured models directory."
            )
        model_source = str(path if path.exists() else path.name)
        try:
            from ultralytics import YOLO
        except (ImportError, OSError) as exc:
            raise DetectorUnavailable(
                "Ultralytics/PyTorch is unavailable. Install the optional AI requirements."
            ) from exc

        self.device = device or getattr(settings, "CAMERA_AI_DEVICE", "auto")
        if self.device == "auto":
            try:
                import torch

                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        try:
            self.model = YOLO(model_source)
        except Exception as exc:
            raise DetectorUnavailable("The configured YOLO model could not be loaded.") from exc

    def detect(self, frame, *, confidence=0.45, image_size=320):
        try:
            results = self.model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=float(confidence),
                imgsz=int(image_size),
                classes=list(TRAFFIC_CLASS_IDS),
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise DetectorUnavailable("YOLO inference failed; normal camera streaming is unaffected.") from exc
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        ids_tensor = getattr(boxes, "id", None)
        ids = ids_tensor.int().cpu().tolist() if ids_tensor is not None else [None] * len(xyxy)
        names = getattr(result, "names", TRAFFIC_CLASS_NAMES)
        detections = []
        for coordinates, score, class_value, track_id in zip(xyxy, confidences, classes, ids):
            class_id = int(class_value)
            label = str(names.get(class_id, TRAFFIC_CLASS_NAMES.get(class_id, class_id)))
            detections.append(
                Detection(
                    class_id=class_id,
                    label=label,
                    confidence=float(score),
                    bbox=tuple(float(value) for value in coordinates),
                    track_id=int(track_id) if track_id is not None else None,
                )
            )
        return detections

    def close(self):
        self.model = None
        if str(self.device).startswith("cuda"):
            try:
                import torch

                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
