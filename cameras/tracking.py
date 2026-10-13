"""Tracking-state bookkeeping layered on YOLO's persistent ByteTrack IDs."""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field

from .ai_detection import Detection


def bbox_iou(left, right):
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


@dataclass
class TrackState:
    object_id: int
    category: str
    bbox: tuple[float, float, float, float]
    confidence: float
    history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=30))
    velocity: tuple[float, float] = (0.0, 0.0)
    previous_velocity: tuple[float, float] = (0.0, 0.0)
    direction: str = "stationary"
    consecutive_frames: int = 0
    missed_frames: int = 0
    stationary_frames: int = 0
    status: str = "active"

    @property
    def speed(self):
        return math.hypot(*self.velocity)

    @property
    def previous_speed(self):
        return math.hypot(*self.previous_velocity)

    def update(self, detection: Detection, timestamp: float):
        center = bbox_center(detection.bbox)
        self.previous_velocity = self.velocity
        if self.history:
            previous_time, previous_center = self.history[-1]
            elapsed = max(0.001, timestamp - previous_time)
            self.velocity = (
                (center[0] - previous_center[0]) / elapsed,
                (center[1] - previous_center[1]) / elapsed,
            )
        self.history.append((timestamp, center))
        self.category = detection.label
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.consecutive_frames += 1
        self.missed_frames = 0
        self.status = "active"
        self.stationary_frames = self.stationary_frames + 1 if self.speed < 5.0 else 0
        dx, dy = self.velocity
        if self.speed < 5.0:
            self.direction = "stationary"
        elif abs(dx) >= abs(dy):
            self.direction = "right" if dx > 0 else "left"
        else:
            self.direction = "down" if dy > 0 else "up"

    def as_dict(self):
        return {
            "object_id": self.object_id,
            "category": self.category,
            "bbox": [round(value, 1) for value in self.bbox],
            "confidence": round(self.confidence, 4),
            "position_history": [[round(x, 1), round(y, 1)] for _, (x, y) in self.history],
            "direction": self.direction,
            "velocity_px_s": [round(value, 1) for value in self.velocity],
            "consecutive_frames": self.consecutive_frames,
            "tracking_status": self.status,
        }


class TrackManager:
    """Maintains temporal history and provides IoU IDs if ByteTrack has none."""

    def __init__(self, max_lost_frames=10, fallback_iou_threshold=0.25):
        self.max_lost_frames = max_lost_frames
        self.fallback_iou_threshold = fallback_iou_threshold
        self.tracks: dict[int, TrackState] = {}
        self.next_id = 1

    def _assign_fallback_id(self, detection, used_ids):
        candidates = [
            (bbox_iou(detection.bbox, track.bbox), object_id)
            for object_id, track in self.tracks.items()
            if object_id not in used_ids and track.category == detection.label
        ]
        if candidates:
            overlap, object_id = max(candidates)
            if overlap >= self.fallback_iou_threshold:
                return object_id
        object_id = self.next_id
        self.next_id += 1
        return object_id

    def update(self, detections: list[Detection], timestamp: float):
        used_ids = set()
        active = []
        for detection in detections:
            object_id = detection.track_id
            if object_id is None or object_id in used_ids:
                object_id = self._assign_fallback_id(detection, used_ids)
            self.next_id = max(self.next_id, object_id + 1)
            track = self.tracks.get(object_id)
            if track is None:
                track = TrackState(
                    object_id=object_id,
                    category=detection.label,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                )
                self.tracks[object_id] = track
            track.update(detection, timestamp)
            used_ids.add(object_id)
            active.append(track)

        for object_id, track in list(self.tracks.items()):
            if object_id in used_ids:
                continue
            track.missed_frames += 1
            track.status = "temporarily_lost"
            if track.missed_frames > self.max_lost_frames:
                track.status = "expired"
                del self.tracks[object_id]
        return active

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
