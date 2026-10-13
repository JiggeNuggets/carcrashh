"""CPU-cheap scene features shared by temporal crash model inference.

The classifier consumes tracked motion rather than raw pixels. This keeps the
custom model distinct from YOLO while reusing detections already produced by
the camera AI worker.
"""
from __future__ import annotations

import math

import numpy as np

from .ai_detection import VEHICLE_CLASSES
from .tracking import bbox_center, bbox_iou


FEATURE_NAMES = (
    "vehicle_count",
    "pedestrian_count",
    "mean_speed",
    "max_speed",
    "mean_acceleration",
    "max_acceleration",
    "stationary_ratio",
    "mean_vehicle_area",
    "max_vehicle_area",
    "max_pair_iou",
    "minimum_pair_distance",
    "converging_pair_ratio",
    "opposing_motion_ratio",
    "mean_confidence",
    "mean_track_age",
    "visible_track_ratio",
)


def _safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def scene_motion_features(tracks, frame_shape):
    """Return a normalized float32 feature vector for one sampled frame."""
    height, width = frame_shape[:2]
    diagonal = max(1.0, math.hypot(width, height))
    frame_area = max(1.0, float(width * height))
    tracks = list(tracks)
    vehicles = [track for track in tracks if track.category in VEHICLE_CLASSES]
    pedestrians = [track for track in tracks if track.category == "person"]

    speeds = [track.speed / diagonal for track in vehicles]
    accelerations = [abs(track.speed - track.previous_speed) / diagonal for track in vehicles]
    areas = [
        max(0.0, track.bbox[2] - track.bbox[0])
        * max(0.0, track.bbox[3] - track.bbox[1])
        / frame_area
        for track in vehicles
    ]
    pair_ious = []
    pair_distances = []
    converging = 0
    opposing = 0
    pair_count = 0
    for index, left in enumerate(vehicles):
        for right in vehicles[index + 1 :]:
            pair_count += 1
            pair_ious.append(bbox_iou(left.bbox, right.bbox))
            current_distance = math.dist(bbox_center(left.bbox), bbox_center(right.bbox))
            pair_distances.append(current_distance / diagonal)
            if len(left.history) >= 2 and len(right.history) >= 2:
                previous_distance = math.dist(left.history[-2][1], right.history[-2][1])
                converging += int(previous_distance > current_distance)
            dot = left.velocity[0] * right.velocity[0] + left.velocity[1] * right.velocity[1]
            opposing += int(dot < 0 and left.speed > 5 and right.speed > 5)

    active_count = sum(track.status == "active" for track in tracks)
    values = (
        min(len(vehicles) / 12.0, 1.0),
        min(len(pedestrians) / 8.0, 1.0),
        min(_safe_mean(speeds), 2.0),
        min(max(speeds, default=0.0), 3.0),
        min(_safe_mean(accelerations), 2.0),
        min(max(accelerations, default=0.0), 3.0),
        _safe_mean([float(track.stationary_frames > 0) for track in vehicles]),
        min(_safe_mean(areas), 1.0),
        min(max(areas, default=0.0), 1.0),
        max(pair_ious, default=0.0),
        min(min(pair_distances, default=1.0), 1.0),
        converging / pair_count if pair_count else 0.0,
        opposing / pair_count if pair_count else 0.0,
        _safe_mean([track.confidence for track in tracks]),
        min(_safe_mean([track.consecutive_frames for track in tracks]) / 30.0, 1.0),
        active_count / len(tracks) if tracks else 0.0,
    )
    return np.asarray(values, dtype=np.float32)
