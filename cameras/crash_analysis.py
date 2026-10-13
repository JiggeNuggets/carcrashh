"""Conservative temporal heuristics for suspected collision candidates.

Scores are uncalibrated heuristic suspicion values, not real-world crash
probabilities. Human review and representative labeled-footage evaluation are
required before any operational use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .ai_detection import VEHICLE_CLASSES
from .tracking import bbox_center, bbox_iou


@dataclass(frozen=True)
class CrashEvent:
    timestamp: float
    suspicion_score: float
    severity: str
    vehicle_categories: list[str]
    track_ids: list[int]
    indicators: dict


def _distance(left, right):
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _vector_angle(left, right):
    left_size = math.hypot(*left)
    right_size = math.hypot(*right)
    if left_size < 0.001 or right_size < 0.001:
        return 0.0
    cosine = max(-1.0, min(1.0, (left[0] * right[0] + left[1] * right[1]) / (left_size * right_size)))
    return math.degrees(math.acos(cosine))


class CrashAnalyzer:
    def __init__(self, threshold=0.75, cooldown_seconds=60, processing_fps=2):
        self.threshold = float(threshold)
        self.cooldown_seconds = float(cooldown_seconds)
        self.processing_fps = max(1.0, float(processing_fps))
        self.pair_evidence = {}
        self.candidate_until = {}
        self.motion_signal_until = {}
        self.abnormal_signal_until = {}
        self.last_event_at = None
        self.latest_score = 0.0
        self.latest_indicators = {}

    def update_config(self, *, threshold, cooldown_seconds, processing_fps):
        self.threshold = float(threshold)
        self.cooldown_seconds = float(cooldown_seconds)
        self.processing_fps = max(1.0, float(processing_fps))

    def _update_pair_evidence(self, vehicles, timestamp):
        active_candidates = []
        observed_pairs = set()
        for index, left in enumerate(vehicles):
            for right in vehicles[index + 1 :]:
                key = tuple(sorted((left.object_id, right.object_id)))
                observed_pairs.add(key)
                current_distance = _distance(bbox_center(left.bbox), bbox_center(right.bbox))
                left_width = max(1.0, left.bbox[2] - left.bbox[0])
                right_width = max(1.0, right.bbox[2] - right.bbox[0])
                close = current_distance < 0.8 * (left_width + right_width)
                overlap = bbox_iou(left.bbox, right.bbox)
                converging = False
                if len(left.history) >= 2 and len(right.history) >= 2:
                    previous_distance = _distance(left.history[-2][1], right.history[-2][1])
                    converging = previous_distance - current_distance > max(2.0, 0.03 * (left_width + right_width))
                evidence = self.pair_evidence.get(key, 0)
                # Overlap alone never increases evidence; trajectories must also converge.
                if converging and (close or overlap > 0.02):
                    evidence = min(8, evidence + 1)
                else:
                    evidence = max(0, evidence - 1)
                self.pair_evidence[key] = evidence
                if evidence >= 3:
                    self.candidate_until[key] = timestamp + 3.0
                if self.candidate_until.get(key, float("-inf")) >= timestamp:
                    active_candidates.append((key, evidence, overlap))
        for key in list(self.pair_evidence):
            if key not in observed_pairs:
                self.pair_evidence[key] = max(0, self.pair_evidence[key] - 1)
            if self.pair_evidence[key] == 0 and self.candidate_until.get(key, float("-inf")) < timestamp:
                self.pair_evidence.pop(key, None)
                self.candidate_until.pop(key, None)
        return active_candidates

    def analyze(self, tracks, timestamp: float, *, temporal_score=None, temporal_threshold=0.75):
        vehicles = [track for track in tracks if track.category in VEHICLE_CLASSES]
        candidates = self._update_pair_evidence(vehicles, timestamp)
        candidate_ids = {object_id for key, _, _ in candidates for object_id in key}

        sudden_ids = []
        abnormal_ids = []
        for track in vehicles:
            speed_drop = track.previous_speed > 15 and track.speed < track.previous_speed * 0.4
            direction_change = _vector_angle(track.previous_velocity, track.velocity) > 65 and track.speed > 8
            if speed_drop:
                self.motion_signal_until[track.object_id] = timestamp + 3.0
            if direction_change:
                self.abnormal_signal_until[track.object_id] = timestamp + 3.0
            if self.motion_signal_until.get(track.object_id, float("-inf")) >= timestamp:
                sudden_ids.append(track.object_id)
            if self.abnormal_signal_until.get(track.object_id, float("-inf")) >= timestamp:
                abnormal_ids.append(track.object_id)

        stationary_required = max(3, round(self.processing_fps * 1.5))
        stationary_ids = [
            track.object_id
            for track in vehicles
            if track.object_id in candidate_ids and track.stationary_frames >= stationary_required
        ]
        collision_candidate = bool(candidates)
        sudden_change = bool(candidate_ids.intersection(sudden_ids))
        abnormal_movement = bool(candidate_ids.intersection(abnormal_ids))
        stationary_after_impact = bool(stationary_ids)
        temporal_signal = (
            temporal_score is not None
            and float(temporal_score) >= float(temporal_threshold)
            and bool(vehicles)
        )
        score = 0.0
        if collision_candidate:
            score += 0.45
        if sudden_change:
            score += 0.20
        if abnormal_movement:
            score += 0.10
        if stationary_after_impact:
            score += 0.25
        # The learned score is supporting evidence, not an independent crash
        # confirmation. At the default threshold it still needs temporal
        # trajectory evidence to reach the incident threshold.
        if temporal_signal:
            score += 0.35
        if candidates and max(item[1] for item in candidates) >= 4:
            score += 0.05
        score = round(min(1.0, score), 3)
        indicators = {
            "trajectory_collision_candidate": collision_candidate,
            "sudden_motion_change": sudden_change,
            "abnormal_direction_change": abnormal_movement,
            "stationary_after_candidate_impact": stationary_after_impact,
            "candidate_track_pairs": [list(item[0]) for item in candidates],
            "stationary_track_ids": stationary_ids,
            "temporal_classifier_score": temporal_score,
            "temporal_classifier_signal": temporal_signal,
            "note": "Uncalibrated heuristic evidence; human verification required.",
        }
        self.latest_score = score
        self.latest_indicators = indicators
        if score < self.threshold:
            return None
        if self.last_event_at is not None and timestamp - self.last_event_at < self.cooldown_seconds:
            return None
        self.last_event_at = timestamp
        severity = "undetermined"
        if score >= 0.93:
            severity = "severe"
        elif score >= 0.84:
            severity = "moderate"
        elif score >= self.threshold:
            severity = "minor"
        involved_ids = sorted(candidate_ids)
        categories = sorted({track.category for track in vehicles if track.object_id in candidate_ids})
        return CrashEvent(timestamp, score, severity, categories, involved_ids, indicators)

    def reset(self):
        self.pair_evidence.clear()
        self.candidate_until.clear()
        self.motion_signal_until.clear()
        self.abnormal_signal_until.clear()
        self.last_event_at = None
