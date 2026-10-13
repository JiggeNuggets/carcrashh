"""Recorded-video harness using the production detection/tracking/analyzer stack."""
from __future__ import annotations

import collections
import logging
import time
import uuid
from pathlib import Path

from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone

from .ai_detection import UltralyticsDetector
from .camera_service import STREAM_MAX_FPS, _encode_frames_to_clip
from .crash_analysis import CrashAnalyzer
from .models import CameraSnapshot, RecordedClip, SuspectedIncident
from .tracking import TrackManager
from .temporal_classifier import TemporalClassifierUnavailable, create_temporal_classifier

ALLOWED_VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})
logger = logging.getLogger(__name__)


def safe_test_video_path(value):
    root = Path(getattr(settings, "CAMERA_AI_TEST_VIDEO_ROOT")).resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Video must be inside the configured AI test-video directory.")
    if candidate.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError("Unsupported test-video file type.")
    if not candidate.is_file():
        raise ValueError("Test-video file does not exist.")
    return candidate


class RecordedVideoProcessor:
    def __init__(
        self,
        camera,
        detector_factory=UltralyticsDetector,
        temporal_classifier_factory=create_temporal_classifier,
    ):
        self.camera = camera
        self.detector_factory = detector_factory
        self.temporal_classifier_factory = temporal_classifier_factory
        self.capture = None
        self.detector = None
        self.temporal_classifier = None

    def _create_pending_incident(self, event, jpeg_bytes, pre_frames):
        incident = SuspectedIncident.objects.create(
            camera=self.camera,
            detected_at=timezone.now(),
            suspicion_score=event.suspicion_score,
            severity=event.severity,
            vehicle_categories=event.vehicle_categories,
            indicators={**event.indicators, "track_ids": event.track_ids},
            processing_status=SuspectedIncident.ProcessingStatus.RECORDING,
            source=SuspectedIncident.Source.RECORDED_VIDEO,
        )
        try:
            snapshot = CameraSnapshot(camera=self.camera, note=f"Recorded-video incident {incident.pk}")
            snapshot.image_file.save(
                f"ai_test_{self.camera.pk}_{uuid.uuid4().hex[:8]}.jpg",
                ContentFile(jpeg_bytes),
                save=True,
            )
            incident.snapshot = snapshot
            incident.save(update_fields=("snapshot",))
        except Exception:
            logger.exception("Recorded-video incident snapshot generation failed.")
            incident.media_error = "Recorded-video snapshot could not be generated."
            incident.save(update_fields=("media_error",))
        return {
            "incident": incident,
            "frames": list(pre_frames),
            "post_remaining": int(self.camera.ai_post_event_seconds * STREAM_MAX_FPS),
        }

    def _finalize_pending(self, pending):
        incident = pending["incident"]
        frames = pending["frames"]
        actual_pre = 0.0
        if len(frames) >= 2:
            actual_pre = min(15.0, frames[-1][0] - frames[0][0])
        try:
            clip = _encode_frames_to_clip(
                self.camera.pk,
                frames,
                actual_pre,
                float(self.camera.ai_post_event_seconds),
                RecordedClip.TriggerType.CRASH_DETECTED,
                f"Recorded-video suspected incident {str(incident.pk)[:8]}",
            )
        except Exception:
            logger.exception("Recorded-video incident clip generation failed.")
            clip = None
        if clip:
            incident.video_clip = clip
            incident.processing_status = (
                SuspectedIncident.ProcessingStatus.MEDIA_ERROR
                if incident.media_error
                else SuspectedIncident.ProcessingStatus.READY
            )
            incident.save(update_fields=("video_clip", "processing_status"))
        else:
            incident.processing_status = SuspectedIncident.ProcessingStatus.MEDIA_ERROR
            errors = [incident.media_error, "Recorded-video clip could not be generated."]
            incident.media_error = " ".join(dict.fromkeys(error for error in errors if error))[:255]
            incident.save(update_fields=("processing_status", "media_error"))

    def process(self, path, *, display=False, progress_callback=None):
        import cv2

        safe_path = safe_test_video_path(path)
        tracker = TrackManager()
        analyzer = CrashAnalyzer(
            threshold=self.camera.crash_suspicion_threshold,
            cooldown_seconds=self.camera.incident_cooldown_seconds,
            processing_fps=self.camera.ai_frame_rate,
        )
        self.detector = self.detector_factory()
        temporal_status = "disabled"
        try:
            self.temporal_classifier = self.temporal_classifier_factory()
            if self.temporal_classifier:
                temporal_status = "warming_up"
        except TemporalClassifierUnavailable as exc:
            logger.warning("Recorded-video temporal classifier unavailable: %s", exc)
            temporal_status = "unavailable"
        try:
            self.capture = cv2.VideoCapture(str(safe_path))
        except Exception:
            self.detector.close()
            self.detector = None
            if self.temporal_classifier is not None:
                self.temporal_classifier.close()
                self.temporal_classifier = None
            raise
        if not self.capture.isOpened():
            self.capture.release()
            self.detector.close()
            self.detector = None
            if self.temporal_classifier is not None:
                self.temporal_classifier.close()
                self.temporal_classifier = None
            raise ValueError("OpenCV could not open the test video.")
        source_fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
        source_fps = source_fps if 1 <= source_fps <= 240 else 25.0
        ai_stride = max(1, round(source_fps / max(1, self.camera.ai_frame_rate)))
        recording_stride = max(1, round(source_fps / STREAM_MAX_FPS))
        buffer_frames = collections.deque(
            maxlen=max(1, int(self.camera.buffer_seconds * STREAM_MAX_FPS))
        )
        pending_incidents = []
        frame_index = 0
        inference_count = 0
        incident_count = 0
        started = time.perf_counter()
        display_available = display
        try:
            while True:
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    break
                video_timestamp = frame_index / source_fps
                record = None
                if frame_index % recording_stride == 0:
                    encoded, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                    if encoded:
                        height, width = frame.shape[:2]
                        record = (video_timestamp, jpeg.tobytes(), width, height)
                        buffer_frames.append(record)
                        for pending in list(pending_incidents):
                            pending["frames"].append(record)
                            pending["post_remaining"] -= 1
                            if pending["post_remaining"] <= 0:
                                self._finalize_pending(pending)
                                pending_incidents.remove(pending)

                tracks = []
                if frame_index % ai_stride == 0:
                    detections = self.detector.detect(
                        frame,
                        confidence=self.camera.ai_confidence_threshold,
                        image_size=self.camera.ai_inference_size,
                    )
                    tracks = tracker.update(detections, video_timestamp)
                    temporal_score = None
                    if self.temporal_classifier:
                        try:
                            temporal_score = self.temporal_classifier.update(tracks, frame.shape)
                            temporal_status = "running" if temporal_score is not None else "warming_up"
                        except TemporalClassifierUnavailable as exc:
                            logger.warning("Recorded-video temporal classifier disabled: %s", exc)
                            self.temporal_classifier.close()
                            self.temporal_classifier = None
                            temporal_status = "unavailable"
                    event = analyzer.analyze(
                        tracks,
                        video_timestamp,
                        temporal_score=temporal_score,
                        temporal_threshold=(
                            self.temporal_classifier.decision_threshold
                            if self.temporal_classifier
                            else float(getattr(settings, "CAMERA_AI_TEMPORAL_THRESHOLD", 0.75))
                        ),
                    )
                    inference_count += 1
                    if event and record:
                        pending_incidents.append(
                            self._create_pending_incident(event, record[1], list(buffer_frames))
                        )
                        incident_count += 1
                    if progress_callback:
                        progress_callback(frame_index, detections, tracks, analyzer.latest_score)

                if display_available:
                    annotated = frame.copy()
                    for track in tracks:
                        x1, y1, x2, y2 = (int(value) for value in track.bbox)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (32, 220, 170), 2)
                        cv2.putText(
                            annotated,
                            f"#{track.object_id} {track.category} {track.confidence:.2f}",
                            (x1, max(18, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (32, 220, 170),
                            1,
                            cv2.LINE_AA,
                        )
                    try:
                        cv2.imshow("CrashGuard recorded-video AI test", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        display_available = False
                frame_index += 1
            for pending in pending_incidents:
                self._finalize_pending(pending)
            elapsed = max(0.0001, time.perf_counter() - started)
            return {
                "frames_read": frame_index,
                "inference_frames": inference_count,
                "incidents_created": incident_count,
                "elapsed_seconds": round(elapsed, 3),
                "processing_fps": round(frame_index / elapsed, 2),
                "temporal_classifier_status": temporal_status,
            }
        finally:
            if self.capture is not None:
                self.capture.release()
            if self.detector is not None:
                self.detector.close()
            if self.temporal_classifier is not None:
                self.temporal_classifier.close()
            if display:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
