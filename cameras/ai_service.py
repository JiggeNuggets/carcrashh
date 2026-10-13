"""Bounded, non-blocking per-camera AI inference workers."""
from __future__ import annotations

import atexit
import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass

from django.conf import settings
from django.db import close_old_connections

from .ai_detection import DetectorUnavailable, UltralyticsDetector, VEHICLE_CLASSES
from .crash_analysis import CrashAnalyzer
from .tracking import TrackManager
from .temporal_classifier import TemporalClassifierUnavailable, create_temporal_classifier

logger = logging.getLogger(__name__)

AI_QUEUE_SIZE = 2


@dataclass
class AIState:
    status: str = "stopped"
    message: str = "AI detection is stopped."
    running: bool = False
    inference_fps: float = 0.0
    detected_objects: int = 0
    detected_vehicles: int = 0
    queue_depth: int = 0
    dropped_frames: int = 0
    sequence: int = 0
    latest_frame: bytes | None = None
    latest_tracks: list | None = None
    latest_suspicion_score: float = 0.0
    latest_temporal_score: float | None = None
    temporal_status: str = "disabled"
    latest_event: dict | None = None
    last_processed_at: float | None = None


class AIWorker:
    def __init__(
        self,
        camera,
        detector_factory=UltralyticsDetector,
        incident_handler=None,
        temporal_classifier_factory=create_temporal_classifier,
    ):
        self.camera_id = camera.pk
        self.detector_factory = detector_factory
        self.incident_handler = incident_handler
        self.temporal_classifier_factory = temporal_classifier_factory
        self.queue = queue.Queue(maxsize=AI_QUEUE_SIZE)
        self.condition = threading.Condition(threading.RLock())
        self.stop_event = threading.Event()
        self.thread = None
        self.detector = None
        self.temporal_classifier = None
        self.tracker = TrackManager()
        self.state = AIState(latest_tracks=[])
        self.inference_times = collections.deque(maxlen=20)
        self.last_enqueued_at = None
        self.update_camera_settings(camera)

    def update_camera_settings(self, camera):
        with self.condition:
            self.confidence_threshold = camera.ai_confidence_threshold
            self.frame_rate = camera.ai_frame_rate
            self.inference_size = camera.ai_inference_size
            self.crash_threshold = camera.crash_suspicion_threshold
            self.cooldown_seconds = camera.incident_cooldown_seconds
            self.post_event_seconds = camera.ai_post_event_seconds
            self.temporal_threshold = float(
                getattr(settings, "CAMERA_AI_TEMPORAL_THRESHOLD", 0.75)
            )
            if hasattr(self, "analyzer"):
                self.analyzer.update_config(
                    threshold=self.crash_threshold,
                    cooldown_seconds=self.cooldown_seconds,
                    processing_fps=self.frame_rate,
                )
            else:
                self.analyzer = CrashAnalyzer(
                    threshold=self.crash_threshold,
                    cooldown_seconds=self.cooldown_seconds,
                    processing_fps=self.frame_rate,
                )

    def start(self):
        with self.condition:
            if self.thread and self.thread.is_alive():
                return False
            self.stop_event.clear()
            self.state.running = True
            self.state.status = "loading"
            self.state.message = "Loading the AI detection model..."
            self.thread = threading.Thread(
                target=self._run,
                name=f"ai-worker-{self.camera_id}",
                daemon=True,
            )
            self.thread.start()
            return True

    def stop(self, wait=True):
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass
        with self.condition:
            self.condition.notify_all()
            thread = self.thread
        if wait and thread and thread is not threading.current_thread():
            thread.join(timeout=5)

    def submit(self, frame, timestamp):
        if self.stop_event.is_set() or not self.state.running:
            return False
        minimum_interval = 1.0 / max(1, self.frame_rate)
        if self.last_enqueued_at is not None and timestamp - self.last_enqueued_at < minimum_interval:
            return False
        self.last_enqueued_at = timestamp
        item = (timestamp, frame.copy())
        if self.queue.full():
            try:
                self.queue.get_nowait()
                with self.condition:
                    self.state.dropped_frames += 1
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(item)
            with self.condition:
                self.state.queue_depth = self.queue.qsize()
            return True
        except queue.Full:
            with self.condition:
                self.state.dropped_frames += 1
            return False

    def frames(self):
        with self.condition:
            last_sequence = self.state.sequence
        while not self.stop_event.is_set():
            with self.condition:
                self.condition.wait_for(
                    lambda: self.state.sequence != last_sequence
                    or not self.state.running
                    or self.stop_event.is_set(),
                    timeout=5,
                )
                frame = self.state.latest_frame
                sequence = self.state.sequence
                running = self.state.running
            if frame is not None and sequence != last_sequence:
                last_sequence = sequence
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            elif not running:
                break

    def snapshot(self):
        with self.condition:
            return {
                "status": self.state.status,
                "message": self.state.message,
                "running": self.state.running,
                "inference_fps": self.state.inference_fps,
                "detected_objects": self.state.detected_objects,
                "detected_vehicles": self.state.detected_vehicles,
                "queue_depth": self.state.queue_depth,
                "queue_capacity": self.queue.maxsize,
                "dropped_frames": self.state.dropped_frames,
                "latest_tracks": list(self.state.latest_tracks or []),
                "latest_suspicion_score": self.state.latest_suspicion_score,
                "latest_temporal_score": self.state.latest_temporal_score,
                "temporal_status": self.state.temporal_status,
                "latest_event": self.state.latest_event,
            }

    def _render_overlay(self, frame, tracks, score):
        import cv2

        annotated = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = (int(value) for value in track.bbox)
            color = (32, 220, 170) if track.category in VEHICLE_CLASSES else (70, 180, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"#{track.object_id} {track.category} {track.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            annotated,
            f"Crash suspicion (heuristic): {score:.2f}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 210, 255),
            2,
            cv2.LINE_AA,
        )
        encoded, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        return jpeg.tobytes() if encoded else None

    def _set_failure(self, status, message):
        with self.condition:
            self.state.status = status
            self.state.message = message
            self.state.running = False
            self.condition.notify_all()

    def _run(self):
        close_old_connections()
        try:
            try:
                self.detector = self.detector_factory()
            except DetectorUnavailable as exc:
                self._set_failure("unavailable", str(exc))
                return
            except Exception:
                logger.exception("AI model initialization failed.")
                self._set_failure("error", "AI model initialization failed; normal preview remains available.")
                return
            with self.condition:
                self.state.status = "running"
                self.state.message = "YOLO detection and ByteTrack tracking are active."
            try:
                self.temporal_classifier = self.temporal_classifier_factory()
                with self.condition:
                    self.state.temporal_status = (
                        "warming_up" if self.temporal_classifier else "disabled"
                    )
            except TemporalClassifierUnavailable as exc:
                logger.warning("Temporal classifier unavailable: %s", exc)
                with self.condition:
                    self.state.temporal_status = "unavailable"
            while not self.stop_event.is_set():
                try:
                    item = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    break
                timestamp, frame = item
                started = time.perf_counter()
                try:
                    detections = self.detector.detect(
                        frame,
                        confidence=self.confidence_threshold,
                        image_size=self.inference_size,
                    )
                    tracks = self.tracker.update(detections, timestamp)
                    temporal_score = None
                    if self.temporal_classifier:
                        try:
                            temporal_score = self.temporal_classifier.update(tracks, frame.shape)
                        except TemporalClassifierUnavailable as exc:
                            logger.warning("Temporal classifier disabled after inference error: %s", exc)
                            self.temporal_classifier.close()
                            self.temporal_classifier = None
                            with self.condition:
                                self.state.temporal_status = "unavailable"
                        else:
                            with self.condition:
                                self.state.temporal_status = (
                                    "running" if temporal_score is not None else "warming_up"
                                )
                    event = self.analyzer.analyze(
                        tracks,
                        timestamp,
                        temporal_score=temporal_score,
                        temporal_threshold=(
                            self.temporal_classifier.decision_threshold
                            if self.temporal_classifier
                            else self.temporal_threshold
                        ),
                    )
                    overlay = self._render_overlay(frame, tracks, self.analyzer.latest_score)
                    elapsed = max(0.0001, time.perf_counter() - started)
                    self.inference_times.append(elapsed)
                    inference_fps = len(self.inference_times) / sum(self.inference_times)
                    latest_event = self.state.latest_event
                    if event:
                        latest_event = {
                            "score": event.suspicion_score,
                            "severity": event.severity,
                            "track_ids": event.track_ids,
                            "detected_at_monotonic": event.timestamp,
                        }
                        if self.incident_handler:
                            try:
                                incident = self.incident_handler(self.camera_id, event)
                                latest_event["incident_id"] = str(incident.pk)
                            except Exception:
                                logger.exception("Suspected incident persistence failed.")
                    with self.condition:
                        self.state.status = "running"
                        self.state.message = "YOLO detection and ByteTrack tracking are active."
                        self.state.inference_fps = round(inference_fps, 2)
                        self.state.detected_objects = len(detections)
                        self.state.detected_vehicles = sum(
                            detection.label in VEHICLE_CLASSES for detection in detections
                        )
                        self.state.queue_depth = self.queue.qsize()
                        self.state.latest_tracks = [track.as_dict() for track in tracks]
                        self.state.latest_suspicion_score = self.analyzer.latest_score
                        self.state.latest_temporal_score = temporal_score
                        self.state.latest_event = latest_event
                        self.state.last_processed_at = timestamp
                        if overlay:
                            self.state.latest_frame = overlay
                            self.state.sequence += 1
                        self.condition.notify_all()
                except DetectorUnavailable as exc:
                    self._set_failure("unavailable", str(exc))
                    break
                except Exception:
                    logger.exception("AI frame processing failed.")
                    with self.condition:
                        self.state.status = "error"
                        self.state.message = "An AI frame failed; the worker will continue with newer frames."
                        self.state.queue_depth = self.queue.qsize()
                        self.condition.notify_all()
        finally:
            if self.detector:
                try:
                    self.detector.close()
                except Exception:
                    pass
            if self.temporal_classifier:
                try:
                    self.temporal_classifier.close()
                except Exception:
                    pass
            with self.condition:
                self.state.running = False
                if self.stop_event.is_set():
                    self.state.status = "stopped"
                    self.state.message = "AI detection is stopped."
                self.condition.notify_all()
            close_old_connections()


class AIRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.workers = {}

    def get(
        self,
        camera,
        *,
        detector_factory=UltralyticsDetector,
        incident_handler=None,
        temporal_classifier_factory=create_temporal_classifier,
    ):
        if incident_handler is None:
            from .incident_service import create_live_incident

            incident_handler = create_live_incident
        with self.lock:
            worker = self.workers.get(camera.pk)
            if worker is None:
                worker = AIWorker(
                    camera,
                    detector_factory=detector_factory,
                    incident_handler=incident_handler,
                    temporal_classifier_factory=temporal_classifier_factory,
                )
                self.workers[camera.pk] = worker
            else:
                worker.update_camera_settings(camera)
            return worker

    def sync(self, camera, allow_start=False):
        from .camera_service import should_autostart_workers

        if not camera.enabled or not camera.ai_enabled:
            self.stop(camera.pk)
            return None
        with self.lock:
            existing = self.workers.get(camera.pk)
        if existing is None and not (allow_start and should_autostart_workers()):
            return None
        worker = self.get(camera)
        worker.start()
        return worker

    def submit(self, camera_id, frame, timestamp):
        with self.lock:
            worker = self.workers.get(camera_id)
        return worker.submit(frame, timestamp) if worker else False

    def stop(self, camera_id):
        with self.lock:
            worker = self.workers.pop(camera_id, None)
        if worker:
            worker.stop()

    def stop_all(self):
        with self.lock:
            workers = list(self.workers.values())
            self.workers.clear()
        for worker in workers:
            worker.stop()

    def start_enabled_cameras(self):
        from .models import Camera

        close_old_connections()
        try:
            for camera in Camera.objects.filter(enabled=True, ai_enabled=True):
                self.get(camera).start()
        except Exception:
            logger.debug("AI camera startup skipped because the database is unavailable.")
        finally:
            close_old_connections()


ai_registry = AIRegistry()
atexit.register(ai_registry.stop_all)
