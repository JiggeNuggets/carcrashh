import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .ai_detection import Detection, DetectorUnavailable, UltralyticsDetector
from .ai_service import AIWorker, ai_registry
from .crash_analysis import CrashAnalyzer, CrashEvent
from .incident_service import create_live_incident
from .models import Camera, CameraSnapshot, RecordedClip, SuspectedIncident
from .tracking import TrackManager
from .video_testing import RecordedVideoProcessor
from .temporal_features import FEATURE_NAMES, scene_motion_features
from .temporal_classifier import (
    TemporalClassifierUnavailable,
    TemporalCrashClassifier,
    create_temporal_classifier,
)


def make_camera(**overrides):
    values = {
        "name": "AI test camera",
        "source_type": Camera.SourceType.USB,
        "usb_device_index": 0,
        "enabled": True,
        "buffer_enabled": True,
    }
    values.update(overrides)
    return Camera.objects.create(**values)


def vehicle(track_id, bbox, confidence=0.9, label="car"):
    return Detection(2, label, confidence, tuple(bbox), track_id)


class AIConfigurationTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user("ai-operator", password="test-password-123")
        self.client.force_login(self.user)

    def tearDown(self):
        ai_registry.stop_all()

    def test_ai_configuration_persists_without_source_retest(self):
        camera = make_camera(ai_enabled=False)
        response = self.client.post(reverse("cameras:edit", args=[camera.pk]), {
            "name": camera.name,
            "source_type": Camera.SourceType.USB,
            "usb_device_index": "0",
            "enabled": "on",
            "buffer_enabled": "on",
            "buffer_seconds": "20",
            "ai_enabled": "on",
            "ai_confidence_threshold": "0.55",
            "ai_frame_rate": "5",
            "ai_inference_size": "480",
            "crash_suspicion_threshold": "0.80",
            "incident_cooldown_seconds": "120",
            "ai_post_event_seconds": "15",
        })
        self.assertRedirects(response, reverse("cameras:list"))
        camera.refresh_from_db()
        self.assertTrue(camera.ai_enabled)
        self.assertEqual(camera.ai_confidence_threshold, 0.55)
        self.assertEqual(camera.ai_frame_rate, 5)
        self.assertEqual(camera.ai_inference_size, 480)
        self.assertEqual(camera.crash_suspicion_threshold, 0.8)
        self.assertEqual(camera.incident_cooldown_seconds, 120)
        self.assertEqual(camera.ai_post_event_seconds, 15)

    def test_ai_disabled_camera_has_no_inference_worker(self):
        camera = make_camera(ai_enabled=False)
        self.assertIsNone(ai_registry.sync(camera, allow_start=True))
        self.assertNotIn(camera.pk, ai_registry.workers)
        response = self.client.get(reverse("cameras:ai_status", args=[camera.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "disabled")


class DetectorAdapterTests(TestCase):
    def test_adapter_parses_mocked_ultralytics_bytetrack_result(self):
        class Tensor:
            def __init__(self, value):
                self.value = value

            def cpu(self):
                return self

            def int(self):
                return self

            def tolist(self):
                return self.value

        boxes = MagicMock()
        boxes.__len__.return_value = 1
        boxes.xyxy = Tensor([[10, 20, 110, 120]])
        boxes.conf = Tensor([0.91])
        boxes.cls = Tensor([2])
        boxes.id = Tensor([17])
        result = MagicMock(boxes=boxes, names={2: "car"})
        detector = UltralyticsDetector.__new__(UltralyticsDetector)
        detector.device = "cpu"
        detector.model = MagicMock()
        detector.model.track.return_value = [result]

        detections = detector.detect(np.zeros((160, 160, 3), dtype=np.uint8), confidence=0.5, image_size=320)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].track_id, 17)
        self.assertEqual(detections[0].label, "car")
        detector.model.track.assert_called_once()
        self.assertEqual(detector.model.track.call_args.kwargs["tracker"], "bytetrack.yaml")
        self.assertTrue(detector.model.track.call_args.kwargs["persist"])

    def test_missing_model_is_reported_without_download(self):
        with tempfile.TemporaryDirectory() as directory, override_settings(
            CAMERA_AI_MODEL_PATH=Path(directory) / "missing.pt",
            CAMERA_AI_ALLOW_MODEL_DOWNLOAD=False,
        ):
            with self.assertRaises(DetectorUnavailable):
                UltralyticsDetector()


class TrackingAndCrashAnalysisTests(TestCase):
    def test_tracking_updates_velocity_history_and_occlusion_state(self):
        tracker = TrackManager(max_lost_frames=2)
        first = tracker.update([vehicle(7, (0, 0, 20, 20))], 0.0)[0]
        second = tracker.update([vehicle(7, (10, 0, 30, 20))], 1.0)[0]
        self.assertEqual(first.object_id, second.object_id)
        self.assertEqual(second.direction, "right")
        self.assertAlmostEqual(second.velocity[0], 10.0)
        self.assertEqual(second.consecutive_frames, 2)
        tracker.update([], 2.0)
        self.assertEqual(tracker.tracks[7].status, "temporarily_lost")
        recovered = tracker.update([vehicle(7, (20, 0, 40, 20))], 3.0)[0]
        self.assertEqual(recovered.status, "active")
        self.assertEqual(recovered.consecutive_frames, 3)

    def _qualifying_sequence(self, analyzer):
        tracker = TrackManager()
        centers = ((30, 270), (90, 210), (110, 190), (128, 172), (135, 165))
        event = None
        for timestamp, (left_center, right_center) in enumerate(centers):
            tracks = tracker.update([
                vehicle(1, (left_center - 30, 0, left_center + 30, 60)),
                vehicle(2, (right_center - 30, 0, right_center + 30, 60), label="truck"),
            ], float(timestamp))
            event = analyzer.analyze(tracks, float(timestamp)) or event
        for timestamp in (5.0, 6.0, 7.0):
            tracks = tracker.update([
                vehicle(1, (105, 0, 165, 60)),
                vehicle(2, (135, 0, 195, 60), label="truck"),
            ], timestamp)
            event = analyzer.analyze(tracks, timestamp) or event
        return event, tracks

    def test_temporal_collision_sequence_emits_suspected_event(self):
        analyzer = CrashAnalyzer(threshold=0.75, cooldown_seconds=60, processing_fps=2)
        event, _tracks = self._qualifying_sequence(analyzer)
        self.assertIsNotNone(event)
        self.assertGreaterEqual(event.suspicion_score, 0.75)
        self.assertTrue(event.indicators["trajectory_collision_candidate"])
        self.assertTrue(
            event.indicators["sudden_motion_change"]
            or event.indicators["stationary_after_candidate_impact"]
        )
        self.assertEqual(set(event.track_ids), {1, 2})

    def test_overlap_without_convergence_is_suppressed(self):
        analyzer = CrashAnalyzer(threshold=0.5, cooldown_seconds=10, processing_fps=2)
        tracker = TrackManager()
        for timestamp in range(8):
            tracks = tracker.update([
                vehicle(1, (20, 20, 100, 100)),
                vehicle(2, (30, 30, 110, 110), label="truck"),
            ], float(timestamp))
            event = analyzer.analyze(tracks, float(timestamp))
            self.assertIsNone(event)
        self.assertFalse(analyzer.latest_indicators["trajectory_collision_candidate"])

    def test_cooldown_deduplicates_repeated_event(self):
        analyzer = CrashAnalyzer(threshold=0.75, cooldown_seconds=60, processing_fps=2)
        first, tracks = self._qualifying_sequence(analyzer)
        self.assertIsNotNone(first)
        duplicate = analyzer.analyze(tracks, 7.0)
        self.assertIsNone(duplicate)

    def test_temporal_model_is_supporting_not_standalone_evidence(self):
        analyzer = CrashAnalyzer(threshold=0.75, cooldown_seconds=60, processing_fps=2)
        tracker = TrackManager()
        event = None
        for timestamp in range(8):
            tracks = tracker.update([
                vehicle(1, (20, 20, 100, 100)),
                vehicle(2, (30, 30, 110, 110), label="truck"),
            ], float(timestamp))
            event = analyzer.analyze(
                tracks, float(timestamp), temporal_score=0.99, temporal_threshold=0.75
            ) or event
        self.assertIsNone(event)
        self.assertFalse(analyzer.latest_indicators["trajectory_collision_candidate"])
        self.assertTrue(analyzer.latest_indicators["temporal_classifier_signal"])

    def test_scene_motion_feature_schema_is_finite_and_stable(self):
        tracker = TrackManager()
        tracker.update([vehicle(1, (10, 10, 30, 30))], 0.0)
        tracks = tracker.update([vehicle(1, (20, 10, 40, 30))], 1.0)
        features = scene_motion_features(tracks, (100, 200, 3))
        self.assertEqual(features.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(features[2], 0.0)


class TemporalClassifierTests(TestCase):
    def test_disabled_model_uses_safe_fallback(self):
        with override_settings(CAMERA_AI_TEMPORAL_MODEL_PATH=""):
            self.assertIsNone(create_temporal_classifier())

    def test_missing_model_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.onnx"
            with self.assertRaises(TemporalClassifierUnavailable):
                TemporalCrashClassifier(missing)

    def test_schema_mismatch_is_rejected_before_runtime_load(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.onnx"
            model_path.write_bytes(b"placeholder")
            model_path.with_suffix(".json").write_text(
                '{"feature_names": ["wrong"], "sequence_length": 4}', encoding="utf-8"
            )
            with self.assertRaisesMessage(TemporalClassifierUnavailable, "feature schema"):
                TemporalCrashClassifier(model_path)


class IncidentIntegrationTests(TestCase):
    def test_incident_record_and_media_links_are_created(self):
        camera = make_camera(ai_enabled=True)
        snapshot = CameraSnapshot.objects.create(camera=camera, image_file="snapshots/evidence.jpg")
        clip = RecordedClip.objects.create(camera=camera, video_file="clips/evidence.mp4")
        event = CrashEvent(
            timestamp=12.0,
            suspicion_score=0.86,
            severity="moderate",
            vehicle_categories=["car", "truck"],
            track_ids=[4, 9],
            indicators={"trajectory_collision_candidate": True},
        )

        def complete_immediately(**kwargs):
            kwargs["on_complete"](clip, "")
            return MagicMock()

        with patch("cameras.incident_service.capture_snapshot", return_value=snapshot), patch(
            "cameras.incident_service.trigger_clip_capture", side_effect=complete_immediately
        ):
            incident = create_live_incident(camera.pk, event)
        incident.refresh_from_db()
        self.assertEqual(incident.snapshot, snapshot)
        self.assertEqual(incident.video_clip, clip)
        self.assertEqual(incident.processing_status, SuspectedIncident.ProcessingStatus.READY)
        self.assertEqual(incident.review_status, SuspectedIncident.ReviewStatus.PENDING)
        self.assertEqual(incident.indicators["track_ids"], [4, 9])

    def test_incident_pages_require_authentication(self):
        user = get_user_model().objects.create_user("reviewer", password="test-password-123")
        camera = make_camera()
        incident = SuspectedIncident.objects.create(
            camera=camera,
            detected_at=timezone.now(),
            suspicion_score=0.8,
        )
        urls = (reverse("cameras:incidents"), reverse("cameras:incident_detail", args=[incident.pk]))
        for url in urls:
            self.assertEqual(self.client.get(url).status_code, 302)
        self.client.force_login(user)
        for url in urls:
            self.assertEqual(self.client.get(url).status_code, 200)


class AIWorkerTests(TestCase):
    def tearDown(self):
        ai_registry.stop_all()

    def test_queue_is_bounded_and_drops_oldest_frame(self):
        camera = make_camera(ai_frame_rate=5)
        worker = AIWorker(camera, detector_factory=MagicMock)
        worker.state.running = True
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertTrue(worker.submit(frame, 0.0))
        self.assertTrue(worker.submit(frame, 1.0))
        self.assertTrue(worker.submit(frame, 2.0))
        self.assertEqual(worker.queue.qsize(), 2)
        self.assertEqual(worker.state.dropped_frames, 1)

    def test_worker_loads_detector_once_processes_and_stops_cleanly(self):
        camera = make_camera(ai_frame_rate=5)

        class FakeDetector:
            def __init__(self):
                self.detect_calls = 0
                self.closed = False

            def detect(self, frame, **kwargs):
                self.detect_calls += 1
                return []

            def close(self):
                self.closed = True

        detector = FakeDetector()
        factory = MagicMock(return_value=detector)
        worker = AIWorker(camera, detector_factory=factory, incident_handler=MagicMock())
        worker.start()
        worker.submit(np.zeros((32, 32, 3), dtype=np.uint8), 1.0)
        deadline = time.time() + 3
        while worker.state.sequence == 0 and time.time() < deadline:
            time.sleep(0.01)
        worker.stop()
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(detector.detect_calls, 1)
        self.assertTrue(detector.closed)
        self.assertFalse(worker.state.running)
        self.assertEqual(worker.state.status, "stopped")

    def test_unavailable_detector_does_not_break_worker_shutdown(self):
        camera = make_camera()

        def unavailable():
            raise DetectorUnavailable("model unavailable")

        worker = AIWorker(camera, detector_factory=unavailable)
        worker.start()
        worker.thread.join(timeout=2)
        self.assertEqual(worker.state.status, "unavailable")
        self.assertFalse(worker.state.running)

    def test_unavailable_temporal_model_does_not_interrupt_worker(self):
        camera = make_camera(ai_frame_rate=5)
        detector = MagicMock()
        detector.detect.return_value = []

        def unavailable_temporal():
            raise TemporalClassifierUnavailable("invalid model")

        worker = AIWorker(
            camera,
            detector_factory=MagicMock(return_value=detector),
            temporal_classifier_factory=unavailable_temporal,
        )
        worker.start()
        worker.submit(np.zeros((32, 32, 3), dtype=np.uint8), 1.0)
        deadline = time.time() + 3
        while worker.state.sequence == 0 and time.time() < deadline:
            time.sleep(0.01)
        worker.stop()
        self.assertEqual(detector.detect.call_count, 1)
        self.assertEqual(worker.state.temporal_status, "unavailable")
        self.assertEqual(worker.state.status, "stopped")


class RecordedVideoTests(TestCase):
    def test_recorded_video_eof_releases_capture_and_detector(self):
        camera = make_camera(ai_frame_rate=2)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25.0
        capture.read.side_effect = [(True, frame), (False, None)]
        detector = MagicMock()
        detector.detect.return_value = []

        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "traffic.mp4"
            video_path.write_bytes(b"test-placeholder")
            with override_settings(CAMERA_AI_TEST_VIDEO_ROOT=Path(directory)), patch(
                "cv2.VideoCapture", return_value=capture
            ):
                result = RecordedVideoProcessor(
                    camera, detector_factory=MagicMock(return_value=detector)
                ).process(video_path)

        self.assertEqual(result["frames_read"], 1)
        self.assertEqual(result["inference_frames"], 1)
        capture.release.assert_called()
        detector.close.assert_called_once_with()

    def test_unopenable_video_releases_capture_and_detector(self):
        camera = make_camera()
        capture = MagicMock()
        capture.isOpened.return_value = False
        detector = MagicMock()

        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "corrupt.mp4"
            video_path.write_bytes(b"not-a-video")
            with override_settings(CAMERA_AI_TEST_VIDEO_ROOT=Path(directory)), patch(
                "cv2.VideoCapture", return_value=capture
            ):
                with self.assertRaisesMessage(ValueError, "OpenCV could not open"):
                    RecordedVideoProcessor(
                        camera, detector_factory=MagicMock(return_value=detector)
                    ).process(video_path)

        capture.release.assert_called_once_with()
        detector.close.assert_called_once_with()

    def test_recorded_incident_survives_clip_encoding_failure(self):
        camera = make_camera()
        processor = RecordedVideoProcessor(camera, detector_factory=MagicMock())
        event = CrashEvent(
            timestamp=1.0,
            suspicion_score=0.8,
            severity="minor",
            vehicle_categories=["car"],
            track_ids=[1],
            indicators={"trajectory_collision_candidate": True},
        )
        frame_record = (1.0, b"jpeg-placeholder", 64, 48)

        with tempfile.TemporaryDirectory() as directory, override_settings(MEDIA_ROOT=Path(directory)):
            pending = processor._create_pending_incident(event, frame_record[1], [frame_record])
            with self.assertLogs("cameras.video_testing", level="ERROR"), patch(
                    "cameras.video_testing._encode_frames_to_clip",
                    side_effect=RuntimeError("encoder failed"),
                ):
                    processor._finalize_pending(pending)

        incident = pending["incident"]
        incident.refresh_from_db()
        self.assertEqual(incident.processing_status, SuspectedIncident.ProcessingStatus.MEDIA_ERROR)
        self.assertIn("clip could not be generated", incident.media_error)
        self.assertIsNotNone(incident.snapshot_id)

    def test_recorded_video_rejects_paths_outside_safe_root(self):
        camera = make_camera()
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as outside:
            outside_video = Path(outside) / "traffic.mp4"
            outside_video.write_bytes(b"video")
            with override_settings(CAMERA_AI_TEST_VIDEO_ROOT=Path(allowed)):
                with self.assertRaises(ValueError):
                    RecordedVideoProcessor(camera, detector_factory=MagicMock()).process(outside_video)
