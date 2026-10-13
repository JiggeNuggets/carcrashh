import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from .camera_service import (
    CameraWorker,
    capture_snapshot,
    registry,
    should_autostart_workers,
    test_connection as check_connection,
)
from .models import Camera, CameraSnapshot, RecordedClip


class CameraAccessTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user("operator", password="test-password-123")

    def test_camera_pages_require_login(self):
        response = self.client.get(reverse("cameras:list"))
        self.assertRedirects(response, f"{reverse('login')}?next=/")

    def test_rtsp_credentials_are_never_rendered(self):
        camera = Camera.objects.create(
            name="Secure camera",
            source_type=Camera.SourceType.RTSP,
            rtsp_url="rtsp://operator:secret-password@192.0.2.10:554/live?token=private",
        )
        self.client.force_login(self.user)
        for url in (reverse("cameras:list"), reverse("cameras:edit", args=[camera.pk])):
            response = self.client.get(url)
            body = response.content.decode()
            self.assertNotIn("secret-password", body)
            self.assertNotIn("token=private", body)

    @patch("cameras.views.test_connection", return_value=(True, "Connection successful."))
    def test_successful_test_token_allows_save(self, _test):
        self.client.force_login(self.user)
        test_response = self.client.post(
            reverse("cameras:test"),
            {"source_type": "usb", "usb_device_index": "0"},
        )
        token = test_response.json()["test_token"]
        response = self.client.post(
            reverse("cameras:add"),
            {
                "name": "USB camera",
                "source_type": "usb",
                "usb_device_index": "0",
                "location": "Desk",
                "enabled": "on",
                "test_token": token,
            },
        )
        self.assertRedirects(response, reverse("cameras:list"))
        self.assertTrue(Camera.objects.filter(name="USB camera", usb_device_index=0).exists())

    @patch("cameras.views.test_connection", return_value=(True, "Connection successful."))
    def test_changing_source_invalidates_test_token(self, _test):
        self.client.force_login(self.user)
        token = self.client.post(
            reverse("cameras:test"),
            {"source_type": "usb", "usb_device_index": "0"},
        ).json()["test_token"]
        response = self.client.post(
            reverse("cameras:add"),
            {
                "name": "Changed camera",
                "source_type": "usb",
                "usb_device_index": "1",
                "enabled": "on",
                "test_token": token,
            },
        )
        self.assertContains(response, "Test this camera successfully before saving", status_code=200)
        self.assertFalse(Camera.objects.filter(name="Changed camera").exists())

    def test_invalid_rtsp_url_is_rejected_before_opening(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("cameras:test"),
            {"source_type": "rtsp", "rtsp_url": "https://example.com/video"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("valid rtsp", response.json()["message"])


    def test_camera_toggle(self):
        self.client.force_login(self.user)
        camera = Camera.objects.create(
            name="Toggle camera",
            source_type=Camera.SourceType.USB,
            usb_device_index=0,
            enabled=True,
        )
        response = self.client.post(reverse("cameras:toggle", args=[camera.pk]))
        self.assertRedirects(response, reverse("cameras:list"))
        camera.refresh_from_db()
        self.assertFalse(camera.enabled)

        # Toggle back
        response = self.client.post(reverse("cameras:toggle", args=[camera.pk]))
        self.assertRedirects(response, reverse("cameras:list"))
        camera.refresh_from_db()
        self.assertTrue(camera.enabled)

    def test_camera_delete(self):
        self.client.force_login(self.user)
        camera = Camera.objects.create(
            name="Delete camera",
            source_type=Camera.SourceType.USB,
            usb_device_index=0,
        )
        get_response = self.client.get(reverse("cameras:delete", args=[camera.pk]))
        self.assertEqual(get_response.status_code, 200)
        self.assertContains(get_response, "Remove “Delete camera”?")

        post_response = self.client.post(reverse("cameras:delete", args=[camera.pk]))
        self.assertRedirects(post_response, reverse("cameras:list"))
        self.assertFalse(Camera.objects.filter(pk=camera.pk).exists())

    @patch("cameras.views.test_connection", return_value=(True, "Connection successful."))
    def test_camera_update(self, _test):
        self.client.force_login(self.user)
        camera = Camera.objects.create(
            name="Initial Name",
            source_type=Camera.SourceType.USB,
            usb_device_index=0,
            location="Gate 1",
        )
        token = self.client.post(
            reverse("cameras:test"),
            {"source_type": "usb", "usb_device_index": "0", "camera_id": camera.pk},
        ).json()["test_token"]
        response = self.client.post(
            reverse("cameras:edit", args=[camera.pk]),
            {
                "name": "Updated Name",
                "source_type": "usb",
                "usb_device_index": "0",
                "location": "Gate 2 (Crash Hotspot)",
                "enabled": "on",
                "test_token": token,
            },
        )
        self.assertRedirects(response, reverse("cameras:list"))
        camera.refresh_from_db()
        self.assertEqual(camera.name, "Updated Name")
        self.assertEqual(camera.location, "Gate 2 (Crash Hotspot)")

    def test_camera_preview_view(self):
        self.client.force_login(self.user)
        camera = Camera.objects.create(
            name="Active camera",
            source_type=Camera.SourceType.USB,
            usb_device_index=0,
            enabled=True,
        )
        response = self.client.get(reverse("cameras:preview", args=[camera.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Live preview")
        self.assertContains(response, camera.name)

        # Disabled camera shows disabled notice
        camera.enabled = False
        camera.save()
        disabled_response = self.client.get(reverse("cameras:preview", args=[camera.pk]))
        self.assertContains(disabled_response, "This camera is disabled")

    def test_camera_status_endpoint(self):
        self.client.force_login(self.user)
        camera = Camera.objects.create(
            name="Status test",
            source_type=Camera.SourceType.USB,
            usb_device_index=0,
        )
        worker = MagicMock()
        worker.snapshot.return_value = {
            "status": Camera.ConnectionStatus.CONNECTED,
            "message": "Live preview connected.",
            "running": True,
            "current_worker_status": Camera.ConnectionStatus.CONNECTED,
            "viewers": 0,
            "fps": 12.0,
            "last_frame_at": None,
            "buffer": {"count": 4, "seconds_cached": 0.3, "max_seconds": 15},
            "buffer_enabled": True,
        }
        with patch.object(registry, "get", return_value=worker):
            response = self.client.get(reverse("cameras:status", args=[camera.pk]))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("message", data)
        self.assertEqual(data["running"], True)
        self.assertIn("buffer", data)

    @patch("cameras.views.trigger_clip_capture")
    def test_camera_trigger_clip_endpoint(self, mock_trigger):
        mock_trigger.return_value = MagicMock()
        self.client.force_login(self.user)
        camera = Camera.objects.create(
            name="Trigger Cam",
            source_type=Camera.SourceType.USB,
            usb_device_index=0,
            enabled=True,
            buffer_seconds=15,
        )
        response = self.client.post(
            reverse("cameras:trigger_clip", args=[camera.pk]),
            {"pre_seconds": "10", "post_seconds": "5", "title": "Test Incident"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        mock_trigger.assert_called_once()

    @patch("cameras.views.capture_snapshot")
    def test_camera_snapshot_endpoint(self, mock_snap):
        snap_mock = MagicMock()
        snap_mock.image_file.url = "/media/snapshots/test.jpg"
        snap_mock.pk = 42
        mock_snap.return_value = snap_mock

        self.client.force_login(self.user)
        camera = Camera.objects.create(
            name="Snap Cam",
            source_type=Camera.SourceType.USB,
            usb_device_index=0,
            enabled=True,
        )
        response = self.client.post(
            reverse("cameras:take_snapshot", args=[camera.pk]),
            {"note": "Operator manual snapshot"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["url"], "/media/snapshots/test.jpg")

    def test_recordings_list_and_deletes(self):
        self.client.force_login(self.user)
        camera = Camera.objects.create(
            name="Archive Cam",
            source_type=Camera.SourceType.USB,
            usb_device_index=0,
        )
        clip = RecordedClip.objects.create(
            camera=camera,
            title="Crash Event Sector 4",
            video_file="clips/dummy.mp4",
            duration_seconds=12.5,
            pre_buffer_seconds=10.0,
            post_buffer_seconds=2.5,
            file_size_bytes=1048576,
        )
        snapshot = CameraSnapshot.objects.create(
            camera=camera,
            image_file="snapshots/dummy.jpg",
            note="Impact marker",
        )

        response = self.client.get(reverse("cameras:recordings"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Crash Event Sector 4")
        self.assertContains(response, "Impact marker")

        # Delete clip
        del_clip_resp = self.client.post(reverse("cameras:clip_delete", args=[clip.pk]))
        self.assertRedirects(del_clip_resp, reverse("cameras:recordings"))
        self.assertFalse(RecordedClip.objects.filter(pk=clip.pk).exists())

        # Delete snapshot
        del_snap_resp = self.client.post(reverse("cameras:snapshot_delete", args=[snapshot.pk]))
        self.assertRedirects(del_snap_resp, reverse("cameras:recordings"))
        self.assertFalse(CameraSnapshot.objects.filter(pk=snapshot.pk).exists())


class RingBufferTests(TestCase):
    def test_circular_frame_buffer_eviction_and_filtering(self):
        import time
        from .camera_service import CircularFrameBuffer

        buf = CircularFrameBuffer(max_seconds=5, target_fps=2)
        # Capacity is 5 * 2 = 10 frames
        self.assertEqual(buf.max_frames, 10)

        # Push 15 frames
        now = time.monotonic()
        for i in range(15):
            buf.push(now + i, f"frame_{i}".encode(), 640, 480)

        # Length is capped at max_frames (10)
        self.assertEqual(len(buf.deque), 10)
        # Oldest 5 frames (0-4) were evicted, first frame is frame_5
        self.assertEqual(buf.deque[0][1], b"frame_5")
        self.assertEqual(buf.latest_frame(), b"frame_14")

        stats = buf.stats()
        self.assertEqual(stats["count"], 10)
        self.assertEqual(stats["max_frames"], 10)



class CaptureCleanupTests(TestCase):
    @patch("cameras.camera_service._open_capture")
    def test_connection_check_always_releases_capture(self, open_capture):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        open_capture.return_value = capture

        ok, _message = check_connection(Camera.SourceType.USB, usb_device_index=0)

        self.assertFalse(ok)
        capture.release.assert_called_once_with()


class BufferSettingsTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user("buffer-operator", password="test-password-123")
        self.client.force_login(self.user)

    def tearDown(self):
        registry.stop_all()

    def test_buffer_only_update_preserves_worker_and_needs_no_retest(self):
        camera = Camera.objects.create(
            name="Buffer camera",
            source_type=Camera.SourceType.USB,
            usb_device_index=0,
            enabled=True,
            buffer_enabled=True,
            buffer_seconds=15,
        )
        worker = registry.get(camera)
        existing_thread = MagicMock()
        existing_thread.is_alive.return_value = True
        worker.thread = existing_thread

        response = self.client.post(
            reverse("cameras:edit", args=[camera.pk]),
            {
                "name": camera.name,
                "source_type": Camera.SourceType.USB,
                "usb_device_index": "0",
                "enabled": "on",
                "buffer_enabled": "on",
                "buffer_seconds": "30",
            },
        )

        self.assertRedirects(response, reverse("cameras:list"))
        camera.refresh_from_db()
        self.assertEqual(camera.buffer_seconds, 30)
        self.assertTrue(camera.buffer_enabled)
        self.assertIs(registry.workers[camera.pk], worker)
        self.assertIs(worker.thread, existing_thread)
        self.assertEqual(worker.ring_buffer.max_seconds, 30)

    def test_buffer_duration_is_limited_to_supported_choices(self):
        camera = Camera.objects.create(
            name="Choice camera",
            source_type=Camera.SourceType.USB,
            usb_device_index=0,
        )
        response = self.client.post(
            reverse("cameras:edit", args=[camera.pk]),
            {
                "name": camera.name,
                "source_type": Camera.SourceType.USB,
                "usb_device_index": "0",
                "enabled": "on",
                "buffer_enabled": "on",
                "buffer_seconds": "25",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Select a valid choice")


class SnapshotServiceTests(TestCase):
    def tearDown(self):
        registry.stop_all()

    def test_snapshot_generation_saves_latest_frame(self):
        camera = Camera.objects.create(
            name="Snapshot service camera",
            source_type=Camera.SourceType.USB,
            usb_device_index=0,
        )
        worker = registry.get(camera)
        worker.state.frame = b"jpeg-test-data"
        with tempfile.TemporaryDirectory() as media_root, override_settings(MEDIA_ROOT=media_root):
            with patch.object(worker, "ensure_running"):
                snapshot = capture_snapshot(camera.pk, note="Manual capture")
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.note, "Manual capture")
            self.assertEqual(Path(snapshot.image_file.path).read_bytes(), b"jpeg-test-data")


class SecureMediaAccessTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user("media-operator", password="test-password-123")
        self.camera = Camera.objects.create(
            name="Media camera",
            source_type=Camera.SourceType.USB,
            usb_device_index=0,
            buffer_enabled=False,
        )

    def test_media_and_status_require_authentication(self):
        clip = RecordedClip.objects.create(camera=self.camera, title="Private clip", video_file="clips/private.mp4")
        snapshot = CameraSnapshot.objects.create(camera=self.camera, image_file="snapshots/private.jpg")
        protected_urls = (
            reverse("cameras:clip_view", args=[clip.pk]),
            reverse("cameras:clip_download", args=[clip.pk]),
            reverse("cameras:snapshot_view", args=[snapshot.pk]),
            reverse("cameras:snapshot_download", args=[snapshot.pk]),
            reverse("cameras:status", args=[self.camera.pk]),
        )
        for url in protected_urls:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 302)
            self.assertIn(reverse("login"), response.url)

        response = self.client.post(reverse("cameras:clip_delete", args=[clip.pk]))
        self.assertEqual(response.status_code, 302)
        self.assertTrue(RecordedClip.objects.filter(pk=clip.pk).exists())

    def test_authenticated_preview_download_and_safe_path_validation(self):
        self.client.force_login(self.user)
        with tempfile.TemporaryDirectory() as media_root, override_settings(MEDIA_ROOT=media_root):
            clips_dir = Path(media_root) / "clips"
            snapshots_dir = Path(media_root) / "snapshots"
            clips_dir.mkdir()
            snapshots_dir.mkdir()
            (clips_dir / "safe.mp4").write_bytes(b"mp4-data")
            (snapshots_dir / "safe.jpg").write_bytes(b"jpeg-data")
            clip = RecordedClip.objects.create(camera=self.camera, title="Safe clip", video_file="clips/safe.mp4")
            snapshot = CameraSnapshot.objects.create(camera=self.camera, image_file="snapshots/safe.jpg")

            preview = self.client.get(reverse("cameras:clip_view", args=[clip.pk]))
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview["Content-Type"], "video/mp4")
            self.assertNotIn("attachment", preview.get("Content-Disposition", ""))
            preview.close()

            download = self.client.get(reverse("cameras:clip_download", args=[clip.pk]))
            self.assertEqual(download.status_code, 200)
            self.assertIn("attachment", download["Content-Disposition"])
            download.close()

            image = self.client.get(reverse("cameras:snapshot_view", args=[snapshot.pk]))
            self.assertEqual(image.status_code, 200)
            self.assertEqual(image["Content-Type"], "image/jpeg")
            image.close()

            unsafe = RecordedClip.objects.create(camera=self.camera, video_file="../outside.mp4")
            self.assertEqual(self.client.get(reverse("cameras:clip_view", args=[unsafe.pk])).status_code, 404)

    def test_status_has_complete_health_payload(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("cameras:status", args=[self.camera.pk]))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        for field in (
            "camera_name",
            "source_type",
            "fps",
            "buffer_seconds",
            "buffer_enabled",
            "last_frame_at",
            "current_worker_status",
        ):
            self.assertIn(field, payload)


class WorkerReliabilityTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(
            name="Reliable camera",
            source_type=Camera.SourceType.USB,
            usb_device_index=0,
            buffer_enabled=True,
        )

    def tearDown(self):
        registry.stop_all()

    @patch("cameras.camera_service._cv2")
    @patch("cameras.camera_service._open_capture")
    def test_worker_reconnects_after_failed_open(self, open_capture, _cv2_mock):
        first_capture = MagicMock()
        first_capture.isOpened.return_value = False
        second_capture = MagicMock()
        second_capture.isOpened.return_value = True
        worker = CameraWorker(self.camera)

        def stop_after_reconnect():
            worker.stop(wait=False)
            return False, None

        second_capture.read.side_effect = stop_after_reconnect
        open_capture.side_effect = (first_capture, second_capture)
        with patch.object(worker, "_wait_to_reconnect", return_value=False), patch.object(worker, "_save_status"):
            worker._run()

        self.assertEqual(open_capture.call_count, 2)
        first_capture.release.assert_called()
        second_capture.release.assert_called()

    def test_stop_releases_active_capture_and_signals_shutdown(self):
        worker = CameraWorker(self.camera)
        capture = MagicMock()
        worker.capture = capture
        worker.stop()
        self.assertTrue(worker.stop_event.is_set())
        capture.release.assert_called_once_with()

    @patch("cameras.camera_service.threading.Thread")
    def test_ensure_running_does_not_duplicate_live_thread(self, thread_class):
        worker = CameraWorker(self.camera)
        thread = MagicMock()
        thread.is_alive.return_value = True
        worker.thread = thread
        self.assertFalse(worker.ensure_running())
        thread_class.assert_not_called()

    def test_management_commands_never_autostart_capture(self):
        with patch("cameras.camera_service.sys.argv", ["manage.py", "migrate"]), patch.dict(
            "cameras.camera_service.os.environ",
            {"CAMERA_WORKERS_AUTOSTART": "1"},
            clear=False,
        ):
            self.assertFalse(should_autostart_workers())

