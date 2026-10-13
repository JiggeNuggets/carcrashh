"""Bounded camera ingestion, health monitoring, and media capture services."""
from __future__ import annotations

import atexit
import collections
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from django.conf import settings
from django.core.files.base import ContentFile
from django.db import close_old_connections
from django.utils import timezone

from .models import Camera, CameraSnapshot, RecordedClip

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
logger = logging.getLogger(__name__)

OPEN_TIMEOUT_MS = 5000
READ_TIMEOUT_MS = 5000
STREAM_MAX_FPS = 12
JPEG_QUALITY = 78
RECONNECT_DELAYS = (1, 2, 5, 10, 30)


def should_autostart_workers() -> bool:
    """Return true only in an explicitly safe web-serving process."""
    override = os.environ.get("CAMERA_WORKERS_AUTOSTART")
    enabled_by_override = (
        override.strip().lower() in {"1", "true", "yes", "on"}
        if override is not None
        else None
    )
    arguments = set(sys.argv[1:])
    management_commands = {
        "check",
        "collectstatic",
        "createsuperuser",
        "makemigrations",
        "migrate",
        "shell",
        "showmigrations",
        "test",
    }
    if arguments & management_commands:
        return False
    if "runserver" in arguments:
        if enabled_by_override is False:
            return False
        return os.environ.get("RUN_MAIN") == "true" or "--noreload" in arguments
    return enabled_by_override is True


def _cv2():
    import cv2

    try:
        cv2.setLogLevel(0)
    except (AttributeError, TypeError):
        pass
    return cv2


def camera_source(source_type, usb_device_index=None, rtsp_url=""):
    return int(usb_device_index) if source_type == Camera.SourceType.USB else rtsp_url


def _open_capture(source_type, usb_device_index=None, rtsp_url=""):
    cv2 = _cv2()
    source = camera_source(source_type, usb_device_index, rtsp_url)
    if source_type == Camera.SourceType.RTSP:
        return cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                READ_TIMEOUT_MS,
            ],
        )
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS)
    return capture


def _unavailable_message(source_type):
    if source_type == Camera.SourceType.USB:
        return (
            "Unable to open this USB camera. Check that it is connected to the Django server, "
            "not in use by another app, and allowed by operating-system privacy settings."
        )
    return (
        "Unable to connect to this RTSP camera. Check the address, network access, credentials, "
        "camera permissions, and whether RTSP is enabled."
    )


def test_connection(source_type, usb_device_index=None, rtsp_url=""):
    """Open one source, read one frame, and always release it."""
    capture = None
    try:
        capture = _open_capture(source_type, usb_device_index, rtsp_url)
        if not capture.isOpened():
            return False, _unavailable_message(source_type)
        ok, frame = capture.read()
        if not ok or frame is None:
            return False, "The camera opened, but no video frame was received before the timeout."
        return True, "Connection successful. A video frame was received."
    except PermissionError:
        return False, "Camera access was denied. Allow camera access for Python and try again."
    except ImportError:
        return False, "OpenCV is not installed. Install the project requirements and try again."
    except Exception:
        logger.info("Camera connection test failed without logging source details.")
        return False, _unavailable_message(source_type)
    finally:
        if capture is not None:
            capture.release()


class CircularFrameBuffer:
    """Thread-safe ring buffer storing timestamped JPEG frames."""

    def __init__(self, max_seconds: int = 20, target_fps: int = STREAM_MAX_FPS):
        self.lock = threading.Lock()
        self.max_seconds = max(1, int(max_seconds))
        self.target_fps = max(1, int(target_fps))
        self.max_frames = self.max_seconds * self.target_fps
        self.deque: collections.deque = collections.deque(maxlen=self.max_frames)

    def update_config(self, max_seconds: int):
        with self.lock:
            self.max_seconds = max(1, int(max_seconds))
            self.max_frames = self.max_seconds * self.target_fps
            self.deque = collections.deque(self.deque, maxlen=self.max_frames)

    def clear(self):
        with self.lock:
            self.deque.clear()

    def push(self, timestamp: float, jpeg_bytes: bytes, width: int, height: int):
        with self.lock:
            self.deque.append((timestamp, jpeg_bytes, width, height))

    def get_frames(self, seconds: float | None = None):
        with self.lock:
            if not self.deque:
                return []
            if seconds is None or seconds >= self.max_seconds:
                return list(self.deque)
            cutoff = time.monotonic() - max(0, seconds)
            return [frame for frame in self.deque if frame[0] >= cutoff]

    def latest(self):
        with self.lock:
            return self.deque[-1] if self.deque else None

    def latest_frame(self):
        latest = self.latest()
        return latest[1] if latest else None

    def stats(self):
        with self.lock:
            count = len(self.deque)
            duration = 0.0
            if count >= 2:
                duration = max(0.0, round(self.deque[-1][0] - self.deque[0][0], 1))
            return {
                "count": count,
                "max_frames": self.max_frames,
                "seconds_cached": duration,
                "max_seconds": self.max_seconds,
            }


@dataclass
class WorkerState:
    status: str = Camera.ConnectionStatus.DISCONNECTED
    message: str = "Waiting for the worker to start."
    frame: bytes | None = None
    sequence: int = 0
    running: bool = False
    fps: float = 0.0
    last_frame_at: object | None = None
    frame_times: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=STREAM_MAX_FPS * 3)
    )


class CameraWorker:
    def __init__(self, camera: Camera):
        self.camera_id = camera.pk
        self.source_type = camera.source_type
        self.usb_device_index = camera.usb_device_index
        self.rtsp_url = camera.rtsp_url
        self.buffer_enabled = camera.buffer_enabled
        self.buffer_seconds = camera.buffer_seconds
        self.ring_buffer = CircularFrameBuffer(camera.buffer_seconds, STREAM_MAX_FPS)
        self.condition = threading.Condition(threading.RLock())
        self.stop_event = threading.Event()
        self.thread = None
        self.clients = 0
        self.state = WorkerState()
        self.capture_lock = threading.Lock()
        self.capture = None
        self.recording_lock = threading.Lock()
        self.current_frame_record = None

    @property
    def source_signature(self):
        return self.source_type, self.usb_device_index, self.rtsp_url

    def update_camera_settings(self, camera: Camera):
        with self.condition:
            was_buffering = self.buffer_enabled
            self.buffer_enabled = camera.buffer_enabled
            self.buffer_seconds = camera.buffer_seconds
            self.ring_buffer.update_config(camera.buffer_seconds)
            if was_buffering and not self.buffer_enabled:
                self.ring_buffer.clear()
            if not self.buffer_enabled and self.clients == 0:
                self.stop_event.set()
            self.condition.notify_all()

    def ensure_running(self):
        with self.condition:
            if self.thread and self.thread.is_alive():
                return False
            self.stop_event.clear()
            self.state.running = True
            self.state.status = Camera.ConnectionStatus.CONNECTING
            self.state.message = "Connecting to camera..."
            self.thread = threading.Thread(
                target=self._run,
                name=f"camera-worker-{self.camera_id}",
                daemon=True,
            )
            self.thread.start()
            return True

    def subscribe(self):
        with self.condition:
            self.clients += 1
        self.ensure_running()

    def unsubscribe(self):
        with self.condition:
            self.clients = max(0, self.clients - 1)
            if self.clients == 0 and not self.buffer_enabled:
                self.stop_event.set()
            self.condition.notify_all()

    def stop(self, wait=True):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
            thread = self.thread
        with self.capture_lock:
            capture = self.capture
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        if wait and thread and thread is not threading.current_thread():
            thread.join(timeout=(READ_TIMEOUT_MS / 1000) + 1)

    def snapshot(self):
        with self.condition:
            last_frame = self.state.last_frame_at
            return {
                "status": self.state.status,
                "message": self.state.message,
                "running": self.state.running,
                "current_worker_status": self.state.status,
                "viewers": self.clients,
                "fps": self.state.fps,
                "last_frame_at": last_frame.isoformat() if last_frame else None,
                "buffer": self.ring_buffer.stats(),
                "buffer_enabled": self.buffer_enabled,
            }

    def latest_record(self):
        with self.condition:
            return self.current_frame_record

    def frames(self):
        self.subscribe()
        with self.condition:
            last_sequence = self.state.sequence
        try:
            while not self.stop_event.is_set():
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.state.sequence != last_sequence
                        or not self.state.running
                        or self.stop_event.is_set(),
                        timeout=5,
                    )
                    frame = self.state.frame
                    sequence = self.state.sequence
                    running = self.state.running
                if frame is not None and sequence != last_sequence:
                    last_sequence = sequence
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                elif not running:
                    break
        finally:
            self.unsubscribe()

    def _set_state(self, status, message, *, frame=None, running=None):
        with self.condition:
            self.state.status = status
            self.state.message = message
            if frame is not None:
                now = time.monotonic()
                self.state.frame = frame
                self.state.sequence += 1
                self.state.last_frame_at = timezone.now()
                self.state.frame_times.append(now)
                if len(self.state.frame_times) >= 2:
                    span = self.state.frame_times[-1] - self.state.frame_times[0]
                    self.state.fps = round((len(self.state.frame_times) - 1) / span, 1) if span else 0.0
            if running is not None:
                self.state.running = running
            self.condition.notify_all()

    def _save_status(self, status, message):
        close_old_connections()
        try:
            Camera.objects.filter(pk=self.camera_id).update(
                last_connection_status=status,
                last_connection_message=message,
                last_tested_at=timezone.now(),
            )
        except Exception:
            logger.debug("Could not persist camera worker status.")
        finally:
            close_old_connections()

    def _wait_to_reconnect(self, failure_count):
        delay = RECONNECT_DELAYS[min(failure_count - 1, len(RECONNECT_DELAYS) - 1)]
        message = f"Reconnecting in {delay} seconds."
        self._set_state(Camera.ConnectionStatus.RECONNECTING, message)
        return self.stop_event.wait(delay)

    def _run(self):
        failures = 0
        try:
            while not self.stop_event.is_set():
                status = Camera.ConnectionStatus.CONNECTING if failures == 0 else Camera.ConnectionStatus.RECONNECTING
                self._set_state(status, "Connecting to camera...", running=True)
                capture = None
                try:
                    capture = _open_capture(self.source_type, self.usb_device_index, self.rtsp_url)
                    with self.capture_lock:
                        self.capture = capture
                    if not capture.isOpened():
                        failures += 1
                        message = _unavailable_message(self.source_type)
                        self._set_state(Camera.ConnectionStatus.DISCONNECTED, message)
                        self._save_status(Camera.ConnectionStatus.DISCONNECTED, message)
                    else:
                        failures = 0
                        message = "Live preview connected."
                        self._set_state(Camera.ConnectionStatus.CONNECTED, message)
                        self._save_status(Camera.ConnectionStatus.CONNECTED, message)
                        cv2 = _cv2()
                        consecutive_read_failures = 0
                        frame_interval = 1 / STREAM_MAX_FPS
                        while not self.stop_event.is_set():
                            started = time.monotonic()
                            ok, frame = capture.read()
                            if not ok or frame is None:
                                consecutive_read_failures += 1
                                if consecutive_read_failures >= 4:
                                    failures = 1
                                    message = "The video stream stopped returning frames."
                                    self._set_state(Camera.ConnectionStatus.DISCONNECTED, message)
                                    self._save_status(Camera.ConnectionStatus.DISCONNECTED, message)
                                    break
                                continue
                            consecutive_read_failures = 0
                            encoded, jpeg = cv2.imencode(
                                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                            )
                            if encoded:
                                jpeg_bytes = jpeg.tobytes()
                                height, width = frame.shape[:2]
                                with self.condition:
                                    self.current_frame_record = (started, jpeg_bytes, width, height)
                                if self.buffer_enabled:
                                    self.ring_buffer.push(started, jpeg_bytes, width, height)
                                self._set_state(Camera.ConnectionStatus.CONNECTED, message, frame=jpeg_bytes)
                                # AI receives a bounded copy through a separate worker;
                                # inference never blocks camera ingestion or MJPEG output.
                                try:
                                    from .ai_service import ai_registry

                                    ai_registry.submit(self.camera_id, frame, started)
                                except Exception:
                                    logger.debug("AI frame submission was skipped.")
                            remaining = frame_interval - (time.monotonic() - started)
                            if remaining > 0:
                                self.stop_event.wait(remaining)
                except PermissionError:
                    failures += 1
                    message = "Camera access was denied by the operating system."
                    self._set_state(Camera.ConnectionStatus.ERROR, message)
                    self._save_status(Camera.ConnectionStatus.ERROR, message)
                except Exception:
                    failures += 1
                    logger.info("Camera worker failed without logging source details.")
                    message = _unavailable_message(self.source_type)
                    self._set_state(Camera.ConnectionStatus.ERROR, message)
                    self._save_status(Camera.ConnectionStatus.ERROR, message)
                finally:
                    with self.capture_lock:
                        self.capture = None
                    if capture is not None:
                        try:
                            capture.release()
                        except Exception:
                            pass

                if not self.stop_event.is_set() and self._wait_to_reconnect(max(1, failures)):
                    break
        finally:
            self._set_state(self.state.status, self.state.message, running=False)
            close_old_connections()


def _delete_field_file(field_file):
    if field_file and field_file.name:
        try:
            field_file.delete(save=False)
        except OSError:
            logger.info("A media file could not be removed during archive pruning.")


def _prune_camera_media(camera_id):
    """Bound disk usage by retaining only the newest configured items per camera."""
    clip_limit = max(1, int(getattr(settings, "CAMERA_MAX_CLIPS_PER_CAMERA", 50)))
    snapshot_limit = max(1, int(getattr(settings, "CAMERA_MAX_SNAPSHOTS_PER_CAMERA", 100)))
    for clip in list(RecordedClip.objects.filter(camera_id=camera_id).order_by("-created_at")[clip_limit:]):
        _delete_field_file(clip.video_file)
        _delete_field_file(clip.thumbnail)
        clip.delete()
    for snapshot in list(CameraSnapshot.objects.filter(camera_id=camera_id).order_by("-created_at")[snapshot_limit:]):
        _delete_field_file(snapshot.image_file)
        snapshot.delete()


def capture_snapshot(camera_id: int, note: str = ""):
    """Save the newest received frame without exposing its storage path."""
    close_old_connections()
    try:
        camera = Camera.objects.get(pk=camera_id)
        worker = registry.get(camera)
        worker.ensure_running()
        frame_bytes = worker.ring_buffer.latest_frame() or worker.state.frame
        if not frame_bytes:
            return None
        snapshot = CameraSnapshot(camera=camera, note=note[:255])
        filename = f"snapshot_{camera.pk}_{uuid.uuid4().hex[:8]}.jpg"
        snapshot.image_file.save(filename, ContentFile(frame_bytes), save=True)
        _prune_camera_media(camera.pk)
        return snapshot
    finally:
        close_old_connections()


def _remux_to_h264(input_path: Path, output_path: Path):
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-y", "-i", str(input_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-movflags", "+faststart", str(output_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except (OSError, subprocess.SubprocessError):
        return False


def _encode_frames_to_clip(camera_id, frames, pre_seconds, post_seconds, trigger_type, title):
    close_old_connections()
    temp_video_path = final_video_path = thumb_path = None
    clip = None
    try:
        if not frames:
            return None
        cv2 = _cv2()
        import numpy as np

        camera = Camera.objects.get(pk=camera_id)
        media_root = Path(settings.MEDIA_ROOT)
        for subdirectory in ("clips", "thumbnails"):
            (media_root / subdirectory).mkdir(parents=True, exist_ok=True)
        uid = uuid.uuid4().hex[:10]
        temp_video_path = media_root / f"clips/temp_{camera.pk}_{uid}.mp4"
        final_video_rel = f"clips/clip_{camera.pk}_{uid}.mp4"
        final_video_path = media_root / final_video_rel
        thumb_rel = f"thumbnails/thumb_{camera.pk}_{uid}.jpg"
        thumb_path = media_root / thumb_rel

        width, height = frames[0][2], frames[0][3]
        trigger_idx = min(len(frames) - 1, max(0, int(pre_seconds * STREAM_MAX_FPS) - 1))
        thumb_path.write_bytes(frames[trigger_idx][1])
        writer = cv2.VideoWriter(str(temp_video_path), cv2.VideoWriter_fourcc(*"mp4v"), STREAM_MAX_FPS, (width, height))
        if not writer.isOpened():
            writer.release()
            return None
        written = 0
        try:
            for _, jpeg_bytes, _, _ in frames:
                decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    continue
                if decoded.shape[1] != width or decoded.shape[0] != height:
                    decoded = cv2.resize(decoded, (width, height))
                writer.write(decoded)
                written += 1
        finally:
            writer.release()
        if not written or not temp_video_path.exists() or not temp_video_path.stat().st_size:
            return None

        if _remux_to_h264(temp_video_path, final_video_path):
            temp_video_path.unlink(missing_ok=True)
        else:
            temp_video_path.replace(final_video_path)
        clip = RecordedClip.objects.create(
            camera=camera,
            title=(title or f"Test incident clip - {camera.name}")[:140],
            trigger_type=trigger_type,
            video_file=final_video_rel,
            thumbnail=thumb_rel,
            duration_seconds=round(written / STREAM_MAX_FPS, 1),
            pre_buffer_seconds=round(pre_seconds, 1),
            post_buffer_seconds=round(post_seconds, 1),
            file_size_bytes=final_video_path.stat().st_size,
        )
        _prune_camera_media(camera.pk)
        return clip
    except Exception:
        logger.exception("Incident clip encoding failed without camera source details.")
        return None
    finally:
        if temp_video_path:
            temp_video_path.unlink(missing_ok=True)
        if clip is None:
            if final_video_path:
                final_video_path.unlink(missing_ok=True)
            if thumb_path:
                thumb_path.unlink(missing_ok=True)
        close_old_connections()


def trigger_clip_capture(camera_id: int, pre_seconds: float = 15.0, post_seconds: float = 10.0, trigger_type: str = RecordedClip.TriggerType.TEST, title: str = "", on_complete=None):
    close_old_connections()
    try:
        camera = Camera.objects.get(pk=camera_id)
        worker = registry.get(camera)
        worker.ensure_running()
        if not worker.recording_lock.acquire(blocking=False):
            return None
        pre_frames = worker.ring_buffer.get_frames(pre_seconds)

        def collector_job():
            close_old_connections()
            try:
                collected_frames = list(pre_frames)
                end_time = time.monotonic() + post_seconds
                with worker.condition:
                    last_sequence = worker.state.sequence
                while time.monotonic() < end_time and not worker.stop_event.is_set():
                    with worker.condition:
                        worker.condition.wait_for(
                            lambda: worker.state.sequence != last_sequence or not worker.state.running or worker.stop_event.is_set(),
                            timeout=1,
                        )
                        sequence = worker.state.sequence
                        running = worker.state.running
                    if sequence != last_sequence:
                        last_sequence = sequence
                        latest = worker.latest_record()
                        if latest and (not collected_frames or latest[0] != collected_frames[-1][0]):
                            collected_frames.append(latest)
                    if not running:
                        break
                actual_pre = 0.0
                if len(pre_frames) >= 2:
                    actual_pre = min(pre_seconds, pre_frames[-1][0] - pre_frames[0][0])
                clip = _encode_frames_to_clip(
                    camera_id, collected_frames, actual_pre, post_seconds, trigger_type, title
                )
                if on_complete:
                    try:
                        on_complete(clip, "" if clip else "Incident clip encoding produced no media.")
                    except Exception:
                        logger.exception("Clip completion callback failed.")
            finally:
                worker.recording_lock.release()
                close_old_connections()

        thread = threading.Thread(target=collector_job, name=f"clip-recorder-{camera_id}", daemon=True)
        thread.start()
        return thread
    finally:
        close_old_connections()


class CameraRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.workers = {}

    def get(self, camera: Camera):
        old_worker = None
        with self.lock:
            worker = self.workers.get(camera.pk)
            current_source = (camera.source_type, camera.usb_device_index, camera.rtsp_url)
            if worker and worker.source_signature != current_source:
                old_worker = self.workers.pop(camera.pk)
                worker = None
            if worker is None:
                worker = CameraWorker(camera)
                self.workers[camera.pk] = worker
            else:
                worker.update_camera_settings(camera)
        if old_worker:
            old_worker.stop()
        return worker

    def sync(self, camera: Camera, allow_start=False):
        if not camera.enabled:
            self.stop(camera.pk)
            return None
        with self.lock:
            existing = self.workers.get(camera.pk)
        if existing is None and not (allow_start and should_autostart_workers()):
            return None
        worker = self.get(camera)
        if camera.buffer_enabled:
            worker.ensure_running()
        elif worker.clients == 0:
            self.stop(camera.pk)
        return worker

    def start_buffered_cameras(self):
        close_old_connections()
        try:
            for camera in Camera.objects.filter(enabled=True, buffer_enabled=True):
                self.get(camera).ensure_running()
        except Exception:
            logger.debug("Buffered camera startup was skipped because the database is unavailable.")
        finally:
            close_old_connections()

    def stop(self, camera_id: int):
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


registry = CameraRegistry()
atexit.register(registry.stop_all)
