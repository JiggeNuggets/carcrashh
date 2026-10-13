import threading
import time

from django.apps import apps
from django.apps import AppConfig


def _start_workers_after_app_loading(camera_service, ai_service):
    """Avoid database access from AppConfig.ready() while preserving autostart."""
    while not apps.ready:
        time.sleep(0.01)
    camera_service.registry.start_buffered_cameras()
    ai_service.ai_registry.start_enabled_cameras()


class CamerasConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "cameras"

    def ready(self):
        # Importing registers process-exit cleanup and lifecycle signals. Camera
        # capture is deliberately limited to the serving process so migrations,
        # tests and other management commands never open hardware.
        from . import camera_service
        from . import ai_service
        from . import signals  # noqa: F401

        if camera_service.should_autostart_workers():
            threading.Thread(
                target=_start_workers_after_app_loading,
                args=(camera_service, ai_service),
                name="camera-worker-autostart",
                daemon=True,
            ).start()
