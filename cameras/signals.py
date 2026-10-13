from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

from .models import Camera, CameraSnapshot, RecordedClip
from .camera_service import registry
from .ai_service import ai_registry

@receiver(post_save, sender=Camera)
def start_buffer_worker(sender, instance: Camera, created, **kwargs):
    """Synchronize settings without restarting a worker unnecessarily."""
    registry.sync(instance, allow_start=True)
    ai_registry.sync(instance, allow_start=True)

@receiver(post_delete, sender=Camera)
def cleanup_worker(sender, instance: Camera, **kwargs):
    """Clean up worker when a camera is deleted."""
    registry.stop(instance.pk)
    ai_registry.stop(instance.pk)


@receiver(post_delete, sender=RecordedClip)
def cleanup_clip_files(sender, instance: RecordedClip, **kwargs):
    """Remove files for direct deletes and Camera cascade deletes."""
    for field_file in (instance.video_file, instance.thumbnail):
        if field_file and field_file.name:
            field_file.delete(save=False)


@receiver(post_delete, sender=CameraSnapshot)
def cleanup_snapshot_file(sender, instance: CameraSnapshot, **kwargs):
    if instance.image_file and instance.image_file.name:
        instance.image_file.delete(save=False)
