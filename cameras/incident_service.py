"""Persistence and media orchestration for suspected AI incidents."""
from __future__ import annotations

import logging

from django.db import close_old_connections
from django.utils import timezone

from .camera_service import capture_snapshot, trigger_clip_capture
from .models import Camera, RecordedClip, SuspectedIncident

logger = logging.getLogger(__name__)


def create_live_incident(camera_id, event):
    """Persist an event immediately, then generate evidence asynchronously."""
    close_old_connections()
    try:
        camera = Camera.objects.get(pk=camera_id)
        incident = SuspectedIncident.objects.create(
            camera=camera,
            detected_at=timezone.now(),
            suspicion_score=event.suspicion_score,
            severity=event.severity,
            vehicle_categories=event.vehicle_categories,
            indicators={**event.indicators, "track_ids": event.track_ids},
            processing_status=SuspectedIncident.ProcessingStatus.RECORDING,
            source=SuspectedIncident.Source.LIVE,
        )
        errors = []
        try:
            snapshot = capture_snapshot(camera.pk, note=f"Suspected incident {incident.incident_id}")
        except Exception:
            logger.exception("Suspected incident snapshot generation failed.")
            snapshot = None
        if snapshot:
            incident.snapshot = snapshot
        else:
            errors.append("Incident snapshot was unavailable.")
        incident.media_error = " ".join(errors)
        incident.save(update_fields=("snapshot", "media_error"))

        def media_complete(clip, error_message=""):
            close_old_connections()
            try:
                current = SuspectedIncident.objects.get(pk=incident.pk)
                media_errors = [value for value in (current.media_error, error_message) if value]
                if clip:
                    current.video_clip = clip
                else:
                    media_errors.append("Incident video clip could not be generated.")
                current.media_error = " ".join(dict.fromkeys(media_errors))[:255]
                current.processing_status = (
                    SuspectedIncident.ProcessingStatus.MEDIA_ERROR
                    if current.media_error
                    else SuspectedIncident.ProcessingStatus.READY
                )
                current.save(update_fields=("video_clip", "media_error", "processing_status"))
            except Exception:
                logger.exception("Suspected incident media status could not be updated.")
            finally:
                close_old_connections()

        thread = trigger_clip_capture(
            camera_id=camera.pk,
            pre_seconds=min(15.0, float(camera.buffer_seconds)),
            post_seconds=float(camera.ai_post_event_seconds),
            trigger_type=RecordedClip.TriggerType.CRASH_DETECTED,
            title=f"Suspected incident {str(incident.incident_id)[:8]} - {camera.name}",
            on_complete=media_complete,
        )
        if thread is None:
            media_complete(None, "Another clip was already being generated for this camera.")
        return incident
    finally:
        close_old_connections()
