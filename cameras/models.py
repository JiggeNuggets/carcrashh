import uuid
from urllib.parse import urlsplit, urlunsplit

from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class Camera(models.Model):
    BUFFER_DURATION_CHOICES = (
        (15, "15 seconds"),
        (20, "20 seconds"),
        (30, "30 seconds"),
    )
    AI_FRAME_RATE_CHOICES = ((1, "1 FPS"), (2, "2 FPS"), (5, "5 FPS"))
    AI_INFERENCE_SIZE_CHOICES = ((320, "320 px"), (480, "480 px"), (640, "640 px"))
    INCIDENT_COOLDOWN_CHOICES = (
        (30, "30 seconds"),
        (60, "1 minute"),
        (120, "2 minutes"),
        (300, "5 minutes"),
    )

    class SourceType(models.TextChoices):
        USB = "usb", "USB camera"
        RTSP = "rtsp", "RTSP camera"

    class ConnectionStatus(models.TextChoices):
        CONNECTING = "connecting", "Connecting"
        CONNECTED = "connected", "Connected"
        DISCONNECTED = "disconnected", "Disconnected"
        RECONNECTING = "reconnecting", "Reconnecting"
        DISABLED = "disabled", "Disabled"
        ERROR = "error", "Error"

    name = models.CharField(max_length=120)
    source_type = models.CharField(max_length=8, choices=SourceType.choices)
    usb_device_index = models.PositiveSmallIntegerField(null=True, blank=True)
    rtsp_url = models.CharField(max_length=2048, blank=True)
    location = models.CharField(max_length=255, blank=True)
    enabled = models.BooleanField(default=True)
    buffer_enabled = models.BooleanField(
        default=True,
        help_text="Keep a continuous rolling pre-crash frame buffer in background",
    )
    buffer_seconds = models.PositiveSmallIntegerField(
        default=15,
        choices=BUFFER_DURATION_CHOICES,
        help_text="Rolling buffer window in seconds",
    )
    ai_enabled = models.BooleanField(
        default=False,
        help_text="Analyze sampled camera frames for suspected vehicle collisions",
    )
    ai_confidence_threshold = models.FloatField(
        default=0.45,
        validators=[MinValueValidator(0.1), MaxValueValidator(0.95)],
        help_text="Minimum object-detection confidence",
    )
    ai_frame_rate = models.PositiveSmallIntegerField(
        default=2,
        choices=AI_FRAME_RATE_CHOICES,
        help_text="Maximum frames per second sent to AI inference",
    )
    ai_inference_size = models.PositiveSmallIntegerField(
        default=320,
        choices=AI_INFERENCE_SIZE_CHOICES,
        help_text="YOLO inference image size",
    )
    crash_suspicion_threshold = models.FloatField(
        default=0.75,
        validators=[MinValueValidator(0.5), MaxValueValidator(0.95)],
        help_text="Heuristic score required to create a suspected incident",
    )
    incident_cooldown_seconds = models.PositiveSmallIntegerField(
        default=60,
        choices=INCIDENT_COOLDOWN_CHOICES,
        help_text="Minimum interval between suspected incidents",
    )
    ai_post_event_seconds = models.PositiveSmallIntegerField(
        default=10,
        choices=((5, "5 seconds"), (10, "10 seconds"), (15, "15 seconds")),
        help_text="Post-event footage recorded after an AI trigger",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    last_connection_status = models.CharField(
        max_length=16,
        choices=ConnectionStatus.choices,
        default=ConnectionStatus.DISCONNECTED,
    )
    last_connection_message = models.CharField(max_length=255, blank=True)
    last_tested_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ("name",)

    def __str__(self):
        return self.name

    def clean(self):
        errors = {}
        if self.source_type == self.SourceType.USB:
            if self.usb_device_index is None:
                errors["usb_device_index"] = "Choose the USB device index."
            self.rtsp_url = ""
        elif self.source_type == self.SourceType.RTSP:
            self.usb_device_index = None
            if not self.rtsp_url:
                errors["rtsp_url"] = "Enter an RTSP URL."
            else:
                try:
                    parts = urlsplit(self.rtsp_url)
                    port = parts.port
                except ValueError:
                    errors["rtsp_url"] = "Enter a valid RTSP URL."
                else:
                    if parts.scheme.lower() not in {"rtsp", "rtsps"} or not parts.hostname:
                        errors["rtsp_url"] = "Use an rtsp:// or rtsps:// URL with a host."
                    elif port is not None and not 1 <= port <= 65535:
                        errors["rtsp_url"] = "The RTSP port must be between 1 and 65535."
        else:
            errors["source_type"] = "Choose a supported camera source."
        if errors:
            raise ValidationError(errors)

    @property
    def source_label(self):
        if self.source_type == self.SourceType.USB:
            return f"Device {self.usb_device_index}"
        return self.redacted_rtsp_url

    @property
    def redacted_rtsp_url(self):
        if not self.rtsp_url:
            return "RTSP source"
        try:
            parts = urlsplit(self.rtsp_url)
            hostname = parts.hostname or "camera"
            host = f"[{hostname}]" if ":" in hostname else hostname
            if parts.port:
                host = f"{host}:{parts.port}"
            if parts.username:
                host = f"{parts.username}:••••@{host}"
            path = parts.path
            if len(path) > 32:
                path = f"{path[:29]}…"
            return urlunsplit((parts.scheme, host, path, "", ""))
        except (TypeError, ValueError):
            return "RTSP source"


class RecordedClip(models.Model):
    class TriggerType(models.TextChoices):
        MANUAL = "manual", "Manual operator trigger"
        CRASH_DETECTED = "crash_detected", "Crash detection event"
        MOTION = "motion", "Motion anomaly"
        TEST = "test", "System drill / test"

    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name="recordings")
    title = models.CharField(max_length=140, blank=True)
    trigger_type = models.CharField(
        max_length=24,
        choices=TriggerType.choices,
        default=TriggerType.MANUAL,
    )
    video_file = models.FileField(upload_to="clips/")
    thumbnail = models.FileField(upload_to="thumbnails/", blank=True)
    duration_seconds = models.FloatField(default=0.0)
    pre_buffer_seconds = models.FloatField(default=0.0)
    post_buffer_seconds = models.FloatField(default=0.0)
    file_size_bytes = models.PositiveBigIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return self.title or f"Clip from {self.camera.name} at {self.created_at:%Y-%m-%d %H:%M:%S}"

    @property
    def file_size_display(self):
        bytes_val = self.file_size_bytes
        if bytes_val < 1024:
            return f"{bytes_val} B"
        elif bytes_val < 1024 * 1024:
            return f"{bytes_val / 1024:.1f} KB"
        return f"{bytes_val / (1024 * 1024):.1f} MB"


class CameraSnapshot(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name="snapshots")
    image_file = models.FileField(upload_to="snapshots/")
    note = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"Snapshot {self.camera.name} at {self.created_at:%Y-%m-%d %H:%M:%S}"


class SuspectedIncident(models.Model):
    class Severity(models.TextChoices):
        MINOR = "minor", "Minor (provisional)"
        MODERATE = "moderate", "Moderate (provisional)"
        SEVERE = "severe", "Severe (provisional)"
        UNDETERMINED = "undetermined", "Undetermined"

    class ProcessingStatus(models.TextChoices):
        DETECTED = "detected", "Detected"
        RECORDING = "recording", "Generating media"
        READY = "ready", "Evidence ready"
        MEDIA_ERROR = "media_error", "Media generation error"

    class ReviewStatus(models.TextChoices):
        PENDING = "pending_review", "Pending Review"
        REVIEWED = "reviewed", "Reviewed"
        DISMISSED = "dismissed", "Dismissed"

    class Source(models.TextChoices):
        LIVE = "live", "Live camera"
        RECORDED_VIDEO = "recorded_video", "Recorded video test"

    incident_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name="incidents")
    detected_at = models.DateTimeField()
    suspicion_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    severity = models.CharField(
        max_length=16,
        choices=Severity.choices,
        default=Severity.UNDETERMINED,
    )
    vehicle_categories = models.JSONField(default=list, blank=True)
    indicators = models.JSONField(default=dict, blank=True)
    snapshot = models.ForeignKey(
        CameraSnapshot,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="incidents",
    )
    video_clip = models.ForeignKey(
        RecordedClip,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="incidents",
    )
    processing_status = models.CharField(
        max_length=20,
        choices=ProcessingStatus.choices,
        default=ProcessingStatus.DETECTED,
    )
    review_status = models.CharField(
        max_length=20,
        choices=ReviewStatus.choices,
        default=ReviewStatus.PENDING,
    )
    source = models.CharField(max_length=20, choices=Source.choices, default=Source.LIVE)
    media_error = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-detected_at",)

    def __str__(self):
        return f"Suspected incident {self.incident_id} - {self.camera.name}"

    @property
    def score_percent(self):
        return round(self.suspicion_score * 100)

