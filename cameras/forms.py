import hashlib
import hmac

from django import forms
from django.core import signing

from .models import Camera

TOKEN_SALT = "cameras.connection-test.v1"


def source_fingerprint(source_type, usb_device_index=None, rtsp_url=""):
    if source_type == Camera.SourceType.USB:
        return f"usb:{usb_device_index}"
    digest = hashlib.sha256((rtsp_url or "").encode("utf-8")).hexdigest()
    return f"rtsp:{digest}"


def create_test_token(source_type, usb_device_index=None, rtsp_url=""):
    return signing.dumps(
        {"source": source_fingerprint(source_type, usb_device_index, rtsp_url)},
        salt=TOKEN_SALT,
        compress=True,
    )


class CameraForm(forms.ModelForm):
    test_token = forms.CharField(required=False, widget=forms.HiddenInput)

    class Meta:
        model = Camera
        fields = (
            "name",
            "source_type",
            "usb_device_index",
            "rtsp_url",
            "location",
            "enabled",
            "buffer_enabled",
            "buffer_seconds",
            "ai_enabled",
            "ai_confidence_threshold",
            "ai_frame_rate",
            "ai_inference_size",
            "crash_suspicion_threshold",
            "incident_cooldown_seconds",
            "ai_post_event_seconds",
        )
        widgets = {
            "name": forms.TextInput(attrs={"placeholder": "e.g. North entrance"}),
            "usb_device_index": forms.NumberInput(attrs={"min": 0, "step": 1}),
            "rtsp_url": forms.PasswordInput(
                render_value=False,
                attrs={
                    "autocomplete": "new-password",
                    "placeholder": "rtsp://user:password@camera-host:554/stream",
                },
            ),
            "location": forms.TextInput(attrs={"placeholder": "Optional physical location"}),
            "buffer_seconds": forms.Select(),
            "ai_confidence_threshold": forms.NumberInput(
                attrs={"min": "0.10", "max": "0.95", "step": "0.05"}
            ),
            "crash_suspicion_threshold": forms.NumberInput(
                attrs={"min": "0.50", "max": "0.95", "step": "0.05"}
            ),
        }
        help_texts = {
            "usb_device_index": "Usually 0 for the first camera attached to this server.",
            "rtsp_url": "Stored locally and never shown again. Use only a camera you are authorized to access.",
            "buffer_seconds": "Choose how much pre-event footage is kept in memory.",
            "ai_confidence_threshold": "Objects below this confidence are ignored.",
            "crash_suspicion_threshold": "A heuristic score, not a calibrated crash probability.",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["source_type"].widget.attrs["data-source-select"] = "true"
        self.fields["usb_device_index"].widget.attrs["inputmode"] = "numeric"
        self.fields["buffer_seconds"].required = False
        self.fields["buffer_seconds"].initial = 15
        self.fields["buffer_enabled"].required = False
        self.fields["buffer_enabled"].initial = True
        self.fields["ai_enabled"].required = False
        for name in (
            "ai_confidence_threshold",
            "ai_frame_rate",
            "ai_inference_size",
            "crash_suspicion_threshold",
            "incident_cooldown_seconds",
            "ai_post_event_seconds",
        ):
            self.fields[name].required = False
        if self.instance.pk and self.instance.source_type == Camera.SourceType.RTSP:
            self.fields["rtsp_url"].required = False
            self.fields["rtsp_url"].help_text = (
                "Leave blank to keep the saved RTSP URL. Enter a new URL to replace it. "
                "Saved credentials are never displayed."
            )

    def clean_rtsp_url(self):
        value = self.cleaned_data.get("rtsp_url", "").strip()
        if (
            not value
            and self.instance.pk
            and self.cleaned_data.get("source_type") == Camera.SourceType.RTSP
        ):
            return self.instance.rtsp_url
        return value

    def clean(self):
        cleaned = super().clean()
        self.source_changed = True
        source_type = cleaned.get("source_type")
        usb_index = cleaned.get("usb_device_index")
        rtsp_url = cleaned.get("rtsp_url", "")

        if source_type == Camera.SourceType.USB:
            cleaned["rtsp_url"] = ""
        elif source_type == Camera.SourceType.RTSP:
            cleaned["usb_device_index"] = None

        if cleaned.get("buffer_seconds") is None:
            cleaned["buffer_seconds"] = 15
        if cleaned.get("buffer_enabled") is None:
            cleaned["buffer_enabled"] = True
        ai_defaults = {
            "ai_confidence_threshold": 0.45,
            "ai_frame_rate": 2,
            "ai_inference_size": 320,
            "crash_suspicion_threshold": 0.75,
            "incident_cooldown_seconds": 60,
            "ai_post_event_seconds": 10,
        }
        for name, default in ai_defaults.items():
            if cleaned.get(name) is None:
                cleaned[name] = getattr(self.instance, name, default) if self.instance.pk else default

        if self.errors:
            return cleaned

        expected = source_fingerprint(source_type, usb_index, rtsp_url)
        existing_source = None
        if self.instance.pk:
            existing_source = source_fingerprint(
                self.instance.source_type,
                self.instance.usb_device_index,
                self.instance.rtsp_url,
            )
        # Non-source edits (including buffer settings) do not require reopening a
        # healthy stream and should not interrupt an active worker.
        if existing_source and hmac.compare_digest(existing_source, expected):
            self.source_changed = False
            return cleaned

        token = cleaned.get("test_token", "")
        try:
            payload = signing.loads(token, salt=TOKEN_SALT, max_age=300)
            tested_source = payload.get("source", "")
        except (signing.BadSignature, signing.SignatureExpired, TypeError):
            tested_source = ""
        if not hmac.compare_digest(tested_source, expected):
            raise forms.ValidationError(
                "Test this camera successfully before saving. Connection tests are valid for 5 minutes."
            )
        return cleaned
