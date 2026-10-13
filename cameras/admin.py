from django.contrib import admin
from .models import Camera, SuspectedIncident


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ("name", "source_type", "location", "enabled", "last_connection_status")
    list_filter = ("source_type", "enabled", "last_connection_status")
    search_fields = ("name", "location")

    def get_exclude(self, request, obj=None):
        # Keep URLs carrying credentials out of the generic admin interface.
        return ("rtsp_url",)


@admin.register(SuspectedIncident)
class SuspectedIncidentAdmin(admin.ModelAdmin):
    list_display = (
        "incident_id",
        "camera",
        "detected_at",
        "suspicion_score",
        "severity",
        "processing_status",
        "review_status",
    )
    list_filter = ("severity", "processing_status", "review_status", "source")
    readonly_fields = ("incident_id", "created_at")
