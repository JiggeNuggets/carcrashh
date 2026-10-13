from django.urls import path
from . import views

app_name = "cameras"

urlpatterns = [
    path("", views.camera_list, name="list"),
    path("cameras/add/", views.camera_create, name="add"),
    path("cameras/test/", views.test_camera_connection, name="test"),
    path("cameras/<int:pk>/edit/", views.camera_update, name="edit"),
    path("cameras/<int:pk>/toggle/", views.camera_toggle, name="toggle"),
    path("cameras/<int:pk>/delete/", views.camera_delete, name="delete"),
    path("cameras/<int:pk>/preview/", views.camera_preview, name="preview"),
    path("cameras/<int:pk>/stream.mjpg", views.camera_stream, name="stream"),
    path("cameras/<int:pk>/status/", views.camera_status, name="status"),
    path("cameras/<int:pk>/ai/status/", views.camera_ai_status, name="ai_status"),
    path("cameras/<int:pk>/ai/toggle/", views.camera_ai_toggle, name="ai_toggle"),
    path("cameras/<int:pk>/ai/stream.mjpg", views.camera_ai_stream, name="ai_stream"),
    path("cameras/<int:pk>/trigger-clip/", views.camera_trigger_clip, name="trigger_clip"),
    path("cameras/<int:pk>/snapshot/", views.camera_take_snapshot, name="take_snapshot"),
    path("recordings/", views.recordings_list, name="recordings"),
    path("recordings/clips/<int:pk>/view/", views.clip_view, name="clip_view"),
    path("recordings/clips/<int:pk>/download/", views.clip_download, name="clip_download"),
    path("recordings/clips/<int:pk>/thumbnail/", views.clip_thumbnail, name="clip_thumbnail"),
    path("recordings/clips/<int:pk>/delete/", views.clip_delete, name="clip_delete"),
    path("recordings/snapshots/<int:pk>/view/", views.snapshot_view, name="snapshot_view"),
    path("recordings/snapshots/<int:pk>/download/", views.snapshot_download, name="snapshot_download"),
    path("recordings/snapshots/<int:pk>/delete/", views.snapshot_delete, name="snapshot_delete"),
    path("incidents/", views.incident_list, name="incidents"),
    path("incidents/<uuid:incident_id>/", views.incident_detail, name="incident_detail"),
]
