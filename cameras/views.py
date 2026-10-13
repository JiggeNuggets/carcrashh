from pathlib import PurePosixPath

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import SuspiciousFileOperation
from django.http import FileResponse, Http404, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from .camera_service import capture_snapshot, registry, test_connection, trigger_clip_capture
from .ai_service import ai_registry
from .forms import CameraForm, create_test_token
from .models import Camera, CameraSnapshot, RecordedClip, SuspectedIncident


@login_required
def camera_list(request):
    cameras = Camera.objects.all()
    counts = {
        "total": cameras.count(),
        "enabled": cameras.filter(enabled=True).count(),
        "connected": cameras.filter(last_connection_status=Camera.ConnectionStatus.CONNECTED).count(),
    }
    return render(request, "cameras/camera_list.html", {"cameras": cameras, "counts": counts})


def _camera_form(request, instance=None):
    form = CameraForm(request.POST or None, instance=instance)
    if request.method == "POST" and form.is_valid():
        camera = form.save(commit=False)
        if form.source_changed:
            camera.last_connection_status = Camera.ConnectionStatus.CONNECTED
            camera.last_connection_message = "Connection test passed."
            camera.last_tested_at = timezone.now()
        camera.save()
        messages.success(request, f'Camera "{camera.name}" saved.')
        return redirect("cameras:list")
    return render(
        request,
        "cameras/camera_form.html",
        {
            "form": form,
            "camera": instance,
            "page_title": "Edit camera" if instance else "Add camera",
            "test_url": reverse("cameras:test"),
        },
    )


@login_required
def camera_create(request):
    return _camera_form(request)


@login_required
def camera_update(request, pk):
    return _camera_form(request, get_object_or_404(Camera, pk=pk))


@login_required
@require_POST
def test_camera_connection(request):
    source_type = request.POST.get("source_type", "")
    camera_id = request.POST.get("camera_id", "")
    existing = get_object_or_404(Camera, pk=int(camera_id)) if camera_id.isdigit() else None
    if source_type not in Camera.SourceType.values:
        return JsonResponse({"ok": False, "message": "Choose USB or RTSP as the source type."}, status=400)

    usb_index = None
    rtsp_url = ""
    if source_type == Camera.SourceType.USB:
        try:
            usb_index = int(request.POST.get("usb_device_index", ""))
            if usb_index < 0:
                raise ValueError
        except (TypeError, ValueError):
            return JsonResponse({"ok": False, "message": "Enter a USB device index of 0 or greater."}, status=400)
    else:
        rtsp_url = request.POST.get("rtsp_url", "").strip()
        if not rtsp_url and existing and existing.source_type == Camera.SourceType.RTSP:
            rtsp_url = existing.rtsp_url
        candidate = Camera(source_type=source_type, rtsp_url=rtsp_url)
        try:
            candidate.clean()
        except Exception:
            return JsonResponse({"ok": False, "message": "Enter a valid rtsp:// or rtsps:// camera URL."}, status=400)

    ok, message = test_connection(source_type, usb_index, rtsp_url)
    payload = {"ok": ok, "message": message}
    if ok:
        payload["test_token"] = create_test_token(source_type, usb_index, rtsp_url)
    if existing:
        existing.last_connection_status = Camera.ConnectionStatus.CONNECTED if ok else Camera.ConnectionStatus.DISCONNECTED
        existing.last_connection_message = message
        existing.last_tested_at = timezone.now()
        existing.save(update_fields=("last_connection_status", "last_connection_message", "last_tested_at"))
    return JsonResponse(payload, status=200 if ok else 422)


@login_required
@require_POST
def camera_toggle(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    camera.enabled = not camera.enabled
    if not camera.enabled:
        camera.last_connection_status = Camera.ConnectionStatus.DISABLED
    elif camera.last_connection_status == Camera.ConnectionStatus.DISABLED:
        camera.last_connection_status = Camera.ConnectionStatus.DISCONNECTED
    camera.save(update_fields=("enabled", "last_connection_status"))
    state = "enabled" if camera.enabled else "disabled"
    messages.success(request, f'Camera "{camera.name}" {state}.')
    return redirect("cameras:list")


@login_required
def camera_delete(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    if request.method == "POST":
        camera_id = camera.pk
        name = camera.name
        camera.delete()
        registry.stop(camera_id)
        messages.success(request, f'Camera "{name}" deleted.')
        return redirect("cameras:list")
    return render(request, "cameras/camera_confirm_delete.html", {"camera": camera})


@login_required
def camera_preview(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    return render(
        request,
        "cameras/camera_preview.html",
        {
            "camera": camera,
            "clips": camera.recordings.all()[:4],
            "snapshots": camera.snapshots.all()[:6],
            "post_event_options": (5, 8, 10, 15),
            "incidents": camera.incidents.all()[:5],
        },
    )


@login_required
@require_GET
def camera_stream(request, pk):
    camera = get_object_or_404(Camera, pk=pk, enabled=True)
    worker = registry.get(camera)
    response = StreamingHttpResponse(worker.frames(), content_type="multipart/x-mixed-replace; boundary=frame")
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response["X-Accel-Buffering"] = "no"
    return response


@login_required
@require_GET
def camera_status(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    worker = None
    if camera.enabled:
        with registry.lock:
            worker = registry.workers.get(camera.pk)
        if worker is None and camera.buffer_enabled:
            worker = registry.get(camera)
            worker.ensure_running()

    if not camera.enabled:
        data = {
            "status": Camera.ConnectionStatus.DISABLED,
            "message": "This camera is disabled.",
            "running": False,
            "current_worker_status": Camera.ConnectionStatus.DISABLED,
            "viewers": 0,
            "fps": 0.0,
            "last_frame_at": None,
            "buffer": {"count": 0, "seconds_cached": 0, "max_seconds": camera.buffer_seconds},
            "buffer_enabled": camera.buffer_enabled,
        }
    elif worker:
        data = worker.snapshot()
    else:
        data = {
            "status": camera.last_connection_status,
            "message": camera.last_connection_message or camera.get_last_connection_status_display(),
            "running": False,
            "current_worker_status": camera.last_connection_status,
            "viewers": 0,
            "fps": 0.0,
            "last_frame_at": None,
            "buffer": {"count": 0, "seconds_cached": 0, "max_seconds": camera.buffer_seconds},
            "buffer_enabled": camera.buffer_enabled,
        }
    data.update(
        enabled=camera.enabled,
        camera_name=camera.name,
        source_type=camera.source_type,
        source_type_label=camera.get_source_type_display(),
        buffer_seconds=camera.buffer_seconds,
        tested_at=camera.last_tested_at.isoformat() if camera.last_tested_at else None,
    )
    return JsonResponse(data)


@login_required
@require_GET
def camera_ai_status(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    worker = None
    if camera.enabled and camera.ai_enabled:
        with ai_registry.lock:
            worker = ai_registry.workers.get(camera.pk)
        if worker is None:
            worker = ai_registry.get(camera)
            worker.start()
    if worker:
        data = worker.snapshot()
    else:
        data = {
            "status": "disabled" if not camera.ai_enabled else "stopped",
            "message": "AI detection is disabled for this camera." if not camera.ai_enabled else "AI detection is stopped.",
            "running": False,
            "inference_fps": 0.0,
            "detected_objects": 0,
            "detected_vehicles": 0,
            "queue_depth": 0,
            "queue_capacity": 2,
            "dropped_frames": 0,
            "latest_tracks": [],
            "latest_suspicion_score": 0.0,
            "latest_temporal_score": None,
            "temporal_status": "disabled",
            "latest_event": None,
        }
    data.update(
        enabled=camera.ai_enabled,
        confidence_threshold=camera.ai_confidence_threshold,
        frame_rate=camera.ai_frame_rate,
        inference_size=camera.ai_inference_size,
        crash_threshold=camera.crash_suspicion_threshold,
        cooldown_seconds=camera.incident_cooldown_seconds,
    )
    return JsonResponse(data)


@login_required
@require_POST
def camera_ai_toggle(request, pk):
    camera = get_object_or_404(Camera, pk=pk, enabled=True)
    camera.ai_enabled = not camera.ai_enabled
    camera.save(update_fields=("ai_enabled",))
    if camera.ai_enabled:
        ai_registry.get(camera).start()
    else:
        ai_registry.stop(camera.pk)
    return JsonResponse({
        "ok": True,
        "enabled": camera.ai_enabled,
        "message": f"AI detection {'enabled' if camera.ai_enabled else 'disabled'}.",
    })


@login_required
@require_GET
def camera_ai_stream(request, pk):
    camera = get_object_or_404(Camera, pk=pk, enabled=True, ai_enabled=True)
    registry.get(camera).ensure_running()
    worker = ai_registry.get(camera)
    worker.start()
    response = StreamingHttpResponse(
        worker.frames(),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response["X-Accel-Buffering"] = "no"
    return response


@login_required
@require_POST
def camera_trigger_clip(request, pk):
    camera = get_object_or_404(Camera, pk=pk, enabled=True)
    try:
        pre_seconds = float(request.POST.get("pre_seconds", camera.buffer_seconds))
        post_seconds = float(request.POST.get("post_seconds", 8.0))
    except (ValueError, TypeError):
        pre_seconds, post_seconds = float(camera.buffer_seconds), 8.0
    pre_seconds = max(0.0, min(pre_seconds, float(camera.buffer_seconds)))
    post_seconds = max(0.0, min(post_seconds, 30.0))
    title = request.POST.get("title", f"Test incident clip - {camera.name}").strip()[:140]
    thread = trigger_clip_capture(
        camera_id=camera.pk,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        trigger_type=RecordedClip.TriggerType.TEST,
        title=title,
    )
    if not thread:
        return JsonResponse({"ok": False, "message": "A clip is already being recorded for this camera."}, status=409)
    return JsonResponse({
        "ok": True,
        "message": f"Capturing test incident clip: up to {pre_seconds:.0f}s pre-event + {post_seconds:.0f}s post-event video.",
    })


@login_required
@require_POST
def camera_take_snapshot(request, pk):
    camera = get_object_or_404(Camera, pk=pk, enabled=True)
    snapshot = capture_snapshot(camera.pk, note=request.POST.get("note", "").strip())
    if not snapshot:
        return JsonResponse({"ok": False, "message": "Could not capture snapshot. No video frame is available yet."}, status=503)
    return JsonResponse({
        "ok": True,
        "message": "Snapshot saved successfully.",
        # Keep the legacy storage URL for API compatibility; it is no longer
        # mounted by Django. Browser access uses the authenticated view URL.
        "url": snapshot.image_file.url,
        "view_url": reverse("cameras:snapshot_view", args=[snapshot.pk]),
        "id": snapshot.pk,
    })


@login_required
def recordings_list(request):
    camera_id = request.GET.get("camera", "")
    clips = RecordedClip.objects.select_related("camera")
    snapshots = CameraSnapshot.objects.select_related("camera")
    if camera_id.isdigit():
        clips = clips.filter(camera_id=int(camera_id))
        snapshots = snapshots.filter(camera_id=int(camera_id))
    return render(request, "cameras/recordings_list.html", {
        "clips": clips,
        "snapshots": snapshots,
        "cameras": Camera.objects.all(),
        "selected_camera": int(camera_id) if camera_id.isdigit() else None,
    })


@login_required
def incident_list(request):
    camera_id = request.GET.get("camera", "")
    incidents = SuspectedIncident.objects.select_related(
        "camera", "snapshot", "video_clip"
    )
    if camera_id.isdigit():
        incidents = incidents.filter(camera_id=int(camera_id))
    return render(request, "cameras/incident_list.html", {
        "incidents": incidents,
        "cameras": Camera.objects.all(),
        "selected_camera": int(camera_id) if camera_id.isdigit() else None,
    })


@login_required
def incident_detail(request, incident_id):
    incident = get_object_or_404(
        SuspectedIncident.objects.select_related("camera", "snapshot", "video_clip"),
        pk=incident_id,
    )
    return render(request, "cameras/incident_detail.html", {"incident": incident})


def _media_response(field_file, *, allowed_prefix, content_type, download=False):
    """Serve only a model-associated file from its expected storage directory."""
    name = (field_file.name or "").replace("\\", "/")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or not name.startswith(allowed_prefix):
        raise Http404("Media file not found")
    try:
        stream = field_file.storage.open(name, "rb")
    except (FileNotFoundError, OSError, ValueError, SuspiciousFileOperation):
        raise Http404("Media file not found")
    response = FileResponse(stream, as_attachment=download, filename=path.name, content_type=content_type)
    response["Cache-Control"] = "private, no-store"
    response["X-Content-Type-Options"] = "nosniff"
    return response


@login_required
@require_GET
def clip_view(request, pk):
    clip = get_object_or_404(RecordedClip, pk=pk)
    return _media_response(clip.video_file, allowed_prefix="clips/", content_type="video/mp4")


@login_required
@require_GET
def clip_download(request, pk):
    clip = get_object_or_404(RecordedClip, pk=pk)
    return _media_response(clip.video_file, allowed_prefix="clips/", content_type="video/mp4", download=True)


@login_required
@require_GET
def clip_thumbnail(request, pk):
    clip = get_object_or_404(RecordedClip, pk=pk)
    return _media_response(clip.thumbnail, allowed_prefix="thumbnails/", content_type="image/jpeg")


@login_required
@require_GET
def snapshot_view(request, pk):
    snapshot = get_object_or_404(CameraSnapshot, pk=pk)
    return _media_response(snapshot.image_file, allowed_prefix="snapshots/", content_type="image/jpeg")


@login_required
@require_GET
def snapshot_download(request, pk):
    snapshot = get_object_or_404(CameraSnapshot, pk=pk)
    return _media_response(snapshot.image_file, allowed_prefix="snapshots/", content_type="image/jpeg", download=True)


@login_required
@require_POST
def clip_delete(request, pk):
    clip = get_object_or_404(RecordedClip, pk=pk)
    if clip.video_file:
        clip.video_file.delete(save=False)
    if clip.thumbnail:
        clip.thumbnail.delete(save=False)
    clip.delete()
    messages.success(request, "Recorded clip deleted.")
    return redirect("cameras:recordings")


@login_required
@require_POST
def snapshot_delete(request, pk):
    snapshot = get_object_or_404(CameraSnapshot, pk=pk)
    if snapshot.image_file:
        snapshot.image_file.delete(save=False)
    snapshot.delete()
    messages.success(request, "Snapshot deleted.")
    return redirect("cameras:recordings")
