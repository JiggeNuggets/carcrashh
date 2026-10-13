# CrashGuard / Roadwatch vehicle monitoring prototype

This Django project manages authorized USB and RTSP cameras, maintains bounded rolling buffers, and optionally analyzes sampled frames with YOLO and ByteTrack. Milestone 3 generates **suspected incident candidates requiring human verification**. A heuristic suspicion score is not a calibrated crash probability and does not confirm an accident, injury, or damage.

The project does not contact emergency services and does not include dispatch, maps, responder alerts, or automated accident confirmation.

## Features

- Login-protected camera, recording, AI-control, and incident pages
- Shared USB/RTSP capture workers and live MJPEG previews
- Configurable 15/20/30-second rolling buffers
- Manual snapshots and pre/post-event MP4 test clips
- Worker health, FPS, reconnect, and resource-cleanup monitoring
- Optional per-camera YOLO object detection for people, cars, motorcycles, buses, and trucks
- Persistent ByteTrack IDs with position history, image-plane velocity, direction, and occlusion state
- Temporal collision, sudden-motion, abnormal-direction, and post-event stationary heuristics
- Per-camera confidence, inference FPS, resolution, suspicion threshold, cooldown, and post-event settings
- Bounded two-frame AI queues that discard stale work under load
- Separate normal and annotated AI previews using one underlying camera connection
- Automatic suspected-incident records, snapshots, and asynchronous clips
- Authenticated incident evidence and a basic incident list/detail view
- Safe recorded-video test command with repeatable MP4 processing
- Per-camera recording retention limits

## Base installation

Use 64-bit Python 3.12, 3.13, or 3.14. This project was validated on Windows
with Python 3.14.3; Django 6.0 and current PyTorch Windows wheels officially
support Python 3.14.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Open <http://127.0.0.1:8000>, sign in, and add an authorized camera.

## Optional AI installation

```powershell
pip install -r requirements-ai.txt
```

`requirements-ai.txt` installs Ultralytics, PyTorch, Torchvision, and ByteTrack's
`lap` association dependency in addition to the base requirements. The current
PyTorch Windows guidance supports Python 3.10-3.14. Confirm the exact wheel
offered for the target OS and accelerator using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/).

The requirements intentionally install `opencv-python`, not a second headless
OpenCV distribution. OpenCV documents that only one of its packages should be
installed in an environment because they all provide the same `cv2` namespace.
If this project environment predates that change, remove the old package before
installing the AI stack:

```powershell
python -m pip uninstall -y opencv-python-headless
python -m pip install -r requirements-ai.txt
python -m pip check
```

Validated local CPU stack on Python 3.14.3: Django 6.0.9, NumPy 2.5.4,
OpenCV 4.14.0, Ultralytics 8.4.176, PyTorch 2.14.1+cpu, and Torchvision
0.29.1+cpu. Compatible patch releases selected by the requirement ranges may
be newer on a fresh installation.

### Licensing

Ultralytics documents its open-source package and standard models under **AGPL-3.0**, with an Enterprise license offered for deployments that cannot comply with AGPL requirements. Review the [Ultralytics licensing guidance](https://docs.ultralytics.com/help/contributing/#what-is-the-purpose-of-the-agpl-30-license) before distribution or commercial deployment. This repository does not grant a separate license for Ultralytics code or weights.

### Model weights

Unexpected model downloads are disabled by default. Put the lightweight `yolo11n.pt` weights at `C:\carrcrash\yolo11n.pt`, or configure another local path:

```powershell
$env:CAMERA_AI_MODEL_PATH = "C:\models\yolo11n.pt"
```

For a one-time development download through Ultralytics:

```powershell
$env:CAMERA_AI_ALLOW_MODEL_DOWNLOAD = "1"
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
$env:CAMERA_AI_ALLOW_MODEL_DOWNLOAD = "0"
```

The application loads the model once per active camera AI worker, never once per frame. If Ultralytics, PyTorch, or the weights are unavailable, the AI worker reports `Model unavailable`; normal camera streaming continues.

### CPU and CUDA

CPU is selected automatically when CUDA is unavailable:

```powershell
$env:CAMERA_AI_DEVICE = "cpu"
```

For an NVIDIA GPU, install the matching PyTorch CUDA build using the official PyTorch selector, verify `torch.cuda.is_available()`, and then configure:

```powershell
$env:CAMERA_AI_DEVICE = "cuda:0"
```

Do not assume a fixed inference rate. Measure the selected model, resolution, frame rate, footage, CPU/GPU, and camera count on the deployment computer.

## Camera AI configuration

Edit a camera and open the **AI detection** section:

- Enable AI detection
- Detection confidence: `0.10`–`0.95`
- Processing rate: 1, 2, or 5 sampled FPS
- Inference resolution: 320, 480, or 640 pixels
- Crash suspicion threshold: `0.50`–`0.95`
- Incident cooldown: 30–300 seconds
- AI post-event footage: 5, 10, or 15 seconds

Settings propagate to an existing AI worker without restarting the camera. Disabling AI stops only inference; the original live preview and rolling buffer continue.

On the live page, use **Normal live view** or **AI detection view**. The AI view shows boxes, labels, confidence, and tracking IDs. Health metrics show inference FPS, queue depth, dropped frames, detected vehicle count, and the latest heuristic score.

## Testing USB and RTSP cameras

1. Add and successfully test the source.
2. Keep the rolling buffer enabled.
3. Edit the camera, enable AI, and choose conservative initial settings (320 px, 1–2 FPS, threshold 0.75 or higher).
4. Open the live preview and confirm the normal feed first.
5. Switch to AI detection view and inspect worker status and overlays.
6. Review candidates on **AI incidents**; do not treat them as confirmed crashes.

USB index `0` is usually the first camera attached to the Django server. RTSP URLs must be reachable from that server. Credentials are not rendered in pages, JSON status responses, or application logs.

## Testing a recorded video

Only files under the configured safe directory are accepted. Copy properly licensed footage into it:

```powershell
New-Item -ItemType Directory -Force media\ai_test_videos
Copy-Item C:\licensed-footage\traffic.mp4 media\ai_test_videos\traffic.mp4
python manage.py process_ai_video traffic.mp4 --camera 1
```

Add `--display` for a local OpenCV window and press `q` to stop:

```powershell
python manage.py process_ai_video traffic.mp4 --camera 1 --display
```

The command uses the selected camera's AI settings, runs the same detector/tracker/analyzer stack, releases the video at EOF, and can process the same video repeatedly. It does not assume the video contains a crash. Candidate incidents use source `Recorded video test`.

To use another safe root:

```powershell
$env:CAMERA_AI_TEST_VIDEO_ROOT = "D:\approved-ai-test-videos"
```

## Worker and deployment configuration

The development autoreloader starts workers only in its serving child process. Tests, migrations, and other management commands never auto-open cameras or models. For a single-process production WSGI deployment:

```powershell
$env:CAMERA_WORKERS_AUTOSTART = "1"
```

Multiple WSGI processes would each own independent capture/model workers. A production multi-process system therefore needs a dedicated capture/inference service or broker.

Other settings:

```powershell
$env:DJANGO_SECRET_KEY = "a-long-random-secret"
$env:DJANGO_DEBUG = "0"
$env:DJANGO_ALLOWED_HOSTS = "monitor.example.internal"
$env:CAMERA_MAX_CLIPS_PER_CAMERA = "50"
$env:CAMERA_MAX_SNAPSHOTS_PER_CAMERA = "100"
```

Saved media is not mounted through Django's public development `/media/` route. Evidence is served through authenticated model-based endpoints with safe path validation. Do not configure a production reverse proxy to expose `MEDIA_ROOT` publicly.

## Validation

```powershell
python -m compileall -q cameras crash_monitor
python manage.py check
python manage.py makemigrations --check --dry-run
python manage.py migrate
python manage.py test
```

Unit tests mock the detector and require no camera, GPU, model download, or network. Passing unit tests validates software behavior—not crash-detection accuracy. Precision, recall, false positives, ID stability, and threshold calibration must be measured on representative labeled footage from each intended camera viewpoint.

## Known limitations

- COCO-pretrained YOLO detects general traffic objects; it is not a crash-classification model.
- Image-plane velocity is pixels per second, never km/h without camera calibration.
- Perspective, camera shake, occlusion, congestion, intersections, weather, and night scenes can create false signals or missed events.
- Provisional severity is a heuristic label and cannot assess injury or physical damage.
- ByteTrack IDs are temporary and may switch after long occlusion.
- No physical camera, labeled crash dataset, or model weights are bundled with the repository.
- FFmpeg improves browser-compatible H.264 output when installed; otherwise OpenCV MP4 output is used.
- This prototype does not implement Milestone 4 triage/auditing or Milestone 5 emergency response.
