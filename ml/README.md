# CrashGuard custom temporal model

Milestone 3.1 keeps object detection and crash recognition separate. The
existing COCO-pretrained YOLO11 detector finds and tracks traffic objects. A
small GRU learns from 16 normalized tracked-motion features over 12 sampled
frames. Its score is supporting evidence for the existing conservative crash
heuristics and is never presented as a confirmed crash or calibrated real-world
probability.

## Dataset inventory acquired locally

Only official code and annotations were downloaded to protect local storage:

| Source | Local material | Verified counts | Full footage | Terms |
|---|---:|---:|---:|---|
| DoTA | Official repository and 10,357,806-byte annotation ZIP | 4,677 videos: 3,275 published train and 1,402 published validation; 4,677 per-video JSON files | About 55 GB extracted frames; intentionally not downloaded locally | Repository code is MIT; verify/accept the dataset footage terms and use for research only unless the owners grant broader rights |
| CCD | Official repository | 1,500 crash clips and 3,000 normal clips documented by the authors | Google Drive access required; intentionally not downloaded locally | Repository code is MIT; footage originates from YouTube and BDD100K, so do not assume the code license grants redistribution rights |
| CADP | Official 568,340-byte annotation JSON | 226 video keys in the current JSON; the authors describe 1,416 segments and 205 fully annotated segments | Not downloaded | Non-commercial research use; prior permission is required for other use |

Sources:

- <https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly>
- <https://github.com/Cogito2012/CarCrashDataset>
- <https://ankitshah009.github.io/accident_forecasting_traffic_camera.html>

Do not combine or redistribute footage until each source's current terms have
been reviewed. Dataset repository licenses may cover code without covering the
underlying videos.

## Colab workflow

Open `CrashGuard_Milestone_3_1_Colab.ipynb` in Google Colab, choose a GPU
runtime, and authorize the Drive mount. The notebook locates or extracts the
training project automatically and requires an explicit dataset-terms flag
before it downloads CCD into Drive.

For a minimal upload, place `CrashGuard_Colab_Training_Bundle.zip` in My Drive
and upload/open the notebook in Colab. The notebook automatically extracts the
bundle to `MyDrive/CrashGuardTraining`; it does not contain `.venv`, media,
dataset footage, credentials, or synthetic smoke artifacts.

The pipeline performs these stages:

1. Validate GPU and Drive space.
2. Acquire the official dataset into Drive after terms acceptance.
3. Build deterministic group-based train/validation/test manifests. Clips from
   the same original YouTube video always remain in one split.
4. Run pretrained YOLO11n and production-compatible tracking to cache compact
   motion features. No pseudo-detections are converted into ground-truth boxes.
5. Train and checkpoint a small GRU.
6. Select its threshold only on validation data.
7. Report held-out TP, FP, TN, FN, precision, recall, F1, confusion matrix,
   accuracy, false-positive rate, detection delay, and false alarms per normal
   minute. Per-window held-out predictions are saved for manual false-positive
   review against the source footage.
8. Compare against the Milestone 3 heuristic score on the same windows.
9. Export `crash_temporal.onnx` and `crash_temporal.json` to Drive.

Before the full extraction, the notebook runs a balanced 24-video real-data
verification job (four crash and four non-crash videos from each split). Its
metrics validate the real-data toolchain only and are kept separate from the
full held-out evaluation.

### CCD acquisition layout

The official Drive folder packages footage as two archives rather than as
individually downloadable clips:

- `Crash-1500.zip`: 793,091,291 bytes
- `Normal.zip`: 6,445,547,827 bytes
- `Crash-1500.txt`: 276,792 bytes

The notebook downloads only those three official files. Archives are staged on
ephemeral Colab disk and extracted into
`MyDrive/CrashGuardData/CarCrash/videos`, so neither large archive is written to
the development laptop and Drive does not retain an extra archive copy. It
validates archive byte counts, ZIP CRCs, final video counts, non-empty files,
and a 20-video OpenCV decoding sample before manifest creation.

If Google Drive throttles `gdown`, sign in at the official CCD Drive link,
right-click each of `Crash-1500.zip`, `Normal.zip`, and `Crash-1500.txt`, choose
**Organize → Add shortcut**, and place the shortcuts under
`My Drive/CrashGuardData`. The Colab runtime must still be authorized to the
same Google account.

DoTA manifest boundaries are derived from each official per-frame
`accident_id`, with `anomaly_end` treated as the source's exclusive summary
field. Ignored clips are excluded. This matters because one non-ignored source
file has a stale summary end beyond its final frame while its frame labels are
valid and contiguous; the preparation script never invents a replacement
label.

Training writes `last.pt` after every epoch. Rerun the notebook training cell
with `--resume` to restore the model, optimizer, early-stopping state, and
history. Per-video feature files are also restartable.

## Local functional smoke test

With the optional training dependencies installed, run:

```powershell
.\.venv\Scripts\python.exe ml\training\smoke_test.py --output ml\smoke_output
```

This intentionally tiny synthetic test validates gradients, validation,
checkpoint recovery, ONNX export/runtime parity, Django loading, and the
disabled-model fallback. Its accuracy and F1 are not real crash-detection
metrics and must not be used for model selection or deployment claims.

CCD and DoTA do not provide consistent, comprehensive vehicle-class bounding
boxes suitable for honest YOLO fine-tuning. CCD's supplied boxes are detector
outputs, and DoTA spatial labels mark anomaly regions. The notebook therefore
keeps the pretrained detector and trains only temporal crash recognition.

## Django deployment

Copy both exported files to a local directory, then run:

```powershell
$env:CAMERA_AI_TEMPORAL_MODEL_PATH = "C:\carrcrash\models\crash_temporal.onnx"
.\.venv\Scripts\Activate.ps1
python manage.py check
python manage.py runserver
```

The JSON sidecar contains the exact feature schema, sequence length, and
validation-selected decision threshold. A mismatch disables the temporal model
without stopping YOLO, normal preview, or the heuristic pipeline.
