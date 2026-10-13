# CCD footage acquisition for CrashGuard

The official CCD Drive folder exposes the training footage as two archives:

| File | Verified size | Purpose |
|---|---:|---|
| `Crash-1500.zip` | 793,091,291 bytes | 1,500 crash clips |
| `Normal.zip` | 6,445,547,827 bytes | 3,000 normal clips |
| `Crash-1500.txt` | 276,792 bytes | Official 50-frame crash labels and source IDs |

The archives are deliberately not downloaded to the development laptop.

## Preferred: let Colab acquire the files

1. Upload `CrashGuard_Colab_Training_Bundle.zip` and
   `CrashGuard_Milestone_3_1_Colab.ipynb` to **My Drive**.
2. In Google Drive, right-click the notebook and select **Open with → Google
   Colaboratory**. If Colaboratory is not listed, choose **Open with → Connect
   more apps**, search for **Colaboratory**, and install it.
3. In Colab, choose **Runtime → Change runtime type → T4 GPU** (or another
   available GPU), then select **Save**.
4. Run the first cell. At the Drive prompt, select the intended Google account,
   review the requested Drive access, and select **Allow**.
5. Review the CCD repository description and the terms that apply to the
   underlying YouTube and BDD100K footage.
6. In the dataset cell, change `ACCEPT_CCD_DATASET_TERMS = False` to `True`,
   then run it. Only the two footage archives and official annotation file are
   requested from the official folder.

Archives are staged on ephemeral Colab storage and extracted to:

```text
My Drive/CrashGuardData/CarCrash/videos/Crash-1500/*.mp4
My Drive/CrashGuardData/CarCrash/videos/Normal/*.mp4
My Drive/CrashGuardData/CarCrash/videos/Crash-1500.txt
```

Allow roughly 15 GB of free Drive space for extracted footage, cached motion
features, checkpoints, and evaluation artifacts.

## Manual Drive shortcut fallback

Use this only if Google throttles the notebook download:

1. Open the official folder:
   <https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhIgsXWtj0DA2MR-F?usp=sharing>
2. Sign in to the same Google account used by Colab.
3. Select `Crash-1500.zip`, `Normal.zip`, and `Crash-1500.txt`.
4. Right-click the selection and choose **Organize → Add shortcut** (the label
   may appear as **Add shortcut to Drive**).
5. Choose **My Drive → CrashGuardData**, then select **Add**.
6. Rerun the notebook dataset cell. It searches `CrashGuardData` for verified
   files before attempting another download.

After acquisition, the notebook verifies archive sizes and CRCs, exactly 1,500
crash and 3,000 normal MP4 files, non-empty files, the annotation file, and an
OpenCV decoding sample. It then creates the video-disjoint manifest and runs a
balanced 24-video real-data verification job before full extraction or
training.
