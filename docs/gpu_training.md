# Fine-tuning the detector on a cloud GPU

Baseline B2 fine-tunes a COCO-pretrained YOLO26 detector on VisDrone-DET. One epoch takes about an hour on a laptop CPU and minutes on a GPU, so training runs in a Kaggle notebook (any GPU notebook works the same way). Nothing in the repository uploads data; the package is uploaded by hand.

## 1. Package the data

```bash
python scripts/package_for_gpu.py
```

This writes `cache/gpu_package/visdrone_yolo.zip` (about 1.6 GB: 6,414 training and 527 validation images in YOLO format, with the 78 images that overlap the tracking test videos removed, D-308) and `manifest.json` with the counts and the zip's SHA-256.

## 2. Upload it as a private dataset

VisDrone has no stated licence, so the package is uploaded as a **private** Kaggle dataset (for example `visdrone-yolo`) and never shared.

## 3. Train

In a new notebook: enable a GPU accelerator and internet access, add the private dataset as an input, and paste one of the training scripts into a cell:

| Script | Model | Input size |
|---|---|---|
| `gpu/train_b2_kaggle.py` | YOLO26n | 640 |
| `gpu/train_b2_kaggle_n960.py` | YOLO26n | 960 (the model used in the results) |
| `gpu/train_b2_kaggle_s640.py` | YOLO26s | 640 |

Run it with **Save & Run All** so it continues in the background. The script stops with an error if no GPU is visible. Training runs for at most 50 epochs and stops early once validation mAP has not improved for 10 (D-315).

## 4. Bring the weights back

Download the run's output archive and unzip it into `cache/train/`, so that `cache/train/<run>/weights/best.pt` exists. Validation scoring, tracking and the comparisons then run locally (`docs/runbook.md`, Phase 3).

## SynDrone fine-tuning (version 2)

The version 1 detector is fine-tuned on SynDrone development towns so that detection on synthetic imagery can be reported two ways: zero-shot and fine-tuned. To avoid scoring a model on its own training towns, there are three leave-one-town-out folds (D-707): each trains on two development towns, selects the epoch on the last 20 % of frames of those towns, and is scored on the third. Calibration and test towns are never used.

1. Build the package (after the development towns' truth exists): `python scripts/syndrone_package_for_gpu.py` writes `cache/gpu_package/syndrone_dev_yolo.zip` (every 5th frame of each camera; ignored objects greyed out) and `syndrone_manifest.json` with counts and the SHA-256.
2. Upload two **private** Kaggle datasets: the zip, and `cache/train/B2_yolo26n_960/weights/best.pt` renamed `B2_yolo26n_960_best.pt` (the starting weights, so the only change from zero-shot is SynDrone training; copy it into `cache/gpu_package/` under that name so both uploads sit together). Before uploading, check the zip's SHA-256 against the manifest. Kaggle may unpack the zip on upload, and the notebook accepts either form.
3. In a GPU notebook with both attached, paste `gpu/train_syndrone_kaggle.py` into one cell and use **Save & Run All**. It trains the three folds in turn, resumes after a session timeout, and refuses to run without a GPU.
4. Download `syndrone_ft_outputs.zip` and unzip it into `cache/train/`, so that `cache/train/syndrone_ft_outputs/syndrone_ft_minus_<Town>/weights/best.pt` exists. Scoring then runs locally (`docs/runbook.md`, Phase 7).

### The version 2 detector (D-FINE N)

The same three folds, for the detector chosen in version 2 (D-740), so the comparison with version 1's fine-tune changes only the detector.

1. Build the package from version 1's: `python scripts/syndrone_coco_package.py` writes `cache/gpu_package/syndrone_dev_coco.zip` (the same images and folds in COCO format) and `syndrone_coco_manifest.json` with counts and the SHA-256.
2. Upload two **private** Kaggle datasets: the zip, and the detector's VisDrone weights `cache/train/dfine_n_640/checkpoint_best.pth`, copied as `cache/gpu_package/dfine_n_640_visdrone_best.pth` (the notebook finds it by that name). Check the zip's SHA-256 against the manifest before uploading.
3. In a GPU notebook with both attached and **Internet** on, paste `gpu/train_dfine_syndrone_kaggle.py` into one cell and use **Save & Run All**. It trains the folds in turn and does not start a fold that would not finish within 11 hours; if it prints "NOT RUN THIS SESSION", set `FOLDS` on its first settings line to the folds it lists and run again in a new session.
4. Download each `dfine_syndrone_minus_<Town>.zip` and unzip it into `cache/train/`, so that `cache/train/dfine_syndrone_minus_<Town>/checkpoint_best.pth` exists. Export, parity check and scoring run locally.

## Detector candidates (version 2)

The version 1 detector's library is AGPL-3.0, so version 2 trains permissively licensed candidates on the same VisDrone data (`docs/model_and_data.md`, detector candidates). The first is RF-DETR (Apache-2.0); the others follow the same pattern.

1. Build the package: `python scripts/visdrone_coco_package.py` writes `cache/gpu_package/visdrone_det_coco.zip` (VisDrone DET train and val in COCO format; the same images and labels as version 1's training set) and `visdrone_coco_manifest.json` with counts and the SHA-256.
2. Upload the zip as a **private** Kaggle dataset, once; every candidate uses it. Kaggle may unpack the zip on upload, and the notebook accepts either form.
3. In a GPU notebook with the dataset attached, paste `gpu/train_rfdetr_kaggle.py` into one cell, set `RUN` on its first settings line (for example `rfdetr_nano_640`), and use **Save & Run All**. One run per session; training stops cleanly after 11 hours if it has not finished, so the outputs are never lost to Kaggle's 12-hour limit.
4. Download `<run>.zip` and unzip it into `cache/train/`, so that `cache/train/<run>/checkpoint_best_total.pth` exists.
5. D-FINE works the same way with `gpu/train_dfine_kaggle.py` (runs `dfine_n_640`, `dfine_n_960`, `dfine_s_640`); the notebook needs **Internet** on, because it fetches the authors' code at a fixed commit and their COCO-only start weights. Its zip holds `checkpoint_best.pth`.
6. Score each run locally with one command: `bash scripts/queues/score_candidate.sh <run> <input size> <rfdetr|dfine>`. It exports to ONNX and checks parity against the training code, scores detection on VisDrone MOT val, caches detections, and tunes the project's tracker on them (log in `cache/logs/candidates/`). The training libraries live in their own environments under `cache/venvs/` (D-717).
