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
