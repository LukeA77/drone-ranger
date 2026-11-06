# Fine-tune the version 1 detector on SynDrone development towns, one
# leave-one-town-out fold after another. Paste into ONE Kaggle notebook cell (GPU on).
#
# Inputs (both PRIVATE Kaggle datasets, uploaded by hand):
#   - syndrone_dev_yolo.zip  (scripts/syndrone_package_for_gpu.py)
#   - B2_yolo26n_960_best.pt (cache/train/B2_yolo26n_960/weights/best.pt, the version 1
#     detector), so the only difference from the zero-shot detector is SynDrone training.
# Output: /kaggle/working/syndrone_ft_outputs.zip (weights and curves of each fold). Put it
# in the project's cache/train/ (docs/gpu_training.md, SynDrone section).
#
# Settings as version 1 (D-315): image size 960, library defaults, seed 0, at most 50 epochs
# with early stopping after 10 without improvement on the fold's selection frames.

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ULTRALYTICS_VERSION = "8.4.182"
IMGSZ = 960
FOLDS = ["Town01", "Town07", "Town10HD"]  # the held-out town of each fold
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", f"ultralytics=={ULTRALYTICS_VERSION}"],
    check=True,
)

roots = [Path("/kaggle/input"), Path("/content"), WORK]
data_dir = next(
    (
        p.parent
        for r in roots
        if r.exists()
        for p in r.rglob("fold_Town01.yaml")
        if p.parent.name == "syndrone_dev_yolo"
    ),
    None,
)
if data_dir is None:
    z = next((p for r in roots if r.exists() for p in r.rglob("syndrone_dev_yolo.zip")), None)
    if z is None:
        raise SystemExit("dataset not found: attach the syndrone_dev_yolo dataset")
    with zipfile.ZipFile(z) as f:
        f.extractall(WORK)
    data_dir = WORK / "syndrone_dev_yolo"
start = next((p for r in roots if r.exists() for p in r.rglob("B2_yolo26n_960_best.pt")), None)
if start is None:
    raise SystemExit("start weights not found: attach B2_yolo26n_960_best.pt")

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("NO GPU FOUND. Session options -> Accelerator -> GPU, then run again.")

OVERRIDES = {"epochs": 50, "patience": 10, "cache": "ram"}
info = {}
for held in FOLDS:
    name = f"syndrone_ft_minus_{held}"
    yaml_path = WORK / f"fold_{held}_run.yaml"
    yaml_path.write_text(f"path: {data_dir}\n" + (data_dir / f"fold_{held}.yaml").read_text())
    if (WORK / name / "weights" / "best.pt").exists() and (WORK / name / "done").exists():
        continue
    last = WORK / name / "weights" / "last.pt"
    t0 = time.time()
    if last.exists():
        YOLO(last).train(resume=True)
    else:
        YOLO(str(start)).train(
            data=str(yaml_path),
            imgsz=IMGSZ,
            project=str(WORK),
            name=name,
            exist_ok=True,
            device=0,
            seed=0,
            deterministic=True,
            **OVERRIDES,
        )
    (WORK / name / "done").write_text("ok")
    info[held] = {"wall_seconds": time.time() - t0}

meta = {
    "ultralytics": ULTRALYTICS_VERSION,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "imgsz": IMGSZ,
    "overrides": OVERRIDES,
    "start_weights": start.name,
    "folds": info,
}
(WORK / "syndrone_ft_run.json").write_text(json.dumps(meta, indent=1))
keep = WORK / "syndrone_ft_outputs"
keep.mkdir(exist_ok=True)
for held in FOLDS:
    src = WORK / f"syndrone_ft_minus_{held}"
    if src.exists():
        shutil.copytree(src, keep / src.name, dirs_exist_ok=True)
shutil.copy(WORK / "syndrone_ft_run.json", keep)
out = shutil.make_archive(str(keep), "zip", WORK, keep.name)
print("done; download", out, "(size", os.path.getsize(out) // 2**20, "MB)")
