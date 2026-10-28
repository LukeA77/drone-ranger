# Fine-tune the detector (baseline B2) on a cloud GPU. Paste into ONE Kaggle notebook cell
# (or Colab; see docs/gpu_training.md). Settings match the local pipeline: YOLO26n, image
# size 960, library default hyperparameters, seed 0. Library version pinned to the one used
# locally so results are comparable.
#
# Inputs: the private dataset made from cache/gpu_package/visdrone_yolo.zip.
# Outputs: /kaggle/working/B2_yolo26n_960_outputs.zip (weights + training curves). Download
# it and put it in the project's cache/train/ folder (instructions in docs/gpu_training.md).

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ULTRALYTICS_VERSION = "8.4.182"
RUN_NAME = "B2_yolo26n_960"
MODEL = "yolo26n.pt"
IMGSZ = 960
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                f"ultralytics=={ULTRALYTICS_VERSION}"], check=True)

# Find the dataset: Kaggle unzips uploads; a raw zip is extracted here if needed.
roots = [Path("/kaggle/input"), Path("/content"), WORK]
data_dir = next((p.parent for r in roots if r.exists()
                 for p in r.rglob("visdrone.yaml") if p.parent.name == "visdrone_yolo"), None)
if data_dir is None:
    z = next((p for r in roots if r.exists() for p in r.rglob("visdrone_yolo.zip")), None)
    if z is None:
        raise SystemExit("dataset not found: attach the visdrone_yolo dataset to this notebook")
    with zipfile.ZipFile(z) as f:
        f.extractall(WORK)
    data_dir = WORK / "visdrone_yolo"
n_train = len(list((data_dir / "images" / "train").iterdir()))
n_val = len(list((data_dir / "images" / "val").iterdir()))
print(f"dataset at {data_dir}: {n_train} train, {n_val} val images (expect 6414 and 527)")

# Dataset YAML with an absolute path for this machine (the input folder is read-only).
yaml_path = WORK / "visdrone_run.yaml"
yaml_path.write_text(f"path: {data_dir}\n" + (data_dir / "visdrone.yaml").read_text())

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

if not torch.cuda.is_available():
    # Refuse to train on CPU: an earlier run silently used the CPU (about 1 hour per epoch).
    raise SystemExit("NO GPU FOUND. In the notebook panel set Session options -> Accelerator "
                     "-> GPU (T4 or P100), then run this cell again.")
device = 0
print("device:", torch.cuda.get_device_name(0))

# Changes from library defaults, with reasons (docs/decisions.md D-315):
#  - epochs 50 (default 100) with early stopping patience 10 (default 100): fine-tuning a
#    pretrained detector converges well before 100 epochs, and the GPU quota is 30 h/week;
#    training stops when DET-val mAP has not improved for 10 epochs (DET val is not a test
#    split, so it may be used for early stopping). The earlier MOT17 run used 30 epochs.
#  - cache="ram": keep decoded images in memory so the GPU is not starved by JPEG decoding.
OVERRIDES = {"epochs": 50, "patience": 10, "cache": "ram"}

last = WORK / RUN_NAME / "weights" / "last.pt"
t0 = time.time()
if last.exists():  # resume after a session time-out
    YOLO(last).train(resume=True)
else:
    YOLO(MODEL).train(data=str(yaml_path), imgsz=IMGSZ, project=str(WORK), name=RUN_NAME,
                      exist_ok=True, device=device, seed=0, deterministic=True, **OVERRIDES)
info = {"run": RUN_NAME, "model": MODEL, "imgsz": IMGSZ, "ultralytics": ULTRALYTICS_VERSION,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "wall_seconds": time.time() - t0, "overrides_vs_library_defaults": OVERRIDES,
        "platform": "kaggle" if Path("/kaggle").exists() else "other"}
(WORK / RUN_NAME / "run.json").write_text(json.dumps(info, indent=1))
out = shutil.make_archive(str(WORK / f"{RUN_NAME}_outputs"), "zip", WORK, RUN_NAME)
print("done; download", out, "(size", os.path.getsize(out) // 2**20, "MB)")
