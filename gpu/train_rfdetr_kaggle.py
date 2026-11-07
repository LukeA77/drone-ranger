# Train an RF-DETR detector candidate on VisDrone DET. Paste into ONE Kaggle notebook cell
# (GPU on), set RUN below, and use Save & Run All. One run per session.
#
# Input (a PRIVATE Kaggle dataset, uploaded by hand): visdrone_det_coco.zip
#   (scripts/visdrone_coco_package.py; zipped or unpacked by Kaggle, both work).
# Output: /kaggle/working/<run>.zip (best checkpoint, metrics.csv, settings, run.json).
#   Put it in the project's cache/train/ and unzip (docs/gpu_training.md, candidates).
#
# Settings: library defaults except input size, at most 50 epochs with early stopping after
# 10 without improvement on VisDrone DET val (as the version 1 detector, D-315), seed 0.
# Training stops cleanly after MAX_HOURS, keeping the checkpoints, so a session never hits
# Kaggle's hard time limit and loses its outputs.
# Local smoke test: SMOKE=4 python gpu/train_rfdetr_kaggle.py <data dir> (4 images, CPU).

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

RUN = "rfdetr_nano_640"  # one of RUNS below
RUNS = {
    "rfdetr_nano_640": ("RFDETRNano", 640),
    "rfdetr_nano_960": ("RFDETRNano", 960),
    "rfdetr_small_640": ("RFDETRSmall", 640),
    "rfdetr_small_960": ("RFDETRSmall", 960),
}
RFDETR_VERSION = "1.12.0"
EPOCHS, PATIENCE, SEED = 50, 10, 0
MAX_HOURS = 11.0  # Kaggle sessions end at 12 h; the rest covers setup and zipping

SMOKE = int(os.environ.get("SMOKE", "0"))
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
roots = [Path("/kaggle/input"), WORK]

if not SMOKE:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"rfdetr[train,onnx]=={RFDETR_VERSION}"],
        check=True,
    )


def find_data() -> Path:
    if SMOKE:
        return Path(sys.argv[1])
    for r in roots:
        if r.exists():
            for p in r.rglob("_annotations.coco.json"):
                if p.parent.name == "train" and (p.parent.parent / "valid").is_dir():
                    return p.parent.parent
    z = next((p for r in roots if r.exists() for p in r.rglob("visdrone_det_coco.zip")), None)
    if z is None:
        raise SystemExit("dataset not found: attach the visdrone_det_coco dataset")
    with zipfile.ZipFile(z) as f:
        f.extractall(WORK)
    return WORK / "visdrone_det_coco"


def subset(src: Path, dst: Path, n: int) -> Path:
    for sub in ("train", "valid"):
        coco = json.loads((src / sub / "_annotations.coco.json").read_text())
        coco["images"] = coco["images"][:n]
        ids = {i["id"] for i in coco["images"]}
        coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in ids]
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for i in coco["images"]:
            shutil.copy2(src / sub / i["file_name"], dst / sub / i["file_name"])
        (dst / sub / "_annotations.coco.json").write_text(json.dumps(coco))
    return dst


def stop_after(hours: float) -> None:
    """Every training trainer RF-DETR builds stops cleanly after `hours` (Lightning max_time)."""
    import rfdetr.training as rt

    build = rt.build_trainer
    s = int(hours * 3600)
    limit = f"{s // 86400:02d}:{s % 86400 // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"

    def build_with_limit(*args: object, **kwargs: object) -> object:
        if kwargs.get("include_training_callbacks", True):
            kwargs.setdefault("max_time", limit)
        return build(*args, **kwargs)

    rt.build_trainer = build_with_limit


from importlib.metadata import version  # noqa: E402

import rfdetr  # noqa: E402
import torch  # noqa: E402

if not SMOKE and not torch.cuda.is_available():
    raise SystemExit("NO GPU FOUND. Session options -> Accelerator -> GPU, then run again.")

cls_name, res = RUNS[RUN]
out = WORK / RUN
out.mkdir(parents=True, exist_ok=True)
data = find_data()
if SMOKE:
    data = subset(data, out / "smoke_data", SMOKE)
t_start = time.time()
stop_after(MAX_HOURS if not SMOKE else 0.05)
torch.manual_seed(SEED)
settings = {
    "dataset_dir": str(data),
    "output_dir": str(out),
    "resolution": res,
    "epochs": EPOCHS if not SMOKE else 2,
    "early_stopping": True,
    "early_stopping_patience": PATIENCE,
    "seed": SEED,
    "batch_size": "auto" if not SMOKE else 2,
    "device": "cuda:0" if not SMOKE else "cpu",
}
if SMOKE:
    settings["num_workers"] = 0
model = getattr(rfdetr, cls_name)()
model.train(**settings)
metrics = out / "metrics.csv"
run = {
    "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "run": RUN,
    "model": cls_name,
    "settings": {k: v for k, v in settings.items() if k not in ("dataset_dir", "output_dir")},
    "max_hours": MAX_HOURS,
    "smoke": SMOKE,
    "elapsed_s": round(time.time() - t_start, 1),
    "rfdetr": version("rfdetr"),
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "python": platform.python_version(),
}
(out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
# Download only what is used: the selected checkpoint (best validation mAP), the per-epoch
# metrics and the settings. last.ckpt (with optimizer state) and the other weights stay here.
keep = WORK / "download" / RUN
keep.mkdir(parents=True, exist_ok=True)
for name in ("checkpoint_best_total.pth", "metrics.csv", "training_config.json", "run.json"):
    if (out / name).exists():
        shutil.copy2(out / name, keep / name)
archive = shutil.make_archive(str(WORK / RUN), "zip", WORK / "download", RUN)
print(json.dumps(run, indent=1))
print("done; download", archive, "(size", os.path.getsize(archive) // 2**20, "MB)")
