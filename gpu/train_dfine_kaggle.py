# Train a D-FINE detector candidate on VisDrone DET. Paste into ONE Kaggle notebook cell
# (GPU on, Internet on), set RUN below, and use Save & Run All. One run per session.
#
# Input (a PRIVATE Kaggle dataset, uploaded by hand): visdrone_det_coco.zip
#   (scripts/visdrone_coco_package.py; zipped or unpacked by Kaggle, both work).
# Output: /kaggle/working/<run>.zip (best checkpoint, training log, config, run.json).
#   Put it in the project's cache/train/ and unzip (docs/gpu_training.md, candidates).
#
# Code: the authors' repository at a fixed commit (Apache-2.0). Start weights: the authors'
# COCO-only checkpoint (D-715 prefers COCO-only weights). Recipe: the authors' custom-dataset
# config, fine-tuned from those weights, with the same epoch budget as the other candidates
# (50, D-717): the authors' fine-tuning schedule (64 epochs, augmentation off from 56) is
# scaled to 50 epochs, augmentation off from 44. The best epoch on VisDrone DET val is kept.
# Training runs in a child process stopped after MAX_HOURS, so a session never hits Kaggle's
# hard time limit; checkpoints are written every epoch.
# Local smoke test: SMOKE=4 python gpu/train_dfine_kaggle.py <data dir> <D-FINE checkout>

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

RUN = "dfine_n_640"  # one of RUNS below
# Batch sizes fit one 16 GB GPU. The authors' 32 is split over 4 GPUs (8 each); at 32 on one
# T4 the first steps ran out of memory: VisDrone images hold about 50 boxes, some hundreds, and
# the decoder's training-only denoising queries grow with the boxes per image.
RUNS = {  # name: (size letter, input size, total batch size)
    "dfine_n_640": ("n", 640, 8),
    "dfine_n_960": ("n", 960, 4),
    "dfine_s_640": ("s", 640, 8),
}
DFINE_COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
WEIGHTS_URL = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{}_coco.pth"
AUTHORS_BATCH = 32  # the authors' total batch size for their learning rates
EPOCHS, AUG_OFF, SEED = 50, 44, 0
NUM_CLASSES = 11  # category ids 1..10 (VisDrone numbers), not remapped; slot 0 unused
MAX_HOURS = 11.0

SMOKE = int(os.environ.get("SMOKE", "0"))
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
roots = [Path("/kaggle/input"), WORK]
size, res, batch = RUNS[RUN]


def sh(*cmd: str, **kw: object) -> None:
    subprocess.run(list(cmd), check=True, **kw)


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


if SMOKE:
    repo = Path(sys.argv[2])
else:
    repo = WORK / "D-FINE"
    if not repo.exists():
        sh("git", "clone", "-q", "https://github.com/Peterande/D-FINE.git", str(repo))
    sh("git", "-C", str(repo), "checkout", "-q", DFINE_COMMIT)
    # matplotlib and pycocotools are imported (src/solver/validator.py; torchvision CocoDetection)
    # but not listed in requirements.txt
    sh(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        str(repo / "requirements.txt"),
        "matplotlib",
        "pycocotools",
    )

import torch  # noqa: E402

if not SMOKE and not torch.cuda.is_available():
    raise SystemExit("NO GPU FOUND. Session options -> Accelerator -> GPU, then run again.")

out = WORK / RUN
out.mkdir(parents=True, exist_ok=True)
data = find_data()
if SMOKE:
    data = subset(data, out / "smoke_data", SMOKE)
    batch = 2
epochs, aug_off = (2, 1) if SMOKE else (EPOCHS, AUG_OFF)
start = out / f"dfine_{size}_coco.pth"
if not start.exists():
    urllib.request.urlretrieve(WEIGHTS_URL.format(size), start)

# Learning rates scale with the batch size (linear rule) when it differs from the authors'.
base_cfg = repo / f"configs/dfine/custom/dfine_hgnetv2_{size}_custom.yml"
base = base_cfg.read_text()
lr_scale = batch / AUTHORS_BATCH
workers = 0 if SMOKE else 4
cfg = f"""__include__: ['{base_cfg.as_posix()}']
output_dir: {out.as_posix()}
num_classes: {NUM_CLASSES}
remap_mscoco_category: False
eval_spatial_size: [{res}, {res}]
epochs: {epochs}
ema:
  warmups: 0
lr_warmup_scheduler:
  warmup_duration: 0
train_dataloader:
  total_batch_size: {batch}
  num_workers: {workers}
  dataset:
    img_folder: {(data / "train").as_posix()}
    ann_file: {(data / "train" / "_annotations.coco.json").as_posix()}
    transforms:
      ops:
        - {{type: RandomPhotometricDistort, p: 0.5}}
        - {{type: RandomZoomOut, fill: 0}}
        - {{type: RandomIoUCrop, p: 0.8}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: RandomHorizontalFlip}}
        - {{type: Resize, size: [{res}, {res}], }}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
      policy:
        name: stop_epoch
        epoch: {aug_off}
        ops: ['RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']
  collate_fn:
    base_size: {res}
    stop_epoch: {aug_off}
    ema_restart_decay: 0.9999
    base_size_repeat: 10
val_dataloader:
  total_batch_size: {max(batch, 2)}
  num_workers: {workers}
  dataset:
    img_folder: {(data / "valid").as_posix()}
    ann_file: {(data / "valid" / "_annotations.coco.json").as_posix()}
    transforms:
      ops:
        - {{type: Resize, size: [{res}, {res}], }}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
"""
if lr_scale != 1.0:
    import yaml

    opt = yaml.safe_load(base)["optimizer"]
    opt["lr"] = opt["lr"] * lr_scale
    for g in opt["params"]:
        if "lr" in g:
            g["lr"] = g["lr"] * lr_scale
    cfg += yaml.safe_dump({"optimizer": opt}, default_flow_style=None)
cfg_path = out / f"{RUN}.yml"
cfg_path.write_text(cfg)

cmd = [sys.executable, "train.py", "-c", str(cfg_path), "-t", str(start), f"--seed={SEED}"]
cmd += ["--device", "cpu"] if SMOKE else ["--use-amp"]
t0 = time.time()
limit_s = int((0.05 if SMOKE else MAX_HOURS) * 3600)
stopped_by_time = False
with open(out / "train_stdout.txt", "w") as log:
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    proc = subprocess.Popen(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        rc = proc.wait(timeout=None if SMOKE else limit_s)
    except subprocess.TimeoutExpired:
        proc.terminate()
        rc = proc.wait()
        stopped_by_time = True
if rc != 0 and not stopped_by_time:
    print((out / "train_stdout.txt").read_text()[-4000:])
    raise SystemExit(f"training failed (exit {rc})")

# The overall best epoch is best_stg2.pth if it came after augmentation was switched off,
# otherwise best_stg1.pth (D-FINE's det_solver.py saves the global best under one of these).
best = next((out / n for n in ("best_stg2.pth", "best_stg1.pth") if (out / n).exists()), None)
run = {
    "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "run": RUN,
    "model": f"D-FINE {size.upper()}",
    "commit": DFINE_COMMIT,
    "start_weights": start.name,
    "input_size": res,
    "epochs": epochs,
    "augmentation_off_from_epoch": aug_off,
    "total_batch_size": batch,
    "lr_scale": lr_scale,
    "num_classes": NUM_CLASSES,
    "seed": SEED,
    "max_hours": MAX_HOURS,
    "stopped_by_time_limit": stopped_by_time,
    "best_checkpoint": best.name if best else None,
    "smoke": SMOKE,
    "elapsed_s": round(time.time() - t0, 1),
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "python": platform.python_version(),
}
(out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
keep = WORK / "download" / RUN
keep.mkdir(parents=True, exist_ok=True)
if best:
    shutil.copy2(best, keep / "checkpoint_best.pth")
for name in ("log.txt", "train_stdout.txt", f"{RUN}.yml", "run.json"):
    if (out / name).exists():
        shutil.copy2(out / name, keep / name)
archive = shutil.make_archive(str(WORK / RUN), "zip", WORK / "download", RUN)
print(json.dumps(run, indent=1))
print("done; download", archive, "(size", os.path.getsize(archive) // 2**20, "MB)")
