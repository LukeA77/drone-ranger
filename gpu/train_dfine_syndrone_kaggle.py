# Fine-tune the version 2 detector (D-FINE N, 640) on SynDrone development towns, one
# leave-one-town-out fold after another (D-707, D-740). Paste into ONE Kaggle notebook cell
# (GPU on, Internet on) and use Save & Run All.
#
# Inputs (PRIVATE Kaggle datasets, uploaded by hand; zipped or unpacked by Kaggle, both work):
#   - syndrone_dev_coco.zip  (scripts/syndrone_coco_package.py; the same images and folds as
#     version 1's SynDrone fine-tune)
#   - dfine_n_640_visdrone_best.pth  (cache/train/dfine_n_640/checkpoint_best.pth, renamed:
#     the version 2 detector trained on VisDrone, so the only change is SynDrone training)
# Output: /kaggle/working/dfine_syndrone_minus_<Town>.zip per fold (best checkpoint, logs,
#   config, run.json). Put them in the project's cache/train/ and unzip.
#
# Recipe: as the detector's VisDrone training (D-717, gpu/train_dfine_kaggle.py): the authors'
# custom-dataset config at the same commit, 640 input, total batch 8 with learning rates scaled
# linearly from the authors' 32, at most 50 epochs with augmentation off from 44, seed 0, the
# best epoch on the fold's selection frames kept. Version 1's fine-tune also had at most
# 50 epochs (D-707). Folds run in order until the session's time budget is used; a fold that
# would not finish is not started, and FOLDS is then set to the remaining ones next session.
# Local smoke test: SMOKE=4 python gpu/train_dfine_syndrone_kaggle.py <data dir> <D-FINE
#   checkout> <start checkpoint>

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

FOLDS = ["Town01", "Town07", "Town10HD"]  # held-out town of each fold; edit for a 2nd session
DFINE_COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
SIZE, RES, BATCH = "n", 640, 8
AUTHORS_BATCH = 32
EPOCHS, AUG_OFF, SEED = 50, 44, 0
NUM_CLASSES = 11  # category ids 1..10 (VisDrone numbers), slot 0 unused, as the start weights
MAX_HOURS = 11.0  # whole session; Kaggle stops a session at 12 h
START_NAME = "dfine_n_640_visdrone_best.pth"

SMOKE = int(os.environ.get("SMOKE", "0"))
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
roots = [Path("/kaggle/input"), WORK]
T0 = time.time()


def sh(*cmd: str, **kw: object) -> None:
    subprocess.run(list(cmd), check=True, **kw)


def find_data() -> Path:
    if SMOKE:
        return Path(sys.argv[1])
    for r in roots:
        if r.exists():
            for p in r.rglob("fold_Town01"):
                if p.is_dir() and (p.parent / "images").is_dir():
                    return p.parent
    z = next((p for r in roots if r.exists() for p in r.rglob("syndrone_dev_coco.zip")), None)
    if z is None:
        raise SystemExit("dataset not found: attach the syndrone_dev_coco dataset")
    with zipfile.ZipFile(z) as f:
        f.extractall(WORK)
    return WORK / "syndrone_dev_coco"


def find_start() -> Path:
    if SMOKE:
        return Path(sys.argv[3])
    p = next((p for r in roots if r.exists() for p in r.rglob(START_NAME)), None)
    if p is None:
        raise SystemExit(f"start weights not found: attach {START_NAME}")
    return p


def smoke_subset(data: Path, held: str, dst: Path, n: int) -> tuple[Path, Path]:
    """The first n images of each split, as small annotation files (local test only)."""
    outs = []
    for split in ("train", "valid"):
        coco = json.loads((data / f"fold_{held}" / split / "_annotations.coco.json").read_text())
        coco["images"] = coco["images"][:n]
        ids = {i["id"] for i in coco["images"]}
        coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in ids]
        p = dst / f"{split}.json"
        dst.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(coco))
        outs.append(p)
    return outs[0], outs[1]


if SMOKE:
    repo = Path(sys.argv[2])
else:
    repo = WORK / "D-FINE"
    if not repo.exists():
        sh("git", "clone", "-q", "https://github.com/Peterande/D-FINE.git", str(repo))
    sh("git", "-C", str(repo), "checkout", "-q", DFINE_COMMIT)
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

data = find_data()
start = find_start()
base_cfg = repo / f"configs/dfine/custom/dfine_hgnetv2_{SIZE}_custom.yml"
base = base_cfg.read_text()
batch = 2 if SMOKE else BATCH
lr_scale = BATCH / AUTHORS_BATCH  # the learning rates of the VisDrone training (batch 8)
epochs, aug_off = (2, 1) if SMOKE else (EPOCHS, AUG_OFF)
workers = 0 if SMOKE else 4
done, skipped, fold_hours = [], [], []

for held in FOLDS:
    run = f"dfine_syndrone_minus_{held}"
    out = WORK / run
    out.mkdir(parents=True, exist_ok=True)
    used_h = (time.time() - T0) / 3600
    est_h = max(fold_hours) if fold_hours else 0.0
    if not SMOKE and fold_hours and used_h + est_h > MAX_HOURS:
        skipped.append(held)
        continue
    if SMOKE:
        train_ann, valid_ann = smoke_subset(data, held, out / "smoke", SMOKE)
    else:
        train_ann = data / f"fold_{held}" / "train" / "_annotations.coco.json"
        valid_ann = data / f"fold_{held}" / "valid" / "_annotations.coco.json"
    cfg = f"""__include__: ['{base_cfg.as_posix()}']
output_dir: {out.as_posix()}
num_classes: {NUM_CLASSES}
remap_mscoco_category: False
eval_spatial_size: [{RES}, {RES}]
epochs: {epochs}
ema:
  warmups: 0
lr_warmup_scheduler:
  warmup_duration: 0
train_dataloader:
  total_batch_size: {batch}
  num_workers: {workers}
  dataset:
    img_folder: {data.as_posix()}
    ann_file: {train_ann.as_posix()}
    transforms:
      ops:
        - {{type: RandomPhotometricDistort, p: 0.5}}
        - {{type: RandomZoomOut, fill: 0}}
        - {{type: RandomIoUCrop, p: 0.8}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: RandomHorizontalFlip}}
        - {{type: Resize, size: [{RES}, {RES}], }}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
      policy:
        name: stop_epoch
        epoch: {aug_off}
        ops: ['RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']
  collate_fn:
    base_size: {RES}
    stop_epoch: {aug_off}
    ema_restart_decay: 0.9999
    base_size_repeat: 10
val_dataloader:
  total_batch_size: {max(batch, 2)}
  num_workers: {workers}
  dataset:
    img_folder: {data.as_posix()}
    ann_file: {valid_ann.as_posix()}
    transforms:
      ops:
        - {{type: Resize, size: [{RES}, {RES}], }}
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
    cfg_path = out / f"{run}.yml"
    cfg_path.write_text(cfg)
    cmd = [sys.executable, "train.py", "-c", str(cfg_path), "-t", str(start), f"--seed={SEED}"]
    cmd += ["--device", "cpu"] if SMOKE else ["--use-amp"]
    t_fold = time.time()
    limit_s = int(0.1 * 3600) if SMOKE else int(max(0.5, MAX_HOURS - used_h) * 3600)
    stopped_by_time = False
    with open(out / "train_stdout.txt", "w") as log:
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        proc = subprocess.Popen(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            rc = proc.wait(timeout=limit_s)
        except subprocess.TimeoutExpired:
            proc.terminate()
            rc = proc.wait()
            stopped_by_time = True
    if rc != 0 and not stopped_by_time:
        print((out / "train_stdout.txt").read_text()[-4000:])
        raise SystemExit(f"{run}: training failed (exit {rc})")
    fold_hours.append((time.time() - t_fold) / 3600)
    best = next((out / n for n in ("best_stg2.pth", "best_stg1.pth") if (out / n).exists()), None)
    info = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": run,
        "held_out_town": held,
        "model": f"D-FINE {SIZE.upper()}",
        "commit": DFINE_COMMIT,
        "start_weights": START_NAME,
        "input_size": RES,
        "epochs": epochs,
        "augmentation_off_from_epoch": aug_off,
        "total_batch_size": batch,
        "lr_scale": lr_scale,
        "num_classes": NUM_CLASSES,
        "seed": SEED,
        "stopped_by_time_limit": stopped_by_time,
        "best_checkpoint": best.name if best else None,
        "smoke": SMOKE,
        "fold_hours": round(fold_hours[-1], 2),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
    }
    (out / "run.json").write_text(json.dumps(info, indent=1) + "\n")
    keep = WORK / "download" / run
    keep.mkdir(parents=True, exist_ok=True)
    if best:
        shutil.copy2(best, keep / "checkpoint_best.pth")
    for name in ("log.txt", "train_stdout.txt", f"{run}.yml", "run.json"):
        if (out / name).exists():
            shutil.copy2(out / name, keep / name)
    archive = shutil.make_archive(str(WORK / run), "zip", WORK / "download", run)
    done.append(held)
    print(json.dumps(info, indent=1))
    print("fold done; download", archive, "(size", os.path.getsize(archive) // 2**20, "MB)")
    if stopped_by_time:
        skipped += [f for f in FOLDS if f not in done and f not in skipped]
        break

print("FOLDS DONE:", done)
if skipped:
    print("NOT RUN THIS SESSION (time budget):", skipped)
    print("Next session: set FOLDS =", skipped, "and Save & Run All again.")
