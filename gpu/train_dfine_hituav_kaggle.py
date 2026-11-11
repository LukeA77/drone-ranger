# Train the version 2 detector architecture (D-FINE N, 640) on HIT-UAV thermal images
# (Phase 13, D-751). Three runs, one after another, in one session. Runs on Kaggle or Colab.
#
# Kaggle: paste into ONE notebook cell (GPU on, Internet on), Save & Run All. Inputs are two
#   PRIVATE Kaggle datasets (zipped or unpacked by Kaggle, both work). Output:
#   /kaggle/working/<run>.zip per run.
# Colab: Runtime -> Change runtime type -> GPU. Inputs come from the Google Drive folder
#   My Drive/drone_ranger_gpu/ if they are there, otherwise from the Kaggle datasets in
#   KAGGLE_DATASETS (Colab secret KAGGLE_API_TOKEN). Training state is copied to
#   My Drive/drone_ranger_gpu/outputs/ every minute, so after a disconnect run the cell again:
#   finished runs are skipped and an unfinished one resumes from its last saved epoch.
#
# Inputs:
#   - hituav_coco.zip  (scripts/hituav_coco_package.py: every image, train and validation
#     annotations for the split by capture day and for the official split; no test labels)
#   - dfine_n_640_visdrone_best.pth  (the version 2 detector trained on VisDrone; the same file
#     as for the SynDrone fine-tune, so that Kaggle dataset can be attached again)
# The authors' COCO weights for the first and third runs are downloaded here.
# Put the output zips in the project's cache/train/ and unzip.
#
# Runs (in this order; a run that would not finish in the session's time budget is not started):
#   dfine_hituav_day_coco       split by capture day, from the authors' COCO weights: the
#                               thermal detector, trained as the colour one was (D-717, D-722)
#   dfine_hituav_day_visdrone   split by capture day, from the VisDrone-trained detector: does
#                               colour aerial training help thermal detection?
#   dfine_hituav_official_coco  official split, from COCO weights: for comparison with
#                               published results, and to measure what the official split's
#                               shared videos add to the score
# Recipe: as the detector's VisDrone training (D-717, D-722, gpu/train_dfine_kaggle.py): the
# authors' custom-dataset config at the same commit, 640 input, total batch 8 with learning
# rates scaled linearly from the authors' 32, the same number of optimiser steps as the colour
# detector's 50 VisDrone epochs (D-757: 40,050 steps, the last 4,806 without augmentation, so
# 164 epochs with augmentation off from 144 on the 1,955 day-split training images), seed 0,
# the best epoch on the split's validation images kept. The same 11-slot output head (VisDrone
# category numbers), so the thermal and colour detectors are scored on each other's data with
# the same groups. Resume as the SynDrone fine-tune (D-742).
# Local smoke test: SMOKE=4 python gpu/train_dfine_hituav_kaggle.py <hituav_coco dir> <D-FINE
#   checkout> <VisDrone checkpoint> <COCO checkpoint>

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

# (run name, split folder in the package, start weights: "coco" or "visdrone")
RUNS = [
    ("dfine_hituav_day_coco", "day", "coco"),
    ("dfine_hituav_day_visdrone", "day", "visdrone"),
    ("dfine_hituav_official_coco", "official", "coco"),
]
DFINE_COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
COCO_URL = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth"
# SHA-256 of that file, downloaded on 2026-11-10 (the VisDrone training used the same URL)
COCO_SHA = "41973938d2784d38a9836990d805b8392855ebf611aba55f0f7add90e110744c"
SIZE, RES, BATCH = "n", 640, 8
AUTHORS_BATCH = 32
SEED = 0
# D-757: the colour detector's training (gpu/train_dfine_kaggle.py, cache/train/dfine_n_640)
# ran 801 steps per epoch for 50 epochs, augmentation off for the last 6. HIT-UAV's training
# sets are about a quarter of VisDrone's, so 50 epochs gave a quarter of the steps and the
# first run was still improving at its end (validation AP50 +0.067 over epochs 34 to 43, against
# +0.018 for the colour run). Each run now gets the colour run's number of steps.
REF_STEPS, REF_NOAUG_STEPS = 50 * 801, 6 * 801
# D-FINE at this commit keeps the learning rate constant (MultiStepLR milestone 500, warm-up off
# here), so more epochs means more steps at the same rate, not a stretched schedule.


def schedule(n_train_images: int, batch: int) -> tuple[int, int]:
    """(epochs, epoch augmentation stops) giving the colour run's number of steps."""
    steps = max(n_train_images // batch, 1)
    epochs = round(REF_STEPS / steps)
    return epochs, epochs - round(REF_NOAUG_STEPS / steps)


NUM_CLASSES = 11  # category ids 1..10 (VisDrone numbers), slot 0 unused, as the colour detector
MAX_HOURS = 11.0  # whole session; Kaggle and free Colab stop a session at 12 h at most
VISDRONE_NAME = "dfine_n_640_visdrone_best.pth"
# SHA-256 of the VisDrone checkpoint (cache/gpu_package/dfine_n_640_visdrone_best.sha256)
VISDRONE_SHA = "c25086fe0597b7377e6dae8a41d02dc48f99a6818ebbdc35d544c44372c87514"
DRIVE_FOLDER = "drone_ranger_gpu"
KAGGLE_DATASETS: list[str] = []  # Colab only: "<user>/<dataset>" for the two inputs
MIRROR = ("best_stg1.pth", "best_stg2.pth", "last.pth", "log.txt")
SYNC_S, SETTLE_S = 60, 20

SMOKE = int(os.environ.get("SMOKE", "0"))
KAGGLE = Path("/kaggle/working").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
COLAB = (
    not KAGGLE
    and importlib.util.find_spec("google.colab") is not None
    and Path("/var/colab/hostname").exists()
)
T0 = time.time()

if COLAB:
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount("/content/drive")
    WORK = Path("/content")
    INPUT = Path("/content/drive/MyDrive") / DRIVE_FOLDER
    INPUT.mkdir(parents=True, exist_ok=True)
    roots = [INPUT, WORK / "kaggle_inputs"]
    KEEP = INPUT / "outputs"
else:
    WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    roots = [Path("/kaggle/input"), WORK]
    KEEP = WORK
KEEP.mkdir(parents=True, exist_ok=True)


def sh(*cmd: str, **kw: object) -> None:
    subprocess.run(list(cmd), check=True, **kw)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def walk(name: str) -> list[Path]:
    """Every file or folder called name under the roots (os.walk follows Kaggle's links)."""
    hits = []
    for r in roots:
        if r.exists():
            for d, dirs, files in os.walk(r, followlinks=True):
                dirs[:] = [n for n in dirs if n != "outputs"]
                hits += [Path(d) / n for n in dirs + files if n == name]
    return hits


def show_inputs() -> None:
    top = roots[0]
    if not top.exists():
        return
    print(f"Contents of {top}:")
    for d, dirs, files in os.walk(top, followlinks=True):
        depth = len(Path(d).relative_to(top).parts)
        if depth >= 3:
            dirs[:] = []
        print("  " * depth + Path(d).name + "/", sorted(files)[:5])


def kaggle_download() -> None:
    """Colab: fetch the inputs from Kaggle datasets when the Drive folder does not hold them."""
    have_data = any(zipfile.is_zipfile(p) for p in walk("hituav_coco.zip")) or any(
        p.is_dir() and (p / "images").is_dir() for p in walk("hituav_coco")
    )
    have_start = any(p.is_file() for p in walk(VISDRONE_NAME))
    if have_data and have_start:
        return
    from google.colab import userdata  # type: ignore[import-not-found]

    for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        try:
            v = userdata.get(k)
        except Exception:  # noqa: BLE001 - a secret that is not set or not shared
            v = None
        if v:
            os.environ[k] = v
    if not os.environ.get("KAGGLE_API_TOKEN") and not (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        raise SystemExit(
            "Inputs are not in the Drive folder, and no Kaggle key is set. In Colab, open the key "
            "icon (Secrets) on the left, add KAGGLE_API_TOKEN (kaggle.com -> Settings -> API -> "
            "Create New Token), switch on notebook access, and run again."
        )
    sh(sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle")
    if not KAGGLE_DATASETS:
        print("Your Kaggle datasets:", flush=True)
        subprocess.run(["kaggle", "datasets", "list", "--mine"], check=False)
        raise SystemExit(
            "Set KAGGLE_DATASETS near the top of the cell to the two datasets above that hold "
            "hituav_coco and the VisDrone weights (the 'ref' column), then run again."
        )
    dst = WORK / "kaggle_inputs"
    for ref in KAGGLE_DATASETS:
        d = dst / ref.replace("/", "__")
        if d.exists() and any(d.iterdir()):
            continue
        print("downloading", ref, "from Kaggle", flush=True)
        d.mkdir(parents=True, exist_ok=True)
        sh("kaggle", "datasets", "download", ref, "-p", str(d), "--unzip")


def find_data() -> Path:
    """The hituav_coco folder: images/ plus day/ and official/ annotation folders."""
    if SMOKE:
        return Path(sys.argv[1])
    local = WORK / "hituav_coco"
    if (local / "images").is_dir() and (local / "day").is_dir():
        return local
    for p in walk("hituav_coco"):
        if p.is_dir() and (p / "images").is_dir() and (p / "day").is_dir():
            return p
    for p in walk("day"):  # Kaggle may unpack the zip without its top folder
        if p.is_dir() and (p.parent / "images").is_dir() and (p.parent / "official").is_dir():
            return p.parent
    z = next((p for p in walk("hituav_coco.zip") if zipfile.is_zipfile(p)), None)
    if z is not None:
        print("unpacking", z, flush=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(WORK)
        return local
    show_inputs()
    raise SystemExit("dataset not found: attach hituav_coco (see the top of the file)")


def find_visdrone() -> Path:
    if SMOKE:
        return Path(sys.argv[3])
    p = next((p for p in walk(VISDRONE_NAME) if p.is_file()), None)
    if p is None:  # uploaded under another name: accept a single .pth input
        pths = [
            Path(d, f)
            for r in roots
            if r != WORK and r.exists()
            for d, _, fs in os.walk(r, followlinks=True)
            if "outputs" not in Path(d).parts
            for f in fs
            if f.endswith(".pth")
        ]
        if len(pths) == 1:
            p = pths[0]
    if p is None:
        show_inputs()
        raise SystemExit(f"VisDrone weights not found: add {VISDRONE_NAME} (see the top)")
    got = sha256(p)
    if got != VISDRONE_SHA:
        raise SystemExit(f"{p} is not the version 2 VisDrone checkpoint (sha256 {got})")
    return p


def find_coco() -> Path:
    p = Path(sys.argv[4]) if SMOKE else WORK / "dfine_n_coco.pth"
    if not p.exists():
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(COCO_URL, tmp)
        os.replace(tmp, p)
    got = sha256(p)
    if got != COCO_SHA:
        raise SystemExit(f"{p} is not the authors' D-FINE N COCO checkpoint (sha256 {got})")
    return p


def smoke_subset(data: Path, split: str, dst: Path, n: int) -> tuple[Path, Path]:
    outs = []
    for sub in ("train", "valid"):
        coco = json.loads((data / split / sub / "_annotations.coco.json").read_text())
        coco["images"] = coco["images"][:n]
        ids = {i["id"] for i in coco["images"]}
        coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in ids]
        p = dst / f"{sub}.json"
        dst.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(coco))
        outs.append(p)
    return outs[0], outs[1]


def patch_resume(repo: Path) -> None:
    """Start D-FINE's best score from DR_PRIOR_TOP1 when it is set (resume only, D-742)."""
    f = repo / "src/solver/det_solver.py"
    text = f.read_text()
    if "DR_PRIOR_TOP1" in text:
        return
    anchor = "        best_stat_print = best_stat.copy()\n"
    if text.count(anchor) != 1:
        raise SystemExit("D-FINE code is not as expected at this commit; resume patch not applied")
    patch = (
        '        _prior = float(__import__("os").environ.get("DR_PRIOR_TOP1", "-1"))\n'
        "        if _prior >= 0:\n"
        '            _seen = best_stat.get("coco_eval_bbox", -1.0)\n'
        '            best_stat["coco_eval_bbox"] = max(_seen, _prior)\n'
        "            top1 = max(top1, _prior)\n"
    )
    f.write_text(text.replace(anchor, patch + anchor))


def log_rows(d: Path) -> list[dict]:
    p = d / "log.txt"
    if not p.exists():
        return []
    rows = []
    for ln in p.read_text().splitlines():
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            pass  # a line cut short by a disconnect
    return rows


def mirror(src: Path, dst: Path, settle_s: float) -> None:
    """Copy the training state to dst, only when no file changed in the last settle_s seconds."""
    files = [src / n for n in MIRROR if (src / n).exists()]
    if not files or any(time.time() - f.stat().st_mtime < settle_s for f in files):
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        d = dst / f.name
        if d.exists() and d.stat().st_size == f.stat().st_size:
            if d.stat().st_mtime >= f.stat().st_mtime:
                continue
        tmp = dst / (f.name + ".part")
        shutil.copy2(f, tmp)
        os.replace(tmp, d)


def watch(run: str, src: Path, dst: Path | None, stop: threading.Event) -> None:
    shown = -1
    while not stop.wait(SYNC_S):
        rows = log_rows(src)
        if rows and rows[-1]["epoch"] != shown:
            r = rows[-1]
            shown = r["epoch"]
            ap = r.get("test_coco_eval_bbox", [float("nan")])[0]
            h = (time.time() - T0) / 3600
            print(f"{run}: epoch {shown + 1}/{epochs} done, AP {ap:.3f}, {h:.1f} h", flush=True)
        if dst is not None:
            try:
                mirror(src, dst, SETTLE_S)
            except OSError as e:
                print("copy to Drive failed, will retry:", e, flush=True)


def prepare_resume(out: Path, saved: Path | None) -> tuple[Path | None, float]:
    """Colab: bring back a run's saved state: (checkpoint to resume, best score so far)."""
    if saved is None or not (saved / "last.pth").exists():
        for n in MIRROR:
            (out / n).unlink(missing_ok=True)
        return None, -1.0
    for n in MIRROR:
        if (saved / n).exists():
            shutil.copy2(saved / n, out / n)
    try:
        last = torch.load(out / "last.pth", map_location="cpu", weights_only=False)["last_epoch"]
    except Exception as e:  # noqa: BLE001 - any unreadable file means start again
        print("saved checkpoint unreadable, the run starts again:", e, flush=True)
        for n in MIRROR:
            (out / n).unlink(missing_ok=True)
        shutil.rmtree(saved)
        return None, -1.0
    (out / "best_stg2.pth").unlink(missing_ok=True)
    (saved / "best_stg2.pth").unlink(missing_ok=True)
    rows = [r for r in log_rows(out) if r["epoch"] <= last]
    (out / "log.txt").write_text("".join(json.dumps(r) + "\n" for r in rows))
    stage1 = [r["test_coco_eval_bbox"][0] for r in rows if r["epoch"] < aug_off]
    print(f"resuming after epoch {last + 1}", flush=True)
    return out / "last.pth", max(stage1, default=-1.0)


if SMOKE:
    repo = Path(sys.argv[2]).resolve()
else:
    repo = WORK / "D-FINE"
    if not repo.exists():
        sh("git", "clone", "-q", "https://github.com/Peterande/D-FINE.git", str(repo))
    sh("git", "-C", str(repo), "checkout", "-q", DFINE_COMMIT)
    patch_resume(repo)
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
    where = "Runtime -> Change runtime type" if COLAB else "Session options -> Accelerator"
    raise SystemExit(f"NO GPU FOUND. {where} -> GPU, then run again.")

if COLAB:
    kaggle_download()
data = find_data()
starts = {"visdrone": find_visdrone(), "coco": find_coco()}
start_sha = {k: sha256(p) for k, p in starts.items()}
print("start weights sha256:", start_sha, flush=True)
base_cfg = repo / f"configs/dfine/custom/dfine_hgnetv2_{SIZE}_custom.yml"
base = base_cfg.read_text()
batch = 2 if SMOKE else BATCH
lr_scale = BATCH / AUTHORS_BATCH
workers = 0 if SMOKE else 4
done, skipped, run_hours = [], [], []

for run, split, start_kind in RUNS:
    if (KEEP / f"{run}.zip").exists() and not SMOKE:
        print(run, "already finished:", KEEP / f"{run}.zip", flush=True)
        done.append(run)
        continue
    out = WORK / run
    out.mkdir(parents=True, exist_ok=True)
    used_h = (time.time() - T0) / 3600
    est_h = max(run_hours) if run_hours else 0.0
    if not SMOKE and run_hours and used_h + est_h > MAX_HOURS:
        skipped.append(run)
        continue
    if SMOKE:
        train_ann, valid_ann = smoke_subset(data, split, out / "smoke", SMOKE)
    else:
        train_ann = data / split / "train" / "_annotations.coco.json"
        valid_ann = data / split / "valid" / "_annotations.coco.json"
    if SMOKE:
        epochs, aug_off = 2, 1
    else:
        n_img = len(json.loads(train_ann.read_text())["images"])
        epochs, aug_off = schedule(n_img, BATCH)
    print(f"{run}: {epochs} epochs, augmentation off from {aug_off}", flush=True)
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
    saved = KEEP / run if COLAB else None
    resume, prior = prepare_resume(out, saved)
    start = starts[start_kind]
    cmd = [sys.executable, "train.py", "-c", str(cfg_path), f"--seed={SEED}"]
    cmd += ["-r", str(resume)] if resume else ["-t", str(start)]
    cmd += ["--device", "cpu"] if SMOKE else ["--use-amp"]
    t_run = time.time()
    limit_s = int(0.1 * 3600) if SMOKE else int(max(0.5, MAX_HOURS - used_h) * 3600)
    stopped_by_time = False
    stop = threading.Event()
    watcher = threading.Thread(target=watch, args=(run, out, saved, stop), daemon=True)
    watcher.start()
    print(run, "training started", flush=True)
    with open(out / "train_stdout.txt", "a") as log:
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        if prior >= 0:
            env["DR_PRIOR_TOP1"] = repr(prior)
        proc = subprocess.Popen(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            rc = proc.wait(timeout=limit_s)
        except subprocess.TimeoutExpired:
            proc.terminate()
            rc = proc.wait()
            stopped_by_time = True
    stop.set()
    watcher.join()
    if saved is not None:
        mirror(out, saved, 0.0)
    if rc != 0 and not stopped_by_time:
        print((out / "train_stdout.txt").read_text()[-4000:])
        raise SystemExit(f"{run}: training failed (exit {rc})")
    run_hours.append((time.time() - t_run) / 3600)
    best = next((out / n for n in ("best_stg2.pth", "best_stg1.pth") if (out / n).exists()), None)
    info = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": run,
        "split": split,
        "model": f"D-FINE {SIZE.upper()}",
        "commit": DFINE_COMMIT,
        "start_weights": start_kind,
        "start_weights_file": start.name,
        "start_weights_sha256": start_sha[start_kind],
        "train_annotations_sha256": sha256(train_ann),
        "valid_annotations_sha256": sha256(valid_ann),
        "input_size": RES,
        "epochs": epochs,
        "augmentation_off_from_epoch": aug_off,
        "total_batch_size": batch,
        "lr_scale": lr_scale,
        "num_classes": NUM_CLASSES,
        "seed": SEED,
        "platform": "colab" if COLAB else "kaggle" if KAGGLE else "local",
        "resumed": resume is not None,
        "stopped_by_time_limit": stopped_by_time,
        "best_checkpoint": best.name if best else None,
        "smoke": SMOKE,
        "run_hours_this_session": round(run_hours[-1], 2),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
    }
    (out / "run.json").write_text(json.dumps(info, indent=1) + "\n")
    if stopped_by_time and COLAB:  # unfinished: resumes next session, so no zip yet
        print(json.dumps(info, indent=1))
        skipped += [r[0] for r in RUNS if r[0] not in done and r[0] not in skipped]
        break
    keep = WORK / "download" / run
    keep.mkdir(parents=True, exist_ok=True)
    if best:
        shutil.copy2(best, keep / "checkpoint_best.pth")
    for name in ("log.txt", "train_stdout.txt", f"{run}.yml", "run.json"):
        if (out / name).exists():
            shutil.copy2(out / name, keep / name)
    archive = Path(shutil.make_archive(str(WORK / run), "zip", WORK / "download", run))
    if KEEP != WORK:
        tmp = KEEP / (archive.name + ".part")
        shutil.copy2(archive, tmp)
        os.replace(tmp, KEEP / archive.name)
        archive = KEEP / archive.name
    done.append(run)
    print(json.dumps(info, indent=1))
    print("run done; download", archive, "(size", archive.stat().st_size // 2**20, "MB)")
    if stopped_by_time:  # Kaggle: kept with its flag set, the rest wait
        skipped += [r[0] for r in RUNS if r[0] not in done and r[0] not in skipped]
        break

print("RUNS DONE:", done)
if skipped:
    print("NOT RUN THIS SESSION (time budget):", skipped)
    if COLAB:
        print("Next session: run the cell again; finished runs are skipped.")
    else:
        print("Next session: remove the finished runs from RUNS and Save & Run All again.")
