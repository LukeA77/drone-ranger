# Fine-tune the version 2 detector (D-FINE N, 640) on SynDrone development towns, one
# leave-one-town-out fold after another (D-707, D-740). Runs on Kaggle or Google Colab.
#
# Kaggle: paste into ONE notebook cell (GPU on, Internet on), Save & Run All. Inputs are two
#   PRIVATE Kaggle datasets (zipped or unpacked by Kaggle, both work). Output:
#   /kaggle/working/dfine_syndrone_minus_<Town>.zip per fold.
# Colab: Runtime -> Change runtime type -> GPU. Inputs come from the Google Drive folder
#   My Drive/drone_ranger_gpu/ if they are there, otherwise from the two Kaggle datasets in
#   KAGGLE_DATASETS (Colab secret KAGGLE_API_TOKEN). Paste this into ONE cell and run it; allow
#   the Drive access it asks for (Drive holds the outputs). Training state is copied to
#   My Drive/drone_ranger_gpu/outputs/ every minute, so
#   after a disconnect just run the cell again: finished folds are skipped and an unfinished one
#   resumes from its last saved epoch. Output: the same zips, in that outputs folder.
#
# Inputs:
#   - syndrone_dev_coco.zip  (scripts/syndrone_coco_package.py; the same images and folds as
#     version 1's SynDrone fine-tune)
#   - dfine_n_640_visdrone_best.pth  (cache/train/dfine_n_640/checkpoint_best.pth, renamed:
#     the version 2 detector trained on VisDrone, so the only change is SynDrone training)
# Put the output zips in the project's cache/train/ and unzip.
#
# Recipe: as the detector's VisDrone training (D-717, gpu/train_dfine_kaggle.py): the authors'
# custom-dataset config at the same commit, 640 input, total batch 8 with learning rates scaled
# linearly from the authors' 32, at most 50 epochs with augmentation off from 44, seed 0, the
# best epoch on the fold's selection frames kept. Version 1's fine-tune also had at most
# 50 epochs (D-707). Folds run in order until the session's time budget is used; a fold that
# would not finish is not started. On Kaggle, set FOLDS to the remaining ones next session.
#
# Resume (D-742): D-FINE's own resume (-r) restarts the "best so far" score from the resumed
# epoch, so a later, worse epoch could replace the best checkpoint. One line is added to the
# cloned D-FINE (patch_resume) that starts it from the best score in the fold's log instead;
# without a resume it changes nothing.
# Local smoke test: SMOKE=4 python gpu/train_dfine_syndrone_kaggle.py <data dir> <D-FINE
#   checkout> <start checkpoint>

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
import zipfile
from datetime import UTC, datetime
from pathlib import Path

FOLDS = ["Town01", "Town07", "Town10HD"]  # held-out town of each fold; edit for a 2nd session
DFINE_COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
SIZE, RES, BATCH = "n", 640, 8
AUTHORS_BATCH = 32
EPOCHS, AUG_OFF, SEED = 50, 44, 0
NUM_CLASSES = 11  # category ids 1..10 (VisDrone numbers), slot 0 unused, as the start weights
MAX_HOURS = 11.0  # whole session; Kaggle and free Colab stop a session at 12 h at most
START_NAME = "dfine_n_640_visdrone_best.pth"
DRIVE_FOLDER = "drone_ranger_gpu"  # Colab: My Drive folder for outputs (and inputs, if there)
# Colab: the two Kaggle datasets to download the inputs from, as "<user>/<dataset>" (the end of
# each dataset's address, kaggle.com/datasets/<user>/<dataset>). Used when the inputs are not in
# the Drive folder; needs the Colab secret KAGGLE_API_TOKEN (or KAGGLE_USERNAME and KAGGLE_KEY).
KAGGLE_DATASETS: list[str] = []
MIRROR = ("best_stg1.pth", "best_stg2.pth", "last.pth", "log.txt")  # copied in this order
SYNC_S, SETTLE_S = 60, 20  # Colab: copy to Drive every minute, files unchanged for 20 s

SMOKE = int(os.environ.get("SMOKE", "0"))
COLAB = importlib.util.find_spec("google.colab") is not None
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


def walk(name: str) -> list[Path]:
    """Every file or folder called name under the roots. os.walk with followlinks, because
    Kaggle mounts datasets as symbolic links and Path.rglob stopped following them in
    Python 3.13."""
    hits = []
    for r in roots:
        if r.exists():
            for d, dirs, files in os.walk(r, followlinks=True):
                dirs[:] = [n for n in dirs if n != "outputs"]
                hits += [Path(d) / n for n in dirs + files if n == name]
    return hits


def show_inputs() -> None:
    """Print what is attached (three levels), so a missing or renamed file is visible."""
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
    """Colab: fetch the inputs from Kaggle datasets when the Drive folder does not hold them
    (Kaggle to Colab is a transfer between data centres, much faster than a home upload)."""
    have_data = any(zipfile.is_zipfile(p) for p in walk("syndrone_dev_coco.zip")) or any(
        p.is_dir() for p in walk("fold_Town01")
    )
    have_start = any(p.is_file() for p in walk(START_NAME))
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
    # current client: older ones on Colab do not read the newer KAGGLE_API_TOKEN keys
    sh(sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle")
    if not KAGGLE_DATASETS:
        print("Your Kaggle datasets:", flush=True)
        subprocess.run(["kaggle", "datasets", "list", "--mine"], check=False)
        raise SystemExit(
            "Set KAGGLE_DATASETS near the top of the cell to the two datasets above that hold "
            "syndrone_dev_coco and the start weights (the 'ref' column), then run again."
        )
    dst = WORK / "kaggle_inputs"
    for ref in KAGGLE_DATASETS:
        d = dst / ref.replace("/", "__")
        if d.exists() and any(d.iterdir()):
            continue
        print("downloading", ref, "from Kaggle", flush=True)
        d.mkdir(parents=True, exist_ok=True)
        sh("kaggle", "datasets", "download", ref, "-p", str(d), "--unzip")


YOLO_NAMES = 10  # VisDrone classes; COCO category id = YOLO id + 1


def yolo_to_coco(read: object, names: list[str], prefix: str, dst: Path) -> None:
    """Write the folds' COCO annotation files from version 1's YOLO package (the same images and
    folds, D-707), exactly as scripts/syndrone_coco_package.py does; `read(rel)` returns a
    file's bytes, `names` lists the package's files (relative to its root, `prefix` stripped)."""
    import io

    from PIL import Image

    jpgs = [n for n in names if n.endswith(".jpg")]
    sizes = {}
    for n in jpgs:
        with Image.open(io.BytesIO(read(n))) as im:
            sizes[n] = im.size
    vis = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle"]
    vis += ["awning-tricycle", "bus", "motor"]
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(vis)]
    folds = sorted(
        n.split("fold_")[1].removesuffix("_train.txt")
        for n in names
        if n.startswith("fold_") and n.endswith("_train.txt")
    )
    for town in folds:
        for split, lst in (("train", "train"), ("valid", "select")):
            images, anns = [], []
            rels = [
                ln.strip().lstrip("./")
                for ln in read(f"fold_{town}_{lst}.txt").decode().splitlines()
                if ln.strip()
            ]
            for img_id, rel in enumerate(rels, start=1):
                w, h = sizes[rel]
                images.append({"id": img_id, "file_name": rel, "width": w, "height": h})
                lab = rel.replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
                for ln in read(lab).decode().splitlines():
                    if not ln.strip():
                        continue
                    c, cx, cy, bw, bh = ln.split()
                    bw_px, bh_px = float(bw) * w, float(bh) * h
                    x, y = float(cx) * w - bw_px / 2, float(cy) * h - bh_px / 2
                    anns.append(
                        {
                            "id": len(anns) + 1,
                            "image_id": img_id,
                            "category_id": int(c) + 1,
                            "bbox": [round(x, 2), round(y, 2), round(bw_px, 2), round(bh_px, 2)],
                            "area": round(bw_px * bh_px, 2),
                            "iscrowd": 0,
                        }
                    )
            out = dst / f"fold_{town}" / split / "_annotations.coco.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    print(f"COCO annotations built from the YOLO package: {len(folds)} folds", flush=True)


def find_data() -> tuple[Path, Path]:
    """(image root, annotation root): file names in the annotations are relative to the image
    root; fold_<Town>/<split>/_annotations.coco.json sit under the annotation root."""
    if SMOKE:
        return Path(sys.argv[1]), Path(sys.argv[1])
    local = WORK / "syndrone_dev_coco"
    if (local / "fold_Town01").is_dir() and (local / "images").is_dir():
        return local, local  # unpacked earlier in this session
    for p in walk("fold_Town01"):
        if p.is_dir() and (p.parent / "images").is_dir():
            return p.parent, p.parent
    z = next((p for p in walk("syndrone_dev_coco.zip") if zipfile.is_zipfile(p)), None)
    if z is not None:
        print("unpacking", z, flush=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(WORK)
        return local, local
    # version 1's YOLO package (same images): build the COCO annotations here, no new upload
    ann = WORK / "syndrone_dev_coco_annotations"
    for p in walk("fold_Town01_train.txt"):
        root = p.parent
        if (root / "images").is_dir() and (root / "labels").is_dir():
            if not (ann / "fold_Town01").is_dir():
                names = [
                    str(Path(d, f).relative_to(root).as_posix())
                    for d, _, fs in os.walk(root, followlinks=True)
                    for f in fs
                ]
                yolo_to_coco(lambda r, root=root: (root / r).read_bytes(), names, "", ann)
            return root, ann
    z = next((p for p in walk("syndrone_dev_yolo.zip") if zipfile.is_zipfile(p)), None)
    if z is not None:
        print("unpacking", z, flush=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(WORK)
        root = WORK / "syndrone_dev_yolo"
        names = [
            str(Path(d, f).relative_to(root).as_posix()) for d, _, fs in os.walk(root) for f in fs
        ]
        yolo_to_coco(lambda r: (root / r).read_bytes(), names, "", ann)
        return root, ann
    show_inputs()
    raise SystemExit(
        "dataset not found: attach syndrone_dev_coco or syndrone_dev_yolo (see the top of the file)"
    )


def find_start() -> Path:
    if SMOKE:
        return Path(sys.argv[3])
    p = next((p for p in walk(START_NAME) if p.is_file()), None)
    if p is None:  # the weights uploaded under another name: accept a single .pth input
        inputs = [r for r in roots if r != WORK]
        pths = [
            Path(d, f)
            for r in inputs
            if r.exists()
            for d, _, fs in os.walk(r, followlinks=True)
            if "outputs" not in Path(d).parts
            for f in fs
            if f.endswith(".pth")
        ]
        if len(pths) == 1:
            p = pths[0]
            print("start weights:", p, flush=True)
    if p is None:
        show_inputs()
        raise SystemExit(f"start weights not found: add {START_NAME} (see the top of the file)")
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
    """Copy the training state to dst, only when no file changed in the last settle_s
    seconds (so the copies come from the same epoch), each via a temporary name."""
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
    """Print each finished epoch; on Colab also copy the training state to Drive."""
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
    """Colab: bring back a fold's saved state. Returns (checkpoint to resume, best score so
    far), or (None, -1) to start the fold from the beginning."""
    if saved is None or not (saved / "last.pth").exists():
        for n in MIRROR:  # leftovers of an earlier attempt in this session
            (out / n).unlink(missing_ok=True)
        return None, -1.0
    for n in MIRROR:
        if (saved / n).exists():
            shutil.copy2(saved / n, out / n)
    try:
        last = torch.load(out / "last.pth", map_location="cpu", weights_only=False)["last_epoch"]
    except Exception as e:  # noqa: BLE001 - any unreadable file means start again
        print("saved checkpoint unreadable, the fold starts again:", e, flush=True)
        for n in MIRROR:
            (out / n).unlink(missing_ok=True)
        shutil.rmtree(saved)
        return None, -1.0
    # last.pth is only written before augmentation stops, so the resumed run repeats the whole
    # second stage; its earlier best is dropped so the repeat alone decides it.
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
data, ann_root = find_data()
start = find_start()
START_SHA = hashlib.sha256(start.read_bytes()).hexdigest()
print("start weights sha256:", START_SHA, flush=True)
base_cfg = repo / f"configs/dfine/custom/dfine_hgnetv2_{SIZE}_custom.yml"
base = base_cfg.read_text()
batch = 2 if SMOKE else BATCH
lr_scale = BATCH / AUTHORS_BATCH  # the learning rates of the VisDrone training (batch 8)
epochs, aug_off = (2, 1) if SMOKE else (EPOCHS, AUG_OFF)
workers = 0 if SMOKE else 4
done, skipped, fold_hours = [], [], []

for held in FOLDS:
    run = f"dfine_syndrone_minus_{held}"
    if (KEEP / f"{run}.zip").exists() and not SMOKE:
        print(run, "already finished:", KEEP / f"{run}.zip", flush=True)
        done.append(held)
        continue
    out = WORK / run
    out.mkdir(parents=True, exist_ok=True)
    used_h = (time.time() - T0) / 3600
    est_h = max(fold_hours) if fold_hours else 0.0
    if not SMOKE and fold_hours and used_h + est_h > MAX_HOURS:
        skipped.append(held)
        continue
    if SMOKE:
        train_ann, valid_ann = smoke_subset(ann_root, held, out / "smoke", SMOKE)
    else:
        train_ann = ann_root / f"fold_{held}" / "train" / "_annotations.coco.json"
        valid_ann = ann_root / f"fold_{held}" / "valid" / "_annotations.coco.json"
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
    cmd = [sys.executable, "train.py", "-c", str(cfg_path), f"--seed={SEED}"]
    cmd += ["-r", str(resume)] if resume else ["-t", str(start)]
    cmd += ["--device", "cpu"] if SMOKE else ["--use-amp"]
    t_fold = time.time()
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
    fold_hours.append((time.time() - t_fold) / 3600)
    best = next((out / n for n in ("best_stg2.pth", "best_stg1.pth") if (out / n).exists()), None)
    info = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": run,
        "held_out_town": held,
        "model": f"D-FINE {SIZE.upper()}",
        "commit": DFINE_COMMIT,
        "start_weights": start.name,
        "start_weights_sha256": START_SHA,
        "input_size": RES,
        "epochs": epochs,
        "augmentation_off_from_epoch": aug_off,
        "total_batch_size": batch,
        "lr_scale": lr_scale,
        "num_classes": NUM_CLASSES,
        "seed": SEED,
        "platform": "colab" if COLAB else "kaggle" if Path("/kaggle").exists() else "local",
        "resumed": resume is not None,
        "stopped_by_time_limit": stopped_by_time,
        "best_checkpoint": best.name if best else None,
        "smoke": SMOKE,
        "fold_hours_this_session": round(fold_hours[-1], 2),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
    }
    (out / "run.json").write_text(json.dumps(info, indent=1) + "\n")
    if stopped_by_time and COLAB:  # unfinished: resumes next session, so no zip yet
        print(json.dumps(info, indent=1))
        skipped += [f for f in FOLDS if f not in done and f not in skipped]
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
    done.append(held)
    print(json.dumps(info, indent=1))
    print("fold done; download", archive, "(size", archive.stat().st_size // 2**20, "MB)")
    if stopped_by_time:  # Kaggle: the fold is kept with its flag set, the rest wait
        skipped += [f for f in FOLDS if f not in done and f not in skipped]
        break

print("FOLDS DONE:", done)
if skipped:
    print("NOT RUN THIS SESSION (time budget):", skipped)
    if COLAB:
        print("Next session: run the cell again; finished folds are skipped.")
    else:
        print("Next session: set FOLDS =", skipped, "and Save & Run All again.")
