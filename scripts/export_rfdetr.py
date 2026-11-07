"""Export a trained RF-DETR candidate to ONNX and check parity with the library.

Usage (in the RF-DETR environment, cache/venvs/rfdetr, D-717):
  cache/venvs/rfdetr/Scripts/python.exe scripts/export_rfdetr.py --run rfdetr_nano_640
         [--checkpoint PATH] [--n-frames 50]

Reads cache/train/<run>/checkpoint_best_total.pth (the Kaggle output, gpu/train_rfdetr_kaggle.py)
and loads it with the library's own checkpoint loader, which restores the training input
size; the size is checked against the run's training_config.json. Exports a fixed-size ONNX
model (batch 1) to weights/<run>.onnx, then runs the library (PyTorch) and the project's
ONNX detector (`aerialrange.detect.onnx_detectors.RFDETROnnx`, ONNX Runtime alone) on the same
seeded sample of VisDrone MOT val frames as scripts/export_onnx.py and reports the largest
box and score differences between matched detections and the share matched in each
direction (IoU >= 0.5, same class). The library side uses its own validation transform and
post-processing, and its labels are turned into classes through the checkpoint's own class
names, so the project's pre- and post-processing and its class mapping are all checked.
Writes results/edge/parity_<run>.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerialrange.data.visdrone import CATEGORIES, TRAIN_CLASSES  # noqa: E402
from aerialrange.detect.onnx_detectors import RFDETROnnx  # noqa: E402
from aerialrange.utils.config import load_config  # noqa: E402

PREP = REPO_ROOT / "cache" / "visdrone" / "mot" / "val"


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy box sets."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def compare(a: tuple, b: tuple) -> dict:
    """One-to-one matching of the same class at IoU >= 0.5, best-overlapping pairs first.

    Pairs are taken by overlap, not score: near-duplicate boxes from neighbouring queries
    would otherwise be paired with the wrong partner and show as large differences.
    """
    (ba, sa, ca), (bb, sb, cb) = a, b
    m = iou(ba, bb)
    if m.size:
        m = np.where(ca[:, None] == cb[None, :], m, 0.0)
    ii, jj = np.nonzero(m >= 0.5)
    order = np.argsort(-m[ii, jj], kind="stable")
    used_a = np.zeros(len(ba), bool)
    used_b = np.zeros(len(bb), bool)
    box_diff, score_diff = [], []
    for i, j in zip(ii[order], jj[order], strict=True):
        if used_a[i] or used_b[j]:
            continue
        used_a[i] = used_b[j] = True
        box_diff.append(float(np.max(np.abs(ba[i] - bb[j]))))
        score_diff.append(float(abs(sa[i] - sb[j])))
    return {
        "n_a": len(ba),
        "n_b": len(bb),
        "matched": len(box_diff),
        "max_box_diff_px": max(box_diff, default=0.0),
        "p99_box_diff_px": float(np.percentile(box_diff, 99)) if box_diff else 0.0,
        "max_score_diff": max(score_diff, default=0.0),
        "box_diffs": box_diff,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--n-frames", type=int, default=50)
    ap.add_argument("--score-floor", type=float, default=None, help="default: run config's")
    args = ap.parse_args(argv)
    import rfdetr
    from PIL import Image

    run_dir = REPO_ROOT / "cache" / "train" / args.run
    ckpt = args.checkpoint or run_dir / "checkpoint_best_total.pth"
    cfg = json.loads((ckpt.parent / "training_config.json").read_text())
    model = rfdetr.RFDETR.from_checkpoint(str(ckpt))
    res = int(model.model_config.resolution)
    trained = int(cfg["model_config"]["resolution"])
    if res != trained:
        print(f"loaded at {res} px but trained at {trained} px")
        return 1
    names = [CATEGORIES[c] for c in TRAIN_CLASSES]
    if list(cfg["class_names"]) != names:
        print("class list differs from the VisDrone training classes")
        return 1
    # The library's label -> class name, independent of the project's slot mapping.
    lib_names = model.class_names
    lib_names = dict(enumerate(lib_names)) if isinstance(lib_names, list) else lib_names
    lib_cls = {int(k): names.index(v) for k, v in lib_names.items() if v in names}
    tmp = run_dir / "onnx_export"
    exported = Path(model.export(output_dir=str(tmp), verbose=False))
    dest = REPO_ROOT / "weights" / f"{args.run}.onnx"
    shutil.copy2(exported, dest)
    shutil.rmtree(tmp, ignore_errors=True)

    floor = args.score_floor
    if floor is None:
        floor = float(load_config("run_accurate").require("detector")["score_floor"])
    onnx_det = RFDETROnnx(dest, score_floor=floor)
    frames = []
    for seq in sorted(p for p in PREP.iterdir() if p.is_dir()):
        info = json.loads((seq / "seqinfo.json").read_text())
        frames += sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
    rng = np.random.default_rng(load_config("project").require("seed"))
    pick = sorted(rng.choice(len(frames), size=min(args.n_frames, len(frames)), replace=False))
    import torch
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64

    if not cfg["train_config"].get("square_resize_div_64"):
        print("only square-resize training is supported")
        return 1
    val_tf = make_coco_transforms_square_div_64("val", res)  # the library's validation input
    ctx = model.model
    net = ctx.model.eval()
    rows = []
    for k in pick:
        img = cv2.imread(str(frames[k]))
        h, w = img.shape[:2]
        x, _ = val_tf(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), None)
        with torch.no_grad():
            out = net(x[None].to(next(net.parameters()).device))
            r = ctx.postprocess(out, target_sizes=torch.tensor([[h, w]]))[0]
        labels = r["labels"].cpu().numpy().astype(int)
        keep = (r["scores"] > floor).cpu().numpy() & np.isin(labels, list(lib_cls))
        a = (
            r["boxes"].cpu().numpy()[keep].astype(float),
            r["scores"].cpu().numpy()[keep].astype(float),
            np.array([lib_cls[v] for v in labels[keep]], int),
        )
        d = onnx_det(img)
        rows.append(compare(a, (d.xyxy, d.score, d.cls)))
    diffs = sum((r["box_diffs"] for r in rows), [])
    tot_a = sum(r["n_a"] for r in rows)
    tot_b = sum(r["n_b"] for r in rows)
    matched = sum(r["matched"] for r in rows)
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "checkpoint": ckpt.name,
        "onnx": dest.name,
        "input_hw": [onnx_det.h, onnx_det.w],
        "score_floor": floor,
        "n_frames": len(pick),
        "rfdetr": version("rfdetr"),
        "onnxruntime": version("onnxruntime"),
        "max_box_diff_px": max(r["max_box_diff_px"] for r in rows),
        "p99_box_diff_px": float(np.percentile(diffs, 99)) if diffs else 0.0,
        "max_score_diff": max(r["max_score_diff"] for r in rows),
        "reference": "library validation transform + library post-processing (PyTorch); "
        "classes from the checkpoint's class names",
        "agreement_library_side": matched / max(tot_a, 1),
        "agreement_onnx_side": matched / max(tot_b, 1),
        "detections_library": tot_a,
        "detections_onnx": tot_b,
    }
    path = REPO_ROOT / "results" / "edge" / f"parity_{args.run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
