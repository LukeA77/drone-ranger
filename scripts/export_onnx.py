"""Export a detector to ONNX and check numerical parity.

Usage: python scripts/export_onnx.py --weights weights/yolo26n.pt --imgsz 640 [--n-frames 50]

Exports with the detector library, then runs the original and the ONNX model (ONNX Runtime,
CPU) on a fixed sample of VisDrone MOT val frames (every k-th frame, seeded) and reports:
the largest absolute difference in box coordinates (px) and scores between matched
detections, and detection-level agreement (fraction matched at IoU >= 0.5 with the same
class, in both directions). `{{PARITY_TOL}}` (configs/edge.yaml) is compared only if set;
the actual differences are always reported. Writes results/edge/parity_<name>.json.

The ONNX input shape matches what the original model sees (D-604): the library letterboxes a
16:9 frame only to the next multiple of the stride (e.g. 960 x 544), so a square static export
sees a differently padded image and gives different detections. `--square` keeps the old
square export for comparison.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from aerialrange.track.evaluate import iou_distance
from aerialrange.utils.config import REPO_ROOT, load_config

PREP = REPO_ROOT / "cache" / "visdrone" / "mot" / "val"


def _dets(model: object, img: np.ndarray, imgsz: int) -> tuple[np.ndarray, ...]:
    r = model.predict(img, imgsz=imgsz, verbose=False)[0]
    xyxy = r.boxes.xyxy.cpu().numpy()
    ltwh = np.column_stack([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]])
    return ltwh, r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)


def rect_height(imgsz: int, h: int, w: int, stride: int) -> int:
    """Input height the library uses for a landscape frame scaled to width `imgsz`: the scaled
    height rounded up to a multiple of the stride."""
    return int(np.ceil(imgsz * h / w / stride) * stride)


def compare(a: tuple, b: tuple) -> dict[str, float]:
    """Greedy one-to-one matching by IoU >= 0.5 and same class; differences of matched pairs."""
    (ba, sa, ca), (bb, sb, cb) = a, b
    d = iou_distance(ba, bb, 0.5)
    if d.size:
        d = np.where(ca[:, None] == cb[None, :], d, np.nan)
    used = np.zeros(len(bb), bool)
    box_diff, score_diff, matched = [], [], 0
    for i in np.argsort(-sa):
        if not d.size:
            break
        row = np.where(used, np.nan, d[i])
        if np.all(np.isnan(row)):
            continue
        j = int(np.nanargmin(row))
        used[j] = True
        matched += 1
        box_diff.append(np.max(np.abs(ba[i] - bb[j])))
        score_diff.append(abs(sa[i] - sb[j]))
    return {
        "n_a": len(ba),
        "n_b": len(bb),
        "matched": matched,
        "max_box_diff_px": float(max(box_diff, default=0.0)),
        "max_score_diff": float(max(score_diff, default=0.0)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--n-frames", type=int, default=50)
    ap.add_argument("--square", action="store_true", help="square input (mismatched, D-604)")
    ap.add_argument("--name", default=None, help="output stem (default: weights stem)")
    args = ap.parse_args(argv)
    import cv2
    import onnxruntime
    import ultralytics
    from ultralytics import YOLO

    src = REPO_ROOT / args.weights
    model = YOLO(src)
    frames = []
    for seq in sorted(p for p in PREP.iterdir() if p.is_dir()):
        info = json.loads((seq / "seqinfo.json").read_text())
        frames += sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
    h0, w0 = cv2.imread(str(frames[0])).shape[:2]
    stride = int(max(model.model.stride))
    shape = args.imgsz if args.square else [rect_height(args.imgsz, h0, w0, stride), args.imgsz]
    onnx_path = Path(model.export(format="onnx", imgsz=shape, dynamic=False))
    tag = "sq" if args.square else "rect"
    dest_onnx = REPO_ROOT / "weights" / f"{args.name or src.stem}_{args.imgsz}_{tag}.onnx"
    if onnx_path.resolve() != dest_onnx.resolve():
        shutil.move(onnx_path, dest_onnx)
    onnx_model = YOLO(dest_onnx, task="detect")
    rng = np.random.default_rng(load_config("project").require("seed"))
    pick = sorted(rng.choice(len(frames), size=min(args.n_frames, len(frames)), replace=False))
    rows = []
    for k in pick:
        img = cv2.imread(str(frames[k]))
        rows.append(compare(_dets(model, img, args.imgsz), _dets(onnx_model, img, args.imgsz)))
    tot_a = sum(r["n_a"] for r in rows)
    tot_b = sum(r["n_b"] for r in rows)
    matched = sum(r["matched"] for r in rows)
    tol = load_config("edge").get("parity_tol")
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "weights": args.weights,
        "onnx": dest_onnx.name,
        "imgsz": args.imgsz,
        "onnx_input_hw": shape if isinstance(shape, list) else [shape, shape],
        "n_frames": len(pick),
        "ultralytics": ultralytics.__version__,
        "onnxruntime": onnxruntime.__version__,
        "max_box_diff_px": max(r["max_box_diff_px"] for r in rows),
        "max_score_diff": max(r["max_score_diff"] for r in rows),
        "agreement_original_side": matched / max(tot_a, 1),
        "agreement_onnx_side": matched / max(tot_b, 1),
        "detections_original": tot_a,
        "detections_onnx": tot_b,
        "parity_tol": tol,
        "within_tol": None if tol is None else bool(max(r["max_box_diff_px"] for r in rows) <= tol),
    }
    dest = REPO_ROOT / "results" / "edge" / f"parity_{dest_onnx.stem}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
