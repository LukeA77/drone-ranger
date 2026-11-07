"""Detection AP per evaluation group and box-height bin on VisDrone MOT frames.

Usage: python scripts/detect_eval.py --run B0_yolo26n_640 --weights weights/yolo26n.pt
       --class-map coco --imgsz 640 [--kind yolo|rfdetr] [--split val] [--frame-stride 10]
       [--tiled --tile 640 --overlap 0.2]

Uses every `frame-stride`-th frame of each sequence (neighbouring frames are near-duplicates).
Ground truth per group comes from scripts/visdrone_prep.py --mot; predictions in ignored
regions are removed first (D-302). AP at IoU `{{IOU_MATCH}}` and averaged over 0.5:0.95,
overall and per height bin. Every kind of detector is scored with the same score floor,
Ultralytics' predict default of 0.25 that version 1's scores used (`detect.factory`).
Writes results/visdrone/<split>/detect_<run>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from aerialrange.data.visdrone import outside_ignored
from aerialrange.detect.evaluate import evaluate_detections
from aerialrange.detect.factory import KINDS, YOLO_PREDICT_CONF
from aerialrange.detect.tiling import detect_tiled
from aerialrange.track.evaluate import read_ignored, read_rows
from aerialrange.utils.config import REPO_ROOT, load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ultralytics_v1.py
from ultralytics_v1 import library_version, make_detector  # noqa: E402

PREP = REPO_ROOT / "cache" / "visdrone" / "mot"
# Box-height bins (pixels): doubling widths; small objects dominate VisDrone (D-312).
HEIGHT_BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]
COCO_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--kind", default="yolo", choices=KINDS)
    ap.add_argument("--split", default="val", choices=["val", "test-dev"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--frame-stride", type=int, default=10)
    ap.add_argument("--tiled", action="store_true")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--overlap", type=float, default=None)
    ap.add_argument("--include-full", action="store_true", help="also detect the full frame")
    args = ap.parse_args(argv)
    if args.split == "test-dev" and not args.final:
        print("test-dev runs once, after all settings are frozen. Use --final.")
        return 1
    if args.tiled and (args.tile is None or args.overlap is None):
        print("--tiled needs --tile and --overlap ({{TILE_SIZE}}, {{TILE_OVERLAP}})")
        return 1
    import cv2

    det_cfg = load_config("detector")
    iou = det_cfg.require("iou_match")
    max_det = det_cfg.require("max_det")
    cmap = {int(k): v for k, v in det_cfg.require("class_maps")[args.class_map].items()}
    groups = sorted(set(cmap.values()))
    # None keeps the YOLO library's own default (0.25), exactly as version 1 was scored.
    floor = None if args.kind == "yolo" else YOLO_PREDICT_CONF
    predict = make_detector(args.kind, args.weights, args.imgsz, floor, max_det, sorted(cmap))

    gt = {g: {} for g in groups}
    det = {g: {} for g in groups}
    t_total, n_img = 0.0, 0
    for seq in sorted(p for p in (PREP / args.split).iterdir() if p.is_dir()):
        info = json.loads((seq / "seqinfo.json").read_text())
        frames = sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
        ign = read_ignored(seq / "ignored.txt")
        g_rows = {g: read_rows(seq / f"gt_{g}.txt", 6) for g in groups}
        for k in range(0, len(frames), args.frame_stride):
            fnum = k + 1
            img = cv2.imread(str(frames[k]))
            t0 = time.perf_counter()
            if args.tiled:
                b, s, c = detect_tiled(
                    img,
                    lambda im, x0, y0: predict(im),
                    args.tile,
                    args.overlap,
                    nms_iou=iou,
                    include_full=args.include_full,
                )
            else:
                b, s, c = predict(img)
            t_total += time.perf_counter() - t0
            n_img += 1
            if fnum in ign and len(b):
                keep = outside_ignored(b, ign[fnum])
                b, s, c = b[keep], s[keep], c[keep]
            key = f"{seq.name}/{fnum}"
            grp = np.array([cmap.get(int(x), "") for x in c])
            for g in groups:
                rows = g_rows[g]
                gt[g][key] = rows[rows[:, 0] == fnum][:, 2:6]
                m = grp == g
                det[g][key] = (b[m], s[m])
        print(f"{seq.name}: done", flush=True)

    res = {}
    for g in groups:
        res[g] = {
            "ap50": evaluate_detections(gt[g], det[g], iou),
            "ap50_95": float(
                np.mean([evaluate_detections(gt[g], det[g], t)["ap"] for t in COCO_THRESHOLDS])
            ),
            "by_height": {
                f"{lo}-{hi if hi < 1e9 else 'inf'}": evaluate_detections(
                    gt[g], det[g], iou, height_range=(lo, hi)
                )
                for lo, hi in HEIGHT_BINS
            },
        }
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "split": args.split,
        "weights": args.weights,
        "class_map": args.class_map,
        "imgsz": args.imgsz,
        "tiled": args.tiled,
        "include_full": args.include_full,
        "tile": args.tile,
        "overlap": args.overlap,
        "frame_stride": args.frame_stride,
        "n_images": n_img,
        "iou_match": iou,
        "max_det": max_det,
        "kind": args.kind,
        "conf_floor": YOLO_PREDICT_CONF,
        "library": library_version(args.kind, args.weights),
        "seconds_per_image_wall": t_total / max(n_img, 1),
        "timing_note": "Wall time, not the latency benchmark protocol (D-601).",
        "results": res,
    }
    dest = REPO_ROOT / "results" / "visdrone" / args.split / f"detect_{args.run}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for g in groups:
        print(f"{g}: AP50 {res[g]['ap50']['ap']:.3f} AP50:95 {res[g]['ap50_95']:.3f}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
