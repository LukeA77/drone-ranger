"""Score the version 2 detector on AU-AIR, the one real drone dataset, on version 1's frames.

Usage: python scripts/auair_detect.py --split calibration
       python scripts/auair_detect.py --split held_out        (once, after the freeze)
       [--run dfine_n_640 --weights weights/dfine_n_640.onnx --kind dfine --imgsz 640]

Same frames, splits, class mapping and scoring as version 1's check in scripts/auair_track.py
(D-506, D-507), so the two compare directly: every `--det-stride`-th frame of each clip in time
order; Human -> person; Car, Truck, Van, Bus -> vehicle; Motorbike, Bicycle and Trailer not
scored; AP at IoU `iou_match` overall and per box-height bin, with the shared score floor of
0.25 (`detect.factory`). AU-AIR played no part in training or tuning the detector.

The held-out split is a one-time test: it runs only if `freeze.py --phase auair_v2 --check`
passes and its result file does not exist yet. Writes results/auair/detect_v2_<run>_<split>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version

import cv2
import numpy as np
import pandas as pd

from aerialrange.detect.evaluate import evaluate_detections
from aerialrange.detect.factory import YOLO_PREDICT_CONF, make_detector
from aerialrange.utils.config import REPO_ROOT, load_config

AUAIR_TO_GROUP = {0: "person", 1: "vehicle", 2: "vehicle", 3: "vehicle", 6: "vehicle"}
HEIGHT_BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]
GROUPS = ("person", "vehicle")


def truth(boxes: list[dict], group: str) -> np.ndarray:
    """AU-AIR boxes of one group as (N, 4) left, top, width, height."""
    rows = [
        [b["left"], b["top"], b["width"], b["height"]]
        for b in boxes
        if AUAIR_TO_GROUP.get(b["class"]) == group
    ]
    return np.asarray(rows, float).reshape(-1, 4)


def stride_frames(frames: pd.DataFrame, clip: int, stride: int) -> pd.DataFrame:
    """Every `stride`-th frame of one clip in time order (version 1's selection)."""
    g = frames[frames["clip"] == clip].sort_values("t_ms")
    return g.iloc[range(0, len(g), stride)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["calibration", "held_out"])
    ap.add_argument("--run", default="dfine_n_640")
    ap.add_argument("--weights", default="weights/dfine_n_640.onnx")
    ap.add_argument("--kind", default="dfine")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--class-map", default="visdrone10")
    ap.add_argument("--det-stride", type=int, default=10)
    args = ap.parse_args(argv)

    dest = REPO_ROOT / "results" / "auair" / f"detect_v2_{args.run}_{args.split}.json"
    if args.split == "held_out":
        if dest.exists():
            print(f"{dest.relative_to(REPO_ROOT)} exists: the held-out test runs once")
            return 1
        chk = [sys.executable, str(REPO_ROOT / "scripts/freeze.py"), "--phase", "auair_v2"]
        if subprocess.run([*chk, "--check"], cwd=REPO_ROOT).returncode != 0:
            print("freeze check failed (or no freeze yet): not running the held-out test")
            return 1

    data, det, au = load_config("data"), load_config("detector"), load_config("auair")
    root = REPO_ROOT / data.require("data_root") / data.require("auair_subdir")
    cmap = {int(k): v for k, v in det.require("class_maps")[args.class_map].items()}
    iou, max_det = det.require("iou_match"), det.require("max_det")
    calib = set(au.require("calib_recordings"))
    frames = pd.read_csv(REPO_ROOT / "cache/auair/frames.csv", dtype={"recording": str})
    boxes = {}
    with (REPO_ROOT / "cache/auair/boxes.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            boxes[r["image_name"]] = r["boxes"]
    predict = make_detector(
        args.kind, REPO_ROOT / args.weights, args.imgsz, YOLO_PREDICT_CONF, max_det, sorted(cmap)
    )

    gt = {g: {} for g in GROUPS}
    dt = {g: {} for g in GROUPS}
    clips, seconds = [], 0.0
    for c in sorted(frames["clip"].unique()):
        sel = stride_frames(frames, c, args.det_stride)
        split = "calibration" if sel["recording"].iloc[0] in calib else "held_out"
        if split != args.split:
            continue
        clips.append(int(c))
        for name, rel in zip(sel["image_name"], sel["image_path"], strict=True):
            img = cv2.imread(str(root / rel))
            if img is None:
                raise SystemExit(f"cannot read {root / rel} (is the data drive connected?)")
            t0 = time.perf_counter()
            b, s, k = predict(img)
            seconds += time.perf_counter() - t0
            grp_of = np.array([cmap.get(int(x), "") for x in k])
            for g in GROUPS:
                gt[g][name] = truth(boxes[name], g)
                m = grp_of == g
                dt[g][name] = (np.asarray(b)[m].reshape(-1, 4), np.asarray(s)[m])
        print(f"clip {c}: {len(sel)} frames", flush=True)

    results = {
        g: {
            "n_images": len(gt[g]),
            "ap50": evaluate_detections(gt[g], dt[g], iou),
            "by_height": {
                f"{lo}-{hi if hi < 1e9 else 'inf'}": evaluate_detections(
                    gt[g], dt[g], iou, height_range=(lo, hi)
                )
                for lo, hi in HEIGHT_BINS
            },
        }
        for g in GROUPS
    }
    v1 = json.loads((REPO_ROOT / "results/auair/detect_B2_yolo26n_960.json").read_text())
    rec = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "weights": args.weights,
        "kind": args.kind,
        "imgsz": args.imgsz,
        "class_map": args.class_map,
        "split": args.split,
        "clips": clips,
        "det_stride": args.det_stride,
        "iou_match": iou,
        "conf_floor": YOLO_PREDICT_CONF,
        "max_det": max_det,
        "class_map_auair": {"person": ["Human"], "vehicle": ["Car", "Truck", "Van", "Bus"]},
        "library": {"onnxruntime": version("onnxruntime")},
        "seconds_per_image_wall": seconds / max(len(gt["person"]), 1),
        "note": "Cross-dataset check on version 1's frames: AU-AIR was not used for training "
        "or tuning. Version 1's result on the same frames is copied in `version1`.",
        "results": results,
        "version1": {
            "source": "results/auair/detect_B2_yolo26n_960.json",
            "run": v1["run"],
            "ap50": {g: v1["results"][args.split][g]["ap50"]["ap"] for g in GROUPS},
        },
    }
    dest.write_text(json.dumps(rec, indent=1, default=float) + "\n", encoding="utf-8")
    for g in GROUPS:
        x = results[g]
        print(
            f"{args.split} {g}: AP50 {x['ap50']['ap']:.3f} ({x['n_images']} images, "
            f"version 1 {rec['version1']['ap50'][g]:.3f})"
        )
    print(f"wrote {dest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
