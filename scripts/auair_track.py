"""Run the chosen detector and tuned tracker on AU-AIR.

Usage: python scripts/auair_track.py --run B2_yolo26n_960 --weights <best.pt> --imgsz 960
       [--clips 0 1 2] [--det-stride 10]

Per clip (frames in time order, 5 fps), live tracking with BoT-SORT and the tuned thresholds
(configs/trackers/botsort_cmc_tuned_<run>.yaml). The tracker was tuned on VisDrone video, which
runs at a higher frame rate, so objects move further between AU-AIR frames; this is a known
difference, not corrected. Tracks go to cache/auair/runs/<run>/clip_NN_<group>.txt (frame
index within the clip, 1-based; id; left, top, width, height; score).

Detection is also scored against AU-AIR's own boxes on every `--det-stride`-th frame (AP50 by
group and box-height bin), mapped like VisDrone (D-507): Human -> person; Car, Truck, Van,
Bus -> vehicle; Motorbike, Bicycle and Trailer are not scored. AU-AIR played no part in
training or tuning, so this is a cross-dataset check. Output:
results/auair/detect_<run>.json, split by calibration and held-out recordings (D-506).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import cv2
import numpy as np
import pandas as pd

from aerialrange.detect.evaluate import evaluate_detections
from aerialrange.track.runner import run_sequence
from aerialrange.utils.config import REPO_ROOT, load_config

AUAIR_TO_GROUP = {0: "person", 1: "vehicle", 2: "vehicle", 3: "vehicle", 6: "vehicle"}
HEIGHT_BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--class-map", default="visdrone10")
    ap.add_argument("--clips", type=int, nargs="*", default=None)
    ap.add_argument("--det-stride", type=int, default=10)
    args = ap.parse_args(argv)
    from ultralytics import YOLO

    data, det, au = load_config("data"), load_config("detector"), load_config("auair")
    root = REPO_ROOT / data.require("data_root") / data.require("auair_subdir")
    cmap = {int(k): v for k, v in det.require("class_maps")[args.class_map].items()}
    iou, max_det = det.require("iou_match"), det.require("max_det")
    calib = set(au.require("calib_recordings"))
    tracker_cfg = str(REPO_ROOT / "configs" / "trackers" / f"botsort_cmc_tuned_{args.run}.yaml")
    fr = pd.read_csv(REPO_ROOT / "cache" / "auair" / "frames.csv", dtype={"recording": str})
    boxes = {}
    with (REPO_ROOT / "cache" / "auair" / "boxes.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            boxes[r["image_name"]] = r["boxes"]
    out = REPO_ROOT / "cache" / "auair" / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    clips = args.clips if args.clips else sorted(fr["clip"].unique())
    gt = {s: {g: {} for g in ("person", "vehicle")} for s in ("calibration", "held_out")}
    dt = {s: {g: {} for g in ("person", "vehicle")} for s in ("calibration", "held_out")}
    timing = {"track_seconds": 0.0, "frames": 0}
    for c in clips:
        g = fr[fr["clip"] == c].sort_values("t_ms")
        split = "calibration" if g["recording"].iloc[0] in calib else "held_out"
        paths = [root / p for p in g["image_path"]]
        if not all((out / f"clip_{c:02d}_{grp}.txt").exists() for grp in ("person", "vehicle")):
            rows, t = run_sequence(model, paths, tracker_cfg, args.imgsz, cmap, max_det=max_det)
            for grp, r in rows.items():
                np.savetxt(out / f"clip_{c:02d}_{grp}.txt", r, fmt="%.2f", delimiter=",")
            timing = {k: timing[k] + t[k] for k in timing}
        for k in range(0, len(g), args.det_stride):
            name = g["image_name"].iloc[k]
            res = model.predict(
                cv2.imread(str(paths[k])),
                imgsz=args.imgsz,
                classes=sorted(cmap),
                max_det=max_det,
                verbose=False,
            )[0]
            xyxy = res.boxes.xyxy.cpu().numpy()
            ltwh = np.column_stack([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]])
            sc, cl = res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)
            grp_of = np.array([cmap.get(int(x), "") for x in cl])
            for grp in ("person", "vehicle"):
                gb = [
                    [b["left"], b["top"], b["width"], b["height"]]
                    for b in boxes[name]
                    if AUAIR_TO_GROUP.get(b["class"]) == grp
                ]
                gt[split][grp][name] = np.asarray(gb, float).reshape(-1, 4)
                m = grp_of == grp
                dt[split][grp][name] = (ltwh[m], sc[m])
        print(f"clip {c} ({split}): {len(g)} frames", flush=True)
    res_out = {}
    for split in ("calibration", "held_out"):
        res_out[split] = {}
        for grp in ("person", "vehicle"):
            if not gt[split][grp]:
                continue
            res_out[split][grp] = {
                "n_images": len(gt[split][grp]),
                "ap50": evaluate_detections(gt[split][grp], dt[split][grp], iou),
                "by_height": {
                    f"{lo}-{hi if hi < 1e9 else 'inf'}": evaluate_detections(
                        gt[split][grp], dt[split][grp], iou, height_range=(lo, hi)
                    )
                    for lo, hi in HEIGHT_BINS
                },
            }
    import ultralytics

    rec = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "weights": args.weights,
        "imgsz": args.imgsz,
        "tracker": tracker_cfg.split("configs")[-1],
        "clips": [int(c) for c in clips],
        "det_stride": args.det_stride,
        "class_map_auair": {"person": ["Human"], "vehicle": ["Car", "Truck", "Van", "Bus"]},
        "note": "Cross-dataset check: AU-AIR was not used for training or tuning. Detection "
        "score floor is the library default (0.25), as for VisDrone AP (D-319).",
        "ultralytics": ultralytics.__version__,
        "timing": timing,
        "results": res_out,
    }
    dest = REPO_ROOT / "results" / "auair" / f"detect_{args.run}.json"
    dest.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    for split, r in res_out.items():
        for grp, x in r.items():
            print(f"{split} {grp}: AP50 {x['ap50']['ap']:.3f} ({x['n_images']} images)")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
