"""Run a detector once over VisDrone MOT sequences and save every frame's detections.

Usage: python scripts/cache_detections.py --run B2_yolo26n_640 --weights <best.pt>
       --class-map visdrone10 --imgsz 640 [--tiled --nms-ios 0.5] [--split val]

Detections go to cache/visdrone/dets/<run>/<split>/<sequence>.npz, for the tracker grid
(scripts/track_grid.py). The score floor is the one `model.track` uses (0.1, the library's
tracking default), so replayed tracking sees the same boxes as live tracking (D-319).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime

import numpy as np

from aerialrange.detect.tiling import detect_tiled
from aerialrange.track.detcache import save_sequence
from aerialrange.utils.config import REPO_ROOT, load_config

PREP = REPO_ROOT / "cache" / "visdrone" / "mot"
# Ultralytics 8.4.182 engine/model.py, Model.track: conf defaults to 0.1 because
# "trackers need low-confidence input".
TRACK_CONF = 0.1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--split", default="val", choices=["val", "test-dev"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--tiled", action="store_true", help="B3: tiled detection (D-317)")
    ap.add_argument("--nms-ios", type=float, default=None, help="tile merge IoS (D-318)")
    args = ap.parse_args(argv)
    if args.split == "test-dev" and not args.final:
        print("test-dev runs once, after all settings are frozen. Use --final.")
        return 1
    import cv2
    from ultralytics import YOLO

    det = load_config("detector")
    max_det = det.require("max_det")
    cmap = {int(k): v for k, v in det.require("class_maps")[args.class_map].items()}
    model = YOLO(args.weights)

    def predict(im: np.ndarray) -> tuple[np.ndarray, ...]:
        r = model.predict(
            im,
            imgsz=args.imgsz,
            classes=sorted(cmap),
            max_det=max_det,
            conf=TRACK_CONF,
            verbose=False,
        )[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        return (
            np.column_stack([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]]),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
        )

    tcfg = det.require("tiling") if args.tiled else None
    out_dir = REPO_ROOT / "cache" / "visdrone" / "dets" / args.run / args.split
    seconds, n = 0.0, 0
    for seq in sorted(p for p in (PREP / args.split).iterdir() if p.is_dir()):
        dest = out_dir / f"{seq.name}.npz"
        if dest.exists():
            print(f"{seq.name}: cached", flush=True)
            continue
        info = json.loads((seq / "seqinfo.json").read_text())
        frames = sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
        dets, shapes = [], []
        for f in frames:
            img = cv2.imread(str(f))
            t0 = time.perf_counter()
            if args.tiled:
                d = detect_tiled(
                    img,
                    lambda im, x0, y0: predict(im),
                    tcfg["tile"],
                    tcfg["overlap"],
                    nms_iou=det.require("iou_match"),
                    include_full=tcfg["include_full"],
                    nms_ios=args.nms_ios,
                )
            else:
                d = predict(img)
            seconds += time.perf_counter() - t0
            n += 1
            dets.append(d)
            shapes.append(img.shape[:2])
        save_sequence(dest, dets, shapes)
        print(f"{seq.name}: {len(frames)} frames", flush=True)
    import ultralytics

    meta = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "weights": str(args.weights),
        "imgsz": args.imgsz,
        "class_map": args.class_map,
        "conf_floor": TRACK_CONF,
        "max_det": max_det,
        "tiled": args.tiled,
        "tiling": tcfg,
        "nms_ios": args.nms_ios,
        "ultralytics": ultralytics.__version__,
        "detect_seconds_this_call": seconds,
        "frames_this_call": n,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
