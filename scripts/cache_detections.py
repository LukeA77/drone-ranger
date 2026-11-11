"""Run a detector once over VisDrone MOT sequences and save every frame's detections.

Usage: python scripts/cache_detections.py --run B2_yolo26n_640 --weights <best.pt>
       --class-map visdrone10 --imgsz 640 [--kind yolo|rfdetr] [--tiled --nms-ios 0.5]
       [--split val]

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
from pathlib import Path

from aerialrange.detect.factory import KINDS
from aerialrange.detect.tiling import detect_tiled
from aerialrange.track.detcache import save_sequence
from aerialrange.utils.config import REPO_ROOT, load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ultralytics_v1.py
from ultralytics_v1 import library_version, make_detector  # noqa: E402

PREP_ROOT = REPO_ROOT / "cache"  # <dataset>/mot/<split>/<sequence>/
# BIRDSAI (D-754): scripts/birdsai_prep.py; train tunes, test runs once
SPLITS = {"visdrone": ("val", "test-dev"), "birdsai": ("train", "test")}
# Ultralytics 8.4.182 engine/model.py, Model.track: conf defaults to 0.1 because
# "trackers need low-confidence input".
TRACK_CONF = 0.1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--kind", default="yolo", choices=KINDS)
    ap.add_argument("--dataset", default="visdrone", choices=sorted(SPLITS))
    ap.add_argument("--split", default="val", choices=["val", "test-dev", "train", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--tiled", action="store_true", help="B3: tiled detection (D-317)")
    ap.add_argument("--nms-ios", type=float, default=None, help="tile merge IoS (D-318)")
    args = ap.parse_args(argv)
    if args.split not in SPLITS[args.dataset]:
        print(f"{args.dataset} splits: {SPLITS[args.dataset]}")
        return 1
    if args.split in ("test-dev", "test") and not args.final:
        print(f"{args.split} runs once, after all settings are frozen. Use --final.")
        return 1
    import cv2

    det = load_config("detector")
    max_det = det.require("max_det")
    cmap = {int(k): v for k, v in det.require("class_maps")[args.class_map].items()}
    predict = make_detector(args.kind, args.weights, args.imgsz, TRACK_CONF, max_det, sorted(cmap))

    tcfg = det.require("tiling") if args.tiled else None
    out_dir = REPO_ROOT / "cache" / args.dataset / "dets" / args.run / args.split
    seconds, n = 0.0, 0
    for seq in sorted(
        p for p in (PREP_ROOT / args.dataset / "mot" / args.split).iterdir() if p.is_dir()
    ):
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
        "kind": args.kind,
        "library": library_version(args.kind, args.weights),
        "detect_seconds_this_call": seconds,
        "frames_this_call": n,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
