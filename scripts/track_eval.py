"""Run a detector + tracker on VisDrone MOT and score it.

Usage: python scripts/track_eval.py --run B0_yolo26n_640_bytetrack --weights weights/yolo26n.pt
       --class-map coco --imgsz 640 --tracker configs/trackers/bytetrack.yaml [--split val]
       [--sequences N]

Predictions go to cache/visdrone/runs/<run>/<split>/; metrics per group to
results/visdrone/<split>/<run>.json. The test-dev split runs once, with --final.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from aerialrange.detect.tiling import detect_tiled
from aerialrange.track.evaluate import accumulate, read_ignored, read_rows, summarise
from aerialrange.track.runner import run_sequence, run_sequence_custom
from aerialrange.utils.config import REPO_ROOT, load_config

PREP = REPO_ROOT / "cache" / "visdrone" / "mot"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test-dev"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--sequences", type=int, default=None, help="limit (smoke tests only)")
    ap.add_argument("--tiled", action="store_true", help="B3: tiled detection (D-317)")
    ap.add_argument("--nms-ios", type=float, default=None, help="tile merge IoS threshold (D-318)")
    args = ap.parse_args(argv)
    if args.split == "test-dev" and not args.final:
        print("test-dev runs once, after all settings are frozen. Use --final.")
        return 1
    from ultralytics import YOLO

    det = load_config("detector")
    iou = det.require("iou_match")
    cmap = {int(k): v for k, v in det.require("class_maps")[args.class_map].items()}
    groups = sorted(set(cmap.values()))
    model = YOLO(args.weights)
    seqs = sorted(p for p in (PREP / args.split).iterdir() if p.is_dir())[: args.sequences]
    pred_dir = REPO_ROOT / "cache" / "visdrone" / "runs" / args.run / args.split
    pred_dir.mkdir(parents=True, exist_ok=True)
    accs = {g: {} for g in groups}
    timing = {"track_seconds": 0.0, "frames": 0}
    for seq in seqs:
        info = json.loads((seq / "seqinfo.json").read_text())
        frames = sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
        if args.tiled:
            # B3: tiled detection (configs/detector.yaml `tiling`, D-317) fed to the tracker.
            tcfg = det.require("tiling")
            max_det = det.require("max_det")

            def predict(im: np.ndarray, max_det: int = max_det) -> tuple[np.ndarray, ...]:
                r = model.predict(
                    im, imgsz=args.imgsz, classes=sorted(cmap), max_det=max_det, verbose=False
                )[0]
                xyxy = r.boxes.xyxy.cpu().numpy()
                return (
                    np.column_stack([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]]),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy().astype(int),
                )

            def detect(img: np.ndarray, tcfg: dict = tcfg) -> tuple[np.ndarray, ...]:
                return detect_tiled(
                    img,
                    lambda im, x0, y0: predict(im),
                    tcfg["tile"],
                    tcfg["overlap"],
                    nms_iou=iou,
                    include_full=tcfg["include_full"],
                    nms_ios=args.nms_ios,
                )

            rows, t = run_sequence_custom(detect, frames, str(REPO_ROOT / args.tracker), cmap)
        else:
            rows, t = run_sequence(
                model,
                frames,
                str(REPO_ROOT / args.tracker),
                args.imgsz,
                cmap,
                max_det=det.require("max_det"),
            )
        timing = {k: timing[k] + t[k] for k in timing}
        ign = read_ignored(seq / "ignored.txt")
        for g in groups:
            np.savetxt(pred_dir / f"{seq.name}_{g}.txt", rows[g], fmt="%.2f", delimiter=",")
            accs[g][seq.name] = accumulate(read_rows(seq / f"gt_{g}.txt", 6), rows[g], ign, iou)
        print(f"{seq.name}: {len(frames)} frames, {t['track_seconds']:.0f} s", flush=True)
    metrics = {g: summarise(accs[g]) for g in groups}
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            ).stdout.strip()
            or None
        )
    except OSError:
        commit = None
    import ultralytics

    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "run": args.run,
        "split": args.split,
        "weights": Path(args.weights).name,
        "imgsz": args.imgsz,
        "tracker": Path(args.tracker).name,
        "tiled": args.tiled,
        "nms_ios": args.nms_ios,
        "class_map": args.class_map,
        "iou_match": iou,
        "max_det": det.require("max_det"),
        "ultralytics": ultralytics.__version__,
        "sequences": [s.name for s in seqs],
        "timing_note": "Wall time of detector+tracker calls; not a latency benchmark "
        "(latency benchmark protocol of D-601 not applied).",
        "timing": timing,
        "metrics": metrics,
    }
    dest = REPO_ROOT / "results" / "visdrone" / args.split / f"{args.run}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for g in groups:
        o = metrics[g]["OVERALL"]
        print(f"{g}: MOTA {o['mota']:.3f} IDF1 {o['idf1']:.3f} IDSW {o['num_switches']:.0f}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
