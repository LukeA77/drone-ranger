"""Track MOT val with a detector's tuned tracker settings and save the tracks and events.

Usage: python scripts/track_tuned.py --dets B2_yolo26n_960

Replays the cached detections (scripts/cache_detections.py) through BoT-SORT with camera-motion
compensation and the tuned thresholds in configs/tracker.yaml (`tuned`, D-321). Writes the
tracks to cache/visdrone/runs/<dets>_tuned/val/<sequence>_<group>.txt and the matching events
(per frame: match, miss, false positive, switch, with object and hypothesis ids) to
cache/visdrone/runs/<dets>_tuned/val/events_<group>.csv, for the failure gallery
(scripts/failure_gallery.py). Validation only.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from aerialrange.track.detcache import Replay, load_sequence
from aerialrange.track.evaluate import accumulate, read_ignored, read_rows, summarise
from aerialrange.track.runner import run_sequence_custom
from aerialrange.utils.config import REPO_ROOT, load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--class-map", default="visdrone10")
    ap.add_argument("--tuned-from", default=None, help="tracker.yaml `tuned` key (default --dets)")
    ap.add_argument("--gmc-downscale", type=int, default=None)
    ap.add_argument("--tag", default=None, help="output run name (default <dets>_tuned)")
    ap.add_argument("--tracker", default="botsort_cmc", help="configs/trackers/<name>.yaml")
    ap.add_argument("--high", type=float, default=None, help="override track_high_thresh")
    ap.add_argument("--new", type=float, default=None, help="override new_track_thresh")
    ap.add_argument("--det-stride", type=int, default=1, help="detect every n-th frame (D-610)")
    args = ap.parse_args(argv)
    det, trk = load_config("detector"), load_config("tracker")
    iou = det.require("iou_match")
    cmap = {int(k): v for k, v in det.require("class_maps")[args.class_map].items()}
    groups = sorted(set(cmap.values()))
    overrides = dict(trk.require("tuned")[args.tuned_from or args.dets])
    if args.gmc_downscale is not None:
        overrides["gmc_downscale"] = args.gmc_downscale
    if args.high is not None:
        overrides["track_high_thresh"] = args.high
    if args.new is not None:
        overrides["new_track_thresh"] = args.new
    tracker_cfg = str(REPO_ROOT / "configs" / "trackers" / f"{args.tracker}.yaml")
    prep = REPO_ROOT / "cache" / "visdrone" / "mot" / "val"
    cache = REPO_ROOT / "cache" / "visdrone" / "dets" / args.dets / "val"
    out = REPO_ROOT / "cache" / "visdrone" / "runs" / (args.tag or f"{args.dets}_tuned") / "val"
    out.mkdir(parents=True, exist_ok=True)
    accs: dict[str, dict] = {g: {} for g in groups}
    for seq in sorted(p for p in prep.iterdir() if p.is_dir()):
        info = json.loads((seq / "seqinfo.json").read_text())
        frames = sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
        replay = Replay(*load_sequence(cache / f"{seq.name}.npz"))
        rows, _ = run_sequence_custom(
            replay, frames, tracker_cfg, cmap, overrides, det_stride=args.det_stride
        )
        ign = read_ignored(seq / "ignored.txt")
        for g in groups:
            np.savetxt(out / f"{seq.name}_{g}.txt", rows[g], fmt="%.2f", delimiter=",")
            accs[g][seq.name] = accumulate(read_rows(seq / f"gt_{g}.txt", 6), rows[g], ign, iou)
        print(f"{seq.name}: done", flush=True)
    for g in groups:
        ev = []
        for s, acc in accs[g].items():
            e = acc.mot_events.reset_index()
            e.insert(0, "sequence", s)
            ev.append(e[e["Type"] != "RAW"])
        pd.concat(ev).to_csv(out / f"events_{g}.csv", index=False)
        o = summarise(accs[g])["OVERALL"]
        print(f"{g}: IDF1 {o['idf1']:.3f} MOTA {o['mota']:.3f} IDSW {o['num_switches']:.0f}")
    summary = {g: summarise(accs[g])["OVERALL"] for g in groups}
    rec = {
        "dets": args.dets,
        "overrides": overrides,
        "det_stride": args.det_stride,
        "metrics": summary,
    }
    (out / "summary.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
