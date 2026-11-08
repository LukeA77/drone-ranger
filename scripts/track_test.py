"""Score the frozen version 2 detector and tracker once on VisDrone MOT test-dev.

Usage: python scripts/track_test.py --dets dfine_n_640 --class-map visdrone10
       --tracker configs/trackers/motion_dfine_n_640.yaml --final

Replays the detector's cached test-dev detections (scripts/cache_detections.py --split test-dev
--final) through the project's tracker with frozen settings, every frame detected, as version
1's test run was (scripts/track_eval.py). Replaying cached detections gives the same tracks as
live tracking (D-319). Predictions go to cache/visdrone/runs/<run>/test-dev/, where
scripts/track_breakdowns.py reads them, and metrics to results/visdrone/test-dev/<run>.json.
The run name is `<dets>__<tracker>`. It refuses to run without --final, and refuses if that
result already exists: the test set is scored once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from aerialrange.track.detcache import Replay, load_sequence
from aerialrange.track.evaluate import accumulate, read_ignored, read_rows, summarise
from aerialrange.track.runner import make_tracker, run_sequence_custom
from aerialrange.utils.config import REPO_ROOT, load_config

SPLIT = "test-dev"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args(argv)
    if not args.final:
        print("test-dev runs once, after all settings are frozen. Use --final.")
        return 1
    run = f"{args.dets}__{Path(args.tracker).stem}"
    dest = REPO_ROOT / "results" / "visdrone" / SPLIT / f"{run}.json"
    if dest.exists():
        print(f"{dest} exists: the test set has been scored with these settings already.")
        return 1
    det = load_config("detector")
    iou = det.require("iou_match")
    cmap = {int(k): v for k, v in det.require("class_maps")[args.class_map].items()}
    groups = sorted(set(cmap.values()))
    prep = REPO_ROOT / "cache" / "visdrone" / "mot" / SPLIT
    cache = REPO_ROOT / "cache" / "visdrone" / "dets" / args.dets / SPLIT
    meta = json.loads((cache / "meta.json").read_text())
    pred_dir = REPO_ROOT / "cache" / "visdrone" / "runs" / run / SPLIT
    pred_dir.mkdir(parents=True, exist_ok=True)
    accs: dict[str, dict] = {g: {} for g in groups}
    seconds, n_frames = 0.0, 0
    seqs = sorted(p for p in prep.iterdir() if p.is_dir())
    for seq in seqs:
        info = json.loads((seq / "seqinfo.json").read_text())
        frames = sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
        replay = Replay(*load_sequence(cache / f"{seq.name}.npz"))
        if len(replay.dets) != len(frames):
            raise ValueError(f"{seq.name}: cache has {len(replay.dets)} frames, not {len(frames)}")
        rows, t = run_sequence_custom(
            replay,
            frames,
            str(REPO_ROOT / args.tracker),
            cmap,
            {},
            det_stride=1,
            build_tracker=make_tracker,
        )
        seconds += t["track_seconds"]
        n_frames += len(frames)
        ign = read_ignored(seq / "ignored.txt")
        for g in groups:
            np.savetxt(pred_dir / f"{seq.name}_{g}.txt", rows[g], fmt="%.2f", delimiter=",")
            accs[g][seq.name] = accumulate(read_rows(seq / f"gt_{g}.txt", 6), rows[g], ign, iou)
        print(f"{seq.name}: {len(frames)} frames", flush=True)
    metrics = {g: summarise(accs[g]) for g in groups}
    commit = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
        ).stdout.strip()
        or None
    )
    tracker_text = (REPO_ROOT / args.tracker).read_bytes()
    o = {g: metrics[g]["OVERALL"] for g in groups}
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "run": run,
        "split": SPLIT,
        "dets": args.dets,
        "detector": meta,
        "tracker": Path(args.tracker).name,
        "tracker_sha256": hashlib.sha256(tracker_text).hexdigest(),
        "det_stride": 1,
        "class_map": args.class_map,
        "iou_match": iou,
        "sequences": [s.name for s in seqs],
        "frames": n_frames,
        "tracker_seconds": seconds,
        "mean_idf1": sum(o[g]["idf1"] for g in groups) / len(groups),
        "mean_mota": sum(o[g]["mota"] for g in groups) / len(groups),
        "metrics": metrics,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for g in groups:
        print(
            f"{g}: MOTA {o[g]['mota']:.3f} IDF1 {o[g]['idf1']:.3f} IDSW {o[g]['num_switches']:.0f}"
        )
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
