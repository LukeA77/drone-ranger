"""Tracker grid on validation (`{{TRACKER_GRID}}`), from cached detections.

Usage: python scripts/track_grid.py --dets B2_yolo26n_640 --class-map visdrone10
       --tracker configs/trackers/botsort_cmc.yaml [--workers 6]

Replays cache/visdrone/dets/<dets>/val/ (scripts/cache_detections.py) through the tracker for
every grid point and scores each with the usual metrics. The grid varies the tracker's two
confidence thresholds, `track_high_thresh` (first-stage matching) and `new_track_thresh`
(starting a track), around the library defaults (0.25, 0.25), keeping new >= high.
`track_low_thresh` stays at its default, 0.1, which is also the cache's score floor.
Selection rule, fixed before any grid result was seen (D-319): highest mean IDF1 over the two
groups; ties broken by mean MOTA. Validation only: there is no test-dev option.
Output: results/visdrone/val/grid_<dets>__<tracker>.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HIGH = (0.15, 0.25, 0.35, 0.45, 0.55)
NEW = (0.25, 0.35, 0.45, 0.55, 0.65)


def grid_points(
    high: tuple[float, ...] = HIGH, new: tuple[float, ...] = NEW
) -> list[dict[str, float]]:
    return [
        {"track_high_thresh": h, "new_track_thresh": n}
        for h, n in itertools.product(high, new)
        if n >= h
    ]


def run_point(
    dets_run: str, tracker: str, overrides: dict[str, float], cmap: dict[int, str], iou: float
) -> dict[str, Any]:
    import cv2

    from aerialrange.track.detcache import Replay, load_sequence
    from aerialrange.track.evaluate import accumulate, read_ignored, read_rows, summarise
    from aerialrange.track.runner import run_sequence_custom
    from aerialrange.utils.config import REPO_ROOT

    cv2.setNumThreads(1)
    prep = REPO_ROOT / "cache" / "visdrone" / "mot" / "val"
    cache = REPO_ROOT / "cache" / "visdrone" / "dets" / dets_run / "val"
    groups = sorted(set(cmap.values()))
    accs: dict[str, dict] = {g: {} for g in groups}
    seconds = 0.0
    for seq in sorted(p for p in prep.iterdir() if p.is_dir()):
        info = json.loads((seq / "seqinfo.json").read_text())
        frames = sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
        replay = Replay(*load_sequence(cache / f"{seq.name}.npz"))
        if len(replay.dets) != len(frames):
            raise ValueError(f"{seq.name}: cache has {len(replay.dets)} frames, not {len(frames)}")
        rows, t = run_sequence_custom(replay, frames, str(REPO_ROOT / tracker), cmap, overrides)
        seconds += t["track_seconds"]
        ign = read_ignored(seq / "ignored.txt")
        for g in groups:
            accs[g][seq.name] = accumulate(read_rows(seq / f"gt_{g}.txt", 6), rows[g], ign, iou)
    metrics = {g: summarise(accs[g]) for g in groups}
    o = {g: metrics[g]["OVERALL"] for g in groups}
    return {
        "overrides": overrides,
        "mean_idf1": sum(o[g]["idf1"] for g in groups) / len(groups),
        "mean_mota": sum(o[g]["mota"] for g in groups) / len(groups),
        "tracker_seconds": seconds,
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    from aerialrange.utils.config import REPO_ROOT, load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--high", type=float, nargs="+", default=HIGH, help="extension grid only")
    ap.add_argument("--new", type=float, nargs="+", default=NEW, help="extension grid only")
    ap.add_argument("--suffix", default="", help="output name suffix for an extension grid")
    args = ap.parse_args(argv)
    high, new = tuple(args.high), tuple(args.new)
    det = load_config("detector")
    iou = det.require("iou_match")
    cmap = {int(k): v for k, v in det.require("class_maps")[args.class_map].items()}
    meta = json.loads(
        (REPO_ROOT / "cache" / "visdrone" / "dets" / args.dets / "val" / "meta.json").read_text()
    )
    pts = grid_points(high, new)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_point, args.dets, args.tracker, p, cmap, iou) for p in pts]
        results = []
        for f in futs:
            r = f.result()
            o = {g: r["metrics"][g]["OVERALL"] for g in r["metrics"]}
            print(
                f"{r['overrides']}: mean IDF1 {r['mean_idf1']:.3f} "
                + " ".join(f"{g} IDF1 {o[g]['idf1']:.3f} MOTA {o[g]['mota']:.3f}" for g in o),
                flush=True,
            )
            results.append(r)
    best = max(results, key=lambda r: (round(r["mean_idf1"], 4), r["mean_mota"]))
    lib_default = {"track_high_thresh": 0.25, "new_track_thresh": 0.25}
    default = next((r for r in results if r["overrides"] == lib_default), None)
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": "val",
        "dets": args.dets,
        "detector": meta,
        "tracker": Path(args.tracker).name,
        "iou_match": iou,
        "grid": {"track_high_thresh": high, "new_track_thresh": new, "constraint": "new >= high"},
        "selection_rule": "max mean IDF1 over groups (rounded to 4 dp), then mean MOTA (D-319)",
        "selected": best["overrides"],
        "default": default["overrides"] if default else None,
        "points": results,
    }
    tag = Path(args.tracker).stem + args.suffix
    dest = REPO_ROOT / "results" / "visdrone" / "val" / f"grid_{args.dets}__{tag}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    ref = f" (default {default['mean_idf1']:.3f})" if default else ""
    print(f"selected {best['overrides']} mean IDF1 {best['mean_idf1']:.3f}{ref}; wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
