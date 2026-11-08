"""Tracker grid on validation (`{{TRACKER_GRID}}`), from cached detections.

Usage: python scripts/track_grid.py --dets B2_yolo26n_640 --class-map visdrone10
       --tracker configs/trackers/botsort_cmc.yaml [--workers 6]

Replays cache/visdrone/dets/<dets>/val/ (scripts/cache_detections.py) through the tracker for
every grid point and scores each with the usual metrics. The grid varies the tracker's two
confidence thresholds, `track_high_thresh` (first-stage matching) and `new_track_thresh`
(starting a track), around the library defaults (0.25, 0.25), keeping new >= high.
`track_low_thresh` stays at its default, 0.1, which is also the cache's score floor.
Selection rule, fixed before any grid result was seen (D-319): highest mean IDF1 over the two
groups; ties broken by mean MOTA.

`--group person|vehicle` (project tracker only, D-733) varies the thresholds of that group's
classes alone (`high_by_class`, `new_by_class`); the other group keeps the tracker file's values.
Rules: people, highest person F2 (CLEAR precision and recall, beta 2), ties by person IDF1;
vehicles, highest vehicle IDF1, ties by vehicle MOTA. Every point also records, per group,
ground-truth object-frames, matched object-frames and identity switches by box height.
Validation only: there is no test-dev option.
Output: results/visdrone/val/grid_<dets>__<tracker>[__<group>].json.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # ultralytics_v1, also in workers

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
    from aerialrange.track.evaluate import (
        HEIGHT_BINS,
        accumulate,
        counts_by_height,
        height_key,
        read_ignored,
        read_rows,
        summarise,
    )
    from aerialrange.track.runner import run_sequence_custom
    from aerialrange.utils.config import REPO_ROOT
    from ultralytics_v1 import make_tracker  # version 1's library trackers, or the project's

    cv2.setNumThreads(1)
    prep = REPO_ROOT / "cache" / "visdrone" / "mot" / "val"
    cache = REPO_ROOT / "cache" / "visdrone" / "dets" / dets_run / "val"
    groups = sorted(set(cmap.values()))
    trk_ov = dict(overrides)
    stride = int(trk_ov.pop("det_stride", 1))  # detection stride (D-610), not a tracker key
    group = trk_ov.pop("group", None)
    if group is not None:  # thresholds for this group's classes only (D-733)
        ids = [c for c, g in cmap.items() if g == group]
        h, n = trk_ov.pop("track_high_thresh"), trk_ov.pop("new_track_thresh")
        trk_ov["high_by_class"] = dict.fromkeys(ids, h)
        trk_ov["new_by_class"] = dict.fromkeys(ids, n)
    accs: dict[str, dict] = {g: {} for g in groups}
    by_h = {g: np.zeros((3, len(HEIGHT_BINS))) for g in groups}
    seconds = 0.0
    for seq in sorted(p for p in prep.iterdir() if p.is_dir()):
        info = json.loads((seq / "seqinfo.json").read_text())
        frames = sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
        replay = Replay(*load_sequence(cache / f"{seq.name}.npz"))
        if len(replay.dets) != len(frames):
            raise ValueError(f"{seq.name}: cache has {len(replay.dets)} frames, not {len(frames)}")
        rows, t = run_sequence_custom(
            replay,
            frames,
            str(REPO_ROOT / tracker),
            cmap,
            trk_ov,
            det_stride=stride,
            build_tracker=make_tracker,
        )
        seconds += t["track_seconds"]
        ign = read_ignored(seq / "ignored.txt")
        for g in groups:
            gt = read_rows(seq / f"gt_{g}.txt", 6)
            accs[g][seq.name] = accumulate(gt, rows[g], ign, iou)
            by_h[g] += counts_by_height(accs[g][seq.name], gt)
    metrics = {g: summarise(accs[g]) for g in groups}
    o = {g: metrics[g]["OVERALL"] for g in groups}
    return {
        "overrides": overrides,
        "mean_idf1": sum(o[g]["idf1"] for g in groups) / len(groups),
        "mean_mota": sum(o[g]["mota"] for g in groups) / len(groups),
        "f2": {g: f_beta(o[g]["precision"], o[g]["recall"], 2.0) for g in groups},
        "by_height": {
            g: {
                height_key(lo, hi): {
                    "gt_object_frames": int(by_h[g][0, b]),
                    "matched": int(by_h[g][1, b]),
                    "switches": int(by_h[g][2, b]),
                }
                for b, (lo, hi) in enumerate(HEIGHT_BINS)
            }
            for g in groups
        },
        "tracker_seconds": seconds,
        "metrics": metrics,
    }


def f_beta(precision: float, recall: float, beta: float) -> float:
    """F-measure weighing recall `beta` times as much as precision (van Rijsbergen)."""
    b2 = beta * beta
    den = b2 * precision + recall
    return (1 + b2) * precision * recall / den if den > 0 else 0.0


RULES = {
    None: "max mean IDF1 over groups (rounded to 4 dp), then mean MOTA (D-319)",
    "person": "max person F2 (rounded to 4 dp), then person IDF1 (D-733)",
    "vehicle": "max vehicle IDF1 (rounded to 4 dp), then vehicle MOTA (D-733)",
}


def select_key(r: dict[str, Any], group: str | None) -> tuple[float, float]:
    """Sort key of a grid point under the selection rule for `group` (higher is better)."""
    if group is None:
        return (round(r["mean_idf1"], 4), r["mean_mota"])
    o = r["metrics"][group]["OVERALL"]
    if group == "person":
        return (round(r["f2"]["person"], 4), o["idf1"])
    return (round(o["idf1"], 4), o["mota"])


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
    ap.add_argument(
        "--group", choices=["person", "vehicle"], help="tune this group's thresholds only (D-733)"
    )
    args = ap.parse_args(argv)
    high, new = tuple(args.high), tuple(args.new)
    det = load_config("detector")
    iou = det.require("iou_match")
    cmap = {int(k): v for k, v in det.require("class_maps")[args.class_map].items()}
    meta = json.loads(
        (REPO_ROOT / "cache" / "visdrone" / "dets" / args.dets / "val" / "meta.json").read_text()
    )
    pts = grid_points(high, new)
    if args.group:
        pts = [{**p, "group": args.group} for p in pts]
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
    best = max(results, key=lambda r: select_key(r, args.group))
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
        "group": args.group,
        "selection_rule": RULES[args.group],
        "selected": best["overrides"],
        "default": default["overrides"] if default else None,
        "points": results,
    }
    tag = Path(args.tracker).stem + (f"__{args.group}" if args.group else "") + args.suffix
    dest = REPO_ROOT / "results" / "visdrone" / "val" / f"grid_{args.dets}__{tag}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    ref = f" (default {default['mean_idf1']:.3f})" if default else ""
    print(f"selected {best['overrides']} mean IDF1 {best['mean_idf1']:.3f}{ref}; wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
