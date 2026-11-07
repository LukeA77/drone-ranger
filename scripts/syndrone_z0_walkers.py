"""Test a walker-based fit of SynDrone's base height z0 against the vehicle-based fit.

Usage: python scripts/syndrone_z0_walkers.py [--towns Town01 Town07 Town10HD Town04 Town05]
       [--stride 25] [--workers 4]

Why: the vehicle fit (D-704) needs vehicles standing on road. In Town02 every vehicle actor
is a parked car that never moves and stands 11 to 150 m from any road, so no frame can be
fitted. Walkers stand on the ground too: for each walker actor, the ground pixels in a 1 m ring
around its footprint, of the ground class most common in that ring (road, road line,
sidewalk, ground or terrain; one class only, so a kerb cannot mix two levels), must lie at the
box's bottom. Each walker gives one offset; a Huber estimate over all walkers of the frame
from the three cameras gives z0 (the same estimator as the vehicle fit).

Test: on development and calibration towns (never test towns), every `stride`-th frame where
the vehicle fit rests on at least 3 vehicles, compare the two estimates. Acceptance rule, set
before any result was seen: the walker fit may replace the vehicle fit where the latter is
missing if, after subtracting the median walker-minus-vehicle difference over these towns (a
fixed bias, e.g. a walker box bottom sitting slightly off its feet), the absolute difference
has median <= 0.05 m and 90th percentile <= 0.10 m in every town. At a 20 m camera, 0.10 m is
a 0.5 % range error, half the pipeline's measured median range error (D-718).
Writes results/syndrone/calibration/z0_walkers_check.json (per-frame values and the rule).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from aerialrange.utils.config import REPO_ROOT

GROUND_TAGS = (6, 7, 8, 14, 22)  # road line, road, sidewalk, ground, terrain (CARLA labelmap)
WALKER_TAG = 40
RING_M = 1.0
MIN_PX = 30
MIN_VEHICLES = 3
RULE = {"median_abs_m": 0.05, "p90_abs_m": 0.10}


def walker_offsets(view, boxes) -> np.ndarray:  # noqa: ANN001
    from scipy.spatial import cKDTree

    from aerialrange.data.syndrone import BOTTOM
    from aerialrange.evaluate.syndrone_truth import _inside_footprint, box_frames

    P = view.rel_points()
    ok = np.isin(view.semantic, GROUND_TAGS) & (view.depth < 999.0)
    pts, cls = P[ok], view.semantic[ok]
    sel = (~boxes.level) & (boxes.tags == WALKER_TAG)
    if len(pts) == 0 or not sel.any():
        return np.zeros(0)
    tree = cKDTree(pts[:, :2])
    cen, axes, half = box_frames(boxes.corners[sel])
    bottoms = boxes.corners[sel][:, BOTTOM, 2].mean(axis=1)
    out = []
    for c, a, hh, zb in zip(cen, axes, half, bottoms, strict=True):
        idx = tree.query_ball_point(c[:2], float(np.hypot(hh[0] + RING_M, hh[1] + RING_M)))
        if len(idx) < MIN_PX:
            continue
        near, nc = pts[idx], cls[idx]
        ring = _inside_footprint(near[:, :2], c, a, hh, RING_M)
        ring &= ~_inside_footprint(near[:, :2], c, a, hh, 0.0)
        if ring.sum() < MIN_PX:
            continue
        u, n = np.unique(nc[ring], return_counts=True)
        r2 = ring & (nc == u[np.argmax(n)])
        if r2.sum() < MIN_PX:
            continue
        out.append(zb - float(np.median(near[r2, 2])))
    return np.asarray(out, float)


def frame_pair(args: tuple[str, int]) -> dict:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from aerialrange.data.syndrone import SynDroneSequence, decode_depth
    from aerialrange.evaluate.syndrone_truth import TruthSettings, View, estimate_z0, huber_location
    from syndrone_truth import HEIGHTS, _store

    town, k = args
    st = _store()
    seqs = {h: SynDroneSequence(st, town, h) for h in HEIGHTS}
    boxes = seqs[80].boxes(k)
    views = []
    for h, q in seqs.items():
        name = q.member("depth", k)
        depth = decode_depth(dict(st.iter_read([name]))[name])
        views.append(View(h, q.camera_record(k), depth, q.semantic(k), q.camera))
    s = TruthSettings()
    fit = estimate_z0(views, boxes, s)
    wo = np.concatenate([walker_offsets(v, boxes) for v in views])
    mu, se, _ = huber_location(wo, s.huber_k)
    return {
        "town": town,
        "frame": k,
        "z0_vehicles": float(fit.z0),
        "n_vehicles": int(fit.n_boxes),
        "z0_walkers": float(mu),
        "se_walkers": float(se),
        "n_walkers": len(wo),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--towns", nargs="+", default=["Town01", "Town07", "Town10HD", "Town04", "Town05"]
    )
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    splits = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["syndrone"]["splits"]
    if set(args.towns) & set(splits["test"]):
        print("test towns are not used to choose the rule")
        return 1
    jobs = [(t, k) for t in args.towns for k in range(0, 3000, args.stride)]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = []
        for i, r in enumerate(ex.map(frame_pair, jobs, chunksize=4)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)} frames", flush=True)
    both = [
        r
        for r in rows
        if r["n_vehicles"] >= MIN_VEHICLES and r["n_walkers"] >= 1 and np.isfinite(r["z0_walkers"])
    ]
    diff = np.array([r["z0_walkers"] - r["z0_vehicles"] for r in both])
    bias = float(np.median(diff)) if len(diff) else float("nan")
    per_town = {}
    for t in args.towns:
        d = np.array([r["z0_walkers"] - r["z0_vehicles"] for r in both if r["town"] == t]) - bias
        per_town[t] = {
            "n_frames": len(d),
            "median_abs_m": float(np.median(np.abs(d))) if len(d) else None,
            "p90_abs_m": float(np.percentile(np.abs(d), 90)) if len(d) else None,
            "median_signed_m": float(np.median(d)) if len(d) else None,
            "share_frames_with_walkers": float(
                np.mean([r["n_walkers"] >= 1 for r in rows if r["town"] == t])
            ),
        }
    passed = all(
        v["n_frames"] > 0
        and v["median_abs_m"] <= RULE["median_abs_m"]
        and v["p90_abs_m"] <= RULE["p90_abs_m"]
        for v in per_town.values()
    )
    out = {
        "towns": args.towns,
        "stride": args.stride,
        "ground_tags": list(GROUND_TAGS),
        "ring_m": RING_M,
        "min_px": MIN_PX,
        "rule": RULE,
        "bias_walkers_minus_vehicles_m": bias,
        "per_town_after_bias": per_town,
        "passed": passed,
        "frames": rows,
    }
    dest = REPO_ROOT / "results" / "syndrone" / "calibration" / "z0_walkers_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "frames"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
