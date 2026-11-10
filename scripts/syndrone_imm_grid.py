"""Grid for the IMM settings that concern measurement and tracking errors (D-735).

Usage: python scripts/syndrone_imm_grid.py --source v2trk_accurate [--workers 4]

Motion settings per group come from results/syndrone/development/imm_fit.json
(scripts/syndrone_imm_fit.py); measurement errors are propagated (`kalman.meas_model:
propagated`, pixel noise of the source run's detector, exact SynDrone poses). The grid varies
`gate_prob`, `reset_after` and `q_stationary` (the same for both groups), replays each point
over the development sequences with scripts/syndrone_refilter.py's stage, and scores all points
together with the reference runs on identical rows (scripts/syndrone_speed_compare.py).
Selection rule (`--rule`): `pooled` (D-735 as first written) is the lowest mean over groups of
the 90th-percentile absolute speed error, ties by the mean distance of 90 % coverage from 0.90;
`strata` (D-735 revised, the default) scores five strata by true speed (people below and from
0.1 m/s; vehicles below 0.1, 0.1 to 3, from 3 m/s): a point qualifies only if no stratum's
90th-percentile error is worse than version 1's (`v2trk_accurate`), and the lowest mean of the
five wins (3 decimal places; ties by coverage as above); if none qualifies, the
constant-velocity filter with propagated errors (`v2trk_prop_cv`) is selected.
`--output conditional` reports the IMM's speed given whether the object moves (D-746).
Runs are written as cache/syndrone/runs/immgrid<suffix>_<k>/.
Output: results/syndrone/development/imm_grid.json.
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aerialrange.data.syndrone import syndrone_camera  # noqa: E402
from aerialrange.data.syndrone_archive import MemberStore  # noqa: E402
from aerialrange.pipeline.range_speed import RangeSpeedStage, replay  # noqa: E402
from aerialrange.utils.config import REPO_ROOT, load_config  # noqa: E402
from syndrone_refilter import flight_meta  # noqa: E402
from syndrone_speed_compare import compare  # noqa: E402

RUNS = REPO_ROOT / "cache" / "syndrone" / "runs"
GATE = (0.99, 0.999, 0.9999)
RESET = (2, 3, 5)
Q_STAT = (0.0, 0.001, 0.01, 0.1)
MEAS_SIGMA_B2_960 = {  # results/visdrone/val/pixel_noise_B2_yolo26n_960.json, overall per group
    "kappa_u": {"person": 0.10745237515098169, "vehicle": 0.03845991265795009},
    "kappa_v": {"person": 0.07897254548183567, "vehicle": 0.05318886219027205},
    "tilt_deg": 0.0,
    "h_m": 0.0,
    "yaw_deg": 0.0,
}
REFERENCES = ("v2trk_accurate", "v2trk_prop_cv")


def point_settings(fit: dict, gate: float, reset: int, qs: float, output: str = "mixture") -> dict:
    keys = (
        "q_cv",
        "q_ct",
        "q_turn",
        "turn_rate_sigma0",
        "switch_rates",
        "mode0",
        "velocity_sigma_moving",
    )
    return {
        g: {
            **{k: fit[g][k] for k in keys},
            "gate_prob": gate,
            "reset_after": reset,
            "q_stationary": qs,
            "output": output,
        }
        for g in fit
    }


def job(name: str, seq: str, source: str, imm: dict, meta_path: str) -> str:
    cfg = load_config("run_accurate")
    cam = syndrone_camera()
    ov: dict[str, dict] = {
        "camera": {"focal_px": cam.fx, "cx": cam.cx, "cy": cam.cy},
        "kalman": {
            "meas_model": "propagated",
            "meas_sigma": MEAS_SIGMA_B2_960,
            "motion_model": "imm",
            "imm": imm,
        },
    }

    def get(k: str) -> Any:
        base = cfg.require(k)
        return {**base, **ov.get(k, {})} if isinstance(base, dict) else ov.get(k, base)

    meta = pd.read_pickle(meta_path)
    tracks = pd.read_csv(RUNS / source / seq / "tracks.csv")
    new = replay(tracks, meta, RangeSpeedStage.from_config(get))
    dst = RUNS / name / seq
    dst.mkdir(parents=True, exist_ok=True)
    new.to_csv(dst / "tracks.csv", index=False)
    (dst / "refilter.json").write_text(json.dumps({"source": source, "overrides": ov}) + "\n")
    return f"{name}/{seq}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gate", type=float, nargs="+", default=GATE, help="extension only")
    ap.add_argument("--reset", type=int, nargs="+", default=RESET, help="extension only")
    ap.add_argument("--q-stat", type=float, nargs="+", default=Q_STAT, help="extension only")
    ap.add_argument("--suffix", default="", help="extension: run names and output suffix")
    ap.add_argument("--rule", choices=["pooled", "strata"], default="strata")
    ap.add_argument(
        "--output",
        choices=["mixture", "conditional"],
        default="mixture",
        help="the IMM's reported estimate (D-746)",
    )
    ap.add_argument(
        "--union-with", help="extension: an earlier grid's output; selection over the union"
    )
    args = ap.parse_args(argv)
    fit = json.loads((REPO_ROOT / "results/syndrone/development/imm_fit.json").read_text())[
        "groups"
    ]
    seqs = sorted(d.name for d in (RUNS / args.source).glob("*_h*"))
    dev = json.loads((REPO_ROOT / "results" / "splits.json").read_text("utf-8"))["syndrone"][
        "splits"
    ]["development"]
    seqs = [s for s in seqs if s.rsplit("_h", 1)[0] in dev]
    meta_dir = REPO_ROOT / "cache" / "syndrone" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    d = load_config("data")
    store = MemberStore(
        REPO_ROOT / d.require("data_root") / d.require("syndrone_subdir"),
        d.require("syndrone_base_url"),
    )
    for s in seqs:
        p = meta_dir / f"{s}.pkl"
        if not p.exists():
            town, h = s.rsplit("_h", 1)
            flight_meta(store, town, int(h)).to_pickle(p)
    points = list(itertools.product(args.gate, args.reset, args.q_stat))
    names = [f"immgrid{args.suffix}_{k:02d}" for k in range(len(points))]
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [
            ex.submit(
                job,
                n,
                s,
                args.source,
                point_settings(fit, *pt, args.output),
                str(meta_dir / f"{s}.pkl"),
            )
            for n, pt in zip(names, points, strict=True)
            for s in seqs
        ]
        for f in futs:
            print(f.result(), flush=True)
    pairs = pd.read_csv(RUNS / args.source / "speed_pairs_development.csv.gz")
    earlier = {}
    if args.union_with:
        prev = json.loads((REPO_ROOT / args.union_with).read_text())
        earlier = {n: {k: v for k, v in p.items() if k != "key"} for n, p in prev["points"].items()}
    all_points = {
        **earlier,
        **{
            n: {"gate_prob": pt[0], "reset_after": pt[1], "q_stationary": pt[2]}
            for n, pt in zip(names, points, strict=True)
        },
    }
    names = list(all_points)
    runs = [*REFERENCES, *names]
    res = compare(pairs, runs)

    def cov_gap(n: str) -> float:
        return sum(abs(res[g]["runs"][n]["coverage_0.90"] - 0.90) for g in res) / len(res)

    def strata_p90(n: str) -> list[float]:
        return [
            v["err_p90_abs_ms"]
            for g in sorted(res)
            for v in res[g]["runs"][n]["by_truth_speed"].values()
        ]

    def key(n: str) -> tuple[float, float]:
        if args.rule == "pooled":
            p90 = sum(res[g]["runs"][n]["err_p90_abs_ms"] for g in res) / len(res)
            return (round(p90, 3), cov_gap(n))
        return (round(float(np.mean(strata_p90(n))), 3), cov_gap(n))

    v1 = strata_p90("v2trk_accurate")
    qualifies = {
        n: all(a <= b for a, b in zip(strata_p90(n), v1, strict=True))
        for n in [*REFERENCES, *names]
    }
    pool = names if args.rule == "pooled" else [n for n in names if qualifies[n]]
    best = min(pool, key=key) if pool else "v2trk_prop_cv"
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-735",
        "source": args.source,
        "rule": args.rule,
        "output": args.output,
        "qualifies_no_stratum_worse_than_v1": qualifies,
        "union_with": args.union_with,
        "points": {n: {**p, "key": key(n)} for n, p in all_points.items()},
        "selected": best,
        "references": {n: key(n) for n in REFERENCES},
        "groups": res,
    }
    dest = REPO_ROOT / "results" / "syndrone" / "development" / f"imm_grid{args.suffix}.json"
    dest.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("selected", best, out["points"].get(best, "reference"))
    for n in [*REFERENCES, best]:
        print(n, key(n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
