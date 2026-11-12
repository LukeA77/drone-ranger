"""How far Phase 16's planner comparison depends on the measured recurrence of false alarms
(D-766). Development seeds only; the test seeds are not run again.

The world places persistent look-alikes from the two-kind fit in
results/multi_drone/clutter_p10_v2_cal.json (share of false alarms 0.80, firing on 13 % of
passes). Its 95 % interval from resampling sites is wide, so the lawnmower, the Phase 14
auction and the frozen Phase 16 sweep are run with the fit's point estimate and with both ends
of the interval (the weakest and the strongest recurrence it allows), and with no recurrence
(Phase 14's world, every false alarm a one-off). The rate of false alarms on a first pass is
the measured 572 per km2 in every world; only how they recur changes.

  python scripts/multidrone16_sensitivity.py --workers 5

Writes results/multi_drone/phase16_sensitivity.json (and _seeds.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multidrone16_run as r16  # noqa: E402
import multidrone_run as mr  # noqa: E402

PLANNERS = ("lawnmower", "auction14", "sweep")


def worlds() -> dict[str, tuple[float, float]]:
    """World name -> (look-alike share of false alarms, fire probability per pass)."""
    cl = json.loads((mr.OUT / "clutter_p10_v2_cal.json").read_text())
    fit = cl["by_min_track_frames"]["1"]["recurrence"][f"{cl['radius_m']:g}"]["persistent"]
    ci = fit["ci95_resample_sites"]
    return {
        "none": (0.0, 0.0),
        "low": (float(ci["share"][0]), float(ci["fire_per_pass"][0])),
        "fit": (float(fit["share"]), float(fit["fire_per_pass"])),
        "high": (float(ci["share"][1]), float(ci["fire_per_pass"][1])),
    }


def job(a: tuple) -> dict:
    """multidrone16_run.job with the look-alike share and fire probability set by `c`."""
    st, c, seed = a
    orig = r16.camera_files

    def files(camera: str, m: int) -> tuple:
        sensor, _, _, radius = orig(camera, m)
        return sensor, c["share"], c["fire"], radius

    r16.camera_files = files
    try:
        return r16.job((st, c, seed))
    finally:
        r16.camera_files = orig


def configs(s16: dict, st: dict) -> list[tuple[str, dict]]:
    if not s16["frozen"]:
        raise ValueError("run after the phase16 freeze: the sweep is the frozen one")
    sweep = r16.sweep_pc(
        s16["spacing_factor"], s16["post"], s16["block_lanes"], s16["lost_after_s"]
    )
    pcs = {"lawnmower": {"kind": "lawnmower"}, "auction14": r16.PHASE14_AUCTION, "sweep": sweep}
    base = {"camera": "primary", "n": st["n_drones"], "k": s16["confirm_k"]}
    base["m"] = s16["min_track_frames"]
    out = []
    for w, (share, fire) in worlds().items():
        for p in PLANNERS:
            out.append((f"{p}_{w}", {**base, "pc": pcs[p], "share": share, "fire": fire}))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args(argv)
    st, s16 = mr.load_settings(), r16.load16()
    cfgs = configs(s16, st)
    seeds = mr.DEV_SEEDS
    jobs = [(st, c, s) for _, c in cfgs for s in seeds]
    print(f"sensitivity: {len(cfgs)} settings x {len(seeds)} seeds = {len(jobs)} searches")
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(job, jobs, chunksize=4))
    by = {
        lab: rows[i : i + len(seeds)]
        for (lab, _), i in zip(cfgs, range(0, len(rows), len(seeds)), strict=True)
    }
    rng = np.random.default_rng(0)
    res = {lab: r16.summary(r, rng) for lab, r in by.items()}
    paired = {}
    for w in worlds():
        for p in ("auction14", "sweep"):
            paired[f"{p}_{w}_minus_lawnmower"] = mr.paired(
                res[f"{p}_{w}"], res[f"lawnmower_{w}"], rng
            )
    for lab, x in res.items():
        ci = x["ci95"]["restricted_mean_time_s"]
        print(
            f"{lab:18s} confirmed {x['restricted_mean_time_s']:7.0f} s ({ci[0]:.0f} to {ci[1]:.0f})"
            f"  found {x['p_found_by_horizon']:.0%}  crew {x['crew_alarms_before_find_mean']:7.0f}",
            flush=True,
        )
    for k, d in paired.items():
        lo, hi = d["restricted_mean_time_diff_ci95"]
        print(f"{k:34s} {d['restricted_mean_time_diff_s']:+7.0f} s ({lo:+.0f} to {hi:+.0f})")
    doc = {
        "stage": "sensitivity",
        "seeds": [seeds[0], seeds[-1]],
        "worlds": {w: {"share": s, "fire_per_pass": f} for w, (s, f) in worlds().items()},
        "settings": {lab: c for lab, c in cfgs},
        "results": {
            lab: {
                **{k: v for k, v in x.items() if not k.startswith("_")},
                "first_detection": {k: v for k, v in x["first_detection"].items() if k != "_t"},
            }
            for lab, x in res.items()
        },
        "paired": paired,
    }
    (mr.OUT / "phase16_sensitivity.json").write_text(json.dumps(doc, indent=1) + "\n")
    (mr.OUT / "phase16_sensitivity_seeds.json").write_text(json.dumps(by) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
