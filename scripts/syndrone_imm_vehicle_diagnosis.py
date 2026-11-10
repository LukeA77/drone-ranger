"""Where the vehicle IMM loses to the constant-velocity filter (D-746, open since D-735).

Usage: python scripts/syndrone_imm_vehicle_diagnosis.py

On the development towns, identical rows (replays of v2trk_accurate): the constant-velocity
filter with propagated errors (v2trk_prop_cv) against the 30 IMM points of the corrected grid
(immgrid_fixveh_*); for the IMM point with the lowest 90th-percentile error on vehicles from
3 m/s, signed errors by track age, range and camera height, the share of rows reading below
half the true speed, and the direction of the rows where the IMM is more than 1 m/s worse.
Prints only (the numbers are recorded in D-746).
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from syndrone_speed_compare import RUNS, run_speeds  # noqa: E402

pairs = pd.read_csv(RUNS / "v2trk_accurate" / "speed_pairs_development.csv.gz")
pairs = pairs[pairs["group"] == "vehicle"]
cv = run_speeds("v2trk_prop_cv").rename(columns={"speed_ms": "cv", "speed_sigma_ms": "cv_s"})
key = ["town", "height", "frame", "track_id"]
best = None
for i in range(30):
    name = f"immgrid_fixveh_{i:02d}"
    im = run_speeds(name).rename(columns={"speed_ms": "imm", "speed_sigma_ms": "imm_s"})
    m = pairs.merge(cv, on=key).merge(im, on=key).dropna(subset=["cv", "imm"])
    mv = m[m["truth_ms"] >= 3.0]
    p90 = np.percentile(np.abs(mv["imm"] - mv["truth_ms"]), 90)
    if best is None or p90 < best[0]:
        best = (p90, name, m)
p90, name, m = best
print("best moving-vehicle point", name, round(p90, 2))
tr = pd.concat(
    [
        pd.read_csv(d / "tracks.csv", usecols=["frame", "track_id", "track_age", "range_m"]).assign(
            town=d.name.rsplit("_h", 1)[0], height=int(d.name.rsplit("_h", 1)[1])
        )
        for d in sorted((RUNS / name).glob("*_h*"))
    ]
)
m = m.merge(tr, on=key)
mv = m[m["truth_ms"] >= 3.0].copy()
for f in ("cv", "imm"):
    mv[f"{f}_err"] = mv[f] - mv["truth_ms"]
print("moving rows", len(mv))
for f in ("cv", "imm"):
    e = mv[f"{f}_err"]
    print(
        f,
        "median signed",
        round(e.median(), 2),
        "p10",
        round(e.quantile(0.1), 2),
        "p90",
        round(e.quantile(0.9), 2),
        "p90 abs",
        round(e.abs().quantile(0.9), 2),
        "share below half the true speed",
        round((mv[f] < 0.5 * mv["truth_ms"]).mean(), 3),
    )
mv["age_bin"] = pd.cut(mv["track_age"], [0, 5, 10, 25, 50, 100, 10_000])
mv["range_bin"] = pd.cut(mv["range_m"], [0, 50, 100, 150, 250, 10_000])
for b in ("age_bin", "range_bin", "height"):
    g = mv.groupby(b, observed=True)
    print(
        pd.DataFrame(
            {
                "n": g.size(),
                "cv_med": g["cv_err"].median().round(2),
                "imm_med": g["imm_err"].median().round(2),
                "cv_p90abs": g["cv_err"].apply(lambda s: s.abs().quantile(0.9)).round(2),
                "imm_p90abs": g["imm_err"].apply(lambda s: s.abs().quantile(0.9)).round(2),
                "imm_collapsed": g.apply(lambda x: (x["imm"] < 0.5 * x["truth_ms"]).mean()).round(
                    3
                ),
            }
        )
    )
# where IMM is far worse than CV: is it reading slow (pulled toward zero) or fast?
worse = mv[(mv["imm_err"].abs() - mv["cv_err"].abs()) > 1.0]
print(
    "rows where IMM is >1 m/s worse:",
    len(worse),
    "of which IMM slower than truth:",
    round((worse["imm_err"] < 0).mean(), 3),
)
