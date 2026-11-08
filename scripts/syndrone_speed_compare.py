"""Speed accuracy of several replays of one SynDrone run, on identical rows (D-735).

Usage: python scripts/syndrone_speed_compare.py --pairs-from v2trk_accurate
       --runs v2trk_accurate v2trk_prop_cv ... [--out <name>]

All runs must share detection and tracking (scripts/syndrone_refilter.py replays of one
source), so their truth-track matches are identical; the matches come from
cache/syndrone/runs/<pairs-from>/speed_pairs_development.csv.gz (scripts/syndrone_evaluate.py).
Each run's speed and speed sigma are joined on (town, height, frame, track) and every run is
scored on the rows where all runs have a speed, so differences come from the filter alone.
Per group, pooled over development sequences: median and 90th-percentile absolute error,
coverage of the speed intervals at 68 / 90 / 95 % (normal intervals from the sigma), the share
of matched rows with a speed (before the common-row restriction), and the same by
line-of-sight depression band and by true speed (parked below 0.1 m/s, slow, moving).
Output: results/syndrone/development/speed_compare_<out>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from aerialrange.evaluate.syndrone_eval import interval_coverage
from aerialrange.utils.config import REPO_ROOT

RUNS = REPO_ROOT / "cache" / "syndrone" / "runs"
LEVELS = (0.68, 0.90, 0.95)
DEP_BANDS = ((0, 20), (20, 40), (40, 60), (60, 91))
# strata by true speed (m/s), D-735: parked, slow and moving; people have few rows above 3 m/s
SPEED_STRATA = {
    "person": ((0.0, 0.1), (0.1, np.inf)),
    "vehicle": ((0.0, 0.1), (0.1, 3.0), (3.0, np.inf)),
}
KEY = ["town", "height", "frame", "track_id"]


def run_speeds(name: str, towns: set[str] | None = None) -> pd.DataFrame:
    parts = []
    for d in sorted((RUNS / name).glob("*_h*")):
        town, h = d.name.rsplit("_h", 1)
        if towns is not None and town not in towns:
            continue
        t = pd.read_csv(
            d / "tracks.csv", usecols=["frame", "track_id", "speed_ms", "speed_sigma_ms"]
        )
        t["town"], t["height"] = town, int(h)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def metrics(err: np.ndarray, sig: np.ndarray) -> dict:
    if not len(err):
        return {"n": 0}
    a = np.abs(err)
    return {
        "n": int(len(err)),
        "err_median_abs_ms": float(np.median(a)),
        "err_p90_abs_ms": float(np.percentile(a, 90)),
        **{f"coverage_{k}": v for k, v in interval_coverage(err, sig, LEVELS).items()},
    }


def compare(pairs: pd.DataFrame, runs: list[str]) -> dict:
    df = pairs.copy()
    have = {}
    for r in runs:
        s = run_speeds(r).rename(columns={"speed_ms": f"sp_{r}", "speed_sigma_ms": f"sg_{r}"})
        df = df.merge(s, on=KEY, how="left")
        have[r] = np.isfinite(df[f"sp_{r}"]) & np.isfinite(df[f"sg_{r}"])
    common = np.logical_and.reduce(list(have.values()))
    out: dict = {}
    for g in sorted(df["group"].unique()):
        gm = (df["group"] == g).to_numpy()
        out[g] = {
            "matched_rows": int(gm.sum()),
            "common_rows": int((gm & common).sum()),
            "runs": {},
        }
        for r in runs:
            m = gm & common
            err = (df.loc[m, f"sp_{r}"] - df.loc[m, "truth_ms"]).to_numpy()
            res = metrics(err, df.loc[m, f"sg_{r}"].to_numpy())
            res["share_matched_with_speed"] = float(have[r][gm].mean())
            dep = df.loc[m, "depression_deg"].to_numpy()
            res["by_depression"] = {
                f"{a}-{b}": metrics(
                    err[(dep >= a) & (dep < b)],
                    df.loc[m, f"sg_{r}"].to_numpy()[(dep >= a) & (dep < b)],
                )
                for a, b in DEP_BANDS
            }
            ts = df.loc[m, "truth_ms"].to_numpy()
            res["by_truth_speed"] = {
                f"{a:g}-{b:g}": metrics(
                    err[(ts >= a) & (ts < b)], df.loc[m, f"sg_{r}"].to_numpy()[(ts >= a) & (ts < b)]
                )
                for a, b in SPEED_STRATA.get(g, ())
            }
            out[g]["runs"][r] = res
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-from", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="development", choices=["development", "calibration"])
    args = ap.parse_args(argv)
    pairs = pd.read_csv(RUNS / args.pairs_from / f"speed_pairs_{args.split}.csv.gz")
    res = compare(pairs, args.runs)
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-735",
        "pairs_from": args.pairs_from,
        "split": args.split,
        "runs": args.runs,
        "groups": res,
    }
    dest = REPO_ROOT / "results" / "syndrone" / args.split / f"speed_compare_{args.out}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for g, v in res.items():
        print(f"{g}: {v['common_rows']} common rows of {v['matched_rows']}")
        for r, m in v["runs"].items():
            cov = "/".join(f"{m.get(f'coverage_{lv:.2f}', np.nan):.3f}" for lv in LEVELS)
            print(
                f"  {r:28s} median {m.get('err_median_abs_ms', np.nan):.3f} "
                f"p90 {m.get('err_p90_abs_ms', np.nan):.3f} cov68/90/95 {cov} "
                f"with speed {m['share_matched_with_speed']:.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
