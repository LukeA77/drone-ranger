"""Choose and fit the calibration of the range and speed intervals (D-737).

Usage: python scripts/syndrone_interval_calibration.py --run <name> [--min-cell 30]

Reads, for the development and calibration towns, the matched rows of one pipeline run
(scripts/syndrone_evaluate.py): range_matches_<split>.csv.gz (range against the footprint
centre) and speed_matches_<split>.csv.gz. The score of a row is z = |error| / sigma. Groups:
line-of-sight depression band, predicted range band, class, and flight (town and camera height).

Methods (each gives, per nominal level, a multiplier q so the interval is +- q sigma):
- `v1_scalar`: one factor by maximum likelihood under a Gaussian, q = z_level x sqrt(mean z^2);
- `conformal`: one split-conformal quantile of z per level;
- `mondrian`: one quantile per cell (depression x range x class); cells with fewer rows than
  ceil(1 / (1 - level)) use the pooled quantile;
- `mondrian_flights`: per cell, each flight's quantile, then the level-quantile across flights.
Chosen by leave-one-flight-out: lowest mean absolute coverage error over (held-out flight, cell
with at least `--min-cell` rows, level); ties by mean interval width. The chosen method is then
fitted on all 15 flights. Output: results/syndrone/calibration/interval_calibration_<run>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

from aerialrange.utils.config import REPO_ROOT

RUNS = REPO_ROOT / "cache" / "syndrone" / "runs"
LEVELS = (0.68, 0.90, 0.95)
DEP_EDGES = (0, 20, 40, 60, 91)
RANGE_EDGES = (0, 50, 100, 200, np.inf)
METHODS = ("v1_scalar", "conformal", "mondrian", "mondrian_flights")


def conformal_q(z: np.ndarray, level: float) -> float:
    """Split-conformal quantile: the ceil((n + 1) level)-th smallest score (inf if too few)."""
    n = len(z)
    k = math.ceil((n + 1) * level)
    if n == 0 or k > n:
        return float("inf")
    return float(np.sort(z)[k - 1])


def load(run: str) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {"range": [], "speed": []}
    for split in ("development", "calibration"):
        r = pd.read_csv(RUNS / run / f"range_matches_{split}.csv.gz")
        r = r[np.isfinite(r["range_sigma_m"]) & (r["range_sigma_m"] > 0)]
        parts["range"].append(
            pd.DataFrame(
                {
                    "z": np.abs(r["pred_range_m"] - r["truth_range_m"]) / r["range_sigma_m"],
                    "dep": r["depression_deg"],
                    "rng": r["pred_range_m"],
                    "group": r["group"],
                    "flight": r["town"] + "_h" + r["height"].astype(str),
                }
            )
        )
        s = pd.read_csv(RUNS / run / f"speed_matches_{split}.csv.gz")
        s = s[np.isfinite(s["sigma_ms"]) & (s["sigma_ms"] > 0) & np.isfinite(s["pred_ms"])]
        parts["speed"].append(
            pd.DataFrame(
                {
                    "z": np.abs(s["pred_ms"] - s["truth_ms"]) / s["sigma_ms"],
                    "dep": s["depression_deg"],
                    "rng": np.nan,  # speed rows carry no range; range band unused for speed
                    "group": s["group"],
                    "flight": s["town"] + "_h" + s["height"].astype(str),
                }
            )
        )
    out = {}
    for k, v in parts.items():
        df = pd.concat(v, ignore_index=True)
        df["dep_band"] = pd.cut(df["dep"], DEP_EDGES, right=False, labels=False)
        df["rng_band"] = (
            pd.cut(df["rng"], RANGE_EDGES, right=False, labels=False) if k == "range" else 0
        )
        df["cell"] = (
            df["group"]
            + "|d"
            + df["dep_band"].astype("Int64").astype(str)
            + "|r"
            + df["rng_band"].astype("Int64").astype(str)
        )
        out[k] = df.dropna(subset=["dep_band"])
    return out


def fit(df: pd.DataFrame, method: str) -> dict:
    """Multipliers per level: {'pooled': {level: q}, 'cells': {cell: {level: q}}}."""
    z = df["z"].to_numpy()
    if method == "v1_scalar":
        s = float(np.sqrt(np.mean(z**2)))
        return {"pooled": {lv: float(norm.ppf(0.5 + lv / 2)) * s for lv in LEVELS}, "cells": {}}
    pooled = {lv: conformal_q(z, lv) for lv in LEVELS}
    if method == "conformal":
        return {"pooled": pooled, "cells": {}}
    cells: dict[str, dict[float, float]] = {}
    for cell, g in df.groupby("cell"):
        qs = {}
        for lv in LEVELS:
            need = math.ceil(1 / (1 - lv))
            if method == "mondrian":
                qs[lv] = conformal_q(g["z"].to_numpy(), lv) if len(g) >= need else pooled[lv]
            else:  # mondrian_flights
                per = [
                    conformal_q(f["z"].to_numpy(), lv)
                    for _, f in g.groupby("flight")
                    if len(f) >= need
                ]
                per = [q for q in per if np.isfinite(q)]
                qs[lv] = float(np.quantile(per, lv)) if per else pooled[lv]
        cells[str(cell)] = qs
    return {"pooled": pooled, "cells": cells}


def multipliers(model: dict, df: pd.DataFrame, lv: float) -> np.ndarray:
    pooled = model["pooled"][lv]
    return np.array([model["cells"].get(c, {}).get(lv, pooled) for c in df["cell"]])


def loo(df: pd.DataFrame, method: str, min_cell: int) -> dict:
    errs, widths, under = [], [], []
    per_level = {lv: [] for lv in LEVELS}
    for fl in sorted(df["flight"].unique()):
        tr, te = df[df["flight"] != fl], df[df["flight"] == fl]
        model = fit(tr, method)
        for lv in LEVELS:
            q = multipliers(model, te, lv)
            cov = te["z"].to_numpy() <= q
            for _, idx in te.groupby("cell").groups.items():
                if len(idx) < min_cell:
                    continue
                m = te.index.get_indexer(idx)
                c = float(cov[m].mean())
                errs.append(abs(c - lv))
                under.append(c < lv - 0.05)
                per_level[lv].append(c)
            widths.append(float(np.mean(q[np.isfinite(q)])) if np.isfinite(q).any() else np.inf)
    return {
        "mean_abs_coverage_error": float(np.mean(errs)) if errs else float("nan"),
        "share_more_than_5_points_under": float(np.mean(under)) if under else float("nan"),
        "mean_multiplier": float(np.mean(widths)),
        "entries": len(errs),
        "median_cell_coverage_by_level": {
            f"{lv:.2f}": float(np.median(v)) if v else None for lv, v in per_level.items()
        },
    }


def coverage_report(df: pd.DataFrame, model: dict) -> dict:
    out: dict = {"pooled": {}, "by_group": {}}
    for lv in LEVELS:
        q = multipliers(model, df, lv)
        cov = df["z"].to_numpy() <= q
        out["pooled"][f"{lv:.2f}"] = float(cov.mean())
        for key in ("group", "dep_band", "rng_band", "flight"):
            for val, idx in df.groupby(key).groups.items():
                m = df.index.get_indexer(idx)
                out["by_group"].setdefault(f"{key}={val}", {})[f"{lv:.2f}"] = float(cov[m].mean())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-cell", type=int, default=30)
    args = ap.parse_args(argv)
    data = load(args.run)
    res: dict = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-737",
        "run": args.run,
        "levels": list(LEVELS),
        "dep_edges": list(DEP_EDGES),
        "range_edges": [float(x) for x in RANGE_EDGES],
        "quantities": {},
    }
    for qty, df in data.items():
        df = df.reset_index(drop=True)
        scores = {m: loo(df, m, args.min_cell) for m in METHODS}
        chosen = min(
            METHODS,
            key=lambda m: (
                round(scores[m]["mean_abs_coverage_error"], 4),
                scores[m]["mean_multiplier"],
            ),
        )
        model = fit(df, chosen)
        res["quantities"][qty] = {
            "rows": int(len(df)),
            "flights": int(df["flight"].nunique()),
            "leave_one_flight_out": scores,
            "chosen": chosen,
            "uncalibrated_coverage": coverage_report(
                df, {"pooled": dict.fromkeys(LEVELS, 0.0), "cells": {}}
            )
            | {
                "pooled_raw": {
                    f"{lv:.2f}": float(np.mean(df["z"].to_numpy() <= norm.ppf(0.5 + lv / 2)))
                    for lv in LEVELS
                }
            },
            "fitted": {
                "pooled": {f"{k:.2f}": v for k, v in model["pooled"].items()},
                "cells": {
                    c: {f"{k:.2f}": v for k, v in q.items()} for c, q in model["cells"].items()
                },
            },
            "in_sample_coverage": coverage_report(df, model),
        }
        print(qty, "chosen", chosen)
        for m in METHODS:
            s = scores[m]
            print(
                f"  {m:17s} mean |cov err| {s['mean_abs_coverage_error']:.4f} "
                f">5pt under {s['share_more_than_5_points_under']:.3f} "
                f"mult {s['mean_multiplier']:.2f} n {s['entries']}"
            )
    dest = (
        REPO_ROOT / "results" / "syndrone" / "calibration" / f"interval_calibration_{args.run}.json"
    )
    dest.write_text(json.dumps(res, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
