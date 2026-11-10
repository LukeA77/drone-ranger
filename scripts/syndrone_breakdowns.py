"""Breakdowns of a SynDrone evaluation by condition (build2 Phase 10, item 4).

Usage: python scripts/syndrone_breakdowns.py --name <run> [--split development]

Reads the matched rows that scripts/syndrone_evaluate.py writes for a run
(cache/syndrone/runs/<run>/range_matches_<split>.csv.gz and speed_matches_<split>.csv.gz) and
its result (results/syndrone/<split>/eval_<run>.json), and reports range and speed errors per
group (person, vehicle) broken down by:
- flight height (20, 50, 80 m);
- camera angle: the line of sight's depression below the horizon, in the bands of the
  configuration's range sigma table (configs/run.yaml `range_sigma_rel_by_depression`);
- true range, in the bands of the interval calibration (scripts/syndrone_interval_calibration.py);
- object size: the truth box height, in version 1's bins (D-317);
- class: person; car, truck and bus for vehicles (CARLA tags 100, 101, 102).
Each cell gives the row count, the median and 90th-percentile absolute relative range error,
the coverage of the calibrated 68 / 90 / 95 % range intervals (of the sigma-based intervals
where the run has no calibrated ones) and, always, of the sigma-based intervals; the median
absolute speed error, the coverage of the 90 % speed interval and the share of rows whose
speed is shown; with 95 % bootstrap intervals
that resample whole sequences (as the evaluation does). Detection AP by object size and by
height is copied from the evaluation; AP is not broken down by camera angle, because a
detection cannot be assigned an angle when it matches no truth box.
Writes results/syndrone/<split>/breakdowns_<run>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aerialrange.evaluate.syndrone_eval import GROUPS, SIZE_BINS_PX
from aerialrange.utils.config import REPO_ROOT, load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from syndrone_evaluate import N_BOOT, RUNS, TRUTH  # noqa: E402
from syndrone_interval_calibration import RANGE_EDGES  # noqa: E402

CLASS_OF_TAG = {40: "person", 100: "car", 101: "truck", 102: "bus"}
Z = {0.68: 0.994457883, 0.90: 1.644853627, 0.95: 1.959963985}  # two-sided normal quantiles
MIN_ROWS = 30  # cells with fewer matched rows are reported without statistics


def band_labels(edges: list[float]) -> list[str]:
    return [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:], strict=True)]


def bands(x: pd.Series, edges: list[float]) -> pd.Series:
    return pd.cut(x, edges, right=False, labels=band_labels(edges)).astype(str)


def tag_lookup(towns: list[str]) -> dict[tuple[str, int], int]:
    out: dict[tuple[str, int], int] = {}
    for t in towns:
        tr = pd.read_csv(TRUTH / t / "objects.csv.gz", usecols=["id", "tag"]).drop_duplicates()
        out.update({(t, int(i)): int(g) for i, g in zip(tr["id"], tr["tag"], strict=True)})
    return out


RANGE_COLS = [
    "truth_range_m",
    "pred_range_m",
    "range_sigma_m",
    "half_68_m",
    "half_90_m",
    "half_95_m",
]
SPEED_COLS = ["truth_ms", "pred_ms", "sigma_ms", "half_90_ms", "shown"]


def range_stats(a: np.ndarray) -> dict:
    """a: rows of RANGE_COLS."""
    truth, pred, sig, halves = a[:, 0], a[:, 1], a[:, 2], a[:, 3:6]
    rel = (pred - truth) / truth
    err = np.abs(pred - truth)
    out = {
        "range_rel_err_median_abs": float(np.median(np.abs(rel))),
        "range_rel_err_p90_abs": float(np.percentile(np.abs(rel), 90)),
        "range_rel_err_median_signed": float(np.median(rel)),
    }
    # coverage of the sigma-based intervals, always: under added navigation error the calibrated
    # intervals (fitted with exact poses) are not expected to hold, the propagated sigma may
    for lv, z in Z.items():
        out[f"range_sigma_coverage_{lv:.2f}"] = float(np.mean(err <= z * sig))
    ok = np.isfinite(halves).all(axis=1)
    if ok.any():
        for i, lv in enumerate(("0.68", "0.90", "0.95")):
            out[f"range_coverage_{lv}"] = float(np.mean(err[ok] <= halves[ok, i]))
        out["range_intervals"] = "calibrated"
    else:
        for lv, z in Z.items():
            out[f"range_coverage_{lv:.2f}"] = float(np.mean(err <= z * sig))
        out["range_intervals"] = "sigma"
    return out


def speed_stats(a: np.ndarray) -> dict:
    """a: rows of SPEED_COLS."""
    err = np.abs(a[:, 1] - a[:, 0])
    h90 = a[:, 3]
    ok = np.isfinite(h90)
    if ok.any():
        cov = float(np.mean(err[ok] <= h90[ok]))
    else:
        cov = float(np.mean(err <= Z[0.90] * a[:, 2]))
    return {
        "speed_err_median_abs_ms": float(np.median(err)),
        "speed_coverage_0.90": cov,
        "speed_shown_share": float(np.mean(a[:, 4] > 0)),
    }


BOOT_KEYS = {
    "range": (
        "range_rel_err_median_abs",
        "range_rel_err_p90_abs",
        "range_coverage_0.90",
        "range_sigma_coverage_0.90",
    ),
    "speed": ("speed_err_median_abs_ms", "speed_coverage_0.90", "speed_shown_share"),
}


def cell(df: pd.DataFrame, kind: str, rng_seed: int = 0) -> dict:
    """Statistics of one cell, with bootstrap intervals over whole sequences."""
    stats, cols = (range_stats, RANGE_COLS) if kind == "range" else (speed_stats, SPEED_COLS)
    out: dict = {"n": len(df), "n_sequences": int(df["seq"].nunique())}
    if len(df) < MIN_ROWS:
        return out
    out.update(stats(df[cols].to_numpy(float)))
    groups = [g[cols].to_numpy(float) for _, g in df.groupby("seq")]
    if len(groups) < 2:
        return out
    rng = np.random.default_rng(rng_seed)
    vals: dict[str, list[float]] = {k: [] for k in BOOT_KEYS[kind]}
    for _ in range(N_BOOT):
        pick = np.vstack([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        s = stats(pick)
        for k in vals:
            vals[k].append(s[k])
    out["ci95_resample_sequences"] = {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="development")
    args = ap.parse_args(argv)
    run = RUNS / args.name
    ev_path = REPO_ROOT / "results" / "syndrone" / args.split / f"eval_{args.name}.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    rng_df = pd.read_csv(run / f"range_matches_{args.split}.csv.gz")
    spd_df = pd.read_csv(run / f"speed_matches_{args.split}.csv.gz")
    tags = tag_lookup(sorted(set(rng_df["town"]) | set(spd_df["town"])))
    dep_edges = list(load_config("run").require("range_sigma_rel_by_depression")["edges_deg"])
    size_edges = [float(x) for x in SIZE_BINS_PX]
    range_edges = [float(x) for x in RANGE_EDGES]
    for df in (rng_df, spd_df):
        df["seq"] = df["town"] + "_h" + df["height"].astype(int).astype(str)
        df["height_band"] = "h" + df["height"].astype(int).astype(str)
        df["depression_band"] = bands(df["depression_deg"], dep_edges)
        df["class"] = [
            CLASS_OF_TAG.get(tags.get((t, int(i))), "other")
            for t, i in zip(df["town"], df["truth_id"], strict=True)
        ]
    rng_df["range_band"] = bands(rng_df["truth_range_m"], range_edges)
    rng_df["size_band"] = bands(rng_df["mask_height_px"], size_edges)
    dims = {
        "height": "height_band",
        "camera_angle_deg": "depression_band",
        "true_range_m": "range_band",
        "object_size_px": "size_band",
        "class": "class",
    }
    result: dict = {
        "name": args.name,
        "split": args.split,
        "sequences": ev["sequences"],
        "n_boot": N_BOOT,
        "min_rows_per_cell": MIN_ROWS,
        "bands": {
            "camera_angle_deg": dep_edges,
            "true_range_m": range_edges,
            "object_size_px": size_edges,
        },
        "notes": {
            "camera_angle_deg": "depression of the line of sight below the horizon",
            "object_size_px": "truth box height (instance mask); speed rows are not broken down "
            "by size, as speed_matches has no box size",
            "true_range_m": "speed rows are not broken down by true range (no range column)",
        },
    }
    for g in GROUPS:
        rg, sg = rng_df[rng_df["group"] == g], spd_df[spd_df["group"] == g]
        res: dict = {
            "overall": {"range": cell(rg, "range"), "speed": cell(sg, "speed")},
            "detection_ap50_by_size_px": {
                k.removeprefix("ap50_"): v
                for k, v in ev[g]["pooled"].items()
                if k.startswith("ap50_") and k != "ap50_all"
            },
            "detection_ap50_by_height": {
                h: v.get("ap50_all") for h, v in ev[g].get("by_height", {}).items()
            },
        }
        for dim, col in dims.items():
            res[dim] = {}
            for b in sorted(set(rg.get(col, pd.Series(dtype=str))) | set(sg.get(col, []))):
                if b in ("nan", "other"):
                    continue
                entry = {}
                if col in rg:
                    entry["range"] = cell(rg[rg[col] == b], "range")
                if col in sg:
                    entry["speed"] = cell(sg[sg[col] == b], "speed")
                res[dim][b] = entry
        result[g] = res
    out = REPO_ROOT / "results" / "syndrone" / args.split / f"breakdowns_{args.name}.json"
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for g in GROUPS:
        print(f"== {g}")
        for dim in dims:
            for b, e in result[g][dim].items():
                r = e.get("range", {})
                if "range_rel_err_median_abs" in r:
                    print(
                        f"  {dim:17s} {b:9s} n={r['n']:6d}"
                        f" median {r['range_rel_err_median_abs']:.3f}"
                        f" p90 {r['range_rel_err_p90_abs']:.3f}"
                        f" cov90 {r['range_coverage_0.90']:.3f}"
                    )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
