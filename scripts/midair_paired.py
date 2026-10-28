"""Paired comparison of the ranging methods on a common set of samples (D-620).

Usage: python scripts/midair_paired.py --split test

The per-method summaries (terrain_analyse.py) take each method's median over its own valid
samples, so methods are compared on different populations. This script compares them only
where they are all valid, on land: terrain from motion against flat ground with the true
height (the fair baseline; it needs a height a real drone does not have), and, separately,
all three methods including the take-off altimeter. It reports medians and 90th percentiles
with 95 % intervals from resampling whole flights, the paired median difference, the share of
samples where terrain from motion is closer, and each flight's median.

A re-analysis of saved outputs: no setting, threshold or model changes, so the one-time test
rule is not affected. Writes results/midair/<split>/terrain/paired.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import numpy as np

from aerialrange.data.midair import WATER_CLASS
from aerialrange.evaluate import stats as st
from aerialrange.utils.config import REPO_ROOT, load_config

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from terrain_analyse import load, methods  # noqa: E402


def p90(x: np.ndarray) -> float:
    return float(np.percentile(x, 90))


def summarise(
    err: dict[str, np.ndarray], g: np.ndarray, n_boot: int, rng: np.random.Generator
) -> dict:
    out = {}
    for name, a in err.items():
        out[name] = {
            "median": float(np.median(a)),
            "median_ci": st.group_bootstrap_ci(a, g, np.median, n_boot, rng),
            "p90": p90(a),
            "p90_ci": st.group_bootstrap_ci(a, g, p90, n_boot, rng),
            "per_flight_median": {str(int(f)): float(np.median(a[g == f])) for f in np.unique(g)},
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["development", "calibration", "test"])
    args = ap.parse_args(argv)
    seed = load_config("project").require("seed")
    n_boot = load_config("uncertainty").require("n_boot")
    rng = np.random.default_rng(seed)
    r, names = load(args.split)
    m = methods(r)
    land = r["cls"] != WATER_CLASS
    g = r["group"]

    pair = land & m["terrain_motion"]["valid"] & m["flat_true_height"]["valid"]
    t, f = np.abs(m["terrain_motion"]["rel"][pair]), np.abs(m["flat_true_height"]["rel"][pair])
    gp = g[pair]
    diff_ci = st.group_bootstrap_ci(
        np.column_stack([t, f]),
        gp,
        lambda x: float(np.median(x[:, 1]) - np.median(x[:, 0])),
        n_boot,
        rng,
    )
    flights_better = sum(
        float(np.median(t[gp == k]) < np.median(f[gp == k])) for k in np.unique(gp)
    )
    paired = {
        "n_samples": int(pair.sum()),
        "share_of_land_samples": float(pair.sum() / land.sum()),
        "methods": summarise({"terrain_motion": t, "flat_true_height": f}, gp, n_boot, rng),
        "median_difference_flat_minus_terrain": float(np.median(f) - np.median(t)),
        "median_difference_ci": diff_ci,
        "share_samples_terrain_closer": float(np.mean(t < f)),
        "flights_where_terrain_median_lower": int(flights_better),
        "n_flights": int(len(np.unique(gp))),
    }

    tri = pair & m["flat_altimeter"]["valid"]
    gt = g[tri]
    three = {
        "n_samples": int(tri.sum()),
        "share_of_land_samples": float(tri.sum() / land.sum()),
        "methods": summarise(
            {
                k: np.abs(m[k]["rel"][tri])
                for k in ("terrain_motion", "flat_true_height", "flat_altimeter")
            },
            gt,
            n_boot,
            rng,
        ),
    }
    rec = {
        "meta": {
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "split": args.split,
            "trajectories": names,
            "n_boot": n_boot,
            "seed": seed,
            "decision": "D-620",
            "samples": "land only; |relative range error|; a sample counts only where "
            "every compared method is valid",
        },
        "terrain_vs_flat_true_height": paired,
        "all_three_methods": three,
    }
    out = REPO_ROOT / "results" / "midair" / args.split / "terrain" / "paired.json"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    p = paired["methods"]
    print(
        f"common samples {paired['n_samples']} ({paired['share_of_land_samples']:.0%} of land): "
        f"terrain {p['terrain_motion']['median']:.3f} {p['terrain_motion']['median_ci']}, "
        f"p90 {p['terrain_motion']['p90']:.3f}; flat true height "
        f"{p['flat_true_height']['median']:.3f} {p['flat_true_height']['median_ci']}, "
        f"p90 {p['flat_true_height']['p90']:.3f}; "
        f"closer on {paired['share_samples_terrain_closer']:.0%}; "
        f"lower median on {flights_better}/{paired['n_flights']} flights"
    )
    tm = three["methods"]
    print(
        f"all three valid {three['n_samples']} ({three['share_of_land_samples']:.0%}): "
        + ", ".join(f"{k} {v['median']:.3f}" for k, v in tm.items())
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
