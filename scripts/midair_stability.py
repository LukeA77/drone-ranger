"""Stability checks that set {{N_BOOT}} and {{N_MC}} ("large enough that
results are stable to the reported precision; show the check").

Usage: python scripts/midair_stability.py --split development

N_BOOT: the interval of the oracle-plane median absolute relative error, at n and 2n
group-bootstrap resamples, for increasing n; reports the endpoint change as a fraction of
the interval width. N_MC: per-sample Monte Carlo range sigma at n and 2n draws on a fixed
subsample, under the middle noise-sweep level; reports the median relative change.
Writes results/midair/<split>/stability.json.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from aerialrange.evaluate.midair_analysis import monte_carlo_sigma
from aerialrange.evaluate.stats import bootstrap_stability
from aerialrange.utils.config import REPO_ROOT, load_config

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from midair_common import load_split_cache, noise_levels  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="development", choices=["development"])
    ap.add_argument("--subsample", type=int, default=300)
    args = ap.parse_args(argv)
    seed = load_config("project").require("seed")
    r, names = load_split_cache(args.split)
    ok = r["oracle_plane.valid"]
    x = np.abs(r["oracle_plane.rel_range_err"][ok])
    g = r["group"][ok]

    boot = [bootstrap_stability(x, g, np.median, n, seed) for n in (250, 500, 1000, 2000)]

    level_name, sig = list(noise_levels().items())[1]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.nonzero(ok)[0], size=args.subsample, replace=False))
    mc = []
    for n in (250, 500, 1000, 2000):
        a = monte_carlo_sigma(r, idx, sig, "oracle_plane", n, np.random.default_rng(seed))
        b = monte_carlo_sigma(r, idx, sig, "oracle_plane", 2 * n, np.random.default_rng(seed + 1))
        rel = np.abs(a - b) / b
        mc.append(
            {
                "n_mc": n,
                "median_rel_change": float(np.nanmedian(rel)),
                "p95_rel_change": float(np.nanpercentile(rel, 95)),
            }
        )
        print(mc[-1], flush=True)

    out = {
        "split": args.split,
        "trajectories": names,
        "seed": seed,
        "n_boot_check": boot,
        "n_mc_check": {"noise_level": level_name, "subsample": len(idx), "rows": mc},
    }
    dest = REPO_ROOT / "results" / "midair" / args.split / "stability.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(boot, indent=1))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
