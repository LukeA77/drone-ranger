"""Walker-based base height z0 for every frame of a town that has no vehicle fit.

Usage: python scripts/syndrone_z0_walker_series.py --town Town02 [--workers 6]

Only for a town where no frame can be fitted from vehicles (Town02, D-720), and only after
the walker fit passed its check on development and calibration towns
(results/syndrone/calibration/z0_walkers_check.json, scripts/syndrone_z0_walkers.py). Writes
cache/syndrone/truth/<town>/z0_walkers.csv (frame, z0 from walkers with the checked bias
removed, its standard error, walkers used), which scripts/syndrone_z0_check.py then uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from aerialrange.utils.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from syndrone_z0_walkers import frame_pair  # noqa: E402

CHECK = REPO_ROOT / "results" / "syndrone" / "calibration" / "z0_walkers_check.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    check = json.loads(CHECK.read_text())
    if not check["passed"]:
        print("the walker fit did not pass its check; not used")
        return 1
    bias = float(check["bias_walkers_minus_vehicles_m"])
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(frame_pair, [(args.town, k) for k in range(3000)], chunksize=8))
    df = pd.DataFrame(rows)
    if (df["n_vehicles"] >= 3).any():
        print(f"{args.town} has vehicle-fitted frames; the vehicle fit applies (D-704)")
        return 1
    out = pd.DataFrame(
        {
            "frame": df["frame"],
            "z0_walkers": df["z0_walkers"] - bias,
            "se_walkers": df["se_walkers"],
            "n_walkers": df["n_walkers"],
        }
    )
    dest = REPO_ROOT / "cache" / "syndrone" / "truth" / args.town / "z0_walkers.csv"
    out.to_csv(dest, index=False)
    print(f"{args.town}: {int((out['n_walkers'] > 0).sum())} of 3000 frames have walkers; {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
