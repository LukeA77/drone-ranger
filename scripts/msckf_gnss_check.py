"""GPS error per flight against ground truth, for diagnosing a filter result (D-741).

Usage: python scripts/msckf_gnss_check.py --split test

Per flight (1 Hz fixes, NED): mean error, standard deviation about that mean, the largest
deviation from it, and the fixes more than 10 m from the flight's median on any axis. Uses
the same error definition as `msckf_gnss_fit.py`. Writes
results/midair/<split>/navigation/gnss_check.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import h5py
import numpy as np

from aerialrange.utils.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from midair_common import midair_root, split_names  # noqa: E402
from msckf_gnss_fit import errors  # noqa: E402

JUMP_M = 10.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="test", choices=("development", "test"))
    args = ap.parse_args()
    flights = {}
    with h5py.File(midair_root() / "sensor_records.hdf5", "r") as f:
        for name in split_names(args.split):
            e, _, _ = errors(f, name)
            dev = e - e.mean(0)
            jumps = np.where((np.abs(e - np.median(e, 0)) > JUMP_M).any(1))[0]
            flights[name] = {
                "fixes": len(e),
                "mean_m": e.mean(0).tolist(),
                "sd_m": dev.std(0).tolist(),
                "max_abs_deviation_m": np.abs(dev).max(0).tolist(),
                "fixes_over_10m_from_median": jumps.tolist(),
            }
            print(f"{name}: max deviation {np.round(np.abs(dev).max(0), 1)} m, jumps {jumps}")
    dest = REPO_ROOT / "results" / "midair" / args.split / "navigation"
    dest.mkdir(parents=True, exist_ok=True)
    res = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-741",
        "split": args.split,
        "jump_threshold_m": JUMP_M,
        "flights": flights,
    }
    (dest / "gnss_check.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
