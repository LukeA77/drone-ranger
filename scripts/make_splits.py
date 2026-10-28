"""Write results/splits.json: the fixed Mid-Air trajectory split.

Uses the trajectory names in the climate's sensor_records.hdf5, the proportions in
configs/data.yaml (midair_split) and the project seed. Refuses to overwrite a different
existing split unless --force is given, so the split cannot drift after tuning has started.
"""

from __future__ import annotations

import argparse
import json
import sys

from aerialrange.data.midair import list_trajectories
from aerialrange.data.splits import split_trajectories
from aerialrange.utils.config import REPO_ROOT, load_config

OUT = REPO_ROOT / "results" / "splits.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    data = load_config("data")
    seed = load_config("project").require("seed")
    climate = data.require("midair_climate")
    h5 = (
        REPO_ROOT
        / data.require("data_root")
        / data.require("midair_subdir")
        / climate
        / "sensor_records.hdf5"
    )
    names = list_trajectories(h5)
    props = data.require("midair_split")
    split = {
        "midair": {
            "seed": seed,
            "proportions": props,
            "source_climate": climate,
            "n_trajectories": len(names),
            "splits": split_trajectories(names, props, seed),
        }
    }
    if OUT.exists() and not args.force:
        old = json.loads(OUT.read_text(encoding="utf-8"))
        if old != split:
            print(f"{OUT} exists and differs; refusing to overwrite (use --force).")
            return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(split, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
