"""Add the SynDrone town-level split to results/splits.json (key `syndrone`).

Whole towns, stratified by town type (configs/data.yaml `syndrone_strata`), with the project
seed. The Mid-Air entry is left untouched. Refuses to change an existing `syndrone` entry
unless --force is given, so the split cannot drift once results exist (D-701).
"""

from __future__ import annotations

import argparse
import json
import sys

from aerialrange.data.splits import split_stratified
from aerialrange.utils.config import REPO_ROOT, load_config

OUT = REPO_ROOT / "results" / "splits.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    data = load_config("data")
    seed = load_config("project").require("seed")
    strata = data.require("syndrone_strata")
    quota = data.require("syndrone_split_per_stratum")
    towns = data.require("syndrone_towns")
    if sorted(t for v in strata.values() for t in v) != sorted(towns):
        print("syndrone_strata does not cover syndrone_towns exactly")
        return 1
    entry = {
        "seed": seed,
        "unit": "town",
        "strata": strata,
        "per_stratum": quota,
        "rest": "calibration",
        "splits": split_stratified(strata, quota, rest="calibration", seed=seed),
        "official_split_not_used": "official SynDrone split is by frame within each town "
        "(frames 0-99, 600-699, ...), so it is not a town-level split (D-701)",
    }
    allsplits = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    old = allsplits.get("syndrone")
    if old is not None and old != entry and not args.force:
        print("results/splits.json already has a different 'syndrone' entry; refusing (--force)")
        return 1
    allsplits["syndrone"] = entry
    OUT.write_text(json.dumps(allsplits, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(entry["splits"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
