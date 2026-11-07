"""Paired comparison of two SynDrone evaluations of the same sequences.

Usage: python scripts/syndrone_compare.py --base v2trk_accurate --new ft_loto_accurate
       [--split development]

Reads results/syndrone/<split>/eval_<name>.json (scripts/syndrone_evaluate.py) for both runs
and writes, per group (person, vehicle) and metric, the per-sequence change (new minus base),
how many sequences improved, and the per-town mean change, to
results/syndrone/<split>/compare_<base>_vs_<new>.json. The runs must cover the same
sequences.
"""

from __future__ import annotations

import argparse
import json
import sys

from aerialrange.utils.config import REPO_ROOT

METRICS = ("ap50_all", "idf1", "hota_HOTA", "mota")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--split", default="development")
    args = ap.parse_args(argv)
    d = REPO_ROOT / "results" / "syndrone" / args.split
    base = json.loads((d / f"eval_{args.base}.json").read_text(encoding="utf-8"))
    new = json.loads((d / f"eval_{args.new}.json").read_text(encoding="utf-8"))
    out: dict = {"base": args.base, "new": args.new, "split": args.split}
    for g in ("person", "vehicle"):
        a, b = base[g]["per_sequence"], new[g]["per_sequence"]
        if set(a) != set(b):
            print(f"{g}: the runs cover different sequences")
            return 1
        res = {}
        for m in METRICS:
            delta = {s: b[s][m] - a[s][m] for s in sorted(a)}
            towns: dict[str, list[float]] = {}
            for s, v in delta.items():
                towns.setdefault(s.rsplit("_h", 1)[0], []).append(v)
            res[m] = {
                "per_sequence": delta,
                "n_improved": sum(v > 0 for v in delta.values()),
                "n_sequences": len(delta),
                "town_mean": {t: sum(v) / len(v) for t, v in towns.items()},
            }
        out[g] = res
    path = d / f"compare_{args.base}_vs_{args.new}.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for g in ("person", "vehicle"):
        r = out[g]["ap50_all"]
        print(g, "AP50 improved in", r["n_improved"], "of", r["n_sequences"], r["town_mean"])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
