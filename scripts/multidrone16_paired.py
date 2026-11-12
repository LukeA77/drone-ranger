"""Paired comparisons for hypotheses H13 to H16 from a Phase 16 stage's per-seed results (D-765).

Reads results/multi_drone/phase16_<stage>_seeds.json (written by multidrone16_run.py) and writes
results/multi_drone/phase16_paired_<stage>.json. Nothing is simulated again. The comparisons were
fixed before the test results were read (D-765). Each is "a minus b" on the same seeds, with a
95 % bootstrap interval over seeds:

  time  restricted mean time to a confirmed find (the horizon when none);
  crew  crew alarms before the confirmed find (or before the horizon);
  bytes bytes per drone per second.

A "cost" comparison is a difference of differences: (a - a0) - (b - b0) per seed, e.g. what
losing a drone costs the sweep minus what it costs the lawnmower.

  python scripts/multidrone16_paired.py test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multidrone_run as mr  # noqa: E402

N_BOOT = mr.N_BOOT
TEAMS = (1, 2, 4, 8)
BUDGETS = (415, 208, 104, 52)
LOSSES = ("0.25", "0.5", "0.75", "0.9")
RANGES = (250, 500, 1000, 2000)


def measures(rows: list[dict]) -> dict[str, np.ndarray]:
    h = rows[0]["horizon"]
    return {
        "time": np.array([r["t"] if r["found"] else h for r in rows], float),
        "crew": np.array([r["crew_before"] for r in rows], float),
        "bytes": np.array([r["bytes"] / max(r["steps"], 1) / r["n"] for r in rows], float),
    }


def interval(d: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(d)
    boots = [float(np.mean(d[rng.integers(0, n, n)])) for _ in range(N_BOOT)]
    lo, hi = (float(np.percentile(boots, q)) for q in (2.5, 97.5))
    return {"mean": float(np.mean(d)), "ci95": [lo, hi], "excludes_0": not lo <= 0 <= hi}


def comparisons(labels: set[str]) -> list[tuple[str, str, str, tuple[str, str] | None]]:
    """(name, a, b, (a0, b0) for a cost comparison, else None), kept to labels present."""
    out: list[tuple[str, str, str, tuple[str, str] | None]] = []
    for n in TEAMS:  # H13: sweep against the lawnmower; context: against the Phase 14 auction
        out.append((f"H13_sweep_minus_lawnmower_n{n}", f"sweep_n{n}", f"lawnmower_n{n}", None))
        out.append((f"sweep_minus_auction14_n{n}", f"sweep_n{n}", f"auction14_share_n{n}", None))
    out.append(("H13_rural_sweep_minus_lawnmower", "rural_sweep_n4", "rural_lawnmower_n4", None))
    # H14: compact sharing against Phase 14's encoding on the same planner; the sweep under
    # Phase 14's byte budgets, packet loss and range, against unlimited and perfect
    out.append(
        ("H14_auction_compact_minus_frozen", "auction14_compact_n4", "auction14_share_n4", None)
    )
    for b in BUDGETS:
        out.append((f"H14_sweep_budget{b}_minus_unlimited", f"sweep_budget{b}", "sweep_n4", None))
    for x in LOSSES:
        out.append((f"sweep_loss{x}_minus_perfect", f"sweep_loss{x}", "sweep_n4", None))
    for r in RANGES:
        out.append((f"sweep_range{r}_minus_unlimited", f"sweep_range{r}", "sweep_n4", None))
    # H15: what losing a drone at half a sweep costs the sweep, minus what it costs the lawnmower
    out.append(
        (
            "H15_lose_cost_sweep_minus_lawnmower",
            "sweep_lose_any",
            "lawnmower_lose_any",
            ("sweep_n4", "lawnmower_n4"),
        )
    )
    out.append(
        ("H15_sweep_lose_minus_lawnmower_lose", "sweep_lose_any", "lawnmower_lose_any", None)
    )
    out.append(("sweep_lose_minus_three_from_start", "sweep_lose_any", "sweep_n3_from_start", None))
    for plan, lost in (
        ("lawnmower", "lawnmower_lose_any"),
        ("sweep", "sweep_lose_any"),
        ("coord", "coord_lose_any"),
        ("coord", "coord_lose_coordinator"),
    ):
        out.append((f"{lost}_minus_no_loss", lost, f"{plan}_n4", None))
    # H16: coordinator against leaderless; losing the coordinator against losing any drone
    for n in TEAMS[1:]:
        out.append((f"H16_coord_minus_sweep_n{n}", f"coord_n{n}", f"sweep_n{n}", None))
    out.append(("H16_rural_coord_minus_sweep", "rural_coord_n4", "rural_sweep_n4", None))
    out.append(
        ("H16_lose_coordinator_minus_lose_any", "coord_lose_coordinator", "coord_lose_any", None)
    )
    out.append(("coord_lose_any_minus_sweep_lose_any", "coord_lose_any", "sweep_lose_any", None))
    keep = []
    for c in out:
        need = {c[1], c[2], *(c[3] or ())}
        if need <= labels:
            keep.append(c)
    return keep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("design3", "test"))
    args = ap.parse_args(argv)
    rows = json.loads((mr.OUT / f"phase16_{args.stage}_seeds.json").read_text())
    m = {lab: measures(r) for lab, r in rows.items()}
    seeds = {lab: [r["seed"] for r in rs] for lab, rs in rows.items()}
    rng = np.random.default_rng(0)
    out: dict[str, dict] = {}
    for name, a, b, base in comparisons(set(rows)):
        if seeds[a] != seeds[b] or (
            base and (seeds[base[0]] != seeds[a] or seeds[base[1]] != seeds[b])
        ):
            raise ValueError(f"{name}: seeds differ")
        res: dict[str, object] = {"a": a, "b": b}
        if base:
            res["a0"], res["b0"] = base
        for k in ("time", "crew", "bytes"):
            d = m[a][k] - m[b][k]
            if base:
                d = d - (m[base[0]][k] - m[base[1]][k])
            res[k] = interval(d, rng)
        out[name] = res
    doc = {"stage": args.stage, "n_boot": N_BOOT, "comparisons": out}
    dest = mr.OUT / f"phase16_paired_{args.stage}.json"
    dest.write_text(json.dumps(doc, indent=1) + "\n")
    for name, r in out.items():
        t, c = r["time"], r["crew"]
        print(
            f"{name:42s} time {t['mean']:+8.0f} ({t['ci95'][0]:+.0f} to {t['ci95'][1]:+.0f})  "
            f"crew {c['mean']:+8.0f} ({c['ci95'][0]:+.0f} to {c['ci95'][1]:+.0f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
