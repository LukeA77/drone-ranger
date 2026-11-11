"""Phase 14 follow-up on development seeds only (D-756): the byte budget without head-of-line
blocking.

Found after the one-time test: in the frozen radio (search/comms.py) a drone's unused byte
allowance is capped at one second's worth, and observations are sent oldest first. An
observation larger than one second's allowance can then never be sent and blocks every later
one, so under budgets below about one observation per second only the auction's claims get
through. The test's budget rows therefore measure that fault, not graceful degradation.

This script runs the chosen configuration (frozen settings, primary sensor, 4 drones) on
development seeds 0..199 with two allowance rules:
  - "capped": the frozen rule (allowance capped at one second's budget);
  - "fill_to_fit": the allowance keeps building until the next message fits (its cap is the
    larger of one second's budget and the size of the next message), so the average rate still
    respects the budget and no message is blocked forever. No new parameter.
The frozen code is not changed: the alternative rule is patched in here (D-755).

Usage: python scripts/multidrone_budget_check.py --workers 12
Writes results/multi_drone/budget_check_p10_v2_cal.json (and _seeds.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multidrone_run as mr  # noqa: E402
from aerialrange.search import comms as cm  # noqa: E402
from aerialrange.search import sim  # noqa: E402

SENSOR = mr.REPO_ROOT / "results" / "multi_drone" / "sensor_p10_v2_cal.json"
_FROZEN_EXCHANGE = cm.exchange


def exchange_fill_to_fit(
    nodes: list[cm.Node],
    pos_ne: np.ndarray,
    radio: cm.Radio,
    dt_s: float,
    rng: np.random.Generator,
) -> list[list[cm.Observation]]:
    """As comms.exchange, but a drone's allowance builds up until its next message fits."""
    n = len(nodes)
    d = np.hypot(*(pos_ne[:, None, :] - pos_ne[None, :, :]).transpose(2, 0, 1))
    in_range = np.ones((n, n), bool) if radio.range_m is None else d <= radio.range_m
    np.fill_diagonal(in_range, False)
    new: list[list[cm.Observation]] = [[] for _ in range(n)]
    base = cm.HEADER_BYTES + cm.VECTOR_ENTRY_BYTES * n
    sends = []
    for i, node in enumerate(nodes):
        nb = [int(j) for j in np.nonzero(in_range[i])[0]]
        budget = None
        if radio.budget_bytes is not None:
            pending = node.packet(nb, None)
            need = base + (cm.observation_bytes(pending[0]) if pending else 0)
            cap = max(radio.budget_bytes * dt_s, need)
            node.credit = min(node.credit + radio.budget_bytes * dt_s, cap)
            budget = node.credit
            if budget < need:  # still saving up for the next message: stay silent
                sends.append((i, None, []))
                continue
        obs = node.packet(nb, budget)
        size = base + sum(cm.observation_bytes(o) for o in obs)
        if budget is not None:
            node.credit -= size
        node.sent_bytes += size
        sends.append((i, node.vector(), obs))
    for i, vec, obs in sends:
        if vec is None:
            continue
        for j in np.nonzero(in_range[i])[0]:
            if radio.loss > 0 and rng.random() < radio.loss:
                continue
            nodes[j].heard[i] = vec
            for o in obs:
                if nodes[j].add(o):
                    new[j].append(o)
    return new


def job(a: tuple) -> dict:
    st, rule, budget, seed = a
    cm.exchange = exchange_fill_to_fit if rule == "fill_to_fit" else _FROZEN_EXCHANGE
    sc = mr.scenario(st, SENSOR, st["height"], st["n_drones"], "ring")
    pc = sim.PlannerConfig(st["planner"], True, st["block_lanes"])
    r = sim.run(sc, pc, cm.Radio(budget_bytes=budget), seed)
    steps = int(round(min(r.t_found_s or sc.horizon_s, sc.horizon_s) / sc.dt_s))
    return {
        "seed": seed,
        "found": r.found,
        "t": r.t_found_s,
        "horizon": sc.horizon_s,
        "fa_before": r.false_alarms_before_find,
        "passes": r.cell_passes,
        "repeat": r.repeat_cell_passes,
        "bytes": r.bytes_sent,
        "steps": steps,
        "n": st["n_drones"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    st = mr.load_settings()
    if not st["frozen"]:
        print("configs/multi_drone.yaml is not frozen")
        return 1
    cases = [("unlimited", None, None)]
    for rule in ("capped", "fill_to_fit"):
        for b in st["test_budget_bytes"]:
            cases.append((f"{rule}_budget{b}", rule, float(b)))
    rows = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for lab, rule, b in cases:
            jobs = [(st, rule or "capped", b, s) for s in mr.DEV_SEEDS]
            rows[lab] = list(ex.map(job, jobs, chunksize=8))
            print(lab, "done", flush=True)
    rng = np.random.default_rng(0)
    res = {lab: mr.summarise(r, rng) for lab, r in rows.items()}
    paired = {
        lab: {"minus_unlimited": mr.paired(res[lab], res["unlimited"], rng)}
        for lab in res
        if lab != "unlimited"
    }
    per_seed = {lab: {"t_s": v.pop("_t"), "found": v.pop("_found")} for lab, v in res.items()}
    out = {
        "stage": "budget_check (development seeds, after the test; D-756)",
        "sensor": str(SENSOR.relative_to(mr.REPO_ROOT)),
        "settings": st,
        "seeds": [mr.DEV_SEEDS[0], mr.DEV_SEEDS[-1]],
        "results": res,
        "paired": paired,
    }
    path = mr.OUT / "budget_check_p10_v2_cal.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    (mr.OUT / "budget_check_p10_v2_cal_seeds.json").write_text(json.dumps(per_seed) + "\n")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
