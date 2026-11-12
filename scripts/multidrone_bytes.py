"""Where Phase 14's sharing bytes go (Phase 16 stage 0, D-764; development seeds only).

Runs the frozen Phase 14 choice (sharing auction, frozen settings, primary camera, 4 drones,
perfect radio) on development seeds and counts every byte the frozen radio sends, by part:
packet header and version vector, observation header, cell runs, detections; and by whose
observation it is (the sender's own, or relayed). Also counts how many times each observation
was broadcast in total, and how many runs observations would need with fewer probability levels
(256 as frozen, 16, none). The frozen code is not changed: `Node.packet` is wrapped to see what
each broadcast carries.

Usage: python scripts/multidrone_bytes.py --workers 5
Writes results/multi_drone/bytes_p10_v2_cal.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multidrone_run as mr  # noqa: E402
from aerialrange.search import comms as cm  # noqa: E402
from aerialrange.search import sim  # noqa: E402

SENSOR = mr.REPO_ROOT / "results" / "multi_drone" / "sensor_p10_v2_cal.json"
PARTS = ("packet_header_vector", "observation_header", "cell_runs", "detections")


def runs_of(cells: np.ndarray, levels: np.ndarray) -> int:
    """Runs of consecutive cell numbers with one level (as comms._observation_bytes)."""
    if len(cells) == 0:
        return 0
    o = np.argsort(cells)
    c, lv = cells[o], levels[o]
    return 1 + int(np.count_nonzero((np.diff(c) != 1) | (np.diff(lv) != 0)))


def job(seed: int) -> dict:
    st = mr.load_settings()
    sc = mr.scenario(st, SENSOR, st["height"], st["n_drones"], "ring")
    pc = sim.PlannerConfig(st["planner"], True, st["block_lanes"])
    parts: Counter = Counter()
    sends: Counter = Counter()  # observation id -> times broadcast
    uniq: dict = {}
    frozen_packet = cm.Node.packet

    def packet(self: cm.Node, nb: list[int], budget: float | None) -> list:
        out = frozen_packet(self, nb, budget)
        # exchange counts a header and vector every round (no budget here, so never silent)
        parts["packet_header_vector"] += cm.HEADER_BYTES + cm.VECTOR_ENTRY_BYTES * self.n_drones
        for o in out:
            b = cm.observation_bytes(o)
            det = cm.DETECTION_BYTES * len(o.detections)
            parts["observation_header"] += cm.OBS_HEADER_BYTES
            parts["detections"] += det
            parts["cell_runs"] += b - cm.OBS_HEADER_BYTES - det
            parts["own" if o.drone == self.me else "relayed"] += b
            sends[o.id] += 1
            uniq[o.id] = o
        return out

    cm.Node.packet = packet
    try:
        r = sim.run(sc, pc, cm.Radio(), seed)
    finally:
        cm.Node.packet = frozen_packet
    runs = {"256": 0, "16": 0, "none": 0}
    cells = dets = 0
    for o in uniq.values():
        c = np.asarray(o.cells, np.int64)
        p = np.asarray(o.pd, float)
        cells += len(c)
        dets += len(o.detections)
        runs["256"] += runs_of(c, np.round(p * 255))
        runs["16"] += runs_of(c, np.round(p * 15))
        runs["none"] += runs_of(c, np.zeros(len(c)))
    t = r.t_found_s if r.found else sc.horizon_s
    return {
        "seed": seed,
        "seconds": float(t),
        "bytes_sent": int(r.bytes_sent),
        "parts": dict(parts),
        "observations": len(uniq),
        "broadcasts": int(sum(sends.values())),
        "cells": cells,
        "detections": dets,
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args(argv)
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(job, mr.DEV_SEEDS))
    tot = sum(r["bytes_sent"] for r in rows)
    parts = Counter()
    for r in rows:
        parts.update(r["parts"])
    claims = tot - sum(parts[k] for k in PARTS)
    secs = sum(r["seconds"] for r in rows)
    st = mr.load_settings()
    res = {
        "configuration": f"{st['planner']}_b{st['block_lanes']}_share_h{st['height']}",
        "n_drones": st["n_drones"],
        "sensor": SENSOR.relative_to(mr.REPO_ROOT).as_posix(),
        "seeds": [mr.DEV_SEEDS[0], mr.DEV_SEEDS[-1]],
        "bytes_per_drone_per_second": tot / (secs * st["n_drones"]),
        "share_of_bytes": {
            **{k: parts[k] / tot for k in PARTS},
            "claims": claims / tot,
        },
        "share_of_observation_bytes": {
            "own": parts["own"] / (parts["own"] + parts["relayed"]),
            "relayed": parts["relayed"] / (parts["own"] + parts["relayed"]),
        },
        "broadcasts_per_observation": sum(r["broadcasts"] for r in rows)
        / sum(r["observations"] for r in rows),
        "cells_per_observation": sum(r["cells"] for r in rows)
        / sum(r["observations"] for r in rows),
        "runs_per_observation": {
            k: sum(r["runs"][k] for r in rows) / sum(r["observations"] for r in rows)
            for k in ("256", "16", "none")
        },
        "detections_per_observation": sum(r["detections"] for r in rows)
        / sum(r["observations"] for r in rows),
    }
    out = mr.OUT / "bytes_p10_v2_cal.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
