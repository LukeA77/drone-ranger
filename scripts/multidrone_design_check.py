"""Phase 14 design check on development seeds only (D-752 addendum 3): why the lawnmower beats
the map-guided planners at 20 m.
1. Paired difference auction b4 vs b8 (share, h20) on seeds 0..199.
2. Mechanism: lawnmower vs auction b4 share h20 with false alarms off and/or pd constant
   (the height's average), seeds 0..99.
Usage: python scripts/multidrone_design_check.py . 12
Writes results/multi_drone/design_check_p10_v2_cal.json."""

import dataclasses
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "scripts"))
import multidrone_run as mr  # noqa: E402
from aerialrange.search import comms as cm  # noqa: E402
from aerialrange.search import sim  # noqa: E402

SENSOR = repo / "results/multi_drone/sensor_p10_v2_cal.json"


def job(a: tuple) -> dict:
    st, pc, seed, fa_off, pd_const = a
    sc = mr.scenario(st, SENSOR, 20, 4, "ring")
    s = sc.sensor
    if fa_off:
        s = dataclasses.replace(s, fa_per_m2=1e-12)
    if pd_const:
        s = dataclasses.replace(s, pd_bins=())
    sc = dataclasses.replace(sc, sensor=s)
    r = sim.run(sc, sim.PlannerConfig(**pc), cm.Radio(), seed)
    return {
        "t": r.t_found_s if r.found else sc.horizon_s,
        "rep": r.repeat_cell_passes,
        "pas": r.cell_passes,
    }


def main() -> None:
    st = mr.load_settings()
    w = int(sys.argv[2])
    out = {}
    with ProcessPoolExecutor(w) as ex:
        b4 = {"kind": "auction", "share": True, "block_lanes": 4}
        b8 = {"kind": "auction", "share": True, "block_lanes": 8}
        t4 = np.array([r["t"] for r in ex.map(job, [(st, b4, s, 0, 0) for s in range(200)])])
        t8 = np.array([r["t"] for r in ex.map(job, [(st, b8, s, 0, 0) for s in range(200)])])
        d = t4 - t8
        rng = np.random.default_rng(0)
        bs = [np.mean(d[rng.integers(0, 200, 200)]) for _ in range(1000)]
        out["paired_b4_minus_b8"] = {
            "mean_s": float(d.mean()),
            "rmt_b4": float(t4.mean()),
            "rmt_b8": float(t8.mean()),
            "ci95": [float(np.percentile(bs, q)) for q in (2.5, 97.5)],
        }
        print(json.dumps(out), flush=True)
        lm = {"kind": "lawnmower"}
        for fa_off in (0, 1):
            for pd_const in (0, 1):
                for lab, pc in (("lawnmower", lm), ("auction_b4", b4), ("auction_b8", b8)):
                    rs = list(ex.map(job, [(st, pc, s, fa_off, pd_const) for s in range(100)]))
                    t = np.array([r["t"] for r in rs])
                    key = (
                        f"{lab}_fa{'off' if fa_off else 'on'}_pd{'const' if pd_const else 'bylen'}"
                    )
                    out[key] = {
                        "rmt": float(t.mean()),
                        "repeat": sum(r["rep"] for r in rs) / sum(r["pas"] for r in rs),
                    }
                    print(key, out[key], flush=True)
    (repo / "results/multi_drone/design_check_p10_v2_cal.json").write_text(
        json.dumps(out, indent=1)
    )


if __name__ == "__main__":
    main()
