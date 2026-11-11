"""Run the Phase 14 multi-drone search experiments (D-752).

Usage:
  python scripts/multidrone_run.py design   # development seeds: choose planner and block size
  python scripts/multidrone_run.py test     # test seeds, once, with the frozen choices
  (--workers N, default 4; --sensor <sensor json> as many times as needed)

Seeds: development 0..199, test 10000..10299 (the person's position and every random outcome
come from the seed, so planners are compared on the same people: paired comparisons).

Each configuration is summarised over its seeds:
  - probability of a find by time t (curve, and at the fleet's single-coverage time T1 and at
    the horizon 3 x T1, where T1 = area / (drones x speed x lane spacing));
  - restricted mean time to find up to the horizon (searches without a find count as the
    horizon), the standard summary of a time-to-event curve cut at a horizon;
  - false alarms raised before the find; repeated coverage (share of cell passes over ground
    already passed); bytes sent per drone per second;
  with 95 % bootstrap intervals over seeds, and paired differences against the configured
  references.
Writes results/multi_drone/<stage>_<sensor name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from aerialrange.search import comms as cm
from aerialrange.search import prior as pr
from aerialrange.search import sim
from aerialrange.search.grid import Grid
from aerialrange.search.sensor import SensorModel
from aerialrange.utils.config import REPO_ROOT, load_config

OUT = REPO_ROOT / "results" / "multi_drone"
DEV_SEEDS = list(range(200))
TEST_SEEDS = list(range(10000, 10300))  # D-752 addendum 2: 300, for run time
CELL_M = 10.0
N_BOOT = 1000
HORIZON_COVERS = 3.0


KEYS = (
    "frozen",
    "speed_ms",
    "area_ring",
    "height",
    "planner",
    "block_lanes",
    "n_drones",
    "design_heights",
    "design_block_lanes",
    "test_n_drones",
    "test_loss",
    "test_range_m",
    "test_budget_bytes",
)


def load_settings() -> dict:
    cfg = load_config("multi_drone")
    return {k: cfg.get(k) for k in KEYS}


def scenario(st: dict, sensor_path: Path, height: int, n: int, prior_kind: str) -> sim.Scenario:
    radii = tuple(1000.0 * r for r in pr.HIKER_TEMPERATE_MOUNTAIN_KM)
    half = radii[st["area_ring"]]
    g = Grid.centred((0.0, 0.0), half, CELL_M)
    p = pr.ring_prior(g, (0.0, 0.0), radii) if prior_kind == "ring" else pr.uniform_prior(g)
    s = SensorModel.from_results(sensor_path, height)
    t1 = g.size * g.cell_area_m2 / (n * st["speed_ms"] * sim.lane_spacing_m(s))
    return sim.Scenario(g, p, s, n, st["speed_ms"], (0.0, 0.0), HORIZON_COVERS * t1)


def one(args: tuple) -> dict:
    st, sensor_path, height, n, prior_kind, pc, radio, seed = args
    sc = scenario(st, Path(sensor_path), height, n, prior_kind)
    r = sim.run(sc, sim.PlannerConfig(**pc), cm.Radio(**radio), seed)
    return {
        "seed": seed,
        "found": r.found,
        "t": r.t_found_s,
        "horizon": sc.horizon_s,
        "fa_before": r.false_alarms_before_find,
        "passes": r.cell_passes,
        "repeat": r.repeat_cell_passes,
        "bytes": r.bytes_sent,
        "steps": int(round(min(r.t_found_s or sc.horizon_s, sc.horizon_s) / sc.dt_s)),
        "n": n,
    }


def summarise(rows: list[dict], rng: np.random.Generator) -> dict:
    h = rows[0]["horizon"]
    t = np.array([r["t"] if r["found"] else h for r in rows])
    found = np.array([r["found"] for r in rows])
    t1 = h / HORIZON_COVERS
    fa = np.array([r["fa_before"] for r in rows], float)
    rep = np.array([r["repeat"] for r in rows], float)
    pas = np.array([r["passes"] for r in rows], float)
    bps = np.array([r["bytes"] / max(r["steps"], 1) / r["n"] for r in rows], float)

    def stats(idx: np.ndarray) -> dict:
        return {
            "p_found_by_T1": float(np.mean(found[idx] & (t[idx] <= t1))),
            "p_found_by_horizon": float(np.mean(found[idx])),
            "restricted_mean_time_s": float(np.mean(t[idx])),
            "false_alarms_before_find_mean": float(np.mean(fa[idx])),
            "repeated_coverage": float(rep[idx].sum() / max(pas[idx].sum(), 1)),
            "bytes_per_drone_s_mean": float(np.mean(bps[idx])),
        }

    allidx = np.arange(len(rows))
    point = stats(allidx)
    boots = [stats(rng.integers(0, len(rows), len(rows))) for _ in range(N_BOOT)]
    ci = {k: [float(np.percentile([b[k] for b in boots], q)) for q in (2.5, 97.5)] for k in point}
    grid_t = np.linspace(0, h, 61)
    curve = [float(np.mean(found & (t <= x))) for x in grid_t]
    return {
        "n_seeds": len(rows),
        "T1_s": t1,
        "horizon_s": h,
        **point,
        "ci95": ci,
        "curve_t_s": grid_t.tolist(),
        "curve_p_found": curve,
        "_t": t.tolist(),
        "_found": found.tolist(),
    }


def paired(a: dict, b: dict, rng: np.random.Generator) -> dict:
    """a minus b on the same seeds: restricted mean time and found-by-T1."""
    ta, tb = np.array(a["_t"]), np.array(b["_t"])
    t1 = a["T1_s"]
    fa_, fb = np.array(a["_found"]), np.array(b["_found"])
    da = ta - tb
    dp = (fa_ & (ta <= t1)).astype(float) - (fb & (tb <= t1)).astype(float)
    n = len(da)
    bd = [np.mean(da[i]) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    bp = [np.mean(dp[i]) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return {
        "restricted_mean_time_diff_s": float(np.mean(da)),
        "restricted_mean_time_diff_ci95": [float(np.percentile(bd, q)) for q in (2.5, 97.5)],
        "p_found_by_T1_diff": float(np.mean(dp)),
        "p_found_by_T1_diff_ci95": [float(np.percentile(bp, q)) for q in (2.5, 97.5)],
    }


def configs(stage: str, st: dict) -> list[tuple[str, dict]]:
    """(label, settings) for each configuration of a stage."""
    out = []
    base = {"height": st["height"], "n": st["n_drones"], "prior": "ring", "radio": {}}
    if stage == "test" and (st["test_budget_bytes"] is None or st["height"] is None):
        raise ValueError("test stage needs height and test_budget_bytes set")
    if stage == "design":
        for h in st["design_heights"]:
            out.append((f"lawnmower_h{h}", {**base, "height": h, "pc": {"kind": "lawnmower"}}))
            for kind in ("greedy", "auction"):
                for bl in st["design_block_lanes"]:
                    for share in (True, False):
                        lab = f"{kind}_b{bl}_{'share' if share else 'noshare'}_h{h}"
                        pc = {"kind": kind, "share": share, "block_lanes": bl}
                        out.append((lab, {**base, "height": h, "pc": pc}))
        return out
    chosen = {"kind": st["planner"], "block_lanes": st["block_lanes"]}
    for n in st["test_n_drones"]:
        b = {**base, "n": n}
        out.append((f"lawnmower_n{n}", {**b, "pc": {"kind": "lawnmower"}}))
        out.append((f"chosen_noshare_n{n}", {**b, "pc": {**chosen, "share": False}}))
        out.append((f"chosen_share_n{n}", {**b, "pc": {**chosen, "share": True}}))
    for loss in st["test_loss"]:
        out.append((f"chosen_share_loss{loss}", {**base, "pc": chosen, "radio": {"loss": loss}}))
    for rng_m in st["test_range_m"]:
        out.append(
            (f"chosen_share_range{rng_m}", {**base, "pc": chosen, "radio": {"range_m": rng_m}})
        )
    for bud in st["test_budget_bytes"]:
        out.append(
            (f"chosen_share_budget{bud}", {**base, "pc": chosen, "radio": {"budget_bytes": bud}})
        )
    out.append(("lawnmower_uniform", {**base, "prior": "uniform", "pc": {"kind": "lawnmower"}}))
    out.append(("chosen_share_uniform", {**base, "prior": "uniform", "pc": chosen}))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("design", "test"))
    ap.add_argument("--sensor", action="append", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="run only these configuration labels")
    args = ap.parse_args(argv)
    st = load_settings()
    if args.stage == "test" and not st.get("frozen"):
        print("configs/multi_drone.yaml is not frozen: set the design choices and frozen: true")
        return 1
    seeds = DEV_SEEDS if args.stage == "design" else TEST_SEEDS
    for sensor in args.sensor:
        name = Path(sensor).stem.removeprefix("sensor_")
        cfgs, not_run = [], {}
        for lab, c in configs(args.stage, st):
            if args.only and lab not in args.only:
                continue
            if SensorModel.from_results(Path(sensor), c["height"]).can_detect:
                cfgs.append((lab, c))
            else:  # nobody can be found: no search to compare
                not_run[lab] = f"no person detected at {c['height']} m in this sensor model"
        jobs, keys = [], []
        for lab, c in cfgs:
            for seed in seeds:
                jobs.append(
                    (st, sensor, c["height"], c["n"], c["prior"], c["pc"], c["radio"], seed)
                )
                keys.append(lab)
        rows: dict[str, list[dict]] = {lab: [] for lab, _ in cfgs}
        with ProcessPoolExecutor(args.workers) as ex:
            for k, (lab, r) in enumerate(zip(keys, ex.map(one, jobs, chunksize=8), strict=True)):
                rows[lab].append(r)
                if (k + 1) % 500 == 0:
                    print(f"{name}: {k + 1}/{len(jobs)}", flush=True)
        rng = np.random.default_rng(0)
        res = {lab: summarise(rows[lab], rng) for lab, _ in cfgs}
        refs = {}
        for lab in res:
            if args.stage == "test" and lab.startswith("chosen_share_n"):
                n = lab.removeprefix("chosen_share_n")
                if f"chosen_noshare_n{n}" not in res or f"lawnmower_n{n}" not in res:
                    continue
                refs[lab] = {
                    "minus_noshare": paired(res[lab], res[f"chosen_noshare_n{n}"], rng),
                    "minus_lawnmower": paired(res[lab], res[f"lawnmower_n{n}"], rng),
                }
            elif args.stage == "test" and lab.startswith(
                ("chosen_share_loss", "chosen_share_range", "chosen_share_budget")
            ):
                if "chosen_share_n4" in res:  # the same team with an unlimited, perfect radio
                    refs[lab] = {
                        "minus_perfect_radio": paired(res[lab], res["chosen_share_n4"], rng)
                    }
            elif lab == "chosen_share_uniform" and "lawnmower_uniform" in res:
                refs[lab] = {"minus_lawnmower": paired(res[lab], res["lawnmower_uniform"], rng)}
        per_seed = {
            lab: {"seed": [r["seed"] for r in rows[lab]], "t_s": v["_t"], "found": v["_found"]}
            for lab, v in res.items()
        }
        for v in res.values():
            v.pop("_t"), v.pop("_found")
        out = {
            "stage": args.stage,
            "sensor": sensor,
            "settings": st,
            "seeds": [seeds[0], seeds[-1]],
            "configs": {lab: c for lab, c in cfgs},
            "results": res,
            "paired": refs,
            "not_run": not_run,
        }
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{args.stage}_{name}.json"
        path.write_text(json.dumps(out, indent=1, default=lambda o: asdict(o)) + "\n")
        print("wrote", path)
        # per-seed outcomes (time to find, searches without a find at the horizon), so any
        # further paired comparison can be checked from the file
        seeds_path = OUT / f"{args.stage}_{name}_seeds.json"
        seeds_path.write_text(json.dumps(per_seed) + "\n")
        print("wrote", seeds_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
