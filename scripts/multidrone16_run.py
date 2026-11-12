"""Phase 16 search simulation runs (D-764): design stages on development seeds, then the
one-time test.

Usage:
  python scripts/multidrone16_run.py design1 --workers 5   # k x lane spacing x persistence
  python scripts/multidrone16_run.py design2 --workers 5   # what the drones do after a sweep
  python scripts/multidrone16_run.py design3 --workers 5   # lost-drone timeout; coordinator
  python scripts/multidrone16_run.py test --workers 5      # once, after freezing (phase16)

Everything not chosen here is Phase 14's (configs/multi_drone.yaml, frozen): area, prior,
height, speed, team size, radio sweeps, horizon (three sweeps' time). The camera is Phase 14's
primary one (results/multi_drone/sensor_p10_v2_cal.json), refitted for a persistence filter
when one is used (sensor_p10_v2_cal_m<m>.json), with the measured recurrence of false alarms
(persistent look-alikes fitted in clutter_p10_v2_cal.json, at the map's 5-sigma radius).
Summaries reuse Phase 14's (multidrone_run.summarise, .paired): restricted mean time to the
confirmed find (no find counts as the horizon), 95 % bootstrap intervals over seeds, paired
differences on the same seeds.

Choice rules (fixed in D-764 before any run): lowest restricted mean time to confirmed find; a
setting whose paired difference with the best has an interval including 0 is a tie; ties go to
the simpler option:
  design1: smaller k, then Phase 14's lane spacing (factor 1), then no filter (smaller m);
    stage 1 sweeps again after each sweep (the option that needs no map);
  design2: after a sweep, in order of simplicity: sweep again; Phase 14's map-guided blocks
    (4 lanes, as frozen); the same on a coverage-only map; map-guided 2- and 8-lane blocks;
  design3: chosen at 50 % loss with one drone lost at half a sweep (D-765); the longest
    timeout among the tied (fewer wrong "lost" verdicts).
Writes results/multi_drone/phase16_<stage>.json (and _seeds.json).
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
from aerialrange.search import engine as en  # noqa: E402
from aerialrange.utils.config import load_config  # noqa: E402

TEST_SEEDS = list(range(20000, 20300))  # D-764: new seeds; Phase 14's test seeds stay used
CAMERAS = {"primary": "p10_v2_cal", "rural": "v2_dfine_cal_rural"}
KEYS16 = (
    "frozen",
    "design_confirm_k",
    "design_spacing_factor",
    "design_min_track_frames",
    "design_post",
    "design_lost_after_s",
    "design_lost_loss",
    "confirm_k",
    "spacing_factor",
    "min_track_frames",
    "post",
    "block_lanes",
    "lost_after_s",
)


def load16() -> dict:
    cfg = load_config("search16")
    return {k: cfg.get(k) for k in KEYS16}


def camera_m(camera: str, m_primary: int) -> int:
    """The persistence value of `camera` given by the same rule as m_primary on the primary
    camera (none, false-alarm median, person median)."""
    prim = json.loads((mr.OUT / "clutter_p10_v2_cal.json").read_text())["persistence_values"]["m"]
    mine = json.loads((mr.OUT / f"clutter_{CAMERAS[camera]}.json").read_text())
    return int(mine["persistence_values"]["m"][prim.index(int(m_primary))])


def camera_files(camera: str, m: int) -> tuple[Path, float, float, float]:
    """(sensor file, look-alike share, look-alike fire probability, radius) for a camera and
    persistence value (the two-kind fit at the map's 5-sigma radius)."""
    name = CAMERAS[camera]
    cl = json.loads((mr.OUT / f"clutter_{name}.json").read_text())
    rec = cl["by_min_track_frames"][str(m)]["recurrence"][f"{cl['radius_m']:g}"]["persistent"]
    sensor = mr.OUT / (f"sensor_{name}.json" if m == 1 else f"sensor_{name}_m{m}.json")
    return sensor, float(rec["share"]), float(rec["fire_per_pass"]), float(cl["radius_m"])


def job(a: tuple) -> dict:
    st, c, seed = a
    sensor, share, fire, radius = camera_files(c["camera"], c["m"])
    sc = mr.scenario(st, sensor, st["height"], c["n"], "ring")
    lose = None
    if c.get("lose") is not None:
        who = c["lose"]
        if who == "any":
            who = int(np.random.default_rng([seed, 4]).integers(c["n"]))
        lose = (int(who), sc.horizon_s / mr.HORIZON_COVERS / 2)  # half one sweep's time
    world = en.World(
        persistent_share=share,
        persistent_fire=fire,
        confirm_k=c["k"],
        confirm_radius_m=radius,
        stop="confirmed",
        lose=lose,
    )
    r = en.run(sc, en.Plan(**c["pc"]), cm.Radio(**c.get("radio", {})), seed, world)
    return {
        "seed": seed,
        "found": r.confirmed,
        "t": r.t_confirmed_s,
        "first_found": r.found,
        "first_t": r.t_found_s,
        "horizon": sc.horizon_s,
        "crew_before": r.crew_alarms_before_confirmed,
        "fa_before": r.false_alarms_before_find,
        "passes": r.cell_passes,
        "repeat": r.repeat_cell_passes,
        "bytes": r.bytes_sent,
        "steps": r.steps,
        "n": c["n"],
    }


def sweep_pc(f: float, post: str, bl: int, lost: float, coord: bool = False) -> dict:
    return {
        "kind": "sweep",
        "sharing": "compact",
        "spacing_factor": f,
        "post": post,
        "block_lanes": bl,
        "lost_after_s": lost,
        "coordinator": coord,
    }


POST_ORDER = ("resweep", "map_b4", "coverage_b4", "map_b2", "map_b8")  # simplest first


PHASE14_AUCTION = {"kind": "auction", "share": True, "block_lanes": 4}


def configs(stage: str, s16: dict, st: dict) -> list[tuple[str, dict]]:
    n4 = st["n_drones"]
    base = {"camera": "primary", "n": n4}
    out: list[tuple[str, dict]] = []
    if stage == "design1":
        for k in s16["design_confirm_k"]:
            for m in s16["design_min_track_frames"]:
                b = {**base, "k": k, "m": m}
                out.append((f"lawnmower_k{k}_m{m}", {**b, "pc": {"kind": "lawnmower"}}))
                if (k, m) == (s16["design_confirm_k"][0], s16["design_min_track_frames"][0]):
                    # context only (not a choice): Phase 14's planner in the new world
                    out.append((f"auction14_k{k}_m{m}", {**b, "pc": PHASE14_AUCTION}))
                for f in s16["design_spacing_factor"]:
                    pc = sweep_pc(f, "resweep", 4, 30.0)
                    out.append((f"sweep_k{k}_f{f:g}_m{m}", {**b, "pc": pc}))
        return out
    k, f, m = s16["confirm_k"], s16["spacing_factor"], s16["min_track_frames"]
    if None in (k, f, m):
        raise ValueError("set confirm_k, spacing_factor and min_track_frames from design1 first")
    b = {**base, "k": k, "m": m}
    if stage == "design2":
        for opt in s16["design_post"]:
            post, _, blk = opt.partition("_b")
            pc = sweep_pc(f, post, int(blk or 4), 30.0)
            out.append((f"sweep_post_{opt}", {**b, "pc": pc}))
        return out
    post, bl = s16["post"], s16["block_lanes"]
    if post is None or bl is None:
        raise ValueError("set post and block_lanes from design2 first")
    if stage == "design3":
        loss = {"loss": s16["design_lost_loss"]}
        for lost in s16["design_lost_after_s"]:
            out.append(
                (f"sweep_T{lost:g}_loss", {**b, "pc": sweep_pc(f, post, bl, lost), "radio": loss})
            )
            # the choice: loss and a drone really lost at half a sweep, so a timeout that is
            # too short (wrong verdicts) and one too long (slow takeover) both cost time
            out.append(
                (
                    f"sweep_T{lost:g}_loss_lose",
                    {**b, "pc": sweep_pc(f, post, bl, lost), "radio": loss, "lose": "any"},
                )
            )
            pc = sweep_pc(f, post, bl, lost, True)
            out.append((f"coord_T{lost:g}_loss", {**b, "pc": pc, "radio": loss}))
        for coord in (False, True):
            lab = "coord_T30_perfect" if coord else "sweep_T30_perfect"
            out.append((lab, {**b, "pc": sweep_pc(f, post, bl, 30.0, coord)}))
        return out
    if stage != "test":
        raise ValueError(f"unknown stage {stage}")
    lost = s16["lost_after_s"]
    if lost is None or not s16["frozen"]:
        raise ValueError("test needs every choice set and frozen: true")
    sw_pc, co_pc = sweep_pc(f, post, bl, lost), sweep_pc(f, post, bl, lost, True)
    for n in st["test_n_drones"]:
        c = {**b, "n": n}
        out.append((f"lawnmower_n{n}", {**c, "pc": {"kind": "lawnmower"}}))
        out.append((f"auction14_share_n{n}", {**c, "pc": PHASE14_AUCTION}))
        out.append((f"auction14_noshare_n{n}", {**c, "pc": {**PHASE14_AUCTION, "share": False}}))
        out.append((f"sweep_n{n}", {**c, "pc": sw_pc}))
        if n > 1:
            out.append((f"coord_n{n}", {**c, "pc": co_pc}))
    out.append(("auction14_compact_n4", {**b, "pc": {**PHASE14_AUCTION, "sharing": "compact"}}))
    for loss in st["test_loss"]:
        if loss > 0:
            out.append((f"sweep_loss{loss:g}", {**b, "pc": sw_pc, "radio": {"loss": loss}}))
    for rng_m in st["test_range_m"]:
        out.append((f"sweep_range{rng_m:g}", {**b, "pc": sw_pc, "radio": {"range_m": rng_m}}))
    for bud in st["test_budget_bytes"]:
        r = {"budget_bytes": float(bud)}
        out.append((f"sweep_budget{bud:g}", {**b, "pc": sw_pc, "radio": r}))
    out.append(("lawnmower_lose_any", {**b, "pc": {"kind": "lawnmower"}, "lose": "any"}))
    out.append(("sweep_lose_any", {**b, "pc": sw_pc, "lose": "any"}))
    out.append(("coord_lose_any", {**b, "pc": co_pc, "lose": "any"}))
    out.append(("coord_lose_coordinator", {**b, "pc": co_pc, "lose": 0}))
    out.append(("sweep_n3_from_start", {**b, "n": n4 - 1, "pc": sw_pc}))
    for lab, pc in (
        ("lawnmower", {"kind": "lawnmower"}),
        ("auction14_share", PHASE14_AUCTION),
        ("sweep", sw_pc),
        ("coord", co_pc),
    ):
        cr = {**b, "camera": "rural", "m": camera_m("rural", m)}
        out.append((f"rural_{lab}_n4", {**cr, "pc": pc}))
    return out


def summary(rows: list[dict], rng: np.random.Generator) -> dict:
    s = mr.summarise(rows, rng)
    first = [{**r, "found": r["first_found"], "t": r["first_t"]} for r in rows]
    f = mr.summarise(first, rng)
    crew = np.array([r["crew_before"] for r in rows], float)
    boots = [float(np.mean(crew[rng.integers(0, len(crew), len(crew))])) for _ in range(mr.N_BOOT)]
    s["crew_alarms_before_find_mean"] = float(crew.mean())
    s["ci95"]["crew_alarms_before_find_mean"] = [
        float(np.percentile(boots, q)) for q in (2.5, 97.5)
    ]
    s["first_detection"] = {
        k: f[k]
        for k in ("restricted_mean_time_s", "p_found_by_T1", "p_found_by_horizon", "ci95", "_t")
    }
    return s


def choose(stage: str, res: dict, rng: np.random.Generator) -> dict:
    """Apply the stage's rule (module docstring) to the sweep settings of the stage."""
    if stage == "design1":
        labs = [lab for lab in res if lab.startswith("sweep_")]

        def simple(lab: str) -> tuple:
            p = {x[0]: x[1:] for x in lab.split("_")[1:]}
            return (int(p["k"]), float(p["f"]) != 1.0, int(p["m"]))

    elif stage == "design2":
        labs = [lab for lab in res if lab.startswith("sweep_post_")]

        def simple(lab: str) -> tuple:
            return (POST_ORDER.index(lab[len("sweep_post_") :]),)

    elif stage == "design3":
        labs = [lab for lab in res if lab.startswith("sweep_T") and lab.endswith("_loss_lose")]

        def simple(lab: str) -> tuple:
            return (-float(lab[len("sweep_T") : -len("_loss_lose")]),)

    else:
        return {}
    best = min(labs, key=lambda x: res[x]["restricted_mean_time_s"])
    tied = [best]
    diffs = {}
    for lab in labs:
        if lab == best:
            continue
        d = mr.paired(res[lab], res[best], rng)
        diffs[lab] = d
        lo, hi = d["restricted_mean_time_diff_ci95"]
        if lo <= 0 <= hi:
            tied.append(lab)
    pick = min(tied, key=simple)
    return {"best": best, "tied_with_best": sorted(tied), "chosen": pick, "minus_best": diffs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("design1", "design2", "design3", "test"))
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args(argv)
    st, s16 = mr.load_settings(), load16()
    cfgs = configs(args.stage, s16, st)
    seeds = TEST_SEEDS if args.stage == "test" else mr.DEV_SEEDS
    out = mr.OUT / f"phase16_{args.stage}.json"
    if args.stage == "test" and out.exists():
        print(f"{out} exists: the test runs once")
        return 1
    jobs = [(st, c, s) for _, c in cfgs for s in seeds]
    print(f"{args.stage}: {len(cfgs)} settings x {len(seeds)} seeds = {len(jobs)} searches")
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(job, jobs, chunksize=4))
    by: dict[str, list[dict]] = {}
    for (lab, _), i in zip(cfgs, range(0, len(rows), len(seeds)), strict=True):
        by[lab] = rows[i : i + len(seeds)]
    rng = np.random.default_rng(0)
    res = {lab: summary(r, rng) for lab, r in by.items()}
    pick = choose(args.stage, res, rng)
    for lab, r in res.items():
        ci = r["ci95"]["restricted_mean_time_s"]
        first = r["first_detection"]["restricted_mean_time_s"]
        print(
            f"{lab:28s} confirmed {r['restricted_mean_time_s']:7.0f} s "
            f"({ci[0]:.0f} to {ci[1]:.0f})  first {first:7.0f} s  "
            f"crew {r['crew_alarms_before_find_mean']:8.1f}  "
            f"B/s {r['bytes_per_drone_s_mean']:7.1f}",
            flush=True,
        )
    if pick:
        print("choice:", json.dumps({k: v for k, v in pick.items() if k != "minus_best"}))
    doc = {
        "stage": args.stage,
        "seeds": [seeds[0], seeds[-1]],
        "settings": {lab: c for lab, c in cfgs},
        "phase14_settings": {k: st[k] for k in ("height", "n_drones", "speed_ms", "area_ring")},
        "choice": pick,
        "results": {
            lab: {k: v for k, v in r.items() if not k.startswith("_")} for lab, r in res.items()
        },
    }
    for r in doc["results"].values():
        r["first_detection"] = {k: v for k, v in r["first_detection"].items() if k != "_t"}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    seeds_out = {lab: rows_ for lab, rows_ in by.items()}
    (mr.OUT / f"phase16_{args.stage}_seeds.json").write_text(json.dumps(seeds_out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
