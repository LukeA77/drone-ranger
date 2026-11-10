"""Replay ranging and speed over a saved SynDrone run with other settings, tracks unchanged.

Usage: python scripts/syndrone_refilter.py --source v2trk_accurate --name <new run>
       [--config run_accurate] [--set kalman.motion_model=imm ...] [--towns Town01 ...]
       [--heights 20 50 80]

Reads cache/syndrone/runs/<source>/<town>_h<h>/tracks.csv and replays the stage after the
tracker (`pipeline/range_speed.py`) with the camera poses of each frame (rebuilt as in
scripts/syndrone_run.py) and the configuration given, with `--set section.key=value` overrides
(values parsed as YAML). Writes cache/syndrone/runs/<name>/<town>_h<h>/tracks.csv, plus the
source's detections.csv copied, so scripts/syndrone_evaluate.py scores it like a full run.
Because detection and tracking are untouched, differences between the two runs come from
ranging and filtering alone. `--check` replays with the source's own settings and fails unless
every row matches the source. Development and calibration towns only, unless --allow-test
(Phase 10's one-time run, after freezing).

`--nav-grade nav_good|nav_typical|nav_poor` adds version 1's assumed navigation errors (D-621,
the values of scripts/terrain_pose_sweep.py) to the exact SynDrone poses before replaying, and
sets the pipeline's pose error terms to that grade's jitter, as the Mid-Air sweep did (a system
configured for its navigation grade). Per axis: a position bias and an attitude bias drawn once
per sequence, plus position and attitude jitter drawn per frame, with fixed seeds. The camera
model has heading and tilt but no roll, so the attitude error enters on those two angles only.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aerialrange.data.syndrone import FPS, N_FRAMES, SynDroneSequence, syndrone_camera
from aerialrange.data.syndrone_archive import MemberStore
from aerialrange.pipeline.intervals import config_from_calibration
from aerialrange.pipeline.range_speed import RangeSpeedStage, exact_pose_overrides, replay
from aerialrange.utils.config import REPO_ROOT, load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terrain_pose_sweep import CONDITIONS as SWEEP  # noqa: E402  version 1's grades (D-621)

RUNS = REPO_ROOT / "cache" / "syndrone" / "runs"
CHECK_TOL = 1e-6  # boxes are read back from CSV text, so outputs differ by rounding (~1e-9)
GRADES = ("nav_good", "nav_typical", "nav_poor")
COMPARE = ["range_m", "range_sigma_m", "speed_ms", "speed_sigma_ms", "heading_deg", "nis_mean"]


def flight_meta(store: MemberStore, town: str, height: int) -> pd.DataFrame:
    """Per-frame camera pose, exactly as scripts/syndrone_run.py builds it."""
    recs = SynDroneSequence(store, town, height).camera_records(range(N_FRAMES))
    return pd.DataFrame(
        {
            "frame": range(1, N_FRAMES + 1),
            "t_s": [k / FPS for k in range(N_FRAMES)],
            "n_m": [r.x for r in recs],
            "e_m": [r.y for r in recs],
            "h_m": [r.h for r in recs],
            "yaw_rad": [math.radians(r.yaw) for r in recs],
            "tilt_deg": [-r.pitch for r in recs],
        }
    ).set_index("frame")


def town_number(town: str) -> int:
    """Town01 -> 1, Town10HD -> 10: part of each sequence's seed."""
    return int("".join(c for c in town[4:] if c.isdigit()))


def perturb_flight(meta: pd.DataFrame, grade: str, town: str, h: int, seed: int) -> pd.DataFrame:
    """The flight's poses with a navigation grade's errors added (see the module docstring).
    Bias: one draw per sequence; jitter: one draw per frame; seeds fixed by sequence and grade."""
    pj, aj, pb, ab = SWEEP[grade]
    ci = list(SWEEP).index(grade)
    seq = town_number(town) * 1000 + h
    rb = np.random.default_rng([seed, seq, ci])
    bp = rb.normal(0.0, pb, 3)  # north, east, height (m)
    ba = rb.normal(0.0, ab, 2)  # heading, tilt (deg)
    jit = np.array(
        [
            np.random.default_rng([seed, seq, ci, int(f)]).normal(0.0, [pj] * 3 + [aj] * 2)
            for f in meta.index
        ]
    ).reshape(-1, 5)
    out = meta.copy()
    for c, k in (("n_m", 0), ("e_m", 1), ("h_m", 2)):
        out[c] = meta[c] + bp[k] + jit[:, k]
    out["yaw_rad"] = meta["yaw_rad"] + np.radians(ba[0] + jit[:, 3])
    out["tilt_deg"] = meta["tilt_deg"] + ba[1] + jit[:, 4]
    return out


def parse_sets(items: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for it in items:
        key, _, val = it.partition("=")
        sec, _, name = key.partition(".")
        if not name:
            raise ValueError(f"--set needs section.key=value, got {it!r}")
        out.setdefault(sec, {})[name] = yaml.safe_load(val)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--name")
    ap.add_argument("--config", default="run_accurate")
    ap.add_argument("--set", nargs="*", default=[], dest="sets")
    ap.add_argument("--towns", nargs="*")
    ap.add_argument("--heights", type=int, nargs="*", default=[20, 50, 80])
    ap.add_argument("--check", action="store_true", help="replay unchanged; must match source")
    ap.add_argument(
        "--intervals", help="interval calibration result to apply (D-737; SynDrone conditions)"
    )
    ap.add_argument(
        "--keep-pose-errors",
        action="store_true",
        help="keep the configured pose error terms (default: zeroed, SynDrone poses are exact)",
    )
    ap.add_argument("--nav-grade", choices=GRADES, help="add a navigation grade's pose errors")
    ap.add_argument("--seed", type=int, default=0, help="seed for --nav-grade")
    ap.add_argument("--allow-test", action="store_true", help="Phase 10's one-time run only")
    args = ap.parse_args(argv)
    if args.nav_grade and (args.check or args.keep_pose_errors):
        ap.error("--nav-grade sets its own pose errors: not with --check or --keep-pose-errors")
    if not args.check and not args.name:
        ap.error("--name is required unless --check")
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text("utf-8"))["syndrone"][
        "splits"
    ]
    towns = args.towns or sp["development"]
    bad = [t for t in towns if t in sp["test"]]
    if bad and not args.allow_test:
        print(f"test towns are not replayed: {bad}")
        return 2
    cfg = load_config(args.config)
    cam = syndrone_camera()
    ov = parse_sets(args.sets)
    ov.setdefault("camera", {}).update({"focal_px": cam.fx, "cx": cam.cx, "cy": cam.cy})
    if not args.keep_pose_errors:
        # SynDrone poses are exact: zero the pose error terms, as scripts/syndrone_run.py does
        # (before this was automatic, replays passed them with --set; D-741 note)
        kal = {**cfg.require("kalman"), **ov.get("kalman", {})}
        for k, v in exact_pose_overrides(kal).get("kalman", {}).items():
            ov.setdefault("kalman", {})[k] = v
    if args.nav_grade:
        pj, aj, _, _ = SWEEP[args.nav_grade]
        kal = {**cfg.require("kalman"), **ov.get("kalman", {})}
        if kal.get("meas_model") == "propagated":
            ov.setdefault("kalman", {})["meas_sigma"] = {
                **kal["meas_sigma"],
                "tilt_deg": float(aj),
                "yaw_deg": float(aj),
                "h_m": float(pj),
            }
    if args.intervals:
        ov["intervals"] = config_from_calibration(REPO_ROOT / args.intervals)

    def get(k: str) -> Any:
        try:
            base = cfg.require(k)
        except KeyError:  # a section given only as an override (intervals)
            if k in ov:
                return ov[k]
            raise
        return {**base, **ov.get(k, {})} if isinstance(base, dict) else ov.get(k, base)

    d = load_config("data")
    store = MemberStore(
        REPO_ROOT / d.require("data_root") / d.require("syndrone_subdir"),
        d.require("syndrone_base_url"),
    )
    worst = 0.0
    for town in towns:
        for h in args.heights:
            src = RUNS / args.source / f"{town}_h{h}"
            if not (src / "tracks.csv").exists():
                print(f"{src.name}: no source run, skipped")
                continue
            tracks = pd.read_csv(src / "tracks.csv")
            meta = flight_meta(store, town, h)
            if args.nav_grade:
                meta = perturb_flight(meta, args.nav_grade, town, h, args.seed)
            new = replay(tracks, meta, RangeSpeedStage.from_config(get))
            if args.check:
                a, b = tracks[COMPARE].to_numpy(float), new[COMPARE].to_numpy(float)
                same_nan = np.array_equal(np.isnan(a), np.isnan(b))
                diff = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0
                worst = max(worst, diff if same_nan else np.inf)
                print(
                    f"{src.name}: {len(a)} rows, NaN pattern same {same_nan}, max diff {diff:.3g}"
                )
                continue
            dst = RUNS / args.name / src.name
            dst.mkdir(parents=True, exist_ok=True)
            new.to_csv(dst / "tracks.csv", index=False)
            for extra in ("detections.csv", "detector.json"):
                if (src / extra).exists():
                    shutil.copy2(src / extra, dst / extra)
            (dst / "refilter.json").write_text(
                json.dumps(
                    {
                        "source": args.source,
                        "config": args.config,
                        "overrides": ov,
                        "nav_grade": args.nav_grade,
                        "nav_grade_errors": SWEEP[args.nav_grade] if args.nav_grade else None,
                        "seed": args.seed if args.nav_grade else None,
                    },
                    indent=1,
                )
                + "\n"
            )
            print(f"{dst}: {len(new)} rows", flush=True)
    if args.check:
        ok = worst < CHECK_TOL
        print("replay reproduces the source" if ok else f"MISMATCH (max diff {worst})")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
