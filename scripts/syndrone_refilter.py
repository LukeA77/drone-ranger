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
every row matches the source. Development and calibration towns only.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aerialrange.data.syndrone import FPS, N_FRAMES, SynDroneSequence, syndrone_camera
from aerialrange.data.syndrone_archive import MemberStore
from aerialrange.pipeline.range_speed import RangeSpeedStage, replay
from aerialrange.utils.config import REPO_ROOT, load_config

RUNS = REPO_ROOT / "cache" / "syndrone" / "runs"
CHECK_TOL = 1e-6  # boxes are read back from CSV text, so outputs differ by rounding (~1e-9)
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
    args = ap.parse_args(argv)
    if not args.check and not args.name:
        ap.error("--name is required unless --check")
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text("utf-8"))["syndrone"][
        "splits"
    ]
    towns = args.towns or sp["development"]
    bad = [t for t in towns if t in sp["test"]]
    if bad:
        print(f"test towns are not replayed: {bad}")
        return 2
    cfg = load_config(args.config)
    cam = syndrone_camera()
    ov = parse_sets(args.sets)
    ov.setdefault("camera", {}).update({"focal_px": cam.fx, "cx": cam.cx, "cy": cam.cy})

    def get(k: str) -> Any:
        base = cfg.require(k)
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
            new = replay(tracks, flight_meta(store, town, h), RangeSpeedStage.from_config(get))
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
                    {"source": args.source, "config": args.config, "overrides": ov}, indent=1
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
