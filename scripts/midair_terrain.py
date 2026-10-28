"""Run terrain-from-motion ranging (plan D) on a Mid-Air split.

Usage:
  python scripts/midair_terrain.py --split development
  python scripts/midair_terrain.py --set-sigma-from-measurement   (records TERRAIN_SIGMA_PX)

Needs the split's evaluation cache (scripts/midair_eval.py) and colour images. Estimates are
written per trajectory to cache/midair/<trajectory>.terrain.npz, in the same sample order as
the evaluation cache. The test split needs --final.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np

from aerialrange.data.midair import MidAirTrajectory
from aerialrange.evaluate.terrain_motion import MotionSettings, evaluate_trajectory_motion
from aerialrange.utils.config import CONFIG_DIR, REPO_ROOT, load_config

CACHE = REPO_ROOT / "cache" / "midair"
NOISE = REPO_ROOT / "results" / "midair" / "development" / "tracking_noise.json"


def motion_settings(require_sigma: bool = True) -> MotionSettings:
    """Terrain settings from configs/geometry.yaml."""
    geo = load_config("geometry")
    v = dict(geo.require("terrain_motion"))
    gaps = tuple(v.pop("frame_gaps"))
    sig = geo.require("terrain_sigma_px_by_gap") if require_sigma else None
    sig_t = tuple(float(sig[str(g)]) for g in gaps) if sig else tuple(np.nan for _ in gaps)
    return MotionSettings(frame_gaps=gaps, sigma_px_by_gap=sig_t, **v)


def set_sigma_from_measurement() -> int:
    """Record the measured robust tracking sigma per gap (mean of u and v) in the config."""
    m = json.loads(NOISE.read_text())["by_gap"]
    sig = {g: round((v["robust_sigma_u_px"] + v["robust_sigma_v_px"]) / 2, 3) for g, v in m.items()}
    p = CONFIG_DIR / "geometry.yaml"
    t = p.read_text()
    new = (
        "terrain_sigma_px_by_gap:\n  value: " + json.dumps(sig) + "\n  placeholder: "
        'TERRAIN_SIGMA_PX\n  source: "Measured: results/midair/development/tracking_noise'
        ".json, robust sigma (mean of u and v) per gap; core noise only, tails are left to "
        'calibration (D-402)"'
    )
    t2 = re.sub(
        r"terrain_sigma_px_by_gap:\n  value: .*?\n  placeholder: TERRAIN_SIGMA_PX\n"
        r"  source: .*?(\n|$)",
        new + "\n",
        t,
        flags=re.S,
    )
    p.write_text(t2)
    print("set", sig)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["development", "calibration", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute existing results")
    ap.add_argument("--set-sigma-from-measurement", action="store_true")
    args = ap.parse_args(argv)
    if args.set_sigma_from_measurement:
        return set_sigma_from_measurement()
    if args.split is None:
        ap.error("--split is required")
    if args.split == "test" and not args.final:
        print("The test split runs once, after all settings are frozen. Use --final.")
        return 1
    data = load_config("data")
    root = (
        REPO_ROOT
        / data.require("data_root")
        / data.require("midair_subdir")
        / data.require("midair_climate")
    )
    s = motion_settings()
    names = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["midair"]["splits"][
        args.split
    ]
    for name in names:
        cache = CACHE / f"{name}.npz"
        if not cache.exists():
            print(f"{name}: no evaluation cache, skipped")
            continue
        if (CACHE / f"{name}.terrain.npz").exists() and not args.force:
            print(f"{name}: already done (use --force to recompute)")
            continue
        if not (root / "color_left" / name / ".frames.done").exists():
            print(f"{name}: no colour images, skipped")
            continue
        samples = dict(np.load(cache))
        traj = MidAirTrajectory(root, name)
        try:
            res = evaluate_trajectory_motion(traj, samples, s)
        finally:
            traj.close()
        np.savez_compressed(CACHE / f"{name}.terrain.npz", **res)
        ok = res["terrain.valid"]
        print(f"{name}: {ok.mean():.3f} valid of {len(ok)} samples", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
