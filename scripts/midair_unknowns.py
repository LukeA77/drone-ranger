"""Run the Mid-Air unknowns checks and write results/midair/unknowns.json.

Usage: python scripts/midair_unknowns.py --climate Kite_training/sunny --trajectories 0 1 2
       [--stride 100] [--min-pixels 2000] [--max-pixels 5000]

Only trajectories whose depth, segmentation and normals folders are fully extracted are used.
Sampling settings are exploratory (they choose which frames to look at, not model parameters)
and are recorded in the output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime

import numpy as np

from aerialrange.evaluate import midair_unknowns as unk
from aerialrange.utils.config import REPO_ROOT, load_config

OUT = REPO_ROOT / "results" / "midair" / "unknowns.json"


def _summary(rows: list[dict], key: str) -> dict[str, float]:
    v = np.array([r[key] for r in rows])
    return (
        {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}
        if len(v)
        else {}
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--climate", default="Kite_training/sunny")
    ap.add_argument("--trajectories", type=int, nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--min-pixels", type=int, default=2000)
    ap.add_argument("--max-pixels", type=int, default=5000)
    args = ap.parse_args(argv)

    data = load_config("data")
    seed = load_config("project").require("seed")
    root = REPO_ROOT / data.require("data_root") / data.require("midair_subdir") / args.climate
    h5 = root / "sensor_records.hdf5"
    names = []
    for t in args.trajectories:
        name = f"trajectory_{t:04d}"
        if all(
            (root / m / name / ".frames.done").exists()
            for m in ("depth", "segmentation", "normals")
        ):
            names.append(name)
        else:
            print(f"skip {name}: not fully downloaded")

    quat, water, normals, sky, mapping = {}, [], [], [], {}
    for name in names:
        print(f"{name} ...", flush=True)
        quat[name] = unk.quaternion_check(h5, name)
        mapping[name] = unk.frame_mapping_check(h5, name)
        water += unk.water_plane_check(
            root, name, args.stride, args.min_pixels, args.max_pixels, seed
        )
        normals += unk.normals_check(root, name, args.stride, args.min_pixels // 4)
        sky.append(unk.sky_check(root, name, args.stride))

    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            ).stdout.strip()
            or None
        )
    except OSError:
        commit = None
    result = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "seed": seed,
        "climate": args.climate,
        "trajectories": names,
        "settings": {
            "stride": args.stride,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
        "quaternion_rms_ms2": quat,
        "frame_mapping": mapping,
        "water_plane": {
            "n_frames": len(water),
            "planar_rms_m": _summary(water, "planar_rms_m"),
            "radial_rms_m": _summary(water, "radial_rms_m"),
            "planar_tilt_deg": _summary(water, "planar_tilt_deg"),
            "radial_tilt_deg": _summary(water, "radial_tilt_deg"),
            "radial_better_fraction": float(
                np.mean([r["radial_rms_m"] < r["planar_rms_m"] for r in water])
            )
            if water
            else None,
            "frames": water,
        },
        "normals_water": {
            "n_frames": len(normals),
            "mean_normal_tilt_deg": _summary(normals, "mean_normal_tilt_deg"),
            "median_tilt_deg": _summary(normals, "median_tilt_deg"),
            "frames": normals,
        },
        "sky": sky,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    brief = {k: result[k] for k in ("quaternion_rms_ms2", "frame_mapping", "sky")}
    brief["water_plane"] = {k: v for k, v in result["water_plane"].items() if k != "frames"}
    brief["normals_water"] = {k: v for k, v in result["normals_water"].items() if k != "frames"}
    print(json.dumps(brief, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
