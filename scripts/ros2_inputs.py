"""Inputs for the ROS 2 integration test (Phase 12), printed for scripts/ros2_integration.sh.

Usage: python scripts/ros2_inputs.py camera [--config run]   -> focal,cx,cy
       python scripts/ros2_inputs.py auair-origin --clip 35   -> lat,lon,height

auair-origin: the geodetic position of the ground point at the clip's local north 0, east 0:
its first GPS fix (cache/auair/frames.csv, the origin `data/auair.py` uses), lowered by that
frame's height above the ground (the clip's metadata.csv).
"""

from __future__ import annotations

import argparse

import pandas as pd

from aerialrange.utils.config import REPO_ROOT, load_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("camera", "auair-origin"))
    ap.add_argument("--config", default="run")
    ap.add_argument("--clip", type=int, default=None)
    args = ap.parse_args()
    if args.what == "camera":
        c = load_config(args.config).require("camera")
        print(f"{c['focal_px']},{c['cx']},{c['cy']}")
        return 0
    frames = pd.read_csv(REPO_ROOT / "cache" / "auair" / "frames.csv")
    first = frames[frames["clip"] == args.clip].sort_values("t_ms").iloc[0]
    meta = pd.read_csv(
        REPO_ROOT / "cache" / "auair" / "flights" / f"clip_{args.clip:02d}" / "metadata.csv"
    )
    h0 = float(meta.sort_values("frame").iloc[0]["h_m"])
    print(f"{first['lat_deg']:.9f},{first['lon_deg']:.9f},{first['alt_m'] - h0:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
