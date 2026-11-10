"""Export the first frames of a SynDrone sequence as a frame folder and flight-data CSV, the
form `aerialrange run` and the ROS 2 recording converter read (Phase 12 integration test).

Usage: python scripts/syndrone_export_frames.py --town Town01 --height 50 --frames 60

Frames are written as PNG (lossless, so every reader sees the same pixels), and the flight
data as scripts/syndrone_run.py builds it (Unreal x, y as north, east; heading; tilt; height
above the floor). Development and calibration towns only. Prints the camera (focal,cx,cy) to
pass as CAMERA to scripts/ros2_integration.sh. Writes cache/ros2_inputs/<town>_h<h>/.
"""

from __future__ import annotations

import argparse
import json
import math

import cv2
import pandas as pd

from aerialrange.data.syndrone import FPS, SynDroneSequence, decode_rgb, syndrone_camera
from aerialrange.data.syndrone_archive import MemberStore
from aerialrange.utils.config import REPO_ROOT, load_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--town", required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()
    splits = json.loads((REPO_ROOT / "results" / "splits.json").read_text(encoding="utf-8"))
    if args.town in splits["syndrone"]["splits"]["test"]:
        print(f"{args.town} is a test town")
        return 2
    d = load_config("data")
    store = MemberStore(
        REPO_ROOT / d.require("data_root") / d.require("syndrone_subdir"),
        d.require("syndrone_base_url"),
    )
    seq = SynDroneSequence(store, args.town, args.height)
    recs = seq.camera_records(range(args.frames))
    out = REPO_ROOT / "cache" / "ros2_inputs" / f"{args.town}_h{args.height}"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    names = [seq.member("rgb", k) for k in range(args.frames)]
    got = dict(store.iter_read(names))
    for k, n in enumerate(names, start=1):
        cv2.imwrite(str(out / "frames" / f"{k:06d}.png"), decode_rgb(got[n]))
    pd.DataFrame(
        {
            "frame": range(1, args.frames + 1),
            "t_s": [k / FPS for k in range(args.frames)],
            "n_m": [r.x for r in recs],
            "e_m": [r.y for r in recs],
            "h_m": [r.h for r in recs],
            "yaw_rad": [math.radians(r.yaw) for r in recs],
            "tilt_deg": [-r.pitch for r in recs],
        }
    ).to_csv(out / "metadata.csv", index=False)
    cam = syndrone_camera()
    print(f"{cam.fx},{cam.cx},{cam.cy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
