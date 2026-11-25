"""Build the ROS 2 demo recording: a synthetic flight (no datasets, no model weights) as a
rosbag2 recording, for `ros2 launch drone_ranger_ros demo.launch.py` and the ROS 2 check in CI.

Run inside the ROS 2 image (docker/ros2.Dockerfile does this at build time):
  python3 scripts/ros2_demo_data.py --out /app/demo

Writes <out>/flight/ (frames, metadata.csv, boxes.csv, truth.csv from
`aerialrange.pipeline.synthetic.make_flight`, the scene of `aerialrange demo`), <out>/bag (the
recording, from make_bag) and <out>/demo.json (what demo.launch.py needs). The boxes carry the
detector's measured box noise, as in `aerialrange demo`. The camera is mounted at the flight's
tilt, so the body flies level. The flight has no real location: its map origin is latitude 0,
longitude 0.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from aerialrange.pipeline.synthetic import make_flight
from aerialrange.utils.config import load_config

FRAMES = 80  # 16 s at the scene's 5 frames a second: the three cars pass through the view
ORIGIN = "0.0,0.0,0.0"  # synthetic: no real place


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=FRAMES)
    args = ap.parse_args()
    km = load_config("run").require("kalman")
    ms = km.get("meas_sigma") if km.get("meas_model") == "propagated" else None
    noise = (ms["kappa_u"]["vehicle"], ms["kappa_v"]["vehicle"]) if ms else None
    scene = make_flight(args.out / "flight", n_frames=args.frames, box_noise=noise)
    flight = pd.read_csv(args.out / "flight" / "metadata.csv")
    if flight["tilt_deg"].nunique() != 1 or flight["h_m"].nunique() != 1:
        raise SystemExit("the demo mount assumes a constant camera tilt and height")
    tilt, height = float(flight["tilt_deg"].iloc[0]), float(flight["h_m"].iloc[0])
    camera = f"{scene['focal_px']},{scene['cx']},{scene['cy']}"
    subprocess.run(
        ["ros2", "run", "drone_ranger_ros", "make_bag",
         "--frames", str(args.out / "flight" / "frames"),
         "--flight", str(args.out / "flight" / "metadata.csv"),
         "--origin", ORIGIN, "--camera", camera, "--mount-tilt-deg", str(tilt),
         "--out", str(args.out / "bag")],
        check=True,
    )  # fmt: skip
    meta = {
        "bag": "bag",
        "boxes_csv": "flight/boxes.csv",
        "truth_csv": "flight/truth.csv",
        "metadata_csv": "flight/metadata.csv",
        "frames": args.frames,
        "camera": camera,
        "mount_tilt_deg": tilt,
        "origin": ORIGIN,
        "description": (
            f"SYNTHETIC flight, not real footage: a camera {height:g} m up, tilted "
            f"{tilt:g} deg down, flies north over flat ground past three cars (one parked, two "
            f"driving), {args.frames} frames. Boxes come from the scene, so no detector or "
            "model weights are used; the cars' true positions are known."
        ),
    }
    (args.out / "demo.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(meta["description"])


if __name__ == "__main__":
    main()
