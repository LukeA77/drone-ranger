#!/usr/bin/env bash
# ROS 2 check with no datasets or model weights (run in CI, inside the ROS 2 image):
#
#   docker run --rm drone-ranger-ros:amd64 bash scripts/ros2_demo_check.sh [out dir]
#
# 1. Reference: `aerialrange run` on the built-in synthetic flight ($DRONE_RANGER_DEMO).
# 2. launch_testing: the same flight as a rosbag2 recording, played into the node, with the
#    camera mount reaching it through tf2; every frame must be processed and the node's
#    tracks.csv must equal the reference (test/test_integration.launch.py).
# 3. demo.launch.py without a display: it must run to the end by itself, process every frame
#    and give the same tracks again.
# RATE (default 0.5) is the playback rate in step 2.
set -euo pipefail
D=${DRONE_RANGER_DEMO:-/app/demo}
OUT=${1:-/tmp/ros2_check}
rm -rf "$OUT"; mkdir -p "$OUT"
cd /app
read -r CAM MOUNT N < <(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))
print(m["camera"], m["mount_tilt_deg"], m["frames"])' "$D/demo.json")

echo "1/3 reference: aerialrange run on the synthetic flight ($N frames)"
aerialrange run --source "$D/flight/frames" --metadata "$D/flight/metadata.csv" \
  --boxes "$D/flight/boxes.csv" --camera "$CAM" --out "$OUT/ref" > "$OUT/ref.log" 2>&1

echo "2/3 launch_testing: the recording through the node, compared with the reference"
DR_BAG="$D/bag" DR_REF="$OUT/ref/tracks.csv" DR_OUT="$OUT/node" DR_MOUNT="$MOUNT" \
DR_CONFIG=run DR_BOXES="$D/flight/boxes.csv" DR_RATE="${RATE:-0.5}" DR_FRAMES="$N" \
  launch_test /ws/src/drone_ranger_ros/test/test_integration.launch.py \
  > "$OUT/launch_test.log" 2>&1 || { tail -n 60 "$OUT/launch_test.log"; exit 1; }
grep -E "^(Ran|OK|FAILED)" "$OUT/launch_test.log" || true

echo "3/3 demo.launch.py, no display"
timeout 600 ros2 launch drone_ranger_ros demo.launch.py rviz:=false output_dir:="$OUT/demo" \
  > "$OUT/demo.log" 2>&1 || { tail -n 60 "$OUT/demo.log"; exit 1; }
python3 - "$OUT" "$N" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out, n = Path(sys.argv[1]), int(sys.argv[2])
ref, got = pd.read_csv(out / "ref/tracks.csv"), pd.read_csv(out / "demo/tracks.csv")
frames = len(pd.read_csv(out / "demo/flight.csv"))
assert frames == n, f"demo processed {frames} of {n} frames"
assert list(got.columns) == list(ref.columns) and len(got) == len(ref), "row count differs"
num = [c for c in ref.columns if ref[c].dtype.kind in "fi"]
same = all(np.allclose(got[c], ref[c], rtol=1e-6, atol=1e-6, equal_nan=True) for c in num)
assert same and (got["track_id"] == ref["track_id"]).all(), "demo tracks differ"
print(f"demo: {frames} frames, {len(got)} rows, equal to aerialrange run within 1e-6")
PY
echo "ROS 2 check passed"
