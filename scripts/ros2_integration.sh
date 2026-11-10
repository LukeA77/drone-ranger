#!/usr/bin/env bash
# ROS 2 integration test (Phase 12, build2 11.2): a recorded flight goes in as a rosbag2
# recording, and the node's tracked objects must equal `aerialrange run` on the same frames.
#
#   bash scripts/ros2_integration.sh <name> <frames dir> <flight CSV> <origin lat,lon,h> [frames]
#
# <origin> is the geodetic position of the ground point at local north 0, east 0 (for an AU-AIR
# clip: its first fix, lowered by that frame's height; scripts/ros2_inputs.py prints it).
# Builds docker/ros2.Dockerfile, then inside the container: the reference run (aerialrange run),
# the recording (make_bag), and the launch test. Outputs in cache/ros2/<name>/.
# CAMERA=focal,cx,cy replaces the config's camera for both (the recording's CameraInfo).
# PLATFORM=linux/arm64 builds and runs the ARM64 image (emulated on an x86 laptop: slow, so
# use a lower RATE; it shows the node works on ARM64, not how fast).
# RATE rule: the gap between frames in playback (frame spacing / RATE) must exceed the node's
# time per frame, or its input queue (10 frames) overflows and frames are lost. Emulated ARM64
# takes about 20 s a frame; AU-AIR frames are about 0.3 s apart, so RATE=0.01 (31 s).
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1; FRAMES=$2; FLIGHT=$3; ORIGIN=$4; N=${5:-100}
CONFIG=${CONFIG:-run}; RATE=${RATE:-0.2}; PLATFORM=${PLATFORM:-linux/amd64}
IMAGE=${IMAGE:-drone-ranger-ros:${PLATFORM#linux/}}
OUT="cache/ros2/$NAME"
rm -rf "$OUT"; mkdir -p "$OUT"
docker buildx build --platform "$PLATFORM" --load -f docker/ros2.Dockerfile -t "$IMAGE" . \
  > "$OUT/build.log" 2>&1
abs() { (cd "$(dirname "$1")" && printf '%s/%s' "$(pwd -W 2>/dev/null || pwd)" "$(basename "$1")"); }
docker run --rm --platform "$PLATFORM" \
  -v "$(abs weights):/app/weights:ro" \
  -v "$(abs "$FRAMES"):/data/frames:ro" \
  -v "$(abs "$FLIGHT"):/data/flight.csv:ro" \
  -v "$(abs "$OUT"):/out" \
  -e CONFIG="$CONFIG" -e N="$N" -e ORIGIN="$ORIGIN" -e RATE="$RATE" -e CAMERA="${CAMERA:-}" \
  "$IMAGE" bash -c '
set -euo pipefail
cd /app
CAM=${CAMERA:-$(python3 scripts/ros2_inputs.py camera --config "$CONFIG")}
aerialrange run --source /data/frames --metadata /data/flight.csv --config "$CONFIG" \
  --camera "$CAM" --max-frames "$N" --out /out/ref > /out/ref.log 2>&1
ros2 run drone_ranger_ros make_bag --frames /data/frames --flight /data/flight.csv \
  --origin "$ORIGIN" --camera "$CAM" --mount-tilt-deg 0 --out /out/bag --max-frames "$N" \
  > /out/bag.log 2>&1
DR_BAG=/out/bag DR_REF=/out/ref/tracks.csv DR_OUT=/out/node DR_MOUNT=0 DR_CONFIG="$CONFIG" \
DR_RATE="$RATE" DR_FRAMES="$N" \
  launch_test /ws/src/drone_ranger_ros/test/test_integration.launch.py > /out/launch_test.log 2>&1
echo "launch_test exit=$?"
cp /app/requirements-ros-freeze.txt /out/
' | tee "$OUT/result.txt"
