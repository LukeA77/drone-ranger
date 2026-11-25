#!/usr/bin/env bash
# Record RViz while demo.launch.py plays a recording into the node, on a virtual display inside
# the RViz image (docker/ros2-viz.Dockerfile), so no desktop or GPU is needed.
#
#   bash scripts/record_rviz.sh <out.mp4> [bag folder] [mount_tilt_deg]
#
# With no bag: the built-in synthetic flight. With a bag (make_bag's output, e.g.
# cache/ros2/<name>/bag): that recording, with the detector of configs/run.yaml (weights/ is
# mounted read-only). RATE (default 1.0) is the playback rate; FPS (default 15) the video's.
# The README loop is cut from the MP4 by scripts/make_readme_media.py.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$1; BAG=${2:-}; MOUNT=${3:-}
RATE=${RATE:-1.0}; FPS=${FPS:-15}; SIZE=1280x720
# the capture leaves out RViz's toolbar (top 62 px) and status bar (bottom 32 px)
GRAB=1280x626; GRAB_AT=0,62
docker build -q -f docker/ros2.Dockerfile -t drone-ranger-ros:amd64 . > /dev/null
docker build -q -f docker/ros2-viz.Dockerfile -t drone-ranger-ros-viz:amd64 . > /dev/null
abs() { (cd "$(dirname "$1")" && printf '%s/%s' "$(pwd -W 2>/dev/null || pwd)" "$(basename "$1")"); }
mkdir -p "$(dirname "$OUT")"
ARGS=(); MOUNTS=(-v "$(abs weights):/app/weights:ro" -v "$(abs "$(dirname "$OUT")"):/out")
if [ -n "$BAG" ]; then
  [ -n "$MOUNT" ] || { echo "a bag needs its mount_tilt_deg (as given to make_bag)"; exit 1; }
  MOUNTS+=(-v "$(abs "$BAG"):/data/bag:ro")
  ARGS=(bag:=/data/bag mount_tilt_deg:="$MOUNT")
fi
docker run --rm "${MOUNTS[@]}" -e LIBGL_ALWAYS_SOFTWARE=1 -e QT_QPA_PLATFORM=xcb \
  drone-ranger-ros-viz:amd64 bash -c '
set -euo pipefail
Xvfb :99 -screen 0 '"$SIZE"'x24 +extension GLX -nolisten tcp > /tmp/xvfb.log 2>&1 &
export DISPLAY=:99
for _ in $(seq 50); do xdpyinfo > /dev/null 2>&1 && break; sleep 0.2; done
# RViz needs a few seconds to open: the recording starts start_delay_s after the node is ready
ros2 launch drone_ranger_ros demo.launch.py '"${ARGS[*]:-}"' rate:='"$RATE"' \
  start_delay_s:=8 exit_at_end:=true output_dir:=/tmp/demo_out > /out/record_rviz.log 2>&1 &
LAUNCH=$!
sleep 6
ffmpeg -loglevel error -y -f x11grab -draw_mouse 0 -video_size '"$GRAB"' -framerate '"$FPS"' -i :99.0+'"$GRAB_AT"' \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p /out/'"$(basename "$OUT")"' &
FF=$!
wait $LAUNCH || true
kill -INT $FF; wait $FF || true
grep -E "wrote|demo\]" /out/record_rviz.log || true
'
echo "wrote $OUT"
