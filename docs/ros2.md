# ROS 2 node (Phase 12)

Drone Ranger runs as a ROS 2 node, so it can sit on a drone's onboard computer next to its autopilot and camera drivers. It takes the camera, the drone's odometry and GPS, and publishes every tracked person and vehicle with a ground position, a range and their error bars.

**Platform:** ROS 2 Jazzy Jalisco on Ubuntu 24.04, in Docker (`docker/ros2.Dockerfile`), for x86-64 and ARM64 (Raspberry Pi 5).

## Topics

| Direction | Topic | Type | Notes |
|---|---|---|---|
| in | `image` | `sensor_msgs/Image` | bgr8, rgb8 or mono8; sensor-data quality of service |
| in | `camera_info` | `sensor_msgs/CameraInfo` | the pinhole intrinsics replace the config's camera |
| in | `odometry` | `nav_msgs/Odometry` | body (forward, left, up) in a world frame (east, north, up) |
| in | `fix` | `sensor_msgs/NavSatFix` | anchors the map; the first fix sets the map's origin |
| out | `detections` | `vision_msgs/Detection2DArray` | the detector's raw boxes |
| out | `tracked_objects` | `drone_ranger_msgs/TrackedObjectArray` | see below |
| out | `markers` | `visualization_msgs/MarkerArray` | error ellipses and labels for RViz |

The four inputs are matched by time stamp (`message_filters`, approximate time, 0.1 s). At shutdown the node writes `tracks.csv` (the same columns as `aerialrange run`), `map.geojson` and `map.html` (docs/map.md), and `timing.json` into `output_dir`.

Each `TrackedObject` carries:
- the track number and class (people are a class and a number only);
- the image box;
- the ground position and its 2 × 2 covariance;
- the range with its sigma and calibrated 68 / 90 / 95 % intervals;
- whether a range was found and why not;
- the reliability flag;
- a velocity with covariance, only where the display rule shows the speed.

## Frames

ROS conventions (REP 103, REP 105) are east, north, up for the world and forward, left, up for the body. The pipeline works in north, east, down. All conversions are in `aerialrange/ros_bridge.py` and pinned by `tests/test_ros_bridge.py`.

The camera's heading and tilt come from the odometry's attitude and the camera's fixed mount (`mount_tilt_deg`, required: the tilt below the body's horizontal plane). The pipeline's camera model has no roll, as in `aerialrange run`. A gimballed camera needs odometry of the camera itself.

## Running

```bash
docker build -f docker/ros2.Dockerfile -t drone-ranger-ros:amd64 .
docker run --rm -it -v "$PWD/weights:/app/weights:ro" drone-ranger-ros:amd64 \
  ros2 launch drone_ranger_ros drone_ranger.launch.py mount_tilt_deg:=30 output_dir:=/tmp/out \
  image:=/camera/image_raw camera_info:=/camera/camera_info odometry:=/odom fix:=/gps/fix
```

Settings come from the repository's `configs/` (the `config` launch argument, `run` by default).

## Integration test

`bash scripts/ros2_integration.sh <name> <frames dir> <flight CSV> <origin> [frames]`:

1. Builds the image.
2. Runs `aerialrange run` on the frames, as the reference.
3. Converts the same frames and flight data into a rosbag2 recording (`make_bag`).
4. Plays the recording into the node under `launch_testing`.
5. Checks that the node's output equals the reference: identical track numbers, classes and flags, and every number within 1e-6.

Results are in "Results" below.

## Results

Run on the laptop (2026-11-10, `cache/ros2/<name>/`). All pass: every frame processed, and the node's output equal to `aerialrange run`'s in the same image.

| Recording | Frames | Rows compared | Result |
|---|---|---|---|
| AU-AIR clip 35 (real flight, D-FINE N) | 318 | 1,484 | identical within 1e-6 |
| SynDrone Town01, 50 m (simulated, its own camera) | 60 | 190 | identical within 1e-6 |
| AU-AIR clip 35, first 40 frames, **ARM64 image** (emulated, rate 0.01) | 40 | 86 | identical within 1e-6 |

The ARM64 run shows that the node builds and gives the same results on ARM64. It says nothing about speed: under emulation a frame takes about 9 s (median), which is the emulator, not the Pi.

**Latency on the laptop** (AU-AIR clip 35, medians, 95th percentile in brackets): the pipeline alone takes 106 ms (258 ms), and the node's whole callback, from the synchronised messages' arrival to the last publish, 120 ms (287 ms). So ROS 2 adds about 14 ms per frame on top of the pipeline: image conversion, building and publishing the messages.

**Raspberry Pi 5: estimated, not measured.** Scaled by the laptop-to-Pi ratio of `results/edge/pi_estimate.json` (2.72, range ×0.75 to ×1.5), that 14 ms would be about 38 ms (28 to 57 ms). The ratio was measured on neural-network inference, not on memory copies and message serialisation, so this figure is weaker than the pipeline's own estimate. It will be replaced when a Pi 5 is measured.

Four things the tests found, all fixed:
- **Dropped frames.** With best-effort delivery (ROS's default for cameras), 25 of 40 full-HD frames were lost. That is correct for a live camera, where a late frame is useless, so it stays the default. The `image_qos:=reliable` option delivers every frame of a recording.
- **Timestamp precision.** A time stamp near 1.7 billion seconds, held as one floating-point number, loses about 0.2 µs, which changed speeds by up to 5 parts per million. The node and the converter now work in whole nanoseconds.
- **Start-up race.** Under ARM64 emulation the node takes about a minute to start, and frames played before it subscribed were lost (38 of 40 arrived). The node now logs `subscribed: waiting for frames`, and the test starts playback on that line instead of after a fixed delay. A launch file on a drone can do the same.
- **Input queue overflow.** If frames arrive faster than the node processes them, its queue (10 frames) overflows and the oldest are lost (27 of 40 at rate 0.05 under emulation). On a live camera that is the intended behaviour. For a recording, the gap between frames during playback must be longer than the node's time per frame (`scripts/ros2_integration.sh` explains the choice of `RATE`).

## Limits

- ARM64 has been run only under emulation, on 40 frames; speed on a Raspberry Pi 5 is not measured.
- A camera on a gimbal needs odometry of the camera itself; the node takes one fixed mount angle.
- The image keeps Ubuntu's numpy 1.26, which ROS needs, so the libraries under the pipeline are older than in the laptop environment (`/app/requirements-ros-freeze.txt`). The comparison runs both sides in the same image; whether results differ between the two environments has not been checked.
