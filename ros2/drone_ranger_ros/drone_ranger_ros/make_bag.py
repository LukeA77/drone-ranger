"""Convert a recorded flight (frames and the pipeline's flight-data CSV) into a rosbag2
recording for the node (Phase 12 test input).

Usage (inside the ROS 2 container):
  ros2 run drone_ranger_ros make_bag --frames <folder of .jpg/.png> --flight <flight CSV>
      --origin <lat>,<lon>,<height> --camera <focal>,<cx>,<cy> --mount-tilt-deg 0
      --out <bag folder> [--max-frames N]

Each frame k becomes four messages with the same stamp (start + t_s): image (bgr8),
camera_info (pinhole K from --camera, image size from the frame), odometry (the camera pose of
the flight CSV as a body pose in ENU, with the camera mounted --mount-tilt-deg below the body:
`aerialrange.ros_bridge.odometry_from_camera_pose`, so the node recovers the same pose) and
fix (the drone's latitude, longitude and height: its local north, east and height above the
ground converted from --origin, the geodetic position of the ground point at local 0, 0).
Frames are read in name order, as `aerialrange run` reads a folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
import rosbag2_py
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CameraInfo, Image, NavSatFix

from aerialrange import ros_bridge as rb
from aerialrange.geometry.frames import ned_to_geodetic

START_S = 1_700_000_000  # an arbitrary positive start time for the recording's stamps
TOPICS = {
    "/image": "sensor_msgs/msg/Image",
    "/camera_info": "sensor_msgs/msg/CameraInfo",
    "/odometry": "nav_msgs/msg/Odometry",
    "/fix": "sensor_msgs/msg/NavSatFix",
}


def stamp(t: float) -> Time:
    # whole nanoseconds: START_S + t in floating point would lose about 0.2 us at this size
    ns = START_S * 1_000_000_000 + int(round(t * 1e9))
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--flight", type=Path, required=True)
    ap.add_argument("--origin", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--mount-tilt-deg", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--frame-id", default="camera")
    args = ap.parse_args()
    lat0, lon0, h0 = (float(x) for x in args.origin.split(","))
    f, cx, cy = (float(x) for x in args.camera.split(","))
    meta = pd.read_csv(args.flight).set_index("frame")
    paths = sorted(list(args.frames.glob("*.jpg")) + list(args.frames.glob("*.png")))
    if args.max_frames is not None:
        paths = paths[: args.max_frames]
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(args.out), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for i, (name, typ) in enumerate(TOPICS.items()):
        try:  # Jazzy's TopicMetadata takes an id first; earlier releases do not
            meta_t = rosbag2_py.TopicMetadata(id=i, name=name, type=typ, serialization_format="cdr")
        except TypeError:
            meta_t = rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr")
        writer.create_topic(meta_t)
    for k, p in enumerate(paths, start=1):
        m = meta.loc[k]
        st = stamp(float(m["t_s"]))
        t_ns = st.sec * 1_000_000_000 + st.nanosec
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        im = Image(height=h, width=w, encoding="bgr8", is_bigendian=0, step=w * 3)
        im.header.stamp, im.header.frame_id = st, args.frame_id
        im.data = img.tobytes()
        ci = CameraInfo(height=h, width=w, distortion_model="plumb_bob")
        ci.header = im.header
        ci.k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
        ci.p = [f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.d = [0.0] * 5
        pos, q = rb.odometry_from_camera_pose(
            float(m["n_m"]),
            float(m["e_m"]),
            float(m["h_m"]),
            float(m["yaw_rad"]),
            float(m["tilt_deg"]),
            args.mount_tilt_deg,
        )
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = st, "map", "base_link"
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = pos
        o = od.pose.pose.orientation
        o.x, o.y, o.z, o.w = (float(x) for x in q)
        lat, lon, alt = ned_to_geodetic(
            [float(m["n_m"]), float(m["e_m"]), -float(m["h_m"])], lat0, lon0, h0
        )
        fx = NavSatFix(latitude=float(lat), longitude=float(lon), altitude=float(alt))
        fx.header.stamp, fx.header.frame_id = st, "gps"
        fx.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        for topic, msg in (("/image", im), ("/camera_info", ci), ("/odometry", od), ("/fix", fx)):
            writer.write(topic, serialize_message(msg), t_ns)
    del writer
    print(f"wrote {args.out}: {len(paths)} frames")


if __name__ == "__main__":
    main()
