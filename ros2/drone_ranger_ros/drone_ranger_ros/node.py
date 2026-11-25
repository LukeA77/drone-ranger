"""Drone Ranger ROS 2 node (Phase 12).

Subscribes (synchronised by time stamp with message_filters):
  image (sensor_msgs/Image, sensor-data QoS), camera_info (sensor_msgs/CameraInfo),
  odometry (nav_msgs/Odometry: body FLU in an ENU world, REP 103 / REP 105), fix
  (sensor_msgs/NavSatFix).
Publishes:
  detections (vision_msgs/Detection2DArray): the detector's raw boxes, on detector frames;
  tracked_objects (drone_ranger_msgs/TrackedObjectArray): every tracked object with its ground
    position and covariance, range with interval, validity and reason, reliability flag, and
    velocity with covariance where the display rule shows the speed;
  markers (visualization_msgs/MarkerArray): the same for RViz (an error ellipse, a label with
    the range, and a line from the camera to the object);
  and, at shutdown, tracks.csv (as `aerialrange run` writes it) and the GeoJSON map layer and
  viewer, in the `output_dir` parameter's folder.

The camera's pose comes from tf2 and the odometry together: the body's pose is the odometry
(matched to the image by time stamp), and the camera's pose on the body is the tf2 transform
from the odometry's child frame (base_link) to the image's frame (camera_optical_frame), looked
up at the image's time stamp, so a fixed mount (a static transform) and a gimbal (a moving one)
both work. drone_ranger.launch.py publishes a fixed mount from its mount_tilt_deg argument.

All processing is `aerialrange.pipeline.run.FramePipeline`, the same object `aerialrange run`
uses, with settings from the repository's configs/ (`config` parameter) and the camera
intrinsics from camera_info; all convention changes are in `aerialrange.ros_bridge`.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import message_filters
import numpy as np
import pandas as pd
import rclpy
import tf2_ros
from drone_ranger_msgs.msg import TrackedObject, TrackedObjectArray
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, NavSatFix
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from aerialrange import ros_bridge as rb
from aerialrange.geometry.frames import ned_to_geodetic
from aerialrange.pipeline.geomap import ellipse_axes, write_map_from_frames
from aerialrange.pipeline.ranging import CameraPose
from aerialrange.pipeline.run import FramePipeline

# message_filters' Python tutorial (docs.ros.org, message_filters, Jazzy):
# ApproximateTimeSynchronizer([...], 10, 0.1): queue of 10 messages, 0.1 s slop
SYNC_QUEUE, SYNC_SLOP_S = 10, 0.1
ENCODINGS = {"bgr8": (3, False), "rgb8": (3, True), "mono8": (1, False)}


def image_to_bgr(msg: Image) -> np.ndarray:
    """BGR array of a sensor_msgs/Image (bgr8, rgb8 or mono8), without cv_bridge."""
    if msg.encoding not in ENCODINGS:
        raise ValueError(f"unsupported image encoding {msg.encoding}")
    ch, rgb = ENCODINGS[msg.encoding]
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    a = a[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
    if ch == 1:
        return np.repeat(a, 3, axis=2)
    return a[:, :, ::-1].copy() if rgb else a.copy()


def _f(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return v


class DroneRangerNode(Node):
    def __init__(self) -> None:
        super().__init__("drone_ranger")
        self.declare_parameter("config", "run")
        # tf2 frames of the camera and the body; empty: the image's and the odometry's frame_id
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("body_frame", "")
        self.declare_parameter("tf_timeout_s", 1.0)
        # a CSV of boxes (frame, left, top, width, height, score, cls) read instead of running
        # the detector, by frame number in arrival order: the synthetic demo needs no weights
        self.declare_parameter("boxes_csv", "")
        self.declare_parameter("ground_up_m", 0.0)  # flat ground height in the odometry frame
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("sync_slop_s", SYNC_SLOP_S)
        # "sensor_data" (best effort, the ROS default for camera streams: a late frame is
        # dropped), "reliable" (every frame from a reliable publisher), or "recording" (every
        # frame of a recording played with config/recording_qos.yaml: all four inputs reliable
        # and transient local, so frames published before discovery finished still arrive)
        self.declare_parameter("image_qos", "sensor_data")
        self.declare_parameter("output_dir", "")  # tracks.csv and the map at shutdown
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.body_frame = str(self.get_parameter("body_frame").value)
        self.tf_timeout = Duration(seconds=float(self.get_parameter("tf_timeout_s").value))
        self.boxes_csv = str(self.get_parameter("boxes_csv").value)
        # tf2's subscriptions are in their own reentrant callback group, so under main()'s
        # multi-threaded executor they keep arriving while a frame callback waits on a lookup;
        # the frame callbacks (the node's default, mutually exclusive group) run one at a time
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ground_up = float(self.get_parameter("ground_up_m").value)
        self.world = str(self.get_parameter("world_frame").value)
        self.config = str(self.get_parameter("config").value)
        self.fp: FramePipeline | None = None
        self.k = 0
        self.t0: int | None = None
        self.origin: tuple[float, float, float] | None = None
        self.rows: list[dict] = []
        self.poses: list[dict] = []
        self.ms: dict[str, list[float]] = {"pipeline": [], "callback": []}
        self.pub_det = self.create_publisher(Detection2DArray, "detections", 10)
        self.pub_obj = self.create_publisher(TrackedObjectArray, "tracked_objects", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "markers", 10)
        qos_name = str(self.get_parameter("image_qos").value)
        if qos_name not in ("sensor_data", "reliable", "recording"):
            raise SystemExit("image_qos must be sensor_data, reliable or recording")
        img_qos, other_qos = qos_profile_sensor_data, QoSProfile(depth=10)
        if qos_name == "reliable":
            img_qos = QoSProfile(depth=SYNC_QUEUE, reliability=ReliabilityPolicy.RELIABLE)
        elif qos_name == "recording":
            img_qos = other_qos = QoSProfile(
                depth=SYNC_QUEUE,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
        subs = [
            message_filters.Subscriber(self, Image, "image", qos_profile=img_qos),
            message_filters.Subscriber(self, CameraInfo, "camera_info", qos_profile=other_qos),
            message_filters.Subscriber(self, Odometry, "odometry", qos_profile=other_qos),
            message_filters.Subscriber(self, NavSatFix, "fix", qos_profile=other_qos),
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subs, SYNC_QUEUE, float(self.get_parameter("sync_slop_s").value)
        )
        self.sync.registerCallback(self.on_frame)
        self.subs = subs
        # launch files and tests can wait for this line before starting a recording
        self.get_logger().info("subscribed: waiting for frames")

    def pipeline(self, info: CameraInfo) -> FramePipeline:
        if self.fp is None:
            K = info.k
            cam = {"focal_px": float(K[0]), "cx": float(K[2]), "cy": float(K[5])}
            boxes = Path(self.boxes_csv) if self.boxes_csv else None
            self.fp = FramePipeline(self.config, overrides={"camera": cam}, boxes_csv=boxes)
            src = f"boxes from {boxes}" if boxes else "detector"
            self.get_logger().info(f"pipeline ready: config {self.config}, camera {cam}, {src}")
        return self.fp

    def mount(self, img: Image, odom: Odometry) -> tuple[list[float], list[float]] | None:
        """(translation, quaternion xyzw) of the camera's frame in the body's frame at the
        image's time stamp, or None (logged) if tf2 has no such transform in time."""
        body = self.body_frame or odom.child_frame_id
        cam = self.camera_frame or img.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                body, cam, Time.from_msg(img.header.stamp), self.tf_timeout
            ).transform
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ) as e:
            self.get_logger().error(
                f"no transform {body} -> {cam}: frame skipped ({type(e).__name__}: {e})",
                throttle_duration_sec=5.0,
            )
            return None
        t, r = tf.translation, tf.rotation
        return [t.x, t.y, t.z], [r.x, r.y, r.z, r.w]

    def on_frame(self, img: Image, info: CameraInfo, odom: Odometry, fix: NavSatFix) -> None:
        t_in = time.perf_counter()
        mount = self.mount(img, odom)
        if mount is None:
            return
        fp = self.pipeline(info)
        # whole nanoseconds since the first frame: a stamp of about 1.7e9 s held as one float
        # loses about 0.2 us, which shows in speeds
        stamp_ns = img.header.stamp.sec * 1_000_000_000 + img.header.stamp.nanosec
        if self.t0 is None:
            self.t0 = stamp_ns
        p, q = odom.pose.pose.position, odom.pose.pose.orientation
        pose = rb.camera_pose_from_frames(
            [p.x, p.y, p.z], [q.x, q.y, q.z, q.w], mount[0], mount[1], self.ground_up
        )
        if self.origin is None:
            # geodetic position of the odometry frame's origin: the fix minus the odometry
            lat, lon, h = ned_to_geodetic(
                [-p.y, -p.x, p.z], fix.latitude, fix.longitude, fix.altitude
            )
            self.origin = (float(lat), float(lon), float(h))
        self.k += 1
        t_s = (stamp_ns - self.t0) * 1e-9
        bgr = image_to_bgr(img)
        t_p = time.perf_counter()
        rows, det, _ = fp.step(bgr, self.k, t_s, pose)
        self.ms["pipeline"].append((time.perf_counter() - t_p) * 1000)
        self.rows += rows
        self.poses.append(
            {
                "frame": self.k,
                "t_s": t_s,
                "n_m": pose.n_m,
                "e_m": pose.e_m,
                "h_m": pose.h_m,
                "yaw_rad": pose.yaw_rad,
                "tilt_deg": pose.tilt_deg,
                "d_m": -p.z,  # local down of the drone
            }
        )
        if det is not None:
            self.pub_det.publish(self.detections_msg(img, det))
        self.pub_obj.publish(self.objects_msg(img, rows))
        self.pub_mk.publish(self.markers_msg(img, rows, pose))
        self.ms["callback"].append((time.perf_counter() - t_in) * 1000)

    def detections_msg(self, img: Image, det: object) -> Detection2DArray:
        out = Detection2DArray()
        out.header = img.header
        names = self.fp.cmap if self.fp is not None else {}
        for (cx, cy, w, h), sc, cl in zip(det.xywh, det.conf, det.cls, strict=True):
            d = Detection2D()
            d.header = img.header
            d.bbox.center.position.x, d.bbox.center.position.y = float(cx), float(cy)
            d.bbox.size_x, d.bbox.size_y = float(w), float(h)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(names.get(int(cl), int(cl)))
            hyp.hypothesis.score = float(sc)
            d.results.append(hyp)
            out.detections.append(d)
        return out

    def objects_msg(self, img: Image, rows: list[dict]) -> TrackedObjectArray:
        out = TrackedObjectArray()
        out.header.stamp = img.header.stamp
        out.header.frame_id = self.world
        out.frame_index = self.k
        for r in rows:
            o = TrackedObject()
            o.track_id = int(r["track_id"])
            o.object_class = str(r["class"])
            o.bbox.center.position.x = float(r["left"] + r["width"] / 2)
            o.bbox.center.position.y = float(r["top"] + r["height"] / 2)
            o.bbox.size_x, o.bbox.size_y = float(r["width"]), float(r["height"])
            o.valid = bool(r["valid"])
            o.reason = str(r["reason"])
            o.reliability = str(r["flag"])
            n, e = _f(r["ground_n_m"]), _f(r["ground_e_m"])
            o.position = Point(x=e, y=n, z=self.ground_up)
            cov = np.array(
                [
                    [_f(r["ground_cov_nn_m2"]), _f(r["ground_cov_ne_m2"])],
                    [_f(r["ground_cov_ne_m2"]), _f(r["ground_cov_ee_m2"])],
                ]
            )
            o.position_covariance = rb.cov_en_from_ne(cov).ravel().tolist()
            o.range_m, o.range_sigma_m = _f(r["range_m"]), _f(r["range_sigma_m"])
            o.range_half_width_m = [_f(r[f"range_half_{lv}_m"]) for lv in (68, 90, 95)]
            o.velocity_valid = bool(r["speed_shown"])
            if o.velocity_valid:
                v, vc = rb.velocity_enu(
                    _f(r["speed_ms"]),
                    _f(r["speed_sigma_ms"]),
                    _f(r["heading_deg"]),
                    _f(r["heading_sigma_deg"]),
                )
                o.velocity = Vector3(x=float(v[0]), y=float(v[1]), z=0.0)
                o.velocity_covariance = vc.ravel().tolist()
            else:
                o.velocity_covariance = [math.nan] * 4
            out.objects.append(o)
        return out

    def markers_msg(self, img: Image, rows: list[dict], pose: CameraPose) -> MarkerArray:
        out = MarkerArray()
        clear = Marker()
        clear.header.frame_id, clear.header.stamp = self.world, img.header.stamp
        clear.action = Marker.DELETEALL
        out.markers.append(clear)
        colours = {"person": (0.04, 0.42, 0.80), "vehicle": (0.78, 0.35, 0.04)}
        # the slant range drawn: a line from the camera to each object's ground point
        lines = Marker()
        lines.header = clear.header
        lines.ns, lines.id, lines.type, lines.action = "range", 0, Marker.LINE_LIST, Marker.ADD
        lines.pose.orientation.w = 1.0
        lines.scale.x = 0.12  # line width, display only
        camera = Point(x=pose.e_m, y=pose.n_m, z=pose.h_m + self.ground_up)
        for i, r in enumerate(rows):
            if not bool(r["valid"]):
                continue
            e, n = _f(r["ground_e_m"]), _f(r["ground_n_m"])
            col = colours.get(str(r["class"]), (0.5, 0.5, 0.5))
            cov = [
                [_f(r["ground_cov_nn_m2"]), _f(r["ground_cov_ne_m2"])],
                [_f(r["ground_cov_ne_m2"]), _f(r["ground_cov_ee_m2"])],
            ]
            a, b, brg = ellipse_axes(np.array(cov))
            yaw_enu = math.radians(90.0 - brg)  # bearing from north -> angle from east
            m = Marker()
            m.header.frame_id, m.header.stamp = self.world, img.header.stamp
            m.ns, m.id, m.type, m.action = "ellipse", i, Marker.CYLINDER, Marker.ADD
            m.pose.position = Point(x=e, y=n, z=self.ground_up)
            m.pose.orientation.z = math.sin(yaw_enu / 2)
            m.pose.orientation.w = math.cos(yaw_enu / 2)
            # display only: a 5 cm thick disc, never thinner than 5 cm so RViz draws it
            m.scale.x, m.scale.y, m.scale.z = 2 * a, max(2 * b, 0.05), 0.05
            m.color.r, m.color.g, m.color.b, m.color.a = (*col, 0.35)
            out.markers.append(m)
            t = Marker()
            t.header = m.header
            t.ns, t.id, t.type, t.action = "label", i, Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position = Point(x=e, y=n, z=self.ground_up + 2.0)
            t.scale.z = 3.0  # text height (m), display only
            t.color.r, t.color.g, t.color.b, t.color.a = (*col, 1.0)
            t.text = f"{r['class']} {int(r['track_id'])}  {_f(r['range_m']):.0f} m"
            out.markers.append(t)
            lines.points += [camera, Point(x=e, y=n, z=self.ground_up)]
            lines.colors += [ColorRGBA(r=col[0], g=col[1], b=col[2], a=0.8)] * 2
        if lines.points:
            out.markers.append(lines)
        return out

    def write_outputs(self) -> None:
        """tracks.csv (the columns `aerialrange run` writes) and the map, into `output_dir`."""
        from aerialrange.pipeline.run import COLUMNS

        d = str(self.get_parameter("output_dir").value)
        if not d or self.fp is None or self.origin is None:
            return
        out = Path(d)
        out.mkdir(parents=True, exist_ok=True)
        tracks = pd.DataFrame(self.rows, columns=COLUMNS)
        tracks.to_csv(out / "tracks.csv", index=False)
        pd.DataFrame(self.poses).to_csv(out / "flight.csv", index=False)
        timing = {
            k: {
                "median_ms": float(np.median(v)),
                "p95_ms": float(np.percentile(v, 95)),
                "n": len(v),
            }
            for k, v in self.ms.items()
            if v
        }
        timing["note"] = (
            "callback: from the synchronised messages' arrival in the node to the last publish "
            "(image conversion, pipeline, message building, publishing); pipeline: "
            "FramePipeline.step alone; the first callback includes loading the model"
        )
        (out / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
        if len(tracks):
            write_map_from_frames(tracks, pd.DataFrame(self.poses), self.origin, self.fp.model, out)
        self.get_logger().info(f"wrote {out} ({self.k} frames, {len(tracks)} rows)")


def main() -> None:
    rclpy.init()
    node = DroneRangerNode()
    # two threads: one for frames, one so tf2 can update while a frame waits on a lookup
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.write_outputs()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
