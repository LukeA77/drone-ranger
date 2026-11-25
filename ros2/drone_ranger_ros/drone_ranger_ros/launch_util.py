"""Launch actions shared by the launch files and the integration test: the Drone Ranger node,
the camera mount as tf2 static transforms, and the odometry as tf2.

Frame tree (REP 103, REP 105):

  map ──(odometry: the flight controller, or odometry_tf)──> base_link
  base_link ──(static: the mount, pitched down by mount_tilt_deg)──> camera_link
  camera_link ──(static: roll -90°, yaw -90°)──> camera_optical_frame   (images are stamped here)

The rotation values come from `aerialrange.ros_bridge`, where tests pin them against the
node's geometry.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from aerialrange.ros_bridge import OPTICAL_RPY

BODY, CAMERA_LINK, OPTICAL = "base_link", "camera_link", "camera_optical_frame"
TOPICS = ("image", "camera_info", "odometry", "fix")


def static_tf(
    parent: str,
    child: str,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    use_sim_time: bool = False,
) -> Node:
    """A static_transform_publisher (angles in radians, no translation)."""
    args = ["--x", "0", "--y", "0", "--z", "0"]
    args += ["--roll", repr(roll), "--pitch", repr(pitch), "--yaw", repr(yaw)]
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=f"tf_{parent}_to_{child}",
        arguments=[*args, "--frame-id", parent, "--child-frame-id", child],
        parameters=[{"use_sim_time": use_sim_time}],
        output="log",
    )


def mount_tf_nodes(mount_tilt_deg: float, use_sim_time: bool = False) -> list[Node]:
    """Static transforms of a camera looking forward, `mount_tilt_deg` below the body's
    horizontal plane (positive pitch about the body's left axis tips forward down)."""
    return [
        static_tf(BODY, CAMERA_LINK, pitch=math.radians(mount_tilt_deg), use_sim_time=use_sim_time),
        static_tf(CAMERA_LINK, OPTICAL, *OPTICAL_RPY, use_sim_time=use_sim_time),
    ]


def odometry_tf_node(odometry: str, use_sim_time: bool = False) -> Node:
    return Node(
        package="drone_ranger_ros",
        executable="odometry_tf",
        name="odometry_tf",
        parameters=[{"use_sim_time": use_sim_time}],
        remappings=[("odometry", odometry)],
        output="log",
    )


def drone_ranger_node(
    params: dict[str, Any], remap: dict[str, str] | None = None, **kwargs: Any
) -> Node:
    """The node, with its parameters and topic remappings (TOPICS keys)."""
    remap = remap or {}
    return Node(
        package="drone_ranger_ros",
        executable="drone_ranger_node",
        name="drone_ranger",
        output="screen",
        parameters=[params],
        remappings=[(k, remap[k]) for k in TOPICS if k in remap],
        # unbuffered logs, so "subscribed: waiting for frames" reaches an event handler at once
        additional_env={"PYTHONUNBUFFERED": "1", "RCUTILS_LOGGING_BUFFERED_STREAM": "0"},
        **kwargs,
    )


def play_recording_cmd(bag: str, rate: str, delay_s: str, clock: bool = False) -> list[str]:
    """`ros2 bag play` for the node's image_qos:=recording: every input reliable and transient
    local (config/recording_qos.yaml), so a frame published before discovery has matched the
    node's subscriptions is still delivered when it matches; play exits only once every
    message is acknowledged (or after 30 s)."""
    qos = Path(get_package_share_directory("drone_ranger_ros")) / "config" / "recording_qos.yaml"
    cmd = ["ros2", "bag", "play", bag, "--rate", rate, "--delay", delay_s]
    cmd += ["--qos-profile-overrides-path", str(qos), "--wait-for-all-acked", "30000"]
    return cmd + (["--clock"] if clock else [])
