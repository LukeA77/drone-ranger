"""Replay a recording into the Drone Ranger node, with RViz showing the tracked objects, their
ranges and error ellipses, the camera image and the tf2 frame tree.

  ros2 launch drone_ranger_ros demo.launch.py                  # the built-in synthetic flight
  ros2 launch drone_ranger_ros demo.launch.py rviz:=false output_dir:=/out   # no display
  ros2 launch drone_ranger_ros demo.launch.py bag:=<recording> mount_tilt_deg:=<deg>

With no `bag`, it plays the synthetic flight built into the ROS 2 image (scripts/ros2_demo_data.py:
known ground truth, boxes from the scene, so no model weights are needed), described by
$DRONE_RANGER_DEMO/demo.json. With `bag`, it plays that recording (topics /image,
/camera_info, /odometry, /fix, as make_bag writes) and runs the detector unless `boxes_csv`
is given.

The recording starts once the node has subscribed, plays on the recording's clock (--clock,
use_sim_time), and the launch ends a few seconds after it finishes (exit_at_end, default: when
RViz is off), so the node writes tracks.csv, the map and timing.json to output_dir.
"""

import json
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from drone_ranger_ros.launch_util import (
    drone_ranger_node,
    mount_tf_nodes,
    odometry_tf_node,
    play_recording_cmd,
)
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit, OnProcessIO
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = {
    "bag": ("", "recording to play (default: the built-in synthetic flight)"),
    "mount_tilt_deg": ("", "camera tilt below the body (degrees); default: the demo's"),
    "boxes_csv": ("", "boxes CSV instead of the detector; default: the demo's, if no bag"),
    "config": ("run", "settings: a name in the repository's configs/"),
    "rate": ("1.0", "playback rate"),
    "start_delay_s": ("2.0", "wait after the node subscribes before playing (s)"),
    "rviz": ("true", "open RViz (needs a display)"),
    "exit_at_end": ("", "end the launch when the recording ends (default: when rviz is false)"),
    "output_dir": ("", "folder for tracks.csv, the map and timing.json at the end"),
}
END_WAIT_S = 5.0  # after playback, for the node to finish its last frames
STARTED = b"subscribed: waiting for frames"


def flag(v: str) -> bool:
    if v.lower() not in ("true", "false"):
        raise ValueError(f"expected true or false, got {v!r}")
    return v.lower() == "true"


def demo_defaults() -> dict:
    d = os.environ.get("DRONE_RANGER_DEMO", "")
    f = Path(d) / "demo.json"
    if not d or not f.exists():
        raise RuntimeError(
            "no bag given and no built-in demo ($DRONE_RANGER_DEMO/demo.json): pass "
            "bag:=<recording> mount_tilt_deg:=<deg>, or run inside the drone-ranger-ros image"
        )
    meta = json.loads(f.read_text())
    return {
        "bag": str(Path(d) / meta["bag"]),
        "boxes_csv": str(Path(d) / meta["boxes_csv"]),
        "mount_tilt_deg": str(meta["mount_tilt_deg"]),
        "description": meta["description"],
    }


def actions(context: LaunchContext) -> list:
    a = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    if not a["bag"]:
        d = demo_defaults()
        print(f"[demo] {d['description']}", flush=True)
        a["bag"] = d["bag"]
        a["boxes_csv"] = a["boxes_csv"] or d["boxes_csv"]
        a["mount_tilt_deg"] = a["mount_tilt_deg"] or d["mount_tilt_deg"]
    if not a["mount_tilt_deg"]:
        raise RuntimeError("set mount_tilt_deg: the tilt the recording was made with (make_bag)")
    rviz = flag(a["rviz"])
    exit_at_end = flag(a["exit_at_end"]) if a["exit_at_end"] else not rviz

    out = mount_tf_nodes(float(a["mount_tilt_deg"]), use_sim_time=True)
    out.append(odometry_tf_node("/odometry", use_sim_time=True))
    node = drone_ranger_node(
        {
            "config": a["config"],
            "boxes_csv": a["boxes_csv"],
            "output_dir": a["output_dir"],
            "image_qos": "recording",  # every frame of the recording (see play_recording_cmd)
            "use_sim_time": True,
        },
        {k: f"/{k}" for k in ("image", "camera_info", "odometry", "fix")},
        sigterm_timeout="60",
    )
    out.append(node)
    if rviz:
        cfg = Path(get_package_share_directory("drone_ranger_ros")) / "rviz" / "demo.rviz"
        out.append(
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", str(cfg)],
                parameters=[{"use_sim_time": True}],
                output="log",
            )
        )
    play = ExecuteProcess(
        cmd=play_recording_cmd(a["bag"], a["rate"], a["start_delay_s"], clock=True),
        output="screen",
    )
    started = [False]

    def on_output(event):  # noqa: ANN001, ANN202
        if started[0] or STARTED not in event.text:
            return None
        started[0] = True
        return [play]

    out.append(
        RegisterEventHandler(
            OnProcessIO(target_action=node, on_stdout=on_output, on_stderr=on_output)
        )
    )
    if exit_at_end:
        out.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=play,
                    on_exit=[
                        TimerAction(
                            period=END_WAIT_S,
                            actions=[EmitEvent(event=Shutdown(reason="recording finished"))],
                        )
                    ],
                )
            )
        )
    return out


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in ARGS.items()]
        + [OpaqueFunction(function=actions)]
    )
