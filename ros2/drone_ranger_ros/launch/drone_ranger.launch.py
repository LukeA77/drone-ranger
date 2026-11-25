"""Start the Drone Ranger node on a drone (or on any live or recorded input).

  ros2 launch drone_ranger_ros drone_ranger.launch.py mount_tilt_deg:=30 \
      image:=/camera/image_raw camera_info:=/camera/camera_info odometry:=/odom fix:=/gps/fix

Arguments (`ros2 launch drone_ranger_ros drone_ranger.launch.py --show-args` lists them):
  config            settings: a name in the repository's configs/ (default run)
  mount_tilt_deg    the camera's tilt below the body's horizontal plane, in degrees; required
                    while publish_mount_tf is true
  publish_mount_tf  publish the mount as static transforms base_link -> camera_link ->
                    camera_optical_frame (default true; false when a robot description or a
                    gimbal driver already publishes the camera's frames)
  publish_odom_tf   broadcast the odometry as map -> base_link (default false: a flight
                    controller normally does this; true for a recording)
  boxes_csv         read boxes from this CSV instead of running the detector (default: detector)
  ground_up_m, world_frame, output_dir, image_qos, use_sim_time: the node's parameters
  image, camera_info, odometry, fix: topic remappings
"""

from drone_ranger_ros.launch_util import (
    drone_ranger_node,
    mount_tf_nodes,
    odometry_tf_node,
)
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

ARGS = {
    "config": ("run", "settings: a name in the repository's configs/"),
    "mount_tilt_deg": ("", "camera tilt below the body's horizontal plane (degrees)"),
    "publish_mount_tf": ("true", "publish base_link -> camera_link -> camera_optical_frame"),
    "publish_odom_tf": ("false", "broadcast the odometry as map -> base_link"),
    "boxes_csv": ("", "boxes CSV read instead of running the detector"),
    "ground_up_m": ("0.0", "flat ground height in the odometry frame (m)"),
    "world_frame": ("map", "frame of the published positions"),
    "output_dir": ("", "folder for tracks.csv, the map and timing.json at shutdown"),
    "image_qos": ("sensor_data", "sensor_data (live camera) or reliable (recordings)"),
    "use_sim_time": ("false", "follow /clock (a recording played with --clock)"),
    "image": ("image", "image topic"),
    "camera_info": ("camera_info", "camera_info topic"),
    "odometry": ("odometry", "odometry topic"),
    "fix": ("fix", "GPS fix topic"),
}


def flag(v: str) -> bool:
    if v.lower() not in ("true", "false"):
        raise ValueError(f"expected true or false, got {v!r}")
    return v.lower() == "true"


def nodes(context: LaunchContext) -> list:
    a = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    sim = flag(a["use_sim_time"])
    out = []
    if flag(a["publish_mount_tf"]):
        if not a["mount_tilt_deg"]:
            raise RuntimeError(
                "set mount_tilt_deg (the camera's tilt below the body, degrees), or "
                "publish_mount_tf:=false if the camera's frames are already published"
            )
        out += mount_tf_nodes(float(a["mount_tilt_deg"]), use_sim_time=sim)
    if flag(a["publish_odom_tf"]):
        out.append(odometry_tf_node(a["odometry"], use_sim_time=sim))
    params = {
        "config": a["config"],
        "boxes_csv": a["boxes_csv"],
        "ground_up_m": float(a["ground_up_m"]),
        "world_frame": a["world_frame"],
        "output_dir": a["output_dir"],
        "image_qos": a["image_qos"],
        "use_sim_time": sim,
    }
    remap = {k: a[k] for k in ("image", "camera_info", "odometry", "fix")}
    # time to write tracks.csv and the map after the interrupt
    out.append(drone_ranger_node(params, remap, sigterm_timeout="60"))
    return out


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in ARGS.items()]
        + [OpaqueFunction(function=nodes)]
    )
