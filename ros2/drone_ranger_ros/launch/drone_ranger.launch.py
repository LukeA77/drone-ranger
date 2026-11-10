"""Start the Drone Ranger node. Arguments: config (a name in the repository's configs/),
mount_tilt_deg (camera tilt below the body, required), ground_up_m, world_frame, output_dir,
and topic remappings image, camera_info, odometry and fix."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGS = {
    "config": "run",
    "mount_tilt_deg": None,  # required: the vehicle's own camera mount
    "ground_up_m": "0.0",
    "world_frame": "map",
    "output_dir": "",
    "image_qos": "sensor_data",
    "image": "image",
    "camera_info": "camera_info",
    "odometry": "odometry",
    "fix": "fix",
}


def generate_launch_description() -> LaunchDescription:
    lc = {k: LaunchConfiguration(k) for k in ARGS}
    node = Node(
        package="drone_ranger_ros",
        executable="drone_ranger_node",
        name="drone_ranger",
        output="screen",
        parameters=[
            {
                "config": lc["config"],
                "mount_tilt_deg": ParameterValue(lc["mount_tilt_deg"], value_type=float),
                "ground_up_m": ParameterValue(lc["ground_up_m"], value_type=float),
                "world_frame": lc["world_frame"],
                "output_dir": lc["output_dir"],
                "image_qos": lc["image_qos"],
            }
        ],
        remappings=[(k, lc[k]) for k in ("image", "camera_info", "odometry", "fix")],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(k) if v is None else DeclareLaunchArgument(k, default_value=v)
            for k, v in ARGS.items()
        ]
        + [node]
    )
