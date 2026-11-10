#!/bin/bash
# Source ROS 2 and the Drone Ranger workspace, then run the command.
set -e
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
exec "$@"
