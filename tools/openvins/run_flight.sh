#!/bin/bash
# Run OpenVINS (serial, mono) on one converted Mid-Air flight.
# Usage (inside the container): /run_flight.sh <flight>
# Expects /data/<flight>/flight.bag and /data/<flight>/config/estimator_config.yaml
# (tools/openvins/make_bag.py); writes /data/<flight>/openvins_estimate.txt.
set -e
f="$1"
. /opt/ros/noetic/setup.bash
. /catkin_ws/devel/setup.bash
roslaunch /mono_serial.launch \
    config_path:=/data/"$f"/config/estimator_config.yaml \
    bag:=/data/"$f"/flight.bag \
    path_est:=/data/"$f"/openvins_estimate.txt
