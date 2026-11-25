# The ROS 2 image plus RViz, a virtual display and ffmpeg: for watching the demo on a desktop
# and for recording the README's RViz clip (scripts/record_rviz.sh). Built on the image of
# docker/ros2.Dockerfile, which must be built first.
#
#   docker build -f docker/ros2.Dockerfile -t drone-ranger-ros:amd64 .
#   docker build -f docker/ros2-viz.Dockerfile -t drone-ranger-ros-viz:amd64 .
#
# On a Linux desktop (X11), the demo with RViz:
#   xhost +local:docker
#   docker run --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix drone-ranger-ros-viz:amd64 \
#     ros2 launch drone_ranger_ros demo.launch.py
ARG BASE=drone-ranger-ros:amd64
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-rviz2 xvfb x11-utils ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
