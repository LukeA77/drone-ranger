# ROS 2 image of the Drone Ranger node (Phase 12): ROS 2 Jazzy on Ubuntu 24.04, for x86-64 and
# ARM64 (the official ros images are built for both).
#
#   docker build -f docker/ros2.Dockerfile -t drone-ranger-ros:amd64 .
#   docker buildx build --platform linux/arm64 -f docker/ros2.Dockerfile -t drone-ranger-ros:arm64 --load .
#
# The demo (a synthetic recording into the node; outputs in ./out):
#   docker run --rm -v "$PWD/out:/out" drone-ranger-ros:amd64 \
#     ros2 launch drone_ranger_ros demo.launch.py rviz:=false output_dir:=/out
# With RViz: docker/ros2-viz.Dockerfile.
#
# numpy: ROS 2's Python message support is compiled against Ubuntu's numpy (1.26), and a module
# built against numpy 1.x cannot load under numpy 2. The package is therefore installed with
# numpy held at the system's version and the other libraries at versions pip finds compatible
# with it (not requirements-lock.txt, which pins numpy 2); the versions installed are written to
# /app/requirements-ros-freeze.txt. `aerialrange run` and the node are compared inside this same
# image (scripts/ros2_integration.sh), so the comparison does not depend on that choice.
FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-numpy python3-pytest \
      ros-jazzy-vision-msgs ros-jazzy-message-filters ros-jazzy-rosbag2-storage-mcap \
      ros-jazzy-launch-testing-ros ros-jazzy-rosbag2-py \
      ros-jazzy-tf2-ros ros-jazzy-tf2-ros-py \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src src
COPY configs configs
COPY scripts scripts
RUN echo "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')" > /tmp/numpy.txt \
    && PIP_CONSTRAINT=/tmp/numpy.txt python3 -m pip install --ignore-installed -e . \
    && python3 -m pip freeze > /app/requirements-ros-freeze.txt \
    && source /opt/ros/jazzy/setup.bash     && python3 -c "import numpy, rclpy, sensor_msgs.msg; print('numpy', numpy.__version__)"

COPY ros2 /ws/src
RUN source /opt/ros/jazzy/setup.bash && cd /ws \
    && colcon build --event-handlers console_direct- \
    && mkdir -p /app/cache /app/results /app/weights

# The demo recording: a synthetic flight with known truth, played by demo.launch.py and checked
# by scripts/ros2_demo_check.sh, so neither needs datasets or model weights. Shell scripts may
# have CRLF line endings in a Windows checkout.
ENV DRONE_RANGER_DEMO=/app/demo
RUN sed -i 's/\r$//' /app/scripts/*.sh \
    && source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash \
    && python3 scripts/ros2_demo_data.py --out "$DRONE_RANGER_DEMO"

COPY docker/ros2_entrypoint.sh /ros2_entrypoint.sh
# line endings may be CRLF in a Windows checkout
RUN sed -i 's/\r$//' /ros2_entrypoint.sh && chmod +x /ros2_entrypoint.sh
ENTRYPOINT ["/ros2_entrypoint.sh"]
CMD ["bash"]
