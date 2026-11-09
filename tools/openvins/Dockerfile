# OpenVINS reference for Phase 11 (D-741). OpenVINS is GPL-3.0: it runs here, in its own
# container, as a reference only; its code never enters src/ (build2 R13). Only its trajectory
# outputs are compared. Pinned to the v2.7 release tag.
FROM ros:noetic-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3-catkin-tools libeigen3-dev libboost-all-dev libceres-dev \
        ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-rosbag \
        ros-noetic-tf ros-noetic-message-filters ros-noetic-nav-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /catkin_ws/src
RUN git clone --depth 1 --branch v2.7 https://github.com/rpng/open_vins.git
WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && catkin build -j2 ov_msckf ov_eval

COPY run_flight.sh /run_flight.sh
COPY mono_serial.launch /mono_serial.launch
RUN chmod +x /run_flight.sh
ENTRYPOINT ["/run_flight.sh"]
