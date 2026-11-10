"""Integration test (build2 11.2): a rosbag2 recording converted from a flight goes in, and the
node's tracked objects must equal `aerialrange run` on the same frames and flight data.

Run by scripts/ros2_integration.sh inside the ROS 2 container (it makes the recording and the
reference first) with:
  DR_BAG     the recording (ros2 run drone_ranger_ros make_bag ...)
  DR_REF     the reference run's tracks.csv (aerialrange run on the same frames)
  DR_OUT     a folder for the node's outputs
  DR_MOUNT   the camera mount tilt used to make the recording (degrees)
  DR_CONFIG  the config name (as for aerialrange run)
  DR_RATE    playback rate (slow enough that no frame is dropped)
  DR_FRAMES  the number of frames in the recording
Skipped when DR_BAG is not set.
"""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

import launch
import launch.event_handlers
import launch_ros.actions
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import numpy as np
import pandas as pd
import pytest
import rclpy
from drone_ranger_msgs.msg import TrackedObjectArray

ENV = {k: os.environ.get(k) for k in ("DR_BAG", "DR_REF", "DR_OUT", "DR_MOUNT", "DR_CONFIG")}
RATE = os.environ.get("DR_RATE", "0.2")
TOL = 1e-6  # poses pass through a quaternion and back: about 1e-12, so this is generous
IDLE_S = 600  # far longer than one frame takes, even under ARM64 emulation (about 20 s)
EXACT = ["frame", "track_id", "class", "valid", "reason", "flag", "speed_shown"]


@pytest.mark.launch_test
def generate_test_description():  # noqa: ANN201 - launch_testing's entry point
    if not ENV["DR_BAG"]:
        pytest.skip("DR_BAG not set: run scripts/ros2_integration.sh")
    node = launch_ros.actions.Node(
        # time to write tracks.csv and the map after the interrupt (slow under emulation)
        sigterm_timeout="300",
        sigkill_timeout="300",
        # unbuffered logs, so the "subscribed" line reaches the event handler at once
        additional_env={"PYTHONUNBUFFERED": "1", "RCUTILS_LOGGING_BUFFERED_STREAM": "0"},
        package="drone_ranger_ros",
        executable="drone_ranger_node",
        name="drone_ranger",
        output="screen",
        parameters=[
            {
                "config": ENV["DR_CONFIG"] or "run",
                "mount_tilt_deg": float(ENV["DR_MOUNT"]),
                "output_dir": ENV["DR_OUT"],
                "image_qos": "reliable",  # every frame of the recording, for the comparison
            }
        ],
        remappings=[
            ("image", "/image"),
            ("camera_info", "/camera_info"),
            ("odometry", "/odometry"),
            ("fix", "/fix"),
        ],
    )
    play = launch.actions.ExecuteProcess(
        cmd=["ros2", "bag", "play", ENV["DR_BAG"], "--rate", RATE, "--delay", "5"],
        output="screen",
    )
    # start playback only once the node has subscribed: under ARM64 emulation its start-up takes
    # about a minute, and frames sent before then are lost
    started: list[bool] = [False]

    def on_output(event):  # noqa: ANN001, ANN202
        if started[0] or b"subscribed: waiting for frames" not in event.text:
            return None
        started[0] = True
        return [play]

    wait = launch.actions.RegisterEventHandler(
        launch.event_handlers.OnProcessIO(
            target_action=node, on_stdout=on_output, on_stderr=on_output
        )
    )
    return launch.LaunchDescription([node, wait, launch_testing.actions.ReadyToTest()]), {
        "node": node,
        "play": play,
    }


class TestPlayback(unittest.TestCase):
    def test_every_frame_is_processed(self, proc_info, play) -> None:  # noqa: ANN001
        """The node publishes a result for every frame of the recording (it may lag playback on
        a slow machine, so the launch is not shut down when playback ends), and the recording
        plays to its end. The listener subscribes at the start, before the first frame."""
        frames = int(os.environ.get("DR_FRAMES", "0"))
        rclpy.init()
        try:
            listener = rclpy.create_node("drone_ranger_test_listener")
            seen: list[int] = [0]
            listener.create_subscription(
                TrackedObjectArray,
                "tracked_objects",
                lambda m: seen.__setitem__(0, max(seen[0], m.frame_index)),
                100,
            )
            # give up once no new frame has arrived for IDLE_S (frames lost on the way never come)
            last, last_seen = time.monotonic(), 0
            while frames and seen[0] < frames and time.monotonic() - last < IDLE_S:
                rclpy.spin_once(listener, timeout_sec=1.0)
                if seen[0] != last_seen:
                    last, last_seen = time.monotonic(), seen[0]
            listener.destroy_node()
        finally:
            rclpy.shutdown()
        proc_info.assertWaitForShutdown(process=play, timeout=600)
        if frames:
            self.assertEqual(seen[0], frames, "the node did not publish every frame")


@launch_testing.post_shutdown_test()
class TestOutputs(unittest.TestCase):
    def test_tracks_equal_the_python_pipeline(self) -> None:
        got = pd.read_csv(Path(ENV["DR_OUT"]) / "tracks.csv")
        ref = pd.read_csv(ENV["DR_REF"])
        frames = int(os.environ.get("DR_FRAMES", "0"))
        if frames:
            flight = pd.read_csv(Path(ENV["DR_OUT"]) / "flight.csv")
            self.assertEqual(len(flight), frames, "the node dropped or repeated frames")
        self.assertEqual(list(got.columns), list(ref.columns))
        self.assertEqual(len(got), len(ref))
        for c in EXACT:
            self.assertTrue((got[c].astype(str) == ref[c].astype(str)).all(), c)
        num = [c for c in ref.columns if c not in EXACT and ref[c].dtype.kind in "fi"]
        for c in num:
            a, b = got[c].to_numpy(float), ref[c].to_numpy(float)
            self.assertTrue(np.allclose(a, b, rtol=TOL, atol=TOL, equal_nan=True), c)

    def test_map_written(self) -> None:
        self.assertTrue((Path(ENV["DR_OUT"]) / "map.geojson").exists())

    def test_node_exits_cleanly(self, proc_info) -> None:  # noqa: ANN001
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
