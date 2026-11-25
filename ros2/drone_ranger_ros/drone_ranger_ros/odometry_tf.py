"""Broadcast each nav_msgs/Odometry message as the tf2 transform header.frame_id ->
child_frame_id (map -> base_link), for a recording or a flight stack that publishes odometry
without tf. On a drone whose flight controller already broadcasts this transform, do not run it.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class OdometryTf(Node):
    def __init__(self) -> None:
        super().__init__("odometry_tf")
        self.broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, "odometry", self.on_odometry, 10)

    def on_odometry(self, msg: Odometry) -> None:
        t = TransformStamped()
        t.header = msg.header
        t.child_frame_id = msg.child_frame_id
        p = msg.pose.pose.position
        t.transform.translation.x, t.transform.translation.y = p.x, p.y
        t.transform.translation.z = p.z
        t.transform.rotation = msg.pose.pose.orientation
        self.broadcaster.sendTransform(t)


def main() -> None:
    rclpy.init()
    node = OdometryTf()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
