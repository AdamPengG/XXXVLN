#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from typing import Optional


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node_name", default="v34c_mock_nav2_server")
    ap.add_argument("--cmd_vel_topic", default="/cmd_vel")
    ap.add_argument("--pulse_steps", type=int, default=24)
    ap.add_argument("--pulse_hz", type=float, default=10.0)
    args = ap.parse_args()

    import rclpy  # type: ignore
    from geometry_msgs.msg import Twist  # type: ignore
    from nav2_msgs.action import NavigateToPose  # type: ignore
    from rclpy.action import ActionServer  # type: ignore
    from rclpy.node import Node  # type: ignore

    rclpy.init(args=None)

    class Server(Node):
        def __init__(self) -> None:
            super().__init__(str(args.node_name))
            self.pub = self.create_publisher(Twist, str(args.cmd_vel_topic), 10)
            self.server = ActionServer(self, NavigateToPose, "/navigate_to_pose", self.execute_cb)

        def execute_cb(self, goal_handle):
            hz = max(1.0, float(args.pulse_hz))
            dt = 1.0 / hz
            pulse = max(1, int(args.pulse_steps))
            tw = Twist()
            tw.linear.x = 0.20
            tw.angular.z = 0.05
            for _ in range(pulse):
                self.pub.publish(tw)
                time.sleep(dt)
            tw.linear.x = 0.0
            tw.angular.z = 0.0
            self.pub.publish(tw)
            goal_handle.succeed()
            result = NavigateToPose.Result()
            return result

    node = Server()
    print("[V34C_MOCK_NAV2] ok=1 action=/navigate_to_pose cmd_vel_pub=1", flush=True)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
