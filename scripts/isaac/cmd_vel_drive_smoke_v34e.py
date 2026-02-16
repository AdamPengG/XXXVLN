#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic_cmd", default="/cmd_vel")
    ap.add_argument("--topic_odom", default="/odom")
    ap.add_argument("--duration_s", type=float, default=3.0)
    ap.add_argument("--linear_x", type=float, default=0.35)
    ap.add_argument("--angular_z", type=float, default=0.0)
    ap.add_argument("--min_move_m", type=float, default=0.5)
    ap.add_argument("--settle_s", type=float, default=1.5)
    ap.add_argument("--tag", default="drive")
    args = ap.parse_args()

    try:
        import rclpy  # type: ignore
        from geometry_msgs.msg import Twist  # type: ignore
        from nav_msgs.msg import Odometry  # type: ignore
        from rclpy.node import Node  # type: ignore
    except Exception as e:
        print(
            f"[V34E_CMDVEL_DRIVE] ok=0 moved_m=0.000 odom_start=(nan,nan) odom_end=(nan,nan) reason=rclpy_missing:{type(e).__name__} tag={args.tag}",
            flush=True,
        )
        return 2

    class DriveNode(Node):
        def __init__(self) -> None:
            super().__init__("v34e_cmdvel_drive_smoke")
            self.pub = self.create_publisher(Twist, str(args.topic_cmd), 10)
            self.last_odom = None
            self.odom_count = 0
            self.create_subscription(Odometry, str(args.topic_odom), self._on_odom, 50)

        def _on_odom(self, msg: Odometry) -> None:
            self.last_odom = msg
            self.odom_count += 1

    rclpy.init(args=None)
    node = DriveNode()
    try:
        t_wait_end = time.time() + 12.0
        while time.time() < t_wait_end and node.last_odom is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.last_odom is None:
            print(
                f"[V34E_CMDVEL_DRIVE] ok=0 moved_m=0.000 odom_start=(nan,nan) odom_end=(nan,nan) reason=odom_unavailable tag={args.tag}",
                flush=True,
            )
            return 3

        start = (
            float(node.last_odom.pose.pose.position.x),
            float(node.last_odom.pose.pose.position.y),
        )
        start_count = int(node.odom_count)

        twist = Twist()
        twist.linear.x = float(args.linear_x)
        twist.angular.z = float(args.angular_z)

        t_end = time.time() + float(args.duration_s)
        while time.time() < t_end:
            node.pub.publish(twist)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.05)

        # Ensure stop command is sent.
        stop = Twist()
        for _ in range(5):
            node.pub.publish(stop)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.05)

        t_settle_end = time.time() + float(args.settle_s)
        while time.time() < t_settle_end:
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.05)

        if node.last_odom is None:
            print(
                f"[V34E_CMDVEL_DRIVE] ok=0 moved_m=0.000 odom_start=({start[0]:.3f},{start[1]:.3f}) odom_end=(nan,nan) reason=odom_after_missing tag={args.tag}",
                flush=True,
            )
            return 4

        end = (
            float(node.last_odom.pose.pose.position.x),
            float(node.last_odom.pose.pose.position.y),
        )
        moved = math.hypot(float(end[0] - start[0]), float(end[1] - start[1]))
        updates = int(node.odom_count - start_count)
        if updates <= 0:
            print(
                f"[V34E_CMDVEL_DRIVE] ok=0 moved_m={moved:.3f} odom_start=({start[0]:.3f},{start[1]:.3f}) odom_end=({end[0]:.3f},{end[1]:.3f}) reason=odom_not_updating tag={args.tag}",
                flush=True,
            )
            return 5

        ok = int(moved > float(args.min_move_m))
        print(
            f"[V34E_CMDVEL_DRIVE] ok={ok} moved_m={moved:.3f} odom_start=({start[0]:.3f},{start[1]:.3f}) odom_end=({end[0]:.3f},{end[1]:.3f}) odom_updates={updates}",
            flush=True,
        )
        return 0 if ok == 1 else 6
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
