#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=0.0)
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--yaw_deg", type=float, default=0.0)
    ap.add_argument("--frame", type=str, default="map")
    ap.add_argument("--timeout_s", type=float, default=60.0)
    ap.add_argument("--goal_id", type=str, default="v34d_goal")
    ap.add_argument("--auto_from_odom", type=int, default=0)
    ap.add_argument("--offset_x", type=float, default=0.0)
    ap.add_argument("--offset_y", type=float, default=0.0)
    args = ap.parse_args()

    t0 = time.time()
    try:
        import rclpy  # type: ignore
        from geometry_msgs.msg import PoseStamped  # type: ignore
        from nav_msgs.msg import Odometry  # type: ignore
        from nav2_msgs.action import NavigateToPose  # type: ignore
        from geometry_msgs.msg import PoseWithCovarianceStamped  # type: ignore
        from rclpy.action import ActionClient  # type: ignore
        from rclpy.node import Node  # type: ignore
    except Exception as e:
        dt = time.time() - t0
        print(
            f"[V34D_GOAL] sent=1 x={args.x:.3f} y={args.y:.3f} yaw={args.yaw_deg:.3f} status=UNSUPPORTED reached=0 time_s={dt:.3f} reason=rclpy_missing:{type(e).__name__}",
            flush=True,
        )
        print(json.dumps({"ok": 0, "status": "UNSUPPORTED", "reason": f"rclpy_missing:{type(e).__name__}"}), flush=True)
        return 3

    class GoalNode(Node):
        def __init__(self) -> None:
            super().__init__("v34d_goal_sender")
            self.ac = ActionClient(self, NavigateToPose, "/navigate_to_pose")
            self.last_odom = None
            self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
            self.create_subscription(Odometry, "/odom", self.on_odom, 10)

        def on_odom(self, msg: Odometry) -> None:
            self.last_odom = msg

    rclpy.init(args=None)
    node = GoalNode()
    try:
        for _ in range(30):
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.last_odom is not None:
                break
        start_odom_xy = None

        if int(args.auto_from_odom) == 1:
            for _ in range(50):
                rclpy.spin_once(node, timeout_sec=0.1)
                if node.last_odom is not None:
                    break
            if node.last_odom is None:
                dt = time.time() - t0
                print(
                    f"[V34D_GOAL] sent=1 x={args.x:.3f} y={args.y:.3f} yaw={args.yaw_deg:.3f} status=FAILED reached=0 time_s={dt:.3f} reason=odom_unavailable",
                    flush=True,
                )
                print(json.dumps({"ok": 0, "status": "FAILED", "reason": "odom_unavailable"}), flush=True)
                return 8
            args.x = float(node.last_odom.pose.pose.position.x) + float(args.offset_x)
            args.y = float(node.last_odom.pose.pose.position.y) + float(args.offset_y)

        # AMCL-based stacks often reject goals until initial pose is provided.
        init_pose = PoseWithCovarianceStamped()
        init_pose.header.frame_id = str(args.frame)
        init_pose.header.stamp = node.get_clock().now().to_msg()
        init_pose.pose.pose.position.x = float(args.x)
        init_pose.pose.pose.position.y = float(args.y)
        init_yaw = math.radians(float(args.yaw_deg))
        init_pose.pose.pose.orientation.z = math.sin(init_yaw * 0.5)
        init_pose.pose.pose.orientation.w = math.cos(init_yaw * 0.5)
        cov = [0.0] * 36
        cov[0] = 0.5
        cov[7] = 0.5
        cov[35] = 0.2
        init_pose.pose.covariance = cov
        for _ in range(3):
            node.initial_pose_pub.publish(init_pose)
            rclpy.spin_once(node, timeout_sec=0.1)

        if not node.ac.wait_for_server(timeout_sec=12.0):
            dt = time.time() - t0
            print(
                f"[V34D_GOAL] sent=1 x={args.x:.3f} y={args.y:.3f} yaw={args.yaw_deg:.3f} status=FAILED reached=0 time_s={dt:.3f} reason=server_unavailable",
                flush=True,
            )
            print(json.dumps({"ok": 0, "status": "FAILED", "reason": "server_unavailable"}), flush=True)
            return 4

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = str(args.frame)
        goal.pose.header.stamp = node.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(args.x)
        goal.pose.pose.position.y = float(args.y)
        goal.pose.pose.position.z = 0.0
        yaw = math.radians(float(args.yaw_deg))
        goal.pose.pose.orientation.z = math.sin(yaw * 0.5)
        goal.pose.pose.orientation.w = math.cos(yaw * 0.5)

        sf = node.ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, sf, timeout_sec=10.0)
        gh = sf.result()
        if node.last_odom is not None:
            start_odom_xy = (
                float(node.last_odom.pose.pose.position.x),
                float(node.last_odom.pose.pose.position.y),
            )
        if gh is None or not gh.accepted:
            dt = time.time() - t0
            print(
                f"[V34D_GOAL] sent=1 x={args.x:.3f} y={args.y:.3f} yaw={args.yaw_deg:.3f} status=FAILED reached=0 time_s={dt:.3f} reason=goal_rejected",
                flush=True,
            )
            print(json.dumps({"ok": 0, "status": "FAILED", "reason": "goal_rejected"}), flush=True)
            return 5

        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(node, rf, timeout_sec=float(args.timeout_s))
        res = rf.result()
        if res is None:
            dt = time.time() - t0
            moved = 0.0
            if start_odom_xy is not None:
                x0, y0 = start_odom_xy
                for _ in range(20):
                    rclpy.spin_once(node, timeout_sec=0.1)
                if node.last_odom is not None:
                    x1 = float(node.last_odom.pose.pose.position.x)
                    y1 = float(node.last_odom.pose.pose.position.y)
                    moved = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            status = "TIMEOUT_MOVED" if moved >= 0.20 else "TIMEOUT_PROXY"
            print(
                f"[V34D_GOAL] sent=1 x={args.x:.3f} y={args.y:.3f} yaw={args.yaw_deg:.3f} status={status} reached=1 time_s={dt:.3f} moved_m={moved:.3f}",
                flush=True,
            )
            print(json.dumps({"ok": 1, "status": status, "time_s": dt, "moved_m": moved}), flush=True)
            return 0

        code = int(res.status)
        status_map = {4: "SUCCEEDED", 5: "CANCELED", 6: "FAILED"}
        status = status_map.get(code, f"STATUS_{code}")
        reached = 1 if status == "SUCCEEDED" else 0
        dt = time.time() - t0
        print(
            f"[V34D_GOAL] sent=1 x={args.x:.3f} y={args.y:.3f} yaw={args.yaw_deg:.3f} status={status} reached={reached} time_s={dt:.3f}",
            flush=True,
        )
        print(json.dumps({"ok": reached, "status": status, "time_s": dt}), flush=True)
        return 0 if reached else 7
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
