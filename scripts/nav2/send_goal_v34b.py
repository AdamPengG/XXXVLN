#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time


def _print_fail(goal_id: str, status: str, reason: str, t0: float, final_dist: float = -1.0, code: int = 2) -> int:
    dt = max(0.0, time.time() - t0)
    print(f"[V34B_NAV2_RESULT] id={goal_id} status={status} time_s={dt:.3f} final_dist={final_dist:.3f} reason={reason}", flush=True)
    payload = {"ok": 0, "status": status, "reason": reason, "time_s": dt, "final_dist": final_dist}
    print(json.dumps(payload), flush=True)
    return int(code)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--yaw_deg", type=float, default=0.0)
    ap.add_argument("--frame", type=str, default="map")
    ap.add_argument("--timeout_s", type=float, default=45.0)
    ap.add_argument("--goal_id", type=str, default="goal_v34b")
    args = ap.parse_args()

    t0 = time.time()
    print(
        f"[V34B_GOAL_SENT] id={args.goal_id} x={args.x:.3f} y={args.y:.3f} yaw={args.yaw_deg:.3f}",
        flush=True,
    )

    try:
        import rclpy  # type: ignore
        from geometry_msgs.msg import PoseStamped  # type: ignore
        from nav2_msgs.action import NavigateToPose  # type: ignore
        from rclpy.action import ActionClient  # type: ignore
        from rclpy.node import Node  # type: ignore
    except Exception:
        return _print_fail(str(args.goal_id), "UNSUPPORTED", "ros2_python_not_available", t0, code=3)

    class Nav2Client(Node):
        def __init__(self) -> None:
            super().__init__("v34b_nav2_goal_client")
            self.client = ActionClient(self, NavigateToPose, "/navigate_to_pose")

    rclpy.init(args=None)
    node = Nav2Client()
    try:
        if not node.client.wait_for_server(timeout_sec=5.0):
            return _print_fail(str(args.goal_id), "FAILED", "navigate_to_pose_server_unavailable", t0, code=4)

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

        send_future = node.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return _print_fail(str(args.goal_id), "FAILED", "goal_rejected", t0, code=5)

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future, timeout_sec=float(args.timeout_s))
        res = result_future.result()
        if res is None:
            return _print_fail(str(args.goal_id), "TIMEOUT", "result_timeout", t0, code=6)

        status_code = int(res.status)
        status_map = {
            4: "SUCCEEDED",
            5: "CANCELED",
            6: "FAILED",
        }
        status = status_map.get(status_code, f"STATUS_{status_code}")
        dt = max(0.0, time.time() - t0)
        final_dist = 0.0 if status == "SUCCEEDED" else -1.0
        print(f"[V34B_NAV2_RESULT] id={args.goal_id} status={status} time_s={dt:.3f} final_dist={final_dist:.3f}", flush=True)
        payload = {
            "ok": 1 if status == "SUCCEEDED" else 0,
            "status": status,
            "time_s": dt,
            "final_dist": final_dist,
        }
        print(json.dumps(payload), flush=True)
        return 0 if status == "SUCCEEDED" else 7
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
