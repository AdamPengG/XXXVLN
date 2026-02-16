#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time


def _emit(goal_id: str, status: str, reached: int, t0: float, reason: str = "", final_dist: float = -1.0) -> None:
    dt = max(0.0, time.time() - t0)
    extra = f" reason={reason}" if reason else ""
    print(
        f"[V34C_GOAL] sent=1 id={goal_id} x={_ARGS.x:.3f} y={_ARGS.y:.3f} yaw={_ARGS.yaw_deg:.3f} status={status} reached={reached} time_s={dt:.3f}{extra}",
        flush=True,
    )
    payload = {
        "ok": int(reached),
        "status": status,
        "reached": int(reached),
        "time_s": dt,
        "reason": reason,
        "final_dist": float(final_dist),
    }
    print(json.dumps(payload), flush=True)


def main() -> int:
    global _ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--yaw_deg", type=float, default=0.0)
    ap.add_argument("--frame", type=str, default="map")
    ap.add_argument("--timeout_s", type=float, default=45.0)
    ap.add_argument("--goal_id", type=str, default="v34c_goal")
    _ARGS = ap.parse_args()

    t0 = time.time()

    try:
        import rclpy  # type: ignore
        from geometry_msgs.msg import PoseStamped  # type: ignore
        from nav2_msgs.action import NavigateToPose  # type: ignore
        from rclpy.action import ActionClient  # type: ignore
        from rclpy.node import Node  # type: ignore
    except Exception as e:
        _emit(_ARGS.goal_id, "UNSUPPORTED", 0, t0, reason=f"rclpy_missing:{type(e).__name__}")
        return 3

    class Client(Node):
        def __init__(self) -> None:
            super().__init__("v34c_nav2_goal_client")
            self.cli = ActionClient(self, NavigateToPose, "/navigate_to_pose")

    rclpy.init(args=None)
    node = Client()
    try:
        if not node.cli.wait_for_server(timeout_sec=8.0):
            _emit(_ARGS.goal_id, "FAILED", 0, t0, reason="navigate_to_pose_unavailable")
            return 4

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = str(_ARGS.frame)
        goal.pose.header.stamp = node.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(_ARGS.x)
        goal.pose.pose.position.y = float(_ARGS.y)
        goal.pose.pose.position.z = 0.0
        yaw = math.radians(float(_ARGS.yaw_deg))
        goal.pose.pose.orientation.z = math.sin(yaw * 0.5)
        goal.pose.pose.orientation.w = math.cos(yaw * 0.5)

        sf = node.cli.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, sf, timeout_sec=10.0)
        gh = sf.result()
        if gh is None or not gh.accepted:
            _emit(_ARGS.goal_id, "FAILED", 0, t0, reason="goal_rejected")
            return 5

        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(node, rf, timeout_sec=float(_ARGS.timeout_s))
        res = rf.result()
        if res is None:
            _emit(_ARGS.goal_id, "TIMEOUT", 0, t0, reason="goal_timeout")
            return 6

        code = int(res.status)
        status_map = {4: "SUCCEEDED", 5: "CANCELED", 6: "FAILED"}
        status = status_map.get(code, f"STATUS_{code}")
        reached = 1 if status == "SUCCEEDED" else 0
        _emit(_ARGS.goal_id, status, reached, t0)
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
