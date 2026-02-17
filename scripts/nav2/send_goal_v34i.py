#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from typing import Optional, Tuple


def _yaw_from_quat(z: float, w: float) -> float:
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def _fmt_feedback(fb: object) -> str:
    if fb is None:
        return "none"
    parts = []
    try:
        dr = getattr(fb, "distance_remaining", None)
        if dr is not None:
            parts.append(f"distance_remaining={float(dr):.3f}")
    except Exception:
        pass
    try:
        nt = getattr(fb, "navigation_time", None)
        if nt is not None and hasattr(nt, "sec"):
            nsec = float(getattr(nt, "nanosec", 0)) / 1e9
            parts.append(f"navigation_time={float(getattr(nt, 'sec', 0)) + nsec:.3f}")
    except Exception:
        pass
    try:
        rt = getattr(fb, "estimated_time_remaining", None)
        if rt is not None and hasattr(rt, "sec"):
            nsec = float(getattr(rt, "nanosec", 0)) / 1e9
            parts.append(f"eta={float(getattr(rt, 'sec', 0)) + nsec:.3f}")
    except Exception:
        pass
    return ",".join(parts) if parts else "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=0.0)
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--yaw_deg", type=float, default=0.0)
    ap.add_argument("--frame", type=str, default="map")
    ap.add_argument("--timeout_s", type=float, default=90.0)
    ap.add_argument("--goal_id", type=str, default="v34i_goal")
    ap.add_argument("--auto_from_odom", type=int, default=0)
    ap.add_argument("--offset_x", type=float, default=1.0)
    ap.add_argument("--offset_y", type=float, default=0.0)
    args = ap.parse_args()

    t0 = time.time()
    try:
        import rclpy  # type: ignore
        from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped  # type: ignore
        from nav_msgs.msg import Odometry  # type: ignore
        from nav2_msgs.action import NavigateToPose  # type: ignore
        from rclpy.action import ActionClient  # type: ignore
        from rclpy.node import Node  # type: ignore
    except Exception as e:
        dt = time.time() - t0
        print(
            f"[V34I_NAV2_RESULT] status=UNSUPPORTED ok=0 goal_id={args.goal_id} reason=rclpy_missing:{type(e).__name__} time_s={dt:.3f}",
            flush=True,
        )
        return 3

    class GoalNode(Node):
        def __init__(self) -> None:
            super().__init__("v34i_goal_sender")
            self.ac = ActionClient(self, NavigateToPose, "/navigate_to_pose")
            self.last_odom: Optional[Odometry] = None
            self.last_feedback: object = None
            self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
            self.create_subscription(Odometry, "/odom", self.on_odom, 10)

        def on_odom(self, msg: Odometry) -> None:
            self.last_odom = msg

        def on_feedback(self, msg: object) -> None:
            try:
                self.last_feedback = getattr(msg, "feedback", None)
            except Exception:
                self.last_feedback = None

    rclpy.init(args=None)
    node = GoalNode()

    def _odom_xy() -> Optional[Tuple[float, float]]:
        if node.last_odom is None:
            return None
        return (
            float(node.last_odom.pose.pose.position.x),
            float(node.last_odom.pose.pose.position.y),
        )

    try:
        for _ in range(40):
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.last_odom is not None:
                break

        if int(args.auto_from_odom) == 1 and node.last_odom is not None:
            ox = float(node.last_odom.pose.pose.position.x)
            oy = float(node.last_odom.pose.pose.position.y)
            qz = float(node.last_odom.pose.pose.orientation.z)
            qw = float(node.last_odom.pose.pose.orientation.w)
            oyaw = _yaw_from_quat(qz, qw)
            args.x = ox + float(args.offset_x) * math.cos(oyaw) - float(args.offset_y) * math.sin(oyaw)
            args.y = oy + float(args.offset_x) * math.sin(oyaw) + float(args.offset_y) * math.cos(oyaw)

        init_pose = PoseWithCovarianceStamped()
        init_pose.header.frame_id = str(args.frame)
        init_pose.header.stamp = node.get_clock().now().to_msg()
        if node.last_odom is not None:
            init_pose.pose.pose.position.x = float(node.last_odom.pose.pose.position.x)
            init_pose.pose.pose.position.y = float(node.last_odom.pose.pose.position.y)
            init_pose.pose.pose.orientation.z = float(node.last_odom.pose.pose.orientation.z)
            init_pose.pose.pose.orientation.w = float(node.last_odom.pose.pose.orientation.w)
        cov = [0.0] * 36
        cov[0] = 0.5
        cov[7] = 0.5
        cov[35] = 0.2
        init_pose.pose.covariance = cov
        for _ in range(6):
            node.initial_pose_pub.publish(init_pose)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.05)

        if not node.ac.wait_for_server(timeout_sec=12.0):
            dt = time.time() - t0
            print(
                f"[V34I_NAV2_RESULT] status=FAILED ok=0 goal_id={args.goal_id} reason=server_unavailable time_s={dt:.3f}",
                flush=True,
            )
            return 4

        start_xy = _odom_xy()
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = str(args.frame)
        goal.pose.header.stamp = node.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(args.x)
        goal.pose.pose.position.y = float(args.y)
        yaw = math.radians(float(args.yaw_deg))
        goal.pose.pose.orientation.z = math.sin(yaw * 0.5)
        goal.pose.pose.orientation.w = math.cos(yaw * 0.5)

        sf = node.ac.send_goal_async(goal, feedback_callback=node.on_feedback)
        rclpy.spin_until_future_complete(node, sf, timeout_sec=10.0)
        gh = sf.result()
        if gh is None or not gh.accepted:
            dt = time.time() - t0
            print(
                f"[V34I_NAV2_RESULT] status=GOAL_REJECTED ok=0 goal_id={args.goal_id} reason=goal_rejected time_s={dt:.3f}",
                flush=True,
            )
            return 5

        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(node, rf, timeout_sec=float(args.timeout_s))
        res = rf.result()
        dt = time.time() - t0

        moved = 0.0
        end_xy = _odom_xy()
        if start_xy is not None and end_xy is not None:
            moved = math.hypot(float(end_xy[0] - start_xy[0]), float(end_xy[1] - start_xy[1]))

        if res is None:
            fb = _fmt_feedback(node.last_feedback)
            print(
                f"[V34I_NAV2_RESULT] status=TIMEOUT ok=0 goal_id={args.goal_id} time_s={dt:.3f} moved_m={moved:.3f} last_feedback={fb}",
                flush=True,
            )
            return 6

        code = int(res.status)
        status_map = {4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}
        status = status_map.get(code, f"STATUS_{code}")
        ok = 1 if status == "SUCCEEDED" else 0
        fb = _fmt_feedback(node.last_feedback)
        print(
            f"[V34I_NAV2_RESULT] status={status} ok={ok} goal_id={args.goal_id} time_s={dt:.3f} moved_m={moved:.3f} last_feedback={fb}",
            flush=True,
        )
        print(
            json.dumps(
                {
                    "ok": int(ok),
                    "status": status,
                    "goal_id": str(args.goal_id),
                    "time_s": float(dt),
                    "moved_m": float(moved),
                    "last_feedback": fb,
                }
            ),
            flush=True,
        )
        return 0 if ok == 1 else 7
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
