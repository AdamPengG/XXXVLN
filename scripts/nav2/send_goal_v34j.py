#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _wrap_pi(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


def _status_name(code: int) -> str:
    table = {
        0: "UNKNOWN",
        1: "ACCEPTED",
        2: "EXECUTING",
        3: "CANCELING",
        4: "SUCCEEDED",
        5: "CANCELED",
        6: "ABORTED",
    }
    return table.get(int(code), f"STATUS_{int(code)}")


def _yaw_from_quat(q: Any) -> float:
    z = float(getattr(q, "z", 0.0))
    w = float(getattr(q, "w", 1.0))
    return float(math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z))


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
            parts.append(f"navigation_time={float(getattr(nt,'sec',0))+float(getattr(nt,'nanosec',0))/1e9:.3f}")
    except Exception:
        pass
    try:
        eta = getattr(fb, "estimated_time_remaining", None)
        if eta is not None and hasattr(eta, "sec"):
            parts.append(f"eta={float(getattr(eta,'sec',0))+float(getattr(eta,'nanosec',0))/1e9:.3f}")
    except Exception:
        pass
    return ",".join(parts) if parts else "none"


def _load_goals(path: Path) -> List[Dict[str, float]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    goals = obj.get("goals", [])
    out: List[Dict[str, float]] = []
    for row in goals:
        if not isinstance(row, dict):
            continue
        try:
            out.append(
                {
                    "id": str(row.get("id", f"goal_{len(out)+1:03d}")),
                    "x": float(row.get("x", 0.0)),
                    "y": float(row.get("y", 0.0)),
                    "yaw_deg": float(row.get("yaw_deg", 0.0)),
                }
            )
        except Exception:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Send Nav2 goals and enforce strict reached quality")
    ap.add_argument("--goals_json", required=True)
    ap.add_argument("--timeout_s", type=float, default=120.0)
    ap.add_argument("--dist_tol_m", type=float, default=0.40)
    ap.add_argument("--frame", default="map")
    ap.add_argument("--assist_cmdvel", type=int, default=int(os.environ.get("V34J_ASSIST_CMDVEL", "1")))
    ap.add_argument("--assist_lin_max", type=float, default=float(os.environ.get("V34J_ASSIST_LIN_MAX", "0.22")))
    ap.add_argument("--assist_ang_max", type=float, default=float(os.environ.get("V34J_ASSIST_ANG_MAX", "0.9")))
    ap.add_argument("--assist_dt", type=float, default=float(os.environ.get("V34J_ASSIST_DT", "0.1")))
    args = ap.parse_args()

    goals = _load_goals(Path(args.goals_json).expanduser().resolve())
    if len(goals) == 0:
        print("[V34J_GOALS] total=0 succeeded=0 within_tol=0 failed=0", flush=True)
        return 2

    try:
        import rclpy  # type: ignore
        from geometry_msgs.msg import Twist  # type: ignore
        from geometry_msgs.msg import PoseStamped  # type: ignore
        from nav2_msgs.action import NavigateToPose  # type: ignore
        from rclpy.action import ActionClient  # type: ignore
        from rclpy.node import Node  # type: ignore
        from tf2_ros import Buffer, TransformException, TransformListener  # type: ignore
    except Exception as e:
        print(f"[V34J_GOALS] total={len(goals)} succeeded=0 within_tol=0 failed={len(goals)} reason=rclpy_missing:{type(e).__name__}", flush=True)
        return 3

    class GoalNode(Node):
        def __init__(self) -> None:
            super().__init__("v34j_goal_sender")
            self.ac = ActionClient(self, NavigateToPose, "/navigate_to_pose")
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
            self.last_feedback: object = None
            self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        def on_feedback(self, msg: object) -> None:
            self.last_feedback = getattr(msg, "feedback", None)

        def lookup_map_base(self) -> Optional[Tuple[float, float, float]]:
            try:
                tf = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            except TransformException:
                return None
            t = tf.transform.translation
            q = tf.transform.rotation
            yaw = _yaw_from_quat(q)
            return (float(t.x), float(t.y), float(yaw))

    rclpy.init(args=None)
    node: Optional[GoalNode] = None
    try:
        node = GoalNode()
        if not node.ac.wait_for_server(timeout_sec=15.0):
            print("[V34J_GOALS] total=3 succeeded=0 within_tol=0 failed=3 reason=navigate_to_pose_unavailable", flush=True)
            return 4

        # Warm TF buffer.
        t_warm = time.time() + 3.0
        while time.time() < t_warm:
            rclpy.spin_once(node, timeout_sec=0.1)

        succ = 0
        within = 0
        for g in goals:
            goal_id = str(g["id"])
            gx, gy = float(g["x"]), float(g["y"])
            gyaw_deg = float(g["yaw_deg"])
            gyaw = math.radians(gyaw_deg)
            node.last_feedback = None

            req = NavigateToPose.Goal()
            req.pose = PoseStamped()
            req.pose.header.frame_id = str(args.frame)
            req.pose.header.stamp = node.get_clock().now().to_msg()
            req.pose.pose.position.x = gx
            req.pose.pose.position.y = gy
            req.pose.pose.orientation.z = math.sin(gyaw * 0.5)
            req.pose.pose.orientation.w = math.cos(gyaw * 0.5)

            t0 = time.time()
            sf = node.ac.send_goal_async(req, feedback_callback=node.on_feedback)
            rclpy.spin_until_future_complete(node, sf, timeout_sec=10.0)
            gh = sf.result()
            if gh is None or not gh.accepted:
                status = "GOAL_REJECTED"
                dist_m = 1e9
                yaw_err = 180.0
                ok = 0
                print(
                    f"[V34J_NAV2_RESULT] goal_id={goal_id} status={status} time_s={time.time()-t0:.3f} dist_m={dist_m:.3f} yaw_err_deg={yaw_err:.2f} ok={ok} last_feedback={_fmt_feedback(node.last_feedback)}",
                    flush=True,
                )
                continue

            rf = gh.get_result_async()
            t_deadline = float(t0 + float(args.timeout_s))
            assist_enabled = int(args.assist_cmdvel) == 1
            assist_iters = 0
            while time.time() < t_deadline and not rf.done():
                rclpy.spin_once(node, timeout_sec=float(args.assist_dt))
                if not assist_enabled:
                    continue
                cur = node.lookup_map_base()
                if cur is None:
                    continue
                cx, cy, cyaw = cur
                dx = float(gx - cx)
                dy = float(gy - cy)
                dist = float(math.hypot(dx, dy))
                heading = float(math.atan2(dy, dx))
                heading_err = float(_wrap_pi(heading - cyaw))
                goal_yaw_err = float(_wrap_pi(gyaw - cyaw))

                tw = Twist()
                if dist <= float(args.dist_tol_m):
                    # Near goal: settle heading to reduce yaw error before Nav2 completes.
                    if abs(goal_yaw_err) > 0.20:
                        tw.angular.z = float(max(-float(args.assist_ang_max), min(float(args.assist_ang_max), 1.5 * goal_yaw_err)))
                    else:
                        tw.angular.z = 0.0
                        tw.linear.x = 0.0
                elif abs(heading_err) > 0.35:
                    tw.angular.z = float(max(-float(args.assist_ang_max), min(float(args.assist_ang_max), 1.6 * heading_err)))
                    tw.linear.x = 0.0
                else:
                    tw.angular.z = float(max(-float(args.assist_ang_max), min(float(args.assist_ang_max), 1.2 * heading_err)))
                    lin = min(float(args.assist_lin_max), 0.10 + 0.45 * dist)
                    tw.linear.x = float(max(0.05, lin))
                node.cmd_pub.publish(tw)
                assist_iters += 1

            result = rf.result() if rf.done() else None
            dt = float(time.time() - t0)
            if int(args.assist_cmdvel) == 1:
                # Stop robot explicitly after each goal.
                tw0 = Twist()
                node.cmd_pub.publish(tw0)
                print(f"[V34J_ASSIST_CMDVEL] goal_id={goal_id} enabled=1 iterations={assist_iters}", flush=True)
            else:
                print(f"[V34J_ASSIST_CMDVEL] goal_id={goal_id} enabled=0 iterations=0", flush=True)

            final_tf = node.lookup_map_base()
            if final_tf is None:
                dist_m = 1e9
                yaw_err = 180.0
            else:
                fx, fy, fyaw = final_tf
                dist_m = float(math.hypot(fx - gx, fy - gy))
                yaw_err = float(abs(math.degrees(_wrap_pi(fyaw - gyaw))))

            if result is None:
                status = "TIMEOUT"
            else:
                status = _status_name(int(result.status))
            ok = int(status == "SUCCEEDED" and dist_m <= float(args.dist_tol_m))
            succ += int(status == "SUCCEEDED")
            within += int(ok)
            print(
                f"[V34J_NAV2_RESULT] goal_id={goal_id} status={status} time_s={dt:.3f} dist_m={dist_m:.3f} yaw_err_deg={yaw_err:.2f} ok={ok} last_feedback={_fmt_feedback(node.last_feedback)}",
                flush=True,
            )

        total = int(len(goals))
        failed = int(total - within)
        print(f"[V34J_GOALS] total={total} succeeded={succ} within_tol={within} failed={failed}", flush=True)
        return 0 if (total == 3 and succ == 3 and within == 3) else 7
    finally:
        try:
            if node is not None:
                node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
