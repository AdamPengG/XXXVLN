#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _wrap_pi(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


def _yaw_from_quat(q: Any) -> float:
    z = float(getattr(q, "z", 0.0))
    w = float(getattr(q, "w", 1.0))
    return float(math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z))


def _load_goals(path: Path) -> List[Dict[str, float | str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    raw = obj.get("goals", []) if isinstance(obj, dict) else []
    out: List[Dict[str, float | str]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            continue
        try:
            out.append(
                {
                    "id": str(row.get("id", f"seg_{i+1:03d}")),
                    "x": float(row.get("x", 0.0)),
                    "y": float(row.get("y", 0.0)),
                    "yaw_deg": float(row.get("yaw_deg", 0.0)),
                }
            )
        except Exception:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Send staged Nav2 goals for v35a topo path")
    ap.add_argument("--goals_json", required=True)
    ap.add_argument("--frame", default="map")
    ap.add_argument("--timeout_s", type=float, default=90.0)
    ap.add_argument("--dist_tol_m", type=float, default=0.40)
    ap.add_argument("--yaw_tol_deg", type=float, default=35.0)
    ap.add_argument("--result_json", default="")
    ap.add_argument("--assist_cmdvel", type=int, default=int(__import__("os").environ.get("V35A_ASSIST_CMDVEL", "1")))
    ap.add_argument("--assist_lin_max", type=float, default=float(__import__("os").environ.get("V35A_ASSIST_LIN_MAX", "0.22")))
    ap.add_argument("--assist_ang_max", type=float, default=float(__import__("os").environ.get("V35A_ASSIST_ANG_MAX", "0.9")))
    ap.add_argument("--assist_dt", type=float, default=float(__import__("os").environ.get("V35A_ASSIST_DT", "0.1")))
    args = ap.parse_args()

    goals = _load_goals(Path(args.goals_json).expanduser().resolve())
    if len(goals) == 0:
        print("[V35A_NAV2_STAGED] total=0 succeeded=0 failed=0 ok=0 reason=no_goals", flush=True)
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
        print(f"[V35A_NAV2_STAGED] total={len(goals)} succeeded=0 failed={len(goals)} ok=0 reason=rclpy_missing:{type(e).__name__}", flush=True)
        return 3

    class StagedNode(Node):
        def __init__(self) -> None:
            super().__init__("v35a_staged_goal_sender")
            self.ac = ActionClient(self, NavigateToPose, "/navigate_to_pose")
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
            self.last_feedback: object = None
            self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        def on_feedback(self, msg: object) -> None:
            self.last_feedback = getattr(msg, "feedback", None)

        def lookup_pose(self) -> Optional[Tuple[float, float, float]]:
            try:
                tf = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            except TransformException:
                return None
            t = tf.transform.translation
            q = tf.transform.rotation
            yaw = _yaw_from_quat(q)
            return (float(t.x), float(t.y), float(yaw))

    rclpy.init(args=None)
    node: Optional[StagedNode] = None
    result_rows: List[Dict[str, object]] = []
    try:
        node = StagedNode()
        if not node.ac.wait_for_server(timeout_sec=15.0):
            print(f"[V35A_NAV2_STAGED] total={len(goals)} succeeded=0 failed={len(goals)} ok=0 reason=navigate_to_pose_unavailable", flush=True)
            return 4

        warm_deadline = time.time() + 3.0
        while time.time() < warm_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

        succeeded = 0
        failed = 0
        for row in goals:
            goal_id = str(row["id"])
            gx = float(row["x"])
            gy = float(row["y"])
            gyaw_deg = float(row["yaw_deg"])
            gyaw = math.radians(gyaw_deg)

            req = NavigateToPose.Goal()
            req.pose = PoseStamped()
            req.pose.header.frame_id = str(args.frame)
            req.pose.header.stamp = node.get_clock().now().to_msg()
            req.pose.pose.position.x = gx
            req.pose.pose.position.y = gy
            req.pose.pose.orientation.z = math.sin(0.5 * gyaw)
            req.pose.pose.orientation.w = math.cos(0.5 * gyaw)

            t0 = time.time()
            sf = node.ac.send_goal_async(req, feedback_callback=node.on_feedback)
            rclpy.spin_until_future_complete(node, sf, timeout_sec=10.0)
            gh = sf.result()
            if gh is None or not gh.accepted:
                status = "GOAL_REJECTED"
                dist_m = 1e9
                yaw_err_deg = 180.0
                ok = 0
                failed += 1
                print(
                    f"[V35A_NAV2_SEG] id={goal_id} status={status} time_s={time.time()-t0:.3f} dist_m={dist_m:.3f} yaw_err_deg={yaw_err_deg:.2f} ok={ok}",
                    flush=True,
                )
                result_rows.append(
                    {
                        "id": goal_id,
                        "status": status,
                        "time_s": float(time.time() - t0),
                        "dist_m": float(dist_m),
                        "yaw_err_deg": float(yaw_err_deg),
                        "ok": int(ok),
                    }
                )
                continue

            rf = gh.get_result_async()
            deadline = float(t0 + float(args.timeout_s))
            assist_enabled = int(args.assist_cmdvel) == 1
            assist_iters = 0
            while time.time() < deadline and not rf.done():
                rclpy.spin_once(node, timeout_sec=float(args.assist_dt))
                if not assist_enabled:
                    continue
                cur = node.lookup_pose()
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
                    if abs(goal_yaw_err) > 0.22:
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
            if assist_enabled:
                tw0 = Twist()
                node.cmd_pub.publish(tw0)
                print(f"[V35A_ASSIST_CMDVEL] goal_id={goal_id} enabled=1 iterations={assist_iters}", flush=True)
            else:
                print(f"[V35A_ASSIST_CMDVEL] goal_id={goal_id} enabled=0 iterations=0", flush=True)
            cur = node.lookup_pose()
            if cur is None:
                dist_m = 1e9
                yaw_err_deg = 180.0
            else:
                cx, cy, cyaw = cur
                dist_m = float(math.hypot(cx - gx, cy - gy))
                yaw_err_deg = float(abs(math.degrees(_wrap_pi(cyaw - gyaw))))

            status = "TIMEOUT" if result is None else _status_name(int(result.status))
            near_goal = int(dist_m <= float(args.dist_tol_m) and yaw_err_deg <= float(args.yaw_tol_deg))
            if status == "TIMEOUT" and near_goal == 1:
                status = "TIMEOUT_NEAR_GOAL"
            ok = int((status in {"SUCCEEDED", "TIMEOUT_NEAR_GOAL"}) and near_goal == 1)
            if ok == 1:
                succeeded += 1
            else:
                failed += 1

            print(
                f"[V35A_NAV2_SEG] id={goal_id} status={status} time_s={dt:.3f} dist_m={dist_m:.3f} yaw_err_deg={yaw_err_deg:.2f} ok={ok}",
                flush=True,
            )
            result_rows.append(
                {
                    "id": goal_id,
                    "status": status,
                    "time_s": float(dt),
                    "dist_m": float(dist_m),
                    "yaw_err_deg": float(yaw_err_deg),
                    "ok": int(ok),
                }
            )

        total = int(len(goals))
        ok_all = int(succeeded == total)
        print(f"[V35A_NAV2_STAGED] total={total} succeeded={succeeded} failed={failed} ok={ok_all}", flush=True)

        if str(args.result_json).strip():
            out = Path(args.result_json).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {
                        "total": total,
                        "succeeded": int(succeeded),
                        "failed": int(failed),
                        "ok": int(ok_all),
                        "segments": result_rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        return 0 if ok_all == 1 else 5
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
