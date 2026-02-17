#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from typing import Optional


def _quat_norm(x: float, y: float, z: float, w: float) -> tuple[float, float, float, float]:
    n = math.sqrt(max(1e-12, x * x + y * y + z * z + w * w))
    return (x / n, y / n, z / n, w / n)


def main() -> int:
    ap = argparse.ArgumentParser(description="Republish /odom pose as dynamic odom->base_link TF")
    ap.add_argument("--odom_topic", default="/odom")
    ap.add_argument("--parent_frame", default="odom")
    ap.add_argument("--child_frame", default="base_link")
    ap.add_argument("--publish_base_scan_static", type=int, default=0)
    ap.add_argument("--scan_child_frame", default="base_scan")
    ap.add_argument("--hz_report_period_s", type=float, default=3.0)
    args = ap.parse_args()

    try:
        import rclpy  # type: ignore
        from geometry_msgs.msg import TransformStamped  # type: ignore
        from nav_msgs.msg import Odometry  # type: ignore
        from rclpy.node import Node  # type: ignore
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster  # type: ignore
    except Exception as e:  # pragma: no cover - runtime environment dependent
        print(
            f"[V34I_ODOM_TF_BRIDGE] ok=0 hz=0.000 frames=\"{args.parent_frame}->{args.child_frame}\" dynamic=0 reason=rclpy_missing:{type(e).__name__}",
            flush=True,
        )
        return 2

    class OdomTfBridge(Node):
        def __init__(self) -> None:
            super().__init__("v34i_odom_tf_bridge")
            self.br = TransformBroadcaster(self)
            self.br_static = StaticTransformBroadcaster(self)
            self.msg_count = 0
            self.last_report_t = time.time()
            self.last_msg_t = 0.0
            self.create_subscription(Odometry, str(args.odom_topic), self.on_odom, 20)

            if int(args.publish_base_scan_static) == 1:
                ts = TransformStamped()
                ts.header.stamp = self.get_clock().now().to_msg()
                ts.header.frame_id = str(args.child_frame)
                ts.child_frame_id = str(args.scan_child_frame)
                ts.transform.rotation.w = 1.0
                self.br_static.sendTransform(ts)

            # Startup anchor.
            print(
                f"[V34I_ODOM_TF_BRIDGE] ok=1 hz=0.000 frames=\"{args.parent_frame}->{args.child_frame}\" dynamic=1",
                flush=True,
            )

        def on_odom(self, msg: Odometry) -> None:
            ts = TransformStamped()
            ts.header.stamp = msg.header.stamp
            ts.header.frame_id = str(args.parent_frame)
            ts.child_frame_id = str(args.child_frame)
            ts.transform.translation.x = float(msg.pose.pose.position.x)
            ts.transform.translation.y = float(msg.pose.pose.position.y)
            ts.transform.translation.z = float(msg.pose.pose.position.z)
            q = msg.pose.pose.orientation
            qx, qy, qz, qw = _quat_norm(float(q.x), float(q.y), float(q.z), float(q.w))
            ts.transform.rotation.x = qx
            ts.transform.rotation.y = qy
            ts.transform.rotation.z = qz
            ts.transform.rotation.w = qw
            self.br.sendTransform(ts)

            self.msg_count += 1
            self.last_msg_t = time.time()
            now = time.time()
            if (now - self.last_report_t) >= float(args.hz_report_period_s):
                hz = float(self.msg_count) / max(1e-6, (now - self.last_report_t))
                print(
                    f"[V34I_ODOM_TF_BRIDGE] ok=1 hz={hz:.3f} frames=\"{args.parent_frame}->{args.child_frame}\" dynamic=1",
                    flush=True,
                )
                self.last_report_t = now
                self.msg_count = 0

    rclpy.init(args=None)
    node: Optional[OdomTfBridge] = None
    try:
        node = OdomTfBridge()
        rclpy.spin(node)
        return 0
    except KeyboardInterrupt:
        return 0
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
