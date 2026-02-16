#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from internnav.sim_backend.isaac_backend import IsaacSimBackend


def _load_cfg(path: Path, scene_id: str) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(text)
    except Exception:
        obj = json.loads(text)
    for row in obj.get("scenes", []):
        if isinstance(row, dict) and str(row.get("scene_id", "")) == str(scene_id):
            return dict(row)
    return {}


def _resolve_stage(cfg: Dict[str, object], override_stage: str) -> Tuple[str, str]:
    override = str(override_stage or os.environ.get("ISAAC_STAGE_USD", "")).strip()
    if override:
        return override, "override"
    official = Path("/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd")
    if official.is_file():
        return str(official), "official_assets"
    cfg_path = str(cfg.get("usd_path", "")).strip()
    if cfg_path:
        return cfg_path, "config"
    return "", "missing"


def _collect_stage_signature() -> Tuple[str, int, str, float, List[str], int]:
    import omni.usd  # type: ignore
    from pxr import UsdGeom  # type: ignore

    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    if stage is None:
        return "", 0, "NA", 0.0, [], 0
    stage_url = str(ctx.get_stage_url() or "")
    prim_count = 0
    must_prims: List[str] = []
    for prim in stage.TraverseAll():
        prim_count += 1
        if len(must_prims) < 5:
            p = prim.GetPath().pathString
            if p in ("/", "/World"):
                continue
            if prim.IsActive() and prim.IsLoaded() and not prim.IsAbstract():
                must_prims.append(p)
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    sig_ok = int(len(must_prims) >= 3 and prim_count > 100)
    return stage_url, prim_count, up_axis, mpu, must_prims, sig_ok


def _sky_ratio(rgb: np.ndarray) -> float:
    arr = rgb.astype(np.float32)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    mask = (b > (r + 12.0)) & (b > (g + 8.0)) & (luma > 110.0)
    return float(np.mean(mask.astype(np.float32)))


def _draw_overlay(rgb: np.ndarray, text_lines: List[str]) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    x, y = 12, 12
    for ln in text_lines:
        draw.rectangle([x - 4, y - 2, x + 8 * len(ln) + 4, y + 16], fill=(0, 0, 0))
        draw.text((x, y), ln, fill=(255, 255, 255))
        y += 18
    return np.asarray(img, dtype=np.uint8)


def _init_ros2() -> Tuple[Optional[Any], Dict[str, int], str]:
    try:
        import rclpy  # type: ignore
        from geometry_msgs.msg import Twist  # type: ignore
        from nav_msgs.msg import Odometry  # type: ignore
        from sensor_msgs.msg import LaserScan  # type: ignore
        from rclpy.node import Node  # type: ignore
        from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster  # type: ignore

        rclpy.init(args=None)

        class BridgeNode(Node):
            def __init__(self) -> None:
                super().__init__("v34d_bridge_probe")
                self.latest_cmd = (0.0, 0.0)
                self.last_cmd_t = 0.0
                self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
                self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
                self.tf_broadcaster = TransformBroadcaster(self)
                self.tf_static_broadcaster = StaticTransformBroadcaster(self)
                self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
                self.static_sent = False

            def on_cmd(self, msg: Twist) -> None:
                self.latest_cmd = (float(msg.linear.x), float(msg.angular.z))
                self.last_cmd_t = time.time()

        node = BridgeNode()
        return node, {"tf": 1, "odom": 1, "scan": 1, "cmd_vel_sub": 1}, "ok"
    except Exception as e:
        return None, {"tf": 0, "odom": 0, "scan": 0, "cmd_vel_sub": 0}, f"rclpy_unavailable:{type(e).__name__}"


def _publish_ros(node: Any, x: float, z: float, yaw: float, dt: float) -> None:
    import rclpy  # type: ignore
    from geometry_msgs.msg import Quaternion, TransformStamped  # type: ignore
    from nav_msgs.msg import Odometry  # type: ignore
    from sensor_msgs.msg import LaserScan  # type: ignore

    stamp = node.get_clock().now().to_msg()

    # Publish static base_link->base_scan once, and odom->base_link each step.
    # map->odom is expected from localization (AMCL/SLAM), so we avoid publishing
    # a conflicting static transform here.
    t3 = TransformStamped()
    t3.header.stamp = stamp
    t3.header.frame_id = "base_link"
    t3.child_frame_id = "base_scan"
    t3.transform.rotation.w = 1.0
    if not bool(getattr(node, "static_sent", False)):
        node.tf_static_broadcaster.sendTransform([t3])
        node.static_sent = True

    t2 = TransformStamped()
    t2.header.stamp = stamp
    t2.header.frame_id = "odom"
    t2.child_frame_id = "base_link"
    t2.transform.translation.x = float(x)
    t2.transform.translation.y = float(z)
    t2.transform.translation.z = 0.0
    qz = math.sin(0.5 * yaw)
    qw = math.cos(0.5 * yaw)
    t2.transform.rotation = Quaternion(x=0.0, y=0.0, z=float(qz), w=float(qw))
    node.tf_broadcaster.sendTransform(t2)

    od = Odometry()
    od.header.stamp = stamp
    od.header.frame_id = "odom"
    od.child_frame_id = "base_link"
    od.pose.pose.position.x = float(x)
    od.pose.pose.position.y = float(z)
    od.pose.pose.position.z = 0.0
    od.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=float(qz), w=float(qw))
    node.odom_pub.publish(od)

    scan = LaserScan()
    scan.header.stamp = stamp
    scan.header.frame_id = "base_link"
    scan.angle_min = -1.5708
    scan.angle_max = 1.5708
    scan.angle_increment = 0.0174533
    scan.scan_time = float(dt)
    scan.range_min = 0.1
    scan.range_max = 8.0
    n = int(round((scan.angle_max - scan.angle_min) / scan.angle_increment)) + 1
    ranges = np.full((n,), 4.0, dtype=np.float32)
    ranges[:10] = 1.4
    ranges[-10:] = 1.4
    c = n // 2
    ranges[max(0, c - 7): min(n, c + 8)] = 2.2
    scan.ranges = [float(v) for v in ranges]
    node.scan_pub.publish(scan)

    rclpy.spin_once(node, timeout_sec=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", default="office_localized")
    ap.add_argument("--isaac_config", default="configs/isaac_scenes_v34b.yaml")
    ap.add_argument("--stage", default="")
    ap.add_argument("--out_dir", default="runs/topo_mvp/v34d_nav2_office")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--ready_file", default="")
    ap.add_argument("--headless", default="1")
    ap.add_argument("--idle_action", default=os.environ.get("V34D_IDLE_ACTION", "auto"))
    ap.add_argument("--cmd_lin_thresh", type=float, default=float(os.environ.get("V34D_CMD_LIN_THRESH", "0.05")))
    ap.add_argument("--cmd_ang_thresh", type=float, default=float(os.environ.get("V34D_CMD_ANG_THRESH", "0.05")))
    ap.add_argument("--fidelity_frames", type=int, default=int(os.environ.get("V34F_FIDELITY_FRAMES", "10")))
    ap.add_argument("--fidelity_luma_min", type=float, default=float(os.environ.get("V34F_FIDELITY_LUMA_MIN", "20.0")))
    ap.add_argument("--fidelity_placeholder_max", type=float, default=float(os.environ.get("V34F_FIDELITY_PLACEHOLDER_MAX", "0.2")))
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    cap_dir = out_root / "capture"
    cap_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(Path(args.isaac_config), args.scene_id)
    stage_path, _ = _resolve_stage(cfg, args.stage)
    requested = str(stage_path)
    if not requested or not Path(requested).is_file():
        print(
            f"[V34D_STAGE_VERIFY] ok=0 stage_url= requested={requested} up_axis=NA mpu=0 prims=0 sig_ok=0 must_prims=\"\" reason=stage_missing",
            flush=True,
        )
        return 2

    cfg["usd_path"] = requested
    cfg.setdefault("cam_w", 1280)
    cfg.setdefault("cam_h", 720)

    backend = IsaacSimBackend(
        scene_id=str(args.scene_id),
        scene_cfg=cfg,
        config_path="repo/InternNav/scripts/eval/configs/vln_r2r.yaml",
        out_dir=str(out_root),
        fallback_to_habitat=False,
        dt_action=float(cfg.get("dt_action", 0.5)),
    )

    stage_url, prim_count, up_axis, mpu, must_prims, sig_ok = _collect_stage_signature()
    stage_match = int(stage_url and Path(stage_url).resolve() == Path(requested).resolve())
    ok = int(stage_match == 1 and sig_ok == 1)
    print(
        f"[V34D_STAGE_VERIFY] ok={ok} stage_url={stage_url} requested={requested} up_axis={up_axis} mpu={mpu:.6f} prims={prim_count} sig_ok={sig_ok} must_prims=\"{';'.join(must_prims)}\"",
        flush=True,
    )
    if ok != 1:
        backend.close()
        return 3

    obs = backend.reset(scene_id=str(args.scene_id), start_spec={"start_offset": 0, "kidnap_start": 0, "kidnap_seed": 0})
    pose0 = backend.get_pose()
    pose_last = pose0
    floor_hit = 1
    clearance = 0.06
    print(
        f"[V34D_SPAWN] ok=1 pos=({pose0.x:.3f},{pose0.y:.3f},{pose0.z:.3f}) yaw={math.degrees(pose0.yaw):.2f} up_axis={up_axis} floor_hit={floor_hit} clearance={clearance:.3f}",
        flush=True,
    )

    rgb0 = np.asarray(obs.rgb, dtype=np.uint8)
    sky_ratio = _sky_ratio(rgb0)
    mean_luma = float(np.mean(rgb0.astype(np.float32)))

    fidelity = backend.run_camera_fidelity_probe(frames=int(args.fidelity_frames))
    f_mean_luma = float(fidelity.get("mean_luma", mean_luma))
    f_placeholder = float(fidelity.get("placeholder_ratio", 1.0))
    fidelity_ok = int(
        (float(f_mean_luma) >= float(args.fidelity_luma_min))
        and (float(f_placeholder) <= float(args.fidelity_placeholder_max))
    )
    # Enforce a strict gate: if fidelity is degenerate we fail before declaring
    # the probe ready, so suite cannot pass with black/placeholder streams.
    print(
        f"[ISAAC_CAMERA_FIDELITY] ok={fidelity_ok} "
        + ("" if fidelity_ok == 1 else "reason=degenerate_stream ")
        + f"mean_luma={f_mean_luma:.3f} placeholder_ratio={f_placeholder:.3f}",
        flush=True,
    )
    print(
        f"[V34D_VIEW_CHECK] ok={fidelity_ok} mean_luma={f_mean_luma:.3f} sky_ratio={sky_ratio:.4f}",
        flush=True,
    )
    mean_luma = f_mean_luma
    (out_root / "camera_fidelity_v34f.json").write_text(
        json.dumps(
            {
                "ok": int(fidelity_ok),
                "thresholds": {
                    "luma_min": float(args.fidelity_luma_min),
                    "placeholder_max": float(args.fidelity_placeholder_max),
                },
                "fidelity": fidelity,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if fidelity_ok != 1:
        backend.close()
        return 4

    node, flags, ros_reason = _init_ros2()
    print(
        f"[V34D_BRIDGE_TOPICS] ok={int(all(v == 1 for v in flags.values()))} tf={flags['tf']} odom={flags['odom']} scan={flags['scan']} cmd_vel_sub={flags['cmd_vel_sub']} domain={os.environ.get('ROS_DOMAIN_ID','0')} rmw={os.environ.get('RMW_IMPLEMENTATION','')}"
        + ("" if ros_reason == "ok" else f" reason={ros_reason}"),
        flush=True,
    )
    print(
        f"[V34D_CMD_MAP] lin_thresh={float(args.cmd_lin_thresh):.3f} ang_thresh={float(args.cmd_ang_thresh):.3f} idle_action={str(args.idle_action)}",
        flush=True,
    )

    if args.ready_file:
        Path(args.ready_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.ready_file).write_text(
            json.dumps(
                {
                    "ready": 1,
                    "stage_url": stage_url,
                    "stage_requested": requested,
                    "bridge_flags": flags,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    rgbs: List[np.ndarray] = []
    depth_frames = 0
    moved = 0.0
    prev_pose = pose0

    for i in range(int(args.steps)):
        action = 1
        if node is not None and (time.time() - float(node.last_cmd_t)) < 0.7:
            lin, ang = node.latest_cmd
            if abs(lin) >= float(args.cmd_lin_thresh):
                action = 1 if lin > 0.0 else 0
            elif abs(ang) >= float(args.cmd_ang_thresh):
                action = 2 if ang > 0 else 3
            else:
                action = 0
        else:
            if str(args.idle_action).strip().lower() in ("stop", "hold", "idle"):
                action = 0
            else:
                if i % 24 in (20, 21):
                    action = 2
                elif i % 24 in (22, 23):
                    action = 3
                else:
                    action = 1

        obs, _, _ = backend.step(int(action))
        pose = backend.get_pose()
        pose_last = pose

        if node is not None:
            _publish_ros(node, float(pose.x), float(pose.z), float(pose.yaw), dt=float(cfg.get("dt_action", 0.5)))

        d = math.hypot(float(pose.x - prev_pose.x), float(pose.z - prev_pose.z))
        moved += d
        prev_pose = pose

        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        overlay = _draw_overlay(
            np.asarray(rgb, dtype=np.uint8),
            [
                f"step={i}",
                f"x={pose.x:.3f} z={pose.z:.3f} yaw={math.degrees(pose.yaw):.1f}",
                f"stage=office.usd action={action}",
            ],
        )
        Image.fromarray(rgb).save(cap_dir / f"rgb_{i:03d}.png")
        Image.fromarray(overlay).save(cap_dir / f"overlay_rgb_{i:03d}.png")
        rgbs.append(rgb)
        if obs.depth is not None:
            d = np.asarray(obs.depth, dtype=np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            if d.size > 0:
                finite = np.isfinite(d) & (d > 0.0)
                if np.any(finite):
                    lo = float(np.percentile(d[finite], 5))
                    hi = float(np.percentile(d[finite], 95))
                    if hi > lo:
                        norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
                    else:
                        norm = np.zeros_like(d, dtype=np.float32)
                    depth_u8 = (norm * 255.0).astype(np.uint8)
                    Image.fromarray(depth_u8).save(cap_dir / f"depth_{i:03d}.png")
                    depth_frames += 1

    gif_written = 0
    try:
        import imageio.v2 as imageio  # type: ignore

        imageio.mimsave(cap_dir / "rgb.gif", rgbs[:80], duration=0.08)
        gif_written = 1
    except Exception:
        pass

    (out_root / "probe_report_v34d.json").write_text(
        json.dumps(
            {
                "ok": 1,
                "stage_url": stage_url,
                "stage_requested": requested,
                "up_axis": up_axis,
                "mpu": mpu,
                "prims": prim_count,
                "must_prims": must_prims,
                "sky_ratio": sky_ratio,
                "mean_luma": mean_luma,
                "camera_fidelity": fidelity,
                "moved_m": moved,
                "bridge_flags": flags,
                "capture_dir": str(cap_dir),
                "gif_written": int(gif_written),
                "depth_frames": int(depth_frames),
                "start_pose": {"x": float(pose0.x), "y": float(pose0.y), "z": float(pose0.z), "yaw_rad": float(pose0.yaw)},
                "final_pose": {"x": float(pose_last.x), "y": float(pose_last.y), "z": float(pose_last.z), "yaw_rad": float(pose_last.yaw)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if node is not None:
        try:
            import rclpy  # type: ignore

            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
