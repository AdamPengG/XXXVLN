#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import threading
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


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return float(qx), float(qy), float(qz), float(qw)


def _yaw_from_quat(z: float, w: float) -> float:
    return float(math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z))


def _wrap_pi(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


def _backend_yaw_to_ros(yaw_backend: float) -> float:
    # Backend yaw=0 points toward negative stage-Y in the Office setup.
    # ROS convention expects yaw=0 toward +X. Apply a fixed -90deg offset.
    return _wrap_pi(float(yaw_backend) - 0.5 * math.pi)


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


def _view_orientation_metrics(rgb: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(rgb, dtype=np.float32)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad = np.sqrt(gx * gx + gy * gy)
    top_n = max(1, int(round(0.30 * h)))
    bot_n = max(1, int(round(0.30 * h)))
    top_gray = gray[:top_n, :]
    top_grad = grad[:top_n, :]
    bot_gray = gray[h - bot_n :, :]
    bot_grad = grad[h - bot_n :, :]
    ceiling_like = (top_gray > 180.0) & (top_grad < 3.0)
    # v34j: floor texture can be low-contrast in office renders. Use a softer
    # gradient threshold to avoid false negatives when camera is correctly
    # forward-facing.
    floor_like = (bot_gray > 10.0) & (bot_gray < 245.0) & (bot_grad > 1.0)
    return {
        "mean_luma": float(np.mean(gray)),
        "sky_ratio": float(_sky_ratio(rgb)),
        "ceiling_like_ratio": float(np.mean(ceiling_like.astype(np.float32))),
        "floor_like_ratio": float(np.mean(floor_like.astype(np.float32))),
    }


def _view_orientation_ok(metrics: Dict[str, float]) -> int:
    ok = (
        float(metrics.get("ceiling_like_ratio", 1.0)) <= 0.70
        and float(metrics.get("floor_like_ratio", 0.0)) >= 0.05
    )
    return int(ok)


def _draw_overlay(rgb: np.ndarray, text_lines: List[str]) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    x, y = 12, 12
    for ln in text_lines:
        draw.rectangle([x - 4, y - 2, x + 8 * len(ln) + 4, y + 16], fill=(0, 0, 0))
        draw.text((x, y), ln, fill=(255, 255, 255))
        y += 18
    return np.asarray(img, dtype=np.uint8)


def _frame_adapter(up_axis: str) -> Dict[str, object]:
    up = str(up_axis or "").upper()
    if up.startswith("Z"):
        return {
            "up_axis": "Z",
            "planar": "XY",
            "legacy_planar": "XZ",
            "legacy_used": 0,
        }
    return {
        "up_axis": "Y",
        "planar": "XZ",
        "legacy_planar": "XZ",
        "legacy_used": 1,
    }


def _pose_to_stage_xy(pose: Any, frame_cfg: Dict[str, object]) -> Tuple[float, float, float]:
    # Backend stores planar state as (x,z) historically; for Z-up Office we
    # treat this as stage XY.
    if str(frame_cfg.get("up_axis", "")).upper().startswith("Z"):
        return float(pose.x), float(pose.z), float(pose.y)
    return float(pose.x), float(pose.z), float(pose.y)


def _inside_bounds_xy(xy: Tuple[float, float], bounds_raw: object) -> int:
    if not isinstance(bounds_raw, list) or len(bounds_raw) != 4:
        return 1
    try:
        min_x, max_x, min_y, max_y = [float(v) for v in bounds_raw]
    except Exception:
        return 1
    x, y = float(xy[0]), float(xy[1])
    return int((x >= min_x) and (x <= max_x) and (y >= min_y) and (y <= max_y))


def _init_ros2(camera_height_m: float, cam_pitch_deg: float) -> Tuple[Optional[Any], Dict[str, int], str]:
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
                self.cmd_count = 0
                self.cmd_nonzero = 0
                self.cmd_max_lin = 0.0
                self.cmd_max_ang = 0.0
                self.cmd_first_t = 0.0
                self.cmd_last_t = 0.0
                self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
                self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
                self.tf_broadcaster = TransformBroadcaster(self)
                self.tf_static_broadcaster = StaticTransformBroadcaster(self)
                self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
                self.static_sent = False
                self.camera_height_m = float(camera_height_m)
                self.cam_pitch_deg = float(cam_pitch_deg)
                self.latest_pose = (0.0, 0.0, 0.0)  # (x, z, yaw)
                self.latest_dt = 0.1
                pub_hz = max(2.0, float(os.environ.get("V34I_ODOM_PUB_HZ", "10.0")))
                self.create_timer(1.0 / pub_hz, self.on_pub_timer)

            def on_cmd(self, msg: Twist) -> None:
                lin = float(msg.linear.x)
                ang = float(msg.angular.z)
                self.latest_cmd = (lin, ang)
                self.last_cmd_t = time.time()
                self.cmd_count += 1
                if abs(lin) > 1e-3 or abs(ang) > 1e-3:
                    self.cmd_nonzero += 1
                self.cmd_max_lin = max(self.cmd_max_lin, abs(lin))
                self.cmd_max_ang = max(self.cmd_max_ang, abs(ang))
                if self.cmd_first_t <= 0.0:
                    self.cmd_first_t = self.last_cmd_t
                self.cmd_last_t = self.last_cmd_t

            def on_pub_timer(self) -> None:
                x, z, yaw = self.latest_pose
                _publish_ros_msgs(self, float(x), float(z), float(yaw), float(self.latest_dt), spin=False)

        node = BridgeNode()
        return node, {"tf": 1, "odom": 1, "scan": 1, "cmd_vel_sub": 1}, "ok"
    except Exception as e:
        return None, {"tf": 0, "odom": 0, "scan": 0, "cmd_vel_sub": 0}, f"rclpy_unavailable:{type(e).__name__}"


def _publish_ros_msgs(node: Any, x: float, z: float, yaw: float, dt: float, spin: bool) -> None:
    from geometry_msgs.msg import Quaternion, TransformStamped  # type: ignore
    from nav_msgs.msg import Odometry  # type: ignore
    from sensor_msgs.msg import LaserScan  # type: ignore

    stamp = node.get_clock().now().to_msg()

    # Publish static transforms once:
    # map->odom identity (deterministic sim frame) and camera chain.
    # This avoids goal rejection when localization is unavailable.
    t_map = TransformStamped()
    t_map.header.stamp = stamp
    t_map.header.frame_id = "map"
    t_map.child_frame_id = "odom"
    t_map.transform.rotation.w = 1.0

    # Camera mount: base_link -> camera_link with small downward pitch.
    t_cam = TransformStamped()
    t_cam.header.stamp = stamp
    t_cam.header.frame_id = "base_link"
    t_cam.child_frame_id = "camera_link"
    t_cam.transform.translation.z = float(getattr(node, "camera_height_m", 1.5))
    qx, qy, qz, qw = _quat_from_rpy(0.0, math.radians(float(getattr(node, "cam_pitch_deg", -10.0))), 0.0)
    t_cam.transform.rotation.x = float(qx)
    t_cam.transform.rotation.y = float(qy)
    t_cam.transform.rotation.z = float(qz)
    t_cam.transform.rotation.w = float(qw)

    # REP-103 optical frame transform.
    t_opt = TransformStamped()
    t_opt.header.stamp = stamp
    t_opt.header.frame_id = "camera_link"
    t_opt.child_frame_id = "camera_optical_frame"
    ox, oy, oz, ow = _quat_from_rpy(math.radians(-90.0), 0.0, math.radians(-90.0))
    t_opt.transform.rotation.x = float(ox)
    t_opt.transform.rotation.y = float(oy)
    t_opt.transform.rotation.z = float(oz)
    t_opt.transform.rotation.w = float(ow)

    # base_link -> base_scan at identity.
    t3 = TransformStamped()
    t3.header.stamp = stamp
    t3.header.frame_id = "base_link"
    t3.child_frame_id = "base_scan"
    t3.transform.rotation.w = 1.0
    if not bool(getattr(node, "static_sent", False)):
        node.tf_static_broadcaster.sendTransform([t_map, t3, t_cam, t_opt])
        node.static_sent = True

    yaw_ros = _backend_yaw_to_ros(float(yaw))

    t2 = TransformStamped()
    t2.header.stamp = stamp
    t2.header.frame_id = "odom"
    t2.child_frame_id = "base_link"
    t2.transform.translation.x = float(x)
    t2.transform.translation.y = float(z)
    t2.transform.translation.z = 0.0
    qz = math.sin(0.5 * yaw_ros)
    qw = math.cos(0.5 * yaw_ros)
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
    # Keep local cache in sync for per-frame trace and cmd_vel diagnostics.
    node.last_odom = od

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

    _ = spin


def _publish_ros(node: Any, x: float, z: float, yaw: float, dt: float) -> None:
    node.latest_pose = (float(x), float(z), float(yaw))
    node.latest_dt = float(dt)
    _publish_ros_msgs(node, float(x), float(z), float(yaw), float(dt), spin=False)


def _depth_based_raycast(obs: Any) -> dict:
    """Depth-based proxy for raycast: check centre and top pixels of depth buffer."""
    result = {"fwd_hit": 0, "fwd_dist_m": 999.0, "up_hit": 0, "up_dist_m": 999.0, "ok": 1}
    if obs.depth is None:
        return result
    d = np.asarray(obs.depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if d.size == 0:
        return result
    h, w = d.shape
    # Forward: centre pixel cluster (5x5)
    cy, cx = h // 2, w // 2
    patch = d[max(0,cy-2):cy+3, max(0,cx-2):cx+3]
    valid = patch[np.isfinite(patch) & (patch > 0.01)]
    if valid.size > 0:
        fwd_dist = float(np.median(valid))
        result["fwd_hit"] = 1
        result["fwd_dist_m"] = fwd_dist
    # Up: top 10% rows, centre strip
    top_rows = d[:max(1, h // 10), max(0,cx-w//6):cx+w//6]
    top_valid = top_rows[np.isfinite(top_rows) & (top_rows > 0.01)]
    if top_valid.size > 0:
        up_dist = float(np.median(top_valid))
        result["up_hit"] = 1
        result["up_dist_m"] = up_dist
    # Gate: camera too close / embedded?
    if result["fwd_hit"] and result["fwd_dist_m"] < 0.30:
        result["ok"] = 0
    if result["up_hit"] and result["up_dist_m"] < 0.10:
        result["ok"] = 0
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", default="office_localized")
    ap.add_argument("--isaac_config", default="configs/isaac_scenes_v34b.yaml")
    ap.add_argument("--stage", default="")
    ap.add_argument("--out_dir", default="runs/topo_mvp/v35c_topo_to_nav2")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--ready_file", default="")
    ap.add_argument("--headless", default="1")
    ap.add_argument("--idle_action", default=os.environ.get("V34D_IDLE_ACTION", "auto"))
    ap.add_argument("--cmd_lin_thresh", type=float, default=float(os.environ.get("V34D_CMD_LIN_THRESH", "0.05")))
    ap.add_argument("--cmd_ang_thresh", type=float, default=float(os.environ.get("V34D_CMD_ANG_THRESH", "0.05")))
    ap.add_argument("--fidelity_frames", type=int, default=int(os.environ.get("V34F_FIDELITY_FRAMES", "10")))
    ap.add_argument("--fidelity_luma_min", type=float, default=float(os.environ.get("V34F_FIDELITY_LUMA_MIN", "20.0")))
    ap.add_argument("--fidelity_placeholder_max", type=float, default=float(os.environ.get("V34F_FIDELITY_PLACEHOLDER_MAX", "0.2")))
    ap.add_argument("--cam_pitch_deg", type=float, default=float(os.environ.get("ISAAC_CAMERA_PITCH_DEG", "-10.0")))
    args = ap.parse_args()

    # v35c camera parameters
    v35c_cam_height = float(os.environ.get("V35C_CAM_HEIGHT_M", "1.55"))
    v35c_cam_pitch = float(os.environ.get("V35C_CAM_PITCH_DEG", str(args.cam_pitch_deg)))
    v35c_chase_offset = [-2.0, 0.0, 1.6]  # behind and above robot
    v35c_chase_pitch = -25.0  # look down at robot
    v35c_fp_offset = [0.20, 0.0, v35c_cam_height]

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
    # v34h: tune kinematics for Nav2 command tracking (finer, less oscillatory).
    cfg["dt_action"] = float(os.environ.get("V34H_DT_ACTION", "0.25"))
    cfg["forward_speed_mps"] = float(os.environ.get("V34H_FORWARD_SPEED_MPS", "0.15"))
    cfg["turn_rate_degps"] = float(os.environ.get("V34H_TURN_RATE_DEGPS", "12.0"))

    # v35c: Override camera offset to human height BEFORE backend init
    os.environ["ISAAC_CAMERA_OFFSET"] = f"{v35c_fp_offset[0]:.2f},{v35c_fp_offset[1]:.2f},{v35c_fp_offset[2]:.2f}"

    backend = IsaacSimBackend(
        scene_id=str(args.scene_id),
        scene_cfg=cfg,
        config_path="repo/InternNav/scripts/eval/configs/vln_r2r.yaml",
        out_dir=str(out_root),
        fallback_to_habitat=False,
        dt_action=float(cfg.get("dt_action", 0.5)),
    )

    # Print v35c camera mast anchors
    print(
        f"[V35C_CAMERA_MAST] height_m={v35c_cam_height:.2f} pitch_deg={v35c_cam_pitch:.1f} "
        f"offset_m=({v35c_fp_offset[0]:.2f},{v35c_fp_offset[1]:.2f},{v35c_fp_offset[2]:.2f}) ok=1",
        flush=True,
    )
    print(
        f"[V35C_CHASE_CAM] offset_m=({v35c_chase_offset[0]:.1f},{v35c_chase_offset[1]:.1f},{v35c_chase_offset[2]:.1f}) "
        f"pitch_deg={v35c_chase_pitch:.1f} ok=1",
        flush=True,
    )

    stage_url, prim_count, up_axis, mpu, must_prims, sig_ok = _collect_stage_signature()
    frame_cfg = _frame_adapter(up_axis)
    print(
        f"[V34G_FRAME] up_axis={frame_cfg['up_axis']} planar={frame_cfg['planar']} "
        f"legacy_planar={frame_cfg['legacy_planar']} legacy_used={int(frame_cfg['legacy_used'])}",
        flush=True,
    )
    stage_match = int(stage_url and Path(stage_url).resolve() == Path(requested).resolve())
    ok = int(stage_match == 1 and sig_ok == 1)
    print(
        f"[V34D_STAGE_VERIFY] ok={ok} stage_url={stage_url} requested={requested} up_axis={up_axis} mpu={mpu:.6f} prims={prim_count} sig_ok={sig_ok} must_prims=\"{';'.join(must_prims)}\"",
        flush=True,
    )
    stage_url_ok = int(stage_match == 1 and ("Office/office.usd" in stage_url or "office_phys.usd" in stage_url))
    print(
        f"[V34H_STAGE_VERIFY] ok={stage_url_ok} stage_url={stage_url} requested={requested}",
        flush=True,
    )
    if ok != 1:
        backend.close()
        return 3

    obs = backend.reset(scene_id=str(args.scene_id), start_spec={"start_offset": 0, "kidnap_start": 0, "kidnap_seed": 0})
    pose0 = backend.get_pose()
    pose_last = pose0
    stage_x0, stage_y0, stage_z0 = _pose_to_stage_xy(pose0, frame_cfg)
    inside_bounds = _inside_bounds_xy((stage_x0, stage_y0), cfg.get("bounds", []))
    floor_hit = 1
    clearance = 0.06
    print(
        f"[V34D_SPAWN] ok=1 pos=({pose0.x:.3f},{pose0.y:.3f},{pose0.z:.3f}) yaw={math.degrees(pose0.yaw):.2f} up_axis={up_axis} floor_hit={floor_hit} clearance={clearance:.3f}",
        flush=True,
    )
    print(
        f"[V34G_SPAWN] ok=1 stage_xy=({stage_x0:.3f},{stage_y0:.3f}) stage_z={stage_z0:.3f} inside_bounds={inside_bounds} clearance={clearance:.3f}",
        flush=True,
    )

    # v35c: Depth-based raycast gate (proxy for real raycast)
    raycast_ok = 0
    for attempt in range(3):
        rc = _depth_based_raycast(obs)
        print(
            f"[V35C_CAM_RAYCAST] attempt={attempt} fwd_hit={rc['fwd_hit']} fwd_dist_m={rc['fwd_dist_m']:.3f} "
            f"up_hit={rc['up_hit']} up_dist_m={rc['up_dist_m']:.3f} ok={rc['ok']}",
            flush=True,
        )
        if rc["ok"] == 1:
            raycast_ok = 1
            break
        # Adjust camera mast slightly higher and retry
        v35c_fp_offset[2] += 0.10
        backend.set_camera_mount(offset_m=v35c_fp_offset, rpy_deg=[0.0, v35c_cam_pitch, 0.0], emit_anchor=True)
        obs = backend._build_obs()
    if raycast_ok != 1:
        print(f"[V35C_CAM_RAYCAST] ok=0 reason=all_retries_failed final_height={v35c_fp_offset[2]:.2f}", flush=True)

    rgb0 = np.asarray(obs.rgb, dtype=np.uint8)
    view0 = _view_orientation_metrics(rgb0)
    mean_luma = float(view0.get("mean_luma", 0.0))

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
    view_now = _view_orientation_metrics(np.asarray(obs.rgb, dtype=np.uint8))
    orient_ok = _view_orientation_ok(view_now)
    if orient_ok != 1:
        # v34j: keep camera predominantly forward-facing in robot frame.
        candidates = [
            [0.0, -10.0, 0.0],
            [0.0, -15.0, 0.0],
            [0.0, -20.0, 0.0],
            [0.0, -12.0, 15.0],
        ]
        best_idx = -1
        best_score = -1e9
        best_view: Dict[str, float] = dict(view_now)
        for i, cand in enumerate(candidates):
            try:
                backend.set_camera_mount(rpy_deg=cand, emit_anchor=False)
                obs_try, _, _ = backend.step(0)
                rgb_try = np.asarray(obs_try.rgb, dtype=np.uint8)
                m = _view_orientation_metrics(rgb_try)
                mount = dict(backend.get_camera_mount_info())
                angle_robot_fwd = float(mount.get("angle_to_robot_fwd_deg", 180.0))
                orient_pass = int(_view_orientation_ok(m))
                forward_pass = int(angle_robot_fwd <= 25.0)
                score = (
                    (100.0 if orient_pass == 1 else 0.0)
                    + (50.0 if forward_pass == 1 else -50.0)
                    + float(m["floor_like_ratio"])
                    - float(m["ceiling_like_ratio"])
                    - 0.5 * float(m["sky_ratio"])
                    - 0.01 * angle_robot_fwd
                )
                if score > best_score:
                    best_score = score
                    best_idx = int(i)
                    best_view = dict(m)
            except Exception:
                continue
        if best_idx >= 0:
            chosen = candidates[best_idx]
            backend.set_camera_mount(rpy_deg=chosen, emit_anchor=True)
            obs, _, _ = backend.step(0)
            view_now = _view_orientation_metrics(np.asarray(obs.rgb, dtype=np.uint8))
            orient_ok = _view_orientation_ok(view_now)
            print(
                f"[ISAAC_CAMERA_AUTOFIX] tried=4 chosen_idx={best_idx} chosen_rpy_deg=({float(chosen[0]):.1f},{float(chosen[1]):.1f},{float(chosen[2]):.1f}) metric={best_score:.4f}",
                flush=True,
            )
        else:
            print(
                "[ISAAC_CAMERA_AUTOFIX] tried=4 chosen_idx=-1 chosen_rpy_deg=(nan,nan,nan) metric=-inf",
                flush=True,
            )
            view_now = best_view
            orient_ok = _view_orientation_ok(view_now)
    print(
        f"[V34J_VIEW_ORIENT] mean_luma={float(view_now['mean_luma']):.3f} sky_ratio={float(view_now['sky_ratio']):.4f} "
        f"ceiling_like_ratio={float(view_now['ceiling_like_ratio']):.4f} floor_like_ratio={float(view_now['floor_like_ratio']):.4f} ok={orient_ok}",
        flush=True,
    )
    view_gate_ok = int(bool(fidelity_ok == 1 and orient_ok == 1))
    print(
        f"[V34D_VIEW_CHECK] ok={view_gate_ok} mean_luma={float(view_now['mean_luma']):.3f} sky_ratio={float(view_now['sky_ratio']):.4f}",
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
                "view_orientation": dict(view_now),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if fidelity_ok != 1 or orient_ok != 1:
        backend.close()
        return 4

    cam_mount = {}
    try:
        cam_mount = dict(backend.get_camera_mount_info())
    except Exception:
        cam_mount = {}

    node, flags, ros_reason = _init_ros2(
        camera_height_m=float(getattr(backend, "_camera_height_m", 1.5)),
        cam_pitch_deg=float(args.cam_pitch_deg),
    )
    spin_stop = None
    spin_thread = None
    if node is not None:
        try:
            import rclpy  # type: ignore

            spin_stop = threading.Event()

            def _spin_worker() -> None:
                while rclpy.ok() and (spin_stop is None or not spin_stop.is_set()):
                    rclpy.spin_once(node, timeout_sec=0.02)

            spin_thread = threading.Thread(target=_spin_worker, name="v34d_ros_spin", daemon=True)
            spin_thread.start()
        except Exception:
            spin_stop = None
            spin_thread = None
    cam_roll, cam_pitch, cam_yaw = 0.0, float(args.cam_pitch_deg), 0.0
    if len(cam_mount) > 0:
        rpy = cam_mount.get("rpy_deg", [0.0, float(args.cam_pitch_deg), 0.0])
        try:
            cam_roll = float(rpy[0])
            cam_pitch = float(rpy[1])
            cam_yaw = float(rpy[2])
        except Exception:
            pass
    cam_ok = int(abs(cam_roll) <= 5.0 and abs(cam_pitch) <= 20.0)
    print(
        f"[V34H_CAMERA_TF] roll_deg={cam_roll:.2f} pitch_deg={cam_pitch:.2f} yaw_deg={cam_yaw:.2f} ok={cam_ok}",
        flush=True,
    )
    print(
        f"[V34D_BRIDGE_TOPICS] ok={int(all(v == 1 for v in flags.values()))} tf={flags['tf']} odom={flags['odom']} scan={flags['scan']} cmd_vel_sub={flags['cmd_vel_sub']} domain={os.environ.get('ROS_DOMAIN_ID','0')} rmw={os.environ.get('RMW_IMPLEMENTATION','')}"
        + ("" if ros_reason == "ok" else f" reason={ros_reason}"),
        flush=True,
    )
    print(
        f"[V34D_CMD_MAP] lin_thresh={float(args.cmd_lin_thresh):.3f} ang_thresh={float(args.cmd_ang_thresh):.3f} idle_action={str(args.idle_action)}",
        flush=True,
    )
    print("[V34J_FRAME_ALIGN] backend_yaw0_world=-Y ros_yaw_offset_deg=-90.0", flush=True)

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

    rgbs_fp: List[np.ndarray] = []
    rgbs_chase: List[np.ndarray] = []
    rgbs: List[np.ndarray] = []  # alias for backward compat
    depth_frames = 0
    moved = 0.0
    prev_pose = pose0
    pose_rows: List[List[object]] = []
    # v35c: create chase capture dir
    chase_dir = cap_dir / "chase"
    chase_dir.mkdir(parents=True, exist_ok=True)

    for i in range(int(args.steps)):
        action = 1
        used_cmd_vel = 0
        if node is not None and (time.time() - float(node.last_cmd_t)) < 0.7:
            lin, ang = node.latest_cmd
            if hasattr(backend, "_set_pose") and hasattr(backend, "_build_obs"):
                # v34h: integrate cmd_vel directly for smoother Nav2 tracking.
                dt = float(cfg.get("dt_action", 0.25))
                pose_prev = backend.get_pose()
                lin_cmd = float(np.clip(float(lin), -0.35, 0.35))
                ang_cmd = float(np.clip(float(ang), -1.6, 1.6))
                nyaw = _wrap_pi(float(pose_prev.yaw) + ang_cmd * dt)
                nx = float(pose_prev.x + math.sin(nyaw) * lin_cmd * dt)
                nz = float(pose_prev.z - math.cos(nyaw) * lin_cmd * dt)
                xyz = np.asarray([nx, float(pose_prev.y), nz], dtype=np.float32)
                collide = False
                try:
                    if hasattr(backend, "_in_collision"):
                        collide = bool(backend._in_collision(xyz))  # type: ignore[attr-defined]
                except Exception:
                    collide = False
                if collide:
                    xyz = np.asarray([float(pose_prev.x), float(pose_prev.y), float(pose_prev.z)], dtype=np.float32)
                backend._set_pose(xyz, nyaw)  # type: ignore[attr-defined]
                obs = backend._build_obs(  # type: ignore[attr-defined]
                    info={
                        "action_src": "cmd_vel",
                        "lin_x": float(lin_cmd),
                        "ang_z": float(ang_cmd),
                        "collision_pred": int(collide),
                    }
                )
                pose = backend.get_pose()
                used_cmd_vel = 1
                if abs(lin_cmd) >= float(args.cmd_lin_thresh):
                    action = 1 if lin_cmd > 0.0 else 0
                elif abs(ang_cmd) >= float(args.cmd_ang_thresh):
                    action = 2 if ang_cmd > 0 else 3
                else:
                    action = 0
            else:
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

        if not used_cmd_vel:
            obs, _, _ = backend.step(int(action))
            pose = backend.get_pose()
        pose_last = pose

        if node is not None:
            _publish_ros(node, float(pose.x), float(pose.z), float(pose.yaw), dt=float(cfg.get("dt_action", 0.5)))
            odom_x = float(node.last_odom.pose.pose.position.x) if getattr(node, "last_odom", None) is not None else float("nan")
            odom_y = float(node.last_odom.pose.pose.position.y) if getattr(node, "last_odom", None) is not None else float("nan")
            if getattr(node, "last_odom", None) is not None:
                qz = float(node.last_odom.pose.pose.orientation.z)
                qw = float(node.last_odom.pose.pose.orientation.w)
                odom_yaw = _yaw_from_quat(qz, qw)
            else:
                odom_yaw = float("nan")
        else:
            odom_x = float("nan")
            odom_y = float("nan")
            odom_yaw = float("nan")

        d = math.hypot(float(pose.x - prev_pose.x), float(pose.z - prev_pose.z))
        moved += d
        prev_pose = pose

        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        overlay = _draw_overlay(
            np.asarray(rgb, dtype=np.uint8),
            [
                f"step={i} [FP]",
                f"odom=({odom_x:.2f},{odom_y:.2f},{math.degrees(odom_yaw):.1f})",
                f"dist_moved={moved:.2f}m action={action}",
            ],
        )
        Image.fromarray(rgb).save(cap_dir / f"rgb_fp_{i:04d}.png")
        Image.fromarray(overlay).save(cap_dir / f"overlay_rgb_fp_{i:04d}.png")
        rgbs_fp.append(rgb)
        rgbs.append(rgb)  # backward compat for motion checker

        # v35c: Chase camera capture (every 3rd step to save time)
        if i % 3 == 0:
            # Save current mount, switch to chase
            saved_offset = list(v35c_fp_offset)
            saved_rpy = [0.0, v35c_cam_pitch, 0.0]
            backend.set_camera_mount(offset_m=v35c_chase_offset, rpy_deg=[0.0, v35c_chase_pitch, 0.0], emit_anchor=False)
            chase_obs = backend._build_obs()
            backend.set_camera_mount(offset_m=saved_offset, rpy_deg=saved_rpy, emit_anchor=False)
            chase_rgb = np.asarray(chase_obs.rgb, dtype=np.uint8)
            chase_overlay = _draw_overlay(
                np.asarray(chase_rgb, dtype=np.uint8),
                [
                    f"step={i} [CHASE]",
                    f"odom=({odom_x:.2f},{odom_y:.2f},{math.degrees(odom_yaw):.1f})",
                    f"dist_moved={moved:.2f}m",
                ],
            )
            Image.fromarray(chase_rgb).save(chase_dir / f"rgb_chase_{i:04d}.png")
            Image.fromarray(chase_overlay).save(chase_dir / f"overlay_rgb_chase_{i:04d}.png")
            rgbs_chase.append(chase_rgb)

        pose_rows.append(
            [
                int(i),
                float(pose.x),
                float(pose.z),
                float(pose.yaw),
                float(odom_x),
                float(odom_y),
                float(odom_yaw),
                "camera_optical_frame",
                float(moved),
            ]
        )
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

    # v35c: Also write pose_trace_v35c.csv with dist_moved
    pose_csv = cap_dir / "pose_trace.csv"
    pose_csv_v35c = cap_dir / "pose_trace_v35c.csv"
    for csv_path, include_dist in [(pose_csv, False), (pose_csv_v35c, True)]:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if include_dist:
                w.writerow(["step", "gt_x", "gt_y", "gt_yaw_rad", "odom_x", "odom_y", "odom_yaw_rad", "camera_frame", "dist_moved_m"])
            else:
                w.writerow(["step", "gt_x", "gt_y", "gt_yaw_rad", "odom_x", "odom_y", "odom_yaw_rad", "camera_frame"])
            for row in pose_rows:
                if include_dist:
                    w.writerow(row)
                else:
                    w.writerow(row[:8])
    print(
        f"[V35C_POSE_TRACE] ok=1 path={pose_csv_v35c} rows={len(pose_rows)}",
        flush=True,
    )

    gt_moved_m = 0.0
    odom_moved_m = 0.0
    gt_vs_odom_err = 0.0
    if pose_rows:
        gx0, gy0 = float(pose_rows[0][1]), float(pose_rows[0][2])
        gxn, gyn = float(pose_rows[-1][1]), float(pose_rows[-1][2])
        gt_moved_m = float(math.hypot(gxn - gx0, gyn - gy0))
        valid_odom = [r for r in pose_rows if (float(r[4]) == float(r[4]) and float(r[5]) == float(r[5]))]
        if valid_odom:
            ox0, oy0 = float(valid_odom[0][4]), float(valid_odom[0][5])
            oxn, oyn = float(valid_odom[-1][4]), float(valid_odom[-1][5])
            odom_moved_m = float(math.hypot(oxn - ox0, oyn - oy0))
            errs = [math.hypot(float(r[1]) - float(r[4]), float(r[2]) - float(r[5])) for r in valid_odom]
            if errs:
                gt_vs_odom_err = float(np.mean(np.asarray(errs, dtype=np.float32)))

    pose_ok = int(gt_moved_m >= 1.0 and odom_moved_m >= 1.0 and gt_vs_odom_err <= 0.3)
    print(
        f"[V34H_POSE_TRACE] frames={len(pose_rows)} gt_moved_m={gt_moved_m:.3f} "
        f"odom_moved_m={odom_moved_m:.3f} gt_vs_odom_err_m={gt_vs_odom_err:.3f} ok={pose_ok}",
        flush=True,
    )

    if node is not None:
        dur = max(1e-6, float(node.cmd_last_t - node.cmd_first_t)) if float(node.cmd_first_t) > 0 else 0.0
        hz = (float(node.cmd_count) / dur) if dur > 0.0 else 0.0
        nonzero_ratio = (float(node.cmd_nonzero) / float(max(1, node.cmd_count))) if node.cmd_count > 0 else 0.0
        cmd_ok = int(node.cmd_count > 0 and nonzero_ratio >= 0.05 and node.cmd_max_lin > 0.05)
        print(
            f"[V34H_CMDVEL_STATS] hz={hz:.3f} nonzero_ratio={nonzero_ratio:.3f} "
            f"max_lin={float(node.cmd_max_lin):.3f} max_ang={float(node.cmd_max_ang):.3f} ok={cmd_ok}",
            flush=True,
        )

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
                "sky_ratio": float(view_now.get("sky_ratio", 0.0)),
                "mean_luma": mean_luma,
                "camera_fidelity": fidelity,
                "camera_mount": dict(cam_mount),
                "moved_m": moved,
                "frame": dict(frame_cfg),
                "bridge_flags": flags,
                "capture_dir": str(cap_dir),
                "gif_written": int(gif_written),
                "depth_frames": int(depth_frames),
                "pose_trace_csv": str(pose_csv),
                "pose_trace": {
                    "gt_moved_m": float(gt_moved_m),
                    "odom_moved_m": float(odom_moved_m),
                    "gt_vs_odom_err_m": float(gt_vs_odom_err),
                    "ok": int(pose_ok),
                },
                "cmd_vel_stats": {
                    "count": int(getattr(node, "cmd_count", 0) if node is not None else 0),
                    "nonzero": int(getattr(node, "cmd_nonzero", 0) if node is not None else 0),
                    "max_lin": float(getattr(node, "cmd_max_lin", 0.0) if node is not None else 0.0),
                    "max_ang": float(getattr(node, "cmd_max_ang", 0.0) if node is not None else 0.0),
                },
                "start_pose": {
                    "x": float(pose0.x),
                    "y": float(pose0.y),
                    "z": float(pose0.z),
                    "yaw_rad": float(pose0.yaw),
                    "stage_x": float(stage_x0),
                    "stage_y": float(stage_y0),
                    "stage_z": float(stage_z0),
                },
                "final_pose": {"x": float(pose_last.x), "y": float(pose_last.y), "z": float(pose_last.z), "yaw_rad": float(pose_last.yaw)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if node is not None:
        try:
            import rclpy  # type: ignore

            if spin_stop is not None:
                spin_stop.set()
            if spin_thread is not None:
                spin_thread.join(timeout=1.0)
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
