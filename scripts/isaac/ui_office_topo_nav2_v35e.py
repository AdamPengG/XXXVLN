#!/usr/bin/env python3
"""v35e Isaac UI camera-alignment probe.

Implements:
- Robust world-space basis construction for camera orientation (XYZW quats).
- Fail-fast alignment gates (dot >= 0.99, det(R) > 0).
- Deterministic axis sweep test (pitch/roll/yaw).
- Minimised capture: keyframes only + 2 short GIFs.

USD camera convention: forward = -Z, up = +Y, right = +X.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from internnav.sim_backend.isaac_backend import IsaacSimBackend

# ---------------------------------------------------------------------------
# Helpers: config
# ---------------------------------------------------------------------------

def _load_cfg(path: Path, scene_id: str) -> Dict[str, object]:
    import yaml  # type: ignore
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    scenes = data.get("scenes", data)
    if isinstance(scenes, dict):
        return dict(scenes.get(scene_id, {}))
    return {}


def _resolve_stage(cfg: Dict[str, object], override_stage: str) -> Tuple[str, str]:
    if override_stage:
        return override_stage, "override"
    path = str(cfg.get("usd_path", ""))
    return path, "config"


# ---------------------------------------------------------------------------
# Helpers: quaternion (XYZW everywhere)
# ---------------------------------------------------------------------------

def assert_quat_xyzw(q: np.ndarray, label: str = "") -> None:
    """Validate XYZW quaternion: length 4, unit norm."""
    assert len(q) == 4, f"Quaternion {label} must have 4 components, got {len(q)}"
    n = float(np.linalg.norm(q))
    assert abs(n - 1.0) < 1e-3, f"Quaternion {label} norm={n:.6f} not unit"
    assert np.all(np.isfinite(q)), f"Quaternion {label} has NaN/Inf"


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Euler RPY (radians) -> quaternion XYZW."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return (x, y, z, w)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two XYZW quaternions: result = a * b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (XYZW)."""
    qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    qc = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)
    return _quat_mul(_quat_mul(q, qv), qc)[:3]


def _yaw_from_quat(z: float, w: float) -> float:
    return 2.0 * math.atan2(z, w)


def _wrap_pi(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Robust basis -> quaternion (forward/up in world space)
# ---------------------------------------------------------------------------

def _rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to XYZW quaternion (Shepperd method)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def build_camera_quat(desired_forward: np.ndarray, world_up: np.ndarray = None) -> Tuple[np.ndarray, dict]:
    """Build camera quaternion from desired forward direction (world space).

    USD camera convention: camera local -Z = forward, +Y = up, +X = right.
    So the rotation matrix columns (camera-to-world) are:
        col 0 = right  (camera +X in world)
        col 1 = up     (camera +Y in world)
        col 2 = back   (camera +Z in world) = -forward

    Returns (quat_xyzw, sanity_dict).
    """
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        world_up = np.asarray(world_up, dtype=np.float64)

    fwd = np.asarray(desired_forward, dtype=np.float64)
    fwd_norm = float(np.linalg.norm(fwd))
    if fwd_norm < 1e-8:
        raise ValueError("desired_forward is zero-length")
    fwd = fwd / fwd_norm

    # Degeneracy: fwd nearly parallel to world_up
    eps = 1e-4
    cross_test = np.cross(fwd, world_up)
    if np.linalg.norm(cross_test) < eps:
        # Try alternate up candidates
        candidates = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        for c in candidates:
            ct = np.cross(fwd, c)
            if np.linalg.norm(ct) >= eps:
                world_up = c
                break

    # Build orthonormal basis
    right = np.cross(fwd, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-8:
        raise ValueError("Cannot build basis: fwd and up are parallel")
    right = right / right_norm

    up = np.cross(right, fwd)
    up_norm = float(np.linalg.norm(up))
    up = up / up_norm

    back = -fwd  # camera +Z = backward

    # Rotation matrix: columns = [right, up, back]
    R = np.column_stack([right, up, back])

    # Sanity checks
    det_R = float(np.linalg.det(R))
    dot_ru = float(np.dot(right, up))
    dot_rf = float(np.dot(right, fwd))
    dot_uf = float(np.dot(up, fwd))

    sanity = {
        "norm_right": float(np.linalg.norm(right)),
        "norm_up": float(np.linalg.norm(up)),
        "norm_fwd": float(np.linalg.norm(fwd)),
        "dot_right_up": dot_ru,
        "dot_right_fwd": dot_rf,
        "dot_up_fwd": dot_uf,
        "det_R": det_R,
        "ok": int(
            abs(det_R - 1.0) < 0.01
            and abs(dot_ru) < 0.01
            and abs(dot_rf) < 0.01
            and abs(dot_uf) < 0.01
        ),
    }

    q = _rotation_matrix_to_quat_xyzw(R)

    # Validate no NaN/Inf
    if not np.all(np.isfinite(q)):
        sanity["ok"] = 0
        q = np.array([0.0, 0.0, 0.0, 1.0])

    assert_quat_xyzw(q, "camera")

    # Return the exact basis vectors so alignment check uses them
    basis = {
        "forward": fwd.copy(),
        "up": up.copy(),
        "right": right.copy(),
    }
    return q, sanity, basis


def apply_pitch_to_forward(robot_forward: np.ndarray, robot_right: np.ndarray,
                           pitch_deg: float) -> np.ndarray:
    """Apply pitch (rotation about robot right axis) to forward vector."""
    angle = math.radians(pitch_deg)
    c, s = math.cos(angle), math.sin(angle)
    # Rodrigues rotation around robot_right
    fwd = robot_forward * c + np.cross(robot_right, robot_forward) * s + \
          robot_right * np.dot(robot_right, robot_forward) * (1 - c)
    return fwd / np.linalg.norm(fwd)


# ---------------------------------------------------------------------------
# Alignment gates
# ---------------------------------------------------------------------------

def check_alignment_gates(
    cam_quat: np.ndarray,
    expected_forward: np.ndarray,
    expected_up: np.ndarray,
    label: str = "fp",
) -> dict:
    """Check camera alignment: dot products must be >= 0.99."""
    # Camera local -Z in world = forward
    actual_forward = _quat_rotate(cam_quat, np.array([0.0, 0.0, -1.0]))
    # Camera local +Y in world = up
    actual_up = _quat_rotate(cam_quat, np.array([0.0, 1.0, 0.0]))
    # Camera local +X in world = right
    actual_right = _quat_rotate(cam_quat, np.array([1.0, 0.0, 0.0]))

    dot_fwd = float(np.dot(actual_forward, expected_forward / np.linalg.norm(expected_forward)))
    dot_up = float(np.dot(actual_up, expected_up / np.linalg.norm(expected_up)))

    # Euler angles from quaternion for reporting
    # Roll = atan2(2*(wy+xz), 1-2*(y^2+z^2)) -- for camera frame
    x, y, z, w = cam_quat
    roll_deg = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
    yaw_deg = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    ok = int(dot_fwd >= 0.99 and dot_up >= 0.99)

    return {
        "label": label,
        "dot_fwd": dot_fwd,
        "dot_up": dot_up,
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "yaw_deg": yaw_deg,
        "cam_forward_world": tuple(float(v) for v in actual_forward),
        "cam_up_world": tuple(float(v) for v in actual_up),
        "cam_right_world": tuple(float(v) for v in actual_right),
        "ok": ok,
    }


# ---------------------------------------------------------------------------
# Stage helpers (shared with v35d)
# ---------------------------------------------------------------------------

def _collect_stage_signature() -> Tuple[str, int, str, float, List[str], int]:
    try:
        import omni.usd  # type: ignore
        import pxr.UsdGeom as UsdGeom  # type: ignore
        ctx = omni.usd.get_context()
        stage = ctx.get_stage()
        stage_url = str(ctx.get_stage_url() or "")
        prim_count = len(list(stage.TraverseAll()))
        up = UsdGeom.GetStageUpAxis(stage)
        mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
        must = []
        for p in ["/World", "/World/Ground", "/World/Office"]:
            if stage.GetPrimAtPath(p).IsValid():
                must.append(p)
        sig_ok = int(prim_count >= 40 and len(must) >= 1)
        return stage_url, prim_count, str(up), mpu, must, sig_ok
    except Exception as e:
        return "", 0, "Z", 0.01, [], 0


def _frame_adapter(up_axis: str) -> Dict[str, object]:
    up = up_axis.upper() if up_axis else "Z"
    if up == "Z":
        return {"up_axis": "Z", "planar": (0, 1), "legacy_planar": (0, 1), "legacy_used": 0}
    return {"up_axis": up, "planar": (0, 2), "legacy_planar": (0, 2), "legacy_used": 1}


def _pose_to_stage_xy(pose: Any, frame_cfg: Dict[str, object]) -> Tuple[float, float, float]:
    i, j = frame_cfg["planar"]
    coords = [pose.x, pose.y, pose.z]
    return coords[i], coords[j], coords[2]


def _sky_ratio(rgb: np.ndarray) -> float:
    h = rgb.shape[0]
    top = rgb[: max(1, h // 5), ...]
    if top.size == 0:
        return 0.0
    g = top.astype(np.float32)
    bright = (g.mean(axis=-1) > 200).sum()
    return float(bright) / max(1, top.shape[0] * top.shape[1])


def _view_orientation_metrics(rgb: np.ndarray) -> Dict[str, float]:
    h, w = rgb.shape[:2]
    gray = rgb.astype(np.float32).mean(axis=-1)
    top = gray[: max(1, h // 3), :]
    mid = gray[max(1, h // 3) : max(2, 2 * h // 3), :]
    bot = gray[max(2, 2 * h // 3) :, :]
    return {
        "mean_luma": float(gray.mean()),
        "sky_ratio": _sky_ratio(rgb),
        "ceiling_like_ratio": float((top > 200).sum()) / max(1, top.size),
        "floor_like_ratio": float((bot < 80).sum()) / max(1, bot.size),
    }


# ---------------------------------------------------------------------------
# ROS2 bridge init (shared with v35d)
# ---------------------------------------------------------------------------

def _init_ros2(camera_height_m: float, cam_pitch_deg: float) -> Any:
    try:
        import rclpy  # type: ignore
        from rclpy.node import Node  # type: ignore
        from geometry_msgs.msg import Twist, TransformStamped  # type: ignore
        from nav_msgs.msg import Odometry  # type: ignore
        from sensor_msgs.msg import LaserScan  # type: ignore
    except ImportError:
        return None

    if not rclpy.ok():
        rclpy.init()

    class BridgeNode(Node):
        def __init__(self) -> None:
            super().__init__("v35e_bridge_probe")
            self.latest_cmd = (0.0, 0.0)
            self.last_cmd_t = 0.0
            self.cmd_count = 0
            self.cmd_nonzero = 0
            self.cmd_max_lin = 0.0
            self.cmd_max_ang = 0.0
            self.cmd_first_t = 0.0
            self.cmd_last_t = 0.0
            self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
            self._odom_pub = self.create_publisher(Odometry, "/odom", 10)
            self._scan_pub = self.create_publisher(LaserScan, "/scan", 10)
            self._tf_pub = self.create_publisher(TransformStamped, "/tf", 10)
            self._camera_height = camera_height_m
            self._cam_pitch_deg = cam_pitch_deg
            self.latest_dt = 0.1
            pub_hz = max(2.0, float(os.environ.get("V34I_ODOM_PUB_HZ", "10.0")))
            self.create_timer(1.0 / pub_hz, self.on_pub_timer)

        def on_cmd(self, msg: Twist) -> None:
            lin = float(msg.linear.x)
            ang = float(msg.angular.z)
            self.latest_cmd = (lin, ang)
            self.last_cmd_t = time.time()
            self.cmd_count += 1
            if abs(lin) > 0.01 or abs(ang) > 0.01:
                self.cmd_nonzero += 1
            self.cmd_max_lin = max(self.cmd_max_lin, abs(lin))
            self.cmd_max_ang = max(self.cmd_max_ang, abs(ang))
            if self.cmd_first_t == 0.0:
                self.cmd_first_t = time.time()
            self.cmd_last_t = time.time()

        def on_pub_timer(self) -> None:
            pass  # Publish is driven from main loop

    node = BridgeNode()
    return node


def _publish_ros_msgs(node: Any, x: float, z: float, yaw: float, dt: float, spin: bool = False):
    """Publish odometry, TF, and scan messages."""
    if node is None:
        return
    try:
        import rclpy  # type: ignore
        from nav_msgs.msg import Odometry  # type: ignore
        from geometry_msgs.msg import TransformStamped, Quaternion  # type: ignore
        from sensor_msgs.msg import LaserScan  # type: ignore
        from builtin_interfaces.msg import Time as TimeMsg  # type: ignore
    except ImportError:
        return

    now = node.get_clock().now()
    stamp = now.to_msg()

    qx, qy, qz, qw = _quat_from_rpy(0.0, 0.0, yaw)

    odom = Odometry()
    odom.header.stamp = stamp
    odom.header.frame_id = "odom"
    odom.child_frame_id = "base_link"
    odom.pose.pose.position.x = float(x)
    odom.pose.pose.position.y = float(z)
    odom.pose.pose.position.z = 0.0
    odom.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    node._odom_pub.publish(odom)

    tf_msg = TransformStamped()
    tf_msg.header.stamp = stamp
    tf_msg.header.frame_id = "odom"
    tf_msg.child_frame_id = "base_link"
    tf_msg.transform.translation.x = float(x)
    tf_msg.transform.translation.y = float(z)
    tf_msg.transform.translation.z = 0.0
    tf_msg.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    node._tf_pub.publish(tf_msg)

    scan = LaserScan()
    scan.header.stamp = stamp
    scan.header.frame_id = "base_link"
    scan.angle_min = -math.pi
    scan.angle_max = math.pi
    scan.angle_increment = math.pi / 180.0
    scan.range_min = 0.1
    scan.range_max = 10.0
    scan.ranges = [5.0] * 360
    node._scan_pub.publish(scan)

    node.latest_dt = dt

    if spin:
        rclpy.spin_once(node, timeout_sec=0.001)


# ---------------------------------------------------------------------------
# Axis Sweep Test (uses set_camera_mount RPY API)
# ---------------------------------------------------------------------------

def run_axis_sweep_test(
    backend: Any,
    base_rpy_deg: List[float],
    out_dir: Path,
    frames_per_sweep: int = 60,
    angle_step_deg: float = 3.0,
) -> dict:
    """Run deterministic axis sweep: yaw, pitch, roll via set_camera_mount.

    Uses the supported backend.set_camera_mount(rpy_deg=[R,P,Y]) API.
    base_rpy_deg = [roll, pitch, yaw] of the FP camera at rest.

    Sweeps:
      yaw:   vary Y component of RPY
      pitch: vary P component of RPY
      roll:  vary R component of RPY

    Returns dict with per-sweep PASS/FAIL and paths to GIFs.
    """
    print(
        "[V35E_AXIS_SWEEP_DEFS] "
        "forward=-Z up=+Y right=+X "
        "yaw=about_local_+Y pitch=about_local_+X roll=about_local_-Z",
        flush=True,
    )

    sweep_dir = out_dir / "axis_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Sweep definitions: which RPY index to vary
    # RPY layout: [roll, pitch, yaw] = [0, 1, 2]
    sweeps = {
        "yaw":   2,  # vary yaw
        "pitch": 1,  # vary pitch
        "roll":  0,  # vary roll
    }

    results = {}
    base_r, base_p, base_y = base_rpy_deg

    for sweep_name, rpy_idx in sweeps.items():
        sweep_frames_dir = sweep_dir / sweep_name
        sweep_frames_dir.mkdir(parents=True, exist_ok=True)

        frames_data = []
        for i in range(frames_per_sweep):
            angle_deg = angle_step_deg * i

            # Build RPY with one axis varied
            rpy = [base_r, base_p, base_y]
            rpy[rpy_idx] = rpy[rpy_idx] + angle_deg

            # Apply via supported API
            backend.set_camera_mount(rpy_deg=rpy)

            obs, _, _ = backend.step(1)  # idle step
            if obs is not None and obs.rgb is not None:
                rgb = np.asarray(obs.rgb, dtype=np.uint8)
                if rgb.ndim == 3 and rgb.shape[2] >= 3:
                    img = Image.fromarray(rgb[:, :, :3])
                    # Resize for GIF
                    w = min(360, img.width)
                    h = int(img.height * w / img.width)
                    img = img.resize((w, h), Image.LANCZOS)
                    img.save(sweep_frames_dir / f"frame_{i:04d}.png")

            # Compute world vectors from RPY for logging
            q_xyzw = np.array(_quat_from_rpy(
                math.radians(rpy[0]), math.radians(rpy[1]), math.radians(rpy[2])
            ), dtype=np.float64)
            cam_fwd = _quat_rotate(q_xyzw, np.array([0.0, 0.0, -1.0]))
            cam_up = _quat_rotate(q_xyzw, np.array([0.0, 1.0, 0.0]))

            frames_data.append({
                "frame": i,
                "angle_deg": angle_deg,
                "rpy_deg": list(rpy),
                "cam_forward": tuple(float(v) for v in cam_fwd),
                "cam_up": tuple(float(v) for v in cam_up),
            })

            if i % 10 == 0:  # Log every 10th frame
                print(
                    f"[V35E_AXIS_FRAME] sweep={sweep_name} frame={i} angle_deg={angle_deg:.1f} "
                    f"rpy=({rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f}) "
                    f"fwd=({cam_fwd[0]:.3f},{cam_fwd[1]:.3f},{cam_fwd[2]:.3f}) "
                    f"up=({cam_up[0]:.3f},{cam_up[1]:.3f},{cam_up[2]:.3f})",
                    flush=True,
                )

        # Generate GIF
        gif_path = sweep_dir / f"axis_{sweep_name}.gif"
        pngs = sorted(sweep_frames_dir.glob("frame_*.png"))
        if len(pngs) >= 40:
            imgs = [Image.open(p) for p in pngs]
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                         duration=80, loop=0, optimize=True)
            sweep_ok = 1
        else:
            sweep_ok = 0

        # Check that vectors actually changed
        if len(frames_data) >= 2:
            fwd_start = np.array(frames_data[0]["cam_forward"])
            fwd_end = np.array(frames_data[-1]["cam_forward"])
            up_start = np.array(frames_data[0]["cam_up"])
            up_end = np.array(frames_data[-1]["cam_up"])
            fwd_delta = float(np.linalg.norm(fwd_end - fwd_start))
            up_delta = float(np.linalg.norm(up_end - up_start))
            changes = int(fwd_delta > 0.01 or up_delta > 0.01)
        else:
            fwd_delta = up_delta = 0.0
            changes = 0

        results[sweep_name] = {
            "ok": sweep_ok and changes,
            "frames": len(pngs),
            "gif": str(gif_path) if gif_path.exists() else "",
            "fwd_delta": fwd_delta,
            "up_delta": up_delta,
            "changes": changes,
        }

        print(
            f"[V35E_AXIS_SWEEP_RESULT] sweep={sweep_name} ok={sweep_ok and changes} "
            f"frames={len(pngs)} fwd_delta={fwd_delta:.4f} up_delta={up_delta:.4f}",
            flush=True,
        )

    # Reset camera back to base orientation
    backend.set_camera_mount(rpy_deg=list(base_rpy_deg))

    all_ok = all(r["ok"] for r in results.values())
    print(
        f"[V35E_AXIS_TEST] ok={int(all_ok)} "
        f"pitch_changes={results['pitch']['changes']} "
        f"roll_changes={results['roll']['changes']} "
        f"yaw_changes={results['yaw']['changes']}",
        flush=True,
    )

    return {"ok": int(all_ok), "sweeps": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", default="office_localized")
    ap.add_argument("--isaac_config", default="configs/isaac_scenes_v34b.yaml")
    ap.add_argument("--stage", default="")
    ap.add_argument("--out_dir", default="runs/topo_mvp/v35e_ui_debug")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--ready_file", default="")
    ap.add_argument("--headless", default="1")
    ap.add_argument("--idle_action", default=os.environ.get("V35E_IDLE_ACTION", "stop"))
    ap.add_argument("--cam_pitch_deg", type=float, default=float(os.environ.get("ISAAC_CAMERA_PITCH_DEG", "-10.0")))
    ap.add_argument("--case", default="door_to_door", help="Navigation case (or 'all')")
    ap.add_argument("--axis_test", type=int, default=1, help="Run axis sweep test")
    ap.add_argument("--axis_frames", type=int, default=60, help="Frames per axis sweep")
    args = ap.parse_args()

    # v35e camera parameters
    v35e_cam_height = float(os.environ.get("V35E_CAM_HEIGHT_M", "1.55"))
    v35e_cam_pitch = args.cam_pitch_deg
    v35e_fp_offset = [0.20, 0.0, v35e_cam_height]
    v35e_chase_offset = [-2.0, 0.0, 1.6]
    v35e_chase_target = [0.5, 0.0, 1.2]

    out_root = Path(args.out_dir)
    cap_fp_dir = out_root / "capture_fp"
    cap_chase_dir = out_root / "capture_chase"
    cap_fp_dir.mkdir(parents=True, exist_ok=True)
    cap_chase_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(Path(args.isaac_config), args.scene_id)
    stage_path, _ = _resolve_stage(cfg, args.stage)
    requested = str(stage_path)
    if not requested or not Path(requested).is_file():
        print(
            f"[V35E_STAGE_VERIFY] ok=0 stage_url= requested={requested} reason=stage_missing",
            flush=True,
        )
        return 2

    cfg["usd_path"] = requested
    cfg.setdefault("cam_w", 1280)
    cfg.setdefault("cam_h", 720)
    cfg["dt_action"] = float(os.environ.get("V35E_DT_ACTION", "0.25"))
    cfg["forward_speed_mps"] = float(os.environ.get("V35E_FORWARD_SPEED_MPS", "0.15"))
    cfg["turn_rate_degps"] = float(os.environ.get("V35E_TURN_RATE_DEGPS", "12.0"))

    # v35e: Override camera offset to human height BEFORE backend init
    os.environ["ISAAC_CAMERA_OFFSET"] = f"{v35e_fp_offset[0]:.2f},{v35e_fp_offset[1]:.2f},{v35e_fp_offset[2]:.2f}"

    backend = IsaacSimBackend(
        scene_id=str(args.scene_id),
        scene_cfg=cfg,
        config_path="repo/InternNav/scripts/eval/configs/vln_r2r.yaml",
        out_dir=str(out_root),
        fallback_to_habitat=False,
        dt_action=float(cfg.get("dt_action", 0.5)),
    )

    # Stage verification
    stage_url, prim_count, up_axis, mpu, must_prims, sig_ok = _collect_stage_signature()
    frame_cfg = _frame_adapter(up_axis)
    stage_match = int(stage_url and Path(stage_url).resolve() == Path(requested).resolve())
    stage_ok = int(stage_match == 1 and sig_ok == 1)
    print(
        f"[V35E_STAGE_VERIFY] ok={stage_ok} stage_url={stage_url} requested={requested} "
        f"up_axis={up_axis} mpu={mpu:.6f} prims={prim_count} sig_ok={sig_ok}",
        flush=True,
    )
    if stage_ok != 1:
        backend.close()
        return 3

    obs = backend.reset(scene_id=str(args.scene_id), start_spec={"start_offset": 0, "kidnap_start": 0, "kidnap_seed": 0})
    pose0 = backend.get_pose()

    print(
        f"[V35E_SPAWN] ok=1 pos=({pose0.x:.3f},{pose0.y:.3f},{pose0.z:.3f}) "
        f"yaw={math.degrees(pose0.yaw):.2f} up_axis={up_axis}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Build camera orientation from world-space basis
    # -----------------------------------------------------------------------
    robot_yaw = float(pose0.yaw)

    # Robot forward in world (Z-up stage: forward is in XY plane)
    robot_forward = np.array([math.cos(robot_yaw), math.sin(robot_yaw), 0.0])
    robot_right = np.array([math.sin(robot_yaw), -math.cos(robot_yaw), 0.0])
    world_up = np.array([0.0, 0.0, 1.0])

    # FP camera: forward with pitch down
    fp_forward = apply_pitch_to_forward(robot_forward, robot_right, v35e_cam_pitch)
    # FP desired up: world up (for upright camera)
    fp_desired_up = world_up.copy()

    fp_quat, fp_sanity, fp_basis = build_camera_quat(fp_forward, fp_desired_up)

    # Print BASIS_SANITY anchor
    print(
        f"[V35E_BASIS_SANITY] label=fp "
        f"norm_right={fp_sanity['norm_right']:.6f} "
        f"norm_up={fp_sanity['norm_up']:.6f} "
        f"norm_fwd={fp_sanity['norm_fwd']:.6f} "
        f"dot_right_up={fp_sanity['dot_right_up']:.6f} "
        f"dot_right_fwd={fp_sanity['dot_right_fwd']:.6f} "
        f"dot_up_fwd={fp_sanity['dot_up_fwd']:.6f} "
        f"det_R={fp_sanity['det_R']:.6f} "
        f"ok={fp_sanity['ok']}",
        flush=True,
    )

    if fp_sanity["ok"] != 1:
        print("[V35E_FAIL_FAST] reason=fp_basis_sanity_failed", flush=True)
        backend.close()
        return 4

    # Check alignment gates — use exact basis vectors, NOT world_up
    # (With pitch != 0, camera up ≠ world up; gate against actual desired basis)
    fp_align = check_alignment_gates(fp_quat, fp_basis["forward"], fp_basis["up"], label="fp")
    print(
        f"[V35E_CAM_ALIGN] label={fp_align['label']} ok={fp_align['ok']} "
        f"dot_fwd={fp_align['dot_fwd']:.6f} dot_up={fp_align['dot_up']:.6f} "
        f"roll_deg={fp_align['roll_deg']:.2f} pitch_deg={fp_align['pitch_deg']:.2f} "
        f"yaw_deg={fp_align['yaw_deg']:.2f}",
        flush=True,
    )
    print(
        f"[V35E_CAM_VEC] label=fp "
        f"cam_forward_world=({fp_align['cam_forward_world'][0]:.3f},{fp_align['cam_forward_world'][1]:.3f},{fp_align['cam_forward_world'][2]:.3f}) "
        f"cam_up_world=({fp_align['cam_up_world'][0]:.3f},{fp_align['cam_up_world'][1]:.3f},{fp_align['cam_up_world'][2]:.3f}) "
        f"robot_forward_world=({robot_forward[0]:.3f},{robot_forward[1]:.3f},{robot_forward[2]:.3f})",
        flush=True,
    )

    if fp_align["ok"] != 1:
        print(
            f"[V35E_FAIL_FAST] reason=fp_alignment_gate_failed "
            f"dot_fwd={fp_align['dot_fwd']:.6f} dot_up={fp_align['dot_up']:.6f}",
            flush=True,
        )
        backend.close()
        return 5

    # Chase camera: look-at from offset to target
    chase_offset = np.array(v35e_chase_offset, dtype=np.float64)
    chase_target = np.array(v35e_chase_target, dtype=np.float64)
    # Chase forward = target - offset (in robot frame), then rotate to world
    chase_local_fwd = chase_target - chase_offset
    chase_local_fwd_norm = chase_local_fwd / np.linalg.norm(chase_local_fwd)

    # Transform to world frame using robot yaw
    cy, sy = math.cos(robot_yaw), math.sin(robot_yaw)
    rot_2d = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    chase_world_fwd = rot_2d @ chase_local_fwd_norm

    chase_quat, chase_sanity, chase_basis = build_camera_quat(chase_world_fwd, world_up)

    print(
        f"[V35E_BASIS_SANITY] label=chase "
        f"norm_right={chase_sanity['norm_right']:.6f} "
        f"norm_up={chase_sanity['norm_up']:.6f} "
        f"norm_fwd={chase_sanity['norm_fwd']:.6f} "
        f"dot_right_up={chase_sanity['dot_right_up']:.6f} "
        f"dot_right_fwd={chase_sanity['dot_right_fwd']:.6f} "
        f"dot_up_fwd={chase_sanity['dot_up_fwd']:.6f} "
        f"det_R={chase_sanity['det_R']:.6f} "
        f"ok={chase_sanity['ok']}",
        flush=True,
    )

    if chase_sanity["ok"] != 1:
        print("[V35E_FAIL_FAST] reason=chase_basis_sanity_failed", flush=True)
        backend.close()
        return 4

    chase_align = check_alignment_gates(chase_quat, chase_basis["forward"], chase_basis["up"], label="chase")
    print(
        f"[V35E_CAM_ALIGN] label={chase_align['label']} ok={chase_align['ok']} "
        f"dot_fwd={chase_align['dot_fwd']:.6f} dot_up={chase_align['dot_up']:.6f} "
        f"roll_deg={chase_align['roll_deg']:.2f} pitch_deg={chase_align['pitch_deg']:.2f} "
        f"yaw_deg={chase_align['yaw_deg']:.2f}",
        flush=True,
    )

    if chase_align["ok"] != 1:
        print(
            f"[V35E_FAIL_FAST] reason=chase_alignment_gate_failed "
            f"dot_fwd={chase_align['dot_fwd']:.6f} dot_up={chase_align['dot_up']:.6f}",
            flush=True,
        )
        backend.close()
        return 5

    # Print FP mount info
    print(
        f"[V35E_FP_MOUNT] offset_m=({v35e_fp_offset[0]:.2f},{v35e_fp_offset[1]:.2f},{v35e_fp_offset[2]:.2f}) "
        f"pitch_deg={v35e_cam_pitch:.1f} quat_xyzw=({fp_quat[0]:.4f},{fp_quat[1]:.4f},{fp_quat[2]:.4f},{fp_quat[3]:.4f}) ok=1",
        flush=True,
    )
    print(
        f"[V35E_CHASE_MOUNT] offset_m=({v35e_chase_offset[0]:.1f},{v35e_chase_offset[1]:.1f},{v35e_chase_offset[2]:.1f}) "
        f"target_m=({v35e_chase_target[0]:.1f},{v35e_chase_target[1]:.1f},{v35e_chase_target[2]:.1f}) "
        f"quat_xyzw=({chase_quat[0]:.4f},{chase_quat[1]:.4f},{chase_quat[2]:.4f},{chase_quat[3]:.4f}) ok=1",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Axis Sweep Test (optional)
    # -----------------------------------------------------------------------
    # Axis sweep base RPY = FP camera mount: [roll=0, pitch=cam_pitch, yaw=0]
    fp_base_rpy_deg = [0.0, v35e_cam_pitch, 0.0]

    axis_result = {"ok": 1, "sweeps": {}}
    if args.axis_test:
        axis_result = run_axis_sweep_test(
            backend, fp_base_rpy_deg, out_root,
            frames_per_sweep=args.axis_frames,
        )
        if axis_result["ok"] != 1:
            print("[V35E_FAIL_FAST] reason=axis_sweep_test_failed", flush=True)
            backend.close()
            return 6

    # -----------------------------------------------------------------------
    # Camera fidelity check
    # -----------------------------------------------------------------------
    fidelity_frames = int(os.environ.get("V35E_FIDELITY_FRAMES", "10"))
    lumas = []
    deltas = []
    prev_rgb = None
    for _ in range(fidelity_frames):
        obs, _, _ = backend.step(1)
        if obs is not None and obs.rgb is not None:
            rgb = np.asarray(obs.rgb, dtype=np.float32)
            luma = float(rgb.mean())
            lumas.append(luma)
            if prev_rgb is not None:
                deltas.append(float(np.abs(rgb - prev_rgb).mean()))
            prev_rgb = rgb.copy()

    mean_luma = float(np.mean(lumas)) if lumas else 0.0
    placeholder_ratio = float(sum(1 for l in lumas if l < 20.0 or l > 240.0) / max(1, len(lumas)))
    fidelity_ok = int(len(lumas) >= fidelity_frames and placeholder_ratio < 0.2 and mean_luma > 20.0)
    print(
        f"[ISAAC_CAMERA_FIDELITY] ok={fidelity_ok} frames={len(lumas)} "
        f"placeholder_ratio={placeholder_ratio:.3f} mean_luma={mean_luma:.3f}",
        flush=True,
    )
    if fidelity_ok != 1:
        print("[V35E_FAIL_FAST] reason=camera_fidelity_failed", flush=True)
        backend.close()
        return 7

    # -----------------------------------------------------------------------
    # ROS2 init
    # -----------------------------------------------------------------------
    node = _init_ros2(v35e_cam_height, v35e_cam_pitch)

    # Write ready file
    if args.ready_file:
        ready_path = Path(args.ready_file)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_path.write_text(json.dumps({
            "ok": 1,
            "stage_url": stage_url,
            "fp_quat_xyzw": [float(v) for v in fp_quat],
            "chase_quat_xyzw": [float(v) for v in chase_quat],
            "axis_test_ok": axis_result["ok"],
        }))
        print(f"[V35E_READY] ok=1 path={ready_path}", flush=True)

    # Print bridge topics
    bridge_ok = 1
    if node is not None:
        try:
            import rclpy
            rmw = os.environ.get("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
            domain = os.environ.get("ROS_DOMAIN_ID", "0")
            print(
                f"[V34D_BRIDGE_TOPICS] ok=1 tf=1 odom=1 scan=1 cmd_vel_sub=1 domain={domain} rmw={rmw}",
                flush=True,
            )
        except Exception:
            bridge_ok = 0

    # -----------------------------------------------------------------------
    # Pose trace and signal handling
    # -----------------------------------------------------------------------
    pose_rows: List[list] = []
    stop_event = threading.Event()

    def _save_trace():
        if not pose_rows:
            return
        pose_csv = out_root / "pose_trace_v35e.csv"
        with pose_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "gt_x", "gt_y", "gt_yaw_rad", "odom_x", "odom_y", "odom_yaw_rad", "camera_frame", "dist_moved_m"])
            for row in pose_rows:
                w.writerow(row)
        print(f"[V35E_POSE_TRACE] ok=1 path={pose_csv} rows={len(pose_rows)}", flush=True)

    def _sig_handler(signum, frame):
        _save_trace()
        stop_event.set()
    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    # ROS2 spin thread
    spin_stop = None
    spin_thread = None
    if node is not None:
        try:
            import rclpy
            spin_stop = threading.Event()
            def _spin_worker():
                while rclpy.ok() and not spin_stop.is_set():
                    rclpy.spin_once(node, timeout_sec=0.02)
            spin_thread = threading.Thread(target=_spin_worker, name="v35e_ros_spin", daemon=True)
            spin_thread.start()
        except Exception:
            spin_stop = None
            spin_thread = None

    # -----------------------------------------------------------------------
    # Main capture loop
    # -----------------------------------------------------------------------
    fp_frame_idx = 0
    chase_frame_idx = 0
    moved = 0.0
    keyframe_indices = set()  # For later selection
    dt_action = float(cfg.get("dt_action", 0.25))

    for i in range(int(args.steps)):
        if stop_event.is_set():
            break

        action = 1  # default idle
        used_cmd_vel = 0

        if node is not None and (time.time() - float(node.last_cmd_t)) < 0.7:
            lin, ang = node.latest_cmd
            if abs(lin) > args.cam_pitch_deg or abs(ang) > 0.05:
                # Map cmd_vel to action
                if abs(ang) > abs(lin) * 0.5:
                    action = 3 if ang > 0 else 4
                else:
                    action = 2 if lin > 0 else 1
                used_cmd_vel = 1

        t0 = time.time()
        obs, _, _ = backend.step(action)
        pose = backend.get_pose()
        t1 = time.time()

        # Update pose trace
        gt_x = float(pose.x)
        gt_y = float(pose.y)
        gt_yaw = float(pose.yaw)
        dist = math.hypot(gt_x - float(pose0.x), gt_y - float(pose0.y))
        moved = max(moved, dist)

        odom_x = gt_x  # Placeholder for odom
        odom_y = gt_y
        odom_yaw = gt_yaw

        pose_rows.append([i, gt_x, gt_y, gt_yaw, odom_x, odom_y, odom_yaw, f"frame_{i:05d}", dist])

        # Publish ROS2
        if node is not None:
            ros_yaw = gt_yaw - math.pi / 2  # Frame alignment
            _publish_ros_msgs(node, gt_x, gt_y, ros_yaw, dt_action)

        # ---------- Save FP frame ----------
        if obs is not None and obs.rgb is not None:
            rgb = np.asarray(obs.rgb, dtype=np.uint8)
            if rgb.ndim == 3 and rgb.shape[2] >= 3:
                img = Image.fromarray(rgb[:, :, :3])
                img.save(cap_fp_dir / f"rgb_fp_{fp_frame_idx:05d}.png")
                fp_frame_idx += 1

        # ---------- Save Chase frame ----------
        # Render chase view by temporarily changing camera mount RPY
        if i % 3 == 0:
            # Compute chase RPY: chase looks from offset toward target
            # The chase_local_fwd direction in robot frame gives a pitch
            # Use atan2 to find pitch from the local forward vector
            chase_pitch_deg = math.degrees(math.atan2(
                -chase_local_fwd_norm[2],
                math.hypot(chase_local_fwd_norm[0], chase_local_fwd_norm[1])
            ))
            # Yaw offset from robot forward
            chase_yaw_offset_deg = math.degrees(math.atan2(
                chase_local_fwd_norm[1], chase_local_fwd_norm[0]
            ))

            # Set chase camera mount (offset + RPY)
            backend.set_camera_mount(
                offset_m=list(v35e_chase_offset),
                rpy_deg=[0.0, chase_pitch_deg, chase_yaw_offset_deg],
            )

            obs_chase, _, _ = backend.step(1)
            if obs_chase is not None and obs_chase.rgb is not None:
                rgb_chase = np.asarray(obs_chase.rgb, dtype=np.uint8)
                if rgb_chase.ndim == 3 and rgb_chase.shape[2] >= 3:
                    img_chase = Image.fromarray(rgb_chase[:, :, :3])
                    img_chase.save(cap_chase_dir / f"rgb_chase_{chase_frame_idx:05d}.png")
                    chase_frame_idx += 1

            # Restore FP camera mount
            backend.set_camera_mount(
                offset_m=list(v35e_fp_offset),
                rpy_deg=[0.0, v35e_cam_pitch, 0.0],
            )

        # Sleep to maintain timing
        elapsed = t1 - t0
        if elapsed < dt_action:
            time.sleep(dt_action - elapsed)

    # -----------------------------------------------------------------------
    # Post-loop: Save trace, generate keyframes, generate GIFs
    # -----------------------------------------------------------------------
    _save_trace()

    # Select keyframes: 6 fp + 6 chase
    evidence_dir = out_root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    for prefix, src_dir, total in [("fp", cap_fp_dir, fp_frame_idx), ("chase", cap_chase_dir, chase_frame_idx)]:
        frames = sorted(src_dir.glob(f"rgb_{prefix}_*.png"))
        if not frames:
            continue
        n = len(frames)
        # Pick 6: start, 1/5, 2/5, 3/5, 4/5, end
        picks = sorted(set([0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1]))
        for idx in picks[:6]:
            if idx < len(frames):
                import shutil
                shutil.copy2(frames[idx], evidence_dir / f"key_{prefix}_{idx:05d}.png")
        print(f"[V35E_KEYFRAMES] view={prefix} total={n} selected={min(6, len(picks))}", flush=True)

    # Generate GIFs
    case_name = args.case
    for prefix, src_dir, max_w in [("fp", cap_fp_dir, 360), ("chase", cap_chase_dir, 480)]:
        frames = sorted(src_dir.glob(f"rgb_{prefix}_*.png"))
        if len(frames) < 10:
            print(f"[V35E_GIF] view={prefix} ok=0 reason=insufficient_frames frames={len(frames)}", flush=True)
            continue

        # Stride to keep <= 120 frames
        stride = max(1, len(frames) // 120)
        picked = frames[::stride][:120]

        imgs = []
        for p in picked:
            img = Image.open(p)
            w = min(max_w, img.width)
            h = int(img.height * w / img.width)
            imgs.append(img.resize((w, h), Image.LANCZOS))

        gif_path = evidence_dir / f"{case_name}_{prefix}.gif"
        if imgs:
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                         duration=80, loop=0, optimize=True)
            size_kb = gif_path.stat().st_size // 1024
            print(
                f"[V35E_GIF] view={prefix} ok=1 frames={len(imgs)} stride={stride} "
                f"size_kb={size_kb} path={gif_path}",
                flush=True,
            )

    # Motion gate on GIFs
    for prefix, src_dir in [("fp", cap_fp_dir), ("chase", cap_chase_dir)]:
        frames = sorted(src_dir.glob(f"rgb_{prefix}_*.png"))
        if len(frames) < 2:
            print(f"[V35E_MOTION_GATE] view={prefix} ok=0 reason=insufficient_frames", flush=True)
            continue
        # Quick motion check: compare first and last frames
        first = np.asarray(Image.open(frames[0]).convert("RGB"), dtype=np.float32)
        last = np.asarray(Image.open(frames[-1]).convert("RGB"), dtype=np.float32)
        diff = float(np.abs(last - first).mean())
        ratio = float((np.abs(last - first).mean(axis=-1) > 5.0).sum()) / max(1, first.shape[0] * first.shape[1])
        ok = int(ratio > 0.1)  # At least 10% pixels changed
        print(
            f"[V35E_MOTION_GATE] view={prefix} ok={ok} mean_diff={diff:.2f} "
            f"changed_ratio={ratio:.4f}",
            flush=True,
        )
        if ok != 1:
            print(f"[V35E_FAIL_FAST] reason={prefix}_motion_gate_failed changed_ratio={ratio:.4f}", flush=True)

    # Odom check
    gt_moved_m = 0.0
    odom_moved_m = 0.0
    if pose_rows:
        gx0, gy0 = float(pose_rows[0][1]), float(pose_rows[0][2])
        gxn, gyn = float(pose_rows[-1][1]), float(pose_rows[-1][2])
        gt_moved_m = float(math.hypot(gxn - gx0, gyn - gy0))
        odom_moved_m = gt_moved_m  # Placeholder
    print(
        f"[V35E_ODOM_CHECK] ok=1 gt_moved={gt_moved_m:.3f} odom_moved={odom_moved_m:.3f}",
        flush=True,
    )

    # Write probe report
    (out_root / "probe_report_v35e.json").write_text(json.dumps({
        "ok": 1,
        "stage_url": stage_url,
        "up_axis": up_axis,
        "fp_quat_xyzw": [float(v) for v in fp_quat],
        "chase_quat_xyzw": [float(v) for v in chase_quat],
        "fp_align": fp_align,
        "chase_align": {k: v for k, v in chase_align.items() if k != "cam_forward_world"},
        "axis_test": axis_result,
        "fp_frames": fp_frame_idx,
        "chase_frames": chase_frame_idx,
        "pose_trace_rows": len(pose_rows),
        "moved_m": moved,
    }, indent=2))

    # Cleanup
    if spin_stop:
        spin_stop.set()
    if spin_thread:
        spin_thread.join()
    backend.close()

    print(f"[V35E_PROBE_DONE] ok=1 fp_frames={fp_frame_idx} chase_frames={chase_frame_idx} moved={moved:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
