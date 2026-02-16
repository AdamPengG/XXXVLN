#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


def _load_cfg(path: Path, scene_id: str) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(text)
    except Exception:
        cfg = json.loads(text)
    for row in cfg.get("scenes", []):
        if isinstance(row, dict) and str(row.get("scene_id", "")) == str(scene_id):
            return dict(row)
    return {}


def _resolve_stage(cfg: Dict[str, object], override_stage: str) -> Tuple[str, str]:
    override = str(override_stage or os.environ.get("ISAAC_STAGE_USD", "")).strip()
    if override:
        return override, "override"
    roots = []
    env_root = str(os.environ.get("ISAAC_ASSETS_ROOT", "")).strip()
    if env_root:
        roots.append(env_root)
    roots.extend(["/home/peng/IsaacAssets", "/home/peng/isaacsim_assets"])
    for root in roots:
        for p in (
            Path(root) / "Assets/Isaac/5.1/Isaac/Environments/Office/office.usd",
            Path(root) / "Isaac/Environments/Office/office.usd",
            Path(root) / "Office/office.usd",
        ):
            if p.is_file():
                return str(p), "official_assets"
    cfg_path = str(cfg.get("usd_path", "")).strip()
    if cfg_path:
        return cfg_path, "config"
    return "", "missing"


def _quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
    qw = float(math.cos(yaw * 0.5))
    qy = float(math.sin(yaw * 0.5))
    return np.array([qw, 0.0, qy, 0.0], dtype=np.float32)


def _raycast_ground_y(x: float, z: float, default_y: float = 0.0) -> float:
    try:
        import omni.physx as omni_physx  # type: ignore

        q = omni_physx.get_physx_scene_query_interface()
        hit = q.raycast_closest((float(x), 8.0, float(z)), (0.0, -1.0, 0.0), 20.0)
        if isinstance(hit, dict) and bool(hit.get("hit", False)):
            pos = hit.get("position", None)
            if pos is not None and len(pos) >= 2:
                return float(pos[1])
    except Exception:
        pass
    return float(default_y)


def _read_rgb(camera) -> np.ndarray | None:
    rgb = None
    try:
        rgba = camera.get_rgba()
        if rgba is not None:
            arr = np.asarray(rgba)
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                rgb = arr[..., :3].astype(np.uint8)
    except Exception:
        rgb = None
    if rgb is None:
        try:
            frame = camera.get_current_frame()
            if isinstance(frame, dict):
                rgba = frame.get("rgba", None)
                if rgba is not None:
                    arr = np.asarray(rgba)
                    if arr.ndim == 3 and arr.shape[-1] >= 3:
                        rgb = arr[..., :3].astype(np.uint8)
        except Exception:
            rgb = None
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", default="office_localized")
    ap.add_argument("--isaac_config", default="configs/isaac_scenes_v34b.yaml")
    ap.add_argument("--stage", default="")
    ap.add_argument("--out_dir", default="runs/topo_mvp/v34b_nav2_demo/physics")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--dt", type=float, default=1.0 / 30.0)
    ap.add_argument("--cmd_steps", type=int, default=45)
    ap.add_argument("--speed_mps", type=float, default=0.60)
    ap.add_argument("--base_clearance", type=float, default=0.03)
    ap.add_argument("--settle_steps", type=int, default=45)
    ap.add_argument("--cam_w", type=int, default=640)
    ap.add_argument("--cam_h", type=int, default=360)
    args = ap.parse_args()

    cfg = _load_cfg(Path(args.isaac_config), args.scene_id)
    stage_path, stage_source = _resolve_stage(cfg, args.stage)
    stage_exists = int(bool(stage_path) and Path(stage_path).is_file())
    print(f"[V34B_ISAAC_STAGE] usd={stage_path} exists={stage_exists} source={stage_source}", flush=True)
    if stage_exists != 1:
        print("[V34B_PHYSICS_DRIVE] ok=0 mode=diff_drive dt=0 steps=0 reason=stage_missing", flush=True)
        return 2

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame_dir = out / "capture"
    frame_dir.mkdir(parents=True, exist_ok=True)

    from omni.isaac.kit import SimulationApp  # type: ignore

    sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    contacts = 0
    tunnel = 0
    max_penetration = 0.0
    camera_fail = 0

    try:
        from omni.isaac.core import World  # type: ignore
        from omni.isaac.core.objects import DynamicCuboid, FixedCuboid  # type: ignore
        from omni.isaac.core.utils.stage import open_stage  # type: ignore
        from omni.isaac.sensor import Camera  # type: ignore

        if not open_stage(stage_path):
            print("[V34B_PHYSICS_DRIVE] ok=0 mode=diff_drive dt=0 steps=0 reason=open_stage_failed", flush=True)
            return 2

        world = World(stage_units_in_meters=1.0, physics_dt=float(args.dt), rendering_dt=float(args.dt))
        world.reset()
        for _ in range(20):
            world.step(render=True)

        start_x = float(cfg.get("physics_start_x", -2.0))
        start_z = float(cfg.get("physics_start_z", -2.0))
        ground_y = _raycast_ground_y(start_x, start_z, default_y=0.0)

        robot_h = 0.26
        robot_half_h = robot_h * 0.5
        robot = world.scene.add(
            DynamicCuboid(
                prim_path="/World/V34B2Robot",
                name="v34b2_robot",
                position=np.array([start_x, ground_y + robot_half_h + float(args.base_clearance), start_z], dtype=np.float32),
                scale=np.array([0.36, robot_h, 0.36], dtype=np.float32),
                color=np.array([0.1, 0.7, 0.2], dtype=np.float32),
                mass=20.0,
            )
        )
        # deterministic blocker wall in front of start (collision regression check)
        wall_z = start_z - 1.20
        wall_half_z = 0.10
        robot_half_z = 0.18
        world.scene.add(
            FixedCuboid(
                prim_path="/World/V34B2Wall",
                name="v34b2_wall",
                position=np.array([start_x, ground_y + 0.70, wall_z], dtype=np.float32),
                scale=np.array([1.80, 1.40, wall_half_z * 2.0], dtype=np.float32),
                color=np.array([0.8, 0.2, 0.2], dtype=np.float32),
            )
        )

        camera = Camera(
            prim_path="/World/V34B2Camera",
            frequency=20,
            resolution=(int(args.cam_w), int(args.cam_h)),
            position=np.array([start_x, ground_y + 1.0, start_z], dtype=np.float32),
            orientation=_quat_wxyz_from_yaw(0.0),
        )
        camera.initialize()
        try:
            camera.add_rgba_to_frame()
        except Exception:
            pass

        world.reset()
        spawn_pos, spawn_quat = robot.get_world_pose()
        spawn_y = float(spawn_pos[1])
        print(
            f"[ISAAC_SPAWN] ok=1 x={float(spawn_pos[0]):.3f} y={spawn_y:.3f} z={float(spawn_pos[2]):.3f} "
            f"ground_y={ground_y:.3f} clearance={float(args.base_clearance):.3f}",
            flush=True,
        )

        settle_contacts = 0
        for _ in range(int(args.settle_steps)):
            robot.set_linear_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            robot.set_angular_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            world.step(render=True)
            pos, _ = robot.get_world_pose()
            expected_y = ground_y + robot_half_h + float(args.base_clearance)
            penetration = max(0.0, expected_y - float(pos[1]))
            max_penetration = max(max_penetration, penetration)
            if penetration > 0.005:
                settle_contacts += 1
            cam_pos = np.array([float(pos[0]), float(pos[1]) + 0.42, float(pos[2])], dtype=np.float32)
            camera.set_world_pose(position=cam_pos, orientation=_quat_wxyz_from_yaw(0.0))

        print(
            f"[ISAAC_SETTLE] ok=1 steps={int(args.settle_steps)} max_penetration={max_penetration:.4f} contacts={int(settle_contacts)}",
            flush=True,
        )

        for i in range(int(args.steps)):
            if i < int(args.cmd_steps):
                robot.set_linear_velocity(np.array([0.0, 0.0, -float(args.speed_mps)], dtype=np.float32))
            else:
                robot.set_linear_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            robot.set_angular_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            prev_pos, _ = robot.get_world_pose()
            world.step(render=True)
            pos, _ = robot.get_world_pose()

            dx = float(pos[0] - prev_pos[0])
            dz = float(pos[2] - prev_pos[2])
            dxy = math.hypot(dx, dz)
            near_wall = float(pos[2]) <= (wall_z + wall_half_z + robot_half_z + 0.05)
            if i < int(args.cmd_steps) and near_wall and dxy < 0.003:
                contacts += 1
            if float(pos[2]) < (wall_z - wall_half_z - robot_half_z - 0.02):
                tunnel = 1

            cam_pos = np.array([float(pos[0]), float(pos[1]) + 0.42, float(pos[2])], dtype=np.float32)
            camera.set_world_pose(position=cam_pos, orientation=_quat_wxyz_from_yaw(0.0))
            world.step(render=True)
            rgb = _read_rgb(camera)
            if rgb is None:
                camera_fail += 1
                img = np.zeros((int(args.cam_h), int(args.cam_w), 3), dtype=np.uint8)
                img[..., :] = 10
                Image.fromarray(img).save(frame_dir / f"rgb_{i:03d}.png")
            else:
                Image.fromarray(rgb).save(frame_dir / f"rgb_{i:03d}.png")

        try:
            import imageio.v2 as imageio  # type: ignore

            frames = []
            for p in sorted(frame_dir.glob("rgb_*.png")):
                frames.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
            if len(frames) > 1:
                imageio.mimsave(out / "physics_drive.gif", frames, duration=0.08)
        except Exception:
            pass

        print(f"[PHYSICS_CAPTURE] ok={1 if camera_fail == 0 else 0} frames={int(args.steps)} dir={frame_dir}", flush=True)
        print(f"[V34B_PHYSICS_DRIVE] ok=1 mode=diff_drive dt={float(args.dt):.4f} steps={int(args.steps)}", flush=True)
        print(f"[V34B_COLLISION_CHECK] ok={1 if tunnel == 0 else 0} contacts={int(contacts)} tunnel={int(tunnel)}", flush=True)

        result = {
            "ok": int(tunnel == 0 and camera_fail == 0),
            "stage": stage_path,
            "spawn": {
                "x": float(spawn_pos[0]),
                "y": float(spawn_pos[1]),
                "z": float(spawn_pos[2]),
                "ground_y": float(ground_y),
            },
            "settle": {
                "steps": int(args.settle_steps),
                "max_penetration": float(max_penetration),
                "contacts": int(settle_contacts),
            },
            "contacts": int(contacts),
            "tunnel": int(tunnel),
            "camera_fail_frames": int(camera_fail),
            "capture_dir": str(frame_dir),
        }
        (out / "physics_drive_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0 if (tunnel == 0 and camera_fail == 0) else 3
    finally:
        try:
            sim_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

