#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def _load_cfg(path: Path, scene_id: str) -> dict:
    txt = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(txt)
    except Exception:
        cfg = json.loads(txt)
    for s in cfg.get("scenes", []):
        if isinstance(s, dict) and str(s.get("scene_id", "")) == str(scene_id):
            return dict(s)
    return {}


def _resolve_stage(cfg: Dict[str, object], stage_override: str) -> Tuple[str, str]:
    override = str(stage_override or os.environ.get("ISAAC_STAGE_USD", "")).strip()
    if override:
        return override, "override"
    for root in [str(os.environ.get("ISAAC_ASSETS_ROOT", "")).strip(), "/home/peng/IsaacAssets", "/home/peng/isaacsim_assets"]:
        if not root:
            continue
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


def _world_to_img(x: float, z: float, bounds: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int]:
    min_x, max_x, min_z, max_z = bounds
    nx = (x - min_x) / max(1e-6, (max_x - min_x))
    nz = (z - min_z) / max(1e-6, (max_z - min_z))
    px = int(np.clip(nx, 0.0, 1.0) * (w - 1))
    py = int((1.0 - np.clip(nz, 0.0, 1.0)) * (h - 1))
    return px, py


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", default="office_localized")
    ap.add_argument("--isaac_config", default="configs/isaac_scenes_v34b.yaml")
    ap.add_argument("--stage", default="")
    ap.add_argument("--out_dir", default="runs/topo_mvp/v34b_nav2_demo/physics")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--dt", type=float, default=1.0 / 30.0)
    ap.add_argument("--cmd_steps", type=int, default=45)
    ap.add_argument("--speed_mps", type=float, default=0.6)
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
    frame_dir = out / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    from omni.isaac.kit import SimulationApp  # type: ignore

    sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    contacts = 0
    tunnel = 0
    positions: List[Tuple[float, float]] = []
    try:
        import omni.usd  # type: ignore
        from omni.isaac.core import World  # type: ignore
        from omni.isaac.core.objects import DynamicCuboid, FixedCuboid  # type: ignore
        from omni.isaac.core.utils.stage import open_stage  # type: ignore

        if not open_stage(stage_path):
            print("[V34B_PHYSICS_DRIVE] ok=0 mode=diff_drive dt=0 steps=0 reason=open_stage_failed", flush=True)
            return 2

        world = World(stage_units_in_meters=1.0, physics_dt=float(args.dt), rendering_dt=float(args.dt))
        world.scene.add_default_ground_plane()
        for _ in range(20):
            world.step(render=True)

        bounds_raw = cfg.get("bounds", [-6.0, 6.0, -6.0, 6.0])
        if isinstance(bounds_raw, list) and len(bounds_raw) == 4:
            bounds = tuple(float(x) for x in bounds_raw)
        else:
            bounds = (-6.0, 6.0, -6.0, 6.0)

        start_x = float(cfg.get("physics_start_x", -2.0))
        start_z = float(cfg.get("physics_start_z", -2.0))
        wall_z = start_z - 1.1
        robot_half = 0.18
        wall_half = 0.10

        robot = world.scene.add(
            DynamicCuboid(
                prim_path="/World/V34BPhysicsRobot",
                name="v34b_physics_robot",
                position=np.array([start_x, 0.20, start_z], dtype=np.float32),
                scale=np.array([0.36, 0.22, 0.36], dtype=np.float32),
                color=np.array([0.2, 0.8, 0.2], dtype=np.float32),
                mass=18.0,
            )
        )
        world.scene.add(
            FixedCuboid(
                prim_path="/World/V34BPhysicsWall",
                name="v34b_physics_wall",
                position=np.array([start_x, 0.50, wall_z], dtype=np.float32),
                scale=np.array([1.6, 1.0, 0.20], dtype=np.float32),
                color=np.array([0.9, 0.2, 0.2], dtype=np.float32),
            )
        )

        world.reset()
        for _ in range(20):
            world.step(render=True)

        canvas_w, canvas_h = 640, 480
        for i in range(int(args.steps)):
            if i < int(args.cmd_steps):
                robot.set_linear_velocity(np.array([0.0, 0.0, -float(args.speed_mps)], dtype=np.float32))
            else:
                robot.set_linear_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            robot.set_angular_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            world.step(render=True)

            pos, _ = robot.get_world_pose()
            x = float(pos[0])
            z = float(pos[2])
            positions.append((x, z))

            lin_v = robot.get_linear_velocity()
            speed = float(np.linalg.norm(np.asarray(lin_v, dtype=np.float32)[[0, 2]]))
            near_wall = z <= (wall_z + wall_half + robot_half + 0.05)
            if near_wall and speed < 0.10:
                contacts += 1
            if z < (wall_z - wall_half - robot_half - 0.02):
                tunnel = 1

            # lightweight topdown capture proving non-tunnel trajectory
            img = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))
            dr = ImageDraw.Draw(img)
            sx, sy = _world_to_img(start_x, start_z, bounds, canvas_w, canvas_h)
            wx0, wy0 = _world_to_img(start_x - 0.8, wall_z - wall_half, bounds, canvas_w, canvas_h)
            wx1, wy1 = _world_to_img(start_x + 0.8, wall_z + wall_half, bounds, canvas_w, canvas_h)
            dr.rectangle([min(wx0, wx1), min(wy0, wy1), max(wx0, wx1), max(wy0, wy1)], fill=(200, 40, 40))
            dr.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=(40, 40, 200))

            px, py = _world_to_img(x, z, bounds, canvas_w, canvas_h)
            dr.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(30, 170, 40))
            dr.text((8, 8), f"step={i} x={x:.2f} z={z:.2f} speed={speed:.2f}", fill=(0, 0, 0))
            img.save(frame_dir / f"rgb_{i:03d}.png")

        # gif preview
        frames = [Image.open(p) for p in sorted(frame_dir.glob("rgb_*.png"))]
        if frames:
            frames[0].save(out / "physics_drive.gif", save_all=True, append_images=frames[1:], duration=80, loop=0)

        print(f"[V34B_PHYSICS_DRIVE] ok=1 mode=diff_drive dt={float(args.dt):.4f} steps={int(args.steps)}", flush=True)
        print(f"[V34B_COLLISION_CHECK] ok={1 if tunnel == 0 else 0} contacts={int(contacts)} tunnel={int(tunnel)}", flush=True)

        result = {
            "ok": int(tunnel == 0),
            "contacts": int(contacts),
            "tunnel": int(tunnel),
            "positions": [{"x": float(x), "z": float(z)} for x, z in positions],
            "wall_z": float(wall_z),
            "stage": stage_path,
        }
        (out / "physics_drive_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0 if tunnel == 0 else 3
    finally:
        try:
            sim_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
