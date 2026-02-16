#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

OFFICE_DEFAULT = "/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"


def _resolve_stage(stage_arg: str) -> str:
    p = str(stage_arg or os.environ.get("ISAAC_STAGE_USD", "")).strip()
    if p:
        return p
    return OFFICE_DEFAULT


def _raycast_ground_y(default_x: float, default_z: float, default_y: float = 0.0) -> tuple[float, int]:
    try:
        import omni.physx as omni_physx  # type: ignore

        q = omni_physx.get_physx_scene_query_interface()
        hit = q.raycast_closest((float(default_x), 10.0, float(default_z)), (0.0, -1.0, 0.0), 40.0)
        if isinstance(hit, dict) and bool(hit.get("hit", False)):
            pos = hit.get("position")
            if pos is not None and len(pos) >= 2:
                return float(pos[1]), 1
    except Exception:
        pass
    return float(default_y), 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="")
    ap.add_argument("--out_dir", default="runs/topo_mvp/v34e_nav2_office_phys/physics")
    ap.add_argument("--x", type=float, default=-2.8)
    ap.add_argument("--z", type=float, default=-2.5)
    ap.add_argument("--clearance", type=float, default=0.06)
    ap.add_argument("--settle_steps", type=int, default=60)
    ap.add_argument("--dt", type=float, default=1.0 / 30.0)
    args = ap.parse_args()

    stage = _resolve_stage(args.stage)
    if not Path(stage).is_file():
        print("[V34E_SPAWN] ok=0 pos=(0,0,0) floor_hit=0 clearance=0.000 reason=stage_missing", flush=True)
        print("[V34E_SETTLE] ok=0 steps=0 max_penetration=0.0000 contacts=0", flush=True)
        print("[V34E_FALL_DETECT] ok=0 fell=1 min_clearance=-1.0000", flush=True)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from omni.isaac.kit import SimulationApp  # type: ignore

    sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    try:
        from omni.isaac.core import World  # type: ignore
        from omni.isaac.core.objects import DynamicCuboid  # type: ignore
        from omni.isaac.core.utils.stage import open_stage  # type: ignore

        if not open_stage(stage):
            print("[V34E_SPAWN] ok=0 pos=(0,0,0) floor_hit=0 clearance=0.000 reason=open_stage_failed", flush=True)
            print("[V34E_SETTLE] ok=0 steps=0 max_penetration=0.0000 contacts=0", flush=True)
            print("[V34E_FALL_DETECT] ok=0 fell=1 min_clearance=-1.0000", flush=True)
            return 2

        world = World(stage_units_in_meters=1.0, physics_dt=float(args.dt), rendering_dt=float(args.dt))
        world.reset()
        for _ in range(20):
            world.step(render=True)

        ground_y, floor_hit = _raycast_ground_y(args.x, args.z, default_y=0.0)
        robot_h = 0.26
        half_h = robot_h * 0.5
        spawn_y = float(ground_y + half_h + float(args.clearance))
        robot = world.scene.add(
            DynamicCuboid(
                prim_path="/World/V34ESpawnRobot",
                name="v34e_spawn_robot",
                position=np.array([float(args.x), spawn_y, float(args.z)], dtype=np.float32),
                scale=np.array([0.36, robot_h, 0.36], dtype=np.float32),
                color=np.array([0.2, 0.8, 0.2], dtype=np.float32),
                mass=20.0,
            )
        )

        world.reset()
        pos, _ = robot.get_world_pose()
        print(
            f"[V34E_SPAWN] ok=1 pos=({float(pos[0]):.3f},{float(pos[1]):.3f},{float(pos[2]):.3f}) floor_hit={int(floor_hit)} clearance={float(args.clearance):.3f}",
            flush=True,
        )

        max_penetration = 0.0
        contacts = 0
        fell = 0
        min_clearance = 1e9

        expected_base_y = float(ground_y + half_h)
        for _ in range(int(args.settle_steps)):
            robot.set_linear_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            robot.set_angular_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            world.step(render=True)
            p, _ = robot.get_world_pose()
            base_y = float(p[1])
            clearance_now = base_y - expected_base_y
            min_clearance = min(min_clearance, clearance_now)
            penetration = max(0.0, expected_base_y - base_y)
            max_penetration = max(max_penetration, penetration)
            if penetration > 0.003:
                contacts += 1
            if base_y < float(ground_y - 0.05):
                fell = 1

        settle_ok = int(fell == 0)
        print(
            f"[V34E_SETTLE] ok={settle_ok} steps={int(args.settle_steps)} max_penetration={max_penetration:.4f} contacts={contacts}",
            flush=True,
        )
        print(
            f"[V34E_FALL_DETECT] ok={settle_ok} fell={fell} min_clearance={min_clearance:.4f}",
            flush=True,
        )

        report = {
            "ok": settle_ok,
            "stage": stage,
            "spawn": {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "ground_y": float(ground_y),
                "clearance": float(args.clearance),
                "floor_hit": int(floor_hit),
            },
            "settle": {
                "steps": int(args.settle_steps),
                "max_penetration": float(max_penetration),
                "contacts": int(contacts),
                "fell": int(fell),
                "min_clearance": float(min_clearance),
            },
        }
        (out_dir / "spawn_settle_v34e.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0 if settle_ok == 1 else 3
    finally:
        try:
            sim_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
