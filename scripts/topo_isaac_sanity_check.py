#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict

import numpy as np


def _wrap_pi(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


def _load_scene_cfg(config_path: str, scene_id: str) -> Dict[str, object]:
    if not os.path.isfile(config_path):
        return {}
    payload = {}
    try:
        import yaml  # type: ignore

        with open(config_path, "r") as f:
            payload = yaml.safe_load(f) or {}
    except Exception:
        try:
            with open(config_path, "r") as f:
                payload = json.load(f)
        except Exception:
            payload = {}
    scenes = payload.get("scenes", []) if isinstance(payload, dict) else []
    for s in scenes:
        if str(s.get("scene_id", "")) == str(scene_id):
            return dict(s)
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", type=str, required=True)
    ap.add_argument("--isaac_config", type=str, required=True)
    ap.add_argument("--start_offset", type=int, default=0)
    ap.add_argument("--steps_forward", type=int, default=10)
    ap.add_argument("--steps_turn", type=int, default=10)
    args = ap.parse_args()

    from internnav.sim_backend.isaac_backend import IsaacSimBackend

    cfg = _load_scene_cfg(args.isaac_config, args.scene_id)
    backend = IsaacSimBackend(
        scene_id=args.scene_id,
        scene_cfg=cfg,
        config_path="",
        out_dir=".",
        fallback_to_habitat=bool(int(os.environ.get("ISAAC_BACKEND_ALLOW_FALLBACK", "0"))),
        dt_action=float(cfg.get("dt_action", 0.5)) if cfg else 0.5,
    )
    obs = backend.reset(
        scene_id=args.scene_id,
        start_spec={"start_offset": int(args.start_offset)},
    )

    fwd_dists = []
    align_cos = []
    for _ in range(int(args.steps_forward)):
        prev = backend.get_pose()
        obs, _, _ = backend.step(1)
        cur = backend.get_pose()
        dx = float(cur.x - prev.x)
        dz = float(cur.z - prev.z)
        dist = float(math.sqrt(dx * dx + dz * dz))
        fwd_dists.append(dist)
        yaw = float(prev.yaw)
        heading = np.array([math.sin(yaw), -math.cos(yaw)], dtype=np.float32)
        disp = np.array([dx, dz], dtype=np.float32)
        if np.linalg.norm(disp) > 1e-6:
            cos = float(np.dot(heading, disp) / (np.linalg.norm(heading) * np.linalg.norm(disp)))
            align_cos.append(cos)

    left_yaws = []
    for _ in range(int(args.steps_turn)):
        prev = backend.get_pose()
        obs, _, _ = backend.step(2)
        cur = backend.get_pose()
        dyaw = _wrap_pi(float(cur.yaw - prev.yaw))
        left_yaws.append(float(dyaw))

    right_yaws = []
    for _ in range(int(args.steps_turn)):
        prev = backend.get_pose()
        obs, _, _ = backend.step(3)
        cur = backend.get_pose()
        dyaw = _wrap_pi(float(cur.yaw - prev.yaw))
        right_yaws.append(float(dyaw))

    fwd_mean = float(np.mean(fwd_dists) if fwd_dists else 0.0)
    fwd_std = float(np.std(fwd_dists) if fwd_dists else 0.0)
    yaw_left_mean = float(np.mean(left_yaws) if left_yaws else 0.0)
    yaw_right_mean = float(np.mean(right_yaws) if right_yaws else 0.0)
    align_mean = float(np.mean(align_cos) if align_cos else 0.0)

    print(f"[ISAAC_SANITY] forward_step_mean={fwd_mean:.4f} forward_step_std={fwd_std:.4f}", flush=True)
    print(f"[ISAAC_SANITY] yaw_left_mean={yaw_left_mean:.4f} yaw_right_mean={yaw_right_mean:.4f}", flush=True)
    print(f"[ISAAC_SANITY] heading_alignment_cos={align_mean:.4f}", flush=True)

    if fwd_mean < 0.02 or align_mean < 0.7:
        print(
            f"[ISAAC_SANITY_FAIL] forward_step_mean={fwd_mean:.4f} heading_alignment_cos={align_mean:.4f}",
            flush=True,
        )
        backend.close()
        return 2
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
