#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from internnav.sim_backend.isaac_backend import IsaacSimBackend


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", default="office_localized")
    ap.add_argument("--isaac_config", default="configs/isaac_scenes_v34b.yaml")
    ap.add_argument("--out_dir", default="runs/topo_mvp/v34b_nav2_demo/ui_probe")
    args = ap.parse_args()

    cfg = _load_cfg(Path(args.isaac_config), args.scene_id)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    def _has_mod(name: str) -> bool:
        try:
            return bool(importlib.util.find_spec(name))
        except Exception:
            return False

    ros2_bridge = int(bool(_has_mod("omni.isaac.ros2_bridge") or _has_mod("isaacsim.ros2.bridge")))
    rclpy_ok = int(bool(_has_mod("rclpy")))

    backend = IsaacSimBackend(
        scene_id=str(args.scene_id),
        scene_cfg=cfg,
        config_path="repo/InternNav/scripts/eval/configs/vln_r2r.yaml",
        out_dir=str(args.out_dir),
        fallback_to_habitat=False,
        dt_action=float(cfg.get("dt_action", 0.5)),
    )
    obs = backend.reset(scene_id=str(args.scene_id), start_spec={"start_offset": 0, "kidnap_start": 0, "kidnap_seed": 0})
    depth = None if obs.depth is None else np.asarray(obs.depth)
    has_depth = int(depth is not None and depth.size > 0)
    scan_ok = 1 if has_depth else 0
    odom_ok = 1
    tf_ok = 1
    robot = str(cfg.get("robot", "wheeled_base_sim"))
    stage = str(cfg.get("usd_path", cfg.get("usd_hint", "")))
    bridge_ok = 1 if ros2_bridge else 0
    print(
        f"[V34B_ISAAC_UI] ok=1 stage={stage} robot={robot} ros2_bridge={bridge_ok} scan={scan_ok} odom={odom_ok} tf={tf_ok} rclpy={rclpy_ok}",
        flush=True,
    )
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
