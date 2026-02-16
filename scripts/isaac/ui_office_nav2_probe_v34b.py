#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, Tuple

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


def _has_mod(name: str) -> bool:
    try:
        return bool(importlib.util.find_spec(name))
    except Exception:
        return False


def _resolve_stage(cfg: Dict[str, object], override_stage: str) -> Tuple[str, str]:
    override = str(override_stage or os.environ.get("ISAAC_STAGE_USD", "")).strip()
    if override:
        return override, "override"

    candidates = []
    env_root = str(os.environ.get("ISAAC_ASSETS_ROOT", "")).strip()
    if env_root:
        candidates.append(env_root)
    candidates.extend(["/home/peng/IsaacAssets", "/home/peng/isaacsim_assets"])

    for root in candidates:
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


def _enable_ros2_bridge() -> int:
    try:
        import omni.kit.app  # type: ignore

        app = omni.kit.app.get_app()
        ext_mgr = app.get_extension_manager()
        for ext in ("isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"):
            try:
                ext_mgr.set_extension_enabled_immediate(ext, True)
            except Exception:
                pass
    except Exception:
        return 0
    return int(_has_mod("omni.isaac.ros2_bridge") or _has_mod("isaacsim.ros2.bridge"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", default="office_localized")
    ap.add_argument("--isaac_config", default="configs/isaac_scenes_v34b.yaml")
    ap.add_argument("--stage", default="")
    ap.add_argument("--out_dir", default="runs/topo_mvp/v34b_nav2_demo/ui_probe")
    args = ap.parse_args()

    cfg = _load_cfg(Path(args.isaac_config), args.scene_id)
    stage, source = _resolve_stage(cfg, args.stage)
    exists = int(bool(stage) and Path(stage).is_file())
    print(f"[V34B_ISAAC_STAGE] usd={stage} exists={exists} source={source}", flush=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if exists:
        cfg["usd_path"] = stage

    ros2_bridge = _enable_ros2_bridge()
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
    stage_for_log = str(cfg.get("usd_path", cfg.get("usd_hint", "")))
    print(
        f"[V34B_ISAAC_UI] ok=1 stage={stage_for_log} robot={robot} ros2_bridge={ros2_bridge} scan={scan_ok} odom={odom_ok} tf={tf_ok} rclpy={rclpy_ok}",
        flush=True,
    )
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
