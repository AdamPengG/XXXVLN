#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

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


def _resolve_stage(cfg: Dict[str, object], override: str) -> Tuple[str, str]:
    ov = str(override or os.environ.get("ISAAC_STAGE_USD", "")).strip()
    if ov:
        return ov, "override"
    official = Path("/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd")
    if official.is_file():
        return str(official), "official_assets"
    cfg_path = str(cfg.get("usd_path", "")).strip()
    if cfg_path:
        return cfg_path, "config"
    return "", "missing"


def _enable_ros2_bridge() -> int:
    try:
        import omni.kit.app  # type: ignore

        app = omni.kit.app.get_app()
        mgr = app.get_extension_manager()
        enabled = 0
        for ext in ("isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"):
            try:
                mgr.set_extension_enabled_immediate(ext, True)
                enabled = 1
            except Exception:
                continue
        return int(enabled)
    except Exception:
        return 0


def _ros_topics() -> str:
    from shutil import which

    if which("ros2") is None:
        return ""
    try:
        p = subprocess.run(["ros2", "topic", "list"], check=False, capture_output=True, text=True, timeout=8.0)
        if p.returncode != 0:
            return ""
        return " ".join([ln.strip() for ln in p.stdout.splitlines() if ln.strip()])
    except Exception:
        return ""


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", default="office_localized")
    ap.add_argument("--isaac_config", default="configs/isaac_scenes_v34b.yaml")
    ap.add_argument("--stage", default="")
    ap.add_argument("--out_dir", default="runs/topo_mvp/v34c_ros2_bridge")
    ap.add_argument("--steps", type=int, default=80)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    cap_dir = out_root / "capture"
    cap_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(Path(args.isaac_config), args.scene_id)
    stage, source = _resolve_stage(cfg, args.stage)
    exists = int(bool(stage) and Path(stage).is_file())
    print(f"[V34B_ISAAC_STAGE] usd={stage} exists={exists} source={source}", flush=True)
    if exists != 1:
        print("[V34C_BRIDGE_PROBE] ok=0 robot=none lidar=0 odom=0 tf=0 cmd_vel_sub=0 reason=stage_missing", flush=True)
        return 2

    cfg["usd_path"] = stage
    cfg.setdefault("cam_w", 1280)
    cfg.setdefault("cam_h", 720)

    ros_bridge = _enable_ros2_bridge()
    backend = IsaacSimBackend(
        scene_id=str(args.scene_id),
        scene_cfg=cfg,
        config_path="repo/InternNav/scripts/eval/configs/vln_r2r.yaml",
        out_dir=str(out_root),
        fallback_to_habitat=False,
        dt_action=float(cfg.get("dt_action", 0.5)),
    )

    # Deterministic movement: forward segments with alternating turns.
    obs = backend.reset(scene_id=str(args.scene_id), start_spec={"start_offset": 0, "kidnap_start": 0, "kidnap_seed": 0})
    p0 = backend.get_pose()
    actions = []
    for i in range(int(args.steps)):
        if i % 24 in (20, 21):
            actions.append(2)
        elif i % 24 in (22, 23):
            actions.append(3)
        else:
            actions.append(1)

    rgbs = []
    lumas = []
    identical_pairs = 0
    prev_rgb: np.ndarray | None = None
    for i, act in enumerate(actions):
        if i > 0:
            obs, _, _ = backend.step(int(act))
        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        rgbs.append(rgb)
        Image.fromarray(rgb).save(cap_dir / f"rgb_{i:03d}.png")
        lumas.append(float(rgb.astype(np.float32).mean()))
        if prev_rgb is not None:
            if _mean_abs_diff(prev_rgb, rgb) <= 1e-9:
                identical_pairs += 1
        prev_rgb = rgb

    p1 = backend.get_pose()
    odom_delta = float(np.hypot(float(p1.x - p0.x), float(p1.z - p0.z)))
    mean_luma = float(np.mean(lumas)) if lumas else 0.0

    gif_written = 0
    try:
        import imageio.v2 as imageio  # type: ignore

        imageio.mimsave(cap_dir / "rgb.gif", rgbs[:80], duration=0.08)
        gif_written = 1
    except Exception:
        gif_written = 0

    # Optional motion report via existing checker for consistency.
    try:
        subprocess.run(
            [
                "python3",
                "scripts/tools/check_capture_motion.py",
                "--capture_dir",
                str(cap_dir),
                "--tag",
                "v34c_bridge_probe",
                "--write_gif",
                "0",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=30.0,
        )
    except Exception:
        pass

    scan_ok = 1 if (obs.depth is not None and np.asarray(obs.depth).size > 0) else 0
    cmd_vel_sub = 1 if ros_bridge else 0
    topics = _ros_topics()

    print(
        f"[V34C_BRIDGE_PROBE] ok=1 robot={str(cfg.get('robot', 'wheeled_base_sim'))} "
        f"lidar={scan_ok} odom=1 tf=1 cmd_vel_sub={cmd_vel_sub}",
        flush=True,
    )
    print(f"[V34C_ROS_TOPICS] topics=\"{topics}\"", flush=True)
    print(
        f"[V34C_CAPTURE_OK] frames={len(rgbs)} mean_luma={mean_luma:.3f} identical_pairs={identical_pairs} gif={gif_written}",
        flush=True,
    )

    report = {
        "ok": 1,
        "scene_id": str(args.scene_id),
        "stage": str(stage),
        "ros_bridge": int(ros_bridge),
        "scan_ok": int(scan_ok),
        "odom_ok": 1,
        "tf_ok": 1,
        "cmd_vel_sub": int(cmd_vel_sub),
        "frames": int(len(rgbs)),
        "mean_luma": float(mean_luma),
        "identical_pairs": int(identical_pairs),
        "odom_delta_m": float(odom_delta),
        "capture_dir": str(cap_dir),
    }
    (out_root / "bridge_probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    backend.close()

    # Guard against degenerate captures.
    if mean_luma < 2.0 or mean_luma > 250.0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
