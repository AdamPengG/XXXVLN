#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from internnav.sim_backend.isaac_backend import IsaacSimBackend
from internnav.sim_backend.isaac_camera_config_v29 import depth_stats, rgb_fidelity_stats


def _load_scene_cfg(config_path: str, scene_id: str) -> Dict[str, object]:
    path = Path(config_path)
    if not path.exists():
        return {}
    payload = {}
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text()) or {}
    except Exception:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        return {}
    for row in payload.get("scenes", []) or []:
        if str(row.get("scene_id", "")) == str(scene_id):
            return dict(row)
    return {}


def run(
    scene_id: str,
    config_path: str,
    out_dir: str,
    frames: int,
    stage_override: str,
) -> int:
    out = Path(out_dir)
    rgb_dir = out / "rgb"
    depth_dir = out / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    scene_cfg = _load_scene_cfg(config_path=config_path, scene_id=scene_id)
    if stage_override:
        scene_cfg = dict(scene_cfg)
        scene_cfg["usd_path"] = stage_override
        scene_cfg["usd_hint"] = Path(stage_override).name

    backend = IsaacSimBackend(
        scene_id=scene_id,
        scene_cfg=scene_cfg,
        config_path="",
        out_dir=str(out),
        fallback_to_habitat=False,
        dt_action=float(scene_cfg.get("dt_action", 0.5)) if scene_cfg else 0.5,
    )
    obs = backend.reset(scene_id=scene_id, start_spec={"start_offset": 0})

    rgbs: List[np.ndarray] = []
    depths: List[np.ndarray] = []
    actions = [1, 2, 1, 3]
    n = max(3, int(frames))
    for i in range(n):
        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        Image.fromarray(rgb).save(rgb_dir / f"rgb_{i:04d}.png")
        rgbs.append(rgb)
        if obs.depth is not None:
            d = np.asarray(obs.depth, dtype=np.float32)
            depths.append(d)
            np.save(depth_dir / f"depth_{i:04d}.npy", d)
        obs, _, _ = backend.step(actions[i % len(actions)])

    rgb_stats = rgb_fidelity_stats(rgbs)
    d_stats = depth_stats(depths)
    camera = {}
    if hasattr(backend, "get_camera_info"):
        try:
            camera = dict(getattr(backend, "get_camera_info")())
        except Exception:
            camera = {}
    meta = {
        "scene_id": scene_id,
        "config_path": config_path,
        "stage_path": str(scene_cfg.get("usd_path", "")),
        "frames": int(n),
        "camera": camera,
        "rgb_stats": rgb_stats,
        "depth_stats": d_stats,
        "renderer_used": str(camera.get("renderer_used", "unknown")),
        "files_rgb": [str(p) for p in sorted(rgb_dir.glob("*.png"))],
    }
    (out / "camera_fidelity_meta.json").write_text(json.dumps(meta, indent=2))
    try:
        backend.close()
    except Exception:
        pass

    ok = (
        float(rgb_stats.get("placeholder_ratio", 1.0)) < 0.95
        and float(rgb_stats.get("mean_luma", 0.0)) > 2.0
        and float(rgb_stats.get("mean_luma", 0.0)) < 252.0
        and float(rgb_stats.get("mean_frame_delta", 0.0)) > 0.0
    )
    if ok:
        print(
            f"[ISAAC_CAMERA_FIDELITY] ok=1 frames={n} "
            f"placeholder_ratio={float(rgb_stats['placeholder_ratio']):.3f} "
            f"mean_luma={float(rgb_stats['mean_luma']):.3f} "
            f"luma_std={float(rgb_stats['luma_std']):.3f} "
            f"mean_frame_delta={float(rgb_stats['mean_frame_delta']):.3f}",
            flush=True,
        )
        if len(depths) > 0:
            print(
                f"[ISAAC_DEPTH] ok={int(d_stats.get('ok', 0.0) > 0.5)} frames={len(depths)} "
                f"min={float(d_stats.get('min', 0.0)):.4f} max={float(d_stats.get('max', 0.0)):.4f} "
                f"invalid_ratio={float(d_stats.get('invalid_ratio', 1.0)):.4f}",
                flush=True,
            )
        return 0
    print(
        f"[ISAAC_CAMERA_FIDELITY] ok=0 reason=degenerate_stream frames={n} "
        f"placeholder_ratio={float(rgb_stats.get('placeholder_ratio', 1.0)):.3f} "
        f"hint=check_renderer_or_camera_cfg",
        flush=True,
    )
    return 4


def main() -> int:
    ap = argparse.ArgumentParser(description="Isaac v29 camera fidelity smoke.")
    ap.add_argument("--scene_id", type=str, default="office_localized")
    ap.add_argument("--isaac_config", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--stage", type=str, default="")
    args = ap.parse_args()
    stage = str(args.stage or "").strip()
    return run(
        scene_id=args.scene_id,
        config_path=args.isaac_config,
        out_dir=args.out_dir,
        frames=int(args.frames),
        stage_override=stage,
    )


if __name__ == "__main__":
    raise SystemExit(main())
