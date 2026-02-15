from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from internnav.sim_backend.isaac_backend import IsaacSimBackend


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


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12)
    return float(np.sum(a * b) / den)


def _is_placeholder_rgb(rgb: np.ndarray) -> bool:
    if rgb is None:
        return True
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return True
    if float(np.var(arr.astype(np.float32))) < 60.0:
        return True
    h, w = arr.shape[:2]
    if h < 16 or w < 16:
        return False
    x_grad = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
    y_grad = np.tile(np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1), (1, w))
    r = arr[..., 0].astype(np.float32) / 255.0
    g = arr[..., 1].astype(np.float32) / 255.0
    b = arr[..., 2].astype(np.float32) / 255.0
    c1 = abs(_corr(r, x_grad))
    c2 = abs(_corr(g, y_grad))
    c3 = abs(_corr(b, 1.0 - x_grad))
    return bool(c1 > 0.88 and c2 > 0.88 and c3 > 0.88)


def run_smoke(
    scene_id: str,
    config_path: str,
    out_dir: str,
    frames: int,
    fail_non_5090: bool,
    stage_override: str = "",
) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene_cfg = _load_scene_cfg(config_path=config_path, scene_id=scene_id)
    stage_override = str(stage_override or "").strip()
    if stage_override:
        scene_cfg = dict(scene_cfg)
        scene_cfg["usd_path"] = stage_override
        scene_cfg["usd_hint"] = Path(stage_override).name

    gpu_name = os.environ.get("ISAAC_GPU_NAME", "").strip()
    if fail_non_5090 and gpu_name and "5090" not in gpu_name:
        print(
            '[ISAAC_RENDER_SMOKE] ok=0 reason=unsupported_or_failed_gpu '
            'hint="use ISAAC_GPU_ID=0 (5090)"',
            flush=True,
        )
        return 3

    os.environ.setdefault("ISAAC_RGB_CAPTURE", "1")
    os.environ.setdefault("ISAAC_RENDER", "1")
    os.environ.setdefault("ISAAC_SKIP_WORLD_STEP", "0")
    os.environ.setdefault("ISAAC_MINIMAL", "0")

    backend = IsaacSimBackend(
        scene_id=scene_id,
        scene_cfg=scene_cfg,
        config_path="",
        out_dir=str(out),
        fallback_to_habitat=False,
        dt_action=float(scene_cfg.get("dt_action", 0.5)) if scene_cfg else 0.5,
    )
    obs = backend.reset(scene_id=scene_id, start_spec={"start_offset": 0})
    actions = [1, 2, 1, 3]
    saved: List[Tuple[str, bool]] = []
    for i in range(max(1, int(frames))):
        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        path = out / f"rgb_{i:04d}.png"
        Image.fromarray(rgb).save(path)
        placeholder = _is_placeholder_rgb(rgb)
        saved.append((str(path), placeholder))
        obs, _, _ = backend.step(actions[i % len(actions)])

    total = len(saved)
    placeholder = sum(1 for _, p in saved if p)
    ratio = float(placeholder / max(1, total))
    meta = {
        "scene_id": scene_id,
        "stage_path": str(scene_cfg.get("usd_path", "")),
        "frames": total,
        "placeholder_count": placeholder,
        "placeholder_ratio": ratio,
        "renderer_requested": os.environ.get("ISAAC_RENDERER", "rtx"),
        "renderer_used": os.environ.get("ISAAC_RENDERER_USED", "unknown"),
        "gpu_id": os.environ.get("ISAAC_GPU_ID", ""),
        "gpu_name": os.environ.get("ISAAC_GPU_NAME", ""),
        "files": [p for p, _ in saved],
    }
    (out / "smoke_meta.json").write_text(json.dumps(meta, indent=2))

    ok = (total >= 1) and (ratio < 0.95)
    if ok:
        print(
            f"[ISAAC_RENDER_SMOKE] ok=1 out_dir={out} frames={total} placeholder_ratio={ratio:.3f}",
            flush=True,
        )
        code = 0
    else:
        print(
            f"[ISAAC_RENDER_SMOKE] ok=0 reason=placeholder_or_empty out_dir={out} "
            f"frames={total} placeholder_ratio={ratio:.3f} hint=check_gpu_and_renderer",
            flush=True,
        )
        code = 4
    try:
        backend.close()
    except Exception:
        pass
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description="Isaac render smoke for v27.")
    ap.add_argument("--scene_id", type=str, default="isaac_office_min")
    ap.add_argument("--isaac_config", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--fail_non_5090", type=int, default=1)
    ap.add_argument("--stage", type=str, default="")
    args = ap.parse_args()
    stage = str(args.stage or "").strip() or str(os.environ.get("ISAAC_STAGE_USD", "")).strip()
    return run_smoke(
        scene_id=args.scene_id,
        config_path=args.isaac_config,
        out_dir=args.out_dir,
        frames=int(args.frames),
        fail_non_5090=bool(int(args.fail_non_5090)),
        stage_override=stage,
    )


if __name__ == "__main__":
    raise SystemExit(main())
