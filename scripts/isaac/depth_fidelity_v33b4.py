#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from internnav.sim_backend.isaac_backend import IsaacSimBackend


def _load_scene_cfg(config_path: str, scene_id: str) -> Dict[str, object]:
    p = Path(config_path)
    if not p.exists():
        return {}
    payload = {}
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        return {}
    for row in payload.get("scenes", []) or []:
        if str(row.get("scene_id", "")) == str(scene_id):
            return dict(row)
    return {}


def _depth_to_png(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 1e-6)
    if not valid.any():
        return np.zeros((d.shape[0], d.shape[1]), dtype=np.uint8)
    vals = d[valid]
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals) + 1e-6)
    img = np.clip((d - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    img[~valid] = 0.0
    return (img * 255.0).astype(np.uint8)


def _depth_delta(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.shape != bb.shape:
        return 0.0
    valid = np.isfinite(aa) & np.isfinite(bb)
    if not valid.any():
        return 0.0
    return float(np.mean(np.abs(aa[valid] - bb[valid])))


def _rgb_luma(rgb: np.ndarray) -> float:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return 0.0
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(np.mean(luma))


def _wrap_pi(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def _action_name(action_id: int) -> str:
    return {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}.get(int(action_id), "UNKNOWN")


def run(scene_id: str, config_path: str, out_dir: str, frames: int, stage_override: str) -> int:
    out = Path(out_dir)
    rgb_dir = out / "rgb"
    depth_npy_dir = out / "depth_npy"
    depth_png_dir = out / "depth_png"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_npy_dir.mkdir(parents=True, exist_ok=True)
    depth_png_dir.mkdir(parents=True, exist_ok=True)

    scene_cfg = _load_scene_cfg(config_path=config_path, scene_id=scene_id)
    if not scene_cfg:
        print(f"[ISAAC_DEPTH_FIDELITY] ok=0 reason=scene_cfg_missing hint={scene_id}@{config_path}", flush=True)
        return 2
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

    try:
        backend.reset(scene_id=scene_id, start_spec={"start_offset": 0})
        n = max(10, int(frames))
        # Deterministic move-before-capture pattern to force observable scene changes.
        actions = [3, 3, 1, 2, 2, 1, 3, 1, 2, 1]
        rgbs: List[np.ndarray] = []
        depths: List[np.ndarray] = []
        depth_deltas: List[float] = []

        prev_rgb = None
        prev_depth = None
        prev_pose = None
        for i in range(n):
            act = int(actions[i % len(actions)])
            print(f"[ISAAC_CAPTURE_STEP] frame={i} action={_action_name(act)}", flush=True)
            obs, _, _ = backend.step(act)

            pose = backend.get_pose()
            if prev_pose is None:
                dpos = 0.0
                dyaw_deg = 0.0
            else:
                dpos = float(math.hypot(float(pose.x) - float(prev_pose.x), float(pose.z) - float(prev_pose.z)))
                dyaw_deg = float(math.degrees(_wrap_pi(float(pose.yaw) - float(prev_pose.yaw))))
            print(
                f"[ISAAC_CAPTURE_POSE] frame={i} x={float(pose.x):.4f} z={float(pose.z):.4f} "
                f"yaw_deg={float(math.degrees(float(pose.yaw))):.3f} dpos_m={dpos:.4f} dyaw_deg={dyaw_deg:.3f}",
                flush=True,
            )
            prev_pose = pose

            rgb = np.asarray(obs.rgb, dtype=np.uint8)
            depth = None if obs.depth is None else np.asarray(obs.depth, dtype=np.float32)
            Image.fromarray(rgb).save(rgb_dir / f"rgb_{i:03d}.png")
            rgbs.append(rgb)
            luma = _rgb_luma(rgb)
            frame_delta = 0.0
            if prev_rgb is not None and prev_rgb.shape == rgb.shape:
                frame_delta = float(np.mean(np.abs(rgb.astype(np.float32) - prev_rgb.astype(np.float32))))
            print(f"[ISAAC_RGB_SAMPLE] step={i} mean_luma={luma:.4f} frame_delta={frame_delta:.6f}", flush=True)

            if depth is None:
                print(f"[ISAAC_DEPTH_SAMPLE] step={i} valid_ratio=0.0000 min=nan p20=nan p50=nan p80=nan max=nan", flush=True)
                print(
                    f"[ISAAC_CAPTURE_HASH] frame={i} rgb_md5={hashlib.md5(np.ascontiguousarray(rgb).tobytes()).hexdigest()} depth_md5=none",
                    flush=True,
                )
            else:
                np.save(depth_npy_dir / f"depth_{i:03d}.npy", depth)
                depth_vis = _depth_to_png(depth)
                Image.fromarray(depth_vis).save(depth_png_dir / f"depth_{i:03d}.png")
                depths.append(depth)
                valid = np.isfinite(depth) & (depth > 1e-6)
                valid_ratio = float(valid.mean())
                if valid.any():
                    vals = depth[valid]
                    mn = float(np.min(vals))
                    p20 = float(np.percentile(vals, 20))
                    p50 = float(np.percentile(vals, 50))
                    p80 = float(np.percentile(vals, 80))
                    mx = float(np.max(vals))
                else:
                    mn = p20 = p50 = p80 = mx = float("nan")
                print(
                    f"[ISAAC_DEPTH_SAMPLE] step={i} valid_ratio={valid_ratio:.4f} min={mn:.4f} p20={p20:.4f} p50={p50:.4f} p80={p80:.4f} max={mx:.4f}",
                    flush=True,
                )
                rgb_md5 = hashlib.md5(np.ascontiguousarray(rgb).tobytes()).hexdigest()
                depth_md5 = hashlib.md5(np.ascontiguousarray(depth).tobytes()).hexdigest()
                print(f"[ISAAC_CAPTURE_HASH] frame={i} rgb_md5={rgb_md5} depth_md5={depth_md5}", flush=True)
                if prev_depth is not None:
                    depth_deltas.append(_depth_delta(prev_depth, depth))

            prev_rgb = rgb
            if depth is not None:
                prev_depth = depth

        depth_mean_delta = float(np.mean(depth_deltas)) if len(depth_deltas) > 0 else 0.0
        depth_std_delta = float(np.std(depth_deltas)) if len(depth_deltas) > 0 else 0.0
        depth_constant = int(depth_mean_delta < 0.002)

        meta = {
            "scene_id": scene_id,
            "config_path": config_path,
            "stage_path": str(scene_cfg.get("usd_path", "")),
            "frames": int(n),
            "depth_mean_delta": depth_mean_delta,
            "depth_std_delta": depth_std_delta,
            "depth_constant": int(depth_constant),
            "renderer_used": str(getattr(backend, "renderer_used", "unknown")),
            "files": {
                "rgb": [str(p) for p in sorted(rgb_dir.glob("*.png"))],
                "depth_npy": [str(p) for p in sorted(depth_npy_dir.glob("*.npy"))],
                "depth_png": [str(p) for p in sorted(depth_png_dir.glob("*.png"))],
            },
        }
        (out / "depth_fidelity_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        ok = (len(depths) >= 3) and (depth_constant == 0)
        if ok:
            print(
                f"[ISAAC_DEPTH_FIDELITY] ok=1 frames={n} depth_mean_delta={depth_mean_delta:.6f} "
                f"depth_std_delta={depth_std_delta:.6f} depth_constant=0",
                flush=True,
            )
            return 0
        print(
            f"[ISAAC_DEPTH_FIDELITY] ok=0 frames={n} depth_mean_delta={depth_mean_delta:.6f} "
            f"depth_std_delta={depth_std_delta:.6f} depth_constant={int(depth_constant)} "
            "hint=depth source may be stale or camera not updating",
            flush=True,
        )
        return 4
    finally:
        try:
            backend.close()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="v33b.4 Isaac depth fidelity diagnostic")
    ap.add_argument("--scene_id", type=str, default="office_localized")
    ap.add_argument("--isaac_config", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--stage", type=str, default="")
    args = ap.parse_args()
    return run(
        scene_id=args.scene_id,
        config_path=args.isaac_config,
        out_dir=args.out_dir,
        frames=int(args.frames),
        stage_override=str(args.stage or "").strip(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
