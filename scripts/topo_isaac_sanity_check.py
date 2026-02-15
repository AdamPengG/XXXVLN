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

    # ── v26 render smoke test ─────────────────────────────────────────────
    # Isaac camera may need several render passes to warm up; do a few extra
    # forward steps to ensure the renderer has produced real pixels.
    for _ in range(5):
        obs, _, _ = backend.step(1)
    rgb = backend.get_rgb()
    rgb_ok = False
    rgb_reason = "not_tested"
    rgb_var = 0.0
    if rgb is not None and rgb.ndim == 3 and rgb.shape[-1] >= 3:
        h_img, w_img = rgb.shape[:2]
        rgb_f = rgb.astype(np.float32)
        rgb_var = float(np.var(rgb_f))

        # Detect known placeholder gradient signature:
        # _synthetic_rgbd produces R ∝ col_index, G ∝ row_index, B ∝ (1-col)
        # Check if R-channel is highly correlated with a horizontal ramp.
        is_placeholder = False
        if h_img > 4 and w_img > 4:
            col_ramp = np.tile(np.linspace(0, 1, w_img, dtype=np.float32), (h_img, 1))
            row_ramp = np.tile(np.linspace(0, 1, h_img, dtype=np.float32).reshape(-1, 1), (1, w_img))
            r_ch = rgb_f[:, :, 0] / 255.0
            g_ch = rgb_f[:, :, 1] / 255.0

            def _corr(a, b):
                a_flat = a.ravel()
                b_flat = b.ravel()
                a_m = a_flat - float(np.mean(a_flat))
                b_m = b_flat - float(np.mean(b_flat))
                denom = float(np.sqrt(np.sum(a_m * a_m) * np.sum(b_m * b_m) + 1e-12))
                return float(np.sum(a_m * b_m) / denom)

            corr_r_col = abs(_corr(r_ch, col_ramp))
            corr_g_row = abs(_corr(g_ch, row_ramp))
            # Both channels matching gradient → placeholder
            if corr_r_col > 0.88 and corr_g_row > 0.88:
                is_placeholder = True

        if is_placeholder:
            rgb_reason = "placeholder_detected"
        elif rgb_var < 100.0:
            rgb_reason = "low_variance"
        else:
            rgb_ok = True
            rgb_reason = "real"
    else:
        rgb_reason = "bad_shape" if rgb is not None else "rgb_none"

    if rgb_ok:
        h_img, w_img = rgb.shape[:2]
        print(
            f"[ISAAC_RENDER_SMOKE] ok=1 width={w_img} height={h_img} var={rgb_var:.1f}",
            flush=True,
        )
    else:
        print(
            f"[ISAAC_RENDER_SMOKE] ok=0 reason={rgb_reason} var={rgb_var:.1f}",
            flush=True,
        )

    if fwd_mean < 0.02 or align_mean < 0.7:
        print(
            f"[ISAAC_SANITY_FAIL] forward_step_mean={fwd_mean:.4f} heading_alignment_cos={align_mean:.4f}",
            flush=True,
        )
        backend.close()
        return 2

    # v26: if RGB capture is requested but render smoke failed, exit non-zero
    rgb_capture_requested = bool(int(os.environ.get("ISAAC_RGB_CAPTURE", "0")))
    if rgb_capture_requested and not rgb_ok:
        print(
            f"[ISAAC_RENDER_SMOKE_FAIL] reason={rgb_reason} rgb_capture_was_requested=1",
            flush=True,
        )
        backend.close()
        return 3

    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
