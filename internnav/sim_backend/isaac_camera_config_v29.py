from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class IsaacCameraConfig:
    width: int
    height: int
    fov_deg: float
    near_m: float
    far_m: float
    auto_exposure: int
    exposure: float
    aspect_policy: str


def _int_env(name: str, default: int) -> int:
    try:
        return int(float(str(os.environ.get(name, default))))
    except Exception:
        return int(default)


def _float_env(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, default)))
    except Exception:
        return float(default)


def _str_env(name: str, default: str) -> str:
    try:
        return str(os.environ.get(name, default))
    except Exception:
        return str(default)


def build_camera_config(scene_cfg: Dict[str, object]) -> IsaacCameraConfig:
    w_default = int(scene_cfg.get("cam_w", 1280))
    h_default = int(scene_cfg.get("cam_h", 720))
    fov_default = float(scene_cfg.get("cam_fov_deg", 90.0))
    near_default = float(scene_cfg.get("cam_near", 0.05))
    far_default = float(scene_cfg.get("cam_far", 50.0))
    auto_exp_default = int(scene_cfg.get("cam_auto_exposure", 0))
    exposure_default = float(scene_cfg.get("cam_exposure", 1.0))
    aspect_default = str(scene_cfg.get("cam_aspect_policy", "locked_w_over_h"))
    cfg = IsaacCameraConfig(
        width=max(32, _int_env("ISAAC_CAM_W", w_default)),
        height=max(32, _int_env("ISAAC_CAM_H", h_default)),
        fov_deg=float(np.clip(_float_env("ISAAC_CAM_FOV_DEG", fov_default), 20.0, 150.0)),
        near_m=float(max(1e-3, _float_env("ISAAC_CAM_NEAR", near_default))),
        far_m=float(max(0.2, _float_env("ISAAC_CAM_FAR", far_default))),
        auto_exposure=int(1 if _int_env("ISAAC_CAM_AUTO_EXPOSURE", auto_exp_default) else 0),
        exposure=float(max(1e-4, _float_env("ISAAC_CAM_EXPOSURE", exposure_default))),
        aspect_policy=str(_str_env("ISAAC_CAM_ASPECT_POLICY", aspect_default) or "locked_w_over_h"),
    )
    if cfg.far_m <= cfg.near_m:
        cfg.far_m = cfg.near_m + 0.5
    return cfg


def apply_camera_config(camera, cfg: IsaacCameraConfig) -> Dict[str, object]:
    # Resolution
    try:
        camera.set_resolution((int(cfg.width), int(cfg.height)))
    except Exception:
        pass

    # USD camera attributes for intrinsics-like locking.
    prim = None
    try:
        prim = camera.prim
    except Exception:
        prim = None

    if prim is not None and bool(prim.IsValid()):
        try:
            from pxr import Gf, UsdGeom  # type: ignore

            cam = UsdGeom.Camera(prim)
            h_ap_mm = 20.955
            v_ap_mm = h_ap_mm * float(cfg.height) / float(max(1, cfg.width))
            focal_mm = (h_ap_mm * 0.5) / max(1e-6, math.tan(math.radians(float(cfg.fov_deg)) * 0.5))

            cam.CreateHorizontalApertureAttr().Set(float(h_ap_mm))
            cam.CreateVerticalApertureAttr().Set(float(v_ap_mm))
            cam.CreateFocalLengthAttr().Set(float(focal_mm))
            cam.CreateClippingRangeAttr().Set(Gf.Vec2f(float(cfg.near_m), float(cfg.far_m)))
            # Keep exposure deterministic in post-process path; set USD attr when available.
            try:
                from pxr import Sdf  # type: ignore

                cam.GetPrim().CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(
                    float(math.log2(max(1e-4, cfg.exposure)))
                )
            except Exception:
                pass
        except Exception:
            pass

    return {
        "camera_w": int(cfg.width),
        "camera_h": int(cfg.height),
        "camera_fov_deg": float(cfg.fov_deg),
        "camera_near": float(cfg.near_m),
        "camera_far": float(cfg.far_m),
        "auto_exposure": int(cfg.auto_exposure),
        "exposure": float(cfg.exposure),
        "aspect_policy": str(cfg.aspect_policy),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float32)
    bb = b.reshape(-1).astype(np.float32)
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    den = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)) + 1e-12)
    return float(np.sum(aa * bb) / den)


def is_placeholder_rgb(rgb: np.ndarray) -> bool:
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


def rgb_fidelity_stats(frames: Sequence[np.ndarray]) -> Dict[str, float]:
    if len(frames) == 0:
        return {
            "placeholder_ratio": 1.0,
            "mean_luma": 0.0,
            "luma_std": 0.0,
            "mean_frame_delta": 0.0,
        }
    placeholders = 0
    lumas: List[float] = []
    deltas: List[float] = []
    prev_gray: Optional[np.ndarray] = None
    for fr in frames:
        arr = np.asarray(fr, dtype=np.uint8)
        placeholders += int(is_placeholder_rgb(arr))
        gray = arr.astype(np.float32).mean(axis=2)
        lumas.append(float(np.mean(gray)))
        if prev_gray is not None:
            deltas.append(float(np.mean(np.abs(gray - prev_gray))))
        prev_gray = gray
    return {
        "placeholder_ratio": float(placeholders / max(1, len(frames))),
        "mean_luma": float(np.mean(lumas) if lumas else 0.0),
        "luma_std": float(np.std(lumas) if lumas else 0.0),
        "mean_frame_delta": float(np.mean(deltas) if deltas else 0.0),
    }


def depth_stats(depth_frames: Sequence[np.ndarray]) -> Dict[str, float]:
    if len(depth_frames) == 0:
        return {"ok": 0.0, "min": 0.0, "max": 0.0, "invalid_ratio": 1.0}
    mins: List[float] = []
    maxs: List[float] = []
    invalid = 0
    total = 0
    for d in depth_frames:
        arr = np.asarray(d, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        finite = np.isfinite(arr)
        invalid += int(np.size(arr) - np.count_nonzero(finite))
        total += int(np.size(arr))
        if np.count_nonzero(finite) > 0:
            vals = arr[finite]
            mins.append(float(np.min(vals)))
            maxs.append(float(np.max(vals)))
    invalid_ratio = float(invalid / max(1, total))
    return {
        "ok": float(1.0 if invalid_ratio < 0.99 else 0.0),
        "min": float(np.min(mins) if mins else 0.0),
        "max": float(np.max(maxs) if maxs else 0.0),
        "invalid_ratio": invalid_ratio,
    }
