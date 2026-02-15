import math
from typing import Tuple

import numpy as np


def _safe_depth(depth: np.ndarray, max_depth: float) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if d.ndim != 2:
        return np.zeros((1, 1), dtype=np.float32)
    d = np.where(np.isfinite(d), d, 0.0)
    d = np.clip(d, 0.0, float(max_depth))
    return d


def compute_scan_context(
    depth: np.ndarray,
    bins_r: int = 20,
    bins_theta: int = 60,
    max_depth: float = 5.0,
) -> np.ndarray:
    depth = _safe_depth(depth, max_depth=max_depth)
    h, w = depth.shape
    if h == 0 or w == 0 or bins_r <= 0 or bins_theta <= 0:
        return np.zeros((max(1, bins_r * bins_theta),), dtype=np.float32)

    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    nx = (x - cx) / max(1.0, cx)
    ny = (y - cy) / max(1.0, cy)
    rr = np.sqrt(nx * nx + ny * ny)
    theta = np.mod(np.arctan2(ny, nx) + math.pi, 2.0 * math.pi)

    valid = (depth > 1e-4) & np.isfinite(depth) & (rr <= 1.0)
    if not np.any(valid):
        return np.zeros((bins_r * bins_theta,), dtype=np.float32)

    r_idx = np.floor(rr[valid] * bins_r).astype(np.int32)
    t_idx = np.floor(theta[valid] * (bins_theta / (2.0 * math.pi))).astype(np.int32)
    r_idx = np.clip(r_idx, 0, bins_r - 1)
    t_idx = np.clip(t_idx, 0, bins_theta - 1)

    # Closer obstacles contribute stronger responses.
    strength = 1.0 - (depth[valid] / max(1e-6, float(max_depth)))
    strength = np.clip(strength, 0.0, 1.0)

    sc = np.zeros((bins_r, bins_theta), dtype=np.float32)
    np.maximum.at(sc, (r_idx, t_idx), strength.astype(np.float32))
    flat = sc.reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(flat))
    if norm > 1e-8:
        flat /= norm
    return flat

