#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def _find_rgb_files(root: Path) -> List[Path]:
    cands: List[Path] = []
    for pat in ("**/rgb_*.png", "**/rgb_*.jpg", "**/rgb_*.jpeg"):
        cands.extend(root.glob(pat))
    # fallback for debug capture naming if needed
    if not cands:
        cands.extend(root.glob("**/*.png"))
    return sorted(set(p for p in cands if p.is_file()))


def _find_depth_npy_files(root: Path) -> List[Path]:
    cands = sorted(set(p for p in root.glob("**/depth_*.npy") if p.is_file()))
    return cands


def _load_image(p: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB"), dtype=np.float32)
    except Exception:
        return None


def _load_depth(p: Path) -> Optional[np.ndarray]:
    try:
        d = np.asarray(np.load(p), dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        if d.ndim != 2:
            return None
        return d
    except Exception:
        return None


def _pair_diffs(frames: List[np.ndarray]) -> Tuple[List[float], int]:
    diffs: List[float] = []
    identical = 0
    for i in range(1, len(frames)):
        a = frames[i - 1]
        b = frames[i]
        if a.shape != b.shape:
            continue
        delta = float(np.mean(np.abs(a - b)))
        diffs.append(delta)
        if delta <= 1e-9:
            identical += 1
    return diffs, identical


def _depth_pair_diffs(depths: List[np.ndarray]) -> Tuple[List[float], int]:
    diffs: List[float] = []
    identical = 0
    for i in range(1, len(depths)):
        a = depths[i - 1]
        b = depths[i]
        if a.shape != b.shape:
            continue
        valid = np.isfinite(a) & np.isfinite(b)
        if not valid.any():
            continue
        delta = float(np.mean(np.abs(a[valid] - b[valid])))
        diffs.append(delta)
        if delta <= 1e-9:
            identical += 1
    return diffs, identical


def _write_reports(root: Path, report: dict) -> None:
    (root / "motion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = []
    md.append("# Capture Motion Report")
    md.append("")
    md.append(f"- frames: {report.get('frames', 0)}")
    md.append(f"- identical_pairs: {report.get('identical_pairs', 0)}")
    md.append(f"- mean_rgb_diff: {report.get('mean_rgb_diff', 0.0):.6f}")
    md.append(f"- mean_depth_diff: {report.get('mean_depth_diff', 0.0):.6f}")
    md.append(f"- rgb_files: {report.get('rgb_files', 0)}")
    md.append(f"- depth_files: {report.get('depth_files', 0)}")
    md.append("")
    (root / "motion_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _try_gif(root: Path, rgbs: List[Path]) -> bool:
    if len(rgbs) == 0:
        return False
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception:
        return False
    frames = []
    for p in rgbs[:30]:
        try:
            frames.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
        except Exception:
            continue
    if len(frames) < 2:
        return False
    out = root / "rgb.gif"
    imageio.mimsave(out, frames, duration=0.2)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Check capture motion by frame differences")
    ap.add_argument("--capture_dir", required=True, type=str)
    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--write_gif", default="1", type=str)
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    if not cap.exists():
        print(f"[CAPTURE_MOTION] frames=0 identical_pairs=0 mean_rgb_diff=0.000000 mean_depth_diff=0.000000 reason=missing_capture_dir tag={args.tag}")
        return 2

    rgb_files_all = _find_rgb_files(cap)
    rgb_files: List[Path] = []
    rgb_frames: List[np.ndarray] = []
    for p in rgb_files_all:
        arr = _load_image(p)
        if arr is None:
            continue
        rgb_files.append(p)
        rgb_frames.append(arr)
    rgb_diffs, rgb_identical = _pair_diffs(rgb_frames)

    depth_files = _find_depth_npy_files(cap)
    depths = []
    for p in depth_files:
        d = _load_depth(p)
        if d is not None:
            depths.append(d)
    depth_diffs, depth_identical = _depth_pair_diffs(depths)

    mean_rgb = float(np.mean(rgb_diffs)) if rgb_diffs else 0.0
    mean_depth = float(np.mean(depth_diffs)) if depth_diffs else 0.0
    identical_pairs = int(max(rgb_identical, depth_identical))
    report = {
        "tag": str(args.tag or ""),
        "capture_dir": str(cap),
        "frames": int(len(rgb_frames)),
        "rgb_files": int(len(rgb_files)),
        "depth_files": int(len(depth_files)),
        "rgb_pair_count": int(len(rgb_diffs)),
        "depth_pair_count": int(len(depth_diffs)),
        "identical_pairs": int(identical_pairs),
        "mean_rgb_diff": float(mean_rgb),
        "mean_depth_diff": float(mean_depth),
    }
    _write_reports(cap, report)

    gif_written = False
    if str(args.write_gif).strip() not in {"0", "false", "False"}:
        gif_written = _try_gif(cap, rgb_files)

    print(
        f"[CAPTURE_MOTION] frames={len(rgb_frames)} identical_pairs={identical_pairs} "
        f"mean_rgb_diff={mean_rgb:.6f} mean_depth_diff={mean_depth:.6f} "
        f"gif={int(gif_written)} tag={args.tag}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
