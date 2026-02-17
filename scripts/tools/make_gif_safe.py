#!/usr/bin/env python3
"""Build a GIF from PNG frames with stride, guaranteeing >= min_frames.

Usage:
  python3 make_gif_safe.py --capture_dir runs/.../capture --out rgb.gif \
      --stride 3 --min_frames 20 --max_width 480
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def _find_rgb(root: Path) -> List[Path]:
    cands: List[Path] = []
    for pat in ("rgb_*.png", "overlay_rgb_*.png"):
        cands.extend(root.glob(pat))
    if not cands:
        cands.extend(root.glob("*.png"))
    return sorted(set(p for p in cands if p.is_file()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Safe GIF builder with min-frame gate")
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--out", required=True, help="Output GIF path")
    ap.add_argument("--stride", type=int, default=3, help="Pick every N-th frame")
    ap.add_argument("--min_frames", type=int, default=20, help="Hard minimum frames in GIF")
    ap.add_argument("--max_width", type=int, default=480, help="Resize to this width max")
    ap.add_argument("--duration_ms", type=int, default=150, help="Per-frame duration ms")
    ap.add_argument("--glob", default="rgb_*.png", help="Glob pattern for source PNGs")
    ap.add_argument("--view", default="", help="View label for anchor (e.g. chase, fp)")
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    if not cap.exists():
        print(f"[GIF_FRAMES] ok=0 reason=capture_dir_missing path={args.out}", flush=True)
        return 2

    # Collect source PNGs
    srcs = sorted(cap.glob(args.glob))
    if not srcs:
        srcs = _find_rgb(cap)
    if not srcs:
        print(f"[GIF_FRAMES] ok=0 reason=no_source_pngs path={args.out} dir={cap}", flush=True)
        return 2

    # Apply stride, but adjust if we'd end up below min_frames
    stride = max(1, args.stride)
    picked = srcs[::stride]
    # If too few, reduce stride dynamically
    while len(picked) < args.min_frames and stride > 1:
        stride -= 1
        picked = srcs[::stride]
    # If STILL too few (source itself is under min_frames), take ALL
    if len(picked) < args.min_frames:
        picked = srcs

    if len(picked) < args.min_frames:
        print(
            f"[GIF_FRAMES] ok=0 reason=insufficient_source_frames "
            f"available={len(srcs)} picked={len(picked)} min={args.min_frames} "
            f"path={args.out}",
            flush=True,
        )
        return 3

    # Load and resize
    frames = []
    for p in picked:
        try:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            if w > args.max_width:
                ratio = args.max_width / w
                im = im.resize((args.max_width, int(h * ratio)), Image.LANCZOS)
            frames.append(im)
        except Exception:
            continue

    if len(frames) < args.min_frames:
        print(
            f"[GIF_FRAMES] ok=0 reason=too_few_loadable_frames "
            f"loaded={len(frames)} min={args.min_frames} path={args.out}",
            flush=True,
        )
        return 3

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    size_kb = out.stat().st_size / 1024
    view_str = f" view={args.view}" if args.view else ""
    print(
        f"[GIF_FRAMES] ok=1 path={out} frames={len(frames)} stride={stride} "
        f"size_kb={size_kb:.0f} max_width={args.max_width}{view_str}",
        flush=True,
    )
    if args.view:
        print(
            f"[V35C_GIF] view={args.view} frames={len(frames)} stride={stride} path={out}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
