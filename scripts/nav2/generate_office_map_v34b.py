#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


@dataclass
class PrimFootprint:
    path: str
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    min_y: float
    max_y: float


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


def _resolve_stage(cfg: Dict[str, object], stage_override: str) -> Tuple[str, str]:
    override = str(stage_override or os.environ.get("ISAAC_STAGE_USD", "")).strip()
    if override:
        return override, "override"

    roots: List[str] = []
    env_root = str(os.environ.get("ISAAC_ASSETS_ROOT", "")).strip()
    if env_root:
        roots.append(env_root)
    roots.extend(["/home/peng/IsaacAssets", "/home/peng/isaacsim_assets"])

    for root in roots:
        for p in (
            Path(root) / "Assets/Isaac/5.1/Isaac/Environments/Office/office.usd",
            Path(root) / "Isaac/Environments/Office/office.usd",
            Path(root) / "Office/office.usd",
        ):
            if p.is_file():
                return str(p), "official_assets"

    usd_path = str(cfg.get("usd_path", "")).strip()
    if usd_path:
        return usd_path, "config"
    return "", "missing"


def _collect_footprints(stage) -> List[PrimFootprint]:
    from pxr import UsdGeom  # type: ignore

    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_], useExtentsHint=True)
    footprints: List[PrimFootprint] = []
    skip_words = (
        "floor",
        "ground",
        "ceiling",
        "sky",
        "light",
        "windowpane",
        "window_glass",
        "collisionproxyinvisible",
    )
    for prim in stage.TraverseAll():
        if not prim.IsActive() or prim.IsAbstract() or not prim.IsLoaded():
            continue
        if not prim.IsA(UsdGeom.Boundable):
            continue

        path = prim.GetPath().pathString
        p_lower = path.lower()
        if any(w in p_lower for w in skip_words):
            continue

        try:
            world = bbox_cache.ComputeWorldBound(prim)
            box = world.ComputeAlignedBox()
        except Exception:
            continue

        min_v = box.GetMin()
        max_v = box.GetMax()
        min_x, min_y, min_z = float(min_v[0]), float(min_v[1]), float(min_v[2])
        max_x, max_y, max_z = float(max_v[0]), float(max_v[1]), float(max_v[2])

        if not (np.isfinite(min_x) and np.isfinite(max_x) and np.isfinite(min_z) and np.isfinite(max_z)):
            continue
        if (max_x - min_x) <= 1e-3 or (max_z - min_z) <= 1e-3:
            continue
        # Ignore floor slabs and overhead geometry for 2D nav occupancy.
        if max_y < 0.08:
            continue
        if min_y > 2.8:
            continue
        if (max_y - min_y) < 0.05 and max_y < 0.25:
            continue
        # Clamp giant envelopes that are usually root bounds.
        if (max_x - min_x) > 80.0 or (max_z - min_z) > 80.0:
            continue

        footprints.append(
            PrimFootprint(
                path=path,
                min_x=min_x,
                max_x=max_x,
                min_z=min_z,
                max_z=max_z,
                min_y=min_y,
                max_y=max_y,
            )
        )
    return footprints


def _rasterize(
    footprints: Iterable[PrimFootprint],
    bounds: Tuple[float, float, float, float],
    resolution: float,
    inflate_m: float,
) -> np.ndarray:
    min_x, max_x, min_z, max_z = bounds
    w = max(10, int(math.ceil((max_x - min_x) / resolution)))
    h = max(10, int(math.ceil((max_z - min_z) / resolution)))
    grid = np.full((h, w), 254, dtype=np.uint8)

    for fp in footprints:
        x0 = max(0, int(math.floor((fp.min_x - min_x) / resolution)))
        x1 = min(w, int(math.ceil((fp.max_x - min_x) / resolution)))
        z0 = max(0, int(math.floor((fp.min_z - min_z) / resolution)))
        z1 = min(h, int(math.ceil((fp.max_z - min_z) / resolution)))
        if x1 <= x0 or z1 <= z0:
            continue
        grid[z0:z1, x0:x1] = 0

    # Border as occupied.
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0

    inflate_cells = max(0, int(round(float(inflate_m) / float(resolution))))
    if inflate_cells > 0:
        occ = grid == 0
        dil = occ.copy()
        offsets = []
        for dy in range(-inflate_cells, inflate_cells + 1):
            for dx in range(-inflate_cells, inflate_cells + 1):
                if dx * dx + dy * dy <= inflate_cells * inflate_cells:
                    offsets.append((dy, dx))
        h_, w_ = occ.shape
        for dy, dx in offsets:
            ys0 = max(0, -dy)
            ys1 = min(h_, h_ - dy)
            yd0 = max(0, dy)
            yd1 = min(h_, h_ + dy)
            xs0 = max(0, -dx)
            xs1 = min(w_, w_ - dx)
            xd0 = max(0, dx)
            xd1 = min(w_, w_ + dx)
            if ys1 <= ys0 or xs1 <= xs0:
                continue
            dil[yd0:yd1, xd0:xd1] |= occ[ys0:ys1, xs0:xs1]
        grid[dil] = 0

    return grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", default="office_localized")
    ap.add_argument("--isaac_config", default="configs/isaac_scenes_v34b.yaml")
    ap.add_argument("--stage", default="")
    ap.add_argument("--out_dir", default="runs/topo_mvp/v34b_nav2_demo/map")
    ap.add_argument("--resolution", type=float, default=0.05)
    ap.add_argument("--inflate_m", type=float, default=0.25)
    ap.add_argument("--allow_empty_map", action="store_true")
    ap.add_argument("--min_occ_ratio", type=float, default=0.03)
    args = ap.parse_args()

    cfg = _load_cfg(Path(args.isaac_config), args.scene_id)
    stage_path, stage_source = _resolve_stage(cfg, args.stage)
    stage_exists = int(bool(stage_path) and Path(stage_path).is_file())
    print(f"[V34B_ISAAC_STAGE] usd={stage_path} exists={stage_exists} source={stage_source}", flush=True)
    if stage_exists != 1:
        print("[V34B_MAP_GEN] ok=0 reason=stage_missing hint=set_ISAAC_STAGE_USD", flush=True)
        return 2

    from omni.isaac.kit import SimulationApp  # type: ignore

    sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    method = "rasterize"
    try:
        import omni.usd  # type: ignore
        from omni.isaac.core.utils.stage import open_stage  # type: ignore

        if not open_stage(stage_path):
            print("[V34B_MAP_GEN] ok=0 reason=open_stage_failed", flush=True)
            return 2
        for _ in range(30):
            sim_app.update()
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[V34B_MAP_GEN] ok=0 reason=stage_none", flush=True)
            return 2

        footprints = _collect_footprints(stage)

        bounds_raw = cfg.get("bounds", [-6.0, 6.0, -6.0, 6.0])
        if isinstance(bounds_raw, list) and len(bounds_raw) == 4:
            bounds = tuple(float(x) for x in bounds_raw)
        elif footprints:
            min_x = min(fp.min_x for fp in footprints) - 0.5
            max_x = max(fp.max_x for fp in footprints) + 0.5
            min_z = min(fp.min_z for fp in footprints) - 0.5
            max_z = max(fp.max_z for fp in footprints) + 0.5
            bounds = (min_x, max_x, min_z, max_z)
        else:
            bounds = (-6.0, 6.0, -6.0, 6.0)

        grid = _rasterize(
            footprints=footprints,
            bounds=bounds,
            resolution=float(args.resolution),
            inflate_m=float(args.inflate_m),
        )

        occ_ratio = float((grid == 0).mean())
        free_ratio = float((grid == 254).mean())
        unknown_ratio = float((grid == 205).mean())

        if (not args.allow_empty_map) and occ_ratio < float(args.min_occ_ratio):
            print(
                f"[V34B_MAP_GEN] ok=0 method={method} reason=map_too_empty occ_ratio={occ_ratio:.4f} min_occ_ratio={float(args.min_occ_ratio):.4f}",
                flush=True,
            )
            return 3

        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pgm_path = out / "office_map.pgm"
        yaml_path = out / "office_map.yaml"
        preview_path = out / "office_map_preview.png"

        Image.fromarray(grid).save(pgm_path)
        # Simple colorized preview for quick QA.
        preview = np.stack([grid, grid, grid], axis=-1)
        preview[grid == 0] = np.array([0, 0, 0], dtype=np.uint8)
        preview[grid == 254] = np.array([255, 255, 255], dtype=np.uint8)
        preview[grid == 205] = np.array([127, 127, 127], dtype=np.uint8)
        Image.fromarray(preview).save(preview_path)

        min_x, max_x, min_z, max_z = bounds
        yaml_text = (
            f"image: {pgm_path.name}\n"
            f"resolution: {float(args.resolution):.4f}\n"
            f"origin: [{min_x:.4f}, {min_z:.4f}, 0.0]\n"
            "negate: 0\n"
            "occupied_thresh: 0.65\n"
            "free_thresh: 0.196\n"
        )
        yaml_path.write_text(yaml_text, encoding="utf-8")

        print(
            f"[V34B_MAP_GEN] ok=1 method={method} res={float(args.resolution):.4f} origin=[{min_x:.4f},{min_z:.4f}] w={int(grid.shape[1])} h={int(grid.shape[0])} map_yaml={yaml_path} map_img={pgm_path}",
            flush=True,
        )
        print(
            f"[V34B_MAP_STATS] occ_ratio={occ_ratio:.4f} free_ratio={free_ratio:.4f} unknown_ratio={unknown_ratio:.4f}",
            flush=True,
        )
        return 0
    finally:
        try:
            sim_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
