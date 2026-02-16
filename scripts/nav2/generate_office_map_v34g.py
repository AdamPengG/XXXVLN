#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

OFFICE_DEFAULT = "/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"


@dataclass
class Box2D:
    path: str
    p0: float
    p1: float
    q0: float
    q1: float
    up0: float
    up1: float


def _resolve_stage(stage_arg: str) -> Tuple[str, str]:
    stage = str(stage_arg or os.environ.get("ISAAC_STAGE_USD", "")).strip()
    if stage:
        return stage, "override"
    if Path(OFFICE_DEFAULT).is_file():
        return OFFICE_DEFAULT, "official_assets"
    return "", "missing"


def _prim_has_collision(prim, UsdPhysics, PhysxSchema) -> bool:
    try:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            attr = prim.GetAttribute("physics:collisionEnabled")
            if attr.IsValid() and attr.HasAuthoredValue() and attr.Get() is False:
                return False
            return True
    except Exception:
        pass
    try:
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            return True
    except Exception:
        pass
    try:
        if hasattr(PhysxSchema, "PhysxCollisionAPI") and prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            return True
    except Exception:
        pass
    return False


def _collect_boxes(stage, up_axis: str, only_colliders: bool = True) -> List[Box2D]:
    from pxr import PhysxSchema, UsdGeom, UsdPhysics  # type: ignore

    up = str(up_axis or "").upper()
    if up.startswith("Z"):
        planar_idx = (0, 1)
        up_idx = 2
    else:
        planar_idx = (0, 2)
        up_idx = 1

    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_], useExtentsHint=True)
    boxes: List[Box2D] = []
    skip_words = (
        "sky",
        "dome",
        "light",
        "camera",
        "sensor",
        "reflection",
        "postprocess",
    )
    floor_words = ("ground", "floor", "ceiling")
    wall_words = ("wall", "door", "window", "pillar", "column", "stair", "railing", "frame")

    for prim in stage.TraverseAll():
        if not prim.IsActive() or prim.IsAbstract() or not prim.IsLoaded():
            continue
        try:
            is_boundable = prim.IsA(UsdGeom.Boundable)
        except Exception:
            is_boundable = False
        if not is_boundable:
            continue
        pth = prim.GetPath().pathString
        lower = pth.lower()
        if any(w in lower for w in skip_words):
            continue
        if only_colliders and (not _prim_has_collision(prim, UsdPhysics, PhysxSchema)):
            continue

        try:
            world = bbox_cache.ComputeWorldBound(prim)
            aabb = world.ComputeAlignedBox()
            mn = aabb.GetMin()
            mx = aabb.GetMax()
        except Exception:
            continue

        arr_min = np.array([float(mn[0]), float(mn[1]), float(mn[2])], dtype=np.float32)
        arr_max = np.array([float(mx[0]), float(mx[1]), float(mx[2])], dtype=np.float32)
        if not np.isfinite(arr_min).all() or not np.isfinite(arr_max).all():
            continue

        p0 = float(arr_min[planar_idx[0]])
        p1 = float(arr_max[planar_idx[0]])
        q0 = float(arr_min[planar_idx[1]])
        q1 = float(arr_max[planar_idx[1]])
        u0 = float(arr_min[up_idx])
        u1 = float(arr_max[up_idx])

        if (p1 - p0) <= 1e-3 or (q1 - q0) <= 1e-3:
            continue
        if (u1 - u0) <= 1e-3:
            continue
        dx = float(p1 - p0)
        dy = float(q1 - q0)
        area = float(dx * dy)
        if area <= 1e-4:
            continue
        # Ignore overhead-only prims and paper-thin ground slabs.
        if u0 > 3.0:
            continue
        # Floor/ground slabs are navigable free space and should not become
        # occupied in 2D nav maps.
        if any(w in lower for w in floor_words):
            continue
        # Drop giant scene-root envelopes.
        if (p1 - p0) >= 150.0 or (q1 - q0) >= 150.0:
            continue

        boxes.append(Box2D(path=pth, p0=p0, p1=p1, q0=q0, q1=q1, up0=u0, up1=u1))

    return boxes


def _compute_bounds(boxes: Iterable[Box2D], pad: float) -> Optional[Tuple[float, float, float, float]]:
    boxes = list(boxes)
    if not boxes:
        return None
    p0 = min(b.p0 for b in boxes) - float(pad)
    p1 = max(b.p1 for b in boxes) + float(pad)
    q0 = min(b.q0 for b in boxes) - float(pad)
    q1 = max(b.q1 for b in boxes) + float(pad)
    if not np.isfinite([p0, p1, q0, q1]).all():
        return None
    if (p1 - p0) < 1.0 or (q1 - q0) < 1.0:
        return None
    return float(p0), float(p1), float(q0), float(q1)


def _parse_bounds(bounds_s: str) -> Optional[Tuple[float, float, float, float]]:
    if not str(bounds_s or "").strip():
        return None
    raw = [x.strip() for x in str(bounds_s).split(",") if x.strip()]
    if len(raw) != 4:
        return None
    try:
        vals = [float(x) for x in raw]
    except Exception:
        return None
    p0, p1, q0, q1 = vals
    if (p1 - p0) < 1.0 or (q1 - q0) < 1.0:
        return None
    return float(p0), float(p1), float(q0), float(q1)


def _intersects_bounds(b: Box2D, bounds: Tuple[float, float, float, float]) -> bool:
    p0, p1, q0, q1 = bounds
    if b.p1 < p0 or b.p0 > p1:
        return False
    if b.q1 < q0 or b.q0 > q1:
        return False
    return True


def _prune_outlier_boxes(boxes: List[Box2D]) -> List[Box2D]:
    if not boxes:
        return []
    areas = np.asarray([(b.p1 - b.p0) * (b.q1 - b.q0) for b in boxes], dtype=np.float32)
    # Scene-adaptive pruning: drop top area outliers that usually correspond to
    # non-convex shells whose AABB floods the map.
    q95 = float(np.percentile(areas, 95))
    q99 = float(np.percentile(areas, 99))
    limit = max(8.0, min(q99, q95 * 2.5))
    keep: List[Box2D] = []
    for b, a in zip(boxes, areas.tolist()):
        if float(a) > limit:
            continue
        keep.append(b)
    return keep if keep else boxes


def _project_box_cells(b: Box2D, p0: float, q0: float, res: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = max(0, int(math.floor((b.p0 - p0) / float(res))))
    x1 = min(w, int(math.ceil((b.p1 - p0) / float(res))))
    y0 = max(0, int(math.floor((b.q0 - q0) / float(res))))
    y1 = min(h, int(math.ceil((b.q1 - q0) / float(res))))
    return x0, x1, y0, y1


def _rasterize_aabb(boxes: Iterable[Box2D], bounds: Tuple[float, float, float, float], res: float, inflate_m: float) -> np.ndarray:
    p0, p1, q0, q1 = [float(v) for v in bounds]
    w = max(32, int(math.ceil((p1 - p0) / float(res))))
    h = max(32, int(math.ceil((q1 - q0) / float(res))))
    grid = np.full((h, w), 254, dtype=np.uint8)
    wall_words = ("wall", "door", "window", "pillar", "column", "stair", "railing", "frame", "partition")

    for b in boxes:
        x0, x1, y0, y1 = _project_box_cells(b, p0, q0, res, w, h)
        if x1 <= x0 or y1 <= y0:
            continue
        dx = max(1, x1 - x0)
        dy = max(1, y1 - y0)
        # For large wall-like boxes, draw only a thin border to avoid filling
        # interior free space caused by non-convex mesh AABBs.
        is_wall_like = any(t in b.path.lower() for t in wall_words)
        large_box = (dx * dy) >= int(max(64, 3.0 / max(res, 1e-3)))
        if is_wall_like and large_box:
            t = max(1, int(round(0.12 / max(res, 1e-3))))
            grid[y0:min(h, y0 + t), x0:x1] = 0
            grid[max(0, y1 - t):y1, x0:x1] = 0
            grid[y0:y1, x0:min(w, x0 + t)] = 0
            grid[y0:y1, max(0, x1 - t):x1] = 0
        else:
            grid[y0:y1, x0:x1] = 0

    # Keep outer border occupied.
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0

    inflate_cells = max(0, int(round(float(inflate_m) / float(res))))
    if inflate_cells > 0:
        occ = grid == 0
        dil = occ.copy()
        h_, w_ = occ.shape
        for dy in range(-inflate_cells, inflate_cells + 1):
            for dx in range(-inflate_cells, inflate_cells + 1):
                if dx * dx + dy * dy > inflate_cells * inflate_cells:
                    continue
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


def _save_map(out_dir: Path, grid: np.ndarray, bounds: Tuple[float, float, float, float], res: float) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pgm = out_dir / "office_map.pgm"
    yaml = out_dir / "office_map.yaml"
    preview = out_dir / "office_map_preview.png"

    Image.fromarray(grid).save(pgm)
    preview_img = np.stack([grid, grid, grid], axis=-1)
    preview_img[grid == 0] = np.array([0, 0, 0], dtype=np.uint8)
    preview_img[grid == 254] = np.array([255, 255, 255], dtype=np.uint8)
    preview_img[grid == 205] = np.array([127, 127, 127], dtype=np.uint8)
    Image.fromarray(preview_img).save(preview)

    p0, _p1, q0, _q1 = bounds
    yaml.write_text(
        "\n".join(
            [
                f"image: {pgm.name}",
                f"resolution: {float(res):.4f}",
                f"origin: [{float(p0):.4f}, {float(q0):.4f}, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return pgm, yaml, preview


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="")
    ap.add_argument("--out_dir", default="runs/topo_mvp/v34g_nav2_office_mapfix/map")
    ap.add_argument("--resolution", type=float, default=float(os.environ.get("V34G_MAP_RES", "0.05")))
    ap.add_argument("--inflate_m", type=float, default=float(os.environ.get("V34G_MAP_INFLATE_M", "0.20")))
    ap.add_argument("--padding_m", type=float, default=float(os.environ.get("V34G_MAP_PAD_M", "0.50")))
    ap.add_argument("--bounds", default=str(os.environ.get("V34G_MAP_BOUNDS", "")))
    ap.add_argument("--min_occ_ratio", type=float, default=float(os.environ.get("V34G_MAP_OCC_MIN", "0.02")))
    ap.add_argument("--max_occ_ratio", type=float, default=float(os.environ.get("V34G_MAP_OCC_MAX", "0.85")))
    ap.add_argument("--allow_empty_map", action="store_true")
    args = ap.parse_args()

    stage_path, source = _resolve_stage(args.stage)
    stage_exists = int(bool(stage_path) and Path(stage_path).is_file())
    print(f"[V34B_ISAAC_STAGE] usd={stage_path} exists={stage_exists} source={source}", flush=True)
    if stage_exists != 1:
        print("[V34G_MAP_GEN] ok=0 method=aabb_rasterize reason=stage_missing", flush=True)
        return 2

    from omni.isaac.kit import SimulationApp  # type: ignore

    sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    try:
        import omni.usd  # type: ignore
        from omni.isaac.core.utils.stage import open_stage  # type: ignore
        from pxr import UsdGeom  # type: ignore

        if not open_stage(stage_path):
            print("[V34G_MAP_GEN] ok=0 method=aabb_rasterize reason=open_stage_failed", flush=True)
            return 2
        for _ in range(30):
            sim_app.update()
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[V34G_MAP_GEN] ok=0 method=aabb_rasterize reason=stage_none", flush=True)
            return 2

        up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()
        planar_name = "XY" if up_axis.startswith("Z") else "XZ"

        cfg_bounds = _parse_bounds(args.bounds)
        coll_boxes = _collect_boxes(stage, up_axis=up_axis, only_colliders=True)
        all_boxes = coll_boxes
        if len(coll_boxes) < 80:
            all_boxes = _collect_boxes(stage, up_axis=up_axis, only_colliders=False)
        if cfg_bounds is not None:
            in_bounds = [b for b in all_boxes if _intersects_bounds(b, cfg_bounds)]
            if len(in_bounds) >= 2:
                all_boxes = in_bounds
        all_boxes = _prune_outlier_boxes(all_boxes)

        bounds = cfg_bounds if cfg_bounds is not None else _compute_bounds(all_boxes, pad=float(args.padding_m))
        if bounds is None:
            print("[V34G_MAP_GEN] ok=0 method=aabb_rasterize reason=empty_bounds", flush=True)
            return 3

        grid = _rasterize_aabb(all_boxes, bounds=bounds, res=float(args.resolution), inflate_m=float(args.inflate_m))

        occ_ratio = float((grid == 0).mean())
        free_ratio = float((grid == 254).mean())
        unknown_ratio = float((grid == 205).mean())

        out_dir = Path(args.out_dir)
        pgm, yaml, preview = _save_map(out_dir, grid, bounds=bounds, res=float(args.resolution))

        preview_arr = np.asarray(Image.open(preview).convert("RGB"), dtype=np.uint8)
        preview_mean = float(preview_arr.mean()) if preview_arr.size > 0 else 255.0
        not_empty = int(occ_ratio >= float(args.min_occ_ratio))
        occ_min_ok = int(occ_ratio >= float(args.min_occ_ratio))
        occ_max_ok = int(occ_ratio <= float(args.max_occ_ratio))
        preview_ok = int(preview_mean < 250.0)
        sanity_ok = int(not_empty and occ_min_ok and occ_max_ok and preview_ok)

        report = {
            "ok": int(sanity_ok),
            "method": "aabb_rasterize",
            "stage": stage_path,
            "up_axis": up_axis,
            "planar": planar_name,
            "resolution": float(args.resolution),
            "inflate_m": float(args.inflate_m),
            "padding_m": float(args.padding_m),
            "bounds_source": "config" if cfg_bounds is not None else "auto",
            "bounds": {
                "p0": float(bounds[0]),
                "p1": float(bounds[1]),
                "q0": float(bounds[2]),
                "q1": float(bounds[3]),
            },
            "grid": {"w": int(grid.shape[1]), "h": int(grid.shape[0])},
            "stats": {
                "occ_ratio": float(occ_ratio),
                "free_ratio": float(free_ratio),
                "unknown_ratio": float(unknown_ratio),
                "preview_mean": float(preview_mean),
            },
            "boxes": {
                "collider_boxes": int(len(coll_boxes)),
                "used_boxes": int(len(all_boxes)),
            },
            "map": {
                "pgm": str(pgm),
                "yaml": str(yaml),
                "preview": str(preview),
            },
            "sanity": {
                "not_empty": int(not_empty),
                "occ_ratio_min_ok": int(occ_min_ok),
                "occ_ratio_max_ok": int(occ_max_ok),
                "preview_not_white": int(preview_ok),
            },
        }
        (out_dir / "map_report_v34g.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(
            f"[V34G_MAP_GEN] ok=1 method=aabb_rasterize res={float(args.resolution):.4f} "
            f"w={int(grid.shape[1])} h={int(grid.shape[0])}",
            flush=True,
        )
        print(
            f"[V34G_MAP_STATS] occ_ratio={occ_ratio:.4f} free_ratio={free_ratio:.4f} unknown_ratio={unknown_ratio:.4f}",
            flush=True,
        )
        print(
            f"[V34G_MAP_SANITY] ok={int(sanity_ok)} not_empty={int(not_empty)} "
            f"occ_ratio_min_ok={int(occ_min_ok)} occ_ratio_max_ok={int(occ_max_ok)}",
            flush=True,
        )

        if (not args.allow_empty_map) and (sanity_ok != 1):
            return 4
        return 0
    finally:
        try:
            sim_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
