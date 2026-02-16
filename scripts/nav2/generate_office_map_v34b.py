#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", default="office_localized")
    ap.add_argument("--isaac_config", default="configs/isaac_scenes_v34b.yaml")
    ap.add_argument("--out_dir", default="runs/topo_mvp/v34b_nav2_demo/map")
    ap.add_argument("--resolution", type=float, default=0.05)
    args = ap.parse_args()

    cfg = _load_cfg(Path(args.isaac_config), args.scene_id)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bounds = cfg.get("bounds", [-6.0, 6.0, -6.0, 6.0])
    if not isinstance(bounds, list) or len(bounds) != 4:
        bounds = [-6.0, 6.0, -6.0, 6.0]
    min_x, max_x, min_z, max_z = [float(x) for x in bounds]
    res = float(args.resolution)
    w = max(10, int(np.ceil((max_x - min_x) / res)))
    h = max(10, int(np.ceil((max_z - min_z) / res)))

    occ = np.full((h, w), 254, dtype=np.uint8)
    obstacles = cfg.get("obstacles", []) if isinstance(cfg, dict) else []
    for ob in obstacles:
        if not isinstance(ob, dict):
            continue
        ox = float(ob.get("x", 0.0))
        oz = float(ob.get("z", 0.0))
        rr = float(ob.get("r", ob.get("radius", 0.6)))
        cx = int((ox - min_x) / res)
        cz = int((oz - min_z) / res)
        rad = max(1, int(np.ceil(rr / res)))
        y0, y1 = max(0, cz - rad), min(h, cz + rad + 1)
        x0, x1 = max(0, cx - rad), min(w, cx + rad + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cz) ** 2 <= rad * rad
        occ[y0:y1, x0:x1][mask] = 0

    # Border walls as occupied.
    occ[0, :] = 0
    occ[-1, :] = 0
    occ[:, 0] = 0
    occ[:, -1] = 0

    pgm_path = out / "office_map.pgm"
    Image.fromarray(occ).save(pgm_path)
    yaml_path = out / "office_map.yaml"
    yaml_text = (
        f"image: {pgm_path.name}\n"
        f"resolution: {res:.4f}\n"
        f"origin: [{min_x:.4f}, {min_z:.4f}, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(
        f"[V34B_MAP_GEN] ok=1 map_yaml={yaml_path} map_img={pgm_path} resolution={res:.4f} origin=[{min_x:.4f},{min_z:.4f}]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
