#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


def _load_map_yaml(path: Path) -> Dict[str, float | str | list[float]]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(text)
    except Exception:
        obj = {}
    if not isinstance(obj, dict):
        obj = {}
    origin = obj.get("origin", [-6.0, -6.0, 0.0])
    if not isinstance(origin, list) or len(origin) < 2:
        origin = [-6.0, -6.0, 0.0]
    return {
        "image": str(obj.get("image", "office_map.pgm")),
        "resolution": float(obj.get("resolution", 0.05)),
        "origin": [float(origin[0]), float(origin[1]), float(origin[2] if len(origin) > 2 else 0.0)],
    }


def _cell_to_world(col: np.ndarray, row: np.ndarray, w: int, h: int, res: float, ox: float, oy: float) -> Tuple[np.ndarray, np.ndarray]:
    x = ox + (col.astype(np.float32) + 0.5) * float(res)
    y = oy + ((h - 1) - row.astype(np.float32) + 0.5) * float(res)
    return x, y


def _world_to_cell(x: float, y: float, w: int, h: int, res: float, ox: float, oy: float) -> Tuple[int, int]:
    col = int(math.floor((float(x) - ox) / float(res)))
    row_from_bottom = int(math.floor((float(y) - oy) / float(res)))
    row = int((h - 1) - row_from_bottom)
    return row, col


def _clean_free_mask(free: np.ndarray) -> np.ndarray:
    st3 = np.ones((3, 3), dtype=bool)
    st5 = np.ones((5, 5), dtype=bool)
    x = ndi.binary_opening(free, structure=st3)
    x = ndi.binary_closing(x, structure=st5)
    # Keep borders blocked to avoid map leakage.
    x[0, :] = False
    x[-1, :] = False
    x[:, 0] = False
    x[:, -1] = False
    return x


def _seeded_split(mask: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Split large connected free regions using distance-transform seeds + watershed_ift."""
    if int(mask.sum()) == 0:
        return np.zeros_like(mask, dtype=np.int32)

    lbl, n = ndi.label(mask)
    out = np.zeros_like(lbl, dtype=np.int32)
    rid = 0

    for comp_id in range(1, int(n) + 1):
        comp = lbl == comp_id
        comp_area = int(comp.sum())
        if comp_area == 0:
            continue
        rid += 1
        if comp_area < 2500:
            out[comp] = rid
            continue

        d = dist.copy()
        d[~comp] = 0.0
        peak_thresh = float(np.percentile(d[comp], 75))
        max_f = ndi.maximum_filter(d, size=21)
        peaks = comp & (d == max_f) & (d >= peak_thresh)
        peaks = ndi.binary_opening(peaks, structure=np.ones((3, 3), dtype=bool))
        markers, mcount = ndi.label(peaks)

        if int(mcount) < 2:
            # Fallback: farthest-point seeding (up to 3 seeds)
            ys, xs = np.where(comp)
            if len(xs) == 0:
                out[comp] = rid
                continue
            pts = np.stack([ys, xs], axis=1).astype(np.float32)
            seed_idx = [int(np.argmax(d[comp]))]
            seed_points = [pts[seed_idx[0]]]
            for _ in range(2):
                p = np.asarray(seed_points, dtype=np.float32)
                dist2 = np.min(((pts[:, None, :] - p[None, :, :]) ** 2).sum(axis=2), axis=1)
                seed_points.append(pts[int(np.argmax(dist2))])
            markers = np.zeros_like(lbl, dtype=np.int32)
            for si, (yy, xx) in enumerate(seed_points[:3], start=1):
                markers[int(yy), int(xx)] = int(si)
            markers, mcount = ndi.label(markers > 0)

        # Watershed on inverse distance (low values near medial axis)
        inv = np.max(d) - d
        inv_u8 = np.clip((inv / (float(inv.max()) + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
        try:
            ws = ndi.watershed_ift(inv_u8, markers.astype(np.int16))
            ws[~comp] = 0
        except Exception:
            ws = np.zeros_like(markers, dtype=np.int32)
            ys, xs = np.where(comp)
            seed_pts = np.stack(np.where(markers > 0), axis=1)
            if seed_pts.shape[0] == 0:
                out[comp] = rid
                continue
            all_pts = np.stack([ys, xs], axis=1)
            d2 = ((all_pts[:, None, :] - seed_pts[None, :, :]) ** 2).sum(axis=2)
            nearest = np.argmin(d2, axis=1)
            ws[ys, xs] = nearest + 1

        sub_ids = sorted(int(x) for x in np.unique(ws[comp]) if int(x) > 0)
        if len(sub_ids) <= 1:
            out[comp] = rid
            continue

        # Keep only sizable subregions; merge tiny ones to nearest large.
        sub_masks: List[Tuple[int, np.ndarray]] = []
        for sid in sub_ids:
            sm = ws == sid
            if int(sm.sum()) < 150:
                continue
            sub_masks.append((sid, sm))
        if len(sub_masks) <= 1:
            # Last-resort fallback for giant merged open spaces:
            # split by k-means on free-space coordinates to avoid single-room collapse.
            if comp_area >= 10000:
                ys, xs = np.where(comp)
                pts = np.stack([ys, xs], axis=1).astype(np.float32)
                k = int(np.clip(round(comp_area / 15000.0), 3, 6))
                if pts.shape[0] >= k:
                    # deterministic farthest-point init
                    seeds = [pts[int(np.argmax(d[comp]))]]
                    for _ in range(k - 1):
                        s = np.asarray(seeds, dtype=np.float32)
                        d2 = np.min(((pts[:, None, :] - s[None, :, :]) ** 2).sum(axis=2), axis=1)
                        seeds.append(pts[int(np.argmax(d2))])
                    centers = np.asarray(seeds, dtype=np.float32)
                    for _ in range(12):
                        d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                        asg = np.argmin(d2, axis=1)
                        new_centers = centers.copy()
                        for ki in range(k):
                            mki = asg == ki
                            if np.any(mki):
                                new_centers[ki] = np.mean(pts[mki], axis=0)
                        if np.allclose(new_centers, centers, atol=0.5):
                            centers = new_centers
                            break
                        centers = new_centers
                    d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                    asg = np.argmin(d2, axis=1)
                    ws = np.zeros_like(lbl, dtype=np.int32)
                    ws[ys, xs] = asg.astype(np.int32) + 1
                    sub_masks = []
                    for ki in range(1, k + 1):
                        mki = ws == ki
                        if int(mki.sum()) < 150:
                            continue
                        sub_masks.append((ki, mki))
            if len(sub_masks) <= 1:
                out[comp] = rid
                continue

        for _, sm in sub_masks:
            rid += 1
            out[sm] = rid

        # Assign leftovers to nearest assigned region in this component.
        leftovers = comp & (out == 0)
        if leftovers.any():
            dfill, idx = ndi.distance_transform_edt(out == 0, return_indices=True)
            rr, cc = np.where(leftovers)
            out[rr, cc] = out[idx[0, rr, cc], idx[1, rr, cc]]

    # Reindex room ids to 0..K-1
    ids = sorted(int(x) for x in np.unique(out) if int(x) > 0)
    remap = {old: i for i, old in enumerate(ids)}
    out2 = np.full_like(out, -1, dtype=np.int32)
    for old, new in remap.items():
        out2[out == old] = int(new)
    return out2


def _compute_rooms(room_cells: np.ndarray, dist: np.ndarray, res: float, ox: float, oy: float) -> Dict[str, object]:
    h, w = room_cells.shape
    rooms: List[Dict[str, object]] = []
    for rid in sorted(int(x) for x in np.unique(room_cells) if int(x) >= 0):
        m = room_cells == rid
        if int(m.sum()) == 0:
            continue
        rr, cc = np.where(m)
        xs, ys = _cell_to_world(cc, rr, w, h, res, ox, oy)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        rooms.append(
            {
                "room_id": int(rid),
                "cell_count": int(m.sum()),
                "centroid": {"x": cx, "y": cy},
                "bbox": {
                    "min": {"x": float(np.min(xs)), "y": float(np.min(ys))},
                    "max": {"x": float(np.max(xs)), "y": float(np.max(ys))},
                },
                "clearance_median": float(np.median(dist[m])) if m.any() else 0.0,
            }
        )
    return {
        "map": {
            "w": int(w),
            "h": int(h),
            "resolution": float(res),
            "origin": [float(ox), float(oy), 0.0],
            "planar_axes": "XY",
            "up_axis": "Z",
        },
        "rooms": rooms,
    }


def _build_doorways(room_cells: np.ndarray, dist: np.ndarray, res: float, ox: float, oy: float) -> Tuple[List[Dict[str, object]], List[Tuple[int, int]]]:
    h, w = room_cells.shape
    pair_points: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = defaultdict(list)

    for r in range(h - 1):
        a = room_cells[r, :]
        b = room_cells[r + 1, :]
        cols = np.where((a >= 0) & (b >= 0) & (a != b))[0]
        for c in cols.tolist():
            ra = int(a[c])
            rb = int(b[c])
            p = (min(ra, rb), max(ra, rb))
            cc = float(min(dist[r, c], dist[r + 1, c]))
            pair_points[p].append((r, c, cc))

    for r in range(h):
        a = room_cells[r, :-1]
        b = room_cells[r, 1:]
        cols = np.where((a >= 0) & (b >= 0) & (a != b))[0]
        for c in cols.tolist():
            ra = int(a[c])
            rb = int(b[c + 1])
            p = (min(ra, rb), max(ra, rb))
            cc = float(min(dist[r, c], dist[r, c + 1]))
            pair_points[p].append((r, c, cc))

    edges: List[Dict[str, object]] = []
    graph_edges: List[Tuple[int, int]] = []
    for (ra, rb), pts in sorted(pair_points.items()):
        if len(pts) < 3:
            continue
        clearances = np.asarray([p[2] for p in pts], dtype=np.float32)
        c_th = float(np.percentile(clearances, 35))
        narrow = [(r, c) for (r, c, cc) in pts if cc <= c_th]
        mask = np.zeros((h, w), dtype=bool)
        for r, c in narrow:
            mask[r, c] = True
        lbl, n = ndi.label(mask)
        doorway_centers: List[Dict[str, float]] = []
        for i in range(1, int(n) + 1):
            m = lbl == i
            if int(m.sum()) < 2:
                continue
            rr, cc = np.where(m)
            xc, yc = _cell_to_world(np.array([int(np.median(cc))]), np.array([int(np.median(rr))]), w, h, res, ox, oy)
            doorway_centers.append({"x": float(xc[0]), "y": float(yc[0])})

        if len(doorway_centers) == 0:
            rr = np.asarray([p[0] for p in pts], dtype=np.int32)
            cc = np.asarray([p[1] for p in pts], dtype=np.int32)
            xc, yc = _cell_to_world(np.array([int(np.median(cc))]), np.array([int(np.median(rr))]), w, h, res, ox, oy)
            doorway_centers.append({"x": float(xc[0]), "y": float(yc[0])})

        edges.append(
            {
                "room_a": int(ra),
                "room_b": int(rb),
                "pair": [int(ra), int(rb)],
                "count": int(len(pts)),
                "narrow_count": int(len(narrow)),
                "clearance_p35_m": float(c_th),
                "passable": int(len(doorway_centers) > 0),
                "doorway_poses": doorway_centers,
            }
        )
        graph_edges.append((int(ra), int(rb)))

    return edges, graph_edges


def _connected_components(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    adj: Dict[int, List[int]] = {int(n): [] for n in nodes}
    for a, b in edges:
        if int(a) == int(b):
            continue
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    seen = set()
    out: List[List[int]] = []
    for n in sorted(adj.keys()):
        if n in seen:
            continue
        q = deque([n])
        seen.add(n)
        comp: List[int] = []
        while q:
            cur = q.popleft()
            comp.append(int(cur))
            for nb in adj.get(cur, []):
                if nb in seen:
                    continue
                seen.add(nb)
                q.append(nb)
        out.append(sorted(comp))
    return out


def _build_topo(room_part: Dict[str, object], doorway_edges: List[Dict[str, object]]) -> Dict[str, object]:
    rooms = room_part.get("rooms", [])
    room_by_id = {int(r["room_id"]): r for r in rooms if isinstance(r, dict)}

    nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []

    nid = 0
    room_center_node: Dict[int, int] = {}
    for rid in sorted(room_by_id.keys()):
        c = room_by_id[rid]["centroid"]
        nodes.append(
            {
                "node_id": int(nid),
                "room_id": int(rid),
                "type": "room_center",
                "x": float(c["x"]),
                "y": float(c["y"]),
                "yaw_deg": 0.0,
            }
        )
        room_center_node[int(rid)] = int(nid)
        nid += 1

    for de in doorway_edges:
        ra = int(de["room_a"])
        rb = int(de["room_b"])
        poses = de.get("doorway_poses", [])
        if not isinstance(poses, list) or len(poses) == 0:
            continue
        p = poses[0]
        px = float(p["x"])
        py = float(p["y"])

        ca = room_by_id.get(ra, {}).get("centroid", {"x": px, "y": py})
        cb = room_by_id.get(rb, {}).get("centroid", {"x": px, "y": py})
        ax, ay = float(ca.get("x", px)), float(ca.get("y", py))
        bx, by = float(cb.get("x", px)), float(cb.get("y", py))

        va = np.asarray([px - ax, py - ay], dtype=np.float32)
        vb = np.asarray([px - bx, py - by], dtype=np.float32)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na > 1e-6:
            va /= na
        if nb > 1e-6:
            vb /= nb
        app_a = (px - float(va[0]) * 0.35, py - float(va[1]) * 0.35)
        app_b = (px - float(vb[0]) * 0.35, py - float(vb[1]) * 0.35)

        n_a = nid
        nodes.append({"node_id": int(n_a), "room_id": int(ra), "type": "doorway_approach", "x": float(app_a[0]), "y": float(app_a[1]), "yaw_deg": float(math.degrees(math.atan2(py - app_a[1], px - app_a[0])))})
        nid += 1

        n_b = nid
        nodes.append({"node_id": int(n_b), "room_id": int(rb), "type": "doorway_approach", "x": float(app_b[0]), "y": float(app_b[1]), "yaw_deg": float(math.degrees(math.atan2(py - app_b[1], px - app_b[0])))})
        nid += 1

        n_d = nid
        nodes.append({"node_id": int(n_d), "room_id": -1, "type": "doorway", "x": float(px), "y": float(py), "yaw_deg": 0.0, "room_pair": [int(ra), int(rb)]})
        nid += 1

        def add_edge(u: int, v: int, etype: str) -> None:
            cost = float(math.hypot(float(nodes[u]["x"]) - float(nodes[v]["x"]), float(nodes[u]["y"]) - float(nodes[v]["y"])))
            edges.append({"u": int(u), "v": int(v), "cost": cost, "type": etype})

        add_edge(room_center_node.get(ra, n_a), n_a, "room_to_door")
        add_edge(room_center_node.get(rb, n_b), n_b, "room_to_door")
        add_edge(n_a, n_d, "door_approach")
        add_edge(n_b, n_d, "door_approach")
        add_edge(n_a, n_b, "cross_room")

    return {"nodes": nodes, "edges": edges}


def _nearest_topo_node(x: float, y: float, nodes: Sequence[Dict[str, object]], room_id: int) -> int:
    best = -1
    best_d = float("inf")
    # Prefer same-room node if available.
    for n in nodes:
        nr = int(n.get("room_id", -1))
        if room_id >= 0 and nr not in (room_id, -1):
            continue
        d = float(math.hypot(float(n["x"]) - x, float(n["y"]) - y))
        if d < best_d:
            best_d = d
            best = int(n["node_id"])
    if best >= 0:
        return int(best)
    for n in nodes:
        d = float(math.hypot(float(n["x"]) - x, float(n["y"]) - y))
        if d < best_d:
            best_d = d
            best = int(n["node_id"])
    return int(best)


def _compute_metrics(seq_rows: Sequence[Dict[str, object]], doorway_edges: Sequence[Dict[str, object]]) -> Dict[str, float | int]:
    steps = len(seq_rows)
    if steps <= 1:
        return {
            "room_consistency": 0.0,
            "topo_consistency": 0.0,
            "doorway_plausibility": 0.0,
            "unknown_rate": 1.0,
            "room_changes": 0,
            "topo_changes": 0,
        }

    room_changes = 0
    topo_changes = 0
    unknown = 0
    seen_pairs = set()
    for i in range(1, steps):
        a = int(seq_rows[i - 1]["room_id"])
        b = int(seq_rows[i]["room_id"])
        if a != b:
            room_changes += 1
            if a >= 0 and b >= 0:
                seen_pairs.add(tuple(sorted((a, b))))
        if int(seq_rows[i - 1]["topo_node_id"]) != int(seq_rows[i]["topo_node_id"]):
            topo_changes += 1

    for r in seq_rows:
        if int(r["room_id"]) < 0 or int(r["topo_node_id"]) < 0:
            unknown += 1

    denom = max(1.0, steps / 10.0)
    room_consistency = max(0.0, 1.0 - room_changes / denom)
    topo_consistency = max(0.0, 1.0 - topo_changes / denom)

    door_pairs = {tuple(sorted((int(e["room_a"]), int(e["room_b"])))) for e in doorway_edges}
    matched = len(seen_pairs & door_pairs)
    doorway_plausibility = float(matched / max(1, len(door_pairs)))
    unknown_rate = float(unknown / max(1, steps))

    return {
        "room_consistency": float(room_consistency),
        "topo_consistency": float(topo_consistency),
        "doorway_plausibility": float(doorway_plausibility),
        "unknown_rate": float(unknown_rate),
        "room_changes": int(room_changes),
        "topo_changes": int(topo_changes),
    }


def _overlay_rooms(img_u8: np.ndarray, room_cells: np.ndarray, out_png: Path) -> None:
    h, w = room_cells.shape
    base = np.repeat(img_u8[:, :, None], 3, axis=2).astype(np.uint8)
    ids = sorted(int(x) for x in np.unique(room_cells) if int(x) >= 0)
    palette = np.array(
        [
            [230, 25, 75],
            [60, 180, 75],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [170, 110, 40],
            [128, 128, 0],
        ],
        dtype=np.uint8,
    )
    for i, rid in enumerate(ids):
        m = room_cells == rid
        col = palette[i % len(palette)]
        base[m] = (0.45 * base[m] + 0.55 * col).astype(np.uint8)

    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)
    for rid in ids:
        rr, cc = np.where(room_cells == rid)
        if len(rr) == 0:
            continue
        cx = int(np.mean(cc))
        cy = int(np.mean(rr))
        draw.text((cx, cy), f"R{rid}", fill=(255, 255, 255))
    img.save(out_png)


def _overlay_doors(img_u8: np.ndarray, doorway_edges: Sequence[Dict[str, object]], res: float, ox: float, oy: float, out_png: Path) -> None:
    h, w = img_u8.shape
    base = np.repeat(img_u8[:, :, None], 3, axis=2).astype(np.uint8)
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)
    for e in doorway_edges:
        for p in e.get("doorway_poses", []):
            x = float(p["x"])
            y = float(p["y"])
            row, col = _world_to_cell(x, y, w, h, res, ox, oy)
            if row < 0 or row >= h or col < 0 or col >= w:
                continue
            draw.ellipse((col - 3, row - 3, col + 3, row + 3), outline=(255, 0, 0), width=2)
            draw.text((col + 5, row - 5), f"{int(e['room_a'])}-{int(e['room_b'])}", fill=(255, 220, 0))
    img.save(out_png)


def _overlay_topo(img_u8: np.ndarray, topo: Dict[str, object], res: float, ox: float, oy: float, out_png: Path) -> None:
    h, w = img_u8.shape
    base = np.repeat(img_u8[:, :, None], 3, axis=2).astype(np.uint8)
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)

    nodes = topo.get("nodes", [])
    edges = topo.get("edges", [])
    node_map = {int(n["node_id"]): n for n in nodes if isinstance(n, dict)}

    for e in edges:
        u = node_map.get(int(e["u"]))
        v = node_map.get(int(e["v"]))
        if u is None or v is None:
            continue
        ru, cu = _world_to_cell(float(u["x"]), float(u["y"]), w, h, res, ox, oy)
        rv, cv = _world_to_cell(float(v["x"]), float(v["y"]), w, h, res, ox, oy)
        draw.line((cu, ru, cv, rv), fill=(0, 255, 255), width=1)

    for n in nodes:
        r, c = _world_to_cell(float(n["x"]), float(n["y"]), w, h, res, ox, oy)
        if r < 0 or r >= h or c < 0 or c >= w:
            continue
        kind = str(n.get("type", ""))
        col = (255, 128, 0) if kind == "room_center" else (255, 0, 255)
        draw.ellipse((c - 2, r - 2, c + 2, r + 2), fill=col)
    img.save(out_png)


def _load_pose_trace(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for i, r in enumerate(rd):
            try:
                rows.append(
                    {
                        "step": float(r.get("step", i)),
                        "x": float(r.get("gt_x", r.get("x", 0.0))),
                        "y": float(r.get("gt_y", r.get("y", 0.0))),
                        "yaw": float(r.get("gt_yaw_rad", r.get("yaw_rad", 0.0))),
                    }
                )
            except Exception:
                continue
    return rows


def run(args: argparse.Namespace) -> int:
    map_yaml = Path(args.map_yaml).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_map_yaml(map_yaml)
    pgm = (map_yaml.parent / str(meta["image"])).resolve()
    img = np.asarray(Image.open(pgm).convert("L"), dtype=np.uint8)
    h, w = img.shape
    res = float(meta["resolution"])
    ox = float(meta["origin"][0])
    oy = float(meta["origin"][1])

    free = img >= 245
    free_clean = _clean_free_mask(free)
    dist = ndi.distance_transform_edt(free_clean) * float(res)

    room_cells = _seeded_split(free_clean, dist)
    room_part = _compute_rooms(room_cells, dist, res, ox, oy)

    doorway_edges, graph_edges = _build_doorways(room_cells, dist, res, ox, oy)
    room_ids = [int(r["room_id"]) for r in room_part.get("rooms", [])]
    comps = _connected_components(room_ids, graph_edges)
    largest = max((len(c) for c in comps), default=0)

    room_graph = {
        "rooms": room_part.get("rooms", []),
        "edges": doorway_edges,
        "components": comps,
    }

    topo = _build_topo(room_part, doorway_edges)

    # Trajectory-room/topo assignment
    traj_rows = _load_pose_trace(Path(args.pose_trace)) if str(args.pose_trace).strip() else []
    seq_rows: List[Dict[str, object]] = []
    nodes = topo.get("nodes", [])
    for r in traj_rows:
        row, col = _world_to_cell(float(r["x"]), float(r["y"]), w, h, res, ox, oy)
        rid = -1
        if 0 <= row < h and 0 <= col < w:
            rid = int(room_cells[row, col])
        tid = _nearest_topo_node(float(r["x"]), float(r["y"]), nodes, rid) if len(nodes) > 0 else -1
        seq_rows.append(
            {
                "step": int(r["step"]),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "yaw_deg": float(math.degrees(float(r["yaw"]))),
                "room_id": int(rid),
                "topo_node_id": int(tid),
                "loc_conf": 1.0,
            }
        )

    traj_csv = out_dir / "traj_room_sequence.csv"
    with traj_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["step", "x", "y", "yaw_deg", "room_id", "topo_node_id", "loc_conf"])
        wr.writeheader()
        for r in seq_rows:
            wr.writerow(r)

    metrics = _compute_metrics(seq_rows, doorway_edges)
    metrics["components"] = int(len(comps))

    (out_dir / "room_partition.json").write_text(
        json.dumps(
            {
                **room_part,
                "room_cell_labels": room_cells.tolist(),
                "method": "occ_cc+morph+dt_seed_split",
                "free_cell_count": int(free_clean.sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "doorway_edges.json").write_text(json.dumps({"edges": doorway_edges}, indent=2), encoding="utf-8")
    (out_dir / "room_graph.json").write_text(json.dumps(room_graph, indent=2), encoding="utf-8")
    (out_dir / "topo_nodes.json").write_text(json.dumps(topo, indent=2), encoding="utf-8")
    (out_dir / "qa_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _overlay_rooms(img, room_cells, out_dir / "overlay_map_rooms.png")
    _overlay_doors(img, doorway_edges, res, ox, oy, out_dir / "overlay_map_doors.png")
    _overlay_topo(img, topo, res, ox, oy, out_dir / "overlay_map_topo.png")

    print(
        f"[V35A_ROOM_PARTITION] ok=1 rooms={len(room_part.get('rooms', []))} method=occ_cc+morph+dt_seed_split",
        flush=True,
    )
    passable = sum(1 for e in doorway_edges if int(e.get("passable", 0)) == 1)
    print(f"[V35A_DOORWAY] ok=1 edges={len(doorway_edges)} passable={passable}", flush=True)
    print(
        f"[V35A_ROOM_GRAPH] ok=1 components={len(comps)} largest_component={largest}",
        flush=True,
    )
    print(f"[V35A_TOPO_NODES] ok=1 nodes={len(topo.get('nodes', []))} edges={len(topo.get('edges', []))}", flush=True)
    print(
        f"[V35A_QA] room_consistency={float(metrics['room_consistency']):.3f} "
        f"topo_consistency={float(metrics['topo_consistency']):.3f} "
        f"doorway_plausibility={float(metrics['doorway_plausibility']):.3f} "
        f"unknown_rate={float(metrics['unknown_rate']):.3f}",
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="v35a room/topo QA from occupancy map")
    ap.add_argument("--map_yaml", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pose_trace", default="")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
