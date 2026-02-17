#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _world_to_cell(x: float, y: float, w: int, h: int, res: float, ox: float, oy: float) -> Tuple[int, int]:
    col = int(math.floor((float(x) - ox) / float(res)))
    row_from_bottom = int(math.floor((float(y) - oy) / float(res)))
    row = int((h - 1) - row_from_bottom)
    return row, col


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_pose_trace(path: Path) -> Optional[Tuple[float, float]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                x = row.get("gt_x", row.get("x"))
                y = row.get("gt_y", row.get("y"))
                if x is None or y is None:
                    continue
                return (float(x), float(y))
    except Exception:
        return None
    return None


def _room_id_from_xy(room_part: Dict[str, object], x: float, y: float) -> int:
    labels = np.asarray(room_part.get("room_cell_labels", []), dtype=np.int32)
    if labels.ndim != 2:
        return -1
    m = room_part.get("map", {})
    w = int(m.get("w", labels.shape[1]))
    h = int(m.get("h", labels.shape[0]))
    res = float(m.get("resolution", 0.05))
    origin = m.get("origin", [0.0, 0.0, 0.0])
    ox = float(origin[0] if isinstance(origin, list) and len(origin) > 0 else 0.0)
    oy = float(origin[1] if isinstance(origin, list) and len(origin) > 1 else 0.0)
    row, col = _world_to_cell(x, y, w, h, res, ox, oy)
    if 0 <= row < h and 0 <= col < w:
        return int(labels[row, col])
    return -1


def _room_centroids(room_part: Dict[str, object]) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    for r in room_part.get("rooms", []):
        if not isinstance(r, dict):
            continue
        rid = int(r.get("room_id", -1))
        c = r.get("centroid", {})
        if not isinstance(c, dict):
            continue
        out[rid] = (float(c.get("x", 0.0)), float(c.get("y", 0.0)))
    return out


def _resolve_room_id(room: str, room_part: Dict[str, object], default_room: int) -> int:
    s = str(room or "").strip().lower()
    if not s:
        return int(default_room)
    try:
        rid = int(s)
        return rid
    except Exception:
        pass
    if s.startswith("room_"):
        tail = s.split("room_", 1)[1]
        try:
            return int(tail)
        except Exception:
            pass
    # fallback: exact alias in room label map not available yet.
    return int(default_room)


def _build_room_graph(room_graph: Dict[str, object]) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], Dict[str, object]]]:
    adj: Dict[int, List[int]] = {}
    edge_map: Dict[Tuple[int, int], Dict[str, object]] = {}
    for e in room_graph.get("edges", []):
        if not isinstance(e, dict):
            continue
        a = int(e.get("room_a", -1))
        b = int(e.get("room_b", -1))
        if a < 0 or b < 0:
            continue
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
        edge_map[(min(a, b), max(a, b))] = e
    for k in adj:
        adj[k] = sorted(set(adj[k]))
    return adj, edge_map


def _bfs_path(adj: Dict[int, List[int]], src: int, dst: int) -> List[int]:
    if src < 0 or dst < 0:
        return []
    if src == dst:
        return [src]
    q = deque([src])
    prev: Dict[int, int] = {src: src}
    while q:
        cur = q.popleft()
        for nb in adj.get(cur, []):
            if nb in prev:
                continue
            prev[nb] = cur
            if nb == dst:
                q.clear()
                break
            q.append(nb)
    if dst not in prev:
        return []
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def _door_pose(edge: Dict[str, object]) -> Optional[Tuple[float, float]]:
    poses = edge.get("doorway_poses", [])
    if isinstance(poses, list) and len(poses) > 0 and isinstance(poses[0], dict):
        return (float(poses[0].get("x", 0.0)), float(poses[0].get("y", 0.0)))
    return None


def _collect_object_catalog(stage_path: Path, room_part: Dict[str, object]) -> List[Dict[str, object]]:
    catalog: List[Dict[str, object]] = []
    keywords = (
        "table",
        "desk",
        "chair",
        "sofa",
        "couch",
        "printer",
        "cabinet",
        "shelf",
        "counter",
        "fridge",
        "microwave",
        "sink",
        "monitor",
        "tv",
        "plant",
        "lamp",
        "stool",
        "drawer",
        "bench",
    )
    deny = ("wall", "floor", "ceiling", "window", "light", "camera", "sky", "ground")

    try:
        from pxr import Usd, UsdGeom  # type: ignore

        stage = Usd.Stage.Open(str(stage_path))
        if stage is None:
            return []
        bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_], useExtentsHint=True)
        seen = set()
        for prim in stage.TraverseAll():
            if not prim.IsActive() or prim.IsAbstract() or not prim.IsLoaded():
                continue
            try:
                if not prim.IsA(UsdGeom.Boundable):
                    continue
            except Exception:
                continue
            pth = prim.GetPath().pathString
            low = pth.lower()
            if any(d in low for d in deny):
                continue
            if not any(k in low for k in keywords):
                continue
            if pth in seen:
                continue
            try:
                wb = bbox_cache.ComputeWorldBound(prim)
                aabb = wb.ComputeAlignedBox()
                mn = aabb.GetMin()
                mx = aabb.GetMax()
                cx = 0.5 * (float(mn[0]) + float(mx[0]))
                cy = 0.5 * (float(mn[1]) + float(mx[1]))
                cz = 0.5 * (float(mn[2]) + float(mx[2]))
                sx = float(mx[0] - mn[0])
                sy = float(mx[1] - mn[1])
                sz = float(mx[2] - mn[2])
            except Exception:
                continue
            if max(sx, sy) > 8.0 or min(sx, sy, sz) <= 1e-4:
                continue
            rid = _room_id_from_xy(room_part, cx, cy)
            name = pth.split("/")[-1]
            cls = next((k for k in keywords if k in low), "object")
            catalog.append(
                {
                    "prim_path": pth,
                    "name": name,
                    "class": cls,
                    "pose": {"x": float(cx), "y": float(cy), "z": float(cz), "yaw_deg": 0.0},
                    "size": {"sx": sx, "sy": sy, "sz": sz},
                    "assigned_room_id": int(rid),
                }
            )
            seen.add(pth)
    except Exception:
        return []

    catalog.sort(key=lambda r: (int(r.get("assigned_room_id", -1)), str(r.get("class", "")), str(r.get("prim_path", ""))))
    return catalog


def _pick_object(catalog: Sequence[Dict[str, object]], q: str, room_id: int) -> Optional[Dict[str, object]]:
    qq = str(q or "").strip().lower()
    if not qq:
        return None
    matches: List[Dict[str, object]] = []
    for obj in catalog:
        text = " ".join(
            [
                str(obj.get("name", "")),
                str(obj.get("class", "")),
                str(obj.get("prim_path", "")),
            ]
        ).lower()
        if qq in text:
            matches.append(obj)
    if room_id >= 0:
        in_room = [m for m in matches if int(m.get("assigned_room_id", -1)) == int(room_id)]
        if len(in_room) > 0:
            matches = in_room
    if len(matches) == 0:
        return None
    return matches[0]


def _compute_approach(object_xy: Tuple[float, float], room_center: Optional[Tuple[float, float]], offset_m: float) -> Tuple[float, float, float]:
    ox, oy = float(object_xy[0]), float(object_xy[1])
    if room_center is None:
        ax = ox - float(offset_m)
        ay = oy
    else:
        cx, cy = float(room_center[0]), float(room_center[1])
        vx, vy = cx - ox, cy - oy
        n = math.hypot(vx, vy)
        if n <= 1e-6:
            ax, ay = ox - float(offset_m), oy
        else:
            ax = ox + (vx / n) * float(offset_m)
            ay = oy + (vy / n) * float(offset_m)
    yaw = math.degrees(math.atan2(oy - ay, ox - ax))
    return float(ax), float(ay), float(yaw)


def _build_staged_goals(
    room_path: Sequence[int],
    edge_map: Dict[Tuple[int, int], Dict[str, object]],
    target_pose: Tuple[float, float, float],
    task_id: str,
) -> List[Dict[str, object]]:
    goals: List[Dict[str, object]] = []

    for i in range(max(0, len(room_path) - 1)):
        a = int(room_path[i])
        b = int(room_path[i + 1])
        e = edge_map.get((min(a, b), max(a, b)))
        if e is None:
            continue
        dp = _door_pose(e)
        if dp is None:
            continue
        goals.append({"id": f"{task_id}_door_{a}_{b}", "x": float(dp[0]), "y": float(dp[1]), "yaw_deg": 0.0, "kind": "doorway"})

    goals.append(
        {
            "id": f"{task_id}_target",
            "x": float(target_pose[0]),
            "y": float(target_pose[1]),
            "yaw_deg": float(target_pose[2]),
            "kind": "target",
        }
    )

    # Deduplicate too-close points.
    dedup: List[Dict[str, object]] = []
    for g in goals:
        if len(dedup) == 0:
            dedup.append(g)
            continue
        p = dedup[-1]
        d = math.hypot(float(g["x"]) - float(p["x"]), float(g["y"]) - float(p["y"]))
        if d < 0.20:
            dedup[-1] = g
        else:
            dedup.append(g)

    # Set yaw toward next goal when possible.
    for i in range(len(dedup) - 1):
        dx = float(dedup[i + 1]["x"]) - float(dedup[i]["x"])
        dy = float(dedup[i + 1]["y"]) - float(dedup[i]["y"])
        if abs(dx) + abs(dy) > 1e-6:
            dedup[i]["yaw_deg"] = float(math.degrees(math.atan2(dy, dx)))

    return dedup


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    room_part = _load_json(Path(args.room_partition_json).expanduser().resolve())
    room_graph = _load_json(Path(args.room_graph_json).expanduser().resolve())

    centroids = _room_centroids(room_part)
    adj, edge_map = _build_room_graph(room_graph)

    start_xy = _parse_pose_trace(Path(args.pose_trace_csv).expanduser().resolve()) if str(args.pose_trace_csv).strip() else None
    if start_xy is None:
        # Fallback to first room centroid.
        start_xy = next(iter(centroids.values()), (0.0, 0.0))
    start_room = _room_id_from_xy(room_part, float(start_xy[0]), float(start_xy[1]))
    if start_room < 0 and len(centroids) > 0:
        start_room = sorted(centroids.keys())[0]

    task = str(args.task).strip().lower()
    if task not in {"doorway", "room", "object"}:
        task = "room"

    target_room = int(start_room)
    target_pose = (float(start_xy[0]), float(start_xy[1]), 0.0)
    source = "room"

    # Build object catalog once (can be empty if pxr unavailable).
    stage_path = Path(args.stage).expanduser().resolve()
    object_catalog = _collect_object_catalog(stage_path, room_part)
    catalog_path = out_dir / "object_catalog.json"
    catalog_path.write_text(json.dumps({"stage": str(stage_path), "objects": object_catalog}, indent=2), encoding="utf-8")
    print(f"[V35A_OBJECT_CATALOG] ok=1 objects={len(object_catalog)} rooms={len(centroids)}", flush=True)

    if task == "doorway":
        # Pick a doorway edge; prefer one connected to start_room.
        chosen_edge = None
        edges = [e for e in room_graph.get("edges", []) if isinstance(e, dict)]
        edges.sort(key=lambda e: (-int(e.get("passable", 0)), -int(e.get("count", 0))))
        for e in edges:
            a = int(e.get("room_a", -1))
            b = int(e.get("room_b", -1))
            if start_room in (a, b):
                chosen_edge = e
                break
        if chosen_edge is None and len(edges) > 0:
            chosen_edge = edges[0]
        if chosen_edge is not None:
            a = int(chosen_edge.get("room_a", start_room))
            b = int(chosen_edge.get("room_b", start_room))
            target_room = int(b if start_room == a else a if start_room == b else b)
            dp = _door_pose(chosen_edge)
            if dp is not None:
                target_pose = (float(dp[0]), float(dp[1]), 0.0)
                source = "doorway"
    elif task == "object":
        room_hint = _resolve_room_id(str(args.room), room_part, start_room)
        picked = _pick_object(object_catalog, str(args.object), room_hint)
        if picked is None:
            target_room = int(room_hint)
            c = centroids.get(target_room, next(iter(centroids.values()), (float(start_xy[0]), float(start_xy[1]))))
            target_pose = (float(c[0]), float(c[1]), 0.0)
            source = "room_fallback"
        else:
            rid = int(picked.get("assigned_room_id", room_hint))
            if rid < 0:
                rid = int(room_hint)
            target_room = int(rid)
            obj_pose = picked.get("pose", {})
            ox = float(obj_pose.get("x", 0.0)) if isinstance(obj_pose, dict) else 0.0
            oy = float(obj_pose.get("y", 0.0)) if isinstance(obj_pose, dict) else 0.0
            target_pose = _compute_approach((ox, oy), centroids.get(target_room), float(args.object_offset_m))
            source = "object"
    else:  # room
        rid = _resolve_room_id(str(args.room), room_part, start_room)
        if rid not in centroids and len(centroids) > 0:
            rid = sorted(centroids.keys())[0]
        target_room = int(rid)
        c = centroids.get(target_room, next(iter(centroids.values()), (float(start_xy[0]), float(start_xy[1]))))
        target_pose = (float(c[0]), float(c[1]), 0.0)
        source = "room"

    if float(args.goal_x) == float(args.goal_x) and float(args.goal_y) == float(args.goal_y):
        gx = float(args.goal_x)
        gy = float(args.goal_y)
        gyaw = float(args.goal_yaw_deg)
        target_pose = (gx, gy, gyaw)
        rid = _room_id_from_xy(room_part, gx, gy)
        if rid >= 0:
            target_room = int(rid)
        source = "override_pose"

    room_path = _bfs_path(adj, int(start_room), int(target_room))
    if len(room_path) == 0:
        room_path = [int(start_room), int(target_room)] if start_room != target_room else [int(start_room)]

    task_id = str(args.task_id or task)
    goals = _build_staged_goals(room_path, edge_map, target_pose, task_id)

    out_payload = {
        "task": task,
        "task_id": task_id,
        "stage": str(stage_path),
        "topo_nodes_json": str(args.topo_nodes_json),
        "start_xy": {"x": float(start_xy[0]), "y": float(start_xy[1])},
        "start_room": int(start_room),
        "target_room": int(target_room),
        "target_pose": {"x": float(target_pose[0]), "y": float(target_pose[1]), "yaw_deg": float(target_pose[2])},
        "source": source,
        "room_path": [int(x) for x in room_path],
        "goals": goals,
        "room": str(args.room),
        "object": str(args.object),
    }

    picked_path = out_dir / f"picked_goal_{task_id}.json"
    staged_path = out_dir / f"staged_goals_{task_id}.json"
    picked_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    staged_path.write_text(json.dumps({"task_id": task_id, "goals": goals}, indent=2), encoding="utf-8")

    print(
        f"[V35A_PICK_GOAL] room={str(args.room)} object={str(args.object)} "
        f"stage_xy=({float(target_pose[0]):.3f},{float(target_pose[1]):.3f}) yaw={float(target_pose[2]):.2f} "
        f"source={source} ok=1",
        flush=True,
    )
    print(f"[V35A_STAGED_PATH] task={task} task_id={task_id} start_room={int(start_room)} target_room={int(target_room)} waypoints={len(goals)} out={staged_path}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Resolve instruction room/object into staged Nav2 goals for v35a")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--room_partition_json", required=True)
    ap.add_argument("--room_graph_json", required=True)
    ap.add_argument("--topo_nodes_json", default="")
    ap.add_argument("--pose_trace_csv", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--task", default="room", choices=["doorway", "room", "object"])
    ap.add_argument("--task_id", default="")
    ap.add_argument("--room", default="")
    ap.add_argument("--object", default="")
    ap.add_argument("--object_offset_m", type=float, default=0.60)
    ap.add_argument("--goal_x", type=float, default=float("nan"))
    ap.add_argument("--goal_y", type=float, default=float("nan"))
    ap.add_argument("--goal_yaw_deg", type=float, default=0.0)
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
