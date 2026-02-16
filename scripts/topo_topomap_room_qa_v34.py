#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from internnav.sim_backend.isaac_backend import IsaacSimBackend


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, dict):
        return {}
    return data


def _load_scene_cfg(config_path: Path, scene_id: str) -> Dict[str, Any]:
    cfg = _load_yaml_or_json(config_path)
    scenes = cfg.get("scenes", []) if isinstance(cfg.get("scenes", []), list) else []
    for s in scenes:
        if isinstance(s, dict) and str(s.get("scene_id", "")) == str(scene_id):
            return dict(s)
    return {}


def _square_actions(steps: int, fwd_steps: int = 8, turn_steps: int = 6) -> List[int]:
    seq: List[int] = []
    while len(seq) < int(steps):
        seq.extend([1] * int(fwd_steps))
        seq.extend([3] * int(turn_steps))
    return seq[: int(steps)]


def _sample_nodes_from_traj(rows: List[Dict[str, Any]], stride: int) -> List[Dict[str, Any]]:
    if len(rows) == 0:
        return []
    keep_idx = set(range(0, len(rows), max(1, int(stride))))
    keep_idx.add(len(rows) - 1)
    nodes: List[Dict[str, Any]] = []
    nid = 0
    for i, r in enumerate(rows):
        if i not in keep_idx:
            continue
        nodes.append(
            {
                "node_id": int(nid),
                "step": int(i),
                "x": float(r["x"]),
                "z": float(r["z"]),
                "y": float(r.get("y", 0.0)),
                "yaw_deg": float(r["yaw_deg"]),
            }
        )
        nid += 1
    return nodes


def _assign_rooms(nodes: List[Dict[str, Any]]) -> Tuple[Dict[int, int], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    if len(nodes) == 0:
        return {}, {}, {"method": "quadrant_median", "stable_hint": "no_nodes"}
    xs = np.array([float(n["x"]) for n in nodes], dtype=np.float32)
    zs = np.array([float(n["z"]) for n in nodes], dtype=np.float32)
    mx = float(np.median(xs))
    mz = float(np.median(zs))
    node_to_room: Dict[int, int] = {}
    for n in nodes:
        x = float(n["x"])
        z = float(n["z"])
        qx = 1 if x > mx else 0
        qz = 1 if z > mz else 0
        rid = int(qx * 2 + qz)
        node_to_room[int(n["node_id"])] = rid
    room_nodes: Dict[int, List[int]] = defaultdict(list)
    for nid, rid in node_to_room.items():
        room_nodes[int(rid)].append(int(nid))
    rooms: Dict[int, Dict[str, Any]] = {}
    for rid, nids in sorted(room_nodes.items()):
        pts = np.array([[nodes[n]["x"], nodes[n]["z"]] for n in nids], dtype=np.float32)
        cx = float(np.mean(pts[:, 0]))
        cz = float(np.mean(pts[:, 1]))
        rooms[int(rid)] = {
            "room_id": int(rid),
            "node_ids": [int(x) for x in nids],
            "centroid": {"x": cx, "z": cz},
            "bbox": {
                "min": {"x": float(np.min(pts[:, 0])), "z": float(np.min(pts[:, 1]))},
                "max": {"x": float(np.max(pts[:, 0])), "z": float(np.max(pts[:, 1]))},
            },
        }
    stable_hint = "stable" if len(rooms) >= 2 else "degenerate_single_room"
    meta = {"method": "quadrant_median", "median_x": mx, "median_z": mz, "stable_hint": stable_hint}
    return node_to_room, rooms, meta


def _build_topo_edges(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    for i in range(1, len(nodes)):
        u = int(nodes[i - 1]["node_id"])
        v = int(nodes[i]["node_id"])
        dx = float(nodes[i]["x"] - nodes[i - 1]["x"])
        dz = float(nodes[i]["z"] - nodes[i - 1]["z"])
        w = float(math.hypot(dx, dz))
        edges.append({"u": u, "v": v, "w": w, "edge_type": "temporal"})
    return edges


def _connected_components(room_nodes: Sequence[int], room_edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    adj: Dict[int, List[int]] = {int(r): [] for r in room_nodes}
    for a, b in room_edges:
        if int(a) == int(b):
            continue
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    seen = set()
    comps: List[List[int]] = []
    for rid in sorted(adj.keys()):
        if rid in seen:
            continue
        q = deque([rid])
        seen.add(rid)
        comp: List[int] = []
        while q:
            cur = q.popleft()
            comp.append(int(cur))
            for nb in adj.get(cur, []):
                if nb in seen:
                    continue
                seen.add(nb)
                q.append(nb)
        comps.append(sorted(comp))
    return comps


def _nearest_node_id(x: float, z: float, nodes: List[Dict[str, Any]]) -> Tuple[int, float]:
    best_nid = -1
    best_d = float("inf")
    for n in nodes:
        d = float(math.hypot(float(n["x"]) - x, float(n["z"]) - z))
        if d < best_d:
            best_d = d
            best_nid = int(n["node_id"])
    return int(best_nid), float(best_d)


def _compute_metrics(seq_rows: List[Dict[str, Any]], doorway_edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps = len(seq_rows)
    if steps <= 1:
        return {
            "room_consistency": 0.0,
            "topo_consistency": 0.0,
            "doorway_plausibility": 0.0,
            "unknown_rate": 1.0,
            "components": 0,
            "room_changes": 0,
            "topo_changes": 0,
        }
    room_changes = 0
    topo_changes = 0
    unknown = 0
    for i in range(1, steps):
        if int(seq_rows[i]["room_id"]) != int(seq_rows[i - 1]["room_id"]):
            room_changes += 1
        if int(seq_rows[i]["topo_node_id"]) != int(seq_rows[i - 1]["topo_node_id"]):
            topo_changes += 1
    for r in seq_rows:
        if int(r["room_id"]) < 0 or int(r["topo_node_id"]) < 0:
            unknown += 1
    denom = max(1.0, steps / 10.0)
    room_consistency = float(max(0.0, 1.0 - (room_changes / denom)))
    topo_consistency = float(max(0.0, 1.0 - (topo_changes / denom)))
    passable = sum(1 for e in doorway_edges if int(e.get("passable", 0)) == 1)
    doorway_plausibility = float(passable / max(1, len(doorway_edges)))
    unknown_rate = float(unknown / max(1, steps))
    return {
        "room_consistency": room_consistency,
        "topo_consistency": topo_consistency,
        "doorway_plausibility": doorway_plausibility,
        "unknown_rate": unknown_rate,
        "room_changes": int(room_changes),
        "topo_changes": int(topo_changes),
    }


def _write_overlay(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    doorway_edges: List[Dict[str, Any]],
    max_frames: int = 20,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(rows), int(max_frames))
    for i in range(n):
        r = rows[i]
        img = Image.fromarray(np.asarray(r["rgb"], dtype=np.uint8))
        draw = ImageDraw.Draw(img)
        lines = [
            f"step={int(r['step'])}",
            f"pose=({float(r['x']):.2f},{float(r['z']):.2f}) yaw={float(r['yaw_deg']):.1f}",
            f"room_id={int(r['room_id'])} topo_node_id={int(r['topo_node_id'])}",
            f"loc_conf={float(r['loc_conf']):.3f}",
        ]
        near_text = ""
        x = float(r["x"])
        z = float(r["z"])
        best_d = float("inf")
        best_edge: Optional[Dict[str, Any]] = None
        for e in doorway_edges:
            cx = float(e.get("center", {}).get("x", 0.0))
            cz = float(e.get("center", {}).get("z", 0.0))
            d = float(math.hypot(cx - x, cz - z))
            if d < best_d:
                best_d = d
                best_edge = e
        if best_edge is not None and best_d < 1.5:
            near_text = (
                f"doorway_candidate rooms={int(best_edge.get('room_u', -1))}"
                f"->{int(best_edge.get('room_v', -1))} d={best_d:.2f}"
            )
            lines.append(near_text)
        y = 8
        for ln in lines:
            draw.rectangle([(6, y - 2), (620, y + 14)], fill=(0, 0, 0, 160))
            draw.text((10, y), ln, fill=(255, 255, 255))
            y += 18
        img.save(out_dir / f"overlay_rgb_{i:03d}.png")
    return int(n)


def run(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    capture_rgb_dir = out_root / "capture" / "rgb"
    capture_depth_png_dir = out_root / "capture" / "depth_png"
    capture_depth_npy_dir = out_root / "capture" / "depth_npy"
    overlay_dir = out_root / "overlay"
    for p in [capture_rgb_dir, capture_depth_png_dir, capture_depth_npy_dir, overlay_dir]:
        p.mkdir(parents=True, exist_ok=True)

    scene_cfg = _load_scene_cfg(Path(args.isaac_config), args.scene_id)
    metrics: Dict[str, Any] = {
        "scene_id": str(args.scene_id),
        "ok": 0,
        "reason": "",
    }
    rows: List[Dict[str, Any]] = []

    def _emit_fail(reason: str) -> None:
        metrics_local = {
            "scene_id": str(args.scene_id),
            "ok": 0,
            "reason": str(reason),
            "room_consistency": 0.0,
            "topo_consistency": 0.0,
            "doorway_plausibility": 0.0,
            "unknown_rate": 1.0,
        }
        (out_root / "qa_metrics.json").write_text(json.dumps(metrics_local, indent=2), encoding="utf-8")
        (out_root / "room_partition.json").write_text(
            json.dumps({"scene_id": str(args.scene_id), "ok": 0, "reason": str(reason), "rooms": {}}, indent=2),
            encoding="utf-8",
        )
        (out_root / "doorway_edges.json").write_text(
            json.dumps({"ok": 0, "reason": str(reason), "edges": []}, indent=2), encoding="utf-8"
        )
        (out_root / "room_graph.json").write_text(
            json.dumps(
                {"scene_id": str(args.scene_id), "ok": 0, "reason": str(reason), "rooms": {}, "edges": [], "components": []},
                indent=2,
            ),
            encoding="utf-8",
        )
        (out_root / "topo_map.json").write_text(
            json.dumps({"scene_id": str(args.scene_id), "ok": 0, "reason": str(reason), "nodes": [], "edges": []}, indent=2),
            encoding="utf-8",
        )
        with (out_root / "traj_room_sequence.csv").open("w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["step", "x", "z", "yaw_deg", "room_id", "topo_node_id", "loc_conf", "collision"])
        print(f"[ROOM_PARTITION] ok=0 rooms=0 method=none stable_hint=runtime_error", flush=True)
        print(f"[DOORWAY_DETECT] ok=0 edges=0 passable=0 blocked=0 reason={reason}", flush=True)
        print(f"[ROOM_GRAPH] ok=0 rooms=0 edges=0 components=0 largest_component=0 reason={reason}", flush=True)
        print(f"[TOPO_MAP] ok=0 nodes=0 edges=0 reason={reason}", flush=True)
        print("[TOPO_QA] room_consistency=0.000 topo_consistency=0.000 doorway_plausibility=0.000 unknown_rate=1.000", flush=True)
        print(f"[TOPO_QA_OVERLAY] ok=0 frames=0 dir={overlay_dir}", flush=True)

    backend: Optional[IsaacSimBackend] = None
    try:
        backend = IsaacSimBackend(
            scene_id=str(args.scene_id),
            scene_cfg=scene_cfg,
            config_path="repo/InternNav/scripts/eval/configs/vln_r2r.yaml",
            out_dir=str(out_root),
            fallback_to_habitat=False,
            dt_action=float(scene_cfg.get("dt_action", 0.5)) if isinstance(scene_cfg, dict) else 0.5,
        )
        start_spec = {
            "start_offset": int(args.start_offset),
            "kidnap_start": 0,
            "kidnap_seed": int(args.seed),
        }
        obs = backend.reset(scene_id=str(args.scene_id), start_spec=start_spec)
        actions = _square_actions(steps=int(args.steps), fwd_steps=int(args.fwd_steps), turn_steps=int(args.turn_steps))

        for i, act in enumerate(actions):
            rgb = np.asarray(obs.rgb, dtype=np.uint8)
            depth = None if obs.depth is None else np.asarray(obs.depth, dtype=np.float32)
            rows.append(
                {
                    "step": int(i),
                    "x": float(obs.pose.x),
                    "y": float(obs.pose.y),
                    "z": float(obs.pose.z),
                    "yaw_deg": float(math.degrees(float(obs.pose.yaw))),
                    "rgb": rgb,
                    "depth": depth,
                    "collision": int(bool(obs.collision)) if obs.collision is not None else 0,
                    "action": int(act),
                    "loc_conf": 1.0,
                }
            )
            if i < int(args.save_frames):
                Image.fromarray(rgb).save(capture_rgb_dir / f"rgb_{i:03d}.png")
                if depth is not None:
                    d = np.asarray(depth, dtype=np.float32)
                    np.save(capture_depth_npy_dir / f"depth_{i:03d}.npy", d)
                    valid = np.isfinite(d) & (d > 1e-4)
                    vis = np.zeros_like(d, dtype=np.float32)
                    if valid.any():
                        lo = float(np.percentile(d[valid], 2))
                        hi = float(np.percentile(d[valid], 98))
                        denom = max(1e-6, hi - lo)
                        vis = np.clip((d - lo) / denom, 0.0, 1.0)
                    vis_u8 = (vis * 255.0).astype(np.uint8)
                    Image.fromarray(vis_u8).save(capture_depth_png_dir / f"depth_{i:03d}.png")
            obs, _, _ = backend.step(int(act))
    except Exception as e:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        _emit_fail(f"qa_runtime_error:{type(e).__name__}:{e}")
        return 0

    try:
        nodes = _sample_nodes_from_traj(rows, stride=int(args.node_stride))
        edges = _build_topo_edges(nodes)
        node_to_room, rooms, room_meta = _assign_rooms(nodes)

        topo_nodes: List[Dict[str, Any]] = []
        for n in nodes:
            nid = int(n["node_id"])
            rid = int(node_to_room.get(nid, -1))
            topo_nodes.append(
                {
                    "node_id": int(nid),
                    "x": float(n["x"]),
                    "y": float(n["y"]),
                    "z": float(n["z"]),
                    "yaw_deg": float(n["yaw_deg"]),
                    "room_id": int(rid),
                }
            )

        # doorway candidates from cross-room temporal edges
        doorway_edges: List[Dict[str, Any]] = []
        room_edge_set = set()
        room_transition_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for e in edges:
            u = int(e["u"])
            v = int(e["v"])
            ru = int(node_to_room.get(u, -1))
            rv = int(node_to_room.get(v, -1))
            if ru < 0 or rv < 0 or ru == rv:
                continue
            key = tuple(sorted((ru, rv)))
            room_edge_set.add(key)
            room_transition_counts[key] += 1
            nu = nodes[u]
            nv = nodes[v]
            cx = 0.5 * (float(nu["x"]) + float(nv["x"]))
            cz = 0.5 * (float(nu["z"]) + float(nv["z"]))
            dist = float(e["w"])
            score = float(1.0 / (1.0 + dist))
            passable = 1 if room_transition_counts[key] > 0 else 0
            doorway_edges.append(
                {
                    "u": int(u),
                    "v": int(v),
                    "room_u": int(ru),
                    "room_v": int(rv),
                    "score": float(score),
                    "distance": float(dist),
                    "center": {"x": cx, "z": cz},
                    "passable": int(passable),
                }
            )

        room_graph_edges: List[Dict[str, Any]] = []
        for a, b in sorted(room_edge_set):
            room_graph_edges.append(
                {
                    "u": int(a),
                    "v": int(b),
                    "doorway_count": int(room_transition_counts.get((a, b), 0)),
                }
            )
        comps = _connected_components(sorted(rooms.keys()), [(e["u"], e["v"]) for e in room_graph_edges])
        largest_comp = max((len(c) for c in comps), default=0)

        # trajectory room/node assignment
        traj_rows: List[Dict[str, Any]] = []
        for r in rows:
            nid, nn_d = _nearest_node_id(float(r["x"]), float(r["z"]), nodes)
            rid = int(node_to_room.get(nid, -1))
            loc_conf = float(math.exp(-nn_d))
            r["topo_node_id"] = int(nid)
            r["room_id"] = int(rid)
            r["loc_conf"] = float(loc_conf)
            traj_rows.append(
                {
                    "step": int(r["step"]),
                    "x": float(r["x"]),
                    "z": float(r["z"]),
                    "yaw_deg": float(r["yaw_deg"]),
                    "room_id": int(rid),
                    "topo_node_id": int(nid),
                    "loc_conf": float(loc_conf),
                    "collision": int(r["collision"]),
                }
            )

        metrics.update(_compute_metrics(traj_rows, doorway_edges))
        metrics.update(
            {
                "ok": 1,
                "rooms": int(len(rooms)),
                "topo_nodes": int(len(nodes)),
                "topo_edges": int(len(edges)),
                "room_components": int(len(comps)),
                "largest_component": int(largest_comp),
                "doorway_edges": int(len(doorway_edges)),
                "doorway_passable": int(sum(1 for e in doorway_edges if int(e.get("passable", 0)) == 1)),
                "doorway_blocked": int(sum(1 for e in doorway_edges if int(e.get("passable", 0)) == 0)),
                "method": str(room_meta.get("method", "unknown")),
                "stable_hint": str(room_meta.get("stable_hint", "unknown")),
            }
        )

        room_partition = {
            "scene_id": str(args.scene_id),
            "method": str(room_meta.get("method", "quadrant_median")),
            "stable_hint": str(room_meta.get("stable_hint", "unknown")),
            "rooms": {str(k): v for k, v in sorted(rooms.items())},
            "node_to_room": {str(k): int(v) for k, v in sorted(node_to_room.items())},
            "meta": room_meta,
        }
        (out_root / "room_partition.json").write_text(json.dumps(room_partition, indent=2), encoding="utf-8")
        (out_root / "doorway_edges.json").write_text(json.dumps({"edges": doorway_edges}, indent=2), encoding="utf-8")
        room_graph = {
            "scene_id": str(args.scene_id),
            "rooms": room_partition["rooms"],
            "edges": room_graph_edges,
            "components": comps,
            "largest_component": int(largest_comp),
        }
        (out_root / "room_graph.json").write_text(json.dumps(room_graph, indent=2), encoding="utf-8")
        topo_map = {
            "scene_id": str(args.scene_id),
            "nodes": topo_nodes,
            "edges": edges,
        }
        (out_root / "topo_map.json").write_text(json.dumps(topo_map, indent=2), encoding="utf-8")
        (out_root / "qa_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        with (out_root / "traj_room_sequence.csv").open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(
                f,
                fieldnames=["step", "x", "z", "yaw_deg", "room_id", "topo_node_id", "loc_conf", "collision"],
            )
            wr.writeheader()
            for r in traj_rows:
                wr.writerow(r)

        overlay_frames = _write_overlay(rows=rows, out_dir=overlay_dir, doorway_edges=doorway_edges, max_frames=int(args.overlay_frames))

        print(
            f"[ROOM_PARTITION] ok=1 rooms={int(len(rooms))} method={str(room_meta.get('method','unknown'))} "
            f"stable_hint={str(room_meta.get('stable_hint','unknown'))}",
            flush=True,
        )
        print(
            f"[DOORWAY_DETECT] ok=1 edges={int(len(doorway_edges))} "
            f"passable={int(metrics['doorway_passable'])} blocked={int(metrics['doorway_blocked'])}",
            flush=True,
        )
        print(
            f"[ROOM_GRAPH] ok=1 rooms={int(len(rooms))} edges={int(len(room_graph_edges))} "
            f"components={int(len(comps))} largest_component={int(largest_comp)}",
            flush=True,
        )
        print(f"[TOPO_MAP] ok=1 nodes={int(len(nodes))} edges={int(len(edges))}", flush=True)
        print(
            f"[TOPO_QA] room_consistency={float(metrics['room_consistency']):.3f} "
            f"topo_consistency={float(metrics['topo_consistency']):.3f} "
            f"doorway_plausibility={float(metrics['doorway_plausibility']):.3f} "
            f"unknown_rate={float(metrics['unknown_rate']):.3f}",
            flush=True,
        )
        print(f"[TOPO_QA_OVERLAY] ok=1 frames={int(overlay_frames)} dir={overlay_dir}", flush=True)
    except Exception as e:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        _emit_fail(f"qa_postproc_error:{type(e).__name__}:{e}")
        return 0
    if backend is not None:
        try:
            backend.close()
        except Exception:
            pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="v34 topo-room QA extractor")
    ap.add_argument("--scene_id", type=str, default="office_localized")
    ap.add_argument("--isaac_config", type=str, default="configs/isaac_scenes_v29.yaml")
    ap.add_argument("--out_root", type=str, default="runs/topo_mvp/v34_topo_room_qa")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--fwd_steps", type=int, default=8)
    ap.add_argument("--turn_steps", type=int, default=6)
    ap.add_argument("--node_stride", type=int, default=4)
    ap.add_argument("--save_frames", type=int, default=20)
    ap.add_argument("--overlay_frames", type=int, default=20)
    ap.add_argument("--start_offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
