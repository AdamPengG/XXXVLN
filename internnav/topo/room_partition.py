import collections
import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from internnav.topo.graph import TopoGraph


@dataclass
class RoomPartitionResult:
    components: List[List[int]]
    room_of_node: Dict[int, int]
    cluster_info: Dict[str, object]
    edge_clearance_debug: Dict[str, object]
    doorway_debug: Dict[str, object]


def _clearance_map_from_meta(node_ids: List[int], node_meta: Dict[str, Dict]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for nid in node_ids:
        meta = node_meta.get(str(int(nid)), {})
        c = meta.get("clearance_m", meta.get("clearance_proxy", 0.0))
        try:
            cv = float(c)
        except Exception:
            cv = 0.0
        if not np.isfinite(cv):
            cv = 0.0
        out[int(nid)] = float(max(0.0, cv))
    return out


def _positions_from_graph(graph: TopoGraph, node_ids: List[int]) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for nid in node_ids:
        out[int(nid)] = np.asarray(graph.nodes[int(nid)].position, dtype=np.float32)
    return out


def _unique_edges(graph: TopoGraph) -> List[Tuple[int, int, float]]:
    rows = []
    for u, nbrs in graph.edges.items():
        uu = int(u)
        for v, w in nbrs.items():
            vv = int(v)
            if uu >= vv:
                continue
            rows.append((uu, vv, float(w)))
    return rows


def _edge_clearance_min_sample(
    pu: np.ndarray,
    pv: np.ndarray,
    node_ids: List[int],
    node_positions: Dict[int, np.ndarray],
    clearance_map: Dict[int, float],
    sample_steps: int,
) -> float:
    samples = max(3, int(sample_steps))
    min_c = float("inf")
    for t in np.linspace(0.0, 1.0, num=samples, dtype=np.float32):
        p = (1.0 - float(t)) * pu + float(t) * pv
        best_nid = None
        best_d = float("inf")
        for nid in node_ids:
            d = float(np.linalg.norm(node_positions[int(nid)] - p))
            if d < best_d:
                best_d = d
                best_nid = int(nid)
        if best_nid is None:
            c = 0.0
        else:
            c = float(clearance_map.get(int(best_nid), 0.0))
        min_c = min(min_c, float(c))
    if not np.isfinite(min_c):
        min_c = 0.0
    return float(max(0.0, min_c))


def _connected_components_filtered(
    node_ids: List[int],
    clearance_map: Dict[int, float],
    edge_clearance: Dict[Tuple[int, int], float],
    delta: float,
    min_component_nodes: int,
) -> Tuple[List[List[int]], List[List[int]], set]:
    kept = set(int(n) for n in node_ids if float(clearance_map.get(int(n), 0.0)) >= float(delta))
    if len(kept) == 0:
        return [], [], set()
    adj: Dict[int, List[int]] = collections.defaultdict(list)
    for (u, v), ec in edge_clearance.items():
        if float(ec) < float(delta):
            continue
        uu = int(u)
        vv = int(v)
        if uu not in kept or vv not in kept:
            continue
        adj[uu].append(vv)
        adj[vv].append(uu)
    seen = set()
    comps = []
    for nid in sorted(kept):
        if nid in seen:
            continue
        stack = [int(nid)]
        seen.add(int(nid))
        comp = []
        while stack:
            u = int(stack.pop())
            comp.append(u)
            for v in adj.get(u, []):
                vv = int(v)
                if vv in seen:
                    continue
                seen.add(vv)
                stack.append(vv)
        comps.append(sorted(comp))
    sig = [c for c in comps if len(c) >= int(min_component_nodes)]
    return comps, sig, kept


def _select_plateau(records: List[Dict[str, object]], total_nodes: int) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    plateaus: List[Dict[str, object]] = []
    i = 0
    while i < len(records):
        k = int(records[i]["K"])
        j = i
        while j + 1 < len(records) and int(records[j + 1]["K"]) == k:
            j += 1
        rec_slice = records[i : j + 1]
        delta_hi = float(records[i]["delta"])
        delta_lo = float(records[j]["delta"])
        delta_len = float(max(0.0, delta_hi - delta_lo))
        plateau_edges = int(j - i + 1)
        med_sizes = [float(r.get("median_size", 0.0)) for r in rec_slice]
        covs = [float(r.get("coverage", 0.0)) for r in rec_slice]
        median_size = float(np.median(np.asarray(med_sizes, dtype=np.float32))) if len(med_sizes) > 0 else 0.0
        coverage = float(np.median(np.asarray(covs, dtype=np.float32))) if len(covs) > 0 else 0.0
        pref_k = 1.0 if 2 <= k <= 10 else 0.35
        score = (delta_len + 1e-5) * float(plateau_edges) * float(pref_k) * (
            float(coverage) + 0.10 * float(median_size) / max(1.0, float(total_nodes))
        )
        plateaus.append(
            {
                "start_idx": int(i),
                "end_idx": int(j),
                "K": int(k),
                "delta_high": float(delta_hi),
                "delta_low": float(delta_lo),
                "plateau_delta": float(delta_len),
                "plateau_edges": int(plateau_edges),
                "median_size": float(median_size),
                "coverage": float(coverage),
                "score": float(score),
            }
        )
        i = j + 1
    if len(plateaus) == 0:
        return {}, []
    plateaus_sorted = sorted(
        plateaus,
        key=lambda x: (
            float(x["score"]),
            1 if 2 <= int(x["K"]) <= 10 else 0,
            float(x["delta_high"]),
        ),
        reverse=True,
    )
    best = dict(plateaus_sorted[0])
    return best, plateaus_sorted


def _flood_fill_assign(
    node_ids: List[int],
    seed_components: List[List[int]],
    edge_clearance: Dict[Tuple[int, int], float],
    node_positions: Dict[int, np.ndarray],
) -> Dict[int, int]:
    room_of_node: Dict[int, int] = {}
    for rid, comp in enumerate(seed_components):
        for n in comp:
            room_of_node[int(n)] = int(rid)

    adjw: Dict[int, List[Tuple[int, float]]] = collections.defaultdict(list)
    for (u, v), ec in edge_clearance.items():
        uu = int(u)
        vv = int(v)
        ww = float(ec)
        adjw[uu].append((vv, ww))
        adjw[vv].append((uu, ww))

    heap: List[Tuple[float, int, int]] = []
    for n, rid in room_of_node.items():
        for v, ec in adjw.get(int(n), []):
            if int(v) in room_of_node:
                continue
            heapq.heappush(heap, (-float(ec), int(rid), int(v)))

    while heap:
        neg_ec, rid, v = heapq.heappop(heap)
        if int(v) in room_of_node:
            continue
        room_of_node[int(v)] = int(rid)
        for nxt, ec in adjw.get(int(v), []):
            if int(nxt) in room_of_node:
                continue
            heapq.heappush(heap, (-float(ec), int(rid), int(nxt)))

    # Remaining isolated nodes: nearest assigned node by geometric distance.
    assigned_nodes = [int(n) for n in room_of_node.keys()]
    for nid in node_ids:
        if int(nid) in room_of_node:
            continue
        if len(assigned_nodes) == 0:
            room_of_node[int(nid)] = 0
            assigned_nodes.append(int(nid))
            continue
        p = node_positions[int(nid)]
        best_room = None
        best_d = float("inf")
        for a in assigned_nodes:
            d = float(np.linalg.norm(node_positions[int(a)] - p))
            if d < best_d:
                best_d = d
                best_room = int(room_of_node[int(a)])
        if best_room is None:
            best_room = int(max(room_of_node.values()) + 1)
        room_of_node[int(nid)] = int(best_room)
        assigned_nodes.append(int(nid))

    # Re-index room ids to stable order by min node id.
    grouped: Dict[int, List[int]] = collections.defaultdict(list)
    for nid, rid in room_of_node.items():
        grouped[int(rid)].append(int(nid))
    ordered = sorted(grouped.items(), key=lambda kv: min(kv[1]))
    rid_map = {int(old): int(new) for new, (old, _) in enumerate(ordered)}
    out = {int(nid): int(rid_map[int(rid)]) for nid, rid in room_of_node.items()}
    return out


def _collect_doorways(
    room_of_node: Dict[int, int],
    edge_clearance: Dict[Tuple[int, int], float],
    clearance_map: Dict[int, float],
    delta_star: float,
) -> Dict[str, object]:
    cross_rows = []
    for (u, v), ec in edge_clearance.items():
        ru = int(room_of_node.get(int(u), -1))
        rv = int(room_of_node.get(int(v), -1))
        if ru < 0 or rv < 0 or ru == rv:
            continue
        cross_rows.append(
            {
                "u": int(u),
                "v": int(v),
                "room_u": int(ru),
                "room_v": int(rv),
                "edge_clearance_m": float(ec),
            }
        )
    if len(cross_rows) == 0:
        return {
            "delta_star": float(delta_star),
            "doorway_edges": [],
            "doorway_nodes": [],
            "cross_room_edges": 0,
            "doorway_edges_count": 0,
            "doorway_nodes_count": 0,
        }
    cross_vals = np.asarray([float(r["edge_clearance_m"]) for r in cross_rows], dtype=np.float32)
    q_low = float(np.quantile(cross_vals, 0.25))
    eps = float(max(0.02, 0.05 * max(1e-6, float(delta_star))))
    doorway_edges = []
    doorway_nodes = set()
    for row in cross_rows:
        ec = float(row["edge_clearance_m"])
        if ec <= float(delta_star + eps) or ec <= float(q_low):
            doorway_edges.append(dict(row))
            if float(clearance_map.get(int(row["u"]), 0.0)) <= float(delta_star + eps):
                doorway_nodes.add(int(row["u"]))
            if float(clearance_map.get(int(row["v"]), 0.0)) <= float(delta_star + eps):
                doorway_nodes.add(int(row["v"]))
    return {
        "delta_star": float(delta_star),
        "cross_room_edges": int(len(cross_rows)),
        "cross_room_q25": float(q_low),
        "doorway_edges_count": int(len(doorway_edges)),
        "doorway_nodes_count": int(len(doorway_nodes)),
        "doorway_edges": doorway_edges,
        "doorway_nodes": sorted(int(x) for x in doorway_nodes),
    }


def partition_rooms_hydralite_edge(
    graph: TopoGraph,
    node_meta: Dict[str, Dict],
    sample_steps: int = 12,
    min_component_nodes: int = 2,
    q_min: float = 0.10,
    q_max: float = 0.80,
    q_points: int = 25,
    mode: str = "PLATEAU",
    min_dilation_m: float = -1.0,
    max_dilation_m: float = -1.0,
) -> RoomPartitionResult:
    node_ids = sorted(int(n) for n in graph.nodes.keys())
    if len(node_ids) <= 1:
        return RoomPartitionResult(
            components=[list(node_ids)],
            room_of_node={int(n): 0 for n in node_ids},
            cluster_info={
                "method": "hydralite_edge_filtration",
                "mode": str(mode),
                "fallback": 1,
                "fallback_reason": "too_few_nodes",
                "delta_star": 0.0,
                "K": 1 if len(node_ids) > 0 else 0,
                "plateau_edges": 0,
                "plateau_delta": 0.0,
                "deltas": [],
                "Ks": [],
                "sizes": [int(len(node_ids))] if len(node_ids) > 0 else [],
            },
            edge_clearance_debug={"edge_clearances": [], "stats": {}},
            doorway_debug={
                "delta_star": 0.0,
                "doorway_edges": [],
                "doorway_nodes": [],
                "cross_room_edges": 0,
                "doorway_edges_count": 0,
                "doorway_nodes_count": 0,
            },
        )

    node_positions = _positions_from_graph(graph, node_ids)
    clearance_map = _clearance_map_from_meta(node_ids, node_meta)
    clear_vals = np.asarray([float(clearance_map[int(n)]) for n in node_ids], dtype=np.float32)
    if clear_vals.size == 0 or float(np.max(clear_vals) - np.min(clear_vals)) < 1e-5:
        comps = [list(node_ids)]
        return RoomPartitionResult(
            components=comps,
            room_of_node={int(n): 0 for n in node_ids},
            cluster_info={
                "method": "hydralite_edge_filtration",
                "mode": str(mode),
                "fallback": 1,
                "fallback_reason": "degenerate_clearance",
                "delta_star": float(np.max(clear_vals)) if clear_vals.size > 0 else 0.0,
                "K": 1,
                "plateau_edges": 0,
                "plateau_delta": 0.0,
                "deltas": [],
                "Ks": [],
                "sizes": [int(len(node_ids))],
            },
            edge_clearance_debug={"edge_clearances": [], "stats": {}},
            doorway_debug={
                "delta_star": float(np.max(clear_vals)) if clear_vals.size > 0 else 0.0,
                "doorway_edges": [],
                "doorway_nodes": [],
                "cross_room_edges": 0,
                "doorway_edges_count": 0,
                "doorway_nodes_count": 0,
            },
        )

    edge_rows = _unique_edges(graph)
    if len(edge_rows) == 0:
        comps = [list(node_ids)]
        return RoomPartitionResult(
            components=comps,
            room_of_node={int(n): 0 for n in node_ids},
            cluster_info={
                "method": "hydralite_edge_filtration",
                "mode": str(mode),
                "fallback": 1,
                "fallback_reason": "no_edges",
                "delta_star": 0.0,
                "K": 1,
                "plateau_edges": 0,
                "plateau_delta": 0.0,
                "deltas": [],
                "Ks": [],
                "sizes": [int(len(node_ids))],
            },
            edge_clearance_debug={"edge_clearances": [], "stats": {}},
            doorway_debug={
                "delta_star": 0.0,
                "doorway_edges": [],
                "doorway_nodes": [],
                "cross_room_edges": 0,
                "doorway_edges_count": 0,
                "doorway_nodes_count": 0,
            },
        )

    edge_clearance: Dict[Tuple[int, int], float] = {}
    edge_debug_rows = []
    for u, v, w in edge_rows:
        pu = node_positions[int(u)]
        pv = node_positions[int(v)]
        ec = _edge_clearance_min_sample(
            pu=pu,
            pv=pv,
            node_ids=node_ids,
            node_positions=node_positions,
            clearance_map=clearance_map,
            sample_steps=sample_steps,
        )
        key = (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))
        edge_clearance[key] = float(ec)
        edge_debug_rows.append({"u": int(u), "v": int(v), "w": float(w), "edge_clearance_m": float(ec)})

    edge_vals = np.asarray([float(r["edge_clearance_m"]) for r in edge_debug_rows], dtype=np.float32)
    if edge_vals.size == 0:
        comps = [list(node_ids)]
        return RoomPartitionResult(
            components=comps,
            room_of_node={int(n): 0 for n in node_ids},
            cluster_info={
                "method": "hydralite_edge_filtration",
                "mode": str(mode),
                "fallback": 1,
                "fallback_reason": "degenerate_edge_clearance",
                "delta_star": 0.0,
                "K": 1,
                "plateau_edges": 0,
                "plateau_delta": 0.0,
                "deltas": [],
                "Ks": [],
                "sizes": [int(len(node_ids))],
            },
            edge_clearance_debug={"edge_clearances": edge_debug_rows, "stats": {}},
            doorway_debug={
                "delta_star": 0.0,
                "doorway_edges": [],
                "doorway_nodes": [],
                "cross_room_edges": 0,
                "doorway_edges_count": 0,
                "doorway_nodes_count": 0,
            },
        )

    q_lo = float(np.clip(q_min, 0.0, 1.0))
    q_hi = float(np.clip(q_max, q_lo + 1e-3, 1.0))
    v_lo = float(np.quantile(edge_vals, q_lo))
    v_hi = float(np.quantile(edge_vals, q_hi))
    if float(min_dilation_m) > 0.0 and float(max_dilation_m) > float(min_dilation_m):
        v_lo = max(float(v_lo), float(min_dilation_m))
        v_hi = min(float(v_hi), float(max_dilation_m))
        if v_hi <= v_lo:
            v_lo = float(np.quantile(edge_vals, q_lo))
            v_hi = float(np.quantile(edge_vals, q_hi))
    uniq = np.unique(np.round(edge_vals, 4))
    vals = [float(x) for x in uniq.tolist() if float(v_lo) <= float(x) <= float(v_hi)]
    vals.sort(reverse=True)
    if len(vals) == 0:
        vals = [float(v_hi), float(v_lo)] if v_hi >= v_lo else [float(v_lo)]
    if len(vals) > int(max(8, q_points)):
        qs = np.linspace(q_lo, q_hi, num=int(max(8, q_points)), dtype=np.float32)
        vals = sorted(
            list(
                {
                    float(np.quantile(edge_vals, float(q)))
                    for q in qs.tolist()
                }
            ),
            reverse=True,
        )

    records: List[Dict[str, object]] = []
    for delta in vals:
        comps, sig, kept = _connected_components_filtered(
            node_ids=node_ids,
            clearance_map=clearance_map,
            edge_clearance=edge_clearance,
            delta=float(delta),
            min_component_nodes=int(min_component_nodes),
        )
        cov = float(sum(len(c) for c in sig) / max(1, len(node_ids)))
        med = float(np.median(np.asarray([len(c) for c in sig], dtype=np.float32))) if len(sig) > 0 else 0.0
        records.append(
            {
                "delta": float(delta),
                "K": int(len(sig)),
                "coverage": float(cov),
                "median_size": float(med),
                "components": comps,
                "sig_components": sig,
                "kept_nodes": int(len(kept)),
            }
        )

    best_plateau, all_plateaus = _select_plateau(records, total_nodes=len(node_ids))
    fallback = 0
    fallback_reason = ""
    if len(best_plateau) == 0:
        fallback = 1
        fallback_reason = "no_plateau"
        delta_star = float(records[0]["delta"]) if len(records) > 0 else 0.0
        seeds = [list(node_ids)]
        K = 1
        plateau_edges = 0
        plateau_delta = 0.0
    else:
        start_idx = int(best_plateau["start_idx"])
        rec = records[start_idx]
        delta_star = float(rec["delta"])
        seeds = [list(map(int, c)) for c in rec.get("sig_components", [])]
        K = int(rec["K"])
        plateau_edges = int(best_plateau["plateau_edges"])
        plateau_delta = float(best_plateau["plateau_delta"])
        if K < 2 or len(seeds) <= 1:
            fallback = 1
            fallback_reason = "insufficient_seed_components"
            seeds = [list(node_ids)]
            K = 1

    room_of_node = _flood_fill_assign(
        node_ids=node_ids,
        seed_components=seeds,
        edge_clearance=edge_clearance,
        node_positions=node_positions,
    )
    grouped: Dict[int, List[int]] = collections.defaultdict(list)
    for nid, rid in room_of_node.items():
        grouped[int(rid)].append(int(nid))
    components = [sorted(v) for _, v in sorted(grouped.items(), key=lambda kv: kv[0])]

    doorway_debug = _collect_doorways(
        room_of_node=room_of_node,
        edge_clearance=edge_clearance,
        clearance_map=clearance_map,
        delta_star=float(delta_star),
    )

    edge_stats = {
        "min": float(np.min(edge_vals)),
        "p10": float(np.quantile(edge_vals, 0.10)),
        "p50": float(np.quantile(edge_vals, 0.50)),
        "p90": float(np.quantile(edge_vals, 0.90)),
        "max": float(np.max(edge_vals)),
    }
    clear_stats = {
        "min": float(np.min(clear_vals)),
        "p10": float(np.quantile(clear_vals, 0.10)),
        "p50": float(np.quantile(clear_vals, 0.50)),
        "p90": float(np.quantile(clear_vals, 0.90)),
        "max": float(np.max(clear_vals)),
    }
    cluster_info = {
        "method": "hydralite_edge_filtration",
        "mode": str(mode),
        "fallback": int(fallback),
        "fallback_reason": str(fallback_reason),
        "delta_star": float(delta_star),
        "K": int(len(components)),
        "seed_K": int(K),
        "plateau_edges": int(plateau_edges),
        "plateau_delta": float(plateau_delta),
        "deltas": [float(r["delta"]) for r in records],
        "Ks": [int(r["K"]) for r in records],
        "scores": [float(best_plateau.get("score", 0.0)) if i == int(best_plateau.get("start_idx", -1)) else 0.0 for i, _ in enumerate(records)],
        "sizes": [int(len(c)) for c in components],
        "clearance_stats": clear_stats,
        "edge_clearance_stats": edge_stats,
        "plateaus": all_plateaus[:20],
        "min_dilation_m": float(min_dilation_m),
        "max_dilation_m": float(max_dilation_m),
    }
    edge_clearance_debug = {
        "sample_steps": int(max(3, sample_steps)),
        "edge_count": int(len(edge_debug_rows)),
        "stats": edge_stats,
        "edge_clearances": edge_debug_rows,
    }
    return RoomPartitionResult(
        components=components,
        room_of_node=room_of_node,
        cluster_info=cluster_info,
        edge_clearance_debug=edge_clearance_debug,
        doorway_debug=doorway_debug,
    )
