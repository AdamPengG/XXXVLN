import argparse
import collections
import json
import math
import os
import random
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import quaternion
from habitat_sim import ShortestPath

from internnav.configs.evaluator import EnvCfg
from internnav.env.habitat_env import HabitatEnv
from internnav.topo.controller import (
    BearingController,
    FollowerController,
    bearing_distance,
)
from internnav.topo.debug_capture import DebugRingBuffer
from internnav.topo.geo_descriptor import compute_scan_context
from internnav.topo.pose_graph import wrap_angle
from internnav.topo.explore import TopoClipEmbedder, TopoEmbedder
from internnav.topo.graph import TopoGraph, load_embeddings
from internnav.topo.plan import dijkstra_path
from internnav.topo.retrieve_goal import retrieve_goal_topk


def _sigmoid(x: float) -> float:
    xx = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-xx)))


def _vote_node(history: Deque[int]) -> Tuple[int, float, Dict[int, int]]:
    counts = collections.Counter(history)
    node_id, count = counts.most_common(1)[0]
    conf = count / max(1, len(history))
    return node_id, conf, dict(counts)


def _build_idx_maps(node_meta: Dict) -> Tuple[Dict[int, int], Dict[int, int]]:
    idx_to_node = {}
    node_to_idx = {}
    for node_id_str, meta in node_meta.items():
        node_id = int(node_id_str)
        idx = int(meta["embed_idx"])
        idx_to_node[idx] = node_id
        node_to_idx[node_id] = idx
    return idx_to_node, node_to_idx


def _node_yaw_from_quat_wxyz(rot_wxyz) -> float:
    q = np.quaternion(float(rot_wxyz[0]), float(rot_wxyz[1]), float(rot_wxyz[2]), float(rot_wxyz[3]))
    rot = quaternion.as_rotation_matrix(q)
    forward = rot @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return float(math.atan2(float(forward[0]), float(-forward[2])))


def _agent_yaw_from_quat(quat_obj) -> float:
    rot = quaternion.as_rotation_matrix(quat_obj)
    forward = rot @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return float(math.atan2(float(forward[0]), float(-forward[2])))


def _bearing_distance_from_yaw(curr_pos: np.ndarray, curr_yaw: float, target_pos: np.ndarray) -> Tuple[float, float]:
    delta = np.array(target_pos, dtype=np.float32) - np.array(curr_pos, dtype=np.float32)
    delta[1] = 0.0
    dist = float(np.linalg.norm(delta[[0, 2]]))
    if dist < 1e-8:
        return 0.0, 0.0
    forward = np.array([math.sin(float(curr_yaw)), 0.0, -math.cos(float(curr_yaw))], dtype=np.float32)
    right = np.array([math.cos(float(curr_yaw)), 0.0, math.sin(float(curr_yaw))], dtype=np.float32)
    f_comp = float(np.dot(delta, forward))
    r_comp = float(np.dot(delta, right))
    bearing = (math.atan2(r_comp, f_comp) + math.pi) % (2.0 * math.pi) - math.pi
    return float(bearing), float(dist)


def _nearest_node(position: np.ndarray, graph: TopoGraph) -> Tuple[int, float]:
    best_id = None
    best_dist = float("inf")
    for nid, node in graph.nodes.items():
        p = np.array(node.position, dtype=np.float32)
        d = float(np.linalg.norm(position - p))
        if d < best_dist:
            best_dist = d
            best_id = int(nid)
    return int(best_id), float(best_dist)


def _select_goal_from_topk(start_node: int, goal_topk_list, edges: Dict[int, Dict[int, float]]) -> Tuple[int, float]:
    ranked = []
    fallback_node, fallback_score = goal_topk_list[0]
    for node_id, score in goal_topk_list:
        path = dijkstra_path(edges, start_node, node_id)
        path_len = len(path)
        if path_len > 1:
            ranked.append((path_len, -score, node_id, score))
    if ranked:
        ranked.sort()
        _, _, node_id, score = ranked[0]
        return int(node_id), float(score)
    return int(fallback_node), float(fallback_score)


def _parse_query_terms(query: str, room_labels: List[str]) -> Tuple[List[str], str]:
    q = str(query).strip().lower()
    room_terms = []
    for label in room_labels:
        lbl = str(label).strip().lower()
        if lbl and lbl in q:
            room_terms.append(lbl)
    tmp = q
    for t in room_terms:
        tmp = tmp.replace(t, " ")
    tokens = [x for x in tmp.replace(",", " ").split() if x]
    stop = {
        "a",
        "an",
        "the",
        "to",
        "in",
        "on",
        "near",
        "at",
        "by",
        "of",
        "and",
        "from",
    }
    obj_tokens = [t for t in tokens if t not in stop]
    object_terms = " ".join(obj_tokens).strip()
    if not object_terms:
        object_terms = q
    return room_terms, object_terms


def _pick_recovery_turn_from_depth(depth_obs, default_turn: int = 2) -> int:
    if depth_obs is None:
        return int(default_turn)
    depth = np.asarray(depth_obs, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.size == 0:
        return int(default_turn)
    width = depth.shape[1]
    if width < 2:
        return int(default_turn)
    mid = width // 2
    left = depth[:, :mid]
    right = depth[:, mid:]
    left_mean = float(np.nanmean(left))
    right_mean = float(np.nanmean(right))
    if not np.isfinite(left_mean) or not np.isfinite(right_mean):
        return int(default_turn)
    # Turn toward the side with larger free space.
    return 2 if left_mean >= right_mean else 3


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32)
    if out.ndim == 1:
        out = out[None, :]
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-6)


def _build_transition_matrix(
    graph: TopoGraph,
    idx_to_node: Dict[int, int],
    node_to_idx: Dict[int, int],
    stay_prob: float,
) -> np.ndarray:
    n = max(0, len(idx_to_node))
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    tmat = np.zeros((n, n), dtype=np.float32)
    stay = float(np.clip(stay_prob, 0.0, 1.0))
    for idx in range(n):
        node_id = int(idx_to_node[int(idx)])
        nbrs = [int(v) for v in graph.edges.get(int(node_id), {}).keys() if int(v) in node_to_idx]
        tmat[idx, idx] += stay
        if len(nbrs) == 0:
            tmat[idx, idx] += (1.0 - stay)
            continue
        share = (1.0 - stay) / float(len(nbrs))
        for nb in nbrs:
            j = int(node_to_idx[int(nb)])
            tmat[idx, j] += float(share)
    row_sum = tmat.sum(axis=1, keepdims=True) + 1e-6
    return tmat / row_sum


def _belief_entropy_norm(belief: np.ndarray) -> float:
    if belief.size <= 1:
        return 0.0
    p = np.clip(belief.astype(np.float64), 1e-12, 1.0)
    ent = -float(np.sum(p * np.log(p)))
    ent_max = float(np.log(float(belief.size)))
    if ent_max <= 1e-9:
        return 0.0
    return float(np.clip(ent / ent_max, 0.0, 1.0))


def _build_node_to_room_map(graph: TopoGraph, place2room: Dict[int, int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    anchors = [int(nid) for nid in graph.nodes.keys() if int(nid) in place2room]
    if len(anchors) == 0:
        return out
    anchor_pos = np.stack(
        [np.asarray(graph.nodes[int(nid)].position, dtype=np.float32) for nid in anchors],
        axis=0,
    )
    for nid, node in graph.nodes.items():
        ni = int(nid)
        if ni in place2room:
            out[ni] = int(place2room[ni])
            continue
        pos = np.asarray(node.position, dtype=np.float32)
        d = np.linalg.norm(anchor_pos - pos[None, :], axis=1)
        ref = int(anchors[int(np.argmin(d))])
        out[ni] = int(place2room.get(ref, -1))
    return out


def _load_doorway_sets(room_graph: Dict) -> Tuple[set, set]:
    doorway = room_graph.get("doorway_debug", {})
    edge_rows = doorway.get("doorway_edges", [])
    node_rows = doorway.get("doorway_nodes", [])
    edge_set = set()
    for row in edge_rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        try:
            a = int(row[0])
            b = int(row[1])
        except Exception:
            continue
        if a <= b:
            edge_set.add((a, b))
        else:
            edge_set.add((b, a))
    node_set = set()
    for row in node_rows:
        try:
            node_set.add(int(row))
        except Exception:
            continue
    return edge_set, node_set


def _load_room_graph(graph_json: str) -> Dict:
    room_graph_path = os.path.join(os.path.dirname(graph_json), "room_graph.json")
    if os.path.isfile(room_graph_path):
        try:
            with open(room_graph_path, "r") as f:
                return json.load(f)
        except Exception:
            return {"rooms": {}, "edges": [], "label_room_counts": {}}
    return {"rooms": {}, "edges": [], "label_room_counts": {}}


def _extract_collision_flag(info: Optional[Dict]) -> Optional[bool]:
    if not isinstance(info, dict):
        return None
    if "collided" in info:
        try:
            return bool(info.get("collided"))
        except Exception:
            return None
    col = info.get("collisions")
    if isinstance(col, dict):
        if "is_collision" in col:
            try:
                return bool(col.get("is_collision"))
            except Exception:
                return None
    return None


def _geodesic_distance(sim, start_pos: np.ndarray, goal_pos: np.ndarray) -> float:
    try:
        sp = ShortestPath()
        sp.requested_start = np.asarray(start_pos, dtype=np.float32)
        sp.requested_end = np.asarray(goal_pos, dtype=np.float32)
        ok = bool(sim.pathfinder.find_path(sp))
        if not ok:
            return float("inf")
        return float(sp.geodesic_distance)
    except Exception:
        return float("inf")


def _run_oracle_probe(
    sim,
    start_pos: np.ndarray,
    start_rot,
    goal_pos: np.ndarray,
    max_steps: int,
    node_reach_thresh: float,
) -> Dict[str, object]:
    result = {
        "oracle_available": 0,
        "oracle_success": 0,
        "oracle_geodesic_start": float("inf"),
        "oracle_geodesic_final": float("inf"),
        "oracle_steps_used": 0,
    }
    if sim is None:
        return result
    goal = np.asarray(goal_pos, dtype=np.float32)
    start = np.asarray(start_pos, dtype=np.float32)
    result["oracle_geodesic_start"] = _geodesic_distance(sim, start, goal)
    try:
        backup = sim.get_agent_state()
        sim.set_agent_state(start, start_rot, reset_sensors=False)
        follower = FollowerController(sim)
        steps = 0
        while steps < int(max_steps):
            st = sim.get_agent_state()
            cur = np.asarray(st.position, dtype=np.float32)
            if float(np.linalg.norm(cur - goal)) <= float(node_reach_thresh):
                result["oracle_success"] = 1
                break
            action = int(follower.act(goal))
            if action == 0:
                break
            sim.step(action)
            steps += 1
        st2 = sim.get_agent_state()
        fin = np.asarray(st2.position, dtype=np.float32)
        result["oracle_geodesic_final"] = _geodesic_distance(sim, fin, goal)
        if float(np.linalg.norm(fin - goal)) <= float(node_reach_thresh):
            result["oracle_success"] = 1
        result["oracle_steps_used"] = int(steps)
        result["oracle_available"] = 1
        sim.set_agent_state(backup.position, backup.rotation, reset_sensors=False)
    except Exception:
        result["oracle_available"] = 0
    return result


def _load_object_graph(graph_json: str, object_graph_json: str = "", object_nodes_json: str = "") -> Tuple[Dict, List[Dict]]:
    base = os.path.dirname(graph_json)
    graph_path = object_graph_json if object_graph_json else os.path.join(base, "object_graph.json")
    nodes_path = object_nodes_json if object_nodes_json else os.path.join(base, "object_nodes.json")
    object_graph = {"objects": {}, "edges_observed_by": [], "edges_contained_in": []}
    object_nodes: List[Dict] = []
    if os.path.isfile(graph_path):
        try:
            with open(graph_path, "r") as f:
                object_graph = json.load(f)
        except Exception:
            object_graph = {"objects": {}, "edges_observed_by": [], "edges_contained_in": []}
    if os.path.isfile(nodes_path):
        try:
            with open(nodes_path, "r") as f:
                payload = json.load(f)
            object_nodes = payload.get("objects", [])
        except Exception:
            object_nodes = []
    if len(object_nodes) == 0 and isinstance(object_graph.get("objects", {}), dict):
        object_nodes = list(object_graph.get("objects", {}).values())
    return object_graph, object_nodes


def _build_room_prompt_embeds(
    clip_embedder: TopoClipEmbedder,
    room_labels: List[str],
) -> Tuple[List[str], np.ndarray]:
    uniq = []
    seen = set()
    for label in room_labels:
        key = str(label).strip().lower()
        if key and key not in seen:
            seen.add(key)
            uniq.append(key)
    if not uniq:
        return [], np.zeros((0, 1), dtype=np.float32)
    rows = [clip_embedder.embed_text(lbl).astype(np.float32) for lbl in uniq]
    arr = np.stack(rows, axis=0)
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-6)
    return uniq, arr


def _refresh_index_maps(node_meta: Dict) -> Tuple[Dict[int, int], Dict[int, int]]:
    idx_to_node, node_to_idx = _build_idx_maps(node_meta)
    return idx_to_node, node_to_idx


def _refresh_embed_norm(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.float32)
    return _normalize_rows(arr.astype(np.float32))


def _score_rooms_for_query(
    query: str,
    query_embed: np.ndarray,
    room_graph: Dict,
    room_prompt_labels: List[str],
    room_prompt_embeds: np.ndarray,
    room_shortlist_k: int,
    room_min: float,
) -> Tuple[List[Tuple[int, str, float]], bool]:
    rooms = room_graph.get("rooms", {}) if isinstance(room_graph, dict) else {}
    if not rooms:
        return [], True
    q = query_embed.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-6)
    room_terms, _ = _parse_query_terms(query, room_prompt_labels)
    room_ranked: List[Tuple[int, str, float]] = []
    for room_id_str, room_info in rooms.items():
        room_id = int(room_id_str)
        label = str(room_info.get("label", "unknown")).strip().lower()
        centroid_embed = np.asarray(room_info.get("mean_embed", []), dtype=np.float32)
        score_centroid = 0.0
        if centroid_embed.ndim == 1 and centroid_embed.size > 0:
            centroid_embed = centroid_embed / (np.linalg.norm(centroid_embed) + 1e-6)
            score_centroid = float(np.dot(centroid_embed, q))
        score_label = 0.0
        if room_prompt_embeds.size > 0 and label:
            try:
                idx = room_prompt_labels.index(label)
                score_label = float(np.dot(room_prompt_embeds[idx], q))
            except ValueError:
                score_label = 0.0
        term_bonus = 0.2 if label in room_terms else 0.0
        room_score = 0.55 * score_centroid + 0.35 * score_label + term_bonus
        room_ranked.append((room_id, label, float(room_score)))
    room_ranked.sort(key=lambda x: x[2], reverse=True)
    topk = room_ranked[: max(1, int(room_shortlist_k))]
    fallback = (len(topk) == 0) or (float(topk[0][2]) < float(room_min))
    return topk, fallback


def _rank_objects_for_query(
    query: str,
    object_embed: np.ndarray,
    object_nodes: List[Dict],
    room_ids: Optional[set],
    object_topk: int = 5,
) -> List[Tuple[int, str, float, int, List[int]]]:
    out: List[Tuple[int, str, float, int, List[int]]] = []
    q = object_embed.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-6)
    q_tokens = set(str(query).lower().replace(",", " ").split())
    for obj in object_nodes:
        obj_id = int(obj.get("object_id", -1))
        if obj_id < 0:
            continue
        room_id = int(obj.get("room_id", -1))
        if room_ids is not None and len(room_ids) > 0 and room_id not in room_ids:
            continue
        emb = np.asarray(obj.get("embed", []), dtype=np.float32)
        score = 0.0
        if emb.ndim == 1 and emb.size > 0:
            emb = emb / (np.linalg.norm(emb) + 1e-6)
            score = float(np.dot(emb, q))
        label = str(obj.get("label", "unknown")).strip().lower()
        if label and label in q_tokens:
            score += 0.12
        elif label and any(tok in label for tok in q_tokens if len(tok) >= 3):
            score += 0.05
        support = [int(x) for x in obj.get("supporting_places", [])]
        out.append((obj_id, label, float(score), room_id, support))
    out.sort(key=lambda x: x[2], reverse=True)
    return out[: max(1, int(object_topk))]


def _select_goal_from_objects(
    start_node: int,
    object_topk: List[Tuple[int, str, float, int, List[int]]],
    edges: Dict[int, Dict[int, float]],
    fallback_goal_topk: List[Tuple[int, float, str]],
) -> Tuple[int, int, float, str]:
    ranked = []
    for obj_id, _, score, room_id, supports in object_topk:
        for place_id in supports:
            path = dijkstra_path(edges, start_node, int(place_id))
            if len(path) == 0:
                continue
            ranked.append((len(path), -float(score), int(place_id), int(obj_id), float(score), int(room_id)))
    if len(ranked) > 0:
        ranked.sort()
        _, _, goal_place, goal_obj, goal_score, goal_room_id = ranked[0]
        return int(goal_place), int(goal_obj), float(goal_score), str(goal_room_id)
    if len(fallback_goal_topk) > 0:
        goal_place = int(fallback_goal_topk[0][0])
        goal_score = float(fallback_goal_topk[0][1])
        goal_room = str(fallback_goal_topk[0][2])
        return goal_place, -1, goal_score, goal_room
    return start_node, -1, 0.0, "unknown"


def _room_first_goal_topk(
    query: str,
    query_embed: np.ndarray,
    object_embed: np.ndarray,
    clip_embeds: np.ndarray,
    idx_to_node: Dict[int, int],
    node_meta: Dict,
    room_graph: Dict,
    room_shortlist_k: int,
    goal_topk: int,
    room_min: float,
    room_prompt_labels: List[str],
    room_prompt_embeds: np.ndarray,
) -> Tuple[List[Tuple[int, float, str]], List[Tuple[int, str, float]], bool]:
    global_topk = retrieve_goal_topk(query_embed, clip_embeds, idx_to_node, topk=goal_topk)
    global_topk_with_room = [
        (
            int(node_id),
            float(score),
            str(node_meta.get(str(int(node_id)), {}).get("room_label", "unknown")),
        )
        for node_id, score in global_topk
    ]
    rooms = room_graph.get("rooms", {}) if isinstance(room_graph, dict) else {}
    if not rooms:
        return global_topk_with_room, [], True

    q = query_embed.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-6)
    room_terms, object_terms = _parse_query_terms(query, room_prompt_labels)
    room_ranked: List[Tuple[int, str, float]] = []

    for room_id_str, room_info in rooms.items():
        room_id = int(room_id_str)
        label = str(room_info.get("label", "unknown")).strip().lower()
        centroid_embed = np.asarray(room_info.get("mean_embed", []), dtype=np.float32)
        score_centroid = 0.0
        if centroid_embed.ndim == 1 and centroid_embed.size > 0:
            centroid_embed = centroid_embed / (np.linalg.norm(centroid_embed) + 1e-6)
            score_centroid = float(np.dot(centroid_embed, q))

        score_label = 0.0
        if room_prompt_embeds.size > 0 and label:
            try:
                idx = room_prompt_labels.index(label)
                score_label = float(np.dot(room_prompt_embeds[idx], q))
            except ValueError:
                score_label = 0.0
        term_bonus = 0.0
        if label in room_terms:
            term_bonus = 0.20
        room_score = 0.55 * score_centroid + 0.35 * score_label + term_bonus
        room_ranked.append((room_id, label, float(room_score)))

    room_ranked.sort(key=lambda x: x[2], reverse=True)
    room_topk = room_ranked[: max(1, int(room_shortlist_k))]
    if len(room_topk) == 0:
        return global_topk_with_room, room_ranked, True
    top_room_score = float(room_topk[0][2])
    if top_room_score < float(room_min):
        return global_topk_with_room, room_topk, True

    allowed_nodes = set()
    for room_id, _, _ in room_topk:
        room = rooms.get(str(room_id), {})
        for nid in room.get("node_ids", []):
            allowed_nodes.add(int(nid))

    if not allowed_nodes:
        return global_topk_with_room, room_topk, True

    shortlist_idx = []
    shortlist_idx_to_node = {}
    for embed_idx, node_id in idx_to_node.items():
        if int(node_id) in allowed_nodes:
            shortlist_idx_to_node[len(shortlist_idx)] = int(node_id)
            shortlist_idx.append(int(embed_idx))
    if len(shortlist_idx) == 0:
        return global_topk_with_room, room_topk, True

    shortlist_embeds = clip_embeds[shortlist_idx].astype(np.float32)
    shortlist_embeds = shortlist_embeds / (np.linalg.norm(shortlist_embeds, axis=1, keepdims=True) + 1e-6)
    q_vec = query_embed.astype(np.float32)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-6)
    o_vec = object_embed.astype(np.float32)
    o_vec = o_vec / (np.linalg.norm(o_vec) + 1e-6)
    q_sims = shortlist_embeds.dot(q_vec)
    o_sims = shortlist_embeds.dot(o_vec)
    if object_terms.strip():
        combined = 0.55 * q_sims + 0.45 * o_sims
    else:
        combined = q_sims

    k = min(int(goal_topk), combined.shape[0])
    top_idx = np.argpartition(-combined, k - 1)[:k]
    top_idx = top_idx[np.argsort(-combined[top_idx])]
    goal_topk_with_room = []
    for i in top_idx:
        node_id = int(shortlist_idx_to_node[int(i)])
        room = str(node_meta.get(str(node_id), {}).get("room_label", "unknown"))
        goal_topk_with_room.append((node_id, float(combined[int(i)]), room))

    if len(goal_topk_with_room) == 0:
        return global_topk_with_room, room_topk, True
    if len(global_topk_with_room) > 0 and (goal_topk_with_room[0][1] + 0.03) < global_topk_with_room[0][1]:
        return global_topk_with_room, room_topk, True
    return goal_topk_with_room, room_topk, False


def _add_node_online(
    graph: TopoGraph,
    node_meta: Dict,
    obs,
    embedder: TopoEmbedder,
    clip_embedder: TopoClipEmbedder,
    vision_prompt: str,
    min_node_spacing: float,
    room_prompt_labels: List[str],
    room_prompt_embeds: np.ndarray,
    geo_bins_r: int,
    geo_bins_theta: int,
    geo_max_depth: float,
    max_nodes: int,
    clip_embeds_list: List[np.ndarray],
    geo_embeds_list: List[np.ndarray],
    save_frame_dir: str = "",
) -> Tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
    rgb = obs.get("rgb")
    if rgb is None:
        return None, None, None, None, 0
    state = obs.get("_agent_state")
    if state is None:
        return None, None, None, None, 0
    depth = obs.get("depth")
    if depth is None:
        depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
    position = [float(state.position[0]), float(state.position[1]), float(state.position[2])]
    rotation = [float(state.rotation.w), float(state.rotation.x), float(state.rotation.y), float(state.rotation.z)]

    if len(graph.nodes) >= int(max_nodes):
        return None, None, None, None, 0

    nearest_id = None
    nearest_dist = float("inf")
    for nid, node in graph.nodes.items():
        d = float(np.linalg.norm(np.array(node.position, dtype=np.float32) - np.array(position, dtype=np.float32)))
        if d < nearest_dist:
            nearest_dist = d
            nearest_id = int(nid)

    if nearest_dist < float(min_node_spacing):
        return None, None, None, None, 0

    emb = embedder.embed_image(rgb, instruction=vision_prompt).astype(np.float16)
    clip_emb = clip_embedder.embed_image(rgb).astype(np.float16)
    geo_emb = compute_scan_context(
        depth=depth,
        bins_r=geo_bins_r,
        bins_theta=geo_bins_theta,
        max_depth=geo_max_depth,
    ).astype(np.float16)
    room_label = "unknown"
    room_score = 0.0
    room_top3 = []
    if room_prompt_embeds.size > 0:
        clip_f = clip_emb.astype(np.float32)
        clip_f = clip_f / (np.linalg.norm(clip_f) + 1e-6)
        sims = room_prompt_embeds.dot(clip_f)
        topk = min(3, sims.shape[0])
        idx = np.argpartition(-sims, topk - 1)[:topk]
        idx = idx[np.argsort(-sims[idx])]
        room_top3 = [(room_prompt_labels[int(i)], float(sims[int(i)])) for i in idx]
        if room_top3:
            room_label, room_score = room_top3[0]

    new_node_id = int(max(graph.nodes.keys()) + 1 if graph.nodes else 0)
    embed_idx = int(
        max([int(meta.get("embed_idx", -1)) for meta in node_meta.values()] + [-1]) + 1
    )
    rgb_path = None
    if save_frame_dir:
        os.makedirs(save_frame_dir, exist_ok=True)
        rgb_path = os.path.join(save_frame_dir, f"fill_node_{new_node_id:04d}.png")
        try:
            from PIL import Image

            Image.fromarray(rgb.astype(np.uint8)).save(rgb_path)
        except Exception:
            rgb_path = None

    graph.add_node(
        node_id=new_node_id,
        step_idx=-1,
        position=position,
        rotation=rotation,
        embed_idx=embed_idx,
        rgb_path=rgb_path,
    )
    if nearest_id is not None:
        w = float(
            np.linalg.norm(
                np.array(graph.nodes[nearest_id].position, dtype=np.float32)
                - np.array(position, dtype=np.float32)
            )
        )
        graph.add_edge(nearest_id, new_node_id, w, bidir=True, edge_type="temporal")

    node_meta[str(new_node_id)] = {
        "embed_idx": embed_idx,
        "clip_embed_idx": embed_idx,
        "geo_idx": embed_idx,
        "position": position,
        "rotation": rotation,
        "rgb_path": rgb_path,
        "room_label": room_label,
        "room_score": float(room_score),
        "room_top3": [[str(lbl), float(score)] for lbl, score in room_top3],
    }
    return new_node_id, emb, clip_emb, geo_emb, 1 if nearest_id is not None else 0


def run_navigation(
    config_path: str,
    graph_json: str,
    embeds_npy: str,
    node_meta_path: str,
    query: str,
    out_dir: str,
    max_steps: int,
    model_path: str,
    pooling: str,
    vote_window: int,
    relocalize_every: int,
    stop_thresh: float,
    turn_thresh: float,
    node_reach_thresh: float,
    controller_mode: str,
    run_id: str,
    start_episode_offset: int = 0,
    vision_prompt: str = "",
    goal_topk: int = 5,
    loc_conf_thresh: float = 0.34,
    kidnap_start: bool = False,
    kidnap_seed: int = 7,
    kidnap_min_nearest_node_dist: float = 0.5,
    kidnap_max_nearest_node_dist: float = 4.0,
    clip_embeds_npy: str = "runs/topo_mvp/node_embeds_clip.npy",
    geo_embeds_npy: str = "runs/topo_mvp/node_geo.npy",
    object_graph_json: str = "",
    object_nodes_json: str = "",
    clip_model_name: str = "openai/clip-vit-base-patch32",
    use_two_stage_goal: bool = False,
    room_shortlist_k: int = 12,
    object_topk: int = 8,
    room_min: float = 0.25,
    stall_window: int = 8,
    stall_min_delta: float = 0.08,
    recovery_mode: str = "depth_turn",
    recovery_turn_steps: int = 3,
    recovery_follower_steps: int = 4,
    recovery_cooldown: int = 5,
    goal_score_thresh: float = 0.0,
    fill_steps: int = 120,
    fill_min_node_spacing: float = 0.35,
    fill_relocalize_low_conf_patience: int = 6,
    fill_retry_nav_extra_steps: int = 80,
    room_query_topk: int = 5,
    loc_fuse_alpha: float = 0.6,
    loc_prior_alpha: float = 0.72,
    loc_prior_sigma: float = 2.2,
    loc_conf_margin_center: float = 0.02,
    loc_conf_gain: float = 12.0,
    belief_temp: float = 0.20,
    belief_stay_prob: float = 0.70,
    belief_mix: float = 0.85,
    belief_entropy_fill_thresh: float = 0.78,
    belief_topk_print: int = 4,
    belief_hop_radius: int = 2,
    belief_geo_radius: float = 3.5,
    belief_noncand_penalty: float = 1.0,
    belief_reset_low_conf: float = 0.18,
    belief_reset_high_entropy: float = 0.98,
    geo_bins_r: int = 20,
    geo_bins_theta: int = 60,
    geo_max_depth: float = 5.0,
    fill_rotate_steps: int = 8,
    fill_forward_steps: int = 12,
    max_nodes: int = 30,
    stop_guard_dist: float = 0.40,
    stop_near_m: float = 1.25,
    stop_near_hold_steps: int = 8,
    stop_near_goal_prob: float = 0.25,
    stop_near_goal_score_min: float = 0.05,
    stop_near_low_conf_guard: float = 0.18,
    doorway_burst_steps: int = 0,
    use_est_pose: bool = False,
    odom_noise_trans: float = 0.02,
    odom_noise_yaw_deg: float = 1.0,
    debug_capture: bool = False,
    debug_only_on_fail: bool = True,
    debug_frame_stride: int = 3,
    debug_ringbuf_steps: int = 220,
    debug_save_depth: bool = False,
    debug_oracle_on_fail: bool = True,
    debug_capture_success_at_maxsteps: bool = True,
    debug_root: str = "",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cond_name = str(os.environ.get("TOPO_RUN_CONDITION", "")).strip()
    if not cond_name:
        parent = os.path.basename(os.path.dirname(out_dir))
        if parent.startswith("condition_"):
            cond_name = parent.replace("condition_", "", 1)
    if not debug_root:
        debug_root = os.path.join(os.path.dirname(os.path.dirname(out_dir)), "debug_runs")
    debug_run_dir = os.path.join(debug_root, cond_name if cond_name else "unknown", run_id)
    debug_buf = DebugRingBuffer(
        enabled=bool(debug_capture),
        ring_steps=int(debug_ringbuf_steps),
        frame_stride=int(debug_frame_stride),
        save_depth=bool(debug_save_depth),
    )
    if bool(debug_capture):
        print(
            f"[TOPO_DEBUG_CAPTURE] run={run_id} only_on_fail={int(bool(debug_only_on_fail))} "
            f"ringbuf={int(debug_ringbuf_steps)} stride={int(debug_frame_stride)}",
            flush=True,
        )

    from habitat_baselines.config.default import get_config as get_habitat_config

    habitat_config = get_habitat_config(config_path)
    env_cfg = EnvCfg(
        env_type="habitat",
        env_settings={
            "habitat_config": habitat_config,
            "output_path": out_dir,
        },
    )
    env = HabitatEnv(env_cfg)

    graph = TopoGraph.load_json(graph_json)
    embeds = load_embeddings(embeds_npy)
    if embeds.ndim == 1:
        embeds = embeds[:, None]
    embeds_list = [embeds[i].astype(np.float16) for i in range(embeds.shape[0])]
    embeds_norm = _refresh_embed_norm(embeds)
    scene_id = graph.meta.get("scene_id")
    with open(node_meta_path, "r") as f:
        meta_blob = json.load(f)
    node_meta = meta_blob.get("node_meta", {})
    idx_to_node, node_to_idx = _refresh_index_maps(node_meta)
    def _refresh_node_pos_by_embed() -> np.ndarray:
        if len(idx_to_node) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        max_idx = max(int(k) for k in idx_to_node.keys())
        arr = np.zeros((max_idx + 1, 3), dtype=np.float32)
        for idx, nid in idx_to_node.items():
            arr[int(idx)] = np.asarray(graph.nodes[int(nid)].position, dtype=np.float32)
        return arr

    node_pos_by_embed = _refresh_node_pos_by_embed()
    clip_embeds = load_embeddings(clip_embeds_npy)
    if clip_embeds.ndim == 1:
        clip_embeds = clip_embeds[:, None]
    clip_embeds_list = [clip_embeds[i].astype(np.float16) for i in range(clip_embeds.shape[0])]
    clip_embeds_norm = _refresh_embed_norm(clip_embeds)
    if os.path.isfile(geo_embeds_npy):
        geo_embeds = load_embeddings(geo_embeds_npy)
        if geo_embeds.ndim == 1:
            geo_embeds = geo_embeds[:, None]
    else:
        geo_dim = int(geo_bins_r) * int(geo_bins_theta)
        geo_embeds = np.zeros((clip_embeds.shape[0], geo_dim), dtype=np.float16)
    if geo_embeds.shape[0] != clip_embeds.shape[0]:
        geo_dim = int(geo_bins_r) * int(geo_bins_theta)
        fixed = np.zeros((clip_embeds.shape[0], geo_dim), dtype=np.float16)
        m = min(fixed.shape[0], geo_embeds.shape[0])
        n = min(fixed.shape[1], geo_embeds.shape[1])
        fixed[:m, :n] = geo_embeds[:m, :n]
        geo_embeds = fixed
    geo_embeds_list = [geo_embeds[i].astype(np.float16) for i in range(geo_embeds.shape[0])]
    geo_embeds_norm = _refresh_embed_norm(geo_embeds)

    embedder = TopoEmbedder(model_path=model_path, pooling=pooling)
    clip_embedder = TopoClipEmbedder(model_name=clip_model_name)
    room_graph = _load_room_graph(graph_json)
    place2room = {
        int(k): int(v)
        for k, v in room_graph.get("place2room", {}).items()
    }
    doorway_edge_set, doorway_node_set = _load_doorway_sets(room_graph)
    node_to_room_path = os.path.join(os.path.dirname(graph_json), "node_to_room.json")
    node_to_room = {}
    if os.path.isfile(node_to_room_path):
        try:
            payload = json.load(open(node_to_room_path, "r"))
            node_to_room = {int(k): int(v) for k, v in payload.get("node_to_room", {}).items()}
        except Exception:
            node_to_room = {}
    if len(node_to_room) == 0:
        node_to_room = _build_node_to_room_map(graph, place2room)
        with open(node_to_room_path, "w") as f:
            json.dump(
                {
                    "method": "nearest_place",
                    "graph_nodes": int(len(graph.nodes)),
                    "mapped": int(len(node_to_room)),
                    "node_to_room": {str(k): int(v) for k, v in sorted(node_to_room.items())},
                },
                f,
                indent=2,
            )
    print(
        f"[TOPO_NODE_TO_ROOM] scene={graph.meta.get('scene_id', 'unknown')} "
        f"graph_nodes={len(graph.nodes)} mapped={len(node_to_room)} method=nearest_place",
        flush=True,
    )
    if object_graph_json == "":
        object_graph_json = os.path.join(os.path.dirname(graph_json), "object_graph.json")
    if object_nodes_json == "":
        object_nodes_json = os.path.join(os.path.dirname(graph_json), "object_nodes.json")
    object_graph, object_nodes = _load_object_graph(
        graph_json,
        object_graph_json=object_graph_json,
        object_nodes_json=object_nodes_json,
    )
    room_labels_from_graph = sorted(
        {
            str(room_info.get("label", "unknown")).strip().lower()
            for room_info in room_graph.get("rooms", {}).values()
            if str(room_info.get("label", "")).strip()
        }
    )
    room_prompt_labels, room_prompt_embeds = _build_room_prompt_embeds(clip_embedder, room_labels_from_graph)
    belief_tmat = _build_transition_matrix(
        graph=graph,
        idx_to_node=idx_to_node,
        node_to_idx=node_to_idx,
        stay_prob=float(belief_stay_prob),
    )
    if belief_tmat.shape[0] > 0:
        belief = np.ones((belief_tmat.shape[0],), dtype=np.float32) / float(belief_tmat.shape[0])
    else:
        belief = np.zeros((0,), dtype=np.float32)
    last_belief_entropy = 1.0
    print(
        f"[TOPO_LOC_BELIEF_CFG] temp={float(belief_temp):.3f} stay={float(belief_stay_prob):.3f} "
        f"mix={float(belief_mix):.3f} entropy_fill_thresh={float(belief_entropy_fill_thresh):.3f}",
        flush=True,
    )

    if scene_id:
        filtered = [ep for ep in env.episodes if scene_id in ep.scene_id]
        env.episodes = filtered
        if len(filtered) > 0:
            env._current_episode_index = int(start_episode_offset) % len(filtered)
        else:
            env._current_episode_index = 0
        env.is_running = len(filtered) > 0
        print(
            f"[TOPO_NAV] scene_filter={scene_id} episodes={len(filtered)} "
            f"start_episode_offset={start_episode_offset}",
            flush=True,
        )

    # Goal retrieval (room-first + object rerank, fallback to place retrieval).
    text_embed = clip_embedder.embed_text(query)
    room_terms, object_terms = _parse_query_terms(query, room_prompt_labels)
    object_embed = clip_embedder.embed_text(object_terms)
    room_topk, room_fallback = _score_rooms_for_query(
        query=query,
        query_embed=text_embed,
        room_graph=room_graph,
        room_prompt_labels=room_prompt_labels,
        room_prompt_embeds=room_prompt_embeds,
        room_shortlist_k=room_shortlist_k,
        room_min=room_min,
    )
    print(
        f"[TOPO_ROOM_QUERY] query=\"{query}\" topk_rooms={room_topk[:max(1, int(room_query_topk))]}",
        flush=True,
    )
    low_room_conf = bool(room_fallback)
    if room_fallback:
        top_room_score = float(room_topk[0][2]) if len(room_topk) > 0 else 0.0
        print(
            f"[TOPO_ROOM_FALLBACK] reason=low_room_conf query=\"{query}\" top_room_score={top_room_score:.4f} "
            f"room_min={float(room_min):.4f}",
            flush=True,
        )
    shortlist_room_ids = None if room_fallback else {int(rid) for rid, _, _ in room_topk}
    object_topk_list = _rank_objects_for_query(
        query=object_terms if object_terms.strip() else query,
        object_embed=object_embed,
        object_nodes=object_nodes,
        room_ids=shortlist_room_ids,
        object_topk=object_topk,
    )
    room_topk_list = list(room_topk)
    print(
        f"[TOPO_OBJECT_TOPK] query=\"{query}\" topk={[(oid, lbl, round(sc, 4), rid) for oid, lbl, sc, rid, _ in object_topk_list[:max(1, int(goal_topk))]]}",
        flush=True,
    )

    # Fallback place-level retrieval list.
    place_goal_topk = retrieve_goal_topk(text_embed, clip_embeds_norm, idx_to_node, topk=max(goal_topk, 8))
    place_goal_topk_room = [
        (
            int(node_id),
            float(score),
            str(node_meta.get(str(int(node_id)), {}).get("room_label", "unknown")),
        )
        for node_id, score in place_goal_topk
    ]
    if len(place_goal_topk_room) == 0:
        env.close()
        raise RuntimeError("Goal retrieval failed: empty top-k list.")
    goal_candidates_preview = [
        (int(pid), round(float(sc), 4), str(room), int(oid))
        for pid, sc, room, oid in [
            (
                int(sup[0]),
                float(score),
                str(room_id),
                int(obj_id),
            )
            for obj_id, _, score, room_id, sup in object_topk_list
            if len(sup) > 0
        ][: max(1, int(goal_topk))]
    ]
    if len(goal_candidates_preview) == 0:
        goal_candidates_preview = [(int(n), round(float(s), 4), str(r), -1) for n, s, r in place_goal_topk_room[:max(1, int(goal_topk))]]
    print(f"[TOPO_GOAL_TOPK] query=\"{query}\" topk={goal_candidates_preview}", flush=True)

    goal_topk_list = [(int(n), float(s)) for n, s, _ in place_goal_topk_room]
    goal_node, goal_score, goal_room = place_goal_topk_room[0]
    goal_object_id = -1
    goal_selected = False

    obs = env.reset()
    if obs is None or not env.is_running:
        env.close()
        raise RuntimeError("No episode available for navigation.")

    start_pose = None
    if kidnap_start:
        sim = env._env.sim
        rng = np.random.default_rng(int(kidnap_seed))
        # Habitat pathfinder RNG is deterministic across processes; advance draws for per-run diversity.
        advance_draws = int(kidnap_seed) % 37
        for _ in range(advance_draws):
            _ = sim.pathfinder.get_random_navigable_point()
        placed = False
        for _ in range(80):
            start_pos = np.array(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            near_node, near_dist = _nearest_node(start_pos, graph)
            if near_dist < kidnap_min_nearest_node_dist or near_dist > kidnap_max_nearest_node_dist:
                continue
            yaw = float(rng.uniform(-math.pi, math.pi))
            rot = np.quaternion(float(math.cos(yaw / 2.0)), 0.0, float(math.sin(yaw / 2.0)), 0.0)
            sim.set_agent_state(start_pos, rot, reset_sensors=False)
            try:
                obs = sim.get_sensor_observations()
            except Exception:
                obs = obs
            start_pose = {
                "x": float(start_pos[0]),
                "y": float(start_pos[1]),
                "z": float(start_pos[2]),
                "yaw": yaw,
                "nearest_node": int(near_node),
                "nearest_node_dist": float(near_dist),
            }
            placed = True
            break
        if not placed:
            # deterministic fallback: use current pose from reset
            st = sim.get_agent_state()
            start_pose = {
                "x": float(st.position[0]),
                "y": float(st.position[1]),
                "z": float(st.position[2]),
                "yaw": _agent_yaw_from_quat(st.rotation),
                "nearest_node": int(_nearest_node(np.array(st.position, dtype=np.float32), graph)[0]),
                "nearest_node_dist": float(_nearest_node(np.array(st.position, dtype=np.float32), graph)[1]),
            }
    else:
        st = env._env.sim.get_agent_state()
        start_pose = {
            "x": float(st.position[0]),
            "y": float(st.position[1]),
            "z": float(st.position[2]),
            "yaw": _agent_yaw_from_quat(st.rotation),
            "nearest_node": int(_nearest_node(np.array(st.position, dtype=np.float32), graph)[0]),
            "nearest_node_dist": float(_nearest_node(np.array(st.position, dtype=np.float32), graph)[1]),
        }
    oracle_start_state = env._env.sim.get_agent_state()
    oracle_start_pos = np.array(
        [float(oracle_start_state.position[0]), float(oracle_start_state.position[1]), float(oracle_start_state.position[2])],
        dtype=np.float32,
    )
    oracle_start_rot = np.quaternion(
        float(oracle_start_state.rotation.w),
        float(oracle_start_state.rotation.x),
        float(oracle_start_state.rotation.y),
        float(oracle_start_state.rotation.z),
    )

    st_est = env._env.sim.get_agent_state()
    est_pos = np.array([float(st_est.position[0]), float(st_est.position[1]), float(st_est.position[2])], dtype=np.float32)
    est_yaw = float(_agent_yaw_from_quat(st_est.rotation))
    odom_rng = random.Random(int(kidnap_seed) + 991)
    odom_noise_yaw = float(np.deg2rad(float(odom_noise_yaw_deg)))

    history: Deque[int] = collections.deque(maxlen=vote_window)
    step = 0
    done = False
    step_budget = int(max_steps)
    last_vote_node = None
    last_conf = 0.0
    trusted_node = None
    path_nodes = []
    path_index = 0
    next_node = goal_node

    follower = None
    bearing_ctrl = None
    if controller_mode == "follower":
        follower = FollowerController(env._env.sim)
    else:
        bearing_ctrl = BearingController(
            stop_thresh=stop_thresh,
            turn_high=turn_thresh,
            turn_low=turn_thresh * 0.6,
        )

    action_hist = {"STOP": 0, "FWD": 0, "LEFT": 0, "RIGHT": 0}
    room_id_seq: List[int] = []
    room_transitions: List[Dict[str, object]] = []
    prev_room_id: Optional[int] = None
    fail_reason = None
    terminated_by: Optional[str] = None
    success_by: Optional[str] = None
    loc_conf_trace = []
    belief_entropy_trace: List[float] = []
    belief_debug_rows: List[Dict[str, object]] = []
    map_node_history: Deque[int] = collections.deque(maxlen=160)
    last_belief_topk: List[Tuple[int, float]] = []
    low_conf_streak = 0
    stall_window = max(2, int(stall_window))
    subgoal_dist_trace: Deque[float] = collections.deque(maxlen=stall_window)
    goal_dist_trace: Deque[float] = collections.deque(maxlen=stall_window)
    last_next_node = None
    recovery_turn_remaining = 0
    recovery_turn_action = 2
    recovery_follow_remaining = 0
    recovery_follow_to_goal = False
    recovery_forward_remaining = 0
    recovery_cooldown_remaining = 0
    recovery_goal_lock = False
    recovery_events = 0
    watchdog_trigger_count = 0
    relocalize_reset_count = 0
    doorway_burst_count = 0
    near_goal_stop_count = 0
    near_goal_stable_count = 0
    goal_prob = 0.0
    fill_triggers = 0
    fill_added_nodes_total = 0
    fill_added_edges_total = 0
    goal_burst_after = max(1, int(os.environ.get("TOPO_RECOVERY_GOAL_BURST_AFTER", "8")))
    goal_burst_steps = max(1, int(os.environ.get("TOPO_RECOVERY_GOAL_BURST_STEPS", str(max(1, int(recovery_follower_steps))))))
    goal_lock_after = max(1, int(os.environ.get("TOPO_RECOVERY_GOAL_LOCK_AFTER", "6")))
    goal_lock_min_dist = float(os.environ.get("TOPO_RECOVERY_GOAL_LOCK_MIN_DIST", "5.0"))
    stop_near_loc_conf_min = float(os.environ.get("TOPO_STOP_NEAR_LOC_CONF_MIN", "0.40"))
    fill_frame_dir = os.path.join(out_dir, "frames")
    last_loc_vision = 0.0
    last_loc_geo = 0.0
    last_loc_fused = 0.0
    last_loc_prior = 0.0
    last_loc_node = -1
    force_relocalize = False
    last_collision_flag: Optional[bool] = None

    def _localize_with_fusion(
        rgb_obs: np.ndarray,
        depth_obs: Optional[np.ndarray],
        prior_pos: np.ndarray,
    ) -> Dict[str, object]:
        img_embed = embedder.embed_image(rgb_obs, instruction=vision_prompt).astype(np.float32)
        img_embed = img_embed / (np.linalg.norm(img_embed) + 1e-6)

        if depth_obs is None:
            depth_obs = np.zeros((rgb_obs.shape[0], rgb_obs.shape[1]), dtype=np.float32)
        geo_query = compute_scan_context(
            depth=depth_obs,
            bins_r=geo_bins_r,
            bins_theta=geo_bins_theta,
            max_depth=geo_max_depth,
        ).astype(np.float32)
        geo_query = geo_query / (np.linalg.norm(geo_query) + 1e-6)

        vision_sims = embeds_norm.dot(img_embed)
        if geo_embeds_norm.shape[0] == embeds_norm.shape[0]:
            geo_sims = geo_embeds_norm.dot(geo_query)
        else:
            geo_sims = np.zeros_like(vision_sims)
        embed_sims = float(loc_fuse_alpha) * vision_sims + (1.0 - float(loc_fuse_alpha)) * geo_sims
        embed_scores = np.clip((embed_sims + 1.0) * 0.5, 0.0, 1.0)

        geo_prior = np.ones_like(embed_scores, dtype=np.float32)
        if node_pos_by_embed.shape[0] == embed_scores.shape[0] and node_pos_by_embed.shape[0] > 0:
            prior_xy = np.asarray(prior_pos, dtype=np.float32)[[0, 2]]
            node_xy = node_pos_by_embed[:, [0, 2]]
            d_xy = np.linalg.norm(node_xy - prior_xy[None, :], axis=1)
            sigma = max(1e-6, float(loc_prior_sigma))
            geo_prior = np.exp(-d_xy / sigma).astype(np.float32)
        final_scores = float(loc_prior_alpha) * embed_scores + (1.0 - float(loc_prior_alpha)) * geo_prior

        best_idx = int(np.argmax(final_scores)) if final_scores.shape[0] > 0 else -1
        if final_scores.shape[0] > 1:
            top2 = np.partition(final_scores, -2)[-2:]
            top1 = float(np.max(top2))
            second = float(np.min(top2))
        else:
            top1 = float(final_scores[best_idx]) if best_idx >= 0 else 0.0
            second = 0.0
        gap = max(0.0, top1 - second)
        conf = _sigmoid((gap - float(loc_conf_margin_center)) * float(loc_conf_gain))

        return {
            "best_idx": best_idx,
            "best_node": int(idx_to_node.get(best_idx, -1)) if best_idx >= 0 else -1,
            "conf": conf,
            "raw_top1": float(top1),
            "raw_top2": float(second),
            "gap": float(gap),
            "vision_best": float(vision_sims[best_idx]) if best_idx >= 0 else 0.0,
            "geo_best": float(geo_sims[best_idx]) if best_idx >= 0 else 0.0,
            "embed_best": float(embed_scores[best_idx]) if best_idx >= 0 else 0.0,
            "prior_best": float(geo_prior[best_idx]) if best_idx >= 0 else 0.0,
            "final_best": float(final_scores[best_idx]) if best_idx >= 0 else 0.0,
            "vote_node": int(idx_to_node.get(best_idx, -1)) if best_idx >= 0 else -1,
            "vision_scores": vision_sims,
            "geo_scores": geo_sims,
            "embed_scores": embed_scores,
            "prior_scores": geo_prior,
            "final_scores": final_scores,
        }

    def _compute_goal_candidates() -> Tuple[
        List[Tuple[int, float, str]],
        List[Tuple[int, str, float, int, List[int]]],
        List[Tuple[int, str, float]],
    ]:
        local_query_embed = clip_embedder.embed_text(query)
        _, local_object_terms = _parse_query_terms(query, room_prompt_labels)
        local_object_embed = clip_embedder.embed_text(local_object_terms)
        local_room_topk, local_room_fallback = _score_rooms_for_query(
            query=query,
            query_embed=local_query_embed,
            room_graph=room_graph,
            room_prompt_labels=room_prompt_labels,
            room_prompt_embeds=room_prompt_embeds,
            room_shortlist_k=room_shortlist_k,
            room_min=room_min,
        )
        print(
            f"[TOPO_ROOM_QUERY] query=\"{query}\" topk_rooms={local_room_topk[:max(1, int(room_query_topk))]}",
            flush=True,
        )
        if local_room_fallback:
            top_room_score = float(local_room_topk[0][2]) if len(local_room_topk) > 0 else 0.0
            print(
                f"[TOPO_ROOM_FALLBACK] reason=low_room_conf query=\"{query}\" "
                f"top_room_score={top_room_score:.4f} room_min={float(room_min):.4f}",
                flush=True,
            )
        local_room_ids = None if local_room_fallback else {int(rid) for rid, _, _ in local_room_topk}
        local_obj_topk = _rank_objects_for_query(
            query=local_object_terms if local_object_terms.strip() else query,
            object_embed=local_object_embed,
            object_nodes=object_nodes,
            room_ids=local_room_ids,
            object_topk=object_topk,
        )
        print(
            f"[TOPO_OBJECT_TOPK] query=\"{query}\" topk={[(oid, lbl, round(sc, 4), rid) for oid, lbl, sc, rid, _ in local_obj_topk[:max(1, int(goal_topk))]]}",
            flush=True,
        )
        raw = retrieve_goal_topk(local_query_embed, clip_embeds_norm, idx_to_node, topk=max(goal_topk, 8))
        local_place_topk = [
            (
                int(node_id),
                float(score),
                str(node_meta.get(str(int(node_id)), {}).get("room_label", "unknown")),
            )
            for node_id, score in raw
        ]
        preview = [
            (int(sup[0]), round(float(score), 4), str(room_id), int(obj_id))
            for obj_id, _, score, room_id, sup in local_obj_topk
            if len(sup) > 0
        ][: max(1, int(goal_topk))]
        if len(preview) == 0:
            preview = [(int(n), round(float(s), 4), str(r), -1) for n, s, r in local_place_topk[:max(1, int(goal_topk))]]
        print(f"[TOPO_GOAL_TOPK] query=\"{query}\" topk={preview}", flush=True)
        return local_place_topk, local_obj_topk, local_room_topk

    def _save_graph_artifacts() -> None:
        if len(embeds_list) == 0 or len(clip_embeds_list) == 0:
            return
        embeds_arr = np.stack(embeds_list, axis=0).astype(np.float16)
        clip_arr = np.stack(clip_embeds_list, axis=0).astype(np.float16)
        geo_arr = np.stack(geo_embeds_list, axis=0).astype(np.float16) if len(geo_embeds_list) > 0 else None
        np.save(embeds_npy, embeds_arr)
        np.save(clip_embeds_npy, clip_arr)
        if geo_arr is not None:
            np.save(geo_embeds_npy, geo_arr)
        graph.save_json(graph_json)
        with open(node_meta_path, "w") as f:
            json.dump({"meta": graph.meta, "node_meta": node_meta}, f, indent=2)

    def _run_fill_map(trigger_reason: str, score_val: float, loc_conf_val: float) -> None:
        nonlocal obs, goal_topk_list, goal_selected, goal_node, goal_score, goal_room, goal_object_id
        nonlocal place_goal_topk_room, object_topk_list, room_topk_list
        nonlocal idx_to_node, node_to_idx, embeds_norm, clip_embeds_norm, geo_embeds_norm, node_pos_by_embed
        nonlocal fill_triggers, fill_added_nodes_total, fill_added_edges_total
        nonlocal step_budget, trusted_node, last_vote_node, low_conf_streak, last_conf
        nonlocal last_loc_vision, last_loc_geo, last_loc_fused, last_loc_prior, last_loc_node
        nonlocal belief_tmat, belief, last_belief_entropy, last_belief_topk

        if controller_mode == "follower" or int(fill_steps) <= 0:
            return
        fill_triggers += 1
        target_room = "unknown"
        if len(room_topk_list) > 0:
            rid, lbl, _ = room_topk_list[0]
            target_room = f"{int(rid)}:{str(lbl)}"
        elif last_vote_node is not None:
            target_room = str(node_meta.get(str(int(last_vote_node)), {}).get("room_label", "unknown"))
        print(
            f"[TOPO_FILL_TRIGGER] reason={trigger_reason} loc_conf={float(loc_conf_val):.3f} "
            f"goal_score={float(score_val):.4f} target_room={target_room}",
            flush=True,
        )
        nodes_before = len(graph.nodes)
        edges_before = sum(1 for u, nbrs in graph.edges.items() for v in nbrs if u < v)
        added_edges_local = 0

        # Phase A: rotate in place to collect around-view descriptors.
        rotate_steps = max(4, min(int(fill_steps), int(fill_rotate_steps)))
        for turn_idx in range(rotate_steps):
            st = env._env.sim.get_agent_state()
            obs_local = dict(obs)
            obs_local["_agent_state"] = st
            new_node_id, new_emb, new_clip, new_geo, edge_incr = _add_node_online(
                graph=graph,
                node_meta=node_meta,
                obs=obs_local,
                embedder=embedder,
                clip_embedder=clip_embedder,
                vision_prompt=vision_prompt,
                min_node_spacing=float(fill_min_node_spacing),
                room_prompt_labels=room_prompt_labels,
                room_prompt_embeds=room_prompt_embeds,
                geo_bins_r=geo_bins_r,
                geo_bins_theta=geo_bins_theta,
                geo_max_depth=geo_max_depth,
                max_nodes=max_nodes,
                clip_embeds_list=clip_embeds_list,
                geo_embeds_list=geo_embeds_list,
                save_frame_dir=fill_frame_dir,
            )
            if new_node_id is not None and new_emb is not None and new_clip is not None and new_geo is not None:
                embeds_list.append(new_emb.astype(np.float16))
                clip_embeds_list.append(new_clip.astype(np.float16))
                geo_embeds_list.append(new_geo.astype(np.float16))
                added_edges_local += int(edge_incr)

            action = 2 if (turn_idx % 2 == 0) else 3
            obs, _, fill_done, _ = env.step(action)
            if fill_done:
                obs = env.reset()
                if obs is None or not env.is_running:
                    break

        # Phase B: short forward burst with deterministic stuck recovery.
        forward_steps = max(0, min(int(fill_forward_steps), int(fill_steps) - rotate_steps))
        stuck_fwd = 0
        for f_idx in range(forward_steps):
            st = env._env.sim.get_agent_state()
            obs_local = dict(obs)
            obs_local["_agent_state"] = st
            new_node_id, new_emb, new_clip, new_geo, edge_incr = _add_node_online(
                graph=graph,
                node_meta=node_meta,
                obs=obs_local,
                embedder=embedder,
                clip_embedder=clip_embedder,
                vision_prompt=vision_prompt,
                min_node_spacing=float(fill_min_node_spacing),
                room_prompt_labels=room_prompt_labels,
                room_prompt_embeds=room_prompt_embeds,
                geo_bins_r=geo_bins_r,
                geo_bins_theta=geo_bins_theta,
                geo_max_depth=geo_max_depth,
                max_nodes=max_nodes,
                clip_embeds_list=clip_embeds_list,
                geo_embeds_list=geo_embeds_list,
                save_frame_dir=fill_frame_dir,
            )
            if new_node_id is not None and new_emb is not None and new_clip is not None and new_geo is not None:
                embeds_list.append(new_emb.astype(np.float16))
                clip_embeds_list.append(new_clip.astype(np.float16))
                geo_embeds_list.append(new_geo.astype(np.float16))
                added_edges_local += int(edge_incr)

            if stuck_fwd >= 2:
                action = _pick_recovery_turn_from_depth(obs.get("depth"), default_turn=2 if (f_idx % 2 == 0) else 3)
                stuck_fwd = 0
            else:
                action = 1

            pre_pos = np.array(st.position, dtype=np.float32)
            obs, _, fill_done, _ = env.step(action)
            post_pos = np.array(env._env.sim.get_agent_state().position, dtype=np.float32)
            moved = float(np.linalg.norm(post_pos - pre_pos))
            if action == 1 and moved < 0.02:
                stuck_fwd += 1
            else:
                stuck_fwd = 0

            if fill_done:
                obs = env.reset()
                if obs is None or not env.is_running:
                    break

        nodes_after = len(graph.nodes)
        edges_after = sum(1 for u, nbrs in graph.edges.items() for v in nbrs if u < v)
        added_nodes = max(0, nodes_after - nodes_before)
        added_edges = max(0, edges_after - edges_before)
        if added_edges_local > 0:
            added_edges = max(added_edges, added_edges_local)
        fill_added_nodes_total += int(added_nodes)
        fill_added_edges_total += int(added_edges)

        if len(embeds_list) > 0:
            embeds_arr = np.stack(embeds_list, axis=0).astype(np.float16)
            embeds_norm = _refresh_embed_norm(embeds_arr)
        if len(clip_embeds_list) > 0:
            clip_arr = np.stack(clip_embeds_list, axis=0).astype(np.float16)
            clip_embeds_norm = _refresh_embed_norm(clip_arr)
        if len(geo_embeds_list) > 0:
            geo_arr = np.stack(geo_embeds_list, axis=0).astype(np.float16)
            geo_embeds_norm = _refresh_embed_norm(geo_arr)
        idx_to_node, node_to_idx = _refresh_index_maps(node_meta)
        belief_tmat = _build_transition_matrix(
            graph=graph,
            idx_to_node=idx_to_node,
            node_to_idx=node_to_idx,
            stay_prob=float(belief_stay_prob),
        )
        if belief_tmat.shape[0] > 0:
            if belief.shape[0] == belief_tmat.shape[0]:
                belief[:] = belief / (np.sum(belief) + 1e-6)
            else:
                belief = np.ones((belief_tmat.shape[0],), dtype=np.float32) / float(belief_tmat.shape[0])
        else:
            belief = np.zeros((0,), dtype=np.float32)
        node_pos_by_embed = _refresh_node_pos_by_embed()
        _save_graph_artifacts()

        # Re-localize once after fill for confidence and replanning diagnostics.
        new_loc_conf = 0.0
        if obs is not None and obs.get("rgb") is not None:
            rgb = obs.get("rgb")
            depth = obs.get("depth")
            if use_est_pose:
                prior_pos = np.asarray(est_pos, dtype=np.float32)
            else:
                st_fill = env._env.sim.get_agent_state()
                prior_pos = np.asarray(st_fill.position, dtype=np.float32)
            loc_row = _localize_with_fusion(rgb_obs=rgb, depth_obs=depth, prior_pos=prior_pos)
            final_scores = np.asarray(loc_row.get("final_scores", []), dtype=np.float32)
            trusted_node = int(loc_row["best_node"])
            if final_scores.size > 0 and belief.shape[0] == final_scores.shape[0]:
                if belief_tmat.shape[0] == final_scores.shape[0]:
                    prior_belief = belief.dot(belief_tmat)
                else:
                    prior_belief = belief.copy()
                temp = max(1e-6, float(belief_temp))
                logits = final_scores / temp
                logits = logits - float(np.max(logits))
                emission = np.exp(logits).astype(np.float32)
                emission = emission / (np.sum(emission) + 1e-6)
                posterior = prior_belief * emission
                posterior = posterior / (np.sum(posterior) + 1e-6)
                belief = (1.0 - float(belief_mix)) * prior_belief + float(belief_mix) * posterior
                belief = belief / (np.sum(belief) + 1e-6)
                bidx = int(np.argmax(belief))
                trusted_node = int(idx_to_node.get(int(bidx), trusted_node))
                if belief.size > 1:
                    top2 = np.partition(belief, -2)[-2:]
                    b1 = float(np.max(top2))
                    b2 = float(np.min(top2))
                else:
                    b1 = float(belief[0])
                    b2 = 0.0
                last_belief_entropy = _belief_entropy_norm(belief)
                max_entropy_use = float(os.environ.get("TOPO_BELIEF_MAX_ENTROPY_USE", "0.88"))
                best_fused_node = int(loc_row.get("best_node", trusted_node))
                if best_fused_node >= 0 and last_belief_entropy > max_entropy_use:
                    trusted_node = int(best_fused_node)
                belief_conf = float(np.clip((b1 - b2) * 4.0, 0.0, 1.0))
                new_loc_conf = 0.90 * float(loc_row["conf"]) + 0.10 * belief_conf
                k = min(max(1, int(belief_topk_print)), belief.shape[0])
                kidx = np.argpartition(-belief, k - 1)[:k]
                kidx = kidx[np.argsort(-belief[kidx])]
                topk = [(int(idx_to_node.get(int(i), -1)), float(belief[int(i)])) for i in kidx.tolist()]
                last_belief_topk = list(topk)
                print(
                    f"[TOPO_LOC_BELIEF] node={trusted_node} conf={new_loc_conf:.3f} entropy={last_belief_entropy:.3f} "
                    f"topk={[(n, round(p, 4)) for n, p in topk]}",
                    flush=True,
                )
                print(
                    f"[TOPO_RELOCALIZE_BELIEF] run={run_id} step={step} map={trusted_node} "
                    f"loc_conf={new_loc_conf:.3f} entropy={last_belief_entropy:.3f} "
                    f"topK={[(n, round(p, 4)) for n, p in topk]}",
                    flush=True,
                )
                belief_debug_rows.append(
                    {
                        "step": int(step),
                        "map_node": int(trusted_node),
                        "loc_conf": float(new_loc_conf),
                        "entropy": float(last_belief_entropy),
                        "topk": [{"node": int(n), "posterior_prob": float(p)} for n, p in topk],
                    }
                )
                if len(belief_debug_rows) > 180:
                    del belief_debug_rows[: len(belief_debug_rows) - 180]
            else:
                new_loc_conf = float(loc_row["conf"])
            belief_entropy_trace.append(float(last_belief_entropy))
            last_vote_node = trusted_node
            map_node_history.append(int(last_vote_node))
            last_conf = new_loc_conf
            last_loc_vision = float(loc_row["vision_best"])
            last_loc_geo = float(loc_row["geo_best"])
            last_loc_fused = float(loc_row["final_best"])
            last_loc_prior = float(loc_row["prior_best"])
            last_loc_node = int(trusted_node)
            history.clear()
            history.append(trusted_node)
            print(
                f"[TOPO_LOC_FUSED] node={last_loc_node} conf={new_loc_conf:.3f} "
                f"embed={float(loc_row['embed_best']):.4f} geo={last_loc_prior:.4f} alpha={float(loc_prior_alpha):.2f}",
                flush=True,
            )
            print(
                f"[TOPO_LOC_CONF] raw_top1={float(loc_row['raw_top1']):.4f} raw_top2={float(loc_row['raw_top2']):.4f} "
                f"gap={float(loc_row['gap']):.4f} conf={float(loc_row['conf']):.4f}",
                flush=True,
            )

        local_place_topk, local_obj_topk, local_room_topk = _compute_goal_candidates()
        room_topk_list = local_room_topk
        if len(local_place_topk) > 0:
            place_goal_topk_room = local_place_topk
            object_topk_list = local_obj_topk
            goal_topk_list = [(int(n), float(s)) for n, s, _ in local_place_topk]
        if last_vote_node is not None:
            goal_node, goal_object_id, goal_score, goal_room = _select_goal_from_objects(
                start_node=int(last_vote_node),
                object_topk=local_obj_topk,
                edges=graph.edges,
                fallback_goal_topk=local_place_topk,
            )
        goal_selected = False
        low_conf_streak = 0
        step_budget += int(fill_retry_nav_extra_steps)
        replanned_path_len = 0
        if last_vote_node is not None:
            replanned_path_len = len(dijkstra_path(graph.edges, int(last_vote_node), int(goal_node)))
        print(
            f"[TOPO_FILL_DONE] added_nodes={added_nodes} added_edges={added_edges} "
            f"new_loc_conf={new_loc_conf:.3f} replanned_path_len={replanned_path_len}",
            flush=True,
        )
        print(
            f"[TOPO_FILL_IMPROVE] before_goal={float(score_val):.4f} after_goal={float(goal_score):.4f} "
            f"before_conf={float(loc_conf_val):.3f} after_conf={float(new_loc_conf):.3f}",
            flush=True,
        )

    def _push_debug_step(
        step_idx: int,
        action_id: int,
        action_name: str,
        bearing_val: float,
        dist_subgoal_val: float,
        dist_goal_val: float,
        fwd_no_motion: bool,
        collision_flag: Optional[bool],
        events: List[str],
        rgb_obs,
        depth_obs,
    ) -> None:
        if not bool(debug_capture):
            return
        if use_est_pose:
            pose_xyz = [float(est_pos[0]), float(est_pos[1]), float(est_pos[2])]
            pose_yaw = float(est_yaw)
        else:
            st_dbg = env._env.sim.get_agent_state()
            pose_xyz = [float(st_dbg.position[0]), float(st_dbg.position[1]), float(st_dbg.position[2])]
            pose_yaw = float(_agent_yaw_from_quat(st_dbg.rotation))
        rec = {
            "step_idx": int(step_idx),
            "run_id": str(run_id),
            "scene_id": str(graph.meta.get("scene_id", "unknown")),
            "query": str(query),
            "condition": str(cond_name),
            "controller_mode": str(controller_mode),
            "action_id": int(action_id),
            "action": str(action_name),
            "pose_used": {
                "x": float(pose_xyz[0]),
                "y": float(pose_xyz[1]),
                "z": float(pose_xyz[2]),
                "yaw": float(pose_yaw),
                "source": "est_pose" if bool(use_est_pose) else "gt_pose",
            },
            "map_node_id": int(last_vote_node) if last_vote_node is not None else -1,
            "subgoal_node_id": int(next_node) if next_node is not None else -1,
            "goal_node_id": int(goal_node),
            "dist_to_subgoal": float(dist_subgoal_val),
            "bearing_to_subgoal": float(bearing_val),
            "dist_to_goal_node": float(dist_goal_val),
            "loc_conf": float(last_conf),
            "belief_entropy": float(last_belief_entropy),
            "belief_topk": [{"node": int(n), "prob": float(p)} for n, p in list(last_belief_topk)[:5]],
            "room_id_current": int(node_to_room.get(int(last_vote_node), -1)) if last_vote_node is not None else -1,
            "recovery_event": list(events),
            "fwd_no_motion": int(bool(fwd_no_motion)),
            "collision": collision_flag if collision_flag is None else int(bool(collision_flag)),
            "goal_score": float(goal_score),
            "goal_prob": float(goal_prob),
            "terminated_by": terminated_by,
        }
        event_frame = bool(events) or bool(fwd_no_motion)
        debug_buf.push(
            record=rec,
            rgb=rgb_obs,
            depth=depth_obs,
            event_frame=event_frame,
        )

    result = {
        "query": query,
        "goal_node": goal_node,
        "goal_object": goal_object_id,
        "goal_room": goal_room,
        "steps_used": 0,
        "reached_goal_node": False,
        "reached_goal_dist": None,
        "final_dist_to_goal_node": None,
        "success": False,
        "controller_mode": controller_mode,
        "path_nodes": [],
        "action_histogram": {},
        "fail_reason": None,
        "terminated_by": None,
        "success_by": None,
        "start_pose": start_pose,
        "goal_score": float(goal_score),
        "avg_loc_conf": None,
        "planned_node_path": [],
        "recovery_events": 0,
        "watchdog_trigger_count": 0,
        "relocalize_reset_count": 0,
        "doorway_burst_count": 0,
        "near_goal_stop_count": 0,
        "fill_triggers": 0,
        "fill_added_nodes": 0,
        "fill_added_edges": 0,
        "use_est_pose": bool(use_est_pose),
        "odom_noise_trans": float(odom_noise_trans),
        "odom_noise_yaw_deg": float(odom_noise_yaw_deg),
    }

    while not done:
        step_events: List[str] = []
        if step >= step_budget:
            if fill_triggers == 0 and controller_mode != "follower":
                _run_fill_map("max_steps_reached", goal_score, last_conf)
                continue
            fail_reason = "max_steps_reached"
            terminated_by = "max_steps"
            print(f"[TOPO_NAV_FAIL] reason={fail_reason} steps={step}", flush=True)
            break
        if force_relocalize or step % max(1, relocalize_every) == 0 or last_vote_node is None:
            force_relocalize = False
            rgb = obs.get("rgb")
            if rgb is None:
                raise RuntimeError("Observation missing 'rgb' key.")
            depth = obs.get("depth")
            if use_est_pose:
                prior_pos = np.asarray(est_pos, dtype=np.float32)
            else:
                st_loc = env._env.sim.get_agent_state()
                prior_pos = np.asarray(st_loc.position, dtype=np.float32)
            loc_row = _localize_with_fusion(rgb_obs=rgb, depth_obs=depth, prior_pos=prior_pos)
            curr_node = int(loc_row["best_node"])
            if curr_node < 0:
                curr_node = int(next(iter(graph.nodes.keys())))

            final_scores = np.asarray(loc_row.get("final_scores", []), dtype=np.float32)
            if final_scores.size > 0:
                if belief.shape[0] != final_scores.shape[0]:
                    belief = np.ones((final_scores.shape[0],), dtype=np.float32) / float(final_scores.shape[0])
                if belief_tmat.shape[0] != final_scores.shape[0]:
                    belief_tmat = _build_transition_matrix(
                        graph=graph,
                        idx_to_node=idx_to_node,
                        node_to_idx=node_to_idx,
                        stay_prob=float(belief_stay_prob),
                    )
                if belief_tmat.shape[0] == final_scores.shape[0]:
                    prior_belief = belief.dot(belief_tmat)
                else:
                    prior_belief = belief.copy()
                temp = max(1e-6, float(belief_temp))
                logits = final_scores / temp
                logits = logits - float(np.max(logits))
                emission = np.exp(logits).astype(np.float32)
                emission = emission / (np.sum(emission) + 1e-6)
                posterior = prior_belief * emission
                posterior = posterior / (np.sum(posterior) + 1e-6)
                belief = (1.0 - float(belief_mix)) * prior_belief + float(belief_mix) * posterior
                belief = belief / (np.sum(belief) + 1e-6)
                belief_idx = int(np.argmax(belief))
                curr_node = int(idx_to_node.get(int(belief_idx), curr_node))
                if belief.size > 1:
                    top2 = np.partition(belief, -2)[-2:]
                    b1 = float(np.max(top2))
                    b2 = float(np.min(top2))
                else:
                    b1 = float(belief[0])
                    b2 = 0.0
                last_belief_entropy = _belief_entropy_norm(belief)
                max_entropy_use = float(os.environ.get("TOPO_BELIEF_MAX_ENTROPY_USE", "0.88"))
                best_fused_node = int(loc_row.get("best_node", curr_node))
                if best_fused_node >= 0 and last_belief_entropy > max_entropy_use:
                    curr_node = int(best_fused_node)
                belief_conf = float(np.clip((b1 - b2) * 4.0, 0.0, 1.0))
                frame_conf = 0.90 * float(loc_row["conf"]) + 0.10 * belief_conf
                k = min(max(1, int(belief_topk_print)), belief.shape[0])
                kidx = np.argpartition(-belief, k - 1)[:k]
                kidx = kidx[np.argsort(-belief[kidx])]
                topk = [(int(idx_to_node.get(int(i), -1)), float(belief[int(i)])) for i in kidx.tolist()]
                last_belief_topk = list(topk)
                print(
                    f"[TOPO_LOC_BELIEF] node={curr_node} conf={frame_conf:.3f} entropy={last_belief_entropy:.3f} "
                    f"topk={[(n, round(p, 4)) for n, p in topk]}",
                    flush=True,
                )
                print(
                    f"[TOPO_RELOCALIZE_BELIEF] run={run_id} step={step} map={curr_node} "
                    f"loc_conf={frame_conf:.3f} entropy={last_belief_entropy:.3f} "
                    f"topK={[(n, round(p, 4)) for n, p in topk]}",
                    flush=True,
                )
                belief_debug_rows.append(
                    {
                        "step": int(step),
                        "map_node": int(curr_node),
                        "loc_conf": float(frame_conf),
                        "entropy": float(last_belief_entropy),
                        "topk": [{"node": int(n), "posterior_prob": float(p)} for n, p in topk],
                    }
                )
                if len(belief_debug_rows) > 180:
                    del belief_debug_rows[: len(belief_debug_rows) - 180]
            else:
                frame_conf = float(loc_row["conf"])
            history.append(curr_node)
            voted_node, vote_ratio, vote_map = _vote_node(history)
            if frame_conf >= loc_conf_thresh or trusted_node is None:
                trusted_node = voted_node
            last_vote_node = trusted_node
            map_node_history.append(int(last_vote_node))
            last_conf = frame_conf
            belief_entropy_trace.append(float(last_belief_entropy))
            last_loc_vision = float(loc_row["vision_best"])
            last_loc_geo = float(loc_row["geo_best"])
            last_loc_fused = float(loc_row["final_best"])
            last_loc_prior = float(loc_row["prior_best"])
            last_loc_node = int(curr_node)
            print(
                f"[TOPO_LOC_FUSED] node={last_loc_node} conf={last_conf:.3f} "
                f"embed={float(loc_row['embed_best']):.4f} geo={last_loc_prior:.4f} alpha={float(loc_prior_alpha):.2f}",
                flush=True,
            )
            print(
                f"[TOPO_LOC_CONF] raw_top1={float(loc_row['raw_top1']):.4f} raw_top2={float(loc_row['raw_top2']):.4f} "
                f"gap={float(loc_row['gap']):.4f} conf={float(loc_row['conf']):.4f}",
                flush=True,
            )
            if last_conf < loc_conf_thresh:
                low_conf_streak += 1
            else:
                low_conf_streak = 0
            if (
                belief.shape[0] > 0
                and low_conf_streak >= 2
                and float(last_conf) <= float(belief_reset_low_conf)
                and float(last_belief_entropy) >= float(belief_reset_high_entropy)
            ):
                belief = np.ones_like(belief, dtype=np.float32) / max(1.0, float(belief.shape[0]))
                relocalize_reset_count += 1
                step_events.append("relocalize_reset_low_conf")
                print(
                    f"[TOPO_RELOCALIZE_RESET] run={run_id} reason=low_conf step={step} "
                    f"loc_conf={float(last_conf):.3f} entropy={float(last_belief_entropy):.3f}",
                    flush=True,
                )
            avg_loc_conf = float((sum(loc_conf_trace) + float(last_conf)) / max(1, len(loc_conf_trace) + 1))
            print(
                f"[TOPO_LOC] node={last_vote_node} conf={last_conf:.2f} vote={vote_map} avg_loc_conf={avg_loc_conf:.3f}",
                flush=True,
            )
            loc_conf_trace.append(float(last_conf))

            if not goal_selected:
                goal_node, goal_object_id, goal_score, goal_room = _select_goal_from_objects(
                    start_node=int(last_vote_node),
                    object_topk=object_topk_list,
                    edges=graph.edges,
                    fallback_goal_topk=place_goal_topk_room,
                )
                goal_selected = True
                print(
                    f"[TOPO_GOAL] goal_node={goal_node} score={goal_score:.4f} room={goal_room} "
                    f"goal_object={goal_object_id} query=\"{query}\"",
                    flush=True,
                )
                if goal_score < float(goal_score_thresh) and fill_triggers == 0:
                    reason = "low_room_conf" if low_room_conf else "low_goal_score"
                    _run_fill_map(reason, goal_score, last_conf)
                    continue

            if low_conf_streak >= int(fill_relocalize_low_conf_patience) and fill_triggers == 0:
                _run_fill_map("low_loc_conf", goal_score, last_conf)
                continue
            if (
                float(last_belief_entropy) >= float(belief_entropy_fill_thresh)
                and float(last_conf) < (float(loc_conf_thresh) * 0.85)
                and fill_triggers == 0
                and controller_mode != "follower"
            ):
                _run_fill_map("high_belief_entropy", goal_score, last_conf)
                continue

            path_nodes = dijkstra_path(graph.edges, last_vote_node, goal_node)
            path_index = 0
            next_node = path_nodes[1] if len(path_nodes) > 1 else goal_node
            step_events.append("replan")
            print(f"[TOPO_PLAN] path_len={len(path_nodes)} next={next_node}", flush=True)

        state = env._env.sim.get_agent_state()
        curr_pos = np.array([state.position[0], state.position[1], state.position[2]], dtype=np.float32)
        curr_rot = state.rotation
        target_pos = np.array(graph.nodes[next_node].position, dtype=np.float32)
        goal_pos_curr = np.array(graph.nodes[goal_node].position, dtype=np.float32)
        if use_est_pose:
            bearing, dist = _bearing_distance_from_yaw(est_pos, est_yaw, target_pos)
            _, dist_goal = _bearing_distance_from_yaw(est_pos, est_yaw, goal_pos_curr)
        else:
            bearing, dist = bearing_distance(curr_pos, curr_rot, target_pos)
            _, dist_goal = bearing_distance(curr_pos, curr_rot, goal_pos_curr)
        print(
            f"[TOPO_LOC_SCORE] vision={last_loc_vision:.4f} geo={last_loc_geo:.4f} "
            f"fused={last_loc_fused:.4f} prior={last_loc_prior:.4f} chosen_node={last_loc_node} conf={last_conf:.3f}",
            flush=True,
        )
        if last_next_node != next_node:
            subgoal_dist_trace.clear()
            goal_dist_trace.clear()
            last_next_node = next_node

        if dist <= node_reach_thresh:
            if next_node == goal_node:
                print(
                    f"[TOPO_NAV_SUCCESS] step={step} reached_node={goal_node} dist={dist:.3f}",
                    flush=True,
                )
                result["reached_goal_node"] = True
                result["reached_goal_dist"] = float(dist)
                terminated_by = "success_stop"
                success_by = "dist_threshold"
                done = True
                break
            if path_index + 1 < len(path_nodes) - 1:
                path_index += 1
                next_node = path_nodes[path_index + 1]
                step_events.append("advance_subgoal")
                print(f"[TOPO_PLAN] path_len={len(path_nodes)} next={next_node}", flush=True)
                target_pos = np.array(graph.nodes[next_node].position, dtype=np.float32)
                if use_est_pose:
                    bearing, dist = _bearing_distance_from_yaw(est_pos, est_yaw, target_pos)
                    _, dist_goal = _bearing_distance_from_yaw(est_pos, est_yaw, goal_pos_curr)
                else:
                    bearing, dist = bearing_distance(curr_pos, curr_rot, target_pos)
                    _, dist_goal = bearing_distance(curr_pos, curr_rot, goal_pos_curr)
                subgoal_dist_trace.clear()
                goal_dist_trace.clear()
                last_next_node = next_node

        subgoal_dist_trace.append(float(dist))
        goal_dist_trace.append(float(dist_goal))
        progress_stall = False
        if len(subgoal_dist_trace) == subgoal_dist_trace.maxlen:
            start_dist = float(subgoal_dist_trace[0])
            best_dist = float(min(subgoal_dist_trace))
            progress_stall = (start_dist - best_dist) < float(stall_min_delta) and dist > node_reach_thresh
        goal_idx = node_to_idx.get(int(goal_node), -1)
        goal_prob = float(belief[int(goal_idx)]) if (0 <= int(goal_idx) < belief.shape[0]) else 0.0
        anti_false_positive = bool(progress_stall and float(last_conf) < float(stop_near_low_conf_guard))
        if (
            float(dist_goal) <= float(stop_near_m)
            and float(goal_score) >= float(stop_near_goal_score_min)
            and (
                float(goal_prob) >= float(stop_near_goal_prob)
                or float(last_conf) >= float(stop_near_loc_conf_min)
            )
            and not anti_false_positive
        ):
            near_goal_stable_count += 1
        else:
            near_goal_stable_count = 0
        stuck_flag = bool(bearing_ctrl.stuck_state()) if bearing_ctrl is not None else False

        if recovery_cooldown_remaining > 0:
            recovery_cooldown_remaining -= 1
        if (
            controller_mode != "follower"
            and (progress_stall or stuck_flag)
            and recovery_turn_remaining == 0
            and recovery_follow_remaining == 0
            and recovery_forward_remaining == 0
            and recovery_cooldown_remaining == 0
        ):
            watchdog_trigger_count += 1
            step_events.append("watchdog_no_progress")
            print(
                f"[TOPO_PROGRESS_WATCHDOG] run={run_id} step={step} reason=no_progress "
                f"dist_subgoal={dist:.3f} dist_goal={float(np.linalg.norm(curr_pos - np.array(graph.nodes[goal_node].position, dtype=np.float32))):.3f} "
                f"mode={controller_mode}",
                flush=True,
            )
            reason = "stall_progress" if progress_stall else "stuck_counter"
            recovery_mode_used = "depth_turn"
            use_follower_burst = recovery_mode == "follower_burst" or (
                recovery_mode == "hybrid" and recovery_events >= 1
            )
            if progress_stall and float(dist_goal) > 6.0 and (watchdog_trigger_count % 4 == 0):
                if belief.shape[0] > 0:
                    belief = np.ones_like(belief, dtype=np.float32) / max(1.0, float(belief.shape[0]))
                    relocalize_reset_count += 1
                    step_events.append("relocalize_reset_no_progress")
                    print(
                        f"[TOPO_RELOCALIZE_RESET] run={run_id} reason=no_progress step={step} "
                        f"loc_conf={float(last_conf):.3f} entropy={float(last_belief_entropy):.3f}",
                        flush=True,
                    )
            if progress_stall:
                trigger = "follower_burst" if use_follower_burst else "replan"
                print(
                    f"[TOPO_STALL] step={step} dist={dist:.3f} trigger={trigger}",
                    flush=True,
                )
                force_relocalize = True
            doorway_context = False
            if int(doorway_burst_steps) > 0 and last_vote_node is not None:
                edge_key = (
                    (int(last_vote_node), int(next_node))
                    if int(last_vote_node) <= int(next_node)
                    else (int(next_node), int(last_vote_node))
                )
                doorway_context = edge_key in doorway_edge_set or int(last_vote_node) in doorway_node_set
            if doorway_context and int(doorway_burst_steps) > 0:
                recovery_forward_remaining = max(1, int(doorway_burst_steps))
                doorway_burst_count += 1
                recovery_mode_used = "doorway_burst"
            elif use_follower_burst:
                try:
                    if follower is None:
                        follower = FollowerController(env._env.sim)
                    recovery_follow_to_goal = int(watchdog_trigger_count) >= int(goal_burst_after)
                    if recovery_follow_to_goal:
                        recovery_follow_remaining = max(1, int(goal_burst_steps))
                    else:
                        recovery_follow_remaining = max(1, int(recovery_follower_steps))
                    recovery_mode_used = "follower_burst"
                except Exception:
                    recovery_follow_remaining = 0
                    recovery_follow_to_goal = False
            if int(watchdog_trigger_count) >= int(goal_lock_after) and float(dist_goal) >= float(goal_lock_min_dist):
                if follower is None:
                    try:
                        follower = FollowerController(env._env.sim)
                    except Exception:
                        follower = None
                if follower is not None:
                    recovery_goal_lock = True
                    recovery_mode_used = "follower_goal_lock"
            if recovery_follow_remaining == 0:
                default_turn = 2 if (recovery_events % 2 == 0) else 3
                recovery_turn_action = _pick_recovery_turn_from_depth(
                    obs.get("depth"),
                    default_turn=default_turn,
                )
                if not recovery_goal_lock:
                    recovery_turn_remaining = max(1, int(recovery_turn_steps))
                    recovery_mode_used = "depth_turn"
                    recovery_follow_to_goal = False
            recovery_cooldown_remaining = max(1, int(recovery_cooldown))
            recovery_events += 1
            subgoal_dist_trace.clear()
            print(
                f"[TOPO_RECOVERY] mode={recovery_mode_used} reason={reason} "
                f"step={step} dist={dist:.3f} stuck={int(stuck_flag)}",
                flush=True,
            )
            step_events.append(f"recovery:{recovery_mode_used}")

        if controller_mode == "follower":
            action = follower.act(target_pos)
            action_name = ["STOP", "FORWARD", "LEFT", "RIGHT"][action]
        else:
            near_goal_force_stop = False
            if near_goal_stable_count >= int(stop_near_hold_steps):
                action = 0
                action_name = "STOP"
                near_goal_force_stop = True
                near_goal_stop_count += 1
                step_events.append("near_goal_stop")
                print(
                    f"[TOPO_NEAR_GOAL_STOP] run={run_id} step={step} dist={dist_goal:.3f} "
                    f"goal_prob={goal_prob:.3f} loc_conf={last_conf:.3f} T={near_goal_stable_count}",
                    flush=True,
                )
            elif recovery_forward_remaining > 0:
                action = 1
                action_name = "FORWARD"
                recovery_forward_remaining -= 1
            elif recovery_goal_lock and follower is not None:
                action = follower.act(goal_pos_curr)
                action_name = ["STOP", "FORWARD", "LEFT", "RIGHT"][action]
                if int(action) == 0 and float(dist_goal) > max(float(node_reach_thresh), float(stop_near_m)):
                    action = 1
                    action_name = "FORWARD"
            elif recovery_follow_remaining > 0 and follower is not None:
                follow_target = goal_pos_curr if recovery_follow_to_goal else target_pos
                action = follower.act(follow_target)
                action_name = ["STOP", "FORWARD", "LEFT", "RIGHT"][action]
                recovery_follow_remaining -= 1
                if recovery_follow_remaining <= 0:
                    recovery_follow_to_goal = False
            elif recovery_turn_remaining > 0:
                action = int(recovery_turn_action)
                action_name = "LEFT" if action == 2 else "RIGHT"
                recovery_turn_remaining -= 1
            else:
                action, action_name = bearing_ctrl.decide(bearing, dist)
            if action == 0 and (not near_goal_force_stop) and (dist > float(stop_guard_dist) or int(next_node) != int(goal_node)):
                if abs(float(bearing)) <= float(turn_thresh):
                    action = 1
                    action_name = "FORWARD"
                elif float(bearing) < 0:
                    action = 2
                    action_name = "LEFT"
                else:
                    action = 3
                    action_name = "RIGHT"
                print(
                    f"[TOPO_STOP_GUARD] dist={dist:.3f} action_before=STOP action_after={action_name}",
                    flush=True,
                )
                step_events.append("stop_guard")
            stuck = bearing_ctrl.stuck_state()
            print(
                f"[TOPO_CTRL] mode=bearing step={step} bearing={bearing:.3f} dist={dist:.3f} "
                f"action={action_name} stuck={stuck} node={last_vote_node} next={next_node}",
                flush=True,
            )
            if recovery_goal_lock and float(dist_goal) <= max(float(stop_near_m), float(node_reach_thresh) * 1.5):
                recovery_goal_lock = False

        print(
            f"[TOPO_NAV] step={step} cur_node={last_vote_node} next_node={next_node} "
            f"dist={dist:.3f} bearing={bearing:.3f} action={action_name}",
            flush=True,
        )
        curr_room_id = int(node_to_room.get(int(last_vote_node), -1)) if last_vote_node is not None else -1
        if prev_room_id is None:
            prev_room_id = int(curr_room_id)
            if curr_room_id >= 0:
                room_id_seq.append(int(curr_room_id))
        else:
            if int(curr_room_id) != int(prev_room_id):
                room_transitions.append(
                    {
                        "step": int(step),
                        "node_id": int(last_vote_node) if last_vote_node is not None else -1,
                        "from_room": int(prev_room_id),
                        "to_room": int(curr_room_id),
                        "loc_conf": float(last_conf),
                        "goal_score": float(goal_score),
                        "bearing": float(bearing),
                        "dist_to_subgoal": float(dist),
                    }
                )
                step_events.append("room_transition")
                prev_room_id = int(curr_room_id)
            if curr_room_id >= 0 and (len(room_id_seq) == 0 or int(room_id_seq[-1]) != int(curr_room_id)):
                room_id_seq.append(int(curr_room_id))
        action_key = "FWD" if action_name == "FORWARD" else action_name
        action_hist[action_key] += 1

        if action == 0:
            if dist <= node_reach_thresh:
                if next_node == goal_node:
                    result["reached_goal_node"] = True
                    result["reached_goal_dist"] = float(dist)
                    terminated_by = "success_stop"
                    success_by = "stop"
                    _push_debug_step(
                        step_idx=step,
                        action_id=int(action),
                        action_name=str(action_name),
                        bearing_val=float(bearing),
                        dist_subgoal_val=float(dist),
                        dist_goal_val=float(dist_goal),
                        fwd_no_motion=False,
                        collision_flag=last_collision_flag,
                        events=step_events,
                        rgb_obs=obs.get("rgb"),
                        depth_obs=obs.get("depth"),
                    )
                    done = True
                    break
            if near_goal_stable_count >= int(stop_near_hold_steps) and float(dist_goal) <= float(stop_near_m):
                result["reached_goal_node"] = True
                result["reached_goal_dist"] = float(dist_goal)
                terminated_by = "success_stop"
                success_by = "stop"
                _push_debug_step(
                    step_idx=step,
                    action_id=int(action),
                    action_name=str(action_name),
                    bearing_val=float(bearing),
                    dist_subgoal_val=float(dist),
                    dist_goal_val=float(dist_goal),
                    fwd_no_motion=False,
                    collision_flag=last_collision_flag,
                    events=step_events,
                    rgb_obs=obs.get("rgb"),
                    depth_obs=obs.get("depth"),
                )
                done = True
                break
            fail_reason = "early_stop_before_node"
            terminated_by = "early_stop"
            print(f"[TOPO_NAV_FAIL] reason={fail_reason} steps={step}", flush=True)
            _push_debug_step(
                step_idx=step,
                action_id=int(action),
                action_name=str(action_name),
                bearing_val=float(bearing),
                dist_subgoal_val=float(dist),
                dist_goal_val=float(dist_goal),
                fwd_no_motion=False,
                collision_flag=last_collision_flag,
                events=step_events + ["fail:early_stop_before_node"],
                rgb_obs=obs.get("rgb"),
                depth_obs=obs.get("depth"),
            )
            if fill_triggers == 0 and controller_mode != "follower":
                _run_fill_map(fail_reason, goal_score, last_conf)
                fail_reason = None
                terminated_by = None
                continue
            done = True
            break

        prev_pos = curr_pos.copy()
        prev_yaw_gt = float(_agent_yaw_from_quat(curr_rot))
        obs, _, done, info = env.step(action)
        step += 1
        last_collision_flag = _extract_collision_flag(info)

        new_state = env._env.sim.get_agent_state()
        new_pos = np.array(
            [new_state.position[0], new_state.position[1], new_state.position[2]], dtype=np.float32
        )
        moved_dist = float(np.linalg.norm(new_pos - prev_pos))
        fwd_no_motion = bool(int(action) == 1 and moved_dist < 0.01)
        if fwd_no_motion:
            step_events.append("fwd_no_motion")
        if last_collision_flag:
            step_events.append("collision")
        if controller_mode != "follower" and bearing_ctrl is not None:
            bearing_ctrl.update_after_step(action, moved_dist)
        if use_est_pose:
            new_yaw_gt = float(_agent_yaw_from_quat(new_state.rotation))
            noisy_moved = max(0.0, moved_dist + odom_rng.gauss(0.0, float(odom_noise_trans)))
            yaw_delta_gt = wrap_angle(new_yaw_gt - prev_yaw_gt)
            yaw_delta_noisy = wrap_angle(yaw_delta_gt + odom_rng.gauss(0.0, float(odom_noise_yaw)))
            est_yaw = wrap_angle(est_yaw + yaw_delta_noisy)
            if int(action) == 1:
                est_pos[0] += float(math.sin(est_yaw) * noisy_moved)
                est_pos[2] += float(-math.cos(est_yaw) * noisy_moved)
            est_pos[1] = float(new_pos[1])

        _push_debug_step(
            step_idx=step,
            action_id=int(action),
            action_name=str(action_name),
            bearing_val=float(bearing),
            dist_subgoal_val=float(dist),
            dist_goal_val=float(dist_goal),
            fwd_no_motion=bool(fwd_no_motion),
            collision_flag=last_collision_flag,
            events=step_events,
            rgb_obs=obs.get("rgb") if isinstance(obs, dict) else None,
            depth_obs=obs.get("depth") if isinstance(obs, dict) else None,
        )

    if not result["reached_goal_node"] and step >= step_budget and not done and fail_reason is None:
        fail_reason = "max_steps_reached"
        terminated_by = "max_steps"
        print(f"[TOPO_NAV_FAIL] reason={fail_reason} steps={step}", flush=True)

    final_state = env._env.sim.get_agent_state()
    final_pos = np.array(
        [final_state.position[0], final_state.position[1], final_state.position[2]], dtype=np.float32
    )
    goal_pos = np.array(graph.nodes[goal_node].position, dtype=np.float32)
    final_dist = float(np.linalg.norm(final_pos - goal_pos))
    result["steps_used"] = step
    result["final_dist_to_goal_node"] = final_dist
    result["success"] = result["reached_goal_node"] or final_dist <= node_reach_thresh
    result["path_nodes"] = path_nodes
    result["action_histogram"] = action_hist
    result["avg_loc_conf"] = float(sum(loc_conf_trace) / len(loc_conf_trace)) if loc_conf_trace else 0.0
    hist_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    if len(loc_conf_trace) > 0:
        counts, _ = np.histogram(np.asarray(loc_conf_trace, dtype=np.float32), bins=np.asarray(hist_bins + [1.000001], dtype=np.float32))
        conf_hist_counts = [int(x) for x in counts.tolist()]
    else:
        conf_hist_counts = [0] * (len(hist_bins))
    result["loc_conf_hist"] = {"bins": hist_bins, "counts": conf_hist_counts}
    print(f"[TOPO_LOC_CONF_HIST] bins={hist_bins} counts={conf_hist_counts}", flush=True)
    result["planned_node_path"] = [int(n) for n in path_nodes]
    path_room_seq = []
    for nid in path_nodes:
        rid = int(node_to_room.get(int(nid), -1))
        if len(path_room_seq) == 0 or int(path_room_seq[-1]) != int(rid):
            path_room_seq.append(int(rid))
    if len(room_id_seq) == 0 and len(path_room_seq) > 0:
        room_id_seq = [int(x) for x in path_room_seq if int(x) >= 0]
    rooms_visited = sorted({int(r) for r in room_id_seq if int(r) >= 0})
    result["room_id_seq"] = [int(x) for x in room_id_seq]
    result["path_room_seq"] = [int(x) for x in path_room_seq]
    result["rooms_visited_count"] = int(len(rooms_visited))
    result["room_transitions"] = room_transitions
    result["room_transitions_count"] = int(len(room_transitions))
    result["recovery_events"] = int(recovery_events)
    result["watchdog_trigger_count"] = int(watchdog_trigger_count)
    result["relocalize_reset_count"] = int(relocalize_reset_count)
    result["doorway_burst_count"] = int(doorway_burst_count)
    result["near_goal_stop_count"] = int(near_goal_stop_count)
    result["goal_score"] = float(goal_score)
    result["goal_room"] = str(goal_room)
    result["goal_object"] = int(goal_object_id)
    result["fill_triggers"] = int(fill_triggers)
    result["fill_added_nodes"] = int(fill_added_nodes_total)
    result["fill_added_edges"] = int(fill_added_edges_total)
    if len(belief_entropy_trace) > 0:
        belief_entropy_summary = {
            "mean": float(np.mean(np.asarray(belief_entropy_trace, dtype=np.float32))),
            "max": float(np.max(np.asarray(belief_entropy_trace, dtype=np.float32))),
            "min": float(np.min(np.asarray(belief_entropy_trace, dtype=np.float32))),
        }
    else:
        belief_entropy_summary = {"mean": 0.0, "max": 0.0, "min": 0.0}
    result["belief_entropy_summary"] = belief_entropy_summary
    if bool(result["success"]):
        fail_reason = None
        if terminated_by is None:
            if int(step) == 0 and not bool(result.get("reached_goal_node", False)):
                terminated_by = "success_start_in_goal"
                success_by = "start_in_goal"
            else:
                terminated_by = "success_stop"
        if success_by is None:
            if terminated_by == "max_steps":
                success_by = "dist_threshold"
            elif int(near_goal_stop_count) > 0:
                success_by = "stop"
            elif bool(result.get("reached_goal_node", False)):
                success_by = "dist_threshold"
            else:
                success_by = "dist_threshold"
    else:
        if fail_reason is None:
            if terminated_by == "max_steps":
                fail_reason = "max_steps_reached"
            elif terminated_by == "early_stop":
                fail_reason = "early_stop_before_node"
            else:
                fail_reason = "final_distance_too_far"
                print(f"[TOPO_NAV_FAIL] reason={fail_reason} steps={step}", flush=True)
        if terminated_by is None:
            if fail_reason == "max_steps_reached":
                terminated_by = "max_steps"
            elif fail_reason == "early_stop_before_node":
                terminated_by = "early_stop"
            else:
                terminated_by = "exception"
        success_by = None

    result["fail_reason"] = fail_reason if not bool(result["success"]) else None
    result["terminated_by"] = str(terminated_by) if terminated_by is not None else None
    result["success_by"] = str(success_by) if success_by is not None else None

    path_pose_path = os.path.join(out_dir, f"path_poses_{run_id}.json")
    path_pose_rows = []
    for nid in path_nodes:
        node = graph.nodes[int(nid)]
        yaw = _node_yaw_from_quat_wxyz(node.rotation)
        path_pose_rows.append(
            {
                "node_id": int(nid),
                "x": float(node.position[0]),
                "y": float(node.position[1]),
                "z": float(node.position[2]),
                "yaw": float(yaw),
            }
        )
    with open(path_pose_path, "w") as f:
        json.dump({"run_id": run_id, "query": query, "path_poses": path_pose_rows}, f, indent=2)

    result_path = os.path.join(out_dir, f"RESULT_{run_id}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "RESULT.json"), "w") as f:
        json.dump(result, f, indent=2)

    terminated_maxsteps = str(result.get("terminated_by", "")) == "max_steps"
    should_flush_debug = bool(debug_capture) and (
        (not bool(debug_only_on_fail))
        or (not bool(result["success"]))
        or (bool(debug_capture_success_at_maxsteps) and terminated_maxsteps)
    )
    oracle_info = {
        "oracle_available": 0,
        "oracle_success": 0,
        "oracle_geodesic_start": float("inf"),
        "oracle_geodesic_final": float("inf"),
        "oracle_steps_used": 0,
    }
    if bool(debug_oracle_on_fail) and (
        (not bool(result["success"])) or (bool(debug_capture_success_at_maxsteps) and terminated_maxsteps)
    ):
        oracle_info = _run_oracle_probe(
            sim=env._env.sim,
            start_pos=oracle_start_pos,
            start_rot=oracle_start_rot,
            goal_pos=goal_pos,
            max_steps=int(max_steps),
            node_reach_thresh=float(node_reach_thresh),
        )
    result.update(oracle_info)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "RESULT.json"), "w") as f:
        json.dump(result, f, indent=2)

    if should_flush_debug:
        debug_meta = {
            "run_id": str(run_id),
            "scene_id": str(graph.meta.get("scene_id", "unknown")),
            "query": str(query),
            "condition": str(cond_name),
            "success": bool(result["success"]),
            "terminated_by": result.get("terminated_by"),
            "fail_reason": result.get("fail_reason"),
            "controller_mode": str(controller_mode),
            "max_steps": int(max_steps),
            "node_reach_thresh": float(node_reach_thresh),
            "goal_node": int(goal_node),
            "goal_score": float(goal_score),
            "settings": {
                "debug_only_on_fail": int(bool(debug_only_on_fail)),
                "debug_frame_stride": int(debug_frame_stride),
                "debug_ringbuf_steps": int(debug_ringbuf_steps),
                "debug_save_depth": int(bool(debug_save_depth)),
                "debug_capture_success_at_maxsteps": int(bool(debug_capture_success_at_maxsteps)),
                "use_est_pose": int(bool(use_est_pose)),
                "odom_noise_trans": float(odom_noise_trans),
                "odom_noise_yaw_deg": float(odom_noise_yaw_deg),
            },
            "oracle": oracle_info,
            "result_path": str(result_path),
            "build_dir": str(os.path.dirname(graph_json)),
        }
        flush_stats = debug_buf.flush(debug_run_dir, debug_meta)
        print(
            f"[TOPO_DEBUG_FLUSH] run={run_id} success={int(bool(result['success']))} "
            f"terminated_by={result.get('terminated_by')} saved_steps={int(flush_stats['saved_steps'])} "
            f"saved_frames={int(flush_stats['saved_frames'])} dir={debug_run_dir}",
            flush=True,
        )

    debug_should_dump = (not bool(result["success"])) or int(watchdog_trigger_count) > 0 or int(relocalize_reset_count) > 0
    if debug_should_dump:
        belief_debug_path = os.path.join(out_dir, f"belief_debug_{run_id}.json")
        with open(belief_debug_path, "w") as f:
            json.dump(
                {
                    "run_id": str(run_id),
                    "success": bool(result["success"]),
                    "watchdog_trigger_count": int(watchdog_trigger_count),
                    "relocalize_reset_count": int(relocalize_reset_count),
                    "doorway_burst_count": int(doorway_burst_count),
                    "near_goal_stop_count": int(near_goal_stop_count),
                    "belief_entropy_summary": belief_entropy_summary,
                    "map_node_history": [int(x) for x in map_node_history],
                    "rows": belief_debug_rows[-120:],
                },
                f,
                indent=2,
            )
    print(f"[TOPO_ACTION_HIST] run={run_id} mode={controller_mode} hist={action_hist}", flush=True)
    print(
        f"[TOPO_RUN_ROOM_SEQ] run={run_id} rooms_visited={int(len(rooms_visited))} "
        f"seq={result['room_id_seq']} transitions={int(len(room_transitions))}",
        flush=True,
    )

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Relocalize + retrieve goal + plan + execute on a TopoGraph.")
    parser.add_argument("--config_path", type=str, default="repo/InternNav/scripts/eval/configs/vln_r2r.yaml")
    parser.add_argument("--graph_json", type=str, default="runs/topo_mvp/graph.json")
    parser.add_argument("--embeds_npy", type=str, default="runs/topo_mvp/node_embeds.npy")
    parser.add_argument("--node_meta", type=str, default="runs/topo_mvp/node_meta.json")
    parser.add_argument("--query", type=str, default="kitchen table")
    parser.add_argument("--out_dir", type=str, default="runs/topo_mvp")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-DualVLN")
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--vote_window", type=int, default=3)
    parser.add_argument("--relocalize_every", type=int, default=3)
    parser.add_argument("--stop_thresh", type=float, default=0.3)
    parser.add_argument("--turn_thresh", type=float, default=0.35)
    parser.add_argument("--node_reach_thresh", type=float, default=0.4)
    parser.add_argument("--controller_mode", type=str, default="bearing", choices=["follower", "bearing"])
    parser.add_argument("--run_id", type=str, default="run1")
    parser.add_argument("--start_episode_offset", type=int, default=0)
    parser.add_argument("--vision_prompt", type=str, default="")
    parser.add_argument("--goal_topk", type=int, default=5)
    parser.add_argument("--loc_conf_thresh", type=float, default=0.34)
    parser.add_argument("--belief_temp", type=float, default=float(os.environ.get("TOPO_BELIEF_TEMP", "0.20")))
    parser.add_argument("--belief_stay_prob", type=float, default=float(os.environ.get("TOPO_BELIEF_STAY_PROB", "0.70")))
    parser.add_argument("--belief_mix", type=float, default=float(os.environ.get("TOPO_BELIEF_MIX", "0.85")))
    parser.add_argument(
        "--belief_entropy_fill_thresh",
        type=float,
        default=float(os.environ.get("TOPO_BELIEF_ENTROPY_FILL_THRESH", "0.78")),
    )
    parser.add_argument(
        "--belief_topk_print",
        type=int,
        default=int(os.environ.get("TOPO_BELIEF_TOPK_PRINT", "4")),
    )
    parser.add_argument(
        "--belief_hop_radius",
        type=int,
        default=int(os.environ.get("TOPO_BELIEF_HOP_RADIUS", "2")),
    )
    parser.add_argument(
        "--belief_geo_radius",
        type=float,
        default=float(os.environ.get("TOPO_BELIEF_GEO_RADIUS", "3.5")),
    )
    parser.add_argument(
        "--belief_noncand_penalty",
        type=float,
        default=float(os.environ.get("TOPO_BELIEF_NONCAND_PENALTY", "1.0")),
    )
    parser.add_argument(
        "--belief_reset_low_conf",
        type=float,
        default=float(os.environ.get("TOPO_BELIEF_RESET_LOW_CONF", "0.18")),
    )
    parser.add_argument(
        "--belief_reset_high_entropy",
        type=float,
        default=float(os.environ.get("TOPO_BELIEF_RESET_HIGH_ENTROPY", "0.98")),
    )
    parser.add_argument("--kidnap_start", type=int, default=0)
    parser.add_argument("--kidnap_seed", type=int, default=7)
    parser.add_argument("--kidnap_min_nearest_node_dist", type=float, default=0.5)
    parser.add_argument("--kidnap_max_nearest_node_dist", type=float, default=4.0)
    parser.add_argument("--clip_embeds_npy", type=str, default="runs/topo_mvp/node_embeds_clip.npy")
    parser.add_argument("--geo_embeds_npy", type=str, default="runs/topo_mvp/node_geo.npy")
    parser.add_argument("--object_graph_json", type=str, default="")
    parser.add_argument("--object_nodes_json", type=str, default="")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--use_two_stage_goal", type=int, default=0)
    parser.add_argument("--room_shortlist_k", type=int, default=12)
    parser.add_argument("--object_topk", type=int, default=int(os.environ.get("TOPO_OBJECT_TOPK", "8")))
    parser.add_argument("--room_min", type=float, default=float(os.environ.get("TOPO_ROOM_MIN", "0.25")))
    parser.add_argument("--room_query_topk", type=int, default=int(os.environ.get("TOPO_ROOM_QUERY_TOPK", "5")))
    parser.add_argument("--goal_score_thresh", type=float, default=float(os.environ.get("TOPO_GOAL_SCORE_THRESH", "0.0")))
    parser.add_argument("--fill_steps", type=int, default=int(os.environ.get("TOPO_FILL_STEPS", "120")))
    parser.add_argument(
        "--fill_min_node_spacing",
        type=float,
        default=float(os.environ.get("TOPO_FILL_MIN_NODE_SPACING", "0.35")),
    )
    parser.add_argument(
        "--fill_relocalize_low_conf_patience",
        type=int,
        default=int(os.environ.get("TOPO_FILL_LOW_CONF_PATIENCE", "6")),
    )
    parser.add_argument(
        "--fill_retry_nav_extra_steps",
        type=int,
        default=int(os.environ.get("TOPO_FILL_RETRY_NAV_EXTRA_STEPS", "80")),
    )
    parser.add_argument("--loc_fuse_alpha", type=float, default=float(os.environ.get("TOPO_LOC_ALPHA", "0.6")))
    parser.add_argument("--loc_prior_alpha", type=float, default=float(os.environ.get("TOPO_LOC_PRIOR_ALPHA", "0.72")))
    parser.add_argument("--loc_prior_sigma", type=float, default=float(os.environ.get("TOPO_LOC_PRIOR_SIGMA", "2.2")))
    parser.add_argument(
        "--loc_conf_margin_center",
        type=float,
        default=float(os.environ.get("TOPO_LOC_CONF_MARGIN_CENTER", "0.02")),
    )
    parser.add_argument(
        "--loc_conf_gain",
        type=float,
        default=float(os.environ.get("TOPO_LOC_CONF_GAIN", "12.0")),
    )
    parser.add_argument("--geo_bins_r", type=int, default=int(os.environ.get("TOPO_GEO_BINS_R", "20")))
    parser.add_argument("--geo_bins_theta", type=int, default=int(os.environ.get("TOPO_GEO_BINS_THETA", "60")))
    parser.add_argument("--geo_max_depth", type=float, default=float(os.environ.get("TOPO_GEO_MAX_DEPTH", "5.0")))
    parser.add_argument("--fill_rotate_steps", type=int, default=int(os.environ.get("TOPO_FILL_ROTATE_STEPS", "8")))
    parser.add_argument("--fill_forward_steps", type=int, default=int(os.environ.get("TOPO_FILL_FORWARD_STEPS", "12")))
    parser.add_argument("--max_nodes", type=int, default=int(os.environ.get("TOPO_MAX_NODES", "30")))
    parser.add_argument("--stop_guard_dist", type=float, default=float(os.environ.get("TOPO_STOP_GUARD_DIST", "0.40")))
    parser.add_argument("--stop_near_m", type=float, default=float(os.environ.get("TOPO_STOP_NEAR_M", "1.25")))
    parser.add_argument(
        "--stop_near_hold_steps",
        type=int,
        default=int(os.environ.get("TOPO_STOP_NEAR_HOLD_STEPS", "8")),
    )
    parser.add_argument(
        "--stop_near_goal_prob",
        type=float,
        default=float(os.environ.get("TOPO_STOP_NEAR_GOAL_PROB", "0.25")),
    )
    parser.add_argument(
        "--stop_near_goal_score_min",
        type=float,
        default=float(os.environ.get("TOPO_STOP_NEAR_GOAL_SCORE_MIN", "0.05")),
    )
    parser.add_argument(
        "--stop_near_low_conf_guard",
        type=float,
        default=float(os.environ.get("TOPO_STOP_NEAR_LOW_CONF_GUARD", "0.18")),
    )
    parser.add_argument(
        "--doorway_burst_steps",
        type=int,
        default=int(os.environ.get("TOPO_DOORWAY_BURST_STEPS", "0")),
    )
    parser.add_argument("--use_est_pose", type=int, default=int(os.environ.get("TOPO_USE_EST_POSE", "0")))
    parser.add_argument("--odom_noise_trans", type=float, default=float(os.environ.get("TOPO_ODOM_NOISE_TRANS", "0.02")))
    parser.add_argument("--odom_noise_yaw_deg", type=float, default=float(os.environ.get("TOPO_ODOM_NOISE_YAW", "1.0")))
    parser.add_argument("--debug_capture", type=int, default=int(os.environ.get("TOPO_DEBUG_CAPTURE", "0")))
    parser.add_argument(
        "--debug_only_on_fail",
        type=int,
        default=int(os.environ.get("TOPO_DEBUG_ONLY_ON_FAIL", "1")),
    )
    parser.add_argument(
        "--debug_frame_stride",
        type=int,
        default=int(os.environ.get("TOPO_DEBUG_FRAME_STRIDE", "3")),
    )
    parser.add_argument(
        "--debug_ringbuf_steps",
        type=int,
        default=int(os.environ.get("TOPO_DEBUG_RINGBUF_STEPS", "220")),
    )
    parser.add_argument(
        "--debug_save_depth",
        type=int,
        default=int(os.environ.get("TOPO_DEBUG_SAVE_DEPTH", "0")),
    )
    parser.add_argument(
        "--debug_oracle_on_fail",
        type=int,
        default=int(os.environ.get("TOPO_DEBUG_ORACLE_ON_FAIL", "1")),
    )
    parser.add_argument(
        "--debug_capture_success_at_maxsteps",
        type=int,
        default=int(os.environ.get("TOPO_DEBUG_CAPTURE_SUCCESS_AT_MAXSTEPS", "1")),
    )
    parser.add_argument(
        "--debug_root",
        type=str,
        default=os.environ.get("TOPO_DEBUG_ROOT", ""),
    )
    parser.add_argument("--stall_window", type=int, default=int(os.environ.get("TOPO_STALL_WINDOW", "8")))
    parser.add_argument("--stall_min_delta", type=float, default=float(os.environ.get("TOPO_STALL_MIN_DELTA", "0.08")))
    parser.add_argument(
        "--recovery_mode",
        type=str,
        default=os.environ.get("TOPO_RECOVERY_MODE", "depth_turn"),
        choices=["depth_turn", "follower_burst", "hybrid"],
    )
    parser.add_argument(
        "--recovery_turn_steps",
        type=int,
        default=int(os.environ.get("TOPO_RECOVERY_TURN_STEPS", "3")),
    )
    parser.add_argument(
        "--recovery_follower_steps",
        type=int,
        default=int(os.environ.get("TOPO_RECOVERY_FOLLOWER_STEPS", "4")),
    )
    parser.add_argument(
        "--recovery_cooldown",
        type=int,
        default=int(os.environ.get("TOPO_RECOVERY_COOLDOWN", "5")),
    )
    args = parser.parse_args()
    env_controller = os.environ.get("TOPO_CONTROLLER", "").strip().lower()
    controller_mode = env_controller if env_controller in {"bearing", "follower"} else args.controller_mode

    run_navigation(
        config_path=args.config_path,
        graph_json=args.graph_json,
        embeds_npy=args.embeds_npy,
        node_meta_path=args.node_meta,
        query=args.query,
        out_dir=args.out_dir,
        max_steps=args.max_steps,
        model_path=args.model_path,
        pooling=args.pooling,
        vote_window=args.vote_window,
        relocalize_every=args.relocalize_every,
        stop_thresh=args.stop_thresh,
        turn_thresh=args.turn_thresh,
        node_reach_thresh=args.node_reach_thresh,
        controller_mode=controller_mode,
        run_id=args.run_id,
        start_episode_offset=args.start_episode_offset,
        vision_prompt=args.vision_prompt,
        goal_topk=args.goal_topk,
        loc_conf_thresh=args.loc_conf_thresh,
        belief_temp=args.belief_temp,
        belief_stay_prob=args.belief_stay_prob,
        belief_mix=args.belief_mix,
        belief_entropy_fill_thresh=args.belief_entropy_fill_thresh,
        belief_topk_print=args.belief_topk_print,
        belief_hop_radius=args.belief_hop_radius,
        belief_geo_radius=args.belief_geo_radius,
        belief_noncand_penalty=args.belief_noncand_penalty,
        belief_reset_low_conf=args.belief_reset_low_conf,
        belief_reset_high_entropy=args.belief_reset_high_entropy,
        kidnap_start=bool(args.kidnap_start),
        kidnap_seed=args.kidnap_seed,
        kidnap_min_nearest_node_dist=args.kidnap_min_nearest_node_dist,
        kidnap_max_nearest_node_dist=args.kidnap_max_nearest_node_dist,
        clip_embeds_npy=args.clip_embeds_npy,
        geo_embeds_npy=args.geo_embeds_npy,
        object_graph_json=args.object_graph_json,
        object_nodes_json=args.object_nodes_json,
        clip_model_name=args.clip_model_name,
        use_two_stage_goal=bool(args.use_two_stage_goal),
        room_shortlist_k=args.room_shortlist_k,
        object_topk=args.object_topk,
        room_min=args.room_min,
        goal_score_thresh=args.goal_score_thresh,
        fill_steps=args.fill_steps,
        fill_min_node_spacing=args.fill_min_node_spacing,
        fill_relocalize_low_conf_patience=args.fill_relocalize_low_conf_patience,
        fill_retry_nav_extra_steps=args.fill_retry_nav_extra_steps,
        room_query_topk=args.room_query_topk,
        loc_fuse_alpha=args.loc_fuse_alpha,
        loc_prior_alpha=args.loc_prior_alpha,
        loc_prior_sigma=args.loc_prior_sigma,
        loc_conf_margin_center=args.loc_conf_margin_center,
        loc_conf_gain=args.loc_conf_gain,
        geo_bins_r=args.geo_bins_r,
        geo_bins_theta=args.geo_bins_theta,
        geo_max_depth=args.geo_max_depth,
        fill_rotate_steps=args.fill_rotate_steps,
        fill_forward_steps=args.fill_forward_steps,
        max_nodes=args.max_nodes,
        stop_guard_dist=args.stop_guard_dist,
        stop_near_m=args.stop_near_m,
        stop_near_hold_steps=args.stop_near_hold_steps,
        stop_near_goal_prob=args.stop_near_goal_prob,
        stop_near_goal_score_min=args.stop_near_goal_score_min,
        stop_near_low_conf_guard=args.stop_near_low_conf_guard,
        doorway_burst_steps=args.doorway_burst_steps,
        use_est_pose=bool(args.use_est_pose),
        odom_noise_trans=args.odom_noise_trans,
        odom_noise_yaw_deg=args.odom_noise_yaw_deg,
        debug_capture=bool(args.debug_capture),
        debug_only_on_fail=bool(args.debug_only_on_fail),
        debug_frame_stride=args.debug_frame_stride,
        debug_ringbuf_steps=args.debug_ringbuf_steps,
        debug_save_depth=bool(args.debug_save_depth),
        debug_oracle_on_fail=bool(args.debug_oracle_on_fail),
        debug_capture_success_at_maxsteps=bool(args.debug_capture_success_at_maxsteps),
        debug_root=args.debug_root,
        stall_window=args.stall_window,
        stall_min_delta=args.stall_min_delta,
        recovery_mode=args.recovery_mode,
        recovery_turn_steps=args.recovery_turn_steps,
        recovery_follower_steps=args.recovery_follower_steps,
        recovery_cooldown=args.recovery_cooldown,
    )


if __name__ == "__main__":
    main()
