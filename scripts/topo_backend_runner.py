#!/usr/bin/env python
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from internnav.sim_backend.base import SimBackend
from internnav.topo.controller import BearingController
from internnav.topo.debug_capture import DebugRingBuffer
from internnav.topo.graph import TopoGraph, load_embeddings
from internnav.topo.local_planner import plan_grid_path
from internnav.topo.plan import dijkstra_path
from internnav.topo.retrieve_goal import retrieve_goal_topk


def _sigmoid(x: float) -> float:
    xx = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-xx)))


def _vote_node(history: Deque[int]) -> Tuple[int, float, Dict[int, int]]:
    counts = collections.Counter(history)
    node_id, count = counts.most_common(1)[0]
    conf = count / max(1, len(history))
    return int(node_id), float(conf), {int(k): int(v) for k, v in counts.items()}


def _build_idx_maps(node_meta: Dict) -> Tuple[Dict[int, int], Dict[int, int]]:
    idx_to_node = {}
    node_to_idx = {}
    for node_id_str, meta in node_meta.items():
        node_id = int(node_id_str)
        idx = int(meta["embed_idx"])
        idx_to_node[idx] = node_id
        node_to_idx[node_id] = idx
    return idx_to_node, node_to_idx


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
        p = np.asarray(node.position, dtype=np.float32)
        d = float(np.linalg.norm(position - p))
        if d < best_dist:
            best_dist = d
            best_id = int(nid)
    return int(best_id if best_id is not None else 0), float(best_dist)


def _load_scene_cfg(config_path: str, scene_id: str) -> Dict[str, object]:
    if not os.path.isfile(config_path):
        return {}
    payload = {}
    try:
        import yaml  # type: ignore

        with open(config_path, "r") as f:
            payload = yaml.safe_load(f) or {}
    except Exception:
        try:
            with open(config_path, "r") as f:
                payload = json.load(f)
        except Exception:
            payload = {}
    scenes = payload.get("scenes", []) if isinstance(payload, dict) else []
    for s in scenes:
        if str(s.get("scene_id", "")) == str(scene_id):
            return dict(s)
    return {}


class SimpleTopoEmbedder:
    """Dependency-light embedder for Isaac bring-up runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = int(max(8, dim))

    def _fit_dim(self, vec: np.ndarray) -> np.ndarray:
        x = np.asarray(vec, dtype=np.float32).reshape(-1)
        if x.shape[0] == self.dim:
            return x
        if x.shape[0] > self.dim:
            return x[: self.dim]
        rep = int(math.ceil(self.dim / max(1, x.shape[0])))
        return np.tile(x, rep)[: self.dim]

    def embed_image(self, rgb: np.ndarray, instruction: str = "") -> np.ndarray:
        arr = np.asarray(rgb, dtype=np.float32)
        if arr.ndim != 3:
            arr = np.zeros((32, 32, 3), dtype=np.float32)
        arr = arr / 255.0
        h, w, _ = arr.shape
        gh, gw = 8, 8
        ys = np.linspace(0, max(0, h - 1), gh).astype(np.int32)
        xs = np.linspace(0, max(0, w - 1), gw).astype(np.int32)
        patch = arr[np.ix_(ys, xs)]
        gray = patch.mean(axis=2).reshape(-1)
        stats = np.array(
            [
                float(arr[..., 0].mean()),
                float(arr[..., 1].mean()),
                float(arr[..., 2].mean()),
                float(arr[..., 0].std()),
                float(arr[..., 1].std()),
                float(arr[..., 2].std()),
            ],
            dtype=np.float32,
        )
        text_hint = self.embed_text(instruction or "")
        vec = np.concatenate([gray.astype(np.float32), stats, text_hint[:16]], axis=0)
        vec = self._fit_dim(vec)
        n = float(np.linalg.norm(vec) + 1e-6)
        return (vec / n).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        s = str(text or "").strip().lower().encode("utf-8")
        if not s:
            vec = np.zeros((self.dim,), dtype=np.float32)
            vec[0] = 1.0
            return vec
        h = hashlib.sha256(s).digest()
        base = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
        vec = np.tile(base, int(math.ceil(self.dim / base.shape[0])))[: self.dim]
        n = float(np.linalg.norm(vec) + 1e-6)
        return (vec / n).astype(np.float32)


def _build_embedders(args: argparse.Namespace, embed_dim: int):
    mode = str(args.embedder_mode).strip().lower()
    if mode == "auto":
        mode = "simple" if str(args.backend).strip().lower() == "isaac" else "internvla"
    if mode == "simple":
        emb = SimpleTopoEmbedder(dim=int(args.simple_embed_dim or embed_dim))
        return emb, emb, mode
    if mode == "clip":
        from internnav.topo.explore import TopoClipEmbedder

        clip = TopoClipEmbedder(model_name=args.clip_model_name)
        return clip, clip, mode
    if mode == "internvla":
        from internnav.topo.explore import TopoClipEmbedder, TopoEmbedder

        internvla = TopoEmbedder(model_path=args.model_path, pooling=args.pooling)
        clip = TopoClipEmbedder(model_name=args.clip_model_name)
        return internvla, clip, mode
    raise ValueError(f"Unsupported embedder_mode={args.embedder_mode}")


def _embed_image_any(embedder: Any, rgb: np.ndarray, instruction: str) -> np.ndarray:
    try:
        return embedder.embed_image(rgb, instruction=instruction)
    except TypeError:
        return embedder.embed_image(rgb)


def _make_backend(
    backend_name: str,
    scene_id: str,
    isaac_cfg_path: str,
    habitat_config_path: str,
    out_dir: str,
) -> SimBackend:
    name = str(backend_name).strip().lower()
    if name == "habitat":
        from internnav.sim_backend.habitat_backend import HabitatSimBackend

        return HabitatSimBackend(config_path=habitat_config_path, out_dir=out_dir)
    if name == "isaac":
        from internnav.sim_backend.isaac_backend import IsaacSimBackend

        scene_cfg = _load_scene_cfg(isaac_cfg_path, scene_id)
        allow_fallback = bool(int(os.environ.get("ISAAC_BACKEND_ALLOW_FALLBACK", "0")))
        return IsaacSimBackend(
            scene_id=scene_id,
            scene_cfg=scene_cfg,
            config_path=habitat_config_path,
            out_dir=out_dir,
            fallback_to_habitat=allow_fallback,
            dt_action=float(scene_cfg.get("dt_action", 0.5)) if scene_cfg else 0.5,
        )
    raise ValueError(f"Unsupported backend: {backend_name}")


def run_backend_navigation(args: argparse.Namespace) -> Dict[str, object]:
    os.makedirs(args.out_dir, exist_ok=True)

    graph = TopoGraph.load_json(args.graph_json)
    with open(args.node_meta, "r") as f:
        node_meta = json.load(f).get("node_meta", {})
    idx_to_node, node_to_idx = _build_idx_maps(node_meta)

    embeds = load_embeddings(args.embeds_npy)
    if embeds.ndim == 1:
        embeds = embeds[:, None]
    embeds_norm = embeds.astype(np.float32)
    embeds_norm = embeds_norm / (np.linalg.norm(embeds_norm, axis=1, keepdims=True) + 1e-6)

    clip_path = args.clip_embeds_npy if args.clip_embeds_npy else args.embeds_npy
    clip_embeds = load_embeddings(clip_path)
    if clip_embeds.ndim == 1:
        clip_embeds = clip_embeds[:, None]
    clip_embeds_norm = clip_embeds.astype(np.float32)
    clip_embeds_norm = clip_embeds_norm / (np.linalg.norm(clip_embeds_norm, axis=1, keepdims=True) + 1e-6)

    image_embedder, text_embedder, embedder_mode = _build_embedders(args=args, embed_dim=int(embeds_norm.shape[1]))
    print(
        f"[TOPO_BACKEND_EMBEDDER] backend={args.backend} mode={embedder_mode} dim={int(embeds_norm.shape[1])}",
        flush=True,
    )

    goal_spec = None
    goal_pose_override: Optional[np.ndarray] = None
    goal_node_override: Optional[int] = None
    query_text = str(args.query)
    if query_text.strip().startswith("{"):
        try:
            spec = json.loads(query_text)
            if isinstance(spec, dict) and "type" in spec and "value" in spec:
                goal_spec = spec
        except Exception:
            goal_spec = None
    if goal_spec is not None:
        gtype = str(goal_spec.get("type", "")).strip().lower()
        gval = goal_spec.get("value")
        if gtype == "text":
            query_text = str(gval)
            print(f"[TOPO_GOAL_SPEC] type=text value={query_text}", flush=True)
        elif gtype == "node_id":
            goal_node_override = int(gval)
            print(f"[TOPO_GOAL_SPEC] type=node_id value={goal_node_override}", flush=True)
        elif gtype == "pose":
            if isinstance(gval, dict):
                gx = float(gval.get("x", 0.0))
                gy = float(gval.get("y", 0.0))
                gz = float(gval.get("z", 0.0))
                goal_pose_override = np.array([gx, gy, gz], dtype=np.float32)
            print(f"[TOPO_GOAL_SPEC] type=pose value={gval}", flush=True)

    if goal_node_override is None:
        query_embed = text_embedder.embed_text(query_text)
        topk = retrieve_goal_topk(query_embed, clip_embeds_norm, idx_to_node, topk=max(3, int(args.goal_topk)))
        if len(topk) == 0:
            raise RuntimeError("Goal retrieval produced empty top-k.")
    else:
        topk = [(int(goal_node_override), 1.0)]

    goal_node = int(topk[0][0])
    goal_score = float(topk[0][1])
    if goal_pose_override is not None:
        goal_node, _ = _nearest_node(goal_pose_override, graph)
        goal_score = 1.0

    try:
        backend = _make_backend(
            backend_name=args.backend,
            scene_id=args.scene_id,
            isaac_cfg_path=args.isaac_config,
            habitat_config_path=args.config_path,
            out_dir=args.out_dir,
        )
    except Exception as e:
        if str(args.backend).strip().lower() == "isaac":
            reason = f"{type(e).__name__}:{str(e).replace(chr(10), ' ')[:240]}"
            print(f"[ISAAC_BACKEND_ERROR] reason={reason}", flush=True)
        raise
    backend_mode = getattr(backend, "backend_mode", args.backend)
    renderer_used = getattr(backend, "renderer_used", "unknown")

    node_xyz = np.stack(
        [np.asarray(node.position, dtype=np.float32) for _, node in sorted(graph.nodes.items())],
        axis=0,
    )
    start_spec = {
        "start_offset": int(args.start_episode_offset),
        "kidnap_start": 1 if int(args.kidnap_start) else 0,
        "kidnap_seed": int(args.kidnap_seed),
        "kidnap_min_nearest_node_dist": float(args.kidnap_min_nearest_node_dist),
        "kidnap_max_nearest_node_dist": float(args.kidnap_max_nearest_node_dist),
        "reference_nodes_xyz": node_xyz.tolist(),
    }
    obs = backend.reset(scene_id=args.scene_id, start_spec=start_spec)

    goal_pos = np.array(graph.nodes[goal_node].position, dtype=np.float32)
    if goal_pose_override is not None:
        goal_pos = np.array(goal_pose_override, dtype=np.float32)

    planner_waypoints: List[Tuple[float, float]] = []
    planner_ok = False
    planner_meta: Dict[str, float] = {}
    waypoint_idx = 0
    if str(args.controller_mode).strip().lower() == "waypoint_follow":
        if hasattr(backend, "get_bounds") and hasattr(backend, "get_obstacles"):
            bounds = list(getattr(backend, "get_bounds")())
            obstacles = list(getattr(backend, "get_obstacles")())
            ok, waypoints, meta = plan_grid_path(
                bounds=bounds,
                obstacles=obstacles,
                start=np.array([obs.pose.x, obs.pose.y, obs.pose.z], dtype=np.float32),
                goal=goal_pos,
                meters_per_cell=float(args.planner_grid_res),
                max_nodes=int(args.planner_max_nodes),
            )
            planner_meta = dict(meta)
            if ok and len(waypoints) > 0:
                planner_ok = True
                planner_waypoints = [(float(x), float(z)) for x, z in waypoints]
                print(
                    f"[ISAAC_LOCAL_PLANNER] mode=grid_astar ok=1 waypoints={len(planner_waypoints)} "
                    f"grid_res={float(args.planner_grid_res):.3f} meters_per_cell={float(args.planner_grid_res):.3f}",
                    flush=True,
                )
            else:
                print("[ISAAC_LOCAL_PLANNER_FAIL] reason=no_path", flush=True)
        else:
            print("[ISAAC_LOCAL_PLANNER_FAIL] reason=no_backend_bounds", flush=True)

    debug_frame_stride = int(args.debug_frame_stride)
    debug_placeholder = bool(int(args.debug_placeholder))
    if bool(args.debug_capture) and (bool(args.debug_only_on_fail) or bool(args.debug_capture_success_at_maxsteps)):
        debug_frame_stride = 1
    debug_buf = DebugRingBuffer(
        enabled=bool(args.debug_capture),
        ring_steps=int(args.debug_ringbuf_steps),
        frame_stride=debug_frame_stride,
        save_depth=bool(args.debug_save_depth),
        placeholder=debug_placeholder,
    )
    if bool(args.debug_capture):
        print(
            f"[TOPO_DEBUG_CAPTURE_CFG] stride={debug_frame_stride} placeholder={int(debug_placeholder)}",
            flush=True,
        )
        print(
            f"[TOPO_DEBUG_CAPTURE] run={args.run_id} only_on_fail={int(bool(args.debug_only_on_fail))} "
            f"ringbuf={int(args.debug_ringbuf_steps)} stride={int(debug_frame_stride)}",
            flush=True,
        )

    map_node, _ = _nearest_node(np.array([obs.pose.x, obs.pose.y, obs.pose.z], dtype=np.float32), graph)
    path_nodes = dijkstra_path(graph.edges, int(map_node), int(goal_node))
    if len(path_nodes) == 0:
        path_nodes = [int(map_node), int(goal_node)]
    if len(path_nodes) == 1:
        path_nodes = [int(map_node), int(goal_node)]
    path_index = 0
    next_node = int(path_nodes[min(1, len(path_nodes) - 1)])

    history: Deque[int] = collections.deque(maxlen=max(1, int(args.vote_window)))
    loc_conf_trace: List[float] = []
    action_hist = {"STOP": 0, "FWD": 0, "LEFT": 0, "RIGHT": 0}
    path_poses: List[Dict[str, object]] = []
    watchdog_trigger_count = 0
    relocalize_reset_count = 0
    doorway_burst_count = 0
    fill_triggers = 0

    turn_thresh_rad = float(args.turn_thresh)
    bearing_ctrl = BearingController(
        stop_thresh=float(args.stop_thresh),
        turn_high=turn_thresh_rad,
        turn_low=turn_thresh_rad * 0.6,
    )
    print(
        f"[TOPO_CTRL_UNITS] bearing_unit=rad turn_thresh_rad={turn_thresh_rad:.4f}",
        flush=True,
    )
    print("[TOPO_CTRL_FRAME] forward_axis=-Z yaw_sign=+", flush=True)
    dist_trace: Deque[float] = collections.deque(maxlen=max(3, int(args.stall_window)))
    goal_dist_trace: Deque[float] = collections.deque(maxlen=max(3, int(args.stall_window)))

    done = False
    success = False
    fail_reason: Optional[str] = None
    terminated_by: Optional[str] = None
    success_by: Optional[str] = None
    step = 0
    last_conf = 0.0
    last_entropy = 1.0
    last_topk: List[Tuple[int, float]] = []
    last_collision: Optional[bool] = None

    try:
        while not done:
            if step >= int(args.max_steps):
                fail_reason = "max_steps_reached"
                terminated_by = "max_steps"
                print(f"[TOPO_NAV_FAIL] reason={fail_reason} steps={step}", flush=True)
                break

            rgb = np.asarray(obs.rgb, dtype=np.uint8)
            depth = None if obs.depth is None else np.asarray(obs.depth, dtype=np.float32)
            img_embed = _embed_image_any(image_embedder, rgb=rgb, instruction=args.vision_prompt).astype(np.float32)
            img_embed = img_embed / (np.linalg.norm(img_embed) + 1e-6)
            sims = embeds_norm.dot(img_embed)
            best_idx = int(np.argmax(sims))
            local_node = int(idx_to_node.get(best_idx, map_node))
            history.append(local_node)
            voted_node, vote_conf, vote_map = _vote_node(history)
            map_node = int(voted_node)

            temp = max(1e-6, float(args.belief_temp))
            logits = sims / temp
            logits = logits - float(np.max(logits))
            probs = np.exp(logits).astype(np.float32)
            probs = probs / (np.sum(probs) + 1e-6)
            entropy = -float(np.sum(np.clip(probs, 1e-8, 1.0) * np.log(np.clip(probs, 1e-8, 1.0))))
            entropy = float(np.clip(entropy / max(1e-6, math.log(max(2, probs.shape[0]))), 0.0, 1.0))
            k = min(max(1, int(args.belief_topk_print)), probs.shape[0])
            kidx = np.argpartition(-probs, k - 1)[:k]
            kidx = kidx[np.argsort(-probs[kidx])]
            last_topk = [(int(idx_to_node.get(int(i), -1)), float(probs[int(i)])) for i in kidx.tolist()]
            top1 = float(probs[kidx[0]]) if k > 0 else 0.0
            top2 = float(probs[kidx[1]]) if k > 1 else 0.0
            margin_conf = _sigmoid((top1 - top2) * 8.0)
            last_conf = float(np.clip(0.5 * vote_conf + 0.5 * margin_conf, 0.0, 1.0))
            last_entropy = float(entropy)

            print(
                f"[TOPO_RELOCALIZE_BELIEF] run={args.run_id} step={step} map={map_node} "
                f"loc_conf={last_conf:.3f} entropy={last_entropy:.3f} "
                f"topK={[(n, round(p, 4)) for n, p in last_topk]}",
                flush=True,
            )

            path_nodes = dijkstra_path(graph.edges, int(map_node), int(goal_node))
            if len(path_nodes) == 0:
                path_nodes = [int(map_node), int(goal_node)]
            if len(path_nodes) == 1:
                path_nodes = [int(map_node), int(goal_node)]
            path_index = 0
            next_node = int(path_nodes[min(1, len(path_nodes) - 1)])
            print(f"[TOPO_PLAN] path_len={len(path_nodes)} next={next_node}", flush=True)

            curr_pos = np.array([obs.pose.x, obs.pose.y, obs.pose.z], dtype=np.float32)
            target_pos = np.array(graph.nodes[next_node].position, dtype=np.float32)
            if str(args.controller_mode).strip().lower() == "waypoint_follow" and planner_ok:
                if waypoint_idx >= len(planner_waypoints):
                    target_pos = np.array(goal_pos, dtype=np.float32)
                else:
                    wx, wz = planner_waypoints[waypoint_idx]
                    target_pos = np.array([wx, float(curr_pos[1]), wz], dtype=np.float32)
            goal_pos_now = np.array(goal_pos, dtype=np.float32)
            bearing, dist = _bearing_distance_from_yaw(curr_pos, float(obs.pose.yaw), target_pos)
            _, dist_goal = _bearing_distance_from_yaw(curr_pos, float(obs.pose.yaw), goal_pos_now)

            if str(args.controller_mode).strip().lower() == "waypoint_follow" and planner_ok:
                if float(dist) <= float(args.waypoint_reach_thresh):
                    waypoint_idx += 1
                    if waypoint_idx >= len(planner_waypoints):
                        success = True
                        terminated_by = "success_stop"
                        success_by = "dist_threshold"
                        print(
                            f"[TOPO_NAV_SUCCESS] step={step} reached_node={goal_node} dist={dist_goal:.3f}",
                            flush=True,
                        )
                        break
            if float(dist) <= float(args.node_reach_thresh):
                if int(next_node) == int(goal_node):
                    success = True
                    terminated_by = "success_stop"
                    success_by = "dist_threshold"
                    print(
                        f"[TOPO_NAV_SUCCESS] step={step} reached_node={goal_node} dist={dist:.3f}",
                        flush=True,
                    )
                    break
                if len(path_nodes) > 2:
                    next_node = int(path_nodes[2])
                    target_pos = np.array(graph.nodes[next_node].position, dtype=np.float32)
                    bearing, dist = _bearing_distance_from_yaw(curr_pos, float(obs.pose.yaw), target_pos)

            if (
                float(dist_goal) <= float(args.stop_near_m)
                and float(goal_score) >= float(args.stop_near_goal_score_min)
                and float(last_conf) >= float(args.stop_near_loc_conf_min)
            ):
                success = True
                terminated_by = "success_stop"
                success_by = "dist_threshold"
                print(
                    f"[TOPO_NEAR_GOAL_STOP] run={args.run_id} step={step} dist={dist_goal:.3f} "
                    f"goal_prob={top1:.3f} loc_conf={last_conf:.3f} T=1",
                    flush=True,
                )
                break

            if args.controller_mode == "follower":
                if dist <= float(args.stop_thresh):
                    action, action_name = 0, "STOP"
                elif abs(bearing) > turn_thresh_rad * 0.8:
                    action, action_name = (2, "LEFT") if bearing < 0.0 else (3, "RIGHT")
                else:
                    action, action_name = 1, "FORWARD"
            else:
                action, action_name = bearing_ctrl.decide(bearing, dist)

            prev_pos = np.array([obs.pose.x, obs.pose.y, obs.pose.z], dtype=np.float32)
            obs_next, env_done, info = backend.step(int(action))
            new_pos = np.array([obs_next.pose.x, obs_next.pose.y, obs_next.pose.z], dtype=np.float32)
            moved = float(np.linalg.norm(new_pos - prev_pos))
            fwd_no_motion = bool(int(action) == 1 and moved < 0.01)
            if args.controller_mode != "follower":
                bearing_ctrl.update_after_step(int(action), moved)
            last_collision = obs_next.collision if obs_next.collision is not None else backend.get_collision_flag()

            dist_trace.append(float(dist))
            goal_dist_trace.append(float(dist_goal))
            if len(dist_trace) == dist_trace.maxlen:
                start_d = float(dist_trace[0])
                best_d = float(min(dist_trace))
                if (start_d - best_d) < float(args.stall_min_delta):
                    watchdog_trigger_count += 1
                    print(
                        f"[TOPO_PROGRESS_WATCHDOG] run={args.run_id} step={step} reason=no_progress "
                        f"dist_subgoal={dist:.3f} dist_goal={dist_goal:.3f} mode={args.controller_mode}",
                        flush=True,
                    )

            action_hist[action_name if action_name != "FORWARD" else "FWD"] += 1
            room_id = int(node_meta.get(str(int(map_node)), {}).get("room_id", -1))
            rec = {
                "step_idx": int(step),
                "run_id": str(args.run_id),
                "scene_id": str(args.scene_id),
                "query": str(args.query),
                "condition": str(args.condition),
                "controller_mode": str(args.controller_mode),
                "action_id": int(action),
                "action": str(action_name if action_name != "FORWARD" else "FWD"),
                "pose_used": {
                    "x": float(obs.pose.x),
                    "y": float(obs.pose.y),
                    "z": float(obs.pose.z),
                    "yaw": float(obs.pose.yaw),
                    "source": str(args.condition),
                },
                "map_node_id": int(map_node),
                "subgoal_node_id": int(next_node),
                "goal_node_id": int(goal_node),
                "waypoint_idx": int(waypoint_idx),
                "planner_ok": int(bool(planner_ok)),
                "dist_to_subgoal": float(dist),
                "bearing_to_subgoal": float(bearing),
                "dist_to_goal_node": float(dist_goal),
                "loc_conf": float(last_conf),
                "belief_entropy": float(last_entropy),
                "belief_topk": [{"node": int(n), "prob": float(p)} for n, p in last_topk],
                "room_id_current": int(room_id),
                "recovery_event": ["watchdog_no_progress"] if watchdog_trigger_count > 0 else [],
                "fwd_no_motion": int(bool(fwd_no_motion)),
                "collision": None if last_collision is None else int(bool(last_collision)),
                "goal_score": float(goal_score),
            }
            debug_buf.push(record=rec, rgb=rgb, depth=depth, event_frame=bool(fwd_no_motion))

            path_poses.append(
                {
                    "step": int(step),
                    "x": float(obs.pose.x),
                    "y": float(obs.pose.y),
                    "z": float(obs.pose.z),
                    "yaw": float(obs.pose.yaw),
                    "map_node": int(map_node),
                    "next_node": int(next_node),
                    "goal_node": int(goal_node),
                    "dist_goal": float(dist_goal),
                }
            )
            print(
                f"[TOPO_CTRL] mode={args.controller_mode} step={step} bearing={bearing:.3f} dist={dist:.3f} "
                f"action={action_name} stuck={int(bearing_ctrl.stuck_state()) if args.controller_mode != 'follower' else 0} "
                f"node={map_node} next={next_node}",
                flush=True,
            )
            obs = obs_next
            done = bool(env_done)
            if done and not success:
                fail_reason = "env_done"
                terminated_by = "early_stop"
                print(f"[TOPO_NAV_FAIL] reason={fail_reason} steps={step}", flush=True)
                break
            step += 1
    finally:
        pass

    final_pose = backend.get_pose()
    goal_pos = np.array(graph.nodes[goal_node].position, dtype=np.float32)
    final_dist = float(np.linalg.norm(np.array([final_pose.x, final_pose.y, final_pose.z], dtype=np.float32) - goal_pos))
    if success and terminated_by is None:
        terminated_by = "success_stop"
        success_by = "stop"
    if not success and fail_reason is None:
        fail_reason = "max_steps_reached" if terminated_by == "max_steps" else "unknown"
    if success:
        fail_reason = None

    result = {
        "run_id": str(args.run_id),
        "scene_id": str(args.scene_id),
        "query": str(args.query),
        "backend": str(args.backend),
        "backend_mode": str(backend_mode),
        "renderer_used": str(renderer_used),
        "success": bool(success),
        "fail_reason": fail_reason if not success else None,
        "terminated_by": str(terminated_by or ("success_stop" if success else "max_steps")),
        "success_by": success_by if success else None,
        "steps_used": int(step),
        "final_dist_to_goal_node": float(final_dist),
        "goal_node": int(goal_node),
        "goal_score": float(goal_score),
        "action_histogram": {str(k): int(v) for k, v in action_hist.items()},
        "avg_loc_conf": float(np.mean(loc_conf_trace)) if len(loc_conf_trace) > 0 else float(last_conf),
        "watchdog_trigger_count": int(watchdog_trigger_count),
        "relocalize_reset_count": int(relocalize_reset_count),
        "doorway_burst_count": int(doorway_burst_count),
        "belief_entropy_summary": {
            "mean": float(last_entropy),
            "max": float(last_entropy),
            "min": float(last_entropy),
        },
        "fill_triggers": int(fill_triggers),
    }

    run_debug_dir = os.path.join(
        args.debug_root,
        str(args.condition),
        str(args.run_id),
    )
    os.makedirs(run_debug_dir, exist_ok=True)

    oracle_payload = {
        "oracle_available": 0,
        "oracle_success": 0,
        "oracle_geodesic_start": None,
        "oracle_geodesic_final": None,
        "oracle_steps_used": 0,
    }
    oracle_always = bool(int(os.environ.get("TOPO_DEBUG_ORACLE_ALWAYS", "0"))) or int(
        args.debug_oracle_on_fail
    ) > 1
    need_oracle = oracle_always or (
        bool(args.debug_oracle_on_fail)
        and (
            (not success)
            or (
                success
                and str(result["terminated_by"]) == "max_steps"
                and bool(args.debug_capture_success_at_maxsteps)
            )
        )
    )
    if need_oracle:
        oracle_payload = backend.oracle_probe(
            goal_position=goal_pos,
            max_steps=int(args.max_steps),
            success_dist=float(args.node_reach_thresh),
        )

    should_flush = bool(args.debug_capture) and (
        (not bool(args.debug_only_on_fail))
        or (not success)
        or (
            success
            and str(result["terminated_by"]) == "max_steps"
            and bool(args.debug_capture_success_at_maxsteps)
        )
    )
    if should_flush:
        meta = {
            "run_id": str(args.run_id),
            "scene_id": str(args.scene_id),
            "query": str(args.query),
            "condition": str(args.condition),
            "success": bool(success),
            "terminated_by": str(result["terminated_by"]),
            "fail_reason": result["fail_reason"],
            "controller_mode": str(args.controller_mode),
            "renderer_used": str(renderer_used),
            "max_steps": int(args.max_steps),
            "node_reach_thresh": float(args.node_reach_thresh),
            "goal_node": int(goal_node),
            "goal_score": float(goal_score),
            "settings": {
                "debug_only_on_fail": int(bool(args.debug_only_on_fail)),
                "debug_frame_stride": int(args.debug_frame_stride),
                "debug_ringbuf_steps": int(args.debug_ringbuf_steps),
                "debug_save_depth": int(bool(args.debug_save_depth)),
                "debug_capture_success_at_maxsteps": int(bool(args.debug_capture_success_at_maxsteps)),
            },
            "oracle": oracle_payload,
            "planner": {
                "ok": int(bool(planner_ok)),
                "waypoints": planner_waypoints,
                "grid_res": float(args.planner_grid_res),
            },
            "build_dir": str(os.path.dirname(args.graph_json)),
            "result_path": os.path.join(args.out_dir, f"RESULT_{args.run_id}.json"),
        }
        saved = debug_buf.flush(out_dir=run_debug_dir, meta=meta)
        print(
            f"[TOPO_DEBUG_FLUSH] run={args.run_id} success={int(bool(success))} "
            f"terminated_by={result['terminated_by']} saved_steps={int(saved.get('saved_steps', 0))} "
            f"saved_frames={int(saved.get('saved_frames', 0))} dir={run_debug_dir}",
            flush=True,
        )

    result_path = os.path.join(args.out_dir, f"RESULT_{args.run_id}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    path_poses = {
        "run_id": str(args.run_id),
        "scene_id": str(args.scene_id),
        "query": str(args.query),
        "goal_node": int(goal_node),
        "poses": path_poses,
    }
    with open(os.path.join(args.out_dir, f"path_poses_{args.run_id}.json"), "w") as f:
        json.dump(path_poses, f, indent=2)

    print(
        f"[TOPO_ACTION_HIST] run={args.run_id} mode={args.controller_mode} hist={json.dumps(result['action_histogram'])}",
        flush=True,
    )
    backend.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Backend-agnostic topo runner (habitat|isaac).")
    ap.add_argument("--backend", type=str, default="habitat", choices=["habitat", "isaac"])
    ap.add_argument("--condition", type=str, default="gt_pose")
    ap.add_argument("--scene_id", type=str, required=True)
    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--graph_json", type=str, required=True)
    ap.add_argument("--embeds_npy", type=str, required=True)
    ap.add_argument("--clip_embeds_npy", type=str, default="")
    ap.add_argument("--node_meta", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--debug_root", type=str, required=True)
    ap.add_argument("--isaac_config", type=str, default="configs/isaac_scenes_v21.yaml")
    ap.add_argument("--config_path", type=str, default="repo/InternNav/scripts/eval/configs/vln_r2r.yaml")
    ap.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-DualVLN")
    ap.add_argument("--pooling", type=str, default="mean")
    ap.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument(
        "--embedder_mode",
        type=str,
        default=os.environ.get("TOPO_BACKEND_EMBEDDER_MODE", "auto"),
        choices=["auto", "internvla", "clip", "simple"],
    )
    ap.add_argument(
        "--simple_embed_dim",
        type=int,
        default=int(os.environ.get("TOPO_BACKEND_SIMPLE_EMBED_DIM", "64")),
    )
    ap.add_argument("--vision_prompt", type=str, default="")
    ap.add_argument("--controller_mode", type=str, default="bearing", choices=["bearing", "follower", "waypoint_follow"])
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--goal_topk", type=int, default=5)
    ap.add_argument("--vote_window", type=int, default=5)
    ap.add_argument("--start_episode_offset", type=int, default=0)
    ap.add_argument("--kidnap_start", type=int, default=1)
    ap.add_argument("--kidnap_seed", type=int, default=7)
    ap.add_argument("--kidnap_min_nearest_node_dist", type=float, default=0.5)
    ap.add_argument("--kidnap_max_nearest_node_dist", type=float, default=3.5)
    ap.add_argument("--node_reach_thresh", type=float, default=0.8)
    ap.add_argument("--stop_thresh", type=float, default=0.35)
    ap.add_argument("--turn_thresh", type=float, default=0.45)
    ap.add_argument("--planner_grid_res", type=float, default=float(os.environ.get("ISAAC_PLANNER_RES", "0.25")))
    ap.add_argument("--planner_max_nodes", type=int, default=int(os.environ.get("ISAAC_PLANNER_MAX_NODES", "20000")))
    ap.add_argument("--waypoint_reach_thresh", type=float, default=float(os.environ.get("ISAAC_WAYPOINT_REACH", "0.45")))
    ap.add_argument("--stall_window", type=int, default=20)
    ap.add_argument("--stall_min_delta", type=float, default=0.1)
    ap.add_argument("--stop_near_m", type=float, default=1.25)
    ap.add_argument("--stop_near_goal_score_min", type=float, default=0.05)
    ap.add_argument("--stop_near_loc_conf_min", type=float, default=0.35)
    ap.add_argument("--belief_temp", type=float, default=0.3)
    ap.add_argument("--belief_topk_print", type=int, default=4)
    ap.add_argument("--debug_capture", type=int, default=int(os.environ.get("TOPO_DEBUG_CAPTURE", "0")))
    ap.add_argument("--debug_only_on_fail", type=int, default=int(os.environ.get("TOPO_DEBUG_ONLY_ON_FAIL", "1")))
    ap.add_argument("--debug_frame_stride", type=int, default=int(os.environ.get("TOPO_DEBUG_FRAME_STRIDE", "3")))
    ap.add_argument("--debug_ringbuf_steps", type=int, default=int(os.environ.get("TOPO_DEBUG_RINGBUF_STEPS", "220")))
    ap.add_argument("--debug_save_depth", type=int, default=int(os.environ.get("TOPO_DEBUG_SAVE_DEPTH", "0")))
    ap.add_argument("--debug_placeholder", type=int, default=int(os.environ.get("TOPO_DEBUG_PLACEHOLDER", "1")))
    ap.add_argument("--debug_oracle_on_fail", type=int, default=int(os.environ.get("TOPO_DEBUG_ORACLE_ON_FAIL", "1")))
    ap.add_argument(
        "--debug_capture_success_at_maxsteps",
        type=int,
        default=int(os.environ.get("TOPO_DEBUG_CAPTURE_SUCCESS_AT_MAXSTEPS", "1")),
    )
    args = ap.parse_args()
    run_backend_navigation(args)


if __name__ == "__main__":
    main()
