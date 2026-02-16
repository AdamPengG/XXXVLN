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


def _wrap_pi(x: float) -> float:
    return float((float(x) + math.pi) % (2.0 * math.pi) - math.pi)


def _count_sign_flips(vals: List[float], eps: float) -> int:
    signs: List[int] = []
    for v in vals:
        if abs(float(v)) < float(eps):
            continue
        signs.append(1 if v > 0.0 else -1)
    if len(signs) <= 1:
        return 0
    flips = 0
    prev = signs[0]
    for s in signs[1:]:
        if s != prev:
            flips += 1
        prev = s
    return int(flips)


def _parse_offsets_deg(text: str) -> List[float]:
    vals: List[float] = []
    for part in str(text).split(","):
        tok = part.strip()
        if tok == "":
            continue
        try:
            vals.append(float(tok))
        except Exception:
            continue
    if 0.0 not in vals:
        vals.insert(0, 0.0)
    uniq = []
    seen = set()
    for v in vals:
        key = round(float(v), 6)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(float(v))
    return uniq if len(uniq) > 0 else [0.0]


def _action_id_to_name(action_id: int) -> str:
    return {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}.get(int(action_id), "UNKNOWN")


def _collect_depth_candidates(obs: Any, backend: SimBackend) -> List[Tuple[str, np.ndarray]]:
    candidates: List[Tuple[str, np.ndarray]] = []
    depth_obj = getattr(obs, "depth", None)
    if depth_obj is not None:
        try:
            d = np.asarray(depth_obj, dtype=np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            if d.ndim == 2 and d.size > 0:
                candidates.append(("obs.depth", d))
        except Exception:
            pass
    info = getattr(obs, "info", {}) if hasattr(obs, "info") else {}
    if isinstance(info, dict):
        for key in ("depth", "camera_depth", "depth_m", "depth_frame"):
            if key not in info or info[key] is None:
                continue
            try:
                d = np.asarray(info[key], dtype=np.float32)
                if d.ndim == 3:
                    d = d[..., 0]
                if d.ndim == 2 and d.size > 0:
                    candidates.append((f"obs.info.{key}", d))
            except Exception:
                continue
    try:
        d2 = backend.get_depth()
        if d2 is not None:
            d = np.asarray(d2, dtype=np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            if d.ndim == 2 and d.size > 0:
                candidates.append(("backend.get_depth", d))
    except Exception:
        pass
    # stable order + dedup by key
    out: List[Tuple[str, np.ndarray]] = []
    seen = set()
    for k, v in candidates:
        if k in seen:
            continue
        seen.add(k)
        out.append((k, v))
    return out


def _extract_depth_frame(obs: Any, backend: SimBackend, preferred_key: str = "") -> Tuple[Optional[np.ndarray], str]:
    candidates = _collect_depth_candidates(obs=obs, backend=backend)
    if len(candidates) == 0:
        return None, ""
    pref = str(preferred_key or "").strip()
    if pref and pref.lower() != "auto":
        for key, arr in candidates:
            if key == pref or key.endswith(pref):
                return arr, key
        return None, ""
    key, arr = candidates[0]
    return arr, key


def _depth_strip_clearance(
    depth: np.ndarray,
    col_center: int,
    strip_w: int,
    y0: int,
    y1: int,
    pctl: float,
    min_valid_ratio: float,
) -> Tuple[float, float]:
    h, w = int(depth.shape[0]), int(depth.shape[1])
    half = max(1, strip_w // 2)
    x0 = max(0, int(col_center - half))
    x1 = min(w, int(col_center + half + 1))
    if y1 <= y0 or x1 <= x0:
        return 0.0, 0.0
    patch = depth[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0, 0.0
    valid = np.isfinite(patch) & (patch > 1e-4)
    valid_ratio = float(valid.mean())
    if valid_ratio < float(min_valid_ratio):
        return 0.0, valid_ratio
    vals = patch[valid]
    clearance = float(np.percentile(vals, float(np.clip(pctl, 0.0, 100.0))))
    return clearance, valid_ratio


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
    camera_info: Dict[str, object] = {}
    if hasattr(backend, "get_camera_info"):
        try:
            camera_info = dict(getattr(backend, "get_camera_info")())
        except Exception:
            camera_info = {}

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
    v33a_stuck_triggers = 0
    v33a_recovery_count = 0
    v33a_stuck_fail = 0

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
    print(
        f"[V33A_STUCK_CFG] window={int(args.stuck_window)} dist_eps={float(args.stuck_dist_eps):.3f} "
        f"yaw_eps={float(args.oscillation_yaw_eps_deg):.1f}",
        flush=True,
    )
    v33b_enabled = bool(int(args.v33b_enable))
    v33b_depth_ready = False
    v33b_depth_disabled = False
    v33b_depth_source = ""
    v33b_depth_caps_logged = False
    v33b_depth_stats_logged = False
    v33b_blocked_detected_count = 0
    v33b_override_count = 0
    v33b_blocked_no_override_count = 0
    v33b_blocked_no_override_reasons: collections.Counter[str] = collections.Counter()
    v33b_blocked_forward_count = 0
    v33b_blocked_forward_overridden_count = 0
    v33b_blocked_forward_pass_through_count = 0
    v33b_blocked_forward_reasons: collections.Counter[str] = collections.Counter()
    v33b_doorway_triggers = 0
    v33b_offsets = _parse_offsets_deg(args.v33b_offsets_deg)
    v33b_fov_deg = float(camera_info.get("camera_fov_deg", 90.0)) if isinstance(camera_info, dict) else 90.0
    v33b_near = float(camera_info.get("camera_near", 0.05)) if isinstance(camera_info, dict) else 0.05
    v33b_far = float(camera_info.get("camera_far", 50.0)) if isinstance(camera_info, dict) else 50.0
    v33b_steer_latch_side = 0
    v33b_steer_latch_steps = 0
    v33b_depth_key_req = str(os.environ.get("V33B_DEPTH_KEY", "obs.depth")).strip() or "obs.depth"
    v33b_depth_key_used = ""
    v33b_depth_missing_key = False
    v33b_depth_prev_roi: Optional[np.ndarray] = None
    v33b_depth_stale_delta_streak = 0
    v33b_depth_same_stats_k = 0
    v33b_depth_prev_blocked_stats: Optional[Tuple[float, float, float]] = None
    v33b_depth_unreliable = False
    v33b_depth_stale_pose_moved = False
    v33b_depth_hash_prev = ""
    v33b_rgb_hash_prev = ""
    v33b_depth_hash_repeat = 0
    v33b_pose_move_k = 0
    v33b_stationary_k = 0
    v33b_robot_stationary_events = 0
    v33b_last_step_dpos = 0.0
    v33b_last_step_dyaw_deg = 0.0
    v33b_collision_hist: Deque[int] = collections.deque(maxlen=6)
    v33b_probe_failures = 0
    if v33b_enabled:
        print(
            f"[V33B_CFG] enable=1 require_depth={int(args.v33b_require_depth)} clearance_m={float(args.v33b_clearance_m):.2f} "
            f"offsets={v33b_offsets} doorway={int(args.v33b_doorway_trigger)} depth_key={v33b_depth_key_req} "
            f"stale_dpos_eps={float(args.v33b_stale_dpos_eps):.3f} stale_dyaw_eps={float(args.v33b_stale_dyaw_eps):.1f} "
            f"stale_k={int(args.v33b_stale_k)} stationary_k={int(args.v33b_stationary_k)}",
            flush=True,
        )
    dist_trace: Deque[float] = collections.deque(maxlen=max(3, int(args.stall_window)))
    goal_dist_trace: Deque[float] = collections.deque(maxlen=max(3, int(args.stall_window)))
    stuck_pos_trace: Deque[np.ndarray] = collections.deque(maxlen=max(4, int(args.stuck_window) + 1))
    stuck_dist_trace: Deque[float] = collections.deque(maxlen=max(4, int(args.stuck_window)))
    yaw_delta_trace: Deque[float] = collections.deque(maxlen=max(4, int(args.stuck_window)))
    recovery_queue: Deque[int] = collections.deque()
    recovery_cooldown = 0

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
            stuck_pos_trace.append(np.array([float(curr_pos[0]), float(curr_pos[2])], dtype=np.float32))
            stuck_dist_trace.append(float(dist))

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

            stuck_triggered = False
            stuck_type = ""
            stuck_details = ""
            if recovery_cooldown > 0:
                recovery_cooldown -= 1
            if not recovery_queue and recovery_cooldown <= 0 and len(stuck_pos_trace) >= max(3, int(args.stuck_window)):
                seg = list(stuck_pos_trace)
                cum_move = 0.0
                for ii in range(1, len(seg)):
                    cum_move += float(np.linalg.norm(seg[ii] - seg[ii - 1]))
                subgoal_improve = 0.0
                if len(stuck_dist_trace) > 0:
                    subgoal_improve = float(stuck_dist_trace[0] - min(stuck_dist_trace))
                flips = _count_sign_flips(list(yaw_delta_trace), eps=math.radians(float(args.oscillation_yaw_eps_deg)))
                turn_big = sum(1 for yy in yaw_delta_trace if abs(float(yy)) >= math.radians(float(args.oscillation_yaw_eps_deg)))
                stuck_plain = cum_move < float(args.stuck_dist_eps) and subgoal_improve < float(args.stall_min_delta)
                oscillation = (
                    turn_big >= int(args.oscillation_min_turns)
                    and flips >= int(args.oscillation_flip_min)
                    and subgoal_improve < float(args.stall_min_delta)
                )
                if stuck_plain or oscillation:
                    stuck_triggered = True
                    stuck_type = "oscillation" if oscillation else "stuck"
                    stuck_details = (
                        f"cum_move={cum_move:.3f},subgoal_improve={subgoal_improve:.3f},"
                        f"flips={flips},turn_big={turn_big}"
                    )

            if stuck_triggered:
                v33a_stuck_triggers += 1
                print(
                    f"[V33A_STUCK] trigger=1 type={stuck_type} step={step} details={stuck_details}",
                    flush=True,
                )
                if v33a_recovery_count >= int(args.stuck_max_recoveries):
                    v33a_stuck_fail = 1
                    fail_reason = "stuck"
                    terminated_by = "early_stop"
                    print(
                        f"[V33A_RECOVERY] giveup count={v33a_recovery_count} fail_type=stuck",
                        flush=True,
                    )
                    break
                v33a_recovery_count += 1
                k = max(1, int(args.stuck_recovery_scan_k))
                recovery_queue = collections.deque([2] * k + [3] * (2 * k) + [2] * k)
                recovery_cooldown = max(2, int(args.stuck_window) // 2)
                print(f"[V33A_RECOVERY] start step={step} kind=scan k={k}", flush=True)
                replan_ok = False
                replan_reason = "graph_replan"
                if str(args.controller_mode).strip().lower() == "waypoint_follow":
                    if hasattr(backend, "get_bounds") and hasattr(backend, "get_obstacles"):
                        ok, waypoints, meta = plan_grid_path(
                            bounds=list(getattr(backend, "get_bounds")()),
                            obstacles=list(getattr(backend, "get_obstacles")()),
                            start=np.array([obs.pose.x, obs.pose.y, obs.pose.z], dtype=np.float32),
                            goal=goal_pos,
                            meters_per_cell=float(args.planner_grid_res),
                            max_nodes=int(args.planner_max_nodes),
                        )
                        planner_meta = dict(meta)
                        if ok and len(waypoints) > 0:
                            planner_ok = True
                            planner_waypoints = [(float(x), float(z)) for x, z in waypoints]
                            waypoint_idx = 0
                            replan_ok = True
                            replan_reason = f"planner_waypoints={len(planner_waypoints)}"
                        else:
                            replan_reason = "planner_no_path"
                    else:
                        replan_reason = "planner_backend_missing"
                else:
                    replan_ok = True
                print(f"[V33A_RECOVERY] replan ok={int(replan_ok)} reason={replan_reason}", flush=True)

            recovery_event_labels: List[str] = []
            if recovery_queue:
                base_action = int(recovery_queue.popleft())
                base_action_name = _action_id_to_name(base_action)
                recovery_event_labels.append("v33a_recovery_scan")
            elif args.controller_mode == "follower":
                if dist <= float(args.stop_thresh):
                    base_action, base_action_name = 0, "STOP"
                elif abs(bearing) > turn_thresh_rad * 0.8:
                    base_action, base_action_name = (2, "LEFT") if bearing < 0.0 else (3, "RIGHT")
                else:
                    base_action, base_action_name = 1, "FORWARD"
            else:
                base_action, base_action_name = bearing_ctrl.decide(bearing, dist)

            action = int(base_action)
            action_name = str(base_action_name)
            v33b_forward_probe_applied = False

            # v33b: depth/costmap local avoidance as an action filter.
            if v33b_enabled and not v33b_depth_disabled:
                print(
                    f"[V33B_POSE_DELTA] step={step} dpos_m={float(v33b_last_step_dpos):.4f} "
                    f"dyaw_deg={float(v33b_last_step_dyaw_deg):.3f}",
                    flush=True,
                )
                if step == 0:
                    cands = _collect_depth_candidates(obs=obs, backend=backend)
                    key_desc: List[str] = []
                    for key, arr in cands:
                        aa = np.asarray(arr, dtype=np.float32)
                        valid = np.isfinite(aa)
                        if valid.any():
                            min_v = float(np.min(aa[valid]))
                            max_v = float(np.max(aa[valid]))
                            mean_v = float(np.mean(aa[valid]))
                        else:
                            min_v = max_v = mean_v = float("nan")
                        key_desc.append(
                            f"{key}:shape={list(aa.shape)} dtype={aa.dtype} min={min_v:.4f} max={max_v:.4f} mean={mean_v:.4f}"
                        )
                    print(f"[V33B_DEPTH_KEYS] keys={key_desc}", flush=True)
                depth_frame, depth_source = _extract_depth_frame(obs, backend, preferred_key=v33b_depth_key_req)
                if depth_frame is None:
                    v33b_depth_missing_key = True
                    if not v33b_depth_caps_logged:
                        avail = [k for k, _ in _collect_depth_candidates(obs=obs, backend=backend)]
                        print(
                            f"[V33B_CAPS] depth=0 reason=depth_missing_key key={v33b_depth_key_req} available_keys={avail}",
                            flush=True,
                        )
                        v33b_depth_caps_logged = True
                    if bool(args.v33b_require_depth):
                        fail_reason = "backend_caps_depth_missing_key"
                        terminated_by = "early_stop"
                        print(
                            f"[TOPO_NAV_FAIL] reason={fail_reason} steps={step}",
                            flush=True,
                        )
                        break
                    v33b_depth_disabled = True
                    print(
                        f"[V33B_LOCAL_AVOID] step={step} skip=1 blocked=0 override=0 "
                        f"reason=depth_missing_key action_in={base_action_name} action_out={action_name}",
                        flush=True,
                    )
                else:
                    d = np.asarray(depth_frame, dtype=np.float32)
                    if d.ndim == 3:
                        d = d[..., 0]
                    v33b_depth_ready = True
                    v33b_depth_source = str(depth_source)
                    v33b_depth_key_used = str(depth_source)
                    if step == 0:
                        print(
                            f"[V33B_DEPTH_KEY_USED] key={v33b_depth_key_used} shape={list(d.shape)} dtype={d.dtype}",
                            flush=True,
                        )
                    if (not v33b_depth_caps_logged) and d.ndim == 2:
                        print(
                            f"[V33B_CAPS] depth=1 h={int(d.shape[0])} w={int(d.shape[1])} "
                            f"fov_deg={float(v33b_fov_deg):.1f} source=obs_key:{depth_source}",
                            flush=True,
                        )
                        v33b_depth_caps_logged = True
                    if (not v33b_depth_disabled) and d.ndim == 2 and d.size > 0:
                        rgb_md5 = hashlib.md5(np.ascontiguousarray(rgb).tobytes()).hexdigest()
                        depth_md5 = hashlib.md5(np.ascontiguousarray(d).tobytes()).hexdigest()
                        rgb_same = int(bool(v33b_rgb_hash_prev) and rgb_md5 == v33b_rgb_hash_prev)
                        depth_same = int(bool(v33b_depth_hash_prev) and depth_md5 == v33b_depth_hash_prev)
                        pose_moved = int(
                            (float(v33b_last_step_dpos) >= float(args.v33b_stale_dpos_eps))
                            or (abs(float(v33b_last_step_dyaw_deg)) >= float(args.v33b_stale_dyaw_eps))
                        )
                        if rgb_same and depth_same:
                            if pose_moved:
                                v33b_pose_move_k += 1
                                v33b_stationary_k = 0
                            else:
                                v33b_stationary_k += 1
                                v33b_pose_move_k = 0
                                if v33b_stationary_k == int(args.v33b_stationary_k):
                                    v33b_robot_stationary_events += 1
                        else:
                            v33b_pose_move_k = 0
                            v33b_stationary_k = 0
                        v33b_depth_hash_repeat = int(v33b_pose_move_k)
                        print(
                            f"[V33B_STALE_CHECK] step={step} pose_moved={pose_moved} rgb_same={rgb_same} depth_same={depth_same} "
                            f"stale_k={int(v33b_pose_move_k)} stationary_k={int(v33b_stationary_k)}",
                            flush=True,
                        )
                        v33b_depth_hash_prev = depth_md5
                        v33b_rgb_hash_prev = rgb_md5
                        if step % 10 == 0:
                            print(
                                f"[V33B_OBS_HASH] step={step} rgb_md5={rgb_md5} depth_md5={depth_md5}",
                                flush=True,
                            )
                        if v33b_pose_move_k >= int(args.v33b_stale_k):
                            v33b_depth_unreliable = True
                            v33b_depth_stale_pose_moved = True
                            v33b_depth_disabled = True
                            print("[V33B_CAPS] depth=0 reason=depth_stale_pose_moved", flush=True)
                            if bool(args.v33b_require_depth):
                                fail_reason = "backend_caps_depth_stale_pose_moved"
                                terminated_by = "early_stop"
                                print(
                                    f"[TOPO_NAV_FAIL] reason={fail_reason} steps={step}",
                                    flush=True,
                                )
                                break
                            print(
                                f"[V33B_LOCAL_AVOID] step={step} skip=1 blocked=0 override=0 "
                                f"reason=depth_stale_pose_moved action_in={base_action_name} action_out={action_name}",
                                flush=True,
                            )
                    if d.ndim == 2 and d.size > 0:
                        h, w = int(d.shape[0]), int(d.shape[1])
                        y0 = int(np.clip(float(args.v33b_strip_y0_frac) * h, 0, max(0, h - 1)))
                        y1 = int(np.clip(float(args.v33b_strip_y1_frac) * h, y0 + 1, h))
                        # Always mask out bottom 10% rows to reduce floor contamination.
                        y_floor_cap = max(y0 + 1, int(0.90 * h))
                        y1 = min(y1, y_floor_cap)
                        if y1 <= y0:
                            y1 = min(h, y0 + 1)
                        strip_w = max(1, int(float(args.v33b_strip_w_frac) * w))
                        low = float(v33b_near + 1e-3)
                        high = float(max(low + 1e-3, v33b_far))
                        d_clip = np.asarray(d, dtype=np.float32).copy()
                        valid_global = np.isfinite(d_clip)
                        valid_global &= d_clip >= low
                        valid_global &= d_clip <= high
                        d_clip[~valid_global] = np.nan

                        offset_metrics: Dict[float, Tuple[float, float]] = {}
                        for off in v33b_offsets:
                            col = int(round((w * 0.5) + (float(off) / max(1e-3, float(v33b_fov_deg))) * w))
                            clr, vr = _depth_strip_clearance(
                                d_clip,
                                col_center=col,
                                strip_w=strip_w,
                                y0=y0,
                                y1=y1,
                                pctl=float(args.v33b_block_pctl),
                                min_valid_ratio=float(args.v33b_min_valid_ratio),
                            )
                            offset_metrics[float(off)] = (float(clr), float(vr))

                        center_off = min(v33b_offsets, key=lambda x: abs(float(x)))
                        center_clr, center_vr = offset_metrics.get(float(center_off), (0.0, 0.0))
                        best_off = max(v33b_offsets, key=lambda x: offset_metrics.get(float(x), (0.0, 0.0))[0])
                        best_clr = offset_metrics.get(float(best_off), (0.0, 0.0))[0]
                        occ_patch = d_clip[y0:y1, :]
                        occ_valid = np.isfinite(occ_patch)
                        occ_ratio = (
                            float(np.mean((occ_patch[occ_valid] < float(args.v33b_clearance_m)).astype(np.float32)))
                            if occ_valid.any()
                            else 1.0
                        )
                        is_forward_like = int(base_action) == 1
                        blocked = False
                        override = 0
                        override_reason = "action_passthrough"
                        chosen_off = float(best_off)
                        chosen_clr = float(best_clr)
                        depth_mean_delta = 0.0
                        center_clear = bool(
                            center_vr >= float(args.v33b_min_valid_ratio)
                            and center_clr >= float(args.v33b_clearance_m) * 1.15
                        )
                        if occ_patch.size > 0:
                            ds_h = max(1, int(occ_patch.shape[0] // 24))
                            ds_w = max(1, int(occ_patch.shape[1] // 32))
                            roi_small = occ_patch[::ds_h, ::ds_w]
                            if (
                                v33b_depth_prev_roi is not None
                                and isinstance(v33b_depth_prev_roi, np.ndarray)
                                and v33b_depth_prev_roi.shape == roi_small.shape
                            ):
                                pair_valid = np.isfinite(roi_small) & np.isfinite(v33b_depth_prev_roi)
                                if pair_valid.any():
                                    depth_mean_delta = float(
                                        np.mean(np.abs(roi_small[pair_valid] - v33b_depth_prev_roi[pair_valid]))
                                    )
                                    if depth_mean_delta < 0.002:
                                        v33b_depth_stale_delta_streak += 1
                                    else:
                                        v33b_depth_stale_delta_streak = 0
                                else:
                                    v33b_depth_stale_delta_streak = 0
                            else:
                                v33b_depth_stale_delta_streak = 0
                            v33b_depth_prev_roi = roi_small.copy()

                        if not is_forward_like:
                            override_reason = "action_not_forward"
                            print(
                                f"[V33B_LOCAL_AVOID] step={step} skip=1 blocked=0 override=0 "
                                f"reason={override_reason} action_in={base_action_name} action_out={action_name}",
                                flush=True,
                            )
                        else:
                            blocked = bool(
                                (center_vr < float(args.v33b_min_valid_ratio))
                                or (center_clr < float(args.v33b_clearance_m))
                            )
                            if blocked:
                                v33b_blocked_detected_count += 1
                                blocked_stats_sig = (
                                    round(float(center_vr), 3),
                                    round(float(center_clr), 3),
                                    round(float(best_clr), 3),
                                )
                                if v33b_depth_prev_blocked_stats == blocked_stats_sig:
                                    v33b_depth_same_stats_k += 1
                                else:
                                    v33b_depth_same_stats_k = 1
                                v33b_depth_prev_blocked_stats = blocked_stats_sig
                            else:
                                v33b_depth_same_stats_k = 0
                                v33b_depth_prev_blocked_stats = None
                            if step == 0 or blocked:
                                print(
                                    f"[V33B_COSTMAP] ok=1 w={w} h={h} res={float(args.planner_grid_res):.3f} "
                                    f"occ_ratio={occ_ratio:.3f}",
                                    flush=True,
                                )
                            vals_win = occ_patch[np.isfinite(occ_patch)]
                            if vals_win.size > 0 and (step == 0 or blocked):
                                p20 = float(np.percentile(vals_win, 20))
                                p50 = float(np.percentile(vals_win, 50))
                                p80 = float(np.percentile(vals_win, 80))
                                print(
                                    f"[V33B_DEPTH_STATS] step={step} valid_ratio={center_vr:.3f} "
                                    f"p20={p20:.3f} p50={p50:.3f} p80={p80:.3f} "
                                    f"min={float(np.min(vals_win)):.3f} max={float(np.max(vals_win)):.3f}",
                                    flush=True,
                                )
                                v33b_depth_stats_logged = True

                            if blocked and (v33b_depth_stale_delta_streak >= 10 or v33b_depth_same_stats_k >= 5):
                                # Keep as diagnostics only. Stale-disable path is pose-aware hash check above.
                                print(
                                    f"[V33B_DEPTH_STALE] trigger=0 mean_delta={depth_mean_delta:.4f} "
                                    f"same_stats_k={int(v33b_depth_same_stats_k)} action_in=FORWARD action_out=FORWARD",
                                    flush=True,
                                )

                            collision_recent = bool(sum(v33b_collision_hist) > 0)
                            can_force_turn = bool(collision_recent or (v33b_probe_failures > 0))
                            if v33b_depth_unreliable:
                                can_force_turn = False

                            # If we are latched into a steering side, keep it while still blocked.
                            if v33b_steer_latch_steps > 0:
                                if center_clear:
                                    v33b_steer_latch_steps = 0
                                    v33b_steer_latch_side = 0
                                    print("[V33B_STEER_LATCH] release=1", flush=True)
                                elif blocked and can_force_turn:
                                    v33b_steer_latch_steps -= 1
                                    if int(v33b_steer_latch_side) != 0:
                                        action = 3 if int(v33b_steer_latch_side) > 0 else 2
                                        action_name = _action_id_to_name(action)
                                        recovery_event_labels.append("v33b_steer_latch")
                                        override = 1
                                        override_reason = "steer_latch"

                            if blocked and int(action) == int(base_action):
                                improvement = float(best_clr - center_clr)
                                chosen_off = float(best_off)
                                should_turn = bool(
                                    abs(chosen_off) >= float(args.v33b_turn_step_deg)
                                    and improvement >= float(args.v33b_min_improve_m)
                                )
                                if can_force_turn:
                                    chosen_side = 1 if chosen_off > 0.0 else -1 if chosen_off < 0.0 else 0
                                    if chosen_side == 0:
                                        # Deterministic fallback when best offset is straight-ahead.
                                        left_opts = [o for o in v33b_offsets if float(o) < 0.0]
                                        right_opts = [o for o in v33b_offsets if float(o) > 0.0]
                                        left_clr = max(
                                            (offset_metrics.get(float(o), (0.0, 0.0))[0] for o in left_opts),
                                            default=0.0,
                                        )
                                        right_clr = max(
                                            (offset_metrics.get(float(o), (0.0, 0.0))[0] for o in right_opts),
                                            default=0.0,
                                        )
                                        if right_clr > left_clr:
                                            chosen_side = 1
                                            chosen_off = 10.0
                                        elif left_clr > right_clr:
                                            chosen_side = -1
                                            chosen_off = -10.0
                                    if chosen_side == 0:
                                        chosen_side = 1 if int(v33b_steer_latch_side) >= 0 else -1
                                        chosen_off = float(args.v33b_turn_step_deg) * float(chosen_side)
                                    if bool(args.v33b_doorway_trigger) and should_turn and chosen_side != 0:
                                        v33b_doorway_triggers += 1
                                        print(
                                            f"[V33B_DOORWAY] trigger=1 mode=sample_offsets best={float(best_off):.1f}",
                                            flush=True,
                                        )
                                    if chosen_side != 0:
                                        action = 3 if chosen_side > 0 else 2
                                        action_name = _action_id_to_name(action)
                                        recovery_event_labels.append("v33b_local_avoid")
                                        v33b_steer_latch_side = int(chosen_side)
                                        v33b_steer_latch_steps = max(1, int(args.v33b_steer_hyst_steps))
                                        side_s = "right" if chosen_side > 0 else "left"
                                        print(
                                            f"[V33B_STEER_LATCH] set=1 side={side_s} steps={int(v33b_steer_latch_steps)}",
                                            flush=True,
                                        )
                                        override = 1
                                        override_reason = (
                                            "blocked_turn_best_offset" if should_turn else "blocked_force_turn"
                                        )
                                    else:
                                        override_reason = "blocked_no_valid_side"
                                else:
                                    if v33b_depth_unreliable:
                                        override_reason = "blocked_depth_stale_pass_through"
                                    else:
                                        override_reason = "blocked_no_collision_probe"
                                        v33b_forward_probe_applied = True

                            if int(action) != int(base_action):
                                v33b_override_count += 1
                                override = 1
                                if override_reason in {"action_passthrough", "action_not_forward"}:
                                    override_reason = "action_changed"

                            if blocked:
                                v33b_blocked_forward_count += 1
                                v33b_blocked_forward_reasons[str(override_reason)] += 1
                                if override == 1:
                                    v33b_blocked_forward_overridden_count += 1
                                else:
                                    v33b_blocked_forward_pass_through_count += 1
                            if blocked and override == 0:
                                if not override_reason:
                                    override_reason = "blocked_no_override_unknown"
                                v33b_blocked_no_override_count += 1
                                v33b_blocked_no_override_reasons[str(override_reason)] += 1

                            print(
                                f"[V33B_LOCAL_AVOID] step={step} blocked={int(blocked)} override={int(override)} "
                                f"reason={override_reason} chosen_offset={float(chosen_off):.1f} "
                                f"clearance={float(chosen_clr):.3f} action_in={base_action_name} action_out={action_name}",
                                flush=True,
                            )

            prev_pos = np.array([obs.pose.x, obs.pose.y, obs.pose.z], dtype=np.float32)
            obs_next, env_done, info = backend.step(int(action))
            new_pos = np.array([obs_next.pose.x, obs_next.pose.y, obs_next.pose.z], dtype=np.float32)
            moved = float(np.linalg.norm(new_pos - prev_pos))
            yaw_delta = _wrap_pi(float(obs_next.pose.yaw) - float(obs.pose.yaw))
            v33b_last_step_dpos = float(moved)
            v33b_last_step_dyaw_deg = float(math.degrees(float(yaw_delta)))
            yaw_delta_trace.append(float(yaw_delta))
            fwd_no_motion = bool(int(action) == 1 and moved < 0.01)
            if args.controller_mode != "follower":
                bearing_ctrl.update_after_step(int(action), moved)
            last_collision = obs_next.collision if obs_next.collision is not None else backend.get_collision_flag()
            v33b_collision_hist.append(1 if bool(last_collision) else 0)
            if v33b_forward_probe_applied:
                new_goal_dist = float(np.linalg.norm(new_pos - goal_pos_now))
                goal_progress = float(dist_goal - new_goal_dist)
                if bool(last_collision) or goal_progress < 0.03:
                    v33b_probe_failures += 1
                else:
                    v33b_probe_failures = 0
            elif int(action) == 1 and moved > 0.05:
                # Forward translation resets probe failure debt.
                v33b_probe_failures = 0

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
                "fwd_no_motion": int(bool(fwd_no_motion)),
                "collision": None if last_collision is None else int(bool(last_collision)),
                "goal_score": float(goal_score),
            }
            if watchdog_trigger_count > 0:
                recovery_event_labels.append("watchdog_no_progress")
            rec["recovery_event"] = recovery_event_labels
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
        "v33a_stuck_trigger_count": int(v33a_stuck_triggers),
        "v33a_recovery_count": int(v33a_recovery_count),
        "v33a_stuck_fail": int(v33a_stuck_fail),
        "v33b": {
            "enabled": int(v33b_enabled),
            "depth_ready": int(v33b_depth_ready),
            "depth_disabled": int(v33b_depth_disabled),
            "depth_source": str(v33b_depth_source),
            "depth_key_requested": str(v33b_depth_key_req),
            "depth_key_used": str(v33b_depth_key_used),
            "depth_missing_key": int(v33b_depth_missing_key),
            "blocked_detected_count": int(v33b_blocked_detected_count),
            "override_count": int(v33b_override_count),
            "blocked_no_override_count": int(v33b_blocked_no_override_count),
            "blocked_no_override_reasons": {
                str(k): int(v) for k, v in sorted(v33b_blocked_no_override_reasons.items())
            },
            "blocked_forward_count": int(v33b_blocked_forward_count),
            "blocked_forward_overridden_count": int(v33b_blocked_forward_overridden_count),
            "blocked_forward_pass_through_count": int(v33b_blocked_forward_pass_through_count),
            "blocked_forward_reasons": {
                str(k): int(v) for k, v in sorted(v33b_blocked_forward_reasons.items())
            },
            "depth_stale_or_invalid": int(v33b_depth_stale_pose_moved),
            "depth_stale_pose_moved": int(v33b_depth_stale_pose_moved),
            "robot_stationary_events": int(v33b_robot_stationary_events),
            "stale_pose_moved_k": int(v33b_pose_move_k),
            "stationary_k": int(v33b_stationary_k),
            "depth_unreliable": int(v33b_depth_unreliable),
            "depth_stale_delta_streak": int(v33b_depth_stale_delta_streak),
            "depth_same_stats_k": int(v33b_depth_same_stats_k),
            "depth_hash_repeat": int(v33b_depth_hash_repeat),
            "forward_probe_failures": int(v33b_probe_failures),
            "doorway_triggers": int(v33b_doorway_triggers),
        },
        "belief_entropy_summary": {
            "mean": float(last_entropy),
            "max": float(last_entropy),
            "min": float(last_entropy),
        },
        "fill_triggers": int(fill_triggers),
        "camera": camera_info,
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
            "camera": camera_info,
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
    ap.add_argument("--stuck_window", type=int, default=int(os.environ.get("STUCK_WINDOW", "20")))
    ap.add_argument("--stuck_dist_eps", type=float, default=float(os.environ.get("STUCK_DIST_EPS", "0.05")))
    ap.add_argument(
        "--oscillation_yaw_eps_deg",
        type=float,
        default=float(os.environ.get("OSCILLATION_YAW_EPS", "10.0")),
    )
    ap.add_argument(
        "--oscillation_flip_min",
        type=int,
        default=int(os.environ.get("OSCILLATION_FLIP_MIN", "4")),
    )
    ap.add_argument(
        "--oscillation_min_turns",
        type=int,
        default=int(os.environ.get("OSCILLATION_MIN_TURNS", "6")),
    )
    ap.add_argument(
        "--stuck_recovery_scan_k",
        type=int,
        default=int(os.environ.get("STUCK_RECOVERY_SCAN_K", "2")),
    )
    ap.add_argument(
        "--stuck_max_recoveries",
        type=int,
        default=int(os.environ.get("STUCK_MAX_RECOVERIES", "3")),
    )
    ap.add_argument("--v33b_enable", type=int, default=int(os.environ.get("V33B_ENABLE", "0")))
    ap.add_argument("--v33b_require_depth", type=int, default=int(os.environ.get("V33B_REQUIRE_DEPTH", "1")))
    ap.add_argument("--v33b_clearance_m", type=float, default=float(os.environ.get("V33B_CLEARANCE_M", "0.65")))
    ap.add_argument(
        "--v33b_min_valid_ratio",
        type=float,
        default=float(os.environ.get("V33B_MIN_VALID_RATIO", "0.30")),
    )
    ap.add_argument("--v33b_strip_w_frac", type=float, default=float(os.environ.get("V33B_STRIP_W_FRAC", "0.08")))
    ap.add_argument("--v33b_strip_y0_frac", type=float, default=float(os.environ.get("V33B_STRIP_Y0_FRAC", "0.25")))
    ap.add_argument("--v33b_strip_y1_frac", type=float, default=float(os.environ.get("V33B_STRIP_Y1_FRAC", "0.85")))
    ap.add_argument(
        "--v33b_offsets_deg",
        type=str,
        default=str(os.environ.get("V33B_OFFSETS_DEG", "0,10,20,30,-10,-20,-30")),
    )
    ap.add_argument("--v33b_block_pctl", type=float, default=float(os.environ.get("V33B_BLOCK_PCTL", "20")))
    ap.add_argument("--v33b_turn_step_deg", type=float, default=float(os.environ.get("V33B_TURN_STEP_DEG", "15")))
    ap.add_argument("--v33b_min_improve_m", type=float, default=float(os.environ.get("V33B_MIN_IMPROVE_M", "0.10")))
    ap.add_argument("--v33b_steer_hyst_steps", type=int, default=int(os.environ.get("V33B_STEER_HYST_STEPS", "5")))
    ap.add_argument("--v33b_doorway_trigger", type=int, default=int(os.environ.get("V33B_DOORWAY_TRIGGER", "1")))
    ap.add_argument("--v33b_stale_dpos_eps", type=float, default=float(os.environ.get("V33B_STALE_DPOS_EPS", "0.02")))
    ap.add_argument("--v33b_stale_dyaw_eps", type=float, default=float(os.environ.get("V33B_STALE_DYAW_EPS", "2.0")))
    ap.add_argument("--v33b_stale_k", type=int, default=int(os.environ.get("V33B_STALE_K", "3")))
    ap.add_argument("--v33b_stationary_k", type=int, default=int(os.environ.get("V33B_STATIONARY_K", "20")))
    ap.add_argument(
        "--v33b_doorway_hyst_steps",
        type=int,
        default=int(os.environ.get("V33B_DOORWAY_HYST_STEPS", "8")),
    )
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
