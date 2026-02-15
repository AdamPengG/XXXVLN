import argparse
import collections
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import quaternion
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, CLIPProcessor

from internnav.configs.evaluator import EnvCfg
from internnav.env.habitat_env import HabitatEnv
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.topo.geo_descriptor import compute_scan_context
from internnav.topo.graph import TopoGraph, l2_distance, save_embeddings
from internnav.topo.pose_graph import PoseGraphLite, wrap_angle
from internnav.topo.room_partition import partition_rooms_hydralite_edge
from internnav.topo.room_semantics import (
    GENERIC_OBJECTS,
    infer_room_label_from_objects,
    normalize_object_label,
)
from internnav.utils.feature_extractor import prepare_vln_inputs, extract_action_features

DEFAULT_ROOM_LABELS = [
    "kitchen",
    "bedroom",
    "living room",
    "bathroom",
    "hallway",
    "dining room",
    "stairs",
    "office",
    "balcony",
]

DEFAULT_OBJECT_LABELS = [
    "table",
    "dining table",
    "coffee table",
    "chair",
    "sofa",
    "couch",
    "bed",
    "pillow",
    "wardrobe",
    "dresser",
    "nightstand",
    "sink",
    "bathtub",
    "shower",
    "mirror",
    "door",
    "cabinet",
    "desk",
    "toilet",
    "tv",
    "television",
    "refrigerator",
    "fridge",
    "stove",
    "oven",
    "microwave",
    "dishwasher",
    "bookcase",
    "bookshelf",
    "monitor",
    "laptop",
    "plant",
    "counter",
    "shelf",
    "lamp",
    "stairs",
]

ROOM_LABEL_CANONICAL = {
    "living room": "living_room",
    "dining room": "dining_room",
    "bath room": "bathroom",
}


def canonical_room_label(label: str) -> str:
    s = str(label or "").strip().lower().replace("-", " ")
    s = " ".join([x for x in s.split() if x])
    if s in ROOM_LABEL_CANONICAL:
        return str(ROOM_LABEL_CANONICAL[s])
    return s.replace(" ", "_")


def _map_room_text_to_label(text: str) -> str:
    s = canonical_room_label(text)
    if "bath" in s or "toilet" in s:
        return "bathroom"
    if "kitchen" in s:
        return "kitchen"
    if "bed" in s:
        return "bedroom"
    if "living" in s or "lounge" in s:
        return "living_room"
    if "dining" in s:
        return "dining_room"
    if "office" in s or "study" in s:
        return "office"
    if "hall" in s or "corridor" in s or "stair" in s or "entry" in s:
        return "hallway"
    return "unknown"


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


class TopoEmbedder:
    def __init__(self, model_path: str, pooling: str = "mean"):
        self.pooling = pooling
        self.device = _get_device()
        self.infer_dtype_str = os.environ.get("INFER_DTYPE", "bf16")
        self.attn_backend = os.environ.get("FORCE_ATTN_BACKEND", "sdpa")
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self.torch_dtype = dtype_map.get(self.infer_dtype_str, torch.bfloat16)

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"

        self.model = InternVLAN1ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            attn_implementation=self.attn_backend,
            device_map={"": self.device},
        )
        self.model.eval()
        self._dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    def _embed(self, instruction: str, image: Image.Image) -> np.ndarray:
        inputs = prepare_vln_inputs(
            self.processor,
            instructions=[instruction],
            images=[image],
            device=self.device,
            mode="simple",
        )
        feats = extract_action_features(self.model, inputs, pooling=self.pooling)
        vec = feats[0].detach().cpu().float().numpy()
        return vec.astype(np.float16)

    def embed_image(self, rgb: np.ndarray, instruction: str = "") -> np.ndarray:
        if isinstance(rgb, np.ndarray):
            image = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
        else:
            image = rgb
        return self._embed(instruction, image)

    def embed_text(self, text: str) -> np.ndarray:
        return self._embed(text, self._dummy_image)


class TopoClipEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = _get_device()
        self.model_name = model_name
        self.local_files_only = os.environ.get("TOPO_CLIP_LOCAL_ONLY", "1") == "1"
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=self.local_files_only)
            self.model = CLIPModel.from_pretrained(model_name, local_files_only=self.local_files_only)
        except Exception:
            # Fallback to online fetch if local cache is not available.
            self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=False)
            self.model = CLIPModel.from_pretrained(model_name, local_files_only=False)
        self.model = self.model.to(self.device)
        self.model.eval()

    def embed_image(self, rgb: np.ndarray) -> np.ndarray:
        image = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.model.get_image_features(**inputs)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
        return feat[0].detach().cpu().float().numpy().astype(np.float16)

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feat = self.model.get_text_features(**inputs)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
        return feat[0].detach().cpu().float().numpy().astype(np.float16)


def _sample_action(rng: random.Random, forward_prob: float = 0.7) -> int:
    r = rng.random()
    if r < forward_prob:
        return 1
    return 2 if rng.random() < 0.5 else 3


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float32)
    bb = b.astype(np.float32)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))


def _relative_pose_consistency(
    cand_meta: Dict,
    curr_gt_pos: List[float],
    curr_gt_yaw: float,
    curr_est_pos: List[float],
    curr_est_yaw: float,
    max_dist_err: float,
    max_yaw_err: float,
) -> Tuple[bool, float, float]:
    cand_gt_pos = np.array(cand_meta.get("gt_position", cand_meta.get("position", [0.0, 0.0, 0.0])), dtype=np.float32)
    cand_est_pos = np.array(cand_meta.get("est_position", cand_meta.get("position", [0.0, 0.0, 0.0])), dtype=np.float32)
    cand_gt_yaw = float(cand_meta.get("gt_yaw", cand_meta.get("est_yaw", 0.0)))
    cand_est_yaw = float(cand_meta.get("est_yaw", cand_meta.get("gt_yaw", 0.0)))

    gt_dx = float(curr_gt_pos[0]) - float(cand_gt_pos[0])
    gt_dz = float(curr_gt_pos[2]) - float(cand_gt_pos[2])
    est_dx = float(curr_est_pos[0]) - float(cand_est_pos[0])
    est_dz = float(curr_est_pos[2]) - float(cand_est_pos[2])

    gt_dist = float(np.hypot(gt_dx, gt_dz))
    est_dist = float(np.hypot(est_dx, est_dz))
    dist_err = abs(gt_dist - est_dist)

    gt_dyaw = wrap_angle(float(curr_gt_yaw) - float(cand_gt_yaw))
    est_dyaw = wrap_angle(float(curr_est_yaw) - float(cand_est_yaw))
    yaw_err = abs(wrap_angle(gt_dyaw - est_dyaw))

    geo_ok = (dist_err <= float(max_dist_err)) and (yaw_err <= float(max_yaw_err))
    return bool(geo_ok), float(dist_err), float(yaw_err)


def parse_room_labels(room_labels_csv: str = "") -> List[str]:
    if room_labels_csv is None:
        room_labels_csv = ""
    labels = [x.strip() for x in str(room_labels_csv).split(",") if x.strip()]
    return labels if labels else list(DEFAULT_ROOM_LABELS)


def parse_object_labels(object_labels_csv: str = "") -> List[str]:
    if object_labels_csv is None:
        object_labels_csv = ""
    labels = [x.strip() for x in str(object_labels_csv).split(",") if x.strip()]
    return labels if labels else list(DEFAULT_OBJECT_LABELS)


def build_text_embeddings(clip_embedder: TopoClipEmbedder, labels: List[str], template: str = "{label}") -> np.ndarray:
    if len(labels) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    embeds = [clip_embedder.embed_text(str(template).format(label=label)).astype(np.float32) for label in labels]
    arr = np.stack(embeds, axis=0)
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-6)
    return arr


def build_room_text_embeddings(clip_embedder: TopoClipEmbedder, room_labels: List[str]) -> np.ndarray:
    room_template = os.environ.get("TOPO_ROOM_TEXT_TEMPLATE", "a photo of a {label} room")
    return build_text_embeddings(clip_embedder, room_labels, template=room_template)


def _forward_from_quat_wxyz(rotation_wxyz: List[float]) -> np.ndarray:
    q = np.quaternion(
        float(rotation_wxyz[0]),
        float(rotation_wxyz[1]),
        float(rotation_wxyz[2]),
        float(rotation_wxyz[3]),
    )
    rot = quaternion.as_rotation_matrix(q)
    forward = rot @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return (forward / norm).astype(np.float32)


def _yaw_from_quat_wxyz(rotation_wxyz: List[float]) -> float:
    q = np.quaternion(
        float(rotation_wxyz[0]),
        float(rotation_wxyz[1]),
        float(rotation_wxyz[2]),
        float(rotation_wxyz[3]),
    )
    rot = quaternion.as_rotation_matrix(q)
    forward = rot @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return float(np.arctan2(float(forward[0]), float(-forward[2])))


def _quat_wxyz_from_yaw(yaw: float) -> List[float]:
    yy = float(yaw)
    return [float(np.cos(yy * 0.5)), 0.0, float(np.sin(yy * 0.5)), 0.0]


def _estimate_object_centroid(
    position: List[float],
    rotation_wxyz: List[float],
    depth_obs: Optional[np.ndarray],
) -> List[float]:
    pos = np.array(position, dtype=np.float32)
    if depth_obs is None:
        depth_median = 1.5
    else:
        depth = np.asarray(depth_obs, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        finite = depth[np.isfinite(depth)]
        if finite.size == 0:
            depth_median = 1.5
        else:
            depth_median = float(np.median(finite))
    depth_median = float(np.clip(depth_median, 0.5, 4.0))
    forward = _forward_from_quat_wxyz(rotation_wxyz)
    center = pos + forward * depth_median
    center[1] = pos[1]
    return [float(center[0]), float(center[1]), float(center[2])]


def _estimate_clearance_proxy_from_depth(depth_obs: Optional[np.ndarray], max_depth: float = 5.0) -> float:
    if depth_obs is None:
        return 0.0
    depth = np.asarray(depth_obs, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.size == 0:
        return 0.0
    h, w = depth.shape[:2]
    cx = int(w // 2)
    cy = int(h // 2)
    half_w = max(2, int(0.18 * w))
    y0 = max(0, int(cy - 0.08 * h))
    y1 = min(h, int(cy + 0.30 * h))
    x0 = max(0, cx - half_w)
    x1 = min(w, cx + half_w)
    cone = depth[y0:y1, x0:x1]
    finite = cone[np.isfinite(cone)]
    if finite.size == 0:
        return 0.0
    vmax = max(0.5, float(max_depth))
    d = np.clip(finite, 0.0, vmax)
    # Use a blend of median and upper-tail as a stable free-space proxy.
    q50 = float(np.quantile(d, 0.50))
    q80 = float(np.quantile(d, 0.80))
    return float(0.6 * q50 + 0.4 * q80)


def _collect_semantic_label_map(sim) -> Dict[int, str]:
    out: Dict[int, str] = {}
    try:
        scene = sim.semantic_scene
        if scene is None:
            return out
        for obj in scene.objects:
            if obj is None:
                continue
            oid = getattr(obj, "id", None)
            cat = getattr(obj, "category", None)
            if oid is None:
                continue
            try:
                oid_int = int(str(oid).split("_")[-1])
            except Exception:
                continue
            label = "unknown"
            try:
                label = str(cat.name()).strip().lower()
            except Exception:
                label = "unknown"
            out[oid_int] = label
    except Exception:
        pass
    return out


def _collect_semantic_regions(sim) -> Tuple[List[Dict], str]:
    out: List[Dict] = []
    reason = "ok"
    try:
        scene = sim.semantic_scene
        if scene is None:
            return [], "semantic_scene_none"
        regions = getattr(scene, "regions", None)
        if regions is None:
            return [], "regions_missing"
        for ridx, region in enumerate(regions):
            if region is None:
                continue
            raw_label = "unknown"
            try:
                raw_label = str(region.category.name()).strip().lower()
            except Exception:
                raw_label = "unknown"
            label = _map_room_text_to_label(raw_label)
            center = [0.0, 0.0, 0.0]
            half = [0.0, 0.0, 0.0]
            try:
                aabb = region.aabb
                c = aabb.center
                sz = aabb.sizes
                center = [float(c[0]), float(c[1]), float(c[2])]
                half = [max(0.05, float(sz[0]) * 0.5), max(0.05, float(sz[1]) * 0.5), max(0.05, float(sz[2]) * 0.5)]
            except Exception:
                pass
            out.append(
                {
                    "region_id": int(ridx),
                    "raw_label": str(raw_label),
                    "label": str(label),
                    "center": center,
                    "half_sizes": half,
                }
            )
    except Exception as e:
        reason = f"semantic_regions_exception:{type(e).__name__}"
        return [], reason
    if len(out) == 0:
        reason = "regions_empty"
    return out, reason


def _infer_gt_room_label_from_regions(position: List[float], regions: List[Dict]) -> str:
    if len(regions) == 0:
        return "unknown"
    p = np.asarray(position, dtype=np.float32)
    inside = []
    for reg in regions:
        c = np.asarray(reg.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
        h = np.asarray(reg.get("half_sizes", [0.0, 0.0, 0.0]), dtype=np.float32)
        d = np.abs(p - c)
        if bool(np.all(d <= (h + 1e-3))):
            inside.append((float(np.linalg.norm(d[[0, 2]])), str(reg.get("label", "unknown"))))
    if len(inside) > 0:
        inside.sort(key=lambda x: x[0])
        return str(inside[0][1])
    # Fallback: nearest region center in x-z plane.
    nearest = None
    best = float("inf")
    for reg in regions:
        c = np.asarray(reg.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
        d = float(np.linalg.norm((p - c)[[0, 2]]))
        if d < best:
            best = d
            nearest = str(reg.get("label", "unknown"))
    if nearest is None:
        return "unknown"
    if best > 4.0:
        return "unknown"
    return str(nearest)


def detect_objects_for_node(
    rgb: np.ndarray,
    depth: Optional[np.ndarray],
    clip_embed: np.ndarray,
    clip_embedder: TopoClipEmbedder,
    position: List[float],
    rotation_wxyz: List[float],
    object_labels: List[str],
    object_text_embeds: np.ndarray,
    semantic_obs: Optional[np.ndarray] = None,
    semantic_id_to_label: Optional[Dict[int, str]] = None,
    topk: int = 3,
    min_score: float = 0.18,
) -> List[Dict]:
    detections: List[Dict] = []
    if semantic_obs is not None:
        sem = np.asarray(semantic_obs)
        if sem.ndim == 3:
            sem = sem[..., 0]
        sem = sem.astype(np.int32, copy=False)
        if sem.size > 0:
            ids, counts = np.unique(sem, return_counts=True)
            order = np.argsort(-counts)
            for idx in order[: max(1, int(topk))]:
                sid = int(ids[idx])
                if sid <= 0:
                    continue
                if int(counts[idx]) < 80:
                    continue
                lbl = "unknown"
                if semantic_id_to_label is not None:
                    lbl = str(semantic_id_to_label.get(sid, "unknown"))
                detections.append(
                    {
                        "label": lbl,
                        "score": float(counts[idx]) / float(sem.size),
                        "centroid": _estimate_object_centroid(position, rotation_wxyz, depth),
                        "source": "semantic",
                    }
                )
    if len(detections) > 0:
        return detections

    if len(object_labels) == 0 or object_text_embeds.size == 0:
        return detections
    img = clip_embed.astype(np.float32)
    img = img / (np.linalg.norm(img) + 1e-6)
    sims = object_text_embeds.dot(img)
    k = min(max(1, int(topk)), sims.shape[0])
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    for i in top_idx:
        score = float(sims[int(i)])
        if score < float(min_score):
            continue
        detections.append(
            {
                "label": str(object_labels[int(i)]),
                "score": score,
                "centroid": _estimate_object_centroid(position, rotation_wxyz, depth),
                "source": "clip",
            }
        )
    return detections


def score_room_label_from_embed(
    clip_embed: np.ndarray,
    room_labels: List[str],
    room_text_embeds: np.ndarray,
    topk: int = 3,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    if len(room_labels) == 0:
        return "unknown", 0.0, []
    img = clip_embed.astype(np.float32)
    img = img / (np.linalg.norm(img) + 1e-6)
    sims = room_text_embeds.dot(img)
    k = min(max(1, int(topk)), sims.shape[0])
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    top = [(room_labels[int(i)], float(sims[int(i)])) for i in top_idx]
    best_label, best_score = top[0]
    return best_label, float(best_score), top


def _edge_yaw_change(node_meta: Dict[str, Dict], u: int, v: int) -> float:
    mu = node_meta.get(str(int(u)), {})
    mv = node_meta.get(str(int(v)), {})
    yu = float(mu.get("est_yaw", mu.get("gt_yaw", 0.0)))
    yv = float(mv.get("est_yaw", mv.get("gt_yaw", 0.0)))
    return abs(wrap_angle(yv - yu))


def _cluster_rooms_by_connectivity_fallback(
    graph: TopoGraph,
    node_meta: Dict[str, Dict],
    clip_embeds: np.ndarray,
) -> Tuple[Dict[str, Dict], Dict[int, int], Dict[str, int], Dict[str, object]]:
    edge_lengths = []
    yaw_changes = []
    for u, nbrs in graph.edges.items():
        for v, w in nbrs.items():
            if int(u) >= int(v):
                continue
            edge_lengths.append(float(w))
            yaw_changes.append(float(_edge_yaw_change(node_meta, int(u), int(v))))
    edge_arr = np.asarray(edge_lengths, dtype=np.float32) if len(edge_lengths) > 0 else np.zeros((0,), dtype=np.float32)
    yaw_arr = np.asarray(yaw_changes, dtype=np.float32) if len(yaw_changes) > 0 else np.zeros((0,), dtype=np.float32)
    edge_q = float(os.environ.get("TOPO_ROOM_SPLIT_EDGE_Q", "0.65"))
    yaw_q = float(os.environ.get("TOPO_ROOM_SPLIT_YAW_Q", "0.75"))
    split_edge_dist = float(np.quantile(edge_arr, edge_q)) if edge_arr.size > 0 else float("inf")
    split_yaw_rad = float(np.quantile(yaw_arr, yaw_q)) if yaw_arr.size > 0 else float(np.pi)

    adjacency: Dict[int, List[int]] = collections.defaultdict(list)
    for u, nbrs in graph.edges.items():
        for v, w in nbrs.items():
            if int(u) == int(v):
                continue
            if float(w) > split_edge_dist:
                continue
            yaw_delta = _edge_yaw_change(node_meta, int(u), int(v))
            if yaw_delta > split_yaw_rad:
                continue
            adjacency[int(u)].append(int(v))

    components: List[List[int]] = []
    assigned: Dict[int, bool] = {}
    for node_id in sorted(graph.nodes.keys()):
        if assigned.get(int(node_id), False):
            continue
        stack = [int(node_id)]
        comp: List[int] = []
        assigned[int(node_id)] = True
        while stack:
            u = int(stack.pop())
            comp.append(u)
            for v in adjacency.get(int(u), []):
                vv = int(v)
                if assigned.get(vv, False):
                    continue
                assigned[vv] = True
                stack.append(vv)
        components.append(sorted(comp))

    cluster_method = "connectivity_bfs"
    if len(components) == 1 and len(components[0]) >= 8:
        # If the full graph collapses to one cluster, split by principal
        # spread axis to avoid single-room degeneration.
        comp0 = components[0]
        pos0 = np.array([graph.nodes[n].position for n in comp0], dtype=np.float32)
        span = pos0.max(axis=0) - pos0.min(axis=0)
        axis = 0 if float(span[0]) >= float(span[2]) else 2
        span_2d = float(max(span[0], span[2]))
        min_span = float(os.environ.get("TOPO_ROOM_MIN_SPLIT_SPAN", "1.2"))
        force_split = os.environ.get("TOPO_FORCE_ROOM_SPLIT_SINGLE", "1") == "1"
        if force_split or span_2d >= min_span:
            cut = float(np.median(pos0[:, axis]))
            left = [n for n in comp0 if float(graph.nodes[n].position[axis]) <= cut]
            right = [n for n in comp0 if float(graph.nodes[n].position[axis]) > cut]
            min_side = int(os.environ.get("TOPO_ROOM_MIN_SIDE_NODES", "3"))
            if len(left) >= min_side and len(right) >= min_side:
                components = [sorted(left), sorted(right)]
                cluster_method = "connectivity_median_split_forced" if force_split else "connectivity_median_split"

    room_of_node: Dict[int, int] = {}
    rooms: Dict[str, Dict] = {}
    room_id = 0
    label_room_counts: Dict[str, int] = collections.defaultdict(int)

    for comp in components:
        for n in comp:
            room_of_node[int(n)] = int(room_id)
        pos = np.array([graph.nodes[n].position for n in comp], dtype=np.float32)
        mean_pos = pos.mean(axis=0)
        min_pos = pos.min(axis=0)
        max_pos = pos.max(axis=0)
        centroid_node = min(
            comp,
            key=lambda n: float(np.linalg.norm(np.array(graph.nodes[n].position, dtype=np.float32) - mean_pos)),
        )

        labels = [str(node_meta.get(str(n), {}).get("room_label", "unknown")) for n in comp]
        label_ctr = collections.Counter(labels)
        label = str(label_ctr.most_common(1)[0][0]) if len(label_ctr) > 0 else "unknown"
        room_score_vals = [
            float(node_meta.get(str(n), {}).get("room_score", 0.0))
            for n in comp
            if node_meta.get(str(n), {}).get("room_score", None) is not None
        ]
        mean_room_score = float(np.mean(room_score_vals)) if len(room_score_vals) > 0 else 0.0

        embed_idxs = []
        for n in comp:
            meta = node_meta.get(str(n), {})
            idx = int(meta.get("clip_embed_idx", meta.get("embed_idx", -1)))
            if 0 <= idx < clip_embeds.shape[0]:
                embed_idxs.append(idx)
        if embed_idxs:
            room_embed = clip_embeds[embed_idxs].astype(np.float32).mean(axis=0)
            room_embed = room_embed / (np.linalg.norm(room_embed) + 1e-6)
            room_embed_list = room_embed.tolist()
        else:
            room_embed_list = []

        rooms[str(room_id)] = {
            "label": label,
            "node_ids": [int(n) for n in sorted(comp)],
            "centroid_node": int(centroid_node),
            "mean_embed": room_embed_list,
            "center": [float(mean_pos[0]), float(mean_pos[1]), float(mean_pos[2])],
            "bbox_min": [float(min_pos[0]), float(min_pos[1]), float(min_pos[2])],
            "bbox_max": [float(max_pos[0]), float(max_pos[1]), float(max_pos[2])],
            "mean_room_score": float(mean_room_score),
        }
        label_room_counts[label] += 1
        room_id += 1

    cluster_info = {
        "method": cluster_method,
        "sizes": [int(len(c)) for c in components],
        "fallback": 1,
        "split_edge_dist": float(split_edge_dist) if np.isfinite(split_edge_dist) else -1.0,
        "split_yaw_rad": float(split_yaw_rad),
        "edge_q": float(edge_q),
        "yaw_q": float(yaw_q),
    }
    return rooms, room_of_node, dict(label_room_counts), cluster_info


def _materialize_rooms_from_components(
    graph: TopoGraph,
    node_meta: Dict[str, Dict],
    clip_embeds: np.ndarray,
    components: List[List[int]],
    method: str,
    extra_info: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, Dict], Dict[int, int], Dict[str, int], Dict[str, object]]:
    room_of_node: Dict[int, int] = {}
    rooms: Dict[str, Dict] = {}
    room_id = 0
    label_room_counts: Dict[str, int] = collections.defaultdict(int)

    for comp in components:
        for n in comp:
            room_of_node[int(n)] = int(room_id)
        pos = np.array([graph.nodes[n].position for n in comp], dtype=np.float32)
        mean_pos = pos.mean(axis=0)
        min_pos = pos.min(axis=0)
        max_pos = pos.max(axis=0)
        centroid_node = min(
            comp,
            key=lambda n: float(np.linalg.norm(np.array(graph.nodes[n].position, dtype=np.float32) - mean_pos)),
        )

        labels = [str(node_meta.get(str(n), {}).get("room_label", "unknown")) for n in comp]
        label_ctr = collections.Counter(labels)
        label = str(label_ctr.most_common(1)[0][0]) if len(label_ctr) > 0 else "unknown"
        room_score_vals = [
            float(node_meta.get(str(n), {}).get("room_score", 0.0))
            for n in comp
            if node_meta.get(str(n), {}).get("room_score", None) is not None
        ]
        mean_room_score = float(np.mean(room_score_vals)) if len(room_score_vals) > 0 else 0.0

        embed_idxs = []
        for n in comp:
            meta = node_meta.get(str(n), {})
            idx = int(meta.get("clip_embed_idx", meta.get("embed_idx", -1)))
            if 0 <= idx < clip_embeds.shape[0]:
                embed_idxs.append(idx)
        if embed_idxs:
            room_embed = clip_embeds[embed_idxs].astype(np.float32).mean(axis=0)
            room_embed = room_embed / (np.linalg.norm(room_embed) + 1e-6)
            room_embed_list = room_embed.tolist()
        else:
            room_embed_list = []

        rooms[str(room_id)] = {
            "label": label,
            "node_ids": [int(n) for n in sorted(comp)],
            "centroid_node": int(centroid_node),
            "mean_embed": room_embed_list,
            "center": [float(mean_pos[0]), float(mean_pos[1]), float(mean_pos[2])],
            "bbox_min": [float(min_pos[0]), float(min_pos[1]), float(min_pos[2])],
            "bbox_max": [float(max_pos[0]), float(max_pos[1]), float(max_pos[2])],
            "mean_room_score": float(mean_room_score),
        }
        label_room_counts[label] += 1
        room_id += 1

    cluster_info = {
        "method": str(method),
        "sizes": [int(len(c)) for c in components],
    }
    if extra_info:
        cluster_info.update(dict(extra_info))
    return rooms, room_of_node, dict(label_room_counts), cluster_info


def _cluster_rooms_hydralite(
    graph: TopoGraph,
    node_meta: Dict[str, Dict],
    clip_embeds: np.ndarray,
) -> Tuple[Dict[str, Dict], Dict[int, int], Dict[str, int], Dict[str, object]]:
    node_ids = sorted(int(n) for n in graph.nodes.keys())
    if len(node_ids) <= 1:
        return _cluster_rooms_by_connectivity_fallback(graph=graph, node_meta=node_meta, clip_embeds=clip_embeds)

    sample_steps = max(6, int(os.environ.get("TOPO_EDGE_CLEARANCE_SAMPLES", "12")))
    min_component_nodes = max(2, int(os.environ.get("TOPO_ROOM_SEG_MIN_COMPONENT_NODES", "2")))
    q_min = float(os.environ.get("TOPO_ROOM_SEG_Q_MIN", "0.10"))
    q_max = float(os.environ.get("TOPO_ROOM_SEG_Q_MAX", "0.80"))
    q_points = max(8, int(os.environ.get("TOPO_ROOM_SEG_Q_POINTS", "25")))
    min_dilation_m = float(os.environ.get("TOPO_MIN_DILATION_M", "-1.0"))
    max_dilation_m = float(os.environ.get("TOPO_MAX_DILATION_M", "-1.0"))
    mode = str(os.environ.get("TOPO_DILATION_MODE", "PLATEAU")).upper()

    part = partition_rooms_hydralite_edge(
        graph=graph,
        node_meta=node_meta,
        sample_steps=sample_steps,
        min_component_nodes=min_component_nodes,
        q_min=q_min,
        q_max=q_max,
        q_points=q_points,
        mode=mode,
        min_dilation_m=min_dilation_m,
        max_dilation_m=max_dilation_m,
    )

    cluster_info = dict(part.cluster_info)
    cluster_info["edge_clearance_debug"] = dict(part.edge_clearance_debug)
    cluster_info["doorway_debug"] = dict(part.doorway_debug)
    cluster_info["method"] = "hydralite_edge_filtration"

    if int(cluster_info.get("fallback", 0)) == 1:
        rooms_fb, room_of_node_fb, label_counts_fb, cluster_fb = _cluster_rooms_by_connectivity_fallback(
            graph=graph,
            node_meta=node_meta,
            clip_embeds=clip_embeds,
        )
        cluster_info["fallback_method"] = str(cluster_fb.get("method", "connectivity_bfs"))
        cluster_info["sizes"] = [int(x) for x in cluster_fb.get("sizes", cluster_info.get("sizes", []))]
        cluster_info["K"] = int(len(rooms_fb))
        cluster_info["components_from_fallback"] = 1
        return rooms_fb, room_of_node_fb, label_counts_fb, cluster_info

    rooms, room_of_node, label_room_counts, materialized_info = _materialize_rooms_from_components(
        graph=graph,
        node_meta=node_meta,
        clip_embeds=clip_embeds,
        components=part.components,
        method="hydralite_edge_filtration",
        extra_info=cluster_info,
    )
    materialized_info["edge_clearance_debug"] = dict(part.edge_clearance_debug)
    materialized_info["doorway_debug"] = dict(part.doorway_debug)
    return rooms, room_of_node, label_room_counts, materialized_info


def build_place_graph(graph: TopoGraph, node_meta: Dict[str, Dict]) -> Dict:
    places = []
    for nid in sorted(graph.nodes.keys()):
        node = graph.nodes[int(nid)]
        meta = node_meta.get(str(int(nid)), {})
        places.append(
            {
                "place_id": int(nid),
                "position": [float(node.position[0]), float(node.position[1]), float(node.position[2])],
                "rotation": [float(x) for x in node.rotation],
                "room_label": str(meta.get("room_label", "unknown")),
                "room_score": float(meta.get("room_score", 0.0)),
                "embed_idx": int(node.embed_idx),
            }
        )
    edges = []
    for u, nbrs in graph.edges.items():
        for v, w in nbrs.items():
            if int(u) >= int(v):
                continue
            key = (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))
            edges.append(
                {
                    "u": int(u),
                    "v": int(v),
                    "w": float(w),
                    "edge_type": str(graph.edge_types.get(key, "temporal")),
                }
            )
    return {"meta": dict(graph.meta), "places": places, "edges": edges}


def build_room_graph(
    graph: TopoGraph,
    node_meta: Dict[str, Dict],
    clip_embeds: np.ndarray,
    room_labels: Optional[List[str]] = None,
    room_text_embeds: Optional[np.ndarray] = None,
    scene_id: str = "unknown",
) -> Dict:
    seg_mode_env = os.environ.get("TOPO_ROOM_SEG_MODE", "").strip().lower()
    if seg_mode_env in {"edge_filtration", "edge", "hydralite"}:
        prefer_hydralite = True
    else:
        prefer_hydralite = os.environ.get("TOPO_ROOM_SEG_METHOD", "hydralite").strip().lower() != "fallback"
    if prefer_hydralite:
        rooms, room_of_node, label_room_counts, cluster_info = _cluster_rooms_hydralite(
            graph=graph,
            node_meta=node_meta,
            clip_embeds=clip_embeds,
        )
    else:
        rooms, room_of_node, label_room_counts, cluster_info = _cluster_rooms_by_connectivity_fallback(
            graph=graph,
            node_meta=node_meta,
            clip_embeds=clip_embeds,
        )

    if room_labels is None or len(room_labels) == 0:
        room_labels = list(DEFAULT_ROOM_LABELS)
    room_labels_norm = [canonical_room_label(x) for x in room_labels]
    room_text_norm = None
    if room_text_embeds is not None and isinstance(room_text_embeds, np.ndarray) and room_text_embeds.size > 0:
        room_text_norm = room_text_embeds.astype(np.float32)
        room_text_norm = room_text_norm / (np.linalg.norm(room_text_norm, axis=1, keepdims=True) + 1e-6)

    clip_margin_th = float(os.environ.get("TOPO_ROOM_CLIP_MARGIN_TH", "0.04"))
    clip_min_score = float(os.environ.get("TOPO_ROOM_CLIP_MIN_SCORE", "0.18"))
    clip_k = max(1, int(os.environ.get("TOPO_ROOM_CLIP_AGG_K", "4")))
    seg_method = str(cluster_info.get("method", "unknown"))
    seg_mode = str(cluster_info.get("mode", "PLATEAU"))
    delta_star = float(cluster_info.get("delta_star", -1.0))
    plateau_edges = int(cluster_info.get("plateau_edges", 0))
    plateau_delta = float(cluster_info.get("plateau_delta", 0.0))
    seg_fallback = int(cluster_info.get("fallback", 0))
    seg_K = len(rooms)
    seg_deltas = [round(float(x), 4) for x in cluster_info.get("deltas", [])]
    seg_Ks = [int(x) for x in cluster_info.get("Ks", [])]
    clear_stats = dict(cluster_info.get("clearance_stats", {}))
    if len(clear_stats) == 0:
        clear_vals = []
        for nid in sorted(graph.nodes.keys()):
            meta = node_meta.get(str(int(nid)), {})
            cv = float(meta.get("clearance_m", meta.get("clearance_proxy", 0.0)))
            if np.isfinite(cv):
                clear_vals.append(float(max(0.0, cv)))
        if len(clear_vals) > 0:
            carr = np.asarray(clear_vals, dtype=np.float32)
            clear_stats = {
                "min": float(np.min(carr)),
                "p10": float(np.quantile(carr, 0.10)),
                "p50": float(np.quantile(carr, 0.50)),
                "p90": float(np.quantile(carr, 0.90)),
                "max": float(np.max(carr)),
            }
    edge_debug = dict(cluster_info.get("edge_clearance_debug", {}))
    edge_stats = dict(cluster_info.get("edge_clearance_stats", edge_debug.get("stats", {})))
    doorway_debug = dict(cluster_info.get("doorway_debug", {}))
    if len(clear_stats) > 0:
        print(
            f"[TOPO_CLEARANCE_STATS] scene={scene_id} nodes={len(graph.nodes)} "
            f"min={float(clear_stats.get('min', 0.0)):.4f} p10={float(clear_stats.get('p10', 0.0)):.4f} "
            f"p50={float(clear_stats.get('p50', 0.0)):.4f} p90={float(clear_stats.get('p90', 0.0)):.4f} "
            f"max={float(clear_stats.get('max', 0.0)):.4f}",
            flush=True,
        )
    if len(edge_stats) > 0:
        print(
            f"[TOPO_EDGE_CLEARANCE] scene={scene_id} edges={int(edge_debug.get('edge_count', 0))} "
            f"sample_steps={int(edge_debug.get('sample_steps', 0))} min={float(edge_stats.get('min', 0.0)):.4f} "
            f"p10={float(edge_stats.get('p10', 0.0)):.4f} p50={float(edge_stats.get('p50', 0.0)):.4f} "
            f"p90={float(edge_stats.get('p90', 0.0)):.4f} max={float(edge_stats.get('max', 0.0)):.4f}",
            flush=True,
        )
    print(
        f"[TOPO_ROOM_SEGMENTATION] method={seg_method} scene={scene_id} "
        f"nodes={len(graph.nodes)} edges={int(edge_debug.get('edge_count', 0))} "
        f"delta_star={delta_star:.4f} K={seg_K} plateau_edges={plateau_edges} "
        f"plateau_delta={plateau_delta:.4f} fallback={seg_fallback} mode={seg_mode}",
        flush=True,
    )
    if len(seg_deltas) > 0 and len(seg_Ks) > 0:
        print(
            f"[TOPO_ROOM_SEG_CURVE] scene={scene_id} deltas={seg_deltas[:30]} Ks={seg_Ks[:30]}",
            flush=True,
        )
    print(
        f"[TOPO_DOORWAYS] scene={scene_id} delta_star={float(doorway_debug.get('delta_star', delta_star)):.4f} "
        f"doorway_edges={int(doorway_debug.get('doorway_edges_count', 0))} "
        f"doorway_nodes={int(doorway_debug.get('doorway_nodes_count', 0))}",
        flush=True,
    )

    # Scene-level DF suppression for object tokens.
    node_labels_per_node: Dict[int, set] = {}
    for nid in sorted(graph.nodes.keys()):
        labels_set = set()
        meta = node_meta.get(str(int(nid)), {})
        for det in meta.get("object_detections", []) or []:
            lbl = normalize_object_label(det.get("label", ""))
            if lbl:
                labels_set.add(str(lbl))
        node_labels_per_node[int(nid)] = labels_set
    N_df = max(1, len(node_labels_per_node))
    df_counter = collections.Counter()
    for _, labels_set in node_labels_per_node.items():
        for lbl in labels_set:
            df_counter[str(lbl)] += 1
    df_weights: Dict[str, float] = {}
    df_min_w = float(os.environ.get("TOPO_OBJ_DF_MIN_W", "0.08"))
    for lbl, df in df_counter.items():
        p = float(df) / float(N_df)
        df_weights[str(lbl)] = float(max(float(df_min_w), max(0.0, 1.0 - p) ** 2))
    df_rows = sorted(
        [(lbl, int(df), float(df) / float(N_df), float(df_weights.get(lbl, 1.0))) for lbl, df in df_counter.items()],
        key=lambda x: (-x[3], -x[1], x[0]),
    )
    print(
        f"[TOPO_OBJ_DF_SUMMARY] scene={scene_id} N={N_df} "
        f"top={[ (x[0], x[1], round(x[2], 3), round(x[3], 3)) for x in df_rows[:10] ]}",
        flush=True,
    )

    # v16 room semantics:
    # 1) Multi-view CLIP aggregation per room with margin gating
    # 2) Object-first strong-rule inference; CLIP only fallback.
    label_room_counts = collections.defaultdict(int)
    for rid_str, room in rooms.items():
        room_nodes = [int(x) for x in room.get("node_ids", [])]
        obj_hist_raw = collections.defaultdict(float)
        obj_hist = collections.defaultdict(float)

        room_nodes_sorted = sorted(
            room_nodes,
            key=lambda nid: int(graph.nodes[int(nid)].step_idx),
        )
        if len(room_nodes_sorted) <= clip_k:
            rep_nodes = list(room_nodes_sorted)
        else:
            rep_idx = np.linspace(0, len(room_nodes_sorted) - 1, num=clip_k, dtype=np.int32).tolist()
            rep_nodes = []
            for i in rep_idx:
                nid = int(room_nodes_sorted[int(i)])
                if nid not in rep_nodes:
                    rep_nodes.append(nid)

        rep_embed_rows = []
        for nid in rep_nodes:
            meta = node_meta.get(str(int(nid)), {})
            clip_idx = int(meta.get("clip_embed_idx", meta.get("embed_idx", -1)))
            if 0 <= clip_idx < clip_embeds.shape[0]:
                rep_embed_rows.append(clip_embeds[clip_idx].astype(np.float32))

        clip_label = "unknown_clip"
        clip_score = 0.0
        clip_topk = []
        clip_margin = 0.0
        if len(rep_embed_rows) > 0 and room_text_norm is not None and room_text_norm.shape[0] > 0:
            room_img = np.stack(rep_embed_rows, axis=0).mean(axis=0)
            room_img = room_img / (np.linalg.norm(room_img) + 1e-6)
            sims = room_text_norm.dot(room_img)
            kk = min(3, sims.shape[0])
            top_idx = np.argpartition(-sims, kk - 1)[:kk]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            clip_topk = [(room_labels_norm[int(i)], float(sims[int(i)])) for i in top_idx]
            top1_label = str(clip_topk[0][0]) if len(clip_topk) > 0 else "unknown_clip"
            top1_score = float(clip_topk[0][1]) if len(clip_topk) > 0 else 0.0
            top2_label = str(clip_topk[1][0]) if len(clip_topk) > 1 else "unknown_clip"
            top2_score = float(clip_topk[1][1]) if len(clip_topk) > 1 else 0.0
            clip_margin = float(top1_score - top2_score)
            if clip_margin >= clip_margin_th and top1_score >= clip_min_score:
                clip_label = top1_label
                clip_score = top1_score
            print(
                f"[TOPO_ROOM_CLIP_AGG] room={rid_str} K={len(rep_nodes)} "
                f"top1={top1_label}:{top1_score:.4f} top2={top2_label}:{top2_score:.4f} "
                f"margin={clip_margin:.4f} clip_label={clip_label}",
                flush=True,
            )
        else:
            print(
                f"[TOPO_ROOM_CLIP_AGG] room={rid_str} K={len(rep_nodes)} top1=none top2=none "
                f"margin=0.0000 clip_label=unknown_clip",
                flush=True,
            )

        for nid in room_nodes:
            meta = node_meta.get(str(int(nid)), {})
            for det in meta.get("object_detections", []) or []:
                obj_label = normalize_object_label(det.get("label", ""))
                if not obj_label:
                    continue
                obj_score = float(det.get("score", 1.0))
                if obj_score <= 0.0:
                    obj_score = 1.0
                obj_hist_raw[obj_label] += float(obj_score)
                obj_hist[obj_label] += float(obj_score) * float(df_weights.get(str(obj_label), 1.0))

        raw_out = {
            str(k): float(v)
            for k, v in sorted(obj_hist_raw.items(), key=lambda kv: kv[1], reverse=True)
            if float(v) > 0.0
        }

        semantic = infer_room_label_from_objects(
            dict(obj_hist),
            clip_label=clip_label,
            clip_score=float(clip_score),
            clip_topk=clip_topk,
            clip_margin=float(clip_margin),
        )
        source = str(semantic.get("label_source", "unknown"))
        final_label = str(semantic.get("label", "unknown"))
        final_score = float(semantic.get("score", 0.0))
        final_topk = [(str(lbl), float(sc)) for lbl, sc in semantic.get("topk", [])]
        strong_hit = str(semantic.get("strong_hit", ""))
        generic_only = bool(semantic.get("generic_only", False))
        top1_label = str(final_topk[0][0]) if len(final_topk) > 0 else "unknown"
        top1_score = float(final_topk[0][1]) if len(final_topk) > 0 else 0.0
        top2_label = str(final_topk[1][0]) if len(final_topk) > 1 else "unknown"
        top2_score = float(final_topk[1][1]) if len(final_topk) > 1 else 0.0
        label_margin = float(top1_score - top2_score)

        object_hist_out = {
            str(k): float(v)
            for k, v in sorted(obj_hist.items(), key=lambda kv: kv[1], reverse=True)
            if float(v) > 0.0
        }
        score_vals = np.asarray([float(max(0.0, sc)) for _, sc in final_topk], dtype=np.float32)
        if score_vals.size > 0 and float(np.sum(score_vals)) > 1e-8:
            probs = score_vals / float(np.sum(score_vals))
            label_entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
        else:
            label_entropy = 0.0
        total_obj_weighted = float(sum(object_hist_out.values()))
        if total_obj_weighted > 1e-8:
            generic_mass = float(
                sum(float(v) for k, v in object_hist_out.items() if str(k) in GENERIC_OBJECTS)
            )
            generic_ratio = float(generic_mass / total_obj_weighted)
        else:
            generic_ratio = 0.0
        room["object_hist_raw"] = raw_out
        room["object_hist"] = object_hist_out
        room["label_source"] = str(source)
        room["label"] = str(final_label)
        room["label_score"] = float(final_score)
        room["label_topk"] = [[str(lbl), float(sc)] for lbl, sc in final_topk]
        room["rule_hits"] = semantic.get("rule_hits", {})
        room["strong_hit"] = str(strong_hit)
        room["generic_only"] = bool(generic_only)
        room["clip_label"] = str(clip_label)
        room["clip_score"] = float(clip_score)
        room["clip_margin"] = float(clip_margin)
        room["clip_topk"] = [[str(lbl), float(sc)] for lbl, sc in clip_topk]
        room["mean_room_score"] = float(final_score)
        room["label_top1"] = [str(top1_label), float(top1_score)]
        room["label_top2"] = [str(top2_label), float(top2_score)]
        room["label_margin"] = float(label_margin)
        room["label_entropy"] = float(label_entropy)
        room["generic_ratio"] = float(generic_ratio)
        label_room_counts[str(room.get("label", "unknown"))] += 1

        total_obj = int(round(float(sum(object_hist_out.values()))))
        top_obj = list(object_hist_out.items())[:5]
        top_obj_raw = list(raw_out.items())[:5]
        print(
            f"[TOPO_ROOM_OBJECTS] room={rid_str} top={[(k, round(v, 3)) for k, v in top_obj]} total={total_obj}",
            flush=True,
        )
        print(
            f"[TOPO_OBJ_DF_ROOM] scene={scene_id} room={rid_str} "
            f"top_raw={[(k, round(v, 3)) for k, v in top_obj_raw]} "
            f"top_weighted={[(k, round(v, 3)) for k, v in top_obj]}",
            flush=True,
        )
        print(
            f"[TOPO_ROOM_OBJECT_RULE] room={rid_str} strong_hit={strong_hit if strong_hit else 'none'} "
            f"generic_only={int(generic_only)} final={final_label}",
            flush=True,
        )
        print(
            f"[TOPO_ROOM_LABEL_V16] room={rid_str} source={source} label={final_label} "
            f"score={float(final_score):.4f} topk={[(lbl, round(sc, 4)) for lbl, sc in final_topk[:3]]}",
            flush=True,
        )
        print(
            f"[TOPO_ROOM_LABEL_CONF] scene={scene_id} room={rid_str} final={final_label} "
            f"margin={label_margin:.4f} entropy={label_entropy:.4f} generic_ratio={generic_ratio:.4f}",
            flush=True,
        )

    edge_pairs = {}
    for u, nbrs in graph.edges.items():
        ru = room_of_node.get(int(u))
        if ru is None:
            continue
        for v, w in nbrs.items():
            rv = room_of_node.get(int(v))
            if rv is None or ru == rv:
                continue
            a, b = sorted((int(ru), int(rv)))
            key = (a, b)
            old = edge_pairs.get(key, float("inf"))
            edge_pairs[key] = min(old, float(w))

    room_edges = []
    doorway_pair_clearance: Dict[Tuple[int, int], float] = {}
    for row in doorway_debug.get("doorway_edges", []) or []:
        uu = int(row.get("u", -1))
        vv = int(row.get("v", -1))
        ru = int(row.get("room_u", room_of_node.get(uu, -1)))
        rv = int(row.get("room_v", room_of_node.get(vv, -1)))
        if ru < 0 or rv < 0 or ru == rv:
            continue
        a, b = sorted((ru, rv))
        key = (int(a), int(b))
        ec = float(row.get("edge_clearance_m", 0.0))
        if key not in doorway_pair_clearance:
            doorway_pair_clearance[key] = float(ec)
        else:
            doorway_pair_clearance[key] = min(float(doorway_pair_clearance[key]), float(ec))

    for (a, b), w in sorted(edge_pairs.items()):
        et = "doorway" if (int(a), int(b)) in doorway_pair_clearance else "adjacent"
        room_edges.append({"u": int(a), "v": int(b), "w": float(w), "edge_type": et})
    doorway_edges = [
        {
            "u": int(a),
            "v": int(b),
            "w": float(edge_pairs.get((int(a), int(b)), 0.0)),
            "edge_clearance_m": float(ec),
            "edge_type": "doorway",
        }
        for (a, b), ec in sorted(doorway_pair_clearance.items())
    ]

    room_adj: Dict[int, List[int]] = collections.defaultdict(list)
    for e in room_edges:
        u, v = int(e["u"]), int(e["v"])
        room_adj[u].append(v)
        room_adj[v].append(u)
    room_component = {}
    comp_id = 0
    for rid_str in rooms.keys():
        rid = int(rid_str)
        if rid in room_component:
            continue
        stack = [rid]
        room_component[rid] = comp_id
        while stack:
            x = stack.pop()
            for y in room_adj.get(x, []):
                if y in room_component:
                    continue
                room_component[y] = comp_id
                stack.append(y)
        comp_id += 1
    for rid_str, room in rooms.items():
        rid = int(rid_str)
        room["community_id"] = int(room_component.get(rid, -1))

    place2room = {str(int(k)): int(v) for k, v in room_of_node.items()}
    k_curve = [[float(d), int(k)] for d, k in zip(cluster_info.get("deltas", []), cluster_info.get("Ks", []))]
    segmentation_debug = {
        "scene_id": str(scene_id),
        "method": str(seg_method),
        "mode": str(seg_mode),
        "delta_star": float(delta_star),
        "K": int(seg_K),
        "plateau_edges": int(plateau_edges),
        "plateau_delta": float(plateau_delta),
        "fallback": int(seg_fallback),
        "fallback_reason": str(cluster_info.get("fallback_reason", "")),
        "deltas": [float(x) for x in cluster_info.get("deltas", [])],
        "Ks": [int(x) for x in cluster_info.get("Ks", [])],
        "scores": [float(x) for x in cluster_info.get("scores", [])],
        "sizes": [int(x) for x in cluster_info.get("sizes", [])],
        "clearance_stats": clear_stats,
        "edge_clearance_stats": edge_stats,
        "k_curve": k_curve,
        "plateaus": cluster_info.get("plateaus", []),
        "doorway_edges_count": int(doorway_debug.get("doorway_edges_count", 0)),
        "doorway_nodes_count": int(doorway_debug.get("doorway_nodes_count", 0)),
    }
    room_graph = {
        "rooms": rooms,
        "edges": room_edges,
        "doorway_edges": doorway_edges,
        "doorway_nodes": doorway_debug.get("doorway_nodes", []),
        "place2room": place2room,
        "place_to_room": dict(place2room),
        "label_room_counts": dict(label_room_counts),
        "cluster_info": cluster_info,
        "segmentation_debug": segmentation_debug,
        "edge_clearance_debug": edge_debug,
        "doorway_debug": doorway_debug,
    }
    return room_graph


def build_object_graph(
    node_meta: Dict[str, Dict],
    clip_embeds: np.ndarray,
    room_graph: Dict,
    object_merge_dist: float = 1.2,
) -> Tuple[Dict, List[Dict]]:
    place2room = {
        int(k): int(v)
        for k, v in room_graph.get("place2room", {}).items()
    }
    object_nodes: List[Dict] = []
    observed_edges = []
    contained_edges = []

    def _merge_target(label: str, centroid: np.ndarray) -> Optional[int]:
        best_idx = None
        best_dist = float("inf")
        for idx, obj in enumerate(object_nodes):
            if str(obj["label"]) != str(label):
                continue
            d = float(np.linalg.norm(np.array(obj["centroid"], dtype=np.float32) - centroid))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_idx is not None and best_dist <= float(object_merge_dist):
            return int(best_idx)
        return None

    for place_id_str, meta in node_meta.items():
        place_id = int(place_id_str)
        dets = meta.get("object_detections", [])
        clip_idx = int(meta.get("clip_embed_idx", meta.get("embed_idx", -1)))
        place_embed = None
        if 0 <= clip_idx < clip_embeds.shape[0]:
            place_embed = clip_embeds[clip_idx].astype(np.float32)
        for det in dets:
            label = str(det.get("label", "unknown")).strip().lower()
            centroid = np.array(det.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float32)
            score = float(det.get("score", 0.0))
            tgt = _merge_target(label, centroid)
            if tgt is None:
                obj_id = len(object_nodes)
                object_nodes.append(
                    {
                        "object_id": int(obj_id),
                        "label": label,
                        "centroid": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
                        "supporting_places": [int(place_id)],
                        "scores": [float(score)],
                        "embed_sum": place_embed.tolist() if place_embed is not None else [],
                        "embed_count": 1 if place_embed is not None else 0,
                    }
                )
                tgt = obj_id
            else:
                obj = object_nodes[tgt]
                prev_centroid = np.array(obj["centroid"], dtype=np.float32)
                cnt = max(1, len(obj["supporting_places"]))
                new_centroid = (prev_centroid * cnt + centroid) / float(cnt + 1)
                obj["centroid"] = [float(new_centroid[0]), float(new_centroid[1]), float(new_centroid[2])]
                if int(place_id) not in obj["supporting_places"]:
                    obj["supporting_places"].append(int(place_id))
                obj["scores"].append(float(score))
                if place_embed is not None:
                    if obj["embed_count"] <= 0 or len(obj["embed_sum"]) == 0:
                        obj["embed_sum"] = place_embed.tolist()
                        obj["embed_count"] = 1
                    else:
                        prev = np.array(obj["embed_sum"], dtype=np.float32)
                        obj["embed_sum"] = (prev + place_embed).tolist()
                        obj["embed_count"] = int(obj["embed_count"]) + 1
            observed_edges.append(
                {
                    "u_place": int(place_id),
                    "v_object": int(tgt),
                    "edge_type": "observed_by",
                    "score": float(score),
                }
            )

    objects_dict = {}
    for obj in object_nodes:
        support = [int(x) for x in sorted(set(obj["supporting_places"]))]
        room_votes = collections.Counter(place2room.get(pid, -1) for pid in support)
        room_id = int(room_votes.most_common(1)[0][0]) if len(room_votes) > 0 else -1
        if room_id >= 0:
            contained_edges.append(
                {
                    "u_object": int(obj["object_id"]),
                    "v_room": int(room_id),
                    "edge_type": "contained_in",
                }
            )
        if int(obj.get("embed_count", 0)) > 0 and len(obj.get("embed_sum", [])) > 0:
            emb = np.array(obj["embed_sum"], dtype=np.float32) / float(obj["embed_count"])
            emb = emb / (np.linalg.norm(emb) + 1e-6)
            emb_list = emb.tolist()
        else:
            emb_list = []
        objects_dict[str(int(obj["object_id"]))] = {
            "object_id": int(obj["object_id"]),
            "label": str(obj["label"]),
            "centroid": [float(x) for x in obj["centroid"]],
            "supporting_places": support,
            "support_count": len(support),
            "score_mean": float(np.mean(obj["scores"])) if len(obj["scores"]) > 0 else 0.0,
            "room_id": int(room_id),
            "embed": emb_list,
        }

    object_graph = {
        "objects": objects_dict,
        "edges_observed_by": observed_edges,
        "edges_contained_in": contained_edges,
        "num_objects": len(objects_dict),
    }
    return object_graph, list(objects_dict.values())


def build_gt_room_audit(
    scene_id: str,
    graph: TopoGraph,
    node_meta: Dict[str, Dict],
    room_graph: Dict,
    semantic_regions: List[Dict],
) -> Dict:
    labels = ["bathroom", "kitchen", "bedroom", "living_room", "dining_room", "office", "hallway", "unknown"]
    li = {lbl: i for i, lbl in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=np.int32)  # gt(non-unknown) x pred
    rows = []
    gt_counts = collections.Counter()
    non_unknown_nodes = 0

    place2room = {
        int(k): int(v)
        for k, v in room_graph.get("place2room", {}).items()
    }
    rooms = room_graph.get("rooms", {})

    for nid in sorted(graph.nodes.keys()):
        meta = node_meta.get(str(int(nid)), {})
        gt_pos = meta.get("gt_position", meta.get("position", graph.nodes[int(nid)].position))
        gt_label = _infer_gt_room_label_from_regions(gt_pos, semantic_regions)
        gt_label = _map_room_text_to_label(gt_label)
        room_id = int(place2room.get(int(nid), -1))
        pred_label = "unknown"
        if room_id >= 0 and str(room_id) in rooms:
            pred_label = canonical_room_label(rooms[str(room_id)].get("label", "unknown"))
        if gt_label not in li:
            gt_label = "unknown"
        if pred_label not in li:
            pred_label = "unknown"
        if gt_label != "unknown":
            mat[li[gt_label], li[pred_label]] += 1
            gt_counts[gt_label] += 1
            non_unknown_nodes += 1
        rows.append(
            {
                "node_id": int(nid),
                "gt_label": str(gt_label),
                "pred_label": str(pred_label),
                "room_id": int(room_id),
                "included_in_agreement": int(gt_label != "unknown"),
            }
        )

    total = int(non_unknown_nodes)
    if total > 0:
        correct = int(sum(int(mat[i, i]) for i in range(len(labels))))
        top1 = float(correct / float(total))
        gt_available = 1
    else:
        top1 = -1.0
        gt_available = 0
    return {
        "scene_id": str(scene_id),
        "gt_available": int(gt_available),
        "labels": labels,
        "matrix": mat.tolist(),
        "top1_agreement": float(top1),
        "total_nodes": int(len(graph.nodes)),
        "non_unknown_nodes": int(non_unknown_nodes),
        "gt_label_counts": dict(gt_counts),
        "per_node": rows,
    }


def build_topograph(
    config_path: str,
    out_dir: str,
    steps: int,
    save_every_k: int,
    model_path: str,
    pooling: str,
    max_episodes: int,
    seed: int,
    scene_id: Optional[str] = None,
    sample_frames: int = 4,
    min_node_spacing: float = 0.10,
    stuck_move_thresh: float = 0.02,
    stuck_forward_limit: int = 3,
    recovery_turn_steps: int = 2,
    vision_prompt: str = "",
    loop_closure_sim_thresh: float = 0.93,
    loop_closure_geo_thresh: float = 2.0,
    loop_closure_min_hops: int = 3,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    start_episode_offset: int = 0,
    room_labels_csv: str = "",
    room_label_topk: int = 3,
    geo_bins_r: int = 20,
    geo_bins_theta: int = 60,
    geo_max_depth: float = 5.0,
    merge_geo_dist_thresh: float = 0.5,
    merge_vision_sim_thresh: float = 0.98,
    merge_geo_sim_thresh: float = 0.98,
    loop_closure_geo_sim_thresh: float = 0.95,
    max_nodes: int = 30,
    object_labels_csv: str = "",
    object_label_topk: int = 3,
    object_min_score: float = 0.18,
    object_merge_dist: float = 1.2,
    use_est_pose: bool = False,
    odom_noise_trans: float = 0.02,
    odom_noise_yaw_deg: float = 1.0,
    lc_rel_trans_err_max: float = 0.85,
    lc_rel_yaw_err_deg_max: float = 35.0,
    lc_weight_min: float = 0.8,
    lc_weight_max: float = 2.5,
    pg_loop_robust: str = "huber",
    pg_loop_robust_scale: float = 0.75,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Force episode cap for MVP runs
    os.environ["EVAL_MODE"] = "1"
    os.environ["MAX_EPISODES"] = str(max_episodes)

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
    if scene_id:
        filtered = [ep for ep in env.episodes if scene_id in ep.scene_id]
        env.episodes = filtered
        if len(filtered) > 0:
            env._current_episode_index = int(start_episode_offset) % len(filtered)
        else:
            env._current_episode_index = 0
        env.is_running = len(filtered) > 0
        print(
            f"[TOPO_BUILD] scene_filter={scene_id} episodes={len(filtered)} start_episode_offset={start_episode_offset}",
            flush=True,
        )

    embedder = TopoEmbedder(model_path=model_path, pooling=pooling)
    clip_embedder = TopoClipEmbedder(model_name=clip_model_name)
    room_labels = parse_room_labels(room_labels_csv)
    room_text_embeds = build_room_text_embeddings(clip_embedder, room_labels)
    object_labels = parse_object_labels(object_labels_csv)
    object_template = os.environ.get("TOPO_OBJECT_TEXT_TEMPLATE", "a photo of a {label}")
    room_template = os.environ.get("TOPO_ROOM_TEXT_TEMPLATE", "a photo of a {label} room")
    object_text_embeds = build_text_embeddings(clip_embedder, object_labels, template=object_template)
    print(
        f"[TOPO_CLIP_TEXT_TEMPLATE] objects_template=\"{object_template}\" rooms_template=\"{room_template}\"",
        flush=True,
    )
    graph = TopoGraph()
    embeds: List[np.ndarray] = []
    clip_embeds: List[np.ndarray] = []
    geo_embeds: List[np.ndarray] = []
    node_meta = {}

    rng = random.Random(seed)
    total_steps = 0
    node_id = 0
    saved_frames = 0
    node_order: List[int] = []
    last_kept_node: Optional[int] = None

    first_scene_id = None
    semantic_id_to_label = _collect_semantic_label_map(env._env.sim)
    semantic_regions, semantic_region_reason = _collect_semantic_regions(env._env.sim)
    forward_stuck_count = 0
    force_turn_steps = 0
    skipped_near_duplicate_nodes = 0
    skipped_max_nodes = 0
    loop_closure_edges = 0
    pose_graph = PoseGraphLite(loop_robust=pg_loop_robust, loop_robust_scale=pg_loop_robust_scale)
    est_x = 0.0
    est_y = 0.0
    est_z = 0.0
    est_yaw = 0.0
    odom_noise_yaw = float(np.deg2rad(float(odom_noise_yaw_deg)))
    lc_rel_yaw_err_max = float(np.deg2rad(float(lc_rel_yaw_err_deg_max)))

    while env.is_running and total_steps < steps:
        obs = env.reset()
        if not env.is_running or obs is None:
            break
        state0 = env._env.sim.get_agent_state()
        est_x = float(state0.position[0])
        est_y = float(state0.position[1])
        est_z = float(state0.position[2])
        est_yaw = float(_yaw_from_quat_wxyz([state0.rotation.w, state0.rotation.x, state0.rotation.y, state0.rotation.z]))

        episode = env.get_current_episode()
        if first_scene_id is None and hasattr(episode, "scene_id"):
            first_scene_id = str(episode.scene_id)
        done = False
        while not done and total_steps < steps:
            if total_steps % save_every_k == 0:
                rgb = obs.get("rgb")
                if rgb is None:
                    raise RuntimeError("Observation missing 'rgb' key.")
                depth = obs.get("depth")
                if depth is None:
                    depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
                semantic_obs = obs.get("semantic")

                state = env._env.sim.get_agent_state()
                gt_position = [float(state.position[0]), float(state.position[1]), float(state.position[2])]
                gt_rotation = [
                    float(state.rotation.w),
                    float(state.rotation.x),
                    float(state.rotation.y),
                    float(state.rotation.z),
                ]
                gt_yaw = float(_yaw_from_quat_wxyz(gt_rotation))
                est_position = [float(est_x), float(gt_position[1]), float(est_z)]
                est_rotation = _quat_wxyz_from_yaw(est_yaw)
                if use_est_pose:
                    position = est_position
                    rotation = est_rotation
                else:
                    position = gt_position
                    rotation = gt_rotation

                emb = embedder.embed_image(rgb, instruction=vision_prompt)
                clip_emb = clip_embedder.embed_image(rgb)
                geo_emb = compute_scan_context(
                    depth=depth,
                    bins_r=geo_bins_r,
                    bins_theta=geo_bins_theta,
                    max_depth=geo_max_depth,
                ).astype(np.float16)
                clearance_proxy = _estimate_clearance_proxy_from_depth(depth_obs=depth, max_depth=float(geo_max_depth))
                room_label, room_score, room_top3 = score_room_label_from_embed(
                    clip_embed=clip_emb,
                    room_labels=room_labels,
                    room_text_embeds=room_text_embeds,
                    topk=room_label_topk,
                )
                object_detections = detect_objects_for_node(
                    rgb=rgb,
                    depth=depth,
                    clip_embed=clip_emb,
                    clip_embedder=clip_embedder,
                    position=gt_position,
                    rotation_wxyz=gt_rotation,
                    object_labels=object_labels,
                    object_text_embeds=object_text_embeds,
                    semantic_obs=semantic_obs,
                    semantic_id_to_label=semantic_id_to_label,
                    topk=object_label_topk,
                    min_score=object_min_score,
                )

                proposed_node_id = int(node_id)
                node_id += 1

                kept_node = None
                best_merge_score = -1.0
                if last_kept_node is not None:
                    last_dist = l2_distance(graph.nodes[last_kept_node].position, position)
                    if last_dist < min_node_spacing:
                        kept_node = int(last_kept_node)
                        best_merge_score = 1.0

                for cand_id in node_order:
                    cand = graph.nodes[cand_id]
                    geo_dist = l2_distance(cand.position, position)
                    if geo_dist >= merge_geo_dist_thresh:
                        continue
                    cand_idx = int(cand.embed_idx)
                    vis_sim = _cosine_sim(clip_emb, clip_embeds[cand_idx])
                    geo_sim = _cosine_sim(geo_emb, geo_embeds[cand_idx])
                    if vis_sim >= merge_vision_sim_thresh and geo_sim >= merge_geo_sim_thresh:
                        score = vis_sim + geo_sim
                        if score > best_merge_score:
                            best_merge_score = score
                            kept_node = int(cand_id)

                if kept_node is not None:
                    skipped_near_duplicate_nodes += 1
                    if last_kept_node is not None and last_kept_node != kept_node:
                        w = l2_distance(graph.nodes[last_kept_node].position, graph.nodes[kept_node].position)
                        graph.add_edge(
                            last_kept_node,
                            kept_node,
                            w,
                            bidir=True,
                            edge_type="temporal",
                        )
                        if str(last_kept_node) in node_meta and str(kept_node) in node_meta:
                            src_prev = node_meta[str(last_kept_node)]
                            src_next = node_meta[str(kept_node)]
                            if use_est_pose:
                                p0 = src_prev.get("est_position", src_prev.get("position", [0.0, 0.0, 0.0]))
                                p1 = src_next.get("est_position", src_next.get("position", [0.0, 0.0, 0.0]))
                                y0 = float(src_prev.get("est_yaw", 0.0))
                                y1 = float(src_next.get("est_yaw", 0.0))
                            else:
                                p0 = src_prev.get("gt_position", src_prev.get("position", [0.0, 0.0, 0.0]))
                                p1 = src_next.get("gt_position", src_next.get("position", [0.0, 0.0, 0.0]))
                                y0 = float(src_prev.get("gt_yaw", 0.0))
                                y1 = float(src_next.get("gt_yaw", 0.0))
                            pose_graph.add_edge(
                                i=int(last_kept_node),
                                j=int(kept_node),
                                dx=float(p1[0]) - float(p0[0]),
                                dz=float(p1[2]) - float(p0[2]),
                                dyaw=wrap_angle(float(y1) - float(y0)),
                                edge_type="sequential",
                                weight=0.5,
                            )
                    last_kept_node = kept_node
                    print(
                        f"[TOPO_PRUNE] merged=1 kept_node={kept_node} dropped_node={proposed_node_id} "
                        f"reason=near_duplicate",
                        flush=True,
                    )
                elif len(graph.nodes) >= int(max_nodes):
                    skipped_max_nodes += 1
                    print(
                        f"[TOPO_PRUNE] merged=1 kept_node={last_kept_node} dropped_node={proposed_node_id} "
                        f"reason=max_nodes_cap",
                        flush=True,
                    )
                else:
                    embed_idx = len(embeds)
                    embeds.append(emb)
                    clip_embeds.append(clip_emb)
                    geo_embeds.append(geo_emb)

                    rgb_path = None
                    if saved_frames < sample_frames:
                        rgb_path = os.path.join(frames_dir, f"node_{proposed_node_id:04d}.png")
                        Image.fromarray(rgb.astype(np.uint8)).save(rgb_path)
                        saved_frames += 1

                    graph.add_node(
                        node_id=proposed_node_id,
                        step_idx=total_steps,
                        position=position,
                        rotation=rotation,
                        embed_idx=embed_idx,
                        rgb_path=rgb_path,
                    )
                    pose_graph.add_node(
                        node_id=proposed_node_id,
                        x=float(est_position[0]),
                        z=float(est_position[2]),
                        yaw=float(est_yaw),
                        gt_x=float(gt_position[0]),
                        gt_z=float(gt_position[2]),
                        gt_yaw=float(gt_yaw),
                    )
                    if last_kept_node is not None and last_kept_node != proposed_node_id:
                        w = l2_distance(graph.nodes[last_kept_node].position, position)
                        graph.add_edge(
                            last_kept_node,
                            proposed_node_id,
                            w,
                            bidir=True,
                            edge_type="temporal",
                        )
                        if str(last_kept_node) in node_meta:
                            src_prev = node_meta[str(last_kept_node)]
                            if use_est_pose:
                                p0 = src_prev.get("est_position", src_prev.get("position", [0.0, 0.0, 0.0]))
                                y0 = float(src_prev.get("est_yaw", 0.0))
                                p1 = est_position
                                y1 = float(est_yaw)
                            else:
                                p0 = src_prev.get("gt_position", src_prev.get("position", [0.0, 0.0, 0.0]))
                                y0 = float(src_prev.get("gt_yaw", 0.0))
                                p1 = gt_position
                                y1 = float(gt_yaw)
                            pose_graph.add_edge(
                                i=int(last_kept_node),
                                j=int(proposed_node_id),
                                dx=float(p1[0]) - float(p0[0]),
                                dz=float(p1[2]) - float(p0[2]),
                                dyaw=wrap_angle(float(y1) - float(y0)),
                                edge_type="sequential",
                                weight=1.0,
                            )
                    # loop closure edges (appearance + geometry + consistency verification)
                    if len(node_order) > loop_closure_min_hops:
                        for cand_id in node_order[: -int(loop_closure_min_hops)]:
                            cand_node = graph.nodes[cand_id]
                            cand_idx = int(cand_node.embed_idx)
                            geo_dist = l2_distance(cand_node.position, position)
                            vis_sim = _cosine_sim(clip_emb, clip_embeds[cand_idx])
                            geo_sim = _cosine_sim(geo_emb, geo_embeds[cand_idx])
                            lc_score = float(0.5 * (float(vis_sim) + float(geo_sim)))
                            print(
                                f"[TOPO_LC_CAND] i={cand_id} j={proposed_node_id} vis={vis_sim:.4f} "
                                f"geo={geo_sim:.4f} score={lc_score:.4f}",
                                flush=True,
                            )

                            reject_reason = None
                            cand_meta = node_meta.get(str(cand_id), {})
                            geo_ok = True
                            dist_err = 0.0
                            yaw_err = 0.0
                            if geo_dist > float(loop_closure_geo_thresh):
                                reject_reason = "geo_far"
                            elif vis_sim < float(loop_closure_sim_thresh):
                                reject_reason = "vis_low"
                            elif geo_sim < float(loop_closure_geo_sim_thresh):
                                reject_reason = "geo_low"
                            elif cand_id in graph.edges.get(proposed_node_id, {}):
                                reject_reason = "already_connected"
                            else:
                                if cand_meta:
                                    geo_ok, dist_err, yaw_err = _relative_pose_consistency(
                                        cand_meta=cand_meta,
                                        curr_gt_pos=gt_position,
                                        curr_gt_yaw=float(gt_yaw),
                                        curr_est_pos=est_position,
                                        curr_est_yaw=float(est_yaw),
                                        max_dist_err=float(lc_rel_trans_err_max),
                                        max_yaw_err=float(lc_rel_yaw_err_max),
                                    )
                                if not geo_ok:
                                    reject_reason = "geo_inconsistent"

                            if reject_reason is not None:
                                print(
                                    f"[TOPO_LC_REJECT] i={cand_id} j={proposed_node_id} reason={reject_reason} "
                                    f"dist={geo_dist:.3f} dist_err={dist_err:.3f} yaw_err={yaw_err:.3f}",
                                    flush=True,
                                )
                                continue

                            graph.add_edge(
                                cand_id,
                                proposed_node_id,
                                geo_dist,
                                bidir=True,
                                edge_type="loop",
                            )
                            if str(cand_id) in node_meta:
                                cand_meta = node_meta[str(cand_id)]
                                cand_gt_pos = cand_meta.get("gt_position", cand_meta.get("position", [0.0, 0.0, 0.0]))
                                cand_gt_yaw = float(cand_meta.get("gt_yaw", 0.0))
                                lc_weight = float(
                                    np.clip(
                                        float(lc_weight_min) + (float(lc_weight_max) - float(lc_weight_min)) * lc_score,
                                        float(lc_weight_min),
                                        float(lc_weight_max),
                                    )
                                )
                                pose_graph.add_edge(
                                    i=int(cand_id),
                                    j=int(proposed_node_id),
                                    dx=float(gt_position[0]) - float(cand_gt_pos[0]),
                                    dz=float(gt_position[2]) - float(cand_gt_pos[2]),
                                    dyaw=wrap_angle(float(gt_yaw) - float(cand_gt_yaw)),
                                    edge_type="loop",
                                    weight=lc_weight,
                                    sim=float(vis_sim),
                                )
                                print(
                                    f"[TOPO_LC_ADD] i={cand_id} j={proposed_node_id} "
                                    f"weight={lc_weight:.3f} robust={pg_loop_robust}",
                                    flush=True,
                                )
                            loop_closure_edges += 1
                            print(
                                f"[TOPO_LOOP] i={cand_id} j={proposed_node_id} sim={vis_sim:.4f}",
                                flush=True,
                            )
                    node_meta[str(proposed_node_id)] = {
                        "embed_idx": embed_idx,
                        "clip_embed_idx": embed_idx,
                        "geo_idx": embed_idx,
                        "position": position,
                        "rotation": rotation,
                        "pose_source": "est" if use_est_pose else "gt",
                        "gt_position": gt_position,
                        "gt_rotation": gt_rotation,
                        "gt_yaw": float(gt_yaw),
                        "est_position": est_position,
                        "est_yaw": float(est_yaw),
                        "rgb_path": rgb_path,
                        "room_label": room_label,
                        "room_score": float(room_score),
                        "room_top3": [[str(lbl), float(score)] for lbl, score in room_top3],
                        "clearance_m": float(clearance_proxy),
                        "clearance_proxy": float(clearance_proxy),
                        "object_detections": object_detections,
                    }
                    print(
                        f"[TOPO_ROOM_LABEL] node={proposed_node_id} label={room_label} score={room_score:.4f} "
                        f"top3={[(lbl, round(score, 4)) for lbl, score in room_top3]}",
                        flush=True,
                    )
                    node_order.append(int(proposed_node_id))
                    last_kept_node = int(proposed_node_id)

            if force_turn_steps > 0:
                action = 2 if rng.random() < 0.5 else 3
                force_turn_steps -= 1
            else:
                action = _sample_action(rng)

            pre_state = env._env.sim.get_agent_state()
            pre_pos = np.array(pre_state.position, dtype=np.float32)
            pre_yaw = float(_yaw_from_quat_wxyz([pre_state.rotation.w, pre_state.rotation.x, pre_state.rotation.y, pre_state.rotation.z]))
            obs, _, done, _ = env.step(action)
            post_state = env._env.sim.get_agent_state()
            post_pos = np.array(post_state.position, dtype=np.float32)
            post_yaw = float(_yaw_from_quat_wxyz([post_state.rotation.w, post_state.rotation.x, post_state.rotation.y, post_state.rotation.z]))
            moved = float(np.linalg.norm(post_pos - pre_pos))

            # Update a lightweight odometry pose with controllable drift.
            noisy_moved = float(max(0.0, moved + rng.gauss(0.0, float(odom_noise_trans))))
            yaw_delta_gt = wrap_angle(post_yaw - pre_yaw)
            yaw_delta_noisy = wrap_angle(yaw_delta_gt + rng.gauss(0.0, float(odom_noise_yaw)))
            est_yaw = wrap_angle(est_yaw + yaw_delta_noisy)
            if action == 1:
                est_x += float(np.sin(est_yaw) * noisy_moved)
                est_z += float(-np.cos(est_yaw) * noisy_moved)
                est_y = float(post_state.position[1])
            else:
                est_y = float(post_state.position[1])

            if action == 1 and moved < stuck_move_thresh:
                forward_stuck_count += 1
            else:
                forward_stuck_count = 0
            if forward_stuck_count >= stuck_forward_limit:
                force_turn_steps = recovery_turn_steps
                forward_stuck_count = 0
                print(
                    f"[TOPO_BUILD_RECOVERY] step={total_steps} moved={moved:.4f} force_turn_steps={recovery_turn_steps}",
                    flush=True,
                )

            total_steps += 1

    env.close()

    # Post-hoc loop closure sweep: catches revisit candidates that were missed
    # online due short-term filtering while still enforcing geo consistency.
    if len(node_order) > int(loop_closure_min_hops):
        for a_idx, cand_id in enumerate(node_order):
            for prop_id in node_order[a_idx + int(loop_closure_min_hops) :]:
                if cand_id == prop_id:
                    continue
                if cand_id in graph.edges.get(prop_id, {}):
                    continue
                cand_node = graph.nodes[int(cand_id)]
                prop_node = graph.nodes[int(prop_id)]
                geo_dist = l2_distance(cand_node.position, prop_node.position)
                cand_eidx = int(cand_node.embed_idx)
                prop_eidx = int(prop_node.embed_idx)
                vis_sim = _cosine_sim(clip_embeds[cand_eidx], clip_embeds[prop_eidx])
                geo_sim = _cosine_sim(geo_embeds[cand_eidx], geo_embeds[prop_eidx])
                lc_score = float(0.5 * (float(vis_sim) + float(geo_sim)))
                print(
                    f"[TOPO_LC_CAND] i={cand_id} j={prop_id} vis={vis_sim:.4f} geo={geo_sim:.4f} score={lc_score:.4f}",
                    flush=True,
                )

                reject_reason = None
                dist_err = 0.0
                yaw_err = 0.0
                if geo_dist > float(loop_closure_geo_thresh):
                    reject_reason = "geo_far"
                elif vis_sim < float(loop_closure_sim_thresh):
                    reject_reason = "vis_low"
                elif geo_sim < float(loop_closure_geo_sim_thresh):
                    reject_reason = "geo_low"
                else:
                    cand_meta = node_meta.get(str(int(cand_id)), {})
                    prop_meta = node_meta.get(str(int(prop_id)), {})
                    if not cand_meta or not prop_meta:
                        reject_reason = "missing_meta"
                    else:
                        geo_ok, dist_err, yaw_err = _relative_pose_consistency(
                            cand_meta=cand_meta,
                            curr_gt_pos=prop_meta.get("gt_position", prop_meta.get("position", [0.0, 0.0, 0.0])),
                            curr_gt_yaw=float(prop_meta.get("gt_yaw", prop_meta.get("est_yaw", 0.0))),
                            curr_est_pos=prop_meta.get("est_position", prop_meta.get("position", [0.0, 0.0, 0.0])),
                            curr_est_yaw=float(prop_meta.get("est_yaw", prop_meta.get("gt_yaw", 0.0))),
                            max_dist_err=float(lc_rel_trans_err_max),
                            max_yaw_err=float(lc_rel_yaw_err_max),
                        )
                        if not geo_ok:
                            reject_reason = "geo_inconsistent"

                if reject_reason is not None:
                    print(
                        f"[TOPO_LC_REJECT] i={cand_id} j={prop_id} reason={reject_reason} "
                        f"dist={geo_dist:.3f} dist_err={dist_err:.3f} yaw_err={yaw_err:.3f}",
                        flush=True,
                    )
                    continue

                graph.add_edge(
                    int(cand_id),
                    int(prop_id),
                    float(geo_dist),
                    bidir=True,
                    edge_type="loop",
                )
                cand_meta = node_meta[str(int(cand_id))]
                prop_meta = node_meta[str(int(prop_id))]
                cand_gt_pos = cand_meta.get("gt_position", cand_meta.get("position", [0.0, 0.0, 0.0]))
                prop_gt_pos = prop_meta.get("gt_position", prop_meta.get("position", [0.0, 0.0, 0.0]))
                cand_gt_yaw = float(cand_meta.get("gt_yaw", cand_meta.get("est_yaw", 0.0)))
                prop_gt_yaw = float(prop_meta.get("gt_yaw", prop_meta.get("est_yaw", 0.0)))
                lc_weight = float(
                    np.clip(
                        float(lc_weight_min) + (float(lc_weight_max) - float(lc_weight_min)) * lc_score,
                        float(lc_weight_min),
                        float(lc_weight_max),
                    )
                )
                pose_graph.add_edge(
                    i=int(cand_id),
                    j=int(prop_id),
                    dx=float(prop_gt_pos[0]) - float(cand_gt_pos[0]),
                    dz=float(prop_gt_pos[2]) - float(cand_gt_pos[2]),
                    dyaw=wrap_angle(float(prop_gt_yaw) - float(cand_gt_yaw)),
                    edge_type="loop",
                    weight=lc_weight,
                    sim=float(vis_sim),
                )
                loop_closure_edges += 1
                print(
                    f"[TOPO_LC_ADD] i={cand_id} j={prop_id} weight={lc_weight:.3f} robust={pg_loop_robust}",
                    flush=True,
                )
                print(
                    f"[TOPO_LOOP] i={cand_id} j={prop_id} sim={vis_sim:.4f}",
                    flush=True,
                )

    if loop_closure_edges <= 0 and len(node_order) > int(loop_closure_min_hops):
        # Fallback for degenerate trajectories: add the best near revisit pair so
        # loop constraints are observable in diagnostics.
        fallback_pairs = []
        for a_idx, cand_id in enumerate(node_order):
            for prop_id in node_order[a_idx + int(loop_closure_min_hops) :]:
                if cand_id == prop_id:
                    continue
                if cand_id in graph.edges.get(prop_id, {}):
                    continue
                cand_node = graph.nodes[int(cand_id)]
                prop_node = graph.nodes[int(prop_id)]
                geo_dist = l2_distance(cand_node.position, prop_node.position)
                if geo_dist > float(os.environ.get("TOPO_LC_FALLBACK_GEO_THRESH", "1.8")):
                    continue
                cand_eidx = int(cand_node.embed_idx)
                prop_eidx = int(prop_node.embed_idx)
                vis_sim = _cosine_sim(clip_embeds[cand_eidx], clip_embeds[prop_eidx])
                geo_sim = _cosine_sim(geo_embeds[cand_eidx], geo_embeds[prop_eidx])
                score = 0.5 * (float(vis_sim) + float(geo_sim)) - 0.05 * float(geo_dist)
                fallback_pairs.append((score, int(cand_id), int(prop_id), float(geo_dist), float(vis_sim), float(geo_sim)))

        fallback_pairs.sort(reverse=True, key=lambda x: x[0])
        max_fb = int(os.environ.get("TOPO_LC_FALLBACK_MAX", "2"))
        added_fb = 0
        for score, cand_id, prop_id, geo_dist, vis_sim, geo_sim in fallback_pairs:
            if added_fb >= max_fb:
                break
            if vis_sim < float(os.environ.get("TOPO_LC_FALLBACK_VIS_MIN", "0.82")):
                continue
            if geo_sim < float(os.environ.get("TOPO_LC_FALLBACK_GEO_MIN", "0.82")):
                continue
            graph.add_edge(
                int(cand_id),
                int(prop_id),
                float(geo_dist),
                bidir=True,
                edge_type="loop",
            )
            cand_meta = node_meta.get(str(int(cand_id)), {})
            prop_meta = node_meta.get(str(int(prop_id)), {})
            cand_gt_pos = cand_meta.get("gt_position", cand_meta.get("position", [0.0, 0.0, 0.0]))
            prop_gt_pos = prop_meta.get("gt_position", prop_meta.get("position", [0.0, 0.0, 0.0]))
            cand_gt_yaw = float(cand_meta.get("gt_yaw", cand_meta.get("est_yaw", 0.0)))
            prop_gt_yaw = float(prop_meta.get("gt_yaw", prop_meta.get("est_yaw", 0.0)))
            lc_weight = float(
                np.clip(
                    float(lc_weight_min) + (float(lc_weight_max) - float(lc_weight_min)) * float(score),
                    float(lc_weight_min),
                    float(lc_weight_max),
                )
            )
            pose_graph.add_edge(
                i=int(cand_id),
                j=int(prop_id),
                dx=float(prop_gt_pos[0]) - float(cand_gt_pos[0]),
                dz=float(prop_gt_pos[2]) - float(cand_gt_pos[2]),
                dyaw=wrap_angle(float(prop_gt_yaw) - float(cand_gt_yaw)),
                edge_type="loop",
                weight=lc_weight,
                sim=float(vis_sim),
            )
            loop_closure_edges += 1
            added_fb += 1
            print(
                f"[TOPO_LC_CAND] i={cand_id} j={prop_id} vis={vis_sim:.4f} geo={geo_sim:.4f} score={float(score):.4f}",
                flush=True,
            )
            print(
                f"[TOPO_LC_ADD] i={cand_id} j={prop_id} weight={lc_weight:.3f} robust={pg_loop_robust}",
                flush=True,
            )
            print(
                f"[TOPO_LOOP] i={cand_id} j={prop_id} sim={vis_sim:.4f}",
                flush=True,
            )

    graph.meta["steps"] = str(total_steps)
    graph.meta["save_every_k"] = str(save_every_k)
    graph.meta["config_path"] = config_path
    if first_scene_id:
        graph.meta["scene_id"] = first_scene_id
    graph.meta["vision_prompt"] = vision_prompt
    graph.meta["loop_closure_edges"] = str(loop_closure_edges)
    graph.meta["clip_model_name"] = clip_model_name
    graph.meta["start_episode_offset"] = str(start_episode_offset)
    graph.meta["geo_bins_r"] = str(geo_bins_r)
    graph.meta["geo_bins_theta"] = str(geo_bins_theta)
    graph.meta["geo_max_depth"] = str(geo_max_depth)
    graph.meta["max_nodes"] = str(max_nodes)
    graph.meta["use_est_pose"] = str(int(bool(use_est_pose)))
    graph.meta["odom_noise_trans"] = str(float(odom_noise_trans))
    graph.meta["odom_noise_yaw_deg"] = str(float(odom_noise_yaw_deg))
    graph.meta["lc_rel_trans_err_max"] = str(float(lc_rel_trans_err_max))
    graph.meta["lc_rel_yaw_err_deg_max"] = str(float(lc_rel_yaw_err_deg_max))
    graph.meta["lc_weight_min"] = str(float(lc_weight_min))
    graph.meta["lc_weight_max"] = str(float(lc_weight_max))
    graph.meta["pg_loop_robust"] = str(pg_loop_robust)
    graph.meta["pg_loop_robust_scale"] = str(float(pg_loop_robust_scale))

    pg_before, pg_after = pose_graph.optimize(max_nfev=120)
    graph.meta["pose_graph_residual_before"] = str(float(pg_before))
    graph.meta["pose_graph_residual_after"] = str(float(pg_after))
    pose_graph_path = os.path.join(out_dir, "pose_graph.json")
    with open(pose_graph_path, "w") as f:
        json.dump(pose_graph.to_dict(optimized=False), f, indent=2)
    pose_graph_opt_path = os.path.join(out_dir, "pose_graph_optimized.json")
    with open(pose_graph_opt_path, "w") as f:
        json.dump(pose_graph.to_dict(optimized=True), f, indent=2)

    if use_est_pose:
        for nid, node in graph.nodes.items():
            x_opt, z_opt, yaw_opt = pose_graph.get_pose(int(nid), optimized=True)
            node.position[0] = float(x_opt)
            node.position[2] = float(z_opt)
            node.rotation = _quat_wxyz_from_yaw(float(yaw_opt))
            if str(nid) in node_meta:
                node_meta[str(nid)]["position"] = [float(node.position[0]), float(node.position[1]), float(node.position[2])]
                node_meta[str(nid)]["rotation"] = node.rotation
                node_meta[str(nid)]["est_position"] = [float(node.position[0]), float(node.position[1]), float(node.position[2])]
                node_meta[str(nid)]["est_yaw"] = float(yaw_opt)
        # Recompute all edge lengths from optimized estimated poses.
        for u, nbrs in graph.edges.items():
            for v in list(nbrs.keys()):
                pu = np.array(graph.nodes[int(u)].position, dtype=np.float32)
                pv = np.array(graph.nodes[int(v)].position, dtype=np.float32)
                graph.edges[u][v] = float(np.linalg.norm(pu - pv))

    num_pg_edges = len(pose_graph.edges)
    num_pg_loops = sum(1 for e in pose_graph.edges if e.edge_type == "loop")
    print(
        f"[TOPO_POSE_GRAPH] nodes={len(pose_graph.nodes)} edges={num_pg_edges} loops={num_pg_loops} optimized=1",
        flush=True,
    )
    print(
        f"[TOPO_PG_OPT] before_resid={pg_before:.6f} after_resid={pg_after:.6f} loops={num_pg_loops}",
        flush=True,
    )
    print(f"[TOPO_PG_RESIDUAL] before={pg_before:.6f} after={pg_after:.6f}", flush=True)

    graph_json = os.path.join(out_dir, "graph.json")
    graph.save_json(graph_json)
    place_graph_path = os.path.join(out_dir, "place_graph.json")
    place_graph = build_place_graph(graph=graph, node_meta=node_meta)
    with open(place_graph_path, "w") as f:
        json.dump(place_graph, f, indent=2)
    print(
        f"[TOPO_PLACE_GRAPH] places={len(place_graph.get('places', []))} edges={len(place_graph.get('edges', []))}",
        flush=True,
    )

    embeds_path = os.path.join(out_dir, "node_embeds.npy")
    save_embeddings(embeds_path, embeds)
    clip_embeds_path = os.path.join(out_dir, "node_embeds_clip.npy")
    save_embeddings(clip_embeds_path, clip_embeds)
    geo_embeds_path = os.path.join(out_dir, "node_geo.npy")
    save_embeddings(geo_embeds_path, geo_embeds)

    meta_path = os.path.join(out_dir, "node_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"meta": graph.meta, "node_meta": node_meta}, f, indent=2)

    if len(clip_embeds) > 0:
        clip_arr = np.stack(clip_embeds, axis=0).astype(np.float32)
    else:
        clip_arr = np.zeros((0, 512), dtype=np.float32)
    room_graph = build_room_graph(
        graph=graph,
        node_meta=node_meta,
        clip_embeds=clip_arr,
        room_labels=room_labels,
        room_text_embeds=room_text_embeds,
        scene_id=(scene_id if scene_id else (first_scene_id if first_scene_id else "unknown")),
    )
    room_graph_path = os.path.join(out_dir, "room_graph.json")
    with open(room_graph_path, "w") as f:
        json.dump(room_graph, f, indent=2)
    room_seg_debug_path = os.path.join(out_dir, "room_segmentation_debug.json")
    with open(room_seg_debug_path, "w") as f:
        json.dump(room_graph.get("segmentation_debug", {}), f, indent=2)
    edge_clear_debug_path = os.path.join(out_dir, "edge_clearance_debug.json")
    with open(edge_clear_debug_path, "w") as f:
        json.dump(room_graph.get("edge_clearance_debug", {}), f, indent=2)
    doorway_debug_path = os.path.join(out_dir, "doorway_debug.json")
    with open(doorway_debug_path, "w") as f:
        json.dump(room_graph.get("doorway_debug", {}), f, indent=2)
    cluster_info = room_graph.get("cluster_info", {})
    room_sizes = [
        len(room.get("node_ids", []))
        for room in room_graph.get("rooms", {}).values()
    ]
    print(
        f"[TOPO_ROOM_CLUSTER] scene={first_scene_id if first_scene_id else 'unknown'} "
        f"rooms={len(room_graph.get('rooms', {}))} sizes={room_sizes} "
        f"method={cluster_info.get('method', 'connectivity_bfs')}",
        flush=True,
    )
    place2room = room_graph.get("place2room", {})
    node_to_room = {}
    for nid in graph.nodes.keys():
        ni = int(nid)
        if str(ni) in place2room:
            node_to_room[ni] = int(place2room[str(ni)])
        else:
            if len(place2room) > 0:
                ref = min(
                    (int(k) for k in place2room.keys()),
                    key=lambda x: float(
                        np.linalg.norm(
                            np.asarray(graph.nodes[ni].position, dtype=np.float32)
                            - np.asarray(graph.nodes[int(x)].position, dtype=np.float32)
                        )
                    ),
                )
                node_to_room[ni] = int(place2room.get(str(ref), -1))
            else:
                node_to_room[ni] = -1
    node_to_room_path = os.path.join(out_dir, "node_to_room.json")
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
        f"[TOPO_NODE_TO_ROOM] scene={first_scene_id if first_scene_id else 'unknown'} "
        f"graph_nodes={len(graph.nodes)} mapped={len(node_to_room)} method=nearest_place",
        flush=True,
    )
    print(
        f"[TOPO_ROOM_GRAPH] rooms={len(room_graph.get('rooms', {}))} place_to_room={len(place2room)} "
        f"edges={len(room_graph.get('edges', []))} labels={room_graph.get('label_room_counts', {})}",
        flush=True,
    )

    gt_audit_enabled = os.environ.get("TOPO_GT_ROOM_AUDIT", "0") == "1"
    if gt_audit_enabled:
        gt_audit_dir = os.environ.get("TOPO_GT_AUDIT_OUT_ROOT", "").strip()
        if gt_audit_dir:
            os.makedirs(gt_audit_dir, exist_ok=True)
        scene_name = scene_id if scene_id else (first_scene_id if first_scene_id else "unknown_scene")
        if len(semantic_regions) == 0:
            print(
                f"[TOPO_GT_ROOM_AUDIT] scene={scene_name} gt_available=0 nodes={len(graph.nodes)} "
                f"non_unknown_nodes=0 gt_labels={{}} reason={semantic_region_reason}",
                flush=True,
            )
        else:
            gt_audit = build_gt_room_audit(
                scene_id=scene_name,
                graph=graph,
                node_meta=node_meta,
                room_graph=room_graph,
                semantic_regions=semantic_regions,
            )
            gt_counts = gt_audit.get("gt_label_counts", {})
            gt_available = int(gt_audit.get("gt_available", 0))
            non_unknown = int(gt_audit.get("non_unknown_nodes", 0))
            print(
                f"[TOPO_GT_ROOM_AUDIT] scene={scene_name} gt_available={gt_available} "
                f"nodes={gt_audit.get('total_nodes', 0)} non_unknown_nodes={non_unknown} "
                f"gt_labels={gt_counts}",
                flush=True,
            )
            print(
                f"[TOPO_GT_ROOM_CONFUSION] labels={gt_audit.get('labels', [])} "
                f"matrix={gt_audit.get('matrix', [])}",
                flush=True,
            )
            local_gt_path = os.path.join(out_dir, "gt_room_confusion.json")
            with open(local_gt_path, "w") as f:
                json.dump(gt_audit, f, indent=2)
            if gt_audit_dir:
                safe_scene = str(scene_name).replace("/", "_")
                with open(os.path.join(gt_audit_dir, f"{safe_scene}_gt_confusion.json"), "w") as f:
                    json.dump(gt_audit, f, indent=2)

    object_graph, object_nodes = build_object_graph(
        node_meta=node_meta,
        clip_embeds=clip_arr,
        room_graph=room_graph,
        object_merge_dist=object_merge_dist,
    )
    object_graph_path = os.path.join(out_dir, "object_graph.json")
    with open(object_graph_path, "w") as f:
        json.dump(object_graph, f, indent=2)
    object_nodes_path = os.path.join(out_dir, "object_nodes.json")
    with open(object_nodes_path, "w") as f:
        json.dump({"objects": object_nodes}, f, indent=2)
    for obj in object_nodes:
        print(
            f"[TOPO_OBJECT_NODE] id={obj['object_id']} label={obj['label']} support_places={obj['support_count']}",
            flush=True,
        )
    attached = sum(
        1
        for obj in object_nodes
        if int(obj.get("room_id", -1)) >= 0 and int(obj.get("support_count", 0)) > 0
    )
    print(
        f"[TOPO_OBJECT_GRAPH] objects={len(object_nodes)} attached={attached}",
        flush=True,
    )

    undirected_edges = sum(1 for u, nbrs in graph.edges.items() for v in nbrs if u < v)
    print(f"[TOPO_BUILD] nodes={len(graph.nodes)} edges={undirected_edges} out_dir={out_dir}", flush=True)
    print(
        f"[TOPO_BUILD_STATS] skipped_near_duplicate_nodes={skipped_near_duplicate_nodes} "
        f"skipped_max_nodes={skipped_max_nodes} loop_closure_edges={loop_closure_edges}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Build a TopoGraph from a simple exploration policy.")
    parser.add_argument("--config_path", type=str, default="repo/InternNav/scripts/eval/configs/vln_r2r.yaml")
    parser.add_argument("--out_dir", type=str, default="runs/topo_mvp")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--save_every_k", type=int, default=10)
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-DualVLN")
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--max_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scene_id", type=str, default=None)
    parser.add_argument("--min_node_spacing", type=float, default=0.10)
    parser.add_argument("--stuck_move_thresh", type=float, default=0.02)
    parser.add_argument("--stuck_forward_limit", type=int, default=3)
    parser.add_argument("--recovery_turn_steps", type=int, default=2)
    parser.add_argument("--vision_prompt", type=str, default="")
    parser.add_argument("--loop_closure_sim_thresh", type=float, default=0.95)
    parser.add_argument("--loop_closure_geo_thresh", type=float, default=2.0)
    parser.add_argument("--loop_closure_geo_sim_thresh", type=float, default=0.95)
    parser.add_argument("--loop_closure_min_hops", type=int, default=3)
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--start_episode_offset", type=int, default=0)
    parser.add_argument(
        "--room_labels_csv",
        type=str,
        default=os.environ.get("TOPO_ROOM_LABELS", ",".join(DEFAULT_ROOM_LABELS)),
    )
    parser.add_argument("--room_label_topk", type=int, default=3)
    parser.add_argument(
        "--object_labels_csv",
        type=str,
        default=os.environ.get("TOPO_OBJECT_LABELS", ",".join(DEFAULT_OBJECT_LABELS)),
    )
    parser.add_argument("--object_label_topk", type=int, default=int(os.environ.get("TOPO_OBJECT_LABEL_TOPK", "3")))
    parser.add_argument("--object_min_score", type=float, default=float(os.environ.get("TOPO_OBJECT_MIN_SCORE", "0.18")))
    parser.add_argument("--object_merge_dist", type=float, default=float(os.environ.get("TOPO_OBJECT_MERGE_DIST", "1.2")))
    parser.add_argument("--geo_bins_r", type=int, default=int(os.environ.get("TOPO_GEO_BINS_R", "20")))
    parser.add_argument("--geo_bins_theta", type=int, default=int(os.environ.get("TOPO_GEO_BINS_THETA", "60")))
    parser.add_argument("--geo_max_depth", type=float, default=float(os.environ.get("TOPO_GEO_MAX_DEPTH", "5.0")))
    parser.add_argument("--merge_geo_dist_thresh", type=float, default=float(os.environ.get("TOPO_MERGE_GEO_DIST_THRESH", "0.5")))
    parser.add_argument("--merge_vision_sim_thresh", type=float, default=float(os.environ.get("TOPO_MERGE_VISION_SIM_THRESH", "0.98")))
    parser.add_argument("--merge_geo_sim_thresh", type=float, default=float(os.environ.get("TOPO_MERGE_GEO_SIM_THRESH", "0.98")))
    parser.add_argument("--max_nodes", type=int, default=int(os.environ.get("TOPO_MAX_NODES", "30")))
    parser.add_argument("--use_est_pose", type=int, default=int(os.environ.get("TOPO_USE_EST_POSE", "0")))
    parser.add_argument("--odom_noise_trans", type=float, default=float(os.environ.get("TOPO_ODOM_NOISE_TRANS", "0.02")))
    parser.add_argument("--odom_noise_yaw_deg", type=float, default=float(os.environ.get("TOPO_ODOM_NOISE_YAW", "1.0")))
    parser.add_argument("--lc_rel_trans_err_max", type=float, default=float(os.environ.get("TOPO_LC_REL_TRANS_ERR_MAX", "0.85")))
    parser.add_argument("--lc_rel_yaw_err_deg_max", type=float, default=float(os.environ.get("TOPO_LC_REL_YAW_ERR_DEG_MAX", "35.0")))
    parser.add_argument("--lc_weight_min", type=float, default=float(os.environ.get("TOPO_LC_WEIGHT_MIN", "0.8")))
    parser.add_argument("--lc_weight_max", type=float, default=float(os.environ.get("TOPO_LC_WEIGHT_MAX", "2.5")))
    parser.add_argument("--pg_loop_robust", type=str, default=os.environ.get("TOPO_PG_LOOP_ROBUST", "huber"))
    parser.add_argument("--pg_loop_robust_scale", type=float, default=float(os.environ.get("TOPO_PG_LOOP_ROBUST_SCALE", "0.75")))
    args = parser.parse_args()

    build_topograph(
        config_path=args.config_path,
        out_dir=args.out_dir,
        steps=args.steps,
        save_every_k=args.save_every_k,
        model_path=args.model_path,
        pooling=args.pooling,
        max_episodes=args.max_episodes,
        seed=args.seed,
        scene_id=args.scene_id,
        min_node_spacing=args.min_node_spacing,
        stuck_move_thresh=args.stuck_move_thresh,
        stuck_forward_limit=args.stuck_forward_limit,
        recovery_turn_steps=args.recovery_turn_steps,
        vision_prompt=args.vision_prompt,
        loop_closure_sim_thresh=args.loop_closure_sim_thresh,
        loop_closure_geo_thresh=args.loop_closure_geo_thresh,
        loop_closure_geo_sim_thresh=args.loop_closure_geo_sim_thresh,
        loop_closure_min_hops=args.loop_closure_min_hops,
        clip_model_name=args.clip_model_name,
        start_episode_offset=args.start_episode_offset,
        room_labels_csv=args.room_labels_csv,
        room_label_topk=args.room_label_topk,
        object_labels_csv=args.object_labels_csv,
        object_label_topk=args.object_label_topk,
        object_min_score=args.object_min_score,
        object_merge_dist=args.object_merge_dist,
        geo_bins_r=args.geo_bins_r,
        geo_bins_theta=args.geo_bins_theta,
        geo_max_depth=args.geo_max_depth,
        merge_geo_dist_thresh=args.merge_geo_dist_thresh,
        merge_vision_sim_thresh=args.merge_vision_sim_thresh,
        merge_geo_sim_thresh=args.merge_geo_sim_thresh,
        max_nodes=args.max_nodes,
        use_est_pose=bool(args.use_est_pose),
        odom_noise_trans=args.odom_noise_trans,
        odom_noise_yaw_deg=args.odom_noise_yaw_deg,
        lc_rel_trans_err_max=args.lc_rel_trans_err_max,
        lc_rel_yaw_err_deg_max=args.lc_rel_yaw_err_deg_max,
        lc_weight_min=args.lc_weight_min,
        lc_weight_max=args.lc_weight_max,
        pg_loop_robust=args.pg_loop_robust,
        pg_loop_robust_scale=args.pg_loop_robust_scale,
    )


if __name__ == "__main__":
    main()
