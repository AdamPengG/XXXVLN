import collections
import os
from typing import Dict, List, Tuple


OBJECT_SYNONYMS = {
    "fridge": "refrigerator",
    "refridgerator": "refrigerator",
    "freezer": "refrigerator",
    "television": "tv",
    "couch": "sofa",
    "settee": "sofa",
    "dining_table": "table",
    "diningtable": "table",
    "coffee-table": "coffee_table",
    "bookshelf": "bookcase",
    "book_shelf": "bookcase",
    "book shelf": "bookcase",
    "wardrobe_cabinet": "wardrobe",
    "closet": "wardrobe",
    "shower_curtain": "shower",
    "bathtub_sink": "sink",
    "bathroom_sink": "sink",
    "office_chair": "chair",
    "computer_monitor": "monitor",
}


STRONG_OBJECTS = {
    "bathroom": {"toilet", "sink", "bathtub", "shower", "mirror"},
    "kitchen": {"refrigerator", "stove", "oven", "microwave", "dishwasher"},
    "bedroom": {"bed", "pillow", "wardrobe", "dresser", "nightstand"},
    "living_room": {"tv"},
    "dining_room": {"dining_table"},
    "office": {"desk", "monitor", "laptop"},
}

MEDIUM_OBJECTS = {
    "bathroom": {"toilet_paper", "towel", "mirror"},
    "kitchen": {"countertop", "kettle", "sink", "table", "chair"},
    "bedroom": {"blanket", "bookcase", "lamp", "chair"},
    "living_room": {"sofa", "coffee_table", "table", "chair", "bookcase", "plant"},
    "dining_room": {"table", "chair", "cabinet"},
    "office": {"desk", "monitor", "laptop", "computer", "bookcase", "chair"},
    "hallway": {"stairs", "stair", "corridor"},
}

GENERIC_OBJECTS = {
    "door",
    "cabinet",
    "shelf",
    "counter",
    "table",
    "chair",
    "plant",
    "lamp",
}


def normalize_object_label(label: str) -> str:
    s = str(label or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    if s in OBJECT_SYNONYMS:
        return str(OBJECT_SYNONYMS[s])
    return s


def _obj_matches(label: str, keyword: str) -> bool:
    if not label or not keyword:
        return False
    if label == keyword:
        return True
    if keyword in label:
        return True
    if label in keyword:
        return True
    return False


def _sorted_topk(score_map: Dict[str, float], k: int = 4) -> List[Tuple[str, float]]:
    rows = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    return [(str(lbl), float(sc)) for lbl, sc in rows[: max(1, int(k))]]


def _score_rule_hits(norm_hist: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, list], bool]:
    strong_scores = collections.defaultdict(float)
    medium_scores = collections.defaultdict(float)
    rule_hits = collections.defaultdict(list)
    generic_only = True

    for obj_label, obj_count in norm_hist.items():
        matched = False
        for room_label, strong_set in STRONG_OBJECTS.items():
            for kw in strong_set:
                if _obj_matches(obj_label, kw):
                    strong_scores[room_label] += float(obj_count)
                    rule_hits[room_label].append((obj_label, float(obj_count), "strong"))
                    matched = True
                    generic_only = False
                    break
            if matched:
                break
        if matched:
            continue
        for room_label, medium_set in MEDIUM_OBJECTS.items():
            for kw in medium_set:
                if _obj_matches(obj_label, kw):
                    medium_scores[room_label] += 0.55 * float(obj_count)
                    rule_hits[room_label].append((obj_label, float(obj_count), "medium"))
                    matched = True
                    generic_only = False
                    break
            if matched:
                break
        if matched:
            continue
        # Generic objects are weak evidence; route them to plausible spaces only.
        if obj_label in GENERIC_OBJECTS:
            base = float(obj_count)
            medium_scores["hallway"] += 0.02 * base
            rule_hits["hallway"].append((obj_label, float(obj_count), "generic"))
            if obj_label in {"counter", "cabinet"}:
                medium_scores["kitchen"] += 0.08 * base
                rule_hits["kitchen"].append((obj_label, float(obj_count), "generic"))
            if obj_label in {"table", "chair"}:
                medium_scores["dining_room"] += 0.08 * base
                medium_scores["kitchen"] += 0.03 * base
                rule_hits["dining_room"].append((obj_label, float(obj_count), "generic"))
                rule_hits["kitchen"].append((obj_label, float(obj_count), "generic"))
            if obj_label in {"shelf"}:
                medium_scores["office"] += 0.06 * base
                rule_hits["office"].append((obj_label, float(obj_count), "generic"))
            if obj_label in {"door"}:
                medium_scores["hallway"] += 0.05 * base
                rule_hits["hallway"].append((obj_label, float(obj_count), "generic_door"))
        else:
            # Unknown object token: tiny neutral evidence.
            for room_label in ("kitchen", "living_room", "bedroom", "bathroom", "office"):
                medium_scores[room_label] += 0.02 * float(obj_count)
    return dict(strong_scores), dict(medium_scores), dict(rule_hits), bool(generic_only)


def infer_room_label_from_objects(
    obj_hist: Dict,
    clip_label: str = "unknown_clip",
    clip_score: float = 0.0,
    clip_topk: List[Tuple[str, float]] = None,
    clip_margin: float = 0.0,
) -> Dict:
    norm_hist = {}
    total = 0.0
    for k, v in dict(obj_hist or {}).items():
        nk = normalize_object_label(k)
        if not nk:
            continue
        vv = float(v)
        if vv <= 0.0:
            continue
        norm_hist[nk] = norm_hist.get(nk, 0.0) + vv
        total += vv

    if total <= 1e-9:
        return {
            "label": "unknown",
            "score": 0.0,
            "topk": [],
            "rule_hits": {},
            "used_rules": False,
            "total_objects": 0.0,
        }

    strong_scores, medium_scores, rule_hits, generic_only = _score_rule_hits(norm_hist)
    room_scores = collections.defaultdict(float)
    for lbl, sc in strong_scores.items():
        room_scores[str(lbl)] += float(sc) * 2.2
    for lbl, sc in medium_scores.items():
        room_scores[str(lbl)] += float(sc)

    if len(room_scores) == 0:
        return {
            "label": "unknown",
            "score": 0.0,
            "topk": [],
            "rule_hits": {},
            "strong_hit": "",
            "generic_only": bool(generic_only),
            "used_rules": False,
            "total_objects": float(total),
        }

    strong_top = _sorted_topk(strong_scores, k=2)
    strong_hit = str(strong_top[0][0]) if len(strong_top) > 0 else ""
    strong_val = float(strong_top[0][1]) if len(strong_top) > 0 else 0.0
    strong_second = float(strong_top[1][1]) if len(strong_top) > 1 else 0.0
    strong_thresh = float(max(0.05, float(os.environ.get("TOPO_ROOM_STRONG_THRESH", "0.22"))))
    strong_ratio_thresh = float(max(1.0, float(os.environ.get("TOPO_ROOM_STRONG_RATIO_THRESH", "1.20"))))
    has_strong = strong_val >= strong_thresh and (
        float(strong_val / max(1e-6, strong_second)) >= strong_ratio_thresh
    )

    topk = _sorted_topk(room_scores, k=4)
    top1_label = str(topk[0][0]) if len(topk) > 0 else "unknown"
    top1_score = float(topk[0][1]) if len(topk) > 0 else 0.0
    top2_score = float(topk[1][1]) if len(topk) > 1 else 0.0
    if top1_label == "living_room":
        kitchen_score = float(room_scores.get("kitchen", 0.0))
        bedroom_score = float(room_scores.get("bedroom", 0.0))
        alt_score = max(kitchen_score, bedroom_score)
        if alt_score >= 0.90 * max(1e-6, top1_score):
            if kitchen_score >= bedroom_score:
                top1_label = "kitchen"
                top1_score = kitchen_score
            else:
                top1_label = "bedroom"
                top1_score = bedroom_score
    total_score = float(sum(room_scores.values()))
    sep = float(max(0.0, top1_score - top2_score))
    top1_share = float(top1_score / max(1e-6, total_score))
    object_share_thresh = float(max(0.20, float(os.environ.get("TOPO_ROOM_OBJ_SHARE_THRESH", "0.34"))))
    object_sep_thresh = float(max(0.01, float(os.environ.get("TOPO_ROOM_OBJ_SEP_THRESH", "0.06"))))
    object_top1_min = float(max(0.01, float(os.environ.get("TOPO_ROOM_OBJ_TOP1_MIN", "0.24"))))
    object_confident = (top1_share >= object_share_thresh and sep >= object_sep_thresh) or (top1_score >= object_top1_min)

    if has_strong or object_confident:
        best_label = str(strong_hit if has_strong and strong_hit else top1_label)
        conf = max(0.0, min(1.0, 0.42 + 0.38 * top1_share + 0.20 * min(1.0, sep)))
        label_source = "objects"
    else:
        best_label = "unknown"
        conf = 0.0
        label_source = "unknown"
        if top1_label not in ("unknown", "unknown_clip"):
            # Keep object-derived fallback to avoid collapsing whole rooms to unknown.
            best_label = str(top1_label)
            conf = float(max(0.0, min(1.0, 0.18 + 0.25 * top1_share + 0.15 * min(1.0, sep))))
            label_source = "objects"
        else:
            clip_lbl = str(clip_label or "unknown_clip")
            if clip_lbl not in ("unknown", "unknown_clip") and float(clip_score) > 0.0:
                best_label = clip_lbl
                conf = float(max(0.0, min(1.0, clip_score)))
                label_source = "clip_fallback"

    return {
        "label": str(best_label),
        "score": float(conf),
        "topk": [(str(lbl), float(sc)) for lbl, sc in topk],
        "rule_hits": {
            str(lbl): [[str(o), float(c), str(w)] for o, c, w in hits]
            for lbl, hits in rule_hits.items()
        },
        "strong_hit": str(strong_hit),
        "generic_only": bool(generic_only),
        "label_source": str(label_source),
        "used_rules": bool(has_strong),
        "clip_margin": float(clip_margin),
        "clip_label": str(clip_label),
        "clip_score": float(clip_score),
        "clip_topk": [(str(lbl), float(sc)) for lbl, sc in (clip_topk or [])],
        "total_objects": float(total),
    }
