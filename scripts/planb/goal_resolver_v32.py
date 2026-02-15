#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_yaml_or_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("goal_catalog_root_not_mapping")
    return data


def _as_float(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)


def _norm_goal(goal: Dict[str, Any]) -> Dict[str, Any]:
    g = dict(goal)
    gid = str(g.get("id", "")).strip()
    gtype = str(g.get("type", "pose")).strip().lower()
    if not gid:
        raise ValueError("goal_id_missing")

    meta = g.get("meta", {}) if isinstance(g.get("meta", {}), dict) else {}
    bucket = str(meta.get("bucket", "unlabeled"))
    aliases = list(meta.get("aliases", [])) if isinstance(meta.get("aliases", []), list) else []

    out: Dict[str, Any] = {
        "id": gid,
        "type": gtype,
        "meta": {
            "bucket": bucket,
            "aliases": aliases,
            "note": str(meta.get("note", "")),
        },
        "raw": g,
    }

    if gtype == "pose":
        val = g.get("value", {}) if isinstance(g.get("value", {}), dict) else {}
        succ = g.get("success", {}) if isinstance(g.get("success", {}), dict) else {}
        out["value"] = {
            "x": _as_float(val, "x", 0.0),
            "y": _as_float(val, "y", 0.0),
            "z": _as_float(val, "z", 0.0),
            "yaw_deg": _as_float(val, "yaw_deg", _as_float(val, "yaw", 0.0)),
        }
        out["success"] = {
            "dist_m": float(succ.get("dist_m", 0.5)),
            "yaw_deg": float(succ.get("yaw_deg", 15.0)),
        }
        return out

    if gtype == "room":
        val = g.get("value", {}) if isinstance(g.get("value", {}), dict) else {}
        succ = g.get("success", {}) if isinstance(g.get("success", {}), dict) else {}
        region = val.get("region", {}) if isinstance(val.get("region", {}), dict) else {}
        rtype = str(region.get("type", "aabb")).lower()
        rmin = region.get("min", {}) if isinstance(region.get("min", {}), dict) else {}
        rmax = region.get("max", {}) if isinstance(region.get("max", {}), dict) else {}
        anchors: List[Dict[str, float]] = []
        if isinstance(val.get("anchor_poses", []), list):
            for a in val.get("anchor_poses", []):
                if not isinstance(a, dict):
                    continue
                anchors.append(
                    {
                        "x": _as_float(a, "x", 0.0),
                        "y": _as_float(a, "y", 0.0),
                        "z": _as_float(a, "z", 0.0),
                        "yaw_deg": _as_float(a, "yaw_deg", _as_float(a, "yaw", 0.0)),
                    }
                )
        out["value"] = {
            "room_id": str(val.get("room_id", gid)),
            "region": {
                "type": rtype,
                "min": {"x": _as_float(rmin, "x", 0.0), "z": _as_float(rmin, "z", 0.0)},
                "max": {"x": _as_float(rmax, "x", 0.0), "z": _as_float(rmax, "z", 0.0)},
            },
            "anchor_poses": anchors,
        }
        out["success"] = {
            "inside": bool(succ.get("inside", True)),
            "settle_steps": int(succ.get("settle_steps", 5)),
        }
        return out

    if gtype == "object":
        val = g.get("value", {}) if isinstance(g.get("value", {}), dict) else {}
        succ = g.get("success", {}) if isinstance(g.get("success", {}), dict) else {}
        query = val.get("query", {}) if isinstance(val.get("query", {}), dict) else {}
        out["value"] = {
            "query": {
                "name_contains": str(query.get("name_contains", "")),
                "class_equals": str(query.get("class_equals", "")),
                "asset_path_contains": str(query.get("asset_path_contains", "")),
            },
            "require_visible": bool(val.get("require_visible", True)),
            "max_range_m": float(val.get("max_range_m", succ.get("range_m", 1.5))),
        }
        out["success"] = {
            "visible": bool(succ.get("visible", True)),
            "range_m": float(succ.get("range_m", val.get("max_range_m", 1.5))),
        }
        return out

    raise ValueError(f"unsupported_goal_type:{gtype}")


def normalize_catalog(catalog: Dict[str, Any]) -> Dict[str, Any]:
    goals_in = catalog.get("goals", []) if isinstance(catalog.get("goals", []), list) else []
    goals_out: List[Dict[str, Any]] = []
    seen = set()
    for g in goals_in:
        if not isinstance(g, dict):
            continue
        ng = _norm_goal(g)
        if ng["id"] in seen:
            raise ValueError(f"duplicate_goal_id:{ng['id']}")
        seen.add(ng["id"])
        goals_out.append(ng)
    return {
        "version": int(catalog.get("version", 32)),
        "scene": str(catalog.get("scene", "")),
        "goals": goals_out,
    }


def capability_object_semantics() -> Tuple[bool, str]:
    # Default deterministic behavior: disabled unless explicitly enabled.
    import os

    if int(os.environ.get("V32_OBJECT_SEMANTICS", "0")) == 1:
        return True, "env_enabled"
    return False, "object_semantics_not_available"


def representative_pose_for_room(goal: Dict[str, Any]) -> Dict[str, float]:
    val = goal.get("value", {}) if isinstance(goal.get("value", {}), dict) else {}
    anchors = val.get("anchor_poses", []) if isinstance(val.get("anchor_poses", []), list) else []
    if len(anchors) > 0 and isinstance(anchors[0], dict):
        a = anchors[0]
        return {
            "x": float(a.get("x", 0.0)),
            "y": float(a.get("y", 0.0)),
            "z": float(a.get("z", 0.0)),
            "yaw_deg": float(a.get("yaw_deg", 0.0)),
        }
    region = val.get("region", {}) if isinstance(val.get("region", {}), dict) else {}
    rmin = region.get("min", {}) if isinstance(region.get("min", {}), dict) else {}
    rmax = region.get("max", {}) if isinstance(region.get("max", {}), dict) else {}
    cx = 0.5 * (float(rmin.get("x", 0.0)) + float(rmax.get("x", 0.0)))
    cz = 0.5 * (float(rmin.get("z", 0.0)) + float(rmax.get("z", 0.0)))
    return {"x": cx, "y": 0.0, "z": cz, "yaw_deg": 0.0}


def _yaw_err_deg(a: float, b: float) -> float:
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(d)


def is_pose_success(state: Dict[str, float], goal: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
    v = goal.get("value", {})
    s = goal.get("success", {})
    x = float(state.get("x", 0.0))
    z = float(state.get("z", 0.0))
    yaw = float(state.get("yaw_deg", 0.0))
    gx, gz, gyaw = float(v.get("x", 0.0)), float(v.get("z", 0.0)), float(v.get("yaw_deg", 0.0))
    dist = float(math.hypot(x - gx, z - gz))
    yaw_err = float(_yaw_err_deg(yaw, gyaw))
    ok = dist <= float(s.get("dist_m", 0.5)) and yaw_err <= float(s.get("yaw_deg", 15.0))
    return bool(ok), {"dist_m": dist, "yaw_err_deg": yaw_err}


def _inside_aabb(x: float, z: float, rmin: Dict[str, float], rmax: Dict[str, float]) -> bool:
    lo_x, hi_x = sorted([float(rmin.get("x", 0.0)), float(rmax.get("x", 0.0))])
    lo_z, hi_z = sorted([float(rmin.get("z", 0.0)), float(rmax.get("z", 0.0))])
    return (lo_x <= x <= hi_x) and (lo_z <= z <= hi_z)


def is_room_success_from_trace(trace_rows: List[Dict[str, Any]], goal: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    v = goal.get("value", {}) if isinstance(goal.get("value", {}), dict) else {}
    region = v.get("region", {}) if isinstance(v.get("region", {}), dict) else {}
    rmin = region.get("min", {}) if isinstance(region.get("min", {}), dict) else {}
    rmax = region.get("max", {}) if isinstance(region.get("max", {}), dict) else {}
    settle = int(goal.get("success", {}).get("settle_steps", 5))

    consec = 0
    best = 0
    last_inside = False
    for r in trace_rows:
        p = r.get("pose_used", {}) if isinstance(r.get("pose_used", {}), dict) else {}
        x = float(p.get("x", 0.0))
        z = float(p.get("z", 0.0))
        inside = _inside_aabb(x, z, rmin, rmax)
        last_inside = inside
        if inside:
            consec += 1
            best = max(best, consec)
        else:
            consec = 0
    ok = bool(best >= max(1, settle))
    return ok, {"inside_best_streak": int(best), "settle_steps": int(settle), "last_inside": int(last_inside)}


def object_query_summary(goal: Dict[str, Any]) -> str:
    q = goal.get("value", {}).get("query", {}) if isinstance(goal.get("value", {}).get("query", {}), dict) else {}
    for k in ["name_contains", "class_equals", "asset_path_contains"]:
        v = str(q.get(k, "")).strip()
        if v:
            return f"{k}:{v}"
    return "query:unspecified"
