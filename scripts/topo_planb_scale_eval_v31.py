#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from planb.goal_resolver_v32 import (
    capability_object_semantics,
    is_pose_success,
    is_room_success_from_trace,
    normalize_catalog,
    object_query_summary,
    representative_pose_for_room,
)


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _dump_yaml_or_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _sanitize_id(x: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", str(x)).strip("_")
    return s or "run"


def _mk_synth_build(out_dir: Path, scene: str) -> Dict[str, Path]:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    rows, cols = 4, 4
    spacing = 2.0
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    embeds: List[np.ndarray] = []
    clip_embeds: List[np.ndarray] = []
    node_meta: Dict[str, Dict[str, Any]] = {}
    node_to_room: Dict[str, int] = {}
    room_palette = {0: (180, 70, 70), 1: (70, 180, 90), 2: (70, 90, 180), 3: (180, 160, 70)}
    room_labels = ["hallway", "office", "meeting_room", "kitchen"]

    def mk_embed(x: float, z: float, rid: int, dim: int = 64) -> np.ndarray:
        vec = np.zeros((dim,), dtype=np.float32)
        for i in range(dim):
            vec[i] = math.sin(0.17 * i + 0.31 * x) + math.cos(0.13 * i + 0.27 * z) + 0.15 * rid
        n = float(np.linalg.norm(vec) + 1e-6)
        return (vec / n).astype(np.float32)

    nid = 0
    for r in range(rows):
        for c in range(cols):
            x = (c - (cols - 1) / 2.0) * spacing
            z = (r - (rows - 1) / 2.0) * spacing
            room_id = 0 if r < 2 and c < 2 else 1 if r < 2 else 2 if c < 2 else 3
            rgb = np.zeros((160, 240, 3), dtype=np.uint8)
            rgb[..., 0] = int(room_palette[room_id][0] + 10 * c)
            rgb[..., 1] = int(room_palette[room_id][1] + 10 * r)
            rgb[..., 2] = int(room_palette[room_id][2])
            frame = out_dir / "frames" / f"node_{nid:04d}.png"
            Image.fromarray(rgb).save(frame)

            nodes.append(
                {
                    "node_id": nid,
                    "step_idx": nid,
                    "position": [float(x), 0.0, float(z)],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "embed_idx": nid,
                    "timestamp": float(1000.0 + nid),
                    "rgb_path": str(frame),
                }
            )
            emb = mk_embed(x, z, room_id, dim=64)
            embeds.append(emb)
            clip_embeds.append(emb)
            node_meta[str(nid)] = {
                "embed_idx": nid,
                "clip_embed_idx": nid,
                "geo_idx": nid,
                "position": [float(x), 0.0, float(z)],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "room_id": int(room_id),
                "room_label": room_labels[room_id],
                "rgb_path": str(frame),
            }
            node_to_room[str(nid)] = int(room_id)
            nid += 1

    def node_id(rr: int, cc: int) -> int:
        return rr * cols + cc

    for r in range(rows):
        for c in range(cols):
            u = node_id(r, c)
            if c + 1 < cols:
                edges.append({"u": u, "v": node_id(r, c + 1), "w": spacing, "edge_type": "temporal"})
            if r + 1 < rows:
                edges.append({"u": u, "v": node_id(r + 1, c), "w": spacing, "edge_type": "temporal"})

    graph = {"meta": {"scene_id": scene, "backend": "isaac_v31"}, "nodes": nodes, "edges": edges}
    (out_dir / "graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    np.save(out_dir / "node_embeds.npy", np.stack(embeds, axis=0).astype(np.float32))
    np.save(out_dir / "node_embeds_clip.npy", np.stack(clip_embeds, axis=0).astype(np.float32))
    (out_dir / "node_meta.json").write_text(
        json.dumps({"meta": {"scene_id": scene}, "node_meta": node_meta}, indent=2), encoding="utf-8"
    )
    (out_dir / "node_to_room.json").write_text(
        json.dumps({"scene_id": scene, "node_to_room": node_to_room}, indent=2), encoding="utf-8"
    )
    print(f"[TOPO_BUILD] nodes={len(nodes)} edges={len(edges)} out_dir={out_dir}", flush=True)

    return {
        "graph_json": out_dir / "graph.json",
        "embeds_npy": out_dir / "node_embeds.npy",
        "clip_embeds_npy": out_dir / "node_embeds_clip.npy",
        "node_meta": out_dir / "node_meta.json",
    }


def _build_start_scene_cfg(
    base_cfg: Dict[str, Any],
    scene_id: str,
    start: Dict[str, Any],
    out_cfg_path: Path,
) -> None:
    data = dict(base_cfg)
    scenes = list(data.get("scenes", [])) if isinstance(data.get("scenes", []), list) else []
    out_scenes: List[Dict[str, Any]] = []
    found = False
    pose = start.get("pose", {}) if isinstance(start.get("pose", {}), dict) else {}
    sx = float(pose.get("x", 0.0))
    sz = float(pose.get("z", 0.0))
    sy = float(pose.get("y", 0.0))
    syaw = float(pose.get("yaw_deg", pose.get("yaw", 0.0)))

    for s in scenes:
        if not isinstance(s, dict):
            continue
        ss = dict(s)
        if str(ss.get("scene_id", "")) == str(scene_id):
            ss["starts"] = [{"x": sx, "y": sy, "z": sz, "yaw_deg": syaw}]
            found = True
        out_scenes.append(ss)
    if not found:
        out_scenes.append(
            {
                "scene_id": str(scene_id),
                "starts": [{"x": sx, "y": sy, "z": sz, "yaw_deg": syaw}],
            }
        )
    data["scenes"] = out_scenes
    _dump_yaml_or_json(out_cfg_path, data)


def _load_trace_stats(debug_run_dir: Path) -> Dict[str, Any]:
    trace_rows = _load_trace_rows(debug_run_dir / "trace.jsonl")
    if not trace_rows:
        return {
            "collision_count": 0,
            "stuck_count": 0,
            "steps": 0,
            "stationary_ratio": 0.0,
            "last_pose": None,
        }
    rows = trace_rows
    collision_count = sum(int(r.get("collision", 0)) for r in rows)
    stuck_count = 0
    prev: Optional[Tuple[float, float]] = None
    last_pose: Optional[Dict[str, float]] = None
    for r in rows:
        p = r.get("pose_used", {}) if isinstance(r.get("pose_used", {}), dict) else {}
        cur = (float(p.get("x", 0.0)), float(p.get("z", 0.0)))
        last_pose = {
            "x": float(p.get("x", 0.0)),
            "y": float(p.get("y", 0.0)),
            "z": float(p.get("z", 0.0)),
            "yaw": float(p.get("yaw", 0.0)),
            "yaw_deg": float(math.degrees(float(p.get("yaw", 0.0)))),
        }
        if prev is not None:
            d = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            if d < 0.01:
                stuck_count += 1
        prev = cur
    stationary_ratio = float(stuck_count) / float(max(1, len(rows) - 1))
    return {
        "collision_count": int(collision_count),
        "stuck_count": int(stuck_count),
        "steps": int(len(rows)),
        "stationary_ratio": float(stationary_ratio),
        "last_pose": last_pose,
    }


def _load_trace_rows(trace_path: Path) -> List[Dict[str, Any]]:
    if not trace_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows


def _parse_v33a_log(log_path: Path) -> Dict[str, int]:
    out = {"stuck_triggers": 0, "recoveries": 0, "giveup": 0}
    if not log_path.exists():
        return out
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "[V33A_STUCK]" in line and "trigger=1" in line:
            out["stuck_triggers"] += 1
        if "[V33A_RECOVERY]" in line and "start" in line:
            out["recoveries"] += 1
        if "[V33A_RECOVERY]" in line and "giveup" in line:
            out["giveup"] += 1
    return out


def _parse_v33b_log(log_path: Path) -> Dict[str, Any]:
    out = {
        "blocked": 0,
        "overrides": 0,
        "blocked_no_override": 0,
        "doorway": 0,
        "caps_depth0": 0,
    }
    reason_counts: Dict[str, int] = {}
    if not log_path.exists():
        return out
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "[V33B_LOCAL_AVOID]" in line:
            if "blocked=1" in line:
                out["blocked"] += 1
            if "override=1" in line:
                out["overrides"] += 1
            elif "blocked=1" in line and "override=0" in line:
                out["blocked_no_override"] += 1
                reason = "unknown"
                try:
                    reason = line.split("reason=", 1)[1].split()[0].strip()
                except Exception:
                    reason = "unknown"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        if "[V33B_DOORWAY]" in line and "trigger=1" in line:
            out["doorway"] += 1
        if "[V33B_CAPS]" in line and "depth=0" in line:
            out["caps_depth0"] += 1
    out["blocked_no_override_reasons"] = reason_counts
    return out


def _classify_fail(
    result: Optional[Dict[str, Any]],
    timed_out: bool,
    exit_code: int,
    trace_stats: Dict[str, Any],
) -> Tuple[str, str]:
    if timed_out:
        return "timeout", "subprocess_timeout"
    if result is None:
        return ("backend_crash", f"exit_code_{exit_code}") if exit_code != 0 else ("unknown", "missing_result_json")

    success = bool(result.get("success", False))
    if success:
        return "success", "success"

    fail_reason = str(result.get("fail_reason", "") or "")
    term = str(result.get("terminated_by", "") or "")
    steps = int(result.get("steps_used", 0) or 0)
    max_steps = int(result.get("max_steps", 0) or 0)

    if "stuck" in fail_reason.lower():
        return "stuck", fail_reason
    if "backend_caps" in fail_reason.lower():
        return "backend_caps", fail_reason
    if "room_settle_not_met" in fail_reason:
        return "max_steps", fail_reason
    if "invalid_goal" in fail_reason:
        return "invalid_goal", fail_reason
    if "max_steps" in fail_reason or term == "max_steps" or (max_steps > 0 and steps >= max_steps):
        return "max_steps", fail_reason or term or "max_steps"
    if int(trace_stats.get("collision_count", 0)) > 0:
        return "collision", fail_reason or term or "collision_detected"
    if float(trace_stats.get("stationary_ratio", 0.0)) >= 0.35 or int(trace_stats.get("stuck_count", 0)) >= 10:
        return "stuck", fail_reason or term or "stationary_pose"
    if exit_code != 0:
        return "backend_crash", fail_reason or f"exit_code_{exit_code}"
    return "unknown", fail_reason or term or "unknown"


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "run_id",
        "scene_id",
        "start_id",
        "start_bucket",
        "goal_id",
        "goal_type",
        "goal_bucket",
        "compat_mode",
        "room_id",
        "object_query",
        "success",
        "fail_type",
        "reason",
        "steps_used",
        "final_dist_to_goal_node",
        "exit_code",
        "timed_out",
        "v33a_stuck_triggers",
        "v33a_recovery_count",
        "v33a_recovery_giveup",
        "v33b_blocked_count",
        "v33b_override_count",
        "v33b_doorway_trigger_count",
        "v33b_caps_depth0",
        "log_path",
        "debug_index",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in keys})


def _write_index(path: Path, summary: Dict[str, Any], failure_cards_rel: str) -> None:
    html = f"""<!doctype html>
<html>
<head><meta charset=\"utf-8\"/><title>V31 PlanB Scale Eval</title>
<style>body{{font-family:Arial,sans-serif;background:#f7f8fa;margin:0}} .top{{background:#111827;color:#fff;padding:12px 16px}} .wrap{{padding:12px}} pre{{background:#fff;border:1px solid #ddd;padding:12px;overflow:auto}}</style>
</head>
<body>
<div class=\"top\">V31 PlanB Scale Eval</div>
<div class=\"wrap\">
  <p><a href=\"{failure_cards_rel}\">Open failure cards</a></p>
  <pre>{json.dumps(summary, indent=2)}</pre>
</div>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _anchor_tag_from_version(version: int) -> str:
    return "V32_SCALE_EVAL" if int(version) >= 32 else "V31_SCALE_EVAL"


def _as_pose_query_from_goal(goal: Dict[str, Any]) -> Tuple[Dict[str, Any], int, str, str]:
    goal_type = str(goal.get("type", "pose")).strip().lower()
    goal_value = goal.get("value", {}) if isinstance(goal.get("value", {}), dict) else {}
    compat_mode = 0
    room_id = ""
    object_q = ""
    if goal_type == "pose":
        query = {
            "type": "pose",
            "value": {
                "x": float(goal_value.get("x", 0.0)),
                "y": float(goal_value.get("y", 0.0)),
                "z": float(goal_value.get("z", 0.0)),
                "yaw_deg": float(goal_value.get("yaw_deg", 0.0)),
            },
        }
        return query, compat_mode, room_id, object_q

    if goal_type == "room":
        compat_mode = 1
        room_id = str(goal_value.get("room_id", ""))
        pose = representative_pose_for_room(goal)
        query = {
            "type": "pose",
            "value": {
                "x": float(pose.get("x", 0.0)),
                "y": float(pose.get("y", 0.0)),
                "z": float(pose.get("z", 0.0)),
                "yaw_deg": float(pose.get("yaw_deg", 0.0)),
            },
        }
        return query, compat_mode, room_id, object_q

    if goal_type == "object":
        compat_mode = 1
        object_q = object_query_summary(goal)
        query = {
            "type": "text",
            "value": object_q,
        }
        return query, compat_mode, room_id, object_q

    raise ValueError(f"unsupported_goal_type:{goal_type}")


def main() -> int:
    ap = argparse.ArgumentParser(description="v31 Plan-B scale evaluator")
    ap.add_argument("--config", default="configs/isaac_planb_eval_v31.yaml")
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()

    root = Path("/home/peng/DualVLN")
    cfg_path = root / args.config
    cfg = _load_yaml_or_json(cfg_path)

    scene_id = str(cfg.get("scene", "office_localized"))
    scenes_cfg_path = root / str(cfg.get("scenes_config", "configs/isaac_scenes_v29.yaml"))
    goal_catalog_path = root / str(cfg.get("goal_catalog", "configs/isaac_goal_catalog_v30.yaml"))

    selection = cfg.get("selection", {}) if isinstance(cfg.get("selection"), dict) else {}
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), dict) else {}

    max_goals = int(selection.get("max_goals", 50))
    max_starts = int(selection.get("max_starts", 3))
    include_buckets = set(str(x) for x in selection.get("include_buckets", []))
    if args.small:
        max_goals = min(max_goals, 5)
        max_starts = min(max_starts, 1)

    out_root = root / str(out_cfg.get("root", "runs/topo_mvp/v31_planb_scale_eval"))
    logs_dir = out_root / "logs"
    builds_dir = out_root / "builds"
    runtime_dir = out_root / "runtime"
    debug_root = out_root / "debug_runs"
    for p in [logs_dir, builds_dir, runtime_dir, debug_root]:
        p.mkdir(parents=True, exist_ok=True)

    master_log = logs_dir / "topo_planb_scale_eval_v31.log"
    master_log.write_text("", encoding="utf-8")

    starts_all = cfg.get("starts", []) if isinstance(cfg.get("starts"), list) else []
    starts: List[Dict[str, Any]] = []
    for s in starts_all[:max_starts]:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id", f"start_{len(starts)+1:03d}"))
        pose = s.get("pose", {}) if isinstance(s.get("pose", {}), dict) else {}
        starts.append(
            {
                "id": _sanitize_id(sid),
                "bucket": str(s.get("bucket", "unlabeled")),
                "pose": {
                    "x": float(pose.get("x", 0.0)),
                    "y": float(pose.get("y", 0.0)),
                    "z": float(pose.get("z", 0.0)),
                    "yaw_deg": float(pose.get("yaw_deg", pose.get("yaw", 0.0))),
                },
            }
        )

    catalog_raw = _load_yaml_or_json(goal_catalog_path)
    catalog = normalize_catalog(catalog_raw)
    goal_catalog_version = int(catalog.get("version", cfg.get("version", 31)))
    goals_all = catalog.get("goals", []) if isinstance(catalog.get("goals"), list) else []
    goals: List[Dict[str, Any]] = []
    for g in goals_all:
        if not isinstance(g, dict):
            continue
        meta = g.get("meta", {}) if isinstance(g.get("meta", {}), dict) else {}
        bucket = str(meta.get("bucket", "unlabeled"))
        if include_buckets and bucket not in include_buckets:
            continue
        goals.append(
            {
                "id": _sanitize_id(str(g.get("id", f"goal_{len(goals)+1:03d}"))),
                "type": str(g.get("type", "pose")).strip().lower(),
                "bucket": bucket,
                "aliases": list(meta.get("aliases", [])) if isinstance(meta.get("aliases", []), list) else [],
                "value": g.get("value", {}),
                "success": g.get("success", {}),
                "meta": meta,
            }
        )
        if len(goals) >= max_goals:
            break

    anchor_tag = _anchor_tag_from_version(goal_catalog_version)
    object_caps_ok, object_caps_reason = capability_object_semantics()
    if goal_catalog_version >= 32:
        print(f"[V32_CAPS] object_semantics={int(object_caps_ok)} reason={object_caps_reason}", flush=True)

    runs_total = len(starts) * len(goals)
    start_anchor = (
        f"[{anchor_tag}] start scene={scene_id} starts={len(starts)} goals={len(goals)} "
        f"runs_total={runs_total} out_root={out_root}"
    )
    print(start_anchor, flush=True)
    with master_log.open("a", encoding="utf-8") as f:
        f.write(start_anchor + "\n")

    if len(starts) == 0 or len(goals) == 0:
        raise RuntimeError("No starts or goals selected; check eval config/catalog.")

    build_scene_dir = builds_dir / scene_id
    build_art = _mk_synth_build(build_scene_dir, scene_id)

    scenes_cfg = _load_yaml_or_json(scenes_cfg_path)
    condition = str(run_cfg.get("condition", "gt_pose"))
    max_steps = int(run_cfg.get("max_steps", 60))
    timeout_s = int(run_cfg.get("timeout_s", 600))
    build_debug_ui_mode = str(run_cfg.get("build_debug_ui", "failures_only")).strip().lower()

    random.seed(int(run_cfg.get("seed", 0)))
    np.random.seed(int(run_cfg.get("seed", 0)))

    run_out_dir = out_root / f"condition_{condition}" / scene_id
    run_out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    for si, s in enumerate(starts):
        start_cfg_path = runtime_dir / f"isaac_scene_cfg_{scene_id}__{s['id']}.yaml"
        _build_start_scene_cfg(scenes_cfg, scene_id=scene_id, start=s, out_cfg_path=start_cfg_path)

        for gi, g in enumerate(goals):
            run_id = _sanitize_id(f"{scene_id}__{s['id']}__{g['id']}")
            goal_type = str(g.get("type", "pose")).strip().lower()
            query, compat_mode, room_id, object_q = _as_pose_query_from_goal(g)
            query_json = json.dumps(query, ensure_ascii=False)

            goal_line = (
                f"[V31_GOAL] run_id={run_id} goal_id={g['id']} goal_type={goal_type} "
                f"compat_mode={compat_mode}"
            )
            print(goal_line, flush=True)
            with master_log.open("a", encoding="utf-8") as f:
                f.write(goal_line + "\n")

            if goal_catalog_version >= 32:
                detail = object_q if goal_type == "object" else json.dumps(query.get("value", {}), ensure_ascii=False)
                print(
                    f"[V32_GOAL] id={g['id']} type={goal_type} resolved=1 details={detail}",
                    flush=True,
                )

            log_path = logs_dir / f"{run_id}.log"

            if goal_type == "object" and not object_caps_ok:
                reason = object_caps_reason
                print(f"[V32_CAPS] object_semantics=0 reason={reason}", flush=True)
                rec = {
                    "ts": dt.datetime.now().isoformat(),
                    "run_id": run_id,
                    "scene_id": scene_id,
                    "condition": condition,
                    "start_id": s["id"],
                    "start_bucket": s.get("bucket", "unlabeled"),
                    "start_pose": s.get("pose", {}),
                    "goal_id": g["id"],
                    "goal_type": goal_type,
                    "goal_bucket": g.get("bucket", "unlabeled"),
                    "goal_aliases": g.get("aliases", []),
                    "goal_pose": {},
                    "compat_mode": int(compat_mode),
                    "room_id": room_id,
                    "object_query": object_q,
                    "success": 0,
                    "fail_type": "invalid_goal",
                    "reason": reason,
                    "steps_used": 0,
                    "final_dist_to_goal_node": -1.0,
                    "exit_code": 0,
                    "timed_out": 0,
                    "renderer_used": "unsupported",
                    "placeholder_ratio": None,
                    "camera": {},
                    "result_json": "",
                    "log_path": str(log_path),
                    "debug_index": "",
                    "trace_stats": {},
                    "v33a_stuck_triggers": 0,
                    "v33a_recovery_count": 0,
                    "v33a_recovery_giveup": 0,
                    "v33b_blocked_count": 0,
                    "v33b_override_count": 0,
                    "v33b_doorway_trigger_count": 0,
                    "v33b_caps_depth0": 0,
                }
                with log_path.open("w", encoding="utf-8") as lf:
                    lf.write(f"# run_id={run_id}\n")
                    lf.write(f"[V32_CAPS] object_semantics=0 reason={reason}\n")
                records.append(rec)
                line = (
                    f"[V31_RUN] id={run_id} success=0 fail_type=invalid_goal reason={reason} "
                    f"steps=0 final_dist=-1.000 log={log_path} debug_index="
                )
                print(line, flush=True)
                with master_log.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                continue

            node_reach_thresh = 0.5
            if goal_type == "pose":
                node_reach_thresh = float(g.get("success", {}).get("dist_m", 0.5) or 0.5)

            cmd = [
                "bash",
                "scripts/gpu/run_isaac_on_5090.sh",
                "--",
                "bash",
                "scripts/topo_run_backend.sh",
                "--backend",
                "isaac",
                "--condition",
                condition,
                "--scene_id",
                scene_id,
                "--run_id",
                run_id,
                "--query",
                query_json,
                "--graph_json",
                str(build_art["graph_json"]),
                "--embeds_npy",
                str(build_art["embeds_npy"]),
                "--clip_embeds_npy",
                str(build_art["clip_embeds_npy"]),
                "--node_meta",
                str(build_art["node_meta"]),
                "--out_dir",
                str(run_out_dir),
                "--debug_root",
                str(debug_root),
                "--isaac_config",
                str(start_cfg_path),
                "--config_path",
                "repo/InternNav/scripts/eval/configs/vln_r2r.yaml",
                "--controller_mode",
                "waypoint_follow",
                "--embedder_mode",
                "simple",
                "--simple_embed_dim",
                "64",
                "--max_steps",
                str(max_steps),
                "--goal_topk",
                "3",
                "--start_episode_offset",
                "0",
                "--kidnap_start",
                "0",
                "--node_reach_thresh",
                str(node_reach_thresh),
                "--planner_grid_res",
                "0.25",
                "--waypoint_reach_thresh",
                "0.45",
                "--debug_capture",
                "1",
                "--debug_only_on_fail",
                "1" if build_debug_ui_mode == "failures_only" else "0",
                "--debug_frame_stride",
                "1",
                "--debug_ringbuf_steps",
                "220",
                "--debug_save_depth",
                "0",
                "--debug_oracle_on_fail",
                "1",
                "--debug_capture_success_at_maxsteps",
                "1",
                "--debug_placeholder",
                "1",
            ]

            timed_out = False
            exit_code = 0
            with log_path.open("w", encoding="utf-8") as lf:
                lf.write(f"# run_id={run_id}\n# cmd={' '.join(cmd)}\n")
                lf.flush()
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(root),
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        timeout=timeout_s,
                        check=False,
                    )
                    exit_code = int(proc.returncode)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    exit_code = 124
                    lf.write(f"\n[V31_RUN_TIMEOUT] run_id={run_id} timeout_s={timeout_s}\n")

            result_path = run_out_dir / f"RESULT_{run_id}.json"
            result: Optional[Dict[str, Any]] = None
            if result_path.exists():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except Exception:
                    result = None

            debug_run_dir = debug_root / condition / run_id
            trace_rows = _load_trace_rows(debug_run_dir / "trace.jsonl")
            trace_stats = _load_trace_stats(debug_run_dir)
            v33a_stats = _parse_v33a_log(log_path)
            v33b_stats = _parse_v33b_log(log_path)
            fail_type, reason = _classify_fail(result=result, timed_out=timed_out, exit_code=exit_code, trace_stats=trace_stats)
            if fail_type == "unknown" and int(v33a_stats.get("giveup", 0)) > 0:
                fail_type, reason = "stuck", "v33a_recovery_giveup"
            if fail_type == "unknown" and int(v33b_stats.get("caps_depth0", 0)) > 0:
                fail_type, reason = "backend_caps", "backend_caps_depth_missing"

            success = bool(result.get("success", False)) if isinstance(result, dict) else False
            steps_used = int(result.get("steps_used", 0)) if isinstance(result, dict) else 0
            final_dist = float(result.get("final_dist_to_goal_node", -1.0)) if isinstance(result, dict) else -1.0
            renderer_used = str(result.get("renderer_used", "unknown")) if isinstance(result, dict) else "unknown"
            camera = result.get("camera", {}) if isinstance(result, dict) and isinstance(result.get("camera", {}), dict) else {}
            placeholder_ratio = None
            if isinstance(camera.get("fidelity", {}), dict):
                placeholder_ratio = float(camera.get("fidelity", {}).get("placeholder_ratio", 0.0))
            result_v33b = result.get("v33b", {}) if isinstance(result, dict) and isinstance(result.get("v33b", {}), dict) else {}
            blocked_count = int(
                result_v33b.get(
                    "blocked_detected_count",
                    result_v33b.get("blocked_count", v33b_stats.get("blocked", 0)),
                )
                or 0
            )
            override_count = int(result_v33b.get("override_count", v33b_stats.get("overrides", 0)) or 0)
            blocked_no_override_count = int(
                result_v33b.get("blocked_no_override_count", v33b_stats.get("blocked_no_override", 0)) or 0
            )
            blocked_no_override_reasons = result_v33b.get(
                "blocked_no_override_reasons",
                v33b_stats.get("blocked_no_override_reasons", {}),
            )
            if not isinstance(blocked_no_override_reasons, dict):
                blocked_no_override_reasons = {}
            doorway_trigger_count = int(result_v33b.get("doorway_triggers", v33b_stats.get("doorway", 0)) or 0)
            blocked_forward_count = int(result_v33b.get("blocked_forward_count", blocked_count) or 0)
            blocked_forward_overridden = int(result_v33b.get("blocked_forward_overridden_count", override_count) or 0)
            blocked_forward_pass_through = int(
                result_v33b.get("blocked_forward_pass_through_count", blocked_no_override_count) or 0
            )
            blocked_forward_reasons = result_v33b.get("blocked_forward_reasons", blocked_no_override_reasons)
            if not isinstance(blocked_forward_reasons, dict):
                blocked_forward_reasons = {}

            if goal_type == "room":
                room_ok, room_metrics = is_room_success_from_trace(trace_rows, g)
                if room_ok:
                    success = True
                    fail_type = "success"
                    reason = "room_region_settled"
                    print(
                        f"[V32_SUCCESS] id={g['id']} type=room ok=1 metric={json.dumps(room_metrics, ensure_ascii=False)}",
                        flush=True,
                    )
                elif success:
                    success = False
                    fail_type = "max_steps"
                    reason = "room_settle_not_met"
            elif goal_type == "pose" and success:
                last_pose = trace_stats.get("last_pose")
                if isinstance(last_pose, dict):
                    pose_ok, pose_metric = is_pose_success(last_pose, g)
                    if pose_ok:
                        print(
                            f"[V32_SUCCESS] id={g['id']} type=pose ok=1 metric={json.dumps(pose_metric, ensure_ascii=False)}",
                            flush=True,
                        )

            build_debug = (
                build_debug_ui_mode == "all"
                or (build_debug_ui_mode == "failures_only" and not success)
            ) and debug_run_dir.exists()
            debug_index = ""
            if build_debug:
                try:
                    subprocess.run(
                        [
                            "python3",
                            "scripts/topo_debug_ui_build_report.py",
                            "--run_dir",
                            str(debug_run_dir),
                            "--build_root",
                            str(builds_dir),
                        ],
                        cwd=str(root),
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    rep = debug_run_dir / "report" / "index.html"
                    if rep.exists():
                        debug_index = str(rep)
                except Exception:
                    debug_index = ""

            rec = {
                "ts": dt.datetime.now().isoformat(),
                "run_id": run_id,
                "scene_id": scene_id,
                "condition": condition,
                "start_id": s["id"],
                "start_bucket": s.get("bucket", "unlabeled"),
                "start_pose": s.get("pose", {}),
                "goal_id": g["id"],
                "goal_type": goal_type,
                "goal_bucket": g.get("bucket", "unlabeled"),
                "goal_aliases": g.get("aliases", []),
                "goal_pose": query.get("value", {}),
                "compat_mode": int(compat_mode),
                "room_id": room_id,
                "object_query": object_q,
                "success": int(bool(success)),
                "fail_type": fail_type,
                "reason": reason,
                "steps_used": int(steps_used),
                "final_dist_to_goal_node": float(final_dist),
                "exit_code": int(exit_code),
                "timed_out": int(bool(timed_out)),
                "renderer_used": renderer_used,
                "placeholder_ratio": placeholder_ratio,
                "camera": camera,
                "result_json": str(result_path) if result_path.exists() else "",
                "log_path": str(log_path),
                "debug_index": debug_index,
                "trace_stats": trace_stats,
                "v33a_stuck_triggers": int(v33a_stats.get("stuck_triggers", 0)),
                "v33a_recovery_count": int(v33a_stats.get("recoveries", 0)),
                "v33a_recovery_giveup": int(v33a_stats.get("giveup", 0)),
                "v33b_blocked_count": blocked_count,
                "v33b_override_count": override_count,
                "v33b_blocked_no_override_count": blocked_no_override_count,
                "v33b_blocked_no_override_reasons": {
                    str(k): int(v) for k, v in sorted(blocked_no_override_reasons.items())
                },
                "v33b_blocked_forward_count": blocked_forward_count,
                "v33b_blocked_forward_overridden_count": blocked_forward_overridden,
                "v33b_blocked_forward_pass_through_count": blocked_forward_pass_through,
                "v33b_blocked_forward_reasons": {
                    str(k): int(v) for k, v in sorted(blocked_forward_reasons.items())
                },
                "v33b_doorway_trigger_count": doorway_trigger_count,
                "v33b_caps_depth0": int(v33b_stats.get("caps_depth0", 0)),
            }
            records.append(rec)

            line = (
                f"[V31_RUN] id={run_id} success={int(bool(success))} fail_type={fail_type} reason={reason} "
                f"steps={steps_used} final_dist={final_dist:.3f} log={log_path} debug_index={debug_index}"
            )
            print(line, flush=True)
            with master_log.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    try:
        subprocess.run(
            ["python3", "scripts/topo_debug_ui_index.py", "--root", str(debug_root)],
            cwd=str(root),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

    failure_cases = out_root / "failure_cases.jsonl"
    with failure_cases.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_csv(out_root / "runs.csv", records)

    success_n = sum(int(r.get("success", 0)) for r in records)
    fail_n = len(records) - success_n
    by_fail = Counter(str(r.get("fail_type", "unknown")) for r in records)
    by_bucket = Counter(str(r.get("goal_bucket", "unlabeled")) for r in records)
    by_start = Counter(str(r.get("start_id", "")) for r in records)
    by_goal = Counter(str(r.get("goal_id", "")) for r in records)
    by_goal_type = Counter(str(r.get("goal_type", "pose")) for r in records)
    stuck_triggers_total = int(sum(int(r.get("v33a_stuck_triggers", 0) or 0) for r in records))
    recoveries_total = int(sum(int(r.get("v33a_recovery_count", 0) or 0) for r in records))
    stuck_failures = int(sum(1 for r in records if str(r.get("fail_type", "")) == "stuck"))
    v33b_blocked_total = int(sum(int(r.get("v33b_blocked_count", 0) or 0) for r in records))
    v33b_override_total = int(sum(int(r.get("v33b_override_count", 0) or 0) for r in records))
    v33b_blocked_no_override_total = int(sum(int(r.get("v33b_blocked_no_override_count", 0) or 0) for r in records))
    v33b_blocked_forward_total = int(sum(int(r.get("v33b_blocked_forward_count", 0) or 0) for r in records))
    v33b_blocked_forward_overridden_total = int(
        sum(int(r.get("v33b_blocked_forward_overridden_count", 0) or 0) for r in records)
    )
    v33b_blocked_forward_pass_total = int(
        sum(int(r.get("v33b_blocked_forward_pass_through_count", 0) or 0) for r in records)
    )
    v33b_doorway_total = int(sum(int(r.get("v33b_doorway_trigger_count", 0) or 0) for r in records))
    v33b_depth_missing = int(sum(int(r.get("v33b_caps_depth0", 0) or 0) for r in records))
    v33b_no_override_reasons = Counter()
    for r in records:
        reasons = r.get("v33b_blocked_no_override_reasons", {})
        if not isinstance(reasons, dict):
            continue
        for k, v in reasons.items():
            try:
                v33b_no_override_reasons[str(k)] += int(v)
            except Exception:
                continue
    v33b_forward_reasons = Counter()
    for r in records:
        reasons = r.get("v33b_blocked_forward_reasons", {})
        if not isinstance(reasons, dict):
            continue
        for k, v in reasons.items():
            try:
                v33b_forward_reasons[str(k)] += int(v)
            except Exception:
                continue

    summary = {
        "version": int(goal_catalog_version),
        "scene": scene_id,
        "condition": condition,
        "runs_total": len(records),
        "success": int(success_n),
        "failure": int(fail_n),
        "by_fail_type": dict(sorted(by_fail.items())),
        "by_goal_bucket": dict(sorted(by_bucket.items())),
        "by_start_id": dict(sorted(by_start.items())),
        "by_goal_id": dict(sorted(by_goal.items())),
        "by_goal_type": dict(sorted(by_goal_type.items())),
        "starts_used": len(starts),
        "goals_used": len(goals),
        "v33a": {
            "stuck_triggers": stuck_triggers_total,
            "recoveries": recoveries_total,
            "stuck_failures": stuck_failures,
        },
        "v33b": {
            "blocked_total": v33b_blocked_total,
            "overrides_total": v33b_override_total,
            "blocked_no_override_total": v33b_blocked_no_override_total,
            "blocked_no_override_reasons": dict(sorted(v33b_no_override_reasons.items())),
            "blocked_forward_total": v33b_blocked_forward_total,
            "blocked_forward_overridden_total": v33b_blocked_forward_overridden_total,
            "blocked_forward_pass_through_total": v33b_blocked_forward_pass_total,
            "blocked_forward_reasons": dict(sorted(v33b_forward_reasons.items())),
            "doorway_triggers_total": v33b_doorway_total,
            "depth_caps_missing_runs": v33b_depth_missing,
        },
        "small_mode": int(bool(args.small)),
        "paths": {
            "failure_cases": str(failure_cases),
            "runs_csv": str(out_root / "runs.csv"),
            "debug_index": str(debug_root / "index.html"),
        },
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    index_path = out_root / "index.html"
    _write_index(index_path, summary, failure_cards_rel="failure_cards/index.html")

    by_fail_str = ",".join(f"{k}:{v}" for k, v in sorted(by_fail.items()))
    done_anchor = (
        f"[{anchor_tag}] done runs_total={len(records)} success={success_n} failure={fail_n} "
        f"by_fail_type=\"{by_fail_str}\" version={goal_catalog_version}"
    )
    out_anchor = (
        f"[{anchor_tag}_OUT] root={out_root} failure_cases={failure_cases} "
        f"summary={summary_path} index={index_path}"
    )
    print(done_anchor, flush=True)
    print(out_anchor, flush=True)
    v33a_anchor = (
        f"[V33A_SUMMARY] stuck_triggers={stuck_triggers_total} "
        f"recoveries={recoveries_total} stuck_failures={stuck_failures}"
    )
    v33b_anchor = (
        f"[V33B_SUMMARY] blocked={v33b_blocked_total} overrides={v33b_override_total} "
        f"blocked_no_override={v33b_blocked_no_override_total} "
        f"blocked_no_override_reasons=\"{','.join(f'{k}:{v}' for k, v in sorted(v33b_no_override_reasons.items()))}\" "
        f"blocked_forward={v33b_blocked_forward_total} "
        f"blocked_forward_overridden={v33b_blocked_forward_overridden_total} "
        f"blocked_forward_pass_through={v33b_blocked_forward_pass_total} "
        f"blocked_forward_reasons=\"{','.join(f'{k}:{v}' for k, v in sorted(v33b_forward_reasons.items()))}\" "
        f"doorway_triggers={v33b_doorway_total} depth_caps_missing_runs={v33b_depth_missing}"
    )
    print(v33a_anchor, flush=True)
    print(v33b_anchor, flush=True)
    if anchor_tag != "V31_SCALE_EVAL":
        legacy_done = (
            f"[V31_SCALE_EVAL] done runs_total={len(records)} success={success_n} failure={fail_n} "
            f"by_fail_type=\"{by_fail_str}\" version={goal_catalog_version}"
        )
        legacy_out = (
            f"[V31_SCALE_EVAL_OUT] root={out_root} failure_cases={failure_cases} "
            f"summary={summary_path} index={index_path} version={goal_catalog_version}"
        )
        print(legacy_done, flush=True)
        print(legacy_out, flush=True)
    if goal_catalog_version >= 32:
        v32_out = (
            f"[V32_SCALE_EVAL_OUT] root={out_root} failure_cases={failure_cases} "
            f"summary={summary_path} index={index_path}"
        )
        print(v32_out, flush=True)
    with master_log.open("a", encoding="utf-8") as f:
        f.write(done_anchor + "\n")
        f.write(out_anchor + "\n")
        f.write(v33a_anchor + "\n")
        f.write(v33b_anchor + "\n")
        if anchor_tag != "V31_SCALE_EVAL":
            f.write(legacy_done + "\n")
            f.write(legacy_out + "\n")
        if goal_catalog_version >= 32:
            f.write(v32_out + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
