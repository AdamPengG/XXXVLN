#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Tuple

# Allow importing outer runner helpers when executed from Isaac python.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def _load_catalog(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {
            "version": 30,
            "assets_root": os.environ.get("ISAAC_ASSETS_ROOT", "/home/peng/isaacsim_assets"),
            "scene": "office_localized",
            "goals": [],
        }
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("catalog_root_not_mapping")
        return data
    except Exception:
        # JSON fallback (YAML is a superset).
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("catalog_root_not_mapping")
        return data


def _dump_catalog(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import yaml  # type: ignore

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        return
    except Exception:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def _normalize_aliases(vals: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in vals:
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _next_goal_id(existing_ids: List[str], prefix: str) -> str:
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    mx = 0
    for gid in existing_ids:
        m = pat.match(str(gid))
        if not m:
            continue
        mx = max(mx, int(m.group(1)))
    return f"{prefix}{mx + 1:03d}"


def _validate_catalog(data: Dict[str, Any]) -> Tuple[int, Dict[str, int]]:
    goals = data.get("goals", [])
    if not isinstance(goals, list):
        raise ValueError("goals_must_be_list")

    seen = set()
    bucket_counts: Dict[str, int] = {}

    for g in goals:
        if not isinstance(g, dict):
            raise ValueError("goal_entry_must_be_mapping")
        gid = str(g.get("id", "")).strip()
        if not gid:
            raise ValueError("goal_id_missing")
        if gid in seen:
            raise ValueError(f"duplicate_goal_id:{gid}")
        seen.add(gid)

        gtype = str(g.get("type", "")).strip().lower()
        if gtype != "pose":
            raise ValueError(f"unsupported_goal_type:{gtype}")

        val = g.get("value", {})
        if not isinstance(val, dict):
            raise ValueError(f"value_must_be_mapping:{gid}")
        for k in ("x", "z", "yaw_deg"):
            if k not in val:
                raise ValueError(f"missing_value_field:{gid}:{k}")
            float(val[k])

        succ = g.get("success", {})
        if not isinstance(succ, dict):
            raise ValueError(f"success_must_be_mapping:{gid}")
        float(succ.get("dist_m", 0.0))
        float(succ.get("yaw_deg", 0.0))

        meta = g.get("meta", {})
        if not isinstance(meta, dict):
            raise ValueError(f"meta_must_be_mapping:{gid}")
        bucket = str(meta.get("bucket", "unlabeled")).strip() or "unlabeled"
        bucket_counts[bucket] = int(bucket_counts.get(bucket, 0) + 1)

        aliases = meta.get("aliases", [])
        if aliases is None:
            aliases = []
        if not isinstance(aliases, list):
            raise ValueError(f"aliases_must_be_list:{gid}")

    return int(len(goals)), bucket_counts


def _capture_pose(scene_id: str, isaac_config: str, habitat_config: str, out_dir: str, start_offset: int) -> Dict[str, float]:
    from topo_backend_runner import _make_backend  # lazy import; keeps startup light

    backend = _make_backend(
        backend_name="isaac",
        scene_id=scene_id,
        isaac_cfg_path=isaac_config,
        habitat_config_path=habitat_config,
        out_dir=out_dir,
    )
    # NOTE: do not call backend.close() here; in Isaac python this can terminate
    # the process before YAML append/anchor emission. Process teardown is enough.
    backend.reset(scene_id=scene_id, start_spec={"start_offset": int(start_offset)})
    pose = backend.get_pose()

    yaw_deg = (math.degrees(float(pose.yaw)) + 180.0) % 360.0 - 180.0
    out = {
        "x": float(pose.x),
        "y": float(pose.y),
        "z": float(pose.z),
        "yaw_deg": float(yaw_deg),
    }
    print(
        f"[ISAAC_POSE] x={out['x']:.3f} y={out['y']:.3f} z={out['z']:.3f} "
        f"yaw_deg={out['yaw_deg']:.2f} frame=world_xzyaw_deg",
        flush=True,
    )
    return out


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="v30 Plan-B goal authoring tool")
    ap.add_argument("--mode", choices=["record", "validate", "list"], required=True)
    ap.add_argument("--out_yaml", default="configs/isaac_goal_catalog_v30.yaml")
    ap.add_argument("--scene_id", default="office_localized")
    ap.add_argument("--isaac_config", default="configs/isaac_scenes_v29.yaml")
    ap.add_argument("--habitat_config", default="repo/InternNav/scripts/eval/configs/vln_r2r.yaml")
    ap.add_argument("--record_out_dir", default="runs/topo_mvp/v30_goal_author/runtime")
    ap.add_argument("--start_offset", type=int, default=0)

    ap.add_argument("--id", dest="goal_id", default="")
    ap.add_argument("--id_prefix", default="office_pose_")
    ap.add_argument("--bucket", default="med")
    ap.add_argument("--alias", action="append", default=[])
    ap.add_argument("--note", default="")
    ap.add_argument("--author", default=os.environ.get("USER", "unknown"))
    ap.add_argument("--created_at", default=dt.date.today().isoformat())
    ap.add_argument("--success_dist_m", type=float, default=0.50)
    ap.add_argument("--success_yaw_deg", type=float, default=15.0)
    return ap


def main() -> int:
    args = _build_parser().parse_args()
    try:
        data = _load_catalog(args.out_yaml)
        data.setdefault("version", 30)
        data.setdefault("assets_root", os.environ.get("ISAAC_ASSETS_ROOT", "/home/peng/isaacsim_assets"))
        data.setdefault("scene", args.scene_id)
        data.setdefault("goals", [])

        goals = data.get("goals", [])
        if not isinstance(goals, list):
            raise ValueError("goals_must_be_list")

        if args.mode in {"validate", "list"}:
            goal_count, bucket_counts = _validate_catalog(data)
            buckets_str = ",".join(f"{k}:{bucket_counts[k]}" for k in sorted(bucket_counts.keys()))
            print(
                f"[GOAL_AUTHOR_V30] ok=1 mode=validate out_yaml={args.out_yaml} "
                f"goals={goal_count} buckets=\"{buckets_str}\"",
                flush=True,
            )
            if args.mode == "list":
                for g in goals:
                    meta = g.get("meta", {}) if isinstance(g, dict) else {}
                    print(
                        f"- {g.get('id')} bucket={meta.get('bucket','')} aliases={meta.get('aliases',[])}",
                        flush=True,
                    )
            return 0

        # record mode
        existing_ids = [str(g.get("id", "")) for g in goals if isinstance(g, dict)]
        goal_id = str(args.goal_id).strip() if str(args.goal_id).strip() else _next_goal_id(existing_ids, args.id_prefix)
        if goal_id in existing_ids:
            raise ValueError(f"duplicate_goal_id:{goal_id}")

        pose = _capture_pose(
            scene_id=args.scene_id,
            isaac_config=args.isaac_config,
            habitat_config=args.habitat_config,
            out_dir=args.record_out_dir,
            start_offset=args.start_offset,
        )

        entry = {
            "id": goal_id,
            "type": "pose",
            "value": {
                "x": round(float(pose["x"]), 4),
                "z": round(float(pose["z"]), 4),
                "yaw_deg": round(float(pose["yaw_deg"]), 3),
            },
            "success": {
                "dist_m": float(args.success_dist_m),
                "yaw_deg": float(args.success_yaw_deg),
            },
            "meta": {
                "bucket": str(args.bucket),
                "aliases": _normalize_aliases(args.alias),
                "note": str(args.note),
                "author": str(args.author),
                "created_at": str(args.created_at),
            },
        }
        goals.append(entry)
        data["goals"] = goals
        data["scene"] = str(args.scene_id)
        _dump_catalog(args.out_yaml, data)

        print(
            f"[GOAL_AUTHOR_V30] ok=1 mode=record scene={args.scene_id} out_yaml={args.out_yaml} "
            f"id={goal_id} x={entry['value']['x']:.4f} z={entry['value']['z']:.4f} "
            f"yaw_deg={entry['value']['yaw_deg']:.3f} bucket={entry['meta']['bucket']} "
            f"aliases={entry['meta']['aliases']}",
            flush=True,
        )
        return 0

    except Exception as e:
        print(
            f"[GOAL_AUTHOR_V30] ok=0 reason={type(e).__name__}:{e} "
            f"hint=check_yaml_schema_or_backend_env",
            flush=True,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
