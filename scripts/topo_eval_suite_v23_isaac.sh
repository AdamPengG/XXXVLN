#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
RUN_DIR="${ROOT_DIR}/runs/topo_mvp/v23_isaac"
BUILD_DIR="${RUN_DIR}/builds"
LOG_DIR="${RUN_DIR}/logs"
DEBUG_ROOT="${RUN_DIR}/debug_runs"
CFG_PATH="${ROOT_DIR}/configs/isaac_scenes_v23.yaml"
MASTER_LOG="${LOG_DIR}/topo_eval_suite_v23_isaac.log"

mkdir -p "${RUN_DIR}" "${BUILD_DIR}" "${LOG_DIR}" "${DEBUG_ROOT}"
chmod -R a+rwx "${RUN_DIR}" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export ISAAC_BACKEND_ALLOW_FALLBACK="${ISAAC_BACKEND_ALLOW_FALLBACK:-0}"
export TOPO_DEBUG_CAPTURE="${TOPO_DEBUG_CAPTURE:-1}"
export TOPO_DEBUG_ONLY_ON_FAIL="${TOPO_DEBUG_ONLY_ON_FAIL:-1}"
export TOPO_DEBUG_FRAME_STRIDE="${TOPO_DEBUG_FRAME_STRIDE:-1}"
export TOPO_DEBUG_PLACEHOLDER="${TOPO_DEBUG_PLACEHOLDER:-1}"

: > "${MASTER_LOG}"

echo "[TOPO_V23] cfg=${CFG_PATH}" | tee -a "${MASTER_LOG}"
bash scripts/isaac/run_with_isaac_python.sh -- -c "print('ISAAC_RUNNER_READY')" | tee -a "${MASTER_LOG}"

load_cfg_json() {
  python - "${CFG_PATH}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {}
if path.exists():
    try:
        import yaml  # type: ignore
        payload = yaml.safe_load(path.read_text()) or {}
    except Exception:
        payload = json.loads(path.read_text())
print(json.dumps(payload))
PY
}

CFG_JSON="$(load_cfg_json)"
SCENE_LIMIT="${TOPO_V23_SCENE_LIMIT:-1}"
RUN_LIMIT="${TOPO_V23_RUN_LIMIT:-1}"
START_LIMIT="${TOPO_V23_START_LIMIT:-1}"
QUERY_LIMIT="${TOPO_V23_QUERY_LIMIT:-1}"
CONDITIONS="${TOPO_V23_CONDITIONS:-gt_pose,est_pose_noise}"
MAX_STEPS="${TOPO_V23_MAX_STEPS:-120}"

mapfile -t scenes < <(python - "${CFG_JSON}" "${SCENE_LIMIT}" <<'PY'
import json, sys
cfg = json.loads(sys.argv[1])
lim = int(sys.argv[2])
rows = [str(x.get("scene_id", "")).strip() for x in cfg.get("scenes", []) if str(x.get("scene_id", "")).strip()]
if lim > 0:
    rows = rows[:lim]
for r in rows:
    print(r)
PY
)

mapfile -t queries < <(python - "${CFG_JSON}" "${QUERY_LIMIT}" <<'PY'
import json, sys
cfg = json.loads(sys.argv[1])
lim = int(sys.argv[2])
rows = [str(x) for x in cfg.get("global", {}).get("queries", [])]
if lim > 0:
    rows = rows[:lim]
for r in rows:
    print(r)
PY
)

mapfile -t starts < <(python - "${CFG_JSON}" "${START_LIMIT}" <<'PY'
import json, sys
cfg = json.loads(sys.argv[1])
lim = int(sys.argv[2])
rows = [int(x) for x in cfg.get("global", {}).get("start_offsets", [])]
if lim > 0:
    rows = rows[:lim]
for r in rows:
    print(r)
PY
)

if [ "${#scenes[@]}" -eq 0 ]; then
  echo "No scenes configured in ${CFG_PATH}" >&2
  exit 2
fi

# Sanity check (hard gate)
first_scene="${scenes[0]}"
if ! bash scripts/isaac/run_with_isaac_python.sh -- scripts/topo_isaac_sanity_check.py --scene_id "${first_scene}" --isaac_config "${CFG_PATH}" | tee -a "${MASTER_LOG}"; then
  echo "[ISAAC_SANITY_FAIL] scene=${first_scene} reason=gate_failed" | tee -a "${MASTER_LOG}"
  exit 3
fi

for scene in "${scenes[@]}"; do
  out="${BUILD_DIR}/${scene}"
  mkdir -p "${out}/frames"
  python - "${out}" "${scene}" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

out_dir = Path(sys.argv[1])
scene = sys.argv[2]
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "frames").mkdir(parents=True, exist_ok=True)

rows, cols = 4, 4
spacing = 2.0
nodes = []
edges = []
embeds = []
clip_embeds = []
node_meta = {}
node_to_room = {}

room_palette = {
    0: (180, 70, 70),
    1: (70, 180, 90),
    2: (70, 90, 180),
    3: (180, 160, 70),
}

def mk_embed(x, z, rid, dim=64):
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
        room_id = (0 if r < 2 and c < 2 else 1 if r < 2 else 2 if c < 2 else 3)
        rgba = room_palette[room_id]
        img = np.zeros((160, 240, 3), dtype=np.uint8)
        img[..., 0] = int(rgba[0] + 10 * c)
        img[..., 1] = int(rgba[1] + 10 * r)
        img[..., 2] = int(rgba[2])
        frame = out_dir / "frames" / f"node_{nid:04d}.png"
        Image.fromarray(img).save(frame)

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
            "pose_source": "sim",
            "gt_position": [float(x), 0.0, float(z)],
            "gt_rotation": [1.0, 0.0, 0.0, 0.0],
            "gt_yaw": 0.0,
            "est_position": [float(x), 0.0, float(z)],
            "est_yaw": 0.0,
            "rgb_path": str(frame),
            "room_id": int(room_id),
            "room_label": ["hallway", "office", "meeting_room", "kitchen"][room_id],
            "room_score": 0.85,
            "room_top3": [],
            "clearance_m": 1.0,
            "clearance_proxy": 1.0,
            "object_detections": [],
        }
        node_to_room[str(nid)] = int(room_id)
        nid += 1

def node_id(rr, cc):
    return rr * cols + cc

for r in range(rows):
    for c in range(cols):
        u = node_id(r, c)
        if c + 1 < cols:
            v = node_id(r, c + 1)
            edges.append({"u": u, "v": v, "w": spacing, "edge_type": "temporal"})
        if r + 1 < rows:
            v = node_id(r + 1, c)
            edges.append({"u": u, "v": v, "w": spacing, "edge_type": "temporal"})

graph = {"meta": {"scene_id": scene, "backend": "isaac_v23"}, "nodes": nodes, "edges": edges}
(out_dir / "graph.json").write_text(json.dumps(graph, indent=2))
np.save(out_dir / "node_embeds.npy", np.stack(embeds, axis=0).astype(np.float32))
np.save(out_dir / "node_embeds_clip.npy", np.stack(clip_embeds, axis=0).astype(np.float32))
(out_dir / "node_meta.json").write_text(json.dumps({"meta": {"scene_id": scene}, "node_meta": node_meta}, indent=2))
(out_dir / "node_to_room.json").write_text(json.dumps({"scene_id": scene, "node_to_room": node_to_room}, indent=2))
(out_dir / "room_partition.json").write_text(json.dumps({"scene_id": scene, "rooms": {}}, indent=2))
PY
done

IFS=',' read -r -a conds <<< "${CONDITIONS}"
for cond in "${conds[@]}"; do
  cond_dir="${RUN_DIR}/condition_${cond}"
  cond_log="${LOG_DIR}/topo_eval_suite_v23_isaac_${cond}.log"
  rm -rf "${cond_dir}"
  mkdir -p "${cond_dir}"
  chmod -R a+rwx "${RUN_DIR}" 2>/dev/null || true
  : > "${cond_log}"

  run_count=0
  for scene in "${scenes[@]}"; do
    scene_out="${cond_dir}/${scene}"
    mkdir -p "${scene_out}"
    chmod -R a+rwx "${scene_out}" "${DEBUG_ROOT}" 2>/dev/null || true
    graph_json="${BUILD_DIR}/${scene}/graph.json"
    embeds_npy="${BUILD_DIR}/${scene}/node_embeds.npy"
    clip_npy="${BUILD_DIR}/${scene}/node_embeds_clip.npy"
    node_meta="${BUILD_DIR}/${scene}/node_meta.json"
    for si in "${!starts[@]}"; do
      start_offset="${starts[$si]}"
      for qi in "${!queries[@]}"; do
        query="${queries[$qi]}"
        run_id="${scene}_q${qi}_s${si}_bearing"
        if [ "${RUN_LIMIT}" -gt 0 ] && [ "${run_count}" -ge "${RUN_LIMIT}" ]; then
          break 3
        fi
        echo "[TOPO_EVAL_RUN] backend=isaac condition=${cond} scene=${scene} run_id=${run_id} query=\"${query}\" start_offset=${start_offset}" | tee -a "${MASTER_LOG}"
        (
          cd "${ROOT_DIR}"
          bash scripts/topo_run_backend.sh \
            --backend isaac \
            --condition "${cond}" \
            --scene_id "${scene}" \
            --run_id "${run_id}" \
            --query "${query}" \
            --graph_json "${graph_json}" \
            --embeds_npy "${embeds_npy}" \
            --clip_embeds_npy "${clip_npy}" \
            --node_meta "${node_meta}" \
            --out_dir "${scene_out}" \
            --debug_root "${DEBUG_ROOT}" \
            --isaac_config "${CFG_PATH}" \
            --config_path repo/InternNav/scripts/eval/configs/vln_r2r.yaml \
            --controller_mode bearing \
            --embedder_mode simple \
            --simple_embed_dim 64 \
            --max_steps "${MAX_STEPS}" \
            --goal_topk 3 \
            --start_episode_offset "${start_offset}" \
            --kidnap_start 1 \
            --kidnap_seed "$((4040 + run_count))" \
            --kidnap_min_nearest_node_dist 0.2 \
            --kidnap_max_nearest_node_dist 3.5 \
            --node_reach_thresh 1.2 \
            --debug_capture "${TOPO_DEBUG_CAPTURE}" \
            --debug_only_on_fail "${TOPO_DEBUG_ONLY_ON_FAIL}" \
            --debug_frame_stride "${TOPO_DEBUG_FRAME_STRIDE}" \
            --debug_ringbuf_steps "${TOPO_DEBUG_RINGBUF_STEPS:-220}" \
            --debug_save_depth "${TOPO_DEBUG_SAVE_DEPTH:-0}" \
            --debug_oracle_on_fail "${TOPO_DEBUG_ORACLE_ON_FAIL:-1}" \
            --debug_capture_success_at_maxsteps "${TOPO_DEBUG_CAPTURE_SUCCESS_AT_MAXSTEPS:-1}" \
            --debug_placeholder "${TOPO_DEBUG_PLACEHOLDER}"
        ) >> "${cond_log}" 2>&1
        isaac_anchor="$(
          rg -n "\\[ISAAC_BACKEND\\]" -S "${cond_log}" 2>/dev/null \
            | tail -n 1 \
            | sed 's/^[0-9]\+://'
        )"
        if [ -n "${isaac_anchor}" ]; then
          echo "${isaac_anchor}" | tee -a "${MASTER_LOG}"
        fi
        run_count=$((run_count + 1))
      done
    done
  done

  cond_summary="${RUN_DIR}/condition_${cond}_summary.json"
  cond_fail="${RUN_DIR}/condition_${cond}_failure_taxonomy.json"
  python - "${cond_dir}" "${cond}" "${cond_summary}" "${cond_fail}" <<'PY'
import glob
import json
import os
import statistics
import sys
from collections import Counter

cond_dir, cond, out_summary, out_fail = sys.argv[1:]
rows = []
for p in sorted(glob.glob(os.path.join(cond_dir, "*", "RESULT_*.json"))):
    try:
        rows.append(json.load(open(p)))
    except Exception:
        continue
runs = len(rows)
succ = sum(1 for r in rows if bool(r.get("success", False)))
steps = [int(r.get("steps_used", 0)) for r in rows]
locs = [float(r.get("avg_loc_conf", 0.0)) for r in rows]
backend_mode = "unknown"
if rows:
    backend_mode = str(rows[0].get("backend_mode", rows[0].get("backend", "unknown")))
fails = [r for r in rows if not bool(r.get("success", False))]
tax = Counter([str(r.get("fail_reason", "unknown")) for r in fails])
summary = {
    "condition": cond,
    "backend_mode": backend_mode,
    "runs": runs,
    "success": succ,
    "avg_steps": float(statistics.mean(steps) if steps else 0.0),
    "avg_loc_conf": float(statistics.mean(locs) if locs else 0.0),
}
json.dump(summary, open(out_summary, "w"), indent=2)
json.dump(dict(tax), open(out_fail, "w"), indent=2)
print(
    f"[TOPO_EVAL_SUMMARY_V23_ISAAC] backend={backend_mode} condition={cond} "
    f"runs={runs} success={succ} avg_steps={summary['avg_steps']:.2f} "
    f"avg_loc_conf={summary['avg_loc_conf']:.3f}"
)
PY
done | tee -a "${MASTER_LOG}"

if [ "${TOPO_DEBUG_CAPTURE}" = "1" ]; then
  while IFS= read -r run_dir; do
    if [ -f "${run_dir}/trace.jsonl" ] && [ -f "${run_dir}/meta.json" ]; then
      (
        cd "${ROOT_DIR}"
        python scripts/topo_debug_ui_build_report.py \
          --run_dir "${run_dir}" \
          --build_root "${BUILD_DIR}"
      ) >> "${MASTER_LOG}" 2>&1 || true
    fi
  done < <(find "${DEBUG_ROOT}" -mindepth 2 -maxdepth 2 -type d | sort)
fi

(
  cd "${ROOT_DIR}"
  python scripts/topo_debug_ui_index.py --root "${DEBUG_ROOT}"
) >> "${MASTER_LOG}" 2>&1 || true
echo "[TOPO_DEBUG_INDEX_OK] path=${DEBUG_ROOT}/index.html" | tee -a "${MASTER_LOG}"

python - "${RUN_DIR}" <<'PY'
import glob
import json
import os
import sys

run_dir = sys.argv[1]
out = {"conditions": {}}
for p in sorted(glob.glob(os.path.join(run_dir, "condition_*_summary.json"))):
    key = os.path.basename(p).replace("condition_", "").replace("_summary.json", "")
    out["conditions"][key] = json.load(open(p))
json.dump(out, open(os.path.join(run_dir, "v23_isaac_summary.json"), "w"), indent=2)
print(f"[V23_ISAAC] wrote {os.path.join(run_dir, 'v23_isaac_summary.json')}")
PY
