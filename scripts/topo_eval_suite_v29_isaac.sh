#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"

RUN_DIR="${ROOT_DIR}/runs/topo_mvp/v29_isaac"
BUILD_DIR="${RUN_DIR}/builds"
LOG_DIR="${RUN_DIR}/logs"
DEBUG_ROOT="${RUN_DIR}/debug_runs"
MASTER_LOG="${LOG_DIR}/topo_eval_suite_v29_isaac.log"
CFG_PATH="${ISAAC_CFG:-${ROOT_DIR}/configs/isaac_scenes_v29.yaml}"
SCENE_ID="${ISAAC_SCENE_ID:-office_localized}"
STAGE="${ISAAC_STAGE_USD:-}"

mkdir -p "${RUN_DIR}" "${BUILD_DIR}" "${LOG_DIR}" "${DEBUG_ROOT}"
: > "${MASTER_LOG}"

small=0
if [ "${1:-}" = "--small" ]; then
  small=1
fi

RUN_LIMIT="${TOPO_V29_RUN_LIMIT:-3}"
if [ "${small}" = "1" ]; then
  RUN_LIMIT=1
fi
MAX_STEPS="${TOPO_V29_MAX_STEPS:-60}"

export TOPO_DEBUG_CAPTURE="${TOPO_DEBUG_CAPTURE:-1}"
export TOPO_DEBUG_ONLY_ON_FAIL="${TOPO_DEBUG_ONLY_ON_FAIL:-0}"
export TOPO_DEBUG_FRAME_STRIDE="${TOPO_DEBUG_FRAME_STRIDE:-1}"
export TOPO_DEBUG_RINGBUF_STEPS="${TOPO_DEBUG_RINGBUF_STEPS:-220}"
export TOPO_DEBUG_PLACEHOLDER="${TOPO_DEBUG_PLACEHOLDER:-1}"
export TOPO_DEBUG_CAPTURE_SUCCESS_AT_MAXSTEPS="${TOPO_DEBUG_CAPTURE_SUCCESS_AT_MAXSTEPS:-1}"
export TOPO_DEBUG_ORACLE_ON_FAIL="${TOPO_DEBUG_ORACLE_ON_FAIL:-1}"
export ISAAC_RENDERER="${ISAAC_RENDERER:-rtx}"
export ISAAC_BACKEND_ALLOW_FALLBACK="${ISAAC_BACKEND_ALLOW_FALLBACK:-0}"
export ISAAC_RGB_CAPTURE=1
export ISAAC_RENDER=1
export ISAAC_SKIP_WORLD_STEP=0
export ISAAC_MINIMAL=0
export ISAAC_STAGE_USD="${STAGE}"

echo "[V29_ISAAC_SUITE] start=$(date -Iseconds) run_dir=${RUN_DIR}" | tee -a "${MASTER_LOG}"
python3 scripts/gpu/gpu_router.py --role isaac --output anchors | tee -a "${MASTER_LOG}"
python3 scripts/gpu/gpu_router.py --role habitat --output anchors | tee -a "${MASTER_LOG}"

echo "[V29_ISAAC_SUITE] camera_fidelity_begin" | tee -a "${MASTER_LOG}"
(
  bash scripts/isaac/camera_fidelity_v29.sh \
    --cfg "${CFG_PATH}" \
    --scene "${SCENE_ID}" \
    --stage "${STAGE}" \
    --out_dir "${RUN_DIR}/camera_fidelity"
) 2>&1 | tee -a "${MASTER_LOG}"

echo "[V29_ISAAC_SUITE] render_smoke_begin" | tee -a "${MASTER_LOG}"
(
  ISAAC_SMOKE_RUN_DIR="${RUN_DIR}" \
  ISAAC_CFG="${CFG_PATH}" \
  ISAAC_SCENE_ID="${SCENE_ID}" \
  ISAAC_STAGE_USD="${STAGE}" \
  bash scripts/isaac/render_smoke_v27.sh --stage "${STAGE}"
) 2>&1 | tee -a "${MASTER_LOG}"

gen_build() {
  local out_dir="$1"
  local scene="$2"
  python3 - "${out_dir}" "${scene}" <<'PY'
import json
import math
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
nodes, edges, embeds, clip_embeds = [], [], [], []
node_meta, node_to_room = {}, {}
room_palette = {0: (180, 70, 70), 1: (70, 180, 90), 2: (70, 90, 180), 3: (180, 160, 70)}

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
        rgb = np.zeros((160, 240, 3), dtype=np.uint8)
        rgb[..., 0] = int(room_palette[room_id][0] + 10 * c)
        rgb[..., 1] = int(room_palette[room_id][1] + 10 * r)
        rgb[..., 2] = int(room_palette[room_id][2])
        frame = out_dir / "frames" / f"node_{nid:04d}.png"
        Image.fromarray(rgb).save(frame)
        nodes.append({"node_id": nid, "step_idx": nid, "position": [float(x), 0.0, float(z)], "rotation": [1.0, 0.0, 0.0, 0.0], "embed_idx": nid, "timestamp": float(1000.0 + nid), "rgb_path": str(frame)})
        emb = mk_embed(x, z, room_id, dim=64)
        embeds.append(emb)
        clip_embeds.append(emb)
        node_meta[str(nid)] = {"embed_idx": nid, "clip_embed_idx": nid, "geo_idx": nid, "position": [float(x), 0.0, float(z)], "rotation": [1.0, 0.0, 0.0, 0.0], "room_id": int(room_id), "room_label": ["hallway", "office", "meeting_room", "kitchen"][room_id], "rgb_path": str(frame)}
        node_to_room[str(nid)] = int(room_id)
        nid += 1

def node_id(rr, cc): return rr * cols + cc
for r in range(rows):
    for c in range(cols):
        u = node_id(r, c)
        if c + 1 < cols: edges.append({"u": u, "v": node_id(r, c + 1), "w": spacing, "edge_type": "temporal"})
        if r + 1 < rows: edges.append({"u": u, "v": node_id(r + 1, c), "w": spacing, "edge_type": "temporal"})

graph = {"meta": {"scene_id": scene, "backend": "isaac_v29"}, "nodes": nodes, "edges": edges}
(out_dir / "graph.json").write_text(json.dumps(graph, indent=2))
np.save(out_dir / "node_embeds.npy", np.stack(embeds, axis=0).astype(np.float32))
np.save(out_dir / "node_embeds_clip.npy", np.stack(clip_embeds, axis=0).astype(np.float32))
(out_dir / "node_meta.json").write_text(json.dumps({"meta": {"scene_id": scene}, "node_meta": node_meta}, indent=2))
(out_dir / "node_to_room.json").write_text(json.dumps({"scene_id": scene, "node_to_room": node_to_room}, indent=2))
print(f"[TOPO_BUILD] nodes={len(nodes)} edges={len(edges)} out_dir={out_dir}")
PY
}

scene_build="${BUILD_DIR}/${SCENE_ID}"
mkdir -p "${scene_build}"
gen_build "${scene_build}" "${SCENE_ID}" | tee -a "${MASTER_LOG}"

run_count=0
cond="gt_pose"
scene_out="${RUN_DIR}/condition_${cond}/${SCENE_ID}"
mkdir -p "${scene_out}"
for start_offset in 0 1 2; do
  if [ "${run_count}" -ge "${RUN_LIMIT}" ]; then
    break
  fi
  run_id="${SCENE_ID}_q0_s${start_offset}_bearing"
  query='{"type":"pose","value":{"x":2.8,"y":0.0,"z":2.8,"yaw":0.0}}'
  echo "[TOPO_EVAL_RUN] backend=isaac condition=${cond} scene=${SCENE_ID} run_id=${run_id} start_offset=${start_offset}" | tee -a "${MASTER_LOG}"
  (
    bash scripts/gpu/run_isaac_on_5090.sh -- \
      bash scripts/topo_run_backend.sh \
        --backend isaac \
        --condition "${cond}" \
        --scene_id "${SCENE_ID}" \
        --run_id "${run_id}" \
        --query "${query}" \
        --graph_json "${scene_build}/graph.json" \
        --embeds_npy "${scene_build}/node_embeds.npy" \
        --clip_embeds_npy "${scene_build}/node_embeds_clip.npy" \
        --node_meta "${scene_build}/node_meta.json" \
        --out_dir "${scene_out}" \
        --debug_root "${DEBUG_ROOT}" \
        --isaac_config "${CFG_PATH}" \
        --config_path repo/InternNav/scripts/eval/configs/vln_r2r.yaml \
        --controller_mode waypoint_follow \
        --embedder_mode simple \
        --simple_embed_dim 64 \
        --max_steps "${MAX_STEPS}" \
        --goal_topk 3 \
        --start_episode_offset "${start_offset}" \
        --kidnap_start 1 \
        --kidnap_seed "$((29000 + run_count))" \
        --kidnap_min_nearest_node_dist 0.2 \
        --kidnap_max_nearest_node_dist 3.5 \
        --node_reach_thresh 1.2 \
        --planner_grid_res 0.25 \
        --waypoint_reach_thresh 0.45 \
        --debug_capture "${TOPO_DEBUG_CAPTURE}" \
        --debug_only_on_fail "${TOPO_DEBUG_ONLY_ON_FAIL}" \
        --debug_frame_stride "${TOPO_DEBUG_FRAME_STRIDE}" \
        --debug_ringbuf_steps "${TOPO_DEBUG_RINGBUF_STEPS}" \
        --debug_save_depth "${TOPO_DEBUG_SAVE_DEPTH:-0}" \
        --debug_oracle_on_fail "${TOPO_DEBUG_ORACLE_ON_FAIL}" \
        --debug_capture_success_at_maxsteps "${TOPO_DEBUG_CAPTURE_SUCCESS_AT_MAXSTEPS}" \
        --debug_placeholder "${TOPO_DEBUG_PLACEHOLDER}"
  ) >> "${MASTER_LOG}" 2>&1
  run_count=$((run_count + 1))
done

while IFS= read -r meta_path; do
  run_dir="$(dirname "${meta_path}")"
  python3 scripts/topo_debug_ui_build_report.py --run_dir "${run_dir}" --build_root "${BUILD_DIR}" >> "${MASTER_LOG}" 2>&1 || true
done < <(find "${DEBUG_ROOT}" -type f -name "meta.json" | sort)
python3 scripts/topo_debug_ui_index.py --root "${DEBUG_ROOT}" >> "${MASTER_LOG}" 2>&1 || true
echo "[TOPO_DEBUG_INDEX_OK] path=${DEBUG_ROOT}/index.html" | tee -a "${MASTER_LOG}"

summary_json="${RUN_DIR}/v29_summary_tmp.json"
python3 - "${scene_out}" "${summary_json}" <<'PY'
import glob, json, os, sys
scene_out, summary_json = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(scene_out, "RESULT_*.json"))):
    try:
        rows.append(json.load(open(p)))
    except Exception:
        continue
succ = sum(1 for r in rows if bool(r.get("success", False)))
renderer = rows[0].get("renderer_used", "unknown") if rows else "unknown"
json.dump({"runs": len(rows), "success": succ, "renderer": renderer}, open(summary_json, "w"), indent=2)
PY
runs="$(python3 - <<'PY' "${summary_json}"
import json,sys
d=json.load(open(sys.argv[1]))
print(int(d.get("runs",0)))
PY
)"
succ="$(python3 - <<'PY' "${summary_json}"
import json,sys
d=json.load(open(sys.argv[1]))
print(int(d.get("success",0)))
PY
)"
renderer="$(python3 - <<'PY' "${summary_json}"
import json,sys
d=json.load(open(sys.argv[1]))
print(str(d.get("renderer","unknown")))
PY
)"
rm -f "${summary_json}"

echo "[TOPO_EVAL_SUMMARY_V29_ISAAC] scene=${SCENE_ID} runs=${runs} success=${succ} renderer=${renderer}" | tee -a "${MASTER_LOG}"

exit 0
