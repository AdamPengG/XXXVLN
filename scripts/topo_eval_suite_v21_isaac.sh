#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
RUN_DIR="${ROOT_DIR}/runs/topo_mvp/v21_isaac"
BUILD_DIR="${RUN_DIR}/builds"
LOG_DIR="${RUN_DIR}/logs"
DEBUG_ROOT="${RUN_DIR}/debug_runs"
CONFIG_PATH="${ROOT_DIR}/configs/isaac_scenes_v21.yaml"
BUILD_SOURCE="${TOPO_V21_BUILD_SOURCE:-${ROOT_DIR}/runs/topo_mvp/v20/builds}"

mkdir -p "${RUN_DIR}" "${BUILD_DIR}" "${LOG_DIR}" "${DEBUG_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export VLN_SPLIT=train
export PYTHONHASHSEED=0

MAX_STEPS="${TOPO_V21_MAX_STEPS:-120}"
NODE_REACH_THRESH="${TOPO_V21_NODE_REACH_THRESH:-0.8}"
GOAL_TOPK="${TOPO_V21_GOAL_TOPK:-5}"
START_LIMIT="${TOPO_V21_START_LIMIT:-1}"
SCENE_LIMIT="${TOPO_V21_SCENE_LIMIT:-1}"
RUN_LIMIT="${TOPO_V21_RUN_LIMIT:-0}"
CONDITIONS="${TOPO_V21_CONDITIONS:-gt_pose,est_pose_noise}"

load_cfg_json() {
  python - "${CONFIG_PATH}" <<'PY'
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

mapfile -t scenes < <(python - "${CFG_JSON}" "${SCENE_LIMIT}" <<'PY'
import json, sys
cfg=json.loads(sys.argv[1])
lim=int(sys.argv[2])
rows=[str(x.get("scene_id","")).strip() for x in cfg.get("scenes",[]) if str(x.get("scene_id","")).strip()]
if lim>0:
    rows=rows[:lim]
for r in rows:
    print(r)
PY
)
mapfile -t queries < <(python - "${CFG_JSON}" <<'PY'
import json, sys
cfg=json.loads(sys.argv[1])
rows=[str(x) for x in cfg.get("global",{}).get("queries",[])]
for r in rows:
    print(r)
PY
)
mapfile -t starts < <(python - "${CFG_JSON}" "${START_LIMIT}" <<'PY'
import json, sys
cfg=json.loads(sys.argv[1])
lim=int(sys.argv[2])
rows=[int(x) for x in cfg.get("global",{}).get("start_offsets",[])]
if lim>0:
    rows=rows[:lim]
for r in rows:
    print(r)
PY
)

if [ "${#scenes[@]}" -eq 0 ]; then
  echo "No scenes configured in ${CONFIG_PATH}" >&2
  exit 2
fi
if [ "${#queries[@]}" -eq 0 ]; then
  echo "No queries configured in ${CONFIG_PATH}" >&2
  exit 2
fi
if [ "${#starts[@]}" -eq 0 ]; then
  echo "No start offsets configured in ${CONFIG_PATH}" >&2
  exit 2
fi

for scene in "${scenes[@]}"; do
  src="${BUILD_SOURCE}/${scene}"
  dst="${BUILD_DIR}/${scene}"
  rm -rf "${dst}"
  if [ -d "${src}" ]; then
    cp -a "${src}" "${dst}"
  else
    echo "Missing source build for scene=${scene} at ${src}" >&2
    exit 2
  fi
done

IFS=',' read -r -a conds <<< "${CONDITIONS}"

total_runs=0
for cond in "${conds[@]}"; do
  cond_dir="${RUN_DIR}/condition_${cond}"
  cond_log="${LOG_DIR}/topo_eval_suite_v21_isaac_${cond}.log"
  rm -rf "${cond_dir}"
  mkdir -p "${cond_dir}"
  : > "${cond_log}"

  run_count=0
  for scene in "${scenes[@]}"; do
    scene_out="${cond_dir}/${scene}"
    mkdir -p "${scene_out}"
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
        echo "[TOPO_EVAL_RUN] backend=isaac condition=${cond} scene=${scene} run_id=${run_id} query=\"${query}\" start_offset=${start_offset}" | tee -a "${LOG_DIR}/topo_eval_suite_v21_isaac.log"
        (
          cd "${ROOT_DIR}"
          python scripts/topo_backend_runner.py \
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
            --isaac_config "${CONFIG_PATH}" \
            --config_path repo/InternNav/scripts/eval/configs/vln_r2r.yaml \
            --model_path checkpoints/InternVLA-N1-DualVLN \
            --pooling mean \
            --controller_mode bearing \
            --max_steps "${MAX_STEPS}" \
            --goal_topk "${GOAL_TOPK}" \
            --start_episode_offset "${start_offset}" \
            --kidnap_start 1 \
            --kidnap_seed "$((1044 + run_count))" \
            --kidnap_min_nearest_node_dist 0.5 \
            --kidnap_max_nearest_node_dist 3.5 \
            --node_reach_thresh "${NODE_REACH_THRESH}" \
            --debug_capture "${TOPO_DEBUG_CAPTURE:-0}" \
            --debug_only_on_fail "${TOPO_DEBUG_ONLY_ON_FAIL:-1}" \
            --debug_frame_stride "${TOPO_DEBUG_FRAME_STRIDE:-3}" \
            --debug_ringbuf_steps "${TOPO_DEBUG_RINGBUF_STEPS:-220}" \
            --debug_save_depth "${TOPO_DEBUG_SAVE_DEPTH:-0}" \
            --debug_oracle_on_fail "${TOPO_DEBUG_ORACLE_ON_FAIL:-1}" \
            --debug_capture_success_at_maxsteps "${TOPO_DEBUG_CAPTURE_SUCCESS_AT_MAXSTEPS:-1}"
        ) >> "${cond_log}" 2>&1
        run_count=$((run_count + 1))
      done
    done
  done

  cond_summary="${RUN_DIR}/condition_${cond}_summary.json"
  cond_fail_tax="${RUN_DIR}/condition_${cond}_failure_taxonomy.json"
  python - "${cond_dir}" "${cond}" "${cond_summary}" "${cond_fail_tax}" <<'PY'
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
scores = [float(r.get("goal_score", 0.0)) for r in rows]
fails = [r for r in rows if not bool(r.get("success", False))]
failed_max_steps = sum(1 for r in rows if (not bool(r.get("success", False))) and str(r.get("terminated_by", "")) == "max_steps")
terminated_max_steps = sum(1 for r in rows if str(r.get("terminated_by", "")) == "max_steps")
tax = Counter([str(r.get("fail_reason", "unknown")) for r in fails])
summary = {
    "condition": cond,
    "runs": runs,
    "success": succ,
    "avg_steps": float(statistics.mean(steps) if steps else 0.0),
    "avg_loc_conf": float(statistics.mean(locs) if locs else 0.0),
    "avg_goal_score": float(statistics.mean(scores) if scores else 0.0),
    "failed_max_steps": int(failed_max_steps),
    "terminated_max_steps": int(terminated_max_steps),
}
json.dump(summary, open(out_summary, "w"), indent=2)
json.dump(dict(tax), open(out_fail, "w"), indent=2)
print(
    f"[TOPO_EVAL_SUMMARY_V21_ISAAC] condition={cond} runs={runs} success={succ} "
    f"avg_steps={summary['avg_steps']:.2f} avg_loc_conf={summary['avg_loc_conf']:.3f} "
    f"avg_goal_score={summary['avg_goal_score']:.3f} failed_max_steps={failed_max_steps} "
    f"terminated_max_steps={terminated_max_steps}"
)
PY
  total_runs=$((total_runs + run_count))
done | tee -a "${LOG_DIR}/topo_eval_suite_v21_isaac.log"

if [ "${TOPO_DEBUG_CAPTURE:-0}" = "1" ]; then
  while IFS= read -r run_dir; do
    if [ -f "${run_dir}/trace.jsonl" ] && [ -f "${run_dir}/meta.json" ]; then
      (
        cd "${ROOT_DIR}"
        python scripts/topo_debug_ui_build_report.py \
          --run_dir "${run_dir}" \
          --build_root "${BUILD_DIR}"
      ) >> "${LOG_DIR}/topo_eval_suite_v21_isaac.log" 2>&1 || true
    fi
  done < <(find "${DEBUG_ROOT}" -mindepth 2 -maxdepth 2 -type d | sort)
fi

(
  cd "${ROOT_DIR}"
  python scripts/topo_debug_ui_index.py --root "${DEBUG_ROOT}"
) >> "${LOG_DIR}/topo_eval_suite_v21_isaac.log" 2>&1 || true
echo "[TOPO_DEBUG_INDEX_OK] path=${DEBUG_ROOT}/index.html" | tee -a "${LOG_DIR}/topo_eval_suite_v21_isaac.log"

python - "${RUN_DIR}" <<'PY'
import json
import os
import glob
import statistics
import sys

run_dir = sys.argv[1]
summaries = {}
for p in sorted(glob.glob(os.path.join(run_dir, "condition_*_summary.json"))):
    key = os.path.basename(p).replace("condition_", "").replace("_summary.json", "")
    summaries[key] = json.load(open(p))
out = {"conditions": summaries}
if summaries:
    out["metrics"] = {f"{k}_success": int(v.get("success", 0)) for k, v in summaries.items()}
json.dump(out, open(os.path.join(run_dir, "v21_isaac_summary.json"), "w"), indent=2)
print(f"[V21_ISAAC] wrote {os.path.join(run_dir, 'v21_isaac_summary.json')}")
PY
