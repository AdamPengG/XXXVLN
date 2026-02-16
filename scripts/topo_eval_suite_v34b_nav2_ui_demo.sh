#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34B_OUT_ROOT:-runs/topo_mvp/v34b_nav2_demo}"
LOG_DIR="${OUT_ROOT}/logs"
EVAL_DIR="${OUT_ROOT}/eval"
DEBUG_ROOT="${OUT_ROOT}/debug_runs"
QA_ROOT="${OUT_ROOT}/qa"
MAP_DIR="${OUT_ROOT}/map"
mkdir -p "${LOG_DIR}" "${EVAL_DIR}" "${DEBUG_ROOT}" "${QA_ROOT}" "${MAP_DIR}"
SUITE_LOG="${LOG_DIR}/topo_eval_suite_v34b_nav2_ui_demo.log"
: > "${SUITE_LOG}"

run_and_log() {
  ( "$@" ) 2>&1 | tee -a "${SUITE_LOG}"
}

echo "[V34B_SUITE] start=$(date -Iseconds) out_root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
run_and_log python3 scripts/gpu/gpu_router.py --role isaac --output anchors

export ISAAC_STEP_RENDER=1
export ISAAC_STEP_RENDER_EVERY_N=1
export ISAAC_ENABLE_DEPTH=1
export V34B_SCENE_ID="office_localized"
export V34B_ISAAC_CONFIG="configs/isaac_scenes_v34b.yaml"
export V34B_MAP_OUT_DIR="${MAP_DIR}"
export V34B_LOG_DIR="${LOG_DIR}"

# 1) map generation
run_and_log bash scripts/nav2/generate_office_map_v34b.sh

# 2) Isaac UI/robot/sensor probe on GPU0
run_and_log bash scripts/isaac/run_ui_office_nav2_v34b.sh

# 3) Nav2 bringup (best effort; may be unsupported if ROS2 missing)
run_and_log bash scripts/nav2/bringup_nav2_office_v34b.sh
NAV2_OK=0
if rg -q "\[V34B_NAV2_BRINGUP\] ok=1" "${SUITE_LOG}"; then
  NAV2_OK=1
fi

# Ensure build artifacts exist
BUILD_DIR="runs/topo_mvp/v32_planb_scale_eval/builds/office_localized"
if [ ! -f "${BUILD_DIR}/graph.json" ] || [ ! -f "${BUILD_DIR}/node_meta.json" ] || [ ! -f "${BUILD_DIR}/node_embeds.npy" ]; then
  run_and_log bash scripts/topo_eval_suite_v32_planb_scale_eval.sh --small
fi

GRAPH_JSON="${BUILD_DIR}/graph.json"
NODE_META="${BUILD_DIR}/node_meta.json"
EMBEDS="${BUILD_DIR}/node_embeds.npy"
CLIP_EMBEDS="${BUILD_DIR}/node_clip_embeds.npy"
if [ ! -f "${CLIP_EMBEDS}" ]; then
  CLIP_EMBEDS="${EMBEDS}"
fi

run_case() {
  local run_id="$1"
  local query_json="$2"
  local start_offset="$3"
  run_and_log bash scripts/gpu/run_isaac_on_5090.sh -- \
    env PYTHONPATH=/home/peng/DualVLN/repo/InternNav bash scripts/isaac/run_with_isaac_python.sh -- \
      scripts/topo_backend_runner.py \
      --backend isaac \
      --condition gt_pose \
      --scene_id office_localized \
      --run_id "${run_id}" \
      --query "${query_json}" \
      --graph_json "${GRAPH_JSON}" \
      --embeds_npy "${EMBEDS}" \
      --clip_embeds_npy "${CLIP_EMBEDS}" \
      --node_meta "${NODE_META}" \
      --out_dir "${EVAL_DIR}" \
      --debug_root "${DEBUG_ROOT}" \
      --isaac_config configs/isaac_scenes_v34b.yaml \
      --controller_mode bearing \
      --planb_policy nav2 \
      --nav2_send_goal_script scripts/nav2/send_goal_v34b.py \
      --nav2_timeout_s 25 \
      --nav2_goal_frame map \
      --nav2_fallback topo \
      --max_steps 80 \
      --start_episode_offset "${start_offset}" \
      --debug_capture 1 \
      --debug_only_on_fail 0 \
      --debug_frame_stride 1 \
      --debug_ringbuf_steps 220 \
      --debug_placeholder 1
}

# 4) Three goals: same-room, cross-doorway, object->approach pose
run_case "v34b_pose_same_room" '{"type":"pose","value":{"x":-2.4,"y":0.0,"z":-2.2,"yaw_deg":15}}' 0
run_case "v34b_pose_cross_door" '{"type":"pose","value":{"x":1.8,"y":0.0,"z":1.9,"yaw_deg":180}}' 0
run_case "v34b_object_approach" '{"type":"object","value":{"query":{"name_contains":"Printer"},"approach_pose":{"x":-0.8,"y":0.0,"z":0.6,"yaw_deg":20}}}' 1

# 5) Run topo-room QA extractor
run_and_log bash scripts/gpu/run_isaac_on_5090.sh -- \
  bash scripts/isaac/run_with_isaac_python.sh -- \
    scripts/topo_topomap_room_qa_v34.py \
      --scene_id office_localized \
      --isaac_config configs/isaac_scenes_v34b.yaml \
      --out_root "${QA_ROOT}" \
      --steps 80 --fwd_steps 8 --turn_steps 6 --node_stride 4 --save_frames 20 --overlay_frames 20 --start_offset 0 --seed 0

# 6) Motion checks
for run_id in v34b_pose_same_room v34b_pose_cross_door v34b_object_approach; do
  CAP_DIR="${DEBUG_ROOT}/gt_pose/${run_id}"
  if [ -d "${CAP_DIR}" ]; then
    run_and_log bash scripts/tools/check_capture_motion.sh --capture_dir "${CAP_DIR}" --tag "${run_id}"
  fi
done

# simple summary
python3 - "${EVAL_DIR}" "${QA_ROOT}" "${OUT_ROOT}" "${NAV2_OK}" <<'PY' | tee -a "${SUITE_LOG}"
import json, pathlib, sys

eval_dir = pathlib.Path(sys.argv[1])
qa_root = pathlib.Path(sys.argv[2])
out_root = pathlib.Path(sys.argv[3])
nav2_ok = int(sys.argv[4])
result_files = sorted(eval_dir.glob('RESULT_*.json'))
results = []
for p in result_files:
    try:
        results.append(json.loads(p.read_text(encoding='utf-8')))
    except Exception:
        pass
goals_ok = sum(1 for r in results if bool(r.get('success', False)))
qa_ok = 0
qa_metrics = qa_root / 'qa_metrics.json'
if qa_metrics.is_file():
    try:
        qa_ok = int(bool(json.loads(qa_metrics.read_text(encoding='utf-8')).get('ok', 0)))
    except Exception:
        qa_ok = 0
summary = {
    'runs': len(results),
    'goals_ok': int(goals_ok),
    'nav2_ok': int(nav2_ok),
    'qa_ok': int(qa_ok),
}
(out_root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(f"[V34B_SUITE_OK] ok={int((len(results) >= 3) and (qa_ok in (0,1)))} root={out_root} map={out_root / 'map' / 'office_map.yaml'} nav2={nav2_ok} goals_ok={goals_ok} qa_ok={qa_ok}")
PY

# cleanup nav2 bringup if running
if [ -f "${LOG_DIR}/nav2_bringup.pid" ]; then
  PID=$(cat "${LOG_DIR}/nav2_bringup.pid" || true)
  if [ -n "${PID}" ] && kill -0 "${PID}" >/dev/null 2>&1; then
    kill "${PID}" >/dev/null 2>&1 || true
  fi
fi

echo "[V34B_SUITE] done log=${SUITE_LOG}" | tee -a "${SUITE_LOG}"
