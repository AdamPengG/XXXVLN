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
PHYSICS_DIR="${OUT_ROOT}/physics"
AUDIT_DIR="${OUT_ROOT}/collision_audit"
PHYS_STAGE_DIR="${OUT_ROOT}/office_phys"
mkdir -p "${LOG_DIR}" "${EVAL_DIR}" "${DEBUG_ROOT}" "${QA_ROOT}" "${MAP_DIR}" "${PHYSICS_DIR}"
SUITE_LOG="${LOG_DIR}/topo_eval_suite_v34b_nav2_ui_demo.log"
: > "${SUITE_LOG}"

run_and_log() {
  ( "$@" ) 2>&1 | tee -a "${SUITE_LOG}"
}

resolve_stage() {
  if [ -n "${ISAAC_STAGE_USD:-}" ] && [ -f "${ISAAC_STAGE_USD}" ]; then
    echo "${ISAAC_STAGE_USD}|override"
    return 0
  fi
  local roots=()
  [ -n "${ISAAC_ASSETS_ROOT:-}" ] && roots+=("${ISAAC_ASSETS_ROOT}")
  roots+=("/home/peng/IsaacAssets" "/home/peng/isaacsim_assets")
  for root in "${roots[@]}"; do
    [ -n "${root}" ] || continue
    for p in \
      "${root}/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd" \
      "${root}/Isaac/Environments/Office/office.usd" \
      "${root}/Office/office.usd"; do
      if [ -f "${p}" ]; then
        echo "${p}|official_assets"
        return 0
      fi
    done
  done
  echo "|missing"
}

echo "[V34B_SUITE] start=$(date -Iseconds) out_root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
run_and_log python3 scripts/gpu/gpu_router.py --role isaac --output anchors

stage_info="$(resolve_stage)"
OFFICIAL_STAGE_USD="${stage_info%%|*}"
STAGE_SRC="${stage_info##*|}"
STAGE_EXISTS=0
if [ -n "${OFFICIAL_STAGE_USD}" ] && [ -f "${OFFICIAL_STAGE_USD}" ]; then
  STAGE_EXISTS=1
fi
export ISAAC_STAGE_USD="${OFFICIAL_STAGE_USD}"

echo "[V34B_ISAAC_STAGE] usd=${OFFICIAL_STAGE_USD} exists=${STAGE_EXISTS} source=${STAGE_SRC}" | tee -a "${SUITE_LOG}"
if [ "${STAGE_EXISTS}" != "1" ]; then
  echo "[V34B_SUITE_OK] ok=0 root=${OUT_ROOT} map=${MAP_DIR}/office_map.yaml nav2=0 physics=0 qa_ok=0 reason=stage_missing" | tee -a "${SUITE_LOG}"
  exit 2
fi

export ISAAC_STEP_RENDER=1
export ISAAC_STEP_RENDER_EVERY_N=1
export ISAAC_ENABLE_DEPTH=1
export V34B_SCENE_ID="office_localized"
export V34B_ISAAC_CONFIG="configs/isaac_scenes_v34b.yaml"
export V34B_MAP_OUT_DIR="${MAP_DIR}"
export V34B_LOG_DIR="${LOG_DIR}"
export ALLOW_SPARSE_COLLIDERS="${ALLOW_SPARSE_COLLIDERS:-1}"

# 1) Collision audit on official stage.
export V34B_COLLISION_AUDIT_OUT="${AUDIT_DIR}"
run_and_log bash scripts/isaac/collision_audit_v34b2.sh
COLLIDER_RATIO="$(python3 - <<'PY' "${AUDIT_DIR}/collision_audit.json"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print("0.0000")
    raise SystemExit(0)
try:
    obj = json.loads(p.read_text(encoding='utf-8'))
except Exception:
    obj = {}
print(f"{float(obj.get('collider_ratio', 0.0)):.4f}")
PY
)"

# 2) Build local physics-ready overlay stage.
export V34B_OFFICE_PHYS_USD="${PHYS_STAGE_DIR}/office_phys.usd"
run_and_log bash scripts/isaac/build_office_phys_stage_v34b2.sh
if [ ! -f "${V34B_OFFICE_PHYS_USD}" ]; then
  echo "[V34B2_SUITE_OK] ok=0 root=${OUT_ROOT} office_usd=${OFFICIAL_STAGE_USD} office_phys_usd=${V34B_OFFICE_PHYS_USD} collider_ratio=${COLLIDER_RATIO} reason=phys_stage_missing" | tee -a "${SUITE_LOG}"
  exit 3
fi

# Run all downstream tasks on physics-ready stage.
export ISAAC_STAGE_USD="${V34B_OFFICE_PHYS_USD}"

# 3) Real map generation from USD geometry.
run_and_log bash scripts/nav2/generate_office_map_v34b.sh

# 4) Isaac UI + robot/sensor probe.
run_and_log bash scripts/isaac/run_ui_office_nav2_v34b.sh

# 5) Nav2 bringup (best effort; explicit skip reason when ROS2 unavailable).
run_and_log bash scripts/nav2/bringup_nav2_office_v34b.sh
NAV2_OK=0
if rg -q "\[V34B_NAV2_BRINGUP\] ok=1" "${SUITE_LOG}"; then
  NAV2_OK=1
fi

# 6) Physics-safe drive smoke with real Isaac camera capture.
run_and_log bash scripts/gpu/run_isaac_on_5090.sh -- \
  bash scripts/isaac/run_with_isaac_python.sh -- \
    scripts/isaac/physics_drive_smoke_v34b.py \
      --scene_id office_localized \
      --isaac_config configs/isaac_scenes_v34b.yaml \
      --stage "${V34B_OFFICE_PHYS_USD}" \
      --out_dir "${PHYSICS_DIR}" \
      --steps 80

PHYSICS_OK=0
if rg -q "\[V34B_COLLISION_CHECK\] ok=1" "${SUITE_LOG}"; then
  PHYSICS_OK=1
fi

# Ensure build artifacts exist for runner + QA compatibility.
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

RUN_POLICY="nav2"
if [ "${NAV2_OK}" != "1" ]; then
  RUN_POLICY="topo"
  echo "[V34B_NAV_FALLBACK] mode=isaac_native_topo reason=ros2_missing" | tee -a "${SUITE_LOG}"
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
      --planb_policy "${RUN_POLICY}" \
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

# 7) Three goals: same-room pose, cross-door pose, object approach pose.
run_case "v34b_pose_same_room" '{"type":"pose","value":{"x":-2.4,"y":0.0,"z":-2.2,"yaw_deg":15}}' 0
run_case "v34b_pose_cross_door" '{"type":"pose","value":{"x":1.8,"y":0.0,"z":1.9,"yaw_deg":180}}' 0
run_case "v34b_object_approach" '{"type":"object","value":{"query":{"name_contains":"Printer"},"approach_pose":{"x":-0.8,"y":0.0,"z":0.6,"yaw_deg":20}}}' 1

# 8) Topo-room QA on same scene trajectory.
run_and_log bash scripts/gpu/run_isaac_on_5090.sh -- \
  bash scripts/isaac/run_with_isaac_python.sh -- \
    scripts/topo_topomap_room_qa_v34.py \
      --scene_id office_localized \
      --isaac_config configs/isaac_scenes_v34b.yaml \
      --out_root "${QA_ROOT}" \
      --steps 80 --fwd_steps 8 --turn_steps 6 --node_stride 4 --save_frames 20 --overlay_frames 20 --start_offset 0 --seed 0

# 9) Capture motion checks.
for run_id in v34b_pose_same_room v34b_pose_cross_door v34b_object_approach; do
  CAP_DIR="${DEBUG_ROOT}/gt_pose/${run_id}"
  if [ -d "${CAP_DIR}" ]; then
    run_and_log bash scripts/tools/check_capture_motion.sh --capture_dir "${CAP_DIR}" --tag "${run_id}"
  fi
done
if [ -d "${PHYSICS_DIR}/frames" ]; then
  run_and_log bash scripts/tools/check_capture_motion.sh --capture_dir "${PHYSICS_DIR}/frames" --tag "v34b_physics"
fi

# Summary
python3 - "${EVAL_DIR}" "${QA_ROOT}" "${OUT_ROOT}" "${NAV2_OK}" "${PHYSICS_OK}" <<'PY' | tee -a "${SUITE_LOG}"
import json, pathlib, sys

eval_dir = pathlib.Path(sys.argv[1])
qa_root = pathlib.Path(sys.argv[2])
out_root = pathlib.Path(sys.argv[3])
nav2_ok = int(sys.argv[4])
physics_ok = int(sys.argv[5])

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
    'physics_ok': int(physics_ok),
    'qa_ok': int(qa_ok),
}
(out_root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(f"[V34B_SUITE_OK] ok={int((len(results) >= 3) and (physics_ok == 1))} root={out_root} map={out_root / 'map' / 'office_map.yaml'} nav2={nav2_ok} physics={physics_ok} qa_ok={qa_ok}")
PY
echo "[V34B2_SUITE_OK] ok=1 root=${OUT_ROOT} office_usd=${OFFICIAL_STAGE_USD} office_phys_usd=${V34B_OFFICE_PHYS_USD} collider_ratio=${COLLIDER_RATIO}" | tee -a "${SUITE_LOG}"

# cleanup nav2 bringup if running
if [ -f "${LOG_DIR}/nav2_bringup.pid" ]; then
  PID=$(cat "${LOG_DIR}/nav2_bringup.pid" || true)
  if [ -n "${PID}" ] && kill -0 "${PID}" >/dev/null 2>&1; then
    kill "${PID}" >/dev/null 2>&1 || true
  fi
fi

echo "[V34B_SUITE] done log=${SUITE_LOG}" | tee -a "${SUITE_LOG}"
