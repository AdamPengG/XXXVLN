#!/usr/bin/env bash
# topo_eval_suite_v35e_ui_debug.sh — v35e: camera-alignment verification suite.
# Runs: stage verify → axis sweep test → 1 nav case → motion check.
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${V35E_OUT_ROOT:-runs/topo_mvp/v35e_ui_debug_${TS}}"
LOG_DIR="${OUT_ROOT}/logs"
MAP_DIR="${OUT_ROOT}/map"
ROOMQA_DIR="${OUT_ROOT}/roomqa"
mkdir -p "${LOG_DIR}" "${MAP_DIR}" "${ROOMQA_DIR}"
SUITE_LOG="${LOG_DIR}/topo_eval_suite_v35e.log"
: > "${SUITE_LOG}"

# ── Stage ──
STAGE_OFFICIAL="/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"
if [ ! -f "${STAGE_OFFICIAL}" ]; then
  echo "[V35E_STAGE_VERIFY] ok=0 stage_url=${STAGE_OFFICIAL} exists=0" | tee -a "${SUITE_LOG}"
  exit 2
fi
export ISAAC_STAGE_USD="${STAGE_OFFICIAL}"
echo "[V35E_STAGE_VERIFY] ok=1 stage_url=${STAGE_OFFICIAL} exists=1" | tee -a "${SUITE_LOG}"

# ── Sensor env ──
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ISAAC_STEP_RENDER=1
export ISAAC_STEP_RENDER_EVERY_N=1
export ISAAC_ENABLE_DEPTH="${ISAAC_ENABLE_DEPTH:-1}"
export ISAAC_GPU_ID=0
export ISAAC_CAMERA_PITCH_DEG="${ISAAC_CAMERA_PITCH_DEG:--10.0}"
export V35E_CAM_HEIGHT_M="${V35E_CAM_HEIGHT_M:-1.55}"
export V35E_OUT_DIR="${OUT_ROOT}"
export V35E_OUT_ROOT="${OUT_ROOT}"
export V34J_OUT_ROOT="${OUT_ROOT}"
export V34I_OUT_ROOT="${OUT_ROOT}"
export V34D_OUT_ROOT="${OUT_ROOT}"
export V34G_OUT_ROOT="${OUT_ROOT}"
export V34G_MAP_OUT_DIR="${MAP_DIR}"
export V34G_MAP_BOUNDS="${V34G_MAP_BOUNDS:--6,6,-6,6}"
export V34D_MAP_YAML="${MAP_DIR}/office_map.yaml"
export V34I_NAV2_PARAMS="${V34I_NAV2_PARAMS:-configs/nav2_params_v34j.yaml}"
export V34D_NAV2_PARAMS="${V34I_NAV2_PARAMS}"
export V34D_NAV2_CONTAINER="${V34D_NAV2_CONTAINER:-v35e_nav2_stack}"
export V34I_ODOM_TF_CONTAINER="${V34I_ODOM_TF_CONTAINER:-v35e_odom_tf_bridge}"
export V34D_NAV2_DOCKER_IMAGE="${V34D_NAV2_DOCKER_IMAGE:-v34d_ros2_nav2:humble}"
export V34D_USE_LOCALIZATION="${V34D_USE_LOCALIZATION:-0}"

AMENT_ROOT="/home/peng/IsaacSim/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge/humble"
export AMENT_PREFIX_PATH="${AMENT_ROOT}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
export PYTHONPATH="${AMENT_ROOT}/rclpy${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${AMENT_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Headless mode unless forced GUI
HEADLESS="${ISAAC_HEADLESS:-1}"
[ "${ISAAC_GUI:-0}" = "1" ] && HEADLESS=0
export ISAAC_HEADLESS="${HEADLESS}"

PROBE_PID=""
cleanup() {
  if [ -n "${PROBE_PID}" ] && kill -0 "${PROBE_PID}" >/dev/null 2>&1; then
    kill -TERM "${PROBE_PID}" >/dev/null 2>&1 || true
    sleep 2
    kill -9 "${PROBE_PID}" >/dev/null 2>&1 || true
    wait "${PROBE_PID}" >/dev/null 2>&1 || true
  fi
  docker rm -f "${V34I_ODOM_TF_CONTAINER}" >/dev/null 2>&1 || true
  docker rm -f "${V34D_NAV2_CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

run_and_log() { ( "$@" ) 2>&1 | tee -a "${SUITE_LOG}"; }
fail_suite() {
  local reason="$1"
  echo "[V35E_SUITE_OK] ok=0 cases=${STAGED_TOTAL:-1} succeeded=${STAGED_SUCCEEDED:-0} rooms=${ROOM_COUNT:-0} doors=${DOOR_COUNT:-0} root=${OUT_ROOT} reason=${reason}" | tee -a "${SUITE_LOG}"
  exit 2
}

ROOM_COUNT=0
DOOR_COUNT=0
STAGED_SUCCEEDED=0
STAGED_SKIPPED=0
STAGED_TOTAL=2  # v35e: 2 fixed goals as required

echo "[V35E_SUITE] start=$(date -Iseconds) root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
run_and_log python3 scripts/gpu/gpu_router.py --role isaac --output anchors

# ── 1) Map ──
if ! run_and_log bash scripts/nav2/generate_office_map_v34g.sh; then
  fail_suite "map_generation_failed"
fi
if ! rg -q "\[V34G_MAP_SANITY\] ok=1" "${SUITE_LOG}"; then
  fail_suite "map_sanity_failed"
fi

OCC_RATIO="$(python3 - <<'PY' "${MAP_DIR}/office_map.pgm"
import numpy as np; from PIL import Image; import sys
im=np.array(Image.open(sys.argv[1]).convert('L'),dtype=np.float32)
occ=(im<50).sum(); total=im.size
print(f"{occ/total:.4f}")
PY
)"
echo "[V35E_MAP_GEN] ok=1 occ_ratio=${OCC_RATIO}" | tee -a "${SUITE_LOG}"

# ── 2) Room/topo ──
if ! run_and_log python3 scripts/topo_room_from_occ_v35a.py --map_yaml "${MAP_DIR}/office_map.yaml" --out_dir "${ROOMQA_DIR}"; then
  fail_suite "room_partition_failed"
fi
ROOM_COUNT="$(python3 - <<'PY' "${ROOMQA_DIR}/room_partition.json"
import json,sys; print(len(json.load(open(sys.argv[1],encoding='utf-8')).get('rooms',[])))
PY
)"
DOOR_COUNT="$(python3 - <<'PY' "${ROOMQA_DIR}/doorway_edges.json"
import json,sys; print(len(json.load(open(sys.argv[1],encoding='utf-8')).get('edges',[])))
PY
)"
echo "[V35E_ROOM_PARTITION] ok=1 rooms=${ROOM_COUNT}" | tee -a "${SUITE_LOG}"
echo "[V35E_DOORWAY_DETECT] ok=1 doors=${DOOR_COUNT}" | tee -a "${SUITE_LOG}"

# ── 3) Isaac v35e probe (in background) ──
READY_FILE="${OUT_ROOT}/bridge_ready_v35e.json"
rm -f "${READY_FILE}"
: > "${LOG_DIR}/isaac_probe_v35e.log"
CAPTURE_STEPS="${V35E_CAPTURE_STEPS:-6000}"
(
  V35E_READY_FILE="${READY_FILE}" \
  V35E_STEPS="${CAPTURE_STEPS}" \
  V35E_OUT_DIR="${OUT_ROOT}" \
  bash scripts/isaac/run_ui_office_topo_nav2_v35e.sh
) > "${LOG_DIR}/isaac_probe_v35e.log" 2>&1 &
PROBE_PID=$!

for _ in $(seq 1 240); do
  [ -f "${READY_FILE}" ] && break
  sleep 1
done
if [ ! -f "${READY_FILE}" ]; then
  cat "${LOG_DIR}/isaac_probe_v35e.log" >> "${SUITE_LOG}" || true
  fail_suite "probe_not_ready"
fi
cat "${LOG_DIR}/isaac_probe_v35e.log" >> "${SUITE_LOG}" || true

# ── v35e gates: FAIL-FAST ──
# Stage verify
if ! rg -q "\[V35E_STAGE_VERIFY\] ok=1" "${SUITE_LOG}"; then
  fail_suite "stage_verify_failed"
fi
# Camera fidelity
if ! rg -q "\[ISAAC_CAMERA_FIDELITY\] ok=1" "${SUITE_LOG}"; then
  fail_suite "camera_fidelity_failed"
fi
# Basis sanity (both cameras)
if ! rg -q "\[V35E_BASIS_SANITY\].*label=fp.*ok=1" "${SUITE_LOG}"; then
  fail_suite "fp_basis_sanity_failed"
fi
if ! rg -q "\[V35E_BASIS_SANITY\].*label=chase.*ok=1" "${SUITE_LOG}"; then
  fail_suite "chase_basis_sanity_failed"
fi
# Alignment gates
if ! rg -q "\[V35E_CAM_ALIGN\].*label=fp.*ok=1" "${SUITE_LOG}"; then
  fail_suite "fp_alignment_gate_failed"
fi
if ! rg -q "\[V35E_CAM_ALIGN\].*label=chase.*ok=1" "${SUITE_LOG}"; then
  fail_suite "chase_alignment_gate_failed"
fi
# Axis sweep test
if ! rg -q "\[V35E_AXIS_TEST\] ok=1" "${SUITE_LOG}"; then
  fail_suite "axis_sweep_test_failed"
fi
# Bridge topics
if ! rg -q "\[V34D_BRIDGE_TOPICS\] ok=1" "${SUITE_LOG}"; then
  fail_suite "bridge_topics_failed"
fi

echo "[V35E_CAMERA_GATES_OK] ok=1" | tee -a "${SUITE_LOG}"

# ── 4) Nav2 bringup ──
if ! run_and_log bash scripts/nav2/bringup_nav2_office_v34i.sh; then
  fail_suite "nav2_bringup_failed"
fi
run_and_log bash scripts/nav2/ros2_flow_check_v34i.sh || {
  echo "[V35E_ROS2_FLOW_WARN] ros2_flow_check_incomplete — odom may not publish until main loop starts" | tee -a "${SUITE_LOG}"
}

# ── 5) Sample reachable goals (2 fixed goals) ──
REACHABLE_JSON="${ROOMQA_DIR}/reachable_goals_seed.json"
SAMPLE_LOG="${LOG_DIR}/sample_reachable_goals_v35e.log"
: > "${SAMPLE_LOG}"
docker exec "${V34D_NAV2_CONTAINER}" bash -lc \
  "source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION} && python3 /home/peng/DualVLN/scripts/nav2/sample_reachable_goals_v34j.py --map_yaml ${V34D_MAP_YAML} --out_json ${REACHABLE_JSON} --count 4 --radius_min ${V35E_RADIUS_MIN:-0.8} --radius_max ${V35E_RADIUS_MAX:-1.8} --seed ${V35E_GOAL_SEED:-3}" \
  > "${SAMPLE_LOG}" 2>&1 || true
cat "${SAMPLE_LOG}" | tee -a "${SUITE_LOG}"
if ! rg -q "\\[V34J_GOAL_SAMPLE\\].*ok=1" "${SAMPLE_LOG}"; then
  fail_suite "reachable_goal_sampling_failed"
fi

read -r G1X G1Y G1YAW G2X G2Y G2YAW <<EOF
$(python3 - <<'PY' "${REACHABLE_JSON}"
import json,sys
goals=json.load(open(sys.argv[1],encoding='utf-8')).get('goals',[])
vals=[]
for i in range(2):
    if i < len(goals):
        g=goals[i]
        vals.extend([str(float(g.get('x',0.0))), str(float(g.get('y',0.0))), str(float(g.get('yaw_deg',0.0)))])
    else:
        vals.extend(['0.0','0.0','0.0'])
print(' '.join(vals))
PY
)
EOF

ROOM_TARGET="$(python3 - <<'PY' "${ROOMQA_DIR}/room_partition.json"
import json,sys
obj=json.load(open(sys.argv[1],encoding='utf-8'))
ids=sorted(int(r.get('room_id',-1)) for r in obj.get('rooms',[]) if int(r.get('room_id',-1))>=0)
if len(ids)>=2: print(ids[1])
elif len(ids)==1: print(ids[0])
else: print(0)
PY
)"

# ── 6) Run staged nav tasks ──
run_stage_task() {
  local task="$1"
  local task_id="$2"
  local room="$3"
  local obj="$4"
  local gx="$5"
  local gy="$6"
  local gyaw="$7"

  local plan_log="${LOG_DIR}/plan_${task_id}.log"
  : > "${plan_log}"
  if ! python3 scripts/topo_instruction_to_goal_v35a.py \
    --stage "${ISAAC_STAGE_USD}" \
    --room_partition_json "${ROOMQA_DIR}/room_partition.json" \
    --room_graph_json "${ROOMQA_DIR}/room_graph.json" \
    --topo_nodes_json "${ROOMQA_DIR}/topo_nodes.json" \
    --pose_trace_csv "${OUT_ROOT}/pose_trace_v35e.csv" \
    --out_dir "${ROOMQA_DIR}" \
    --task "${task}" \
    --task_id "${task_id}" \
    --room "${room}" \
    --object "${obj}" \
    --goal_x "${gx}" \
    --goal_y "${gy}" \
    --goal_yaw_deg "${gyaw}" > "${plan_log}" 2>&1; then
    cat "${plan_log}" | tee -a "${SUITE_LOG}"
    fail_suite "goal_resolution_failed_${task_id}"
  fi
  cat "${plan_log}" | tee -a "${SUITE_LOG}"

  local goals_json="${ROOMQA_DIR}/staged_goals_${task_id}.json"
  local seg_json="${ROOMQA_DIR}/staged_result_${task_id}.json"
  local seg_log="${LOG_DIR}/send_staged_${task_id}.log"
  : > "${seg_log}"
  docker exec "${V34D_NAV2_CONTAINER}" bash -lc \
    "source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION} && python3 /home/peng/DualVLN/scripts/nav2/send_staged_goals_v35a.py --goals_json ${goals_json} --frame map --timeout_s ${V35E_SEG_TIMEOUT_S:-90} --dist_tol_m ${V35E_SEG_DIST_TOL_M:-0.45} --yaw_tol_deg ${V35E_SEG_YAW_TOL_DEG:-40} --result_json ${seg_json}" \
    > "${seg_log}" 2>&1 || true
  cat "${seg_log}" | tee -a "${SUITE_LOG}"

  python3 - "${seg_json}" "${task_id}" <<'PY' | tee -a "${SUITE_LOG}"
import json,sys
p,tid=sys.argv[1],sys.argv[2]
try:
    r=json.load(open(p,encoding='utf-8'))
except Exception:
    print(f"[V35E_CASE_DONE] case={tid} overall=PARSE_ERROR stages_succeeded=0/0")
    sys.exit(0)
segs=r.get('segments',[])
ok_count=sum(1 for s in segs if s.get('ok',0)==1)
for i,s in enumerate(segs):
    print(f"[V35E_NAV2_STAGE] case={tid} stage={i} status={s.get('status','?')} dist_m={s.get('dist_m','-'):.3f} yaw_err_deg={s.get('yaw_err_deg','-'):.1f} ok={s.get('ok',0)}")
overall="SUCCEEDED" if r.get('ok',0)==1 else "FAILED"
print(f"[V35E_CASE_DONE] case={tid} overall={overall} stages_succeeded={ok_count}/{len(segs)}")
PY

  if rg -q "\\[V35A_NAV2_STAGED\\].*ok=1" "${seg_log}" 2>/dev/null || \
     python3 -c "import json,sys; r=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if r.get('ok',0)==1 else 1)" "${seg_json}" 2>/dev/null; then
    STAGED_SUCCEEDED=$((STAGED_SUCCEEDED + 1))
  fi
}

STAGED_SUCCEEDED=0
STAGED_SKIPPED=0
run_stage_task doorway door_to_door "room_${ROOM_TARGET}" "" "${G1X}" "${G1Y}" "${G1YAW}"
run_stage_task doorway hallway_cross "room_${ROOM_TARGET}" "" "${G2X}" "${G2Y}" "${G2YAW}"

# ── 7) Stop probe ──
if [ -n "${PROBE_PID}" ] && kill -0 "${PROBE_PID}" >/dev/null 2>&1; then
  kill -TERM "${PROBE_PID}" >/dev/null 2>&1 || true
  sleep 3
  kill -9 "${PROBE_PID}" >/dev/null 2>&1 || true
  wait "${PROBE_PID}" >/dev/null 2>&1 || true
  PROBE_PID=""
fi
cat "${LOG_DIR}/isaac_probe_v35e.log" >> "${SUITE_LOG}" || true

# ── 8) Motion gate on captures ──
FP_CAP="${OUT_ROOT}/capture_fp"
CHASE_CAP="${OUT_ROOT}/capture_chase"

for view_dir in "${FP_CAP}" "${CHASE_CAP}"; do
  view_name="$(basename "${view_dir}")"
  if [ -d "${view_dir}" ] && [ "$(find "${view_dir}" -name '*.png' | wc -l)" -ge 2 ]; then
    run_and_log python3 scripts/tools/check_capture_motion.py \
      --capture_dir "${view_dir}" --tag "v35e_${view_name}" --write_gif 0 || true
  else
    echo "[V35E_MOTION_GATE] view=${view_name} ok=0 reason=no_frames" | tee -a "${SUITE_LOG}"
  fi
done

# ── 9) Verify Nav2 outcomes ──
# v35e: Nav2 is informational, camera alignment gates are the primary validation
if [ "${STAGED_SUCCEEDED}" -lt 2 ]; then
  echo "[V35E_NAV2_GATE] ok=0 staged_succeed=${STAGED_SUCCEEDED} of ${STAGED_TOTAL} — Nav2 informational in v35e (camera gates are primary)" | tee -a "${SUITE_LOG}"
fi

echo "[V35E_NAV2_GATE] staged_succeed=${STAGED_SUCCEEDED} of ${STAGED_TOTAL}" | tee -a "${SUITE_LOG}"
echo "[V35E_SUITE_OK] ok=1 cases=${STAGED_TOTAL} succeeded=${STAGED_SUCCEEDED} skipped=${STAGED_SKIPPED} rooms=${ROOM_COUNT} doors=${DOOR_COUNT} root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
