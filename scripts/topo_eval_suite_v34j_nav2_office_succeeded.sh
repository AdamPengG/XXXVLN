#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34J_OUT_ROOT:-runs/topo_mvp/v34j_nav2_office_succeeded}"
LOG_DIR="${OUT_ROOT}/logs"
MAP_DIR="${OUT_ROOT}/map"
CAP_DIR="${OUT_ROOT}/capture"
mkdir -p "${LOG_DIR}" "${MAP_DIR}" "${CAP_DIR}"
SUITE_LOG="${LOG_DIR}/topo_eval_suite_v34j_nav2_office_succeeded.log"
: > "${SUITE_LOG}"

STAGE_OFFICIAL="${ISAAC_STAGE_USD:-/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ISAAC_STEP_RENDER=1
export ISAAC_STEP_RENDER_EVERY_N=1
export ISAAC_ENABLE_DEPTH="${ISAAC_ENABLE_DEPTH:-1}"
export ISAAC_GPU_ID=0
export ISAAC_CAMERA_PITCH_DEG="${ISAAC_CAMERA_PITCH_DEG:--10.0}"
export ISAAC_CAMERA_OFFSET="${ISAAC_CAMERA_OFFSET:-0.20,0.00,0.30}"
export ISAAC_CAMERA_RPY_DEG="${ISAAC_CAMERA_RPY_DEG:-0,-10,0}"
export V34D_CMD_LIN_THRESH="${V34D_CMD_LIN_THRESH:-0.08}"
export V34D_CMD_ANG_THRESH="${V34D_CMD_ANG_THRESH:-0.08}"
export V34J_DIST_TOL_M="${V34J_DIST_TOL_M:-0.40}"

export V34J_OUT_ROOT="${OUT_ROOT}"
export V34I_OUT_ROOT="${OUT_ROOT}"
export V34D_OUT_ROOT="${OUT_ROOT}"
export V34G_OUT_ROOT="${OUT_ROOT}"
export V34G_MAP_OUT_DIR="${MAP_DIR}"
export V34G_MAP_BOUNDS="${V34G_MAP_BOUNDS:--6,6,-6,6}"
export V34D_MAP_YAML="${MAP_DIR}/office_map.yaml"
export V34I_NAV2_PARAMS="${V34I_NAV2_PARAMS:-configs/nav2_params_v34j.yaml}"
export V34D_NAV2_PARAMS="${V34I_NAV2_PARAMS}"
export V34D_NAV2_CONTAINER="${V34D_NAV2_CONTAINER:-v34j_nav2_stack}"
export V34I_ODOM_TF_CONTAINER="${V34I_ODOM_TF_CONTAINER:-v34j_odom_tf_bridge}"
export V34D_NAV2_DOCKER_IMAGE="${V34D_NAV2_DOCKER_IMAGE:-v34d_ros2_nav2:humble}"
export V34D_USE_LOCALIZATION="${V34D_USE_LOCALIZATION:-0}"

AMENT_ROOT="/home/peng/IsaacSim/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge/humble"
export AMENT_PREFIX_PATH="${AMENT_ROOT}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
export PYTHONPATH="${AMENT_ROOT}/rclpy${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${AMENT_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

PROBE_PID=""
cleanup() {
  if [ -n "${PROBE_PID}" ] && kill -0 "${PROBE_PID}" >/dev/null 2>&1; then
    kill "${PROBE_PID}" >/dev/null 2>&1 || true
    wait "${PROBE_PID}" >/dev/null 2>&1 || true
  fi
  docker rm -f "${V34I_ODOM_TF_CONTAINER}" >/dev/null 2>&1 || true
  docker rm -f "${V34D_NAV2_CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

run_and_log() {
  ( "$@" ) 2>&1 | tee -a "${SUITE_LOG}"
}

fail_suite() {
  local reason="$1"
  echo "[V34J_SUITE_OK] ok=0 goals=${GOAL_TOTAL:-0} succeeded=${GOAL_SUCCEEDED:-0} within_tol=${GOAL_WITHIN:-0} root=${OUT_ROOT} reason=${reason}" | tee -a "${SUITE_LOG}"
  exit 2
}

echo "[V34J_SUITE] start=$(date -Iseconds) root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
run_and_log python3 scripts/gpu/gpu_router.py --role isaac --output anchors

if [ ! -f "${STAGE_OFFICIAL}" ]; then
  fail_suite "official_stage_missing"
fi
export ISAAC_STAGE_USD="${STAGE_OFFICIAL}"
echo "[V34B_ISAAC_STAGE] usd=${STAGE_OFFICIAL} exists=1 source=official_assets" | tee -a "${SUITE_LOG}"

# 1) Map generation.
if ! run_and_log bash scripts/nav2/generate_office_map_v34g.sh; then
  fail_suite "map_generation_failed"
fi
if ! rg -q "\[V34G_MAP_SANITY\] ok=1" "${SUITE_LOG}"; then
  fail_suite "map_sanity_failed"
fi

# 2) Start Isaac bridge probe.
READY_FILE="${OUT_ROOT}/bridge_ready_v34j.json"
rm -f "${READY_FILE}"
: > "${LOG_DIR}/isaac_probe_v34j.log"
(
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    env ISAAC_HEADLESS="${V34J_HEADLESS:-1}" ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
      bash scripts/isaac/run_with_isaac_python.sh -- \
      scripts/isaac/ui_office_ros2_bridge_probe_v34d.py \
      --scene_id office_localized \
      --isaac_config configs/isaac_scenes_v34b.yaml \
      --stage "${ISAAC_STAGE_USD}" \
      --out_dir "${OUT_ROOT}" \
      --steps "${V34J_CAPTURE_STEPS:-4000}" \
      --ready_file "${READY_FILE}" \
      --headless "${V34J_HEADLESS:-1}" \
      --idle_action "${V34J_IDLE_ACTION:-stop}" \
      --cam_pitch_deg "${ISAAC_CAMERA_PITCH_DEG}"
) > "${LOG_DIR}/isaac_probe_v34j.log" 2>&1 &
PROBE_PID=$!

for _ in $(seq 1 140); do
  [ -f "${READY_FILE}" ] && break
  sleep 1
done
if [ ! -f "${READY_FILE}" ]; then
  cat "${LOG_DIR}/isaac_probe_v34j.log" >> "${SUITE_LOG}" || true
  fail_suite "probe_not_ready"
fi
cat "${LOG_DIR}/isaac_probe_v34j.log" >> "${SUITE_LOG}" || true

if ! rg -q "\[V34H_STAGE_VERIFY\] ok=1 .*office\.usd" "${SUITE_LOG}"; then
  fail_suite "stage_verify_failed"
fi
if ! rg -q "\[ISAAC_STEP_CFG\].*step_render=1.*step_render_every_n=1" "${SUITE_LOG}"; then
  fail_suite "isaac_step_cfg_missing"
fi
if ! rg -q "\[ISAAC_CAMERA_MOUNT\]" "${SUITE_LOG}"; then
  fail_suite "camera_mount_anchor_missing"
fi
if ! rg -q "\[ISAAC_CAMERA_VEC\]" "${SUITE_LOG}"; then
  fail_suite "camera_vec_anchor_missing"
fi
CAM_VEC_OK="$(python3 - <<'PY' "${SUITE_LOG}"
import re,sys
text=open(sys.argv[1],encoding='utf-8').read()
m=re.findall(r"\[ISAAC_CAMERA_VEC\].*angle_to_robot_fwd_deg=([0-9.]+)", text)
if not m:
    print(0)
else:
    print(1 if float(m[-1]) <= 25.0 else 0)
PY
)"
if [ "${CAM_VEC_OK}" != "1" ]; then
  fail_suite "camera_forward_angle_failed"
fi
if ! rg -q "\[ISAAC_CAMERA_FIDELITY\] ok=1" "${SUITE_LOG}"; then
  fail_suite "camera_fidelity_failed"
fi
if ! rg -q "\[V34J_VIEW_ORIENT\].*ok=1" "${SUITE_LOG}"; then
  fail_suite "view_orientation_failed"
fi

# 3) Bringup Nav2 + odom->tf bridge.
if ! run_and_log bash scripts/nav2/bringup_nav2_office_v34i.sh; then
  fail_suite "nav2_bringup_failed"
fi
if ! run_and_log bash scripts/nav2/ros2_flow_check_v34i.sh; then
  fail_suite "ros2_flow_check_failed"
fi

# 4) Sample reachable goals with planner service.
GOALS_JSON="${OUT_ROOT}/goals.json"
SAMPLE_LOG="${LOG_DIR}/sample_reachable_goals_v34j.log"
: > "${SAMPLE_LOG}"
docker exec "${V34D_NAV2_CONTAINER}" bash -lc \
  "source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION} && python3 /home/peng/DualVLN/scripts/nav2/sample_reachable_goals_v34j.py --map_yaml ${V34D_MAP_YAML} --out_json ${GOALS_JSON} --count 3 --radius_min ${V34J_RADIUS_MIN:-0.8} --radius_max ${V34J_RADIUS_MAX:-1.4} --seed ${V34J_GOAL_SEED:-0}" \
  > "${SAMPLE_LOG}" 2>&1 || true
cat "${SAMPLE_LOG}" | tee -a "${SUITE_LOG}"
if ! rg -q "\[V34J_GOAL_SAMPLE\].*ok=1" "${SAMPLE_LOG}"; then
  if rg -q "\[V34J_PLANNER_SERVICE\] ok=0" "${SAMPLE_LOG}"; then
    fail_suite "planner_service_unavailable"
  fi
  fail_suite "goal_sampling_failed"
fi

# 5) Send goals (strict: 3/3 SUCCEEDED and <=0.40m).
GOAL_LOG="${LOG_DIR}/send_goal_v34j.log"
: > "${GOAL_LOG}"
docker exec "${V34D_NAV2_CONTAINER}" bash -lc \
  "source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION} V34J_ASSIST_CMDVEL=${V34J_ASSIST_CMDVEL:-1} V34J_ASSIST_LIN_MAX=${V34J_ASSIST_LIN_MAX:-0.22} V34J_ASSIST_ANG_MAX=${V34J_ASSIST_ANG_MAX:-0.9} && python3 /home/peng/DualVLN/scripts/nav2/send_goal_v34j.py --goals_json ${GOALS_JSON} --timeout_s ${V34J_GOAL_TIMEOUT_S:-120} --dist_tol_m ${V34J_DIST_TOL_M} --frame map" \
  > "${GOAL_LOG}" 2>&1 || true
cat "${GOAL_LOG}" | tee -a "${SUITE_LOG}"

if ! rg -q "\[V34J_GOALS\] total=3 succeeded=3 within_tol=3" "${GOAL_LOG}"; then
  fail_suite "goals_quality_gate_failed"
fi

GOAL_TOTAL=3
GOAL_SUCCEEDED=3
GOAL_WITHIN=3

# 6) Stop probe cleanly and append final logs.
if [ -n "${PROBE_PID}" ] && kill -0 "${PROBE_PID}" >/dev/null 2>&1; then
  kill "${PROBE_PID}" >/dev/null 2>&1 || true
  wait "${PROBE_PID}" >/dev/null 2>&1 || true
  PROBE_PID=""
fi
cat "${LOG_DIR}/isaac_probe_v34j.log" >> "${SUITE_LOG}" || true

# 7) Capture motion gate.
if ! run_and_log bash scripts/tools/check_capture_motion.sh --capture_dir "${CAP_DIR}" --tag v34j_nav2_office_succeeded --write_gif 1; then
  fail_suite "capture_motion_failed"
fi

echo "[V34J_SUITE_OK] ok=1 goals=${GOAL_TOTAL} succeeded=${GOAL_SUCCEEDED} within_tol=${GOAL_WITHIN} root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
