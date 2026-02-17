#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34I_OUT_ROOT:-runs/topo_mvp/v34i_nav2_office_succeeded}"
LOG_DIR="${OUT_ROOT}/logs"
MAP_DIR="${OUT_ROOT}/map"
CAP_DIR="${OUT_ROOT}/capture"
mkdir -p "${LOG_DIR}" "${MAP_DIR}" "${CAP_DIR}"
SUITE_LOG="${LOG_DIR}/topo_eval_suite_v34i_nav2_office_succeeded.log"
: > "${SUITE_LOG}"

STAGE_OFFICIAL="${ISAAC_STAGE_USD:-/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ISAAC_STEP_RENDER=1
export ISAAC_STEP_RENDER_EVERY_N=1
export ISAAC_ENABLE_DEPTH="${ISAAC_ENABLE_DEPTH:-1}"
export ISAAC_GPU_ID=0
export ISAAC_CAMERA_PITCH_DEG="${ISAAC_CAMERA_PITCH_DEG:--10.0}"
export V34D_CMD_LIN_THRESH="${V34D_CMD_LIN_THRESH:-0.08}"
export V34D_CMD_ANG_THRESH="${V34D_CMD_ANG_THRESH:-0.08}"

export V34I_OUT_ROOT="${OUT_ROOT}"
export V34D_OUT_ROOT="${OUT_ROOT}"
export V34G_OUT_ROOT="${OUT_ROOT}"
export V34G_MAP_OUT_DIR="${MAP_DIR}"
export V34G_MAP_BOUNDS="${V34G_MAP_BOUNDS:--6,6,-6,6}"
export V34D_MAP_YAML="${MAP_DIR}/office_map.yaml"
export V34I_NAV2_PARAMS="${V34I_NAV2_PARAMS:-configs/nav2_params_v34h.yaml}"
export V34D_NAV2_PARAMS="${V34I_NAV2_PARAMS}"
export V34D_NAV2_CONTAINER="${V34D_NAV2_CONTAINER:-v34i_nav2_stack}"
export V34I_ODOM_TF_CONTAINER="${V34I_ODOM_TF_CONTAINER:-v34i_odom_tf_bridge}"
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
  echo "[V34I_SUITE_OK] ok=0 succeeded=${SUCCEEDED_COUNT:-0} root=${OUT_ROOT} reason=${reason}" | tee -a "${SUITE_LOG}"
  exit 2
}

echo "[V34I_SUITE] start=$(date -Iseconds) root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
run_and_log python3 scripts/gpu/gpu_router.py --role isaac --output anchors

if [ ! -f "${STAGE_OFFICIAL}" ]; then
  fail_suite "official_stage_missing"
fi
export ISAAC_STAGE_USD="${STAGE_OFFICIAL}"
echo "[V34I_STAGE] official_usd=${STAGE_OFFICIAL}" | tee -a "${SUITE_LOG}"

# 1) Map generation.
if ! run_and_log bash scripts/nav2/generate_office_map_v34g.sh; then
  fail_suite "map_generation_failed"
fi
if ! rg -q "\[V34G_MAP_SANITY\] ok=1" "${SUITE_LOG}"; then
  fail_suite "map_sanity_failed"
fi

# 2) Start Isaac probe for bridge/capture.
READY_FILE="${OUT_ROOT}/bridge_ready_v34i.json"
rm -f "${READY_FILE}"
: > "${LOG_DIR}/isaac_probe_v34i.log"
(
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    env ISAAC_HEADLESS="${V34I_HEADLESS:-1}" ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
      bash scripts/isaac/run_with_isaac_python.sh -- \
      scripts/isaac/ui_office_ros2_bridge_probe_v34d.py \
      --scene_id office_localized \
      --isaac_config configs/isaac_scenes_v34b.yaml \
      --stage "${ISAAC_STAGE_USD}" \
      --out_dir "${OUT_ROOT}" \
      --steps "${V34I_CAPTURE_STEPS:-360}" \
      --ready_file "${READY_FILE}" \
      --headless "${V34I_HEADLESS:-1}" \
      --idle_action "${V34I_IDLE_ACTION:-stop}" \
      --cam_pitch_deg "${ISAAC_CAMERA_PITCH_DEG}"
) > "${LOG_DIR}/isaac_probe_v34i.log" 2>&1 &
PROBE_PID=$!

for _ in $(seq 1 140); do
  [ -f "${READY_FILE}" ] && break
  sleep 1
done
if [ ! -f "${READY_FILE}" ]; then
  cat "${LOG_DIR}/isaac_probe_v34i.log" >> "${SUITE_LOG}" || true
  fail_suite "probe_not_ready"
fi
cat "${LOG_DIR}/isaac_probe_v34i.log" >> "${SUITE_LOG}" || true

if ! rg -q "\[V34H_STAGE_VERIFY\] ok=1 .*office\.usd" "${SUITE_LOG}"; then
  fail_suite "stage_verify_failed"
fi
if ! rg -q "\[ISAAC_CAMERA_FIDELITY\] ok=1" "${SUITE_LOG}"; then
  fail_suite "camera_fidelity_failed"
fi
if ! rg -q "\[V34D_BRIDGE_TOPICS\] ok=1" "${SUITE_LOG}"; then
  fail_suite "bridge_topics_failed"
fi

# 3) Bringup nav2 + odom->tf bridge.
if ! run_and_log bash scripts/nav2/bringup_nav2_office_v34i.sh; then
  fail_suite "nav2_bringup_failed"
fi

# 4) ROS2 flow checks (message flow + dynamic TF).
if ! run_and_log bash scripts/nav2/ros2_flow_check_v34i.sh; then
  fail_suite "ros2_flow_check_failed"
fi

# 5) Send deterministic goals; require >=2 SUCCEEDED.
GOAL_TIMEOUT="${V34I_GOAL_TIMEOUT_S:-120}"
GOAL_COUNT=3
SUCCEEDED_COUNT=0
GOAL_POINTS=("-1.70 -2.00" "-1.20 -2.00" "-1.20 -1.40")

for idx in $(seq 1 "${GOAL_COUNT}"); do
  pair="${GOAL_POINTS[$((idx-1))]}"
  goal_x="$(echo "${pair}" | awk '{print $1}')"
  goal_y="$(echo "${pair}" | awk '{print $2}')"
  GOAL_LOG="${LOG_DIR}/send_goal_v34i_${idx}.log"
  : > "${GOAL_LOG}"
  docker exec "${V34D_NAV2_CONTAINER}" bash -lc \
    "source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION} && python3 /home/peng/DualVLN/scripts/nav2/send_goal_v34i.py --x ${goal_x} --y ${goal_y} --yaw_deg 0 --frame map --timeout_s ${GOAL_TIMEOUT} --goal_id v34i_goal_${idx}" \
    > "${GOAL_LOG}" 2>&1 || true
  cat "${GOAL_LOG}" | tee -a "${SUITE_LOG}"
  if rg -q "\[V34I_NAV2_RESULT\] status=SUCCEEDED ok=1 goal_id=v34i_goal_${idx}" "${GOAL_LOG}"; then
    SUCCEEDED_COUNT=$((SUCCEEDED_COUNT + 1))
  fi
done

echo "[V34I_GOALS] total=${GOAL_COUNT} succeeded=${SUCCEEDED_COUNT} failed=$((GOAL_COUNT - SUCCEEDED_COUNT))" | tee -a "${SUITE_LOG}"
if [ "${SUCCEEDED_COUNT}" -lt 2 ]; then
  fail_suite "insufficient_succeeded_goals"
fi

# 6) Let probe continue briefly for capture, then stop.
sleep 2
if [ -n "${PROBE_PID}" ] && kill -0 "${PROBE_PID}" >/dev/null 2>&1; then
  kill "${PROBE_PID}" >/dev/null 2>&1 || true
  wait "${PROBE_PID}" >/dev/null 2>&1 || true
  PROBE_PID=""
fi
cat "${LOG_DIR}/isaac_probe_v34i.log" >> "${SUITE_LOG}" || true

POSE_TRACE_OK=0
if rg -q "\[V34H_POSE_TRACE\] .*ok=1" "${SUITE_LOG}"; then
  POSE_TRACE_OK=1
fi
echo "[V34I_POSE_TRACE] ok=${POSE_TRACE_OK}" | tee -a "${SUITE_LOG}"

# 7) Capture motion check.
if ! run_and_log bash scripts/tools/check_capture_motion.sh --capture_dir "${CAP_DIR}" --tag v34i_nav2_office_succeeded --write_gif 1; then
  fail_suite "capture_motion_failed"
fi

echo "[V34I_SUITE_OK] ok=1 succeeded=${SUCCEEDED_COUNT} root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
