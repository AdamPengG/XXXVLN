#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34C_OUT_ROOT:-runs/topo_mvp/v34c_ros2_bridge}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/isaac_ui_ros2_bridge_v34c.log"
: > "${LOG_FILE}"

SCENE_ID="${V34C_SCENE_ID:-office_localized}"
CFG="${V34C_ISAAC_CONFIG:-configs/isaac_scenes_v34b.yaml}"
HEADLESS="${V34C_HEADLESS:-0}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

resolve_stage() {
  if [ -n "${ISAAC_STAGE_USD:-}" ] && [ -f "${ISAAC_STAGE_USD}" ]; then
    echo "${ISAAC_STAGE_USD}|override"
    return 0
  fi
  local p="/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"
  if [ -f "${p}" ]; then
    echo "${p}|official_assets"
    return 0
  fi
  echo "|missing"
}

stage_info="$(resolve_stage)"
STAGE_PATH="${stage_info%%|*}"
STAGE_SOURCE="${stage_info##*|}"
STAGE_EXISTS=0
if [ -n "${STAGE_PATH}" ] && [ -f "${STAGE_PATH}" ]; then
  STAGE_EXISTS=1
fi

echo "[V34B_ISAAC_STAGE] usd=${STAGE_PATH} exists=${STAGE_EXISTS} source=${STAGE_SOURCE}" | tee -a "${LOG_FILE}"
if [ "${STAGE_EXISTS}" != "1" ]; then
  echo "[V34C_ISAAC_UI_LAUNCH] ok=0 stage=${STAGE_PATH} gpu=0 ros_domain=${ROS_DOMAIN_ID} rmw=${RMW_IMPLEMENTATION} reason=stage_missing" | tee -a "${LOG_FILE}"
  exit 2
fi

export ISAAC_STAGE_USD="${STAGE_PATH}"
export ISAAC_GPU_ID=0
export ISAAC_STEP_RENDER=1
export ISAAC_STEP_RENDER_EVERY_N=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}"

if [ -f /opt/ros/humble/setup.bash ]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi

{
  python3 scripts/gpu/gpu_router.py --role isaac --output anchors
  echo "[V34C_ISAAC_UI_LAUNCH] ok=1 stage=${STAGE_PATH} gpu=0 ros_domain=${ROS_DOMAIN_ID} rmw=${RMW_IMPLEMENTATION}"
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    env ISAAC_HEADLESS="${HEADLESS}" bash scripts/isaac/run_with_isaac_python.sh -- \
      scripts/isaac/ui_office_ros2_bridge_probe_v34c.py \
      --scene_id "${SCENE_ID}" \
      --isaac_config "${CFG}" \
      --stage "${STAGE_PATH}" \
      --out_dir "${OUT_ROOT}" \
      --steps 80
} 2>&1 | tee -a "${LOG_FILE}"
