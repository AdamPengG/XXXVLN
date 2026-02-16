#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34D_OUT_ROOT:-runs/topo_mvp/v34d_nav2_office}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/isaac_ui_ros2_bridge_v34d.log"
: > "${LOG_FILE}"

SCENE_ID="${V34D_SCENE_ID:-office_localized}"
CFG="${V34D_ISAAC_CONFIG:-configs/isaac_scenes_v34b.yaml}"
HEADLESS="${V34D_HEADLESS:-0}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

STAGE_DEFAULT="/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"
STAGE_PATH="${ISAAC_STAGE_USD:-${STAGE_DEFAULT}}"
STAGE_SOURCE="official_assets"
if [ -n "${ISAAC_STAGE_USD:-}" ]; then
  STAGE_SOURCE="override"
fi
STAGE_EXISTS=0
if [ -f "${STAGE_PATH}" ]; then
  STAGE_EXISTS=1
fi

AMENT_ROOT="/home/peng/IsaacSim/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge/humble"
export AMENT_PREFIX_PATH="${AMENT_ROOT}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
export PYTHONPATH="${AMENT_ROOT}/rclpy${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${AMENT_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export ISAAC_STAGE_USD="${STAGE_PATH}"
export ISAAC_GPU_ID=0
export ISAAC_STEP_RENDER=1
export ISAAC_STEP_RENDER_EVERY_N=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}"

{
  echo "[V34B_ISAAC_STAGE] usd=${STAGE_PATH} exists=${STAGE_EXISTS} source=${STAGE_SOURCE}"
  if [ "${STAGE_EXISTS}" != "1" ]; then
    echo "[V34C_ISAAC_UI_LAUNCH] ok=0 stage=${STAGE_PATH} gpu=0 ros_domain=${ROS_DOMAIN_ID} rmw=${RMW_IMPLEMENTATION} reason=stage_missing"
    exit 2
  fi
  python3 scripts/gpu/gpu_router.py --role isaac --output anchors
  echo "[V34C_ISAAC_UI_LAUNCH] ok=1 stage=${STAGE_PATH} gpu=0 ros_domain=${ROS_DOMAIN_ID} rmw=${RMW_IMPLEMENTATION}"
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    env ISAAC_HEADLESS="${HEADLESS}" bash scripts/isaac/run_with_isaac_python.sh -- \
      scripts/isaac/ui_office_ros2_bridge_probe_v34d.py \
      --scene_id "${SCENE_ID}" \
      --isaac_config "${CFG}" \
      --stage "${STAGE_PATH}" \
      --out_dir "${OUT_ROOT}" \
      --steps "${V34D_CAPTURE_STEPS:-80}" \
      --headless "${HEADLESS}"
} 2>&1 | tee -a "${LOG_FILE}"
