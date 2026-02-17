#!/usr/bin/env bash
# run_ui_office_topo_nav2_v35e.sh — Launches Isaac Sim (GUI or headless) with
# the v35e camera-alignment probe.
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"

# Stage
STAGE_OFFICIAL="/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"
STAGE="${ISAAC_STAGE_USD:-${STAGE_OFFICIAL}}"
if [ ! -f "${STAGE}" ]; then
  echo "[V35E_ISAAC_UI] ok=0 stage=${STAGE} exists=0 reason=stage_not_found"
  exit 2
fi

# GPU
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU_ID}" 2>/dev/null | head -1 || echo unknown)"

# UI mode
HEADLESS="${ISAAC_HEADLESS:-1}"
UI_MODE=$( [ "${HEADLESS}" = "0" ] && echo 1 || echo 0 )

echo "[V35E_ISAAC_UI] ok=1 stage=${STAGE} exists=1 gpu=${GPU_NAME} ui=${UI_MODE} headless=${HEADLESS}"

# Extra args forwarded to the probe
EXTRA_ARGS="${V35E_EXTRA_ARGS:-}"

exec bash scripts/gpu/run_isaac_on_5090.sh -- \
  env ISAAC_HEADLESS="${HEADLESS}" \
    ISAAC_STAGE_USD="${STAGE}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
    RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}" \
    V35E_CAM_HEIGHT_M="${V35E_CAM_HEIGHT_M:-1.55}" \
    ISAAC_CAMERA_PITCH_DEG="${ISAAC_CAMERA_PITCH_DEG:--10.0}" \
  bash scripts/isaac/run_with_isaac_python.sh -- \
  scripts/isaac/ui_office_topo_nav2_v35e.py \
    --scene_id "${V35E_SCENE_ID:-office_localized}" \
    --isaac_config "${V35E_ISAAC_CONFIG:-configs/isaac_scenes_v34b.yaml}" \
    --stage "${STAGE}" \
    --out_dir "${V35E_OUT_DIR:-runs/topo_mvp/v35e_ui_debug}" \
    --steps "${V35E_STEPS:-1200}" \
    --ready_file "${V35E_READY_FILE:-}" \
    --headless "${HEADLESS}" \
    --cam_pitch_deg "${ISAAC_CAMERA_PITCH_DEG:--10.0}" \
    --axis_test "${V35E_AXIS_TEST:-1}" \
    --axis_frames "${V35E_AXIS_FRAMES:-60}" \
    ${EXTRA_ARGS}
