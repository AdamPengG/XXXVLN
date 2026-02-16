#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34B_OUT_ROOT:-runs/topo_mvp/v34b_nav2_demo}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/isaac_ui_probe_v34b.log"
: > "${LOG_FILE}"

SCENE_ID="${V34B_SCENE_ID:-office_localized}"
CFG="${V34B_ISAAC_CONFIG:-configs/isaac_scenes_v34b.yaml}"
HEADLESS="${V34B_HEADLESS:-0}"

export ISAAC_STEP_RENDER=1
export ISAAC_STEP_RENDER_EVERY_N=1
export ISAAC_ENABLE_DEPTH=1
export ISAAC_HEADLESS="${HEADLESS}"

{
  echo "[V34B_UI_LAUNCH] start=$(date -Iseconds) scene=${SCENE_ID} headless=${HEADLESS} cfg=${CFG}"
  python3 scripts/gpu/gpu_router.py --role isaac --output anchors
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    bash scripts/isaac/run_with_isaac_python.sh -- \
      scripts/isaac/ui_office_nav2_probe_v34b.py \
        --scene_id "${SCENE_ID}" \
        --isaac_config "${CFG}" \
        --out_dir "${OUT_ROOT}/ui_probe"
  echo "[V34B_UI_LAUNCH] done log=${LOG_FILE}"
} 2>&1 | tee -a "${LOG_FILE}"
