#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"

CFG="${V31_PLANB_CFG:-configs/isaac_planb_eval_v31.yaml}"
OUT_ROOT="${V31_PLANB_OUT_ROOT:-runs/topo_mvp/v31_planb_scale_eval}"
LOG_DIR="${OUT_ROOT}/logs"
MASTER_LOG="${LOG_DIR}/topo_eval_suite_v31_planb_scale_eval.log"
mkdir -p "${LOG_DIR}"
: > "${MASTER_LOG}"

SMALL_ARG=""
if [ "${1:-}" = "--small" ]; then
  SMALL_ARG="--small"
fi

echo "[V31_SUITE] start=$(date -Iseconds) cfg=${CFG} out_root=${OUT_ROOT} small=${SMALL_ARG:-0}" | tee -a "${MASTER_LOG}"
python3 scripts/gpu/gpu_router.py --role isaac --output anchors | tee -a "${MASTER_LOG}"

(
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    python3 scripts/topo_planb_scale_eval_v31.py --config "${CFG}" ${SMALL_ARG}
) 2>&1 | tee -a "${MASTER_LOG}"

(
  bash scripts/topo_failure_cards_v31.sh \
    "${OUT_ROOT}/failure_cases.jsonl" \
    "${OUT_ROOT}/failure_cards"
) 2>&1 | tee -a "${MASTER_LOG}"

echo "[V31_SUITE] done log=${MASTER_LOG}" | tee -a "${MASTER_LOG}"
