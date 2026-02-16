#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_USD="${V34B_OFFICE_PHYS_USD:-runs/topo_mvp/v34b_nav2_demo/office_phys/office_phys.usd}"
STAGE_ARG="${ISAAC_STAGE_USD:-}"
METHOD="${V34B_COLLIDER_METHOD:-convexHull}"

cmd=(
  scripts/isaac/build_office_phys_stage_v34b2.py
  --out_usd "${OUT_USD}"
  --method "${METHOD}"
)
if [ -n "${STAGE_ARG}" ]; then
  cmd+=(--stage "${STAGE_ARG}")
fi

bash scripts/gpu/run_isaac_on_5090.sh -- \
  bash scripts/isaac/run_with_isaac_python.sh -- "${cmd[@]}"

