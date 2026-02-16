#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

STAGE="${ISAAC_STAGE_USD:-/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd}"
OUT_USD="${V34E_OFFICE_PHYS_USD:-runs/topo_mvp/v34e_nav2_office_phys/office_phys/office_phys.usd}"
METHOD="${V34E_COLLIDER_METHOD:-convexHull}"

bash scripts/gpu/run_isaac_on_5090.sh -- \
  bash scripts/isaac/run_with_isaac_python.sh -- \
    scripts/isaac/build_office_phys_stage_v34e.py \
      --stage "${STAGE}" \
      --out_usd "${OUT_USD}" \
      --method "${METHOD}"
