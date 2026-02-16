#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_DIR="${V34E_COLLISION_AUDIT_OUT:-runs/topo_mvp/v34e_nav2_office_phys/collision_audit}"
STAGE="${ISAAC_STAGE_USD:-/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd}"
ALLOW_SPARSE="${ALLOW_SPARSE_COLLIDERS:-0}"
MIN_RATIO="${COLLIDER_MIN_RATIO:-0.50}"

bash scripts/gpu/run_isaac_on_5090.sh -- \
  bash scripts/isaac/run_with_isaac_python.sh -- \
    scripts/isaac/collision_audit_v34e.py \
      --stage "${STAGE}" \
      --out_dir "${OUT_DIR}" \
      --allow_sparse_colliders "${ALLOW_SPARSE}" \
      --min_ratio "${MIN_RATIO}"
