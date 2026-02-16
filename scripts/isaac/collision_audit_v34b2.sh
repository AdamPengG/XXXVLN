#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_DIR="${V34B_COLLISION_AUDIT_OUT:-runs/topo_mvp/v34b_nav2_demo/collision_audit}"
STAGE_ARG="${ISAAC_STAGE_USD:-}"
ALLOW_SPARSE="${ALLOW_SPARSE_COLLIDERS:-0}"
MIN_RATIO="${COLLIDER_MIN_RATIO:-0.50}"

cmd=(
  scripts/isaac/collision_audit_v34b2.py
  --out_dir "${OUT_DIR}"
  --allow_sparse_colliders "${ALLOW_SPARSE}"
  --min_ratio "${MIN_RATIO}"
)
if [ -n "${STAGE_ARG}" ]; then
  cmd+=(--stage "${STAGE_ARG}")
fi

bash scripts/gpu/run_isaac_on_5090.sh -- \
  bash scripts/isaac/run_with_isaac_python.sh -- "${cmd[@]}"

