#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

STAGE="${ISAAC_STAGE_USD:-/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd}"
OUT_DIR="${V34E_PHYSICS_OUT:-runs/topo_mvp/v34e_nav2_office_phys/physics}"

bash scripts/gpu/run_isaac_on_5090.sh -- \
  bash scripts/isaac/run_with_isaac_python.sh -- \
    scripts/isaac/phys_spawn_settle_v34e.py \
      --stage "${STAGE}" \
      --out_dir "${OUT_DIR}" \
      --x "${V34E_SPAWN_X:--2.8}" \
      --z "${V34E_SPAWN_Z:--2.5}" \
      --clearance "${V34E_SPAWN_CLEARANCE:-0.06}" \
      --settle_steps "${V34E_SETTLE_STEPS:-60}"
