#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34G_OUT_ROOT:-runs/topo_mvp/v34g_nav2_office_mapfix}"
OUT_DIR="${V34G_MAP_OUT_DIR:-${OUT_ROOT}/map}"
mkdir -p "${OUT_DIR}"

STAGE="${ISAAC_STAGE_USD:-/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd}"

bash scripts/isaac/run_with_isaac_python.sh -- \
  scripts/nav2/generate_office_map_v34g.py \
  --stage "${STAGE}" \
  --out_dir "${OUT_DIR}" \
  --resolution "${V34G_MAP_RES:-0.05}" \
  --inflate_m "${V34G_MAP_INFLATE_M:-0.20}" \
  --padding_m "${V34G_MAP_PAD_M:-0.50}" \
  --bounds="${V34G_MAP_BOUNDS:-}" \
  --min_occ_ratio "${V34G_MAP_OCC_MIN:-0.02}" \
  --max_occ_ratio "${V34G_MAP_OCC_MAX:-0.85}"
