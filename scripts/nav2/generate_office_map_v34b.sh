#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_DIR="${V34B_MAP_OUT_DIR:-runs/topo_mvp/v34b_nav2_demo/map}"
SCENE_ID="${V34B_SCENE_ID:-office_localized}"
CFG="${V34B_ISAAC_CONFIG:-configs/isaac_scenes_v34b.yaml}"

python3 scripts/nav2/generate_office_map_v34b.py --scene_id "${SCENE_ID}" --isaac_config "${CFG}" --out_dir "${OUT_DIR}"
