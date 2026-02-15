#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"

FAILURE_CASES="${1:-runs/topo_mvp/v31_planb_scale_eval/failure_cases.jsonl}"
OUT_DIR="${2:-runs/topo_mvp/v31_planb_scale_eval/failure_cards}"
python3 scripts/topo_failure_cards_v31.py --failure_cases "${FAILURE_CASES}" --out_dir "${OUT_DIR}"
