#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
MODE="${1:-}"
TARGET_DIR="${2:-}"
RUN_ROOT="${TOPO_DEBUG_ROOT:-${ROOT_DIR}/runs/topo_mvp/v20/debug_runs}"
BUILD_ROOT="${TOPO_BUILD_ROOT:-${ROOT_DIR}/runs/topo_mvp/v20/builds}"
PORT="${TOPO_UI_PORT:-8000}"

if [ "${MODE}" = "--build-only" ]; then
  if [ -z "${TARGET_DIR}" ]; then
    echo "usage: bash scripts/topo_debug_ui.sh --build-only <run_dir>" >&2
    exit 2
  fi
  python "${ROOT_DIR}/scripts/topo_debug_ui_build_report.py" \
    --run_dir "${TARGET_DIR}" \
    --build_root "${BUILD_ROOT}"
  echo "[TOPO_UI_BUILD_OK] run_dir=${TARGET_DIR}"
  exit 0
fi

if [ "${MODE}" = "--serve" ]; then
  ROOT="${TARGET_DIR:-${RUN_ROOT}}"
  echo "[TOPO_UI_SERVE] url=http://127.0.0.1:${PORT}/index.html"
  python -m http.server "${PORT}" --directory "${ROOT}"
  exit 0
fi

python "${ROOT_DIR}/scripts/topo_debug_ui_index.py" --root "${RUN_ROOT}"
echo "[TOPO_UI_SERVE] url=http://127.0.0.1:${PORT}/index.html"
python -m http.server "${PORT}" --directory "${RUN_ROOT}"
