#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "${ROOT_DIR}"

ASSETS_ROOT="${ISAAC_ASSETS_ROOT:-/home/peng/isaacsim_assets}"
STAGE_DEFAULT="${ASSETS_ROOT}/Office/office.usd"
if [ ! -f "${STAGE_DEFAULT}" ] && [ -f "${ASSETS_ROOT}/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd" ]; then
  STAGE_DEFAULT="${ASSETS_ROOT}/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"
fi

STAGE="${STAGE_DEFAULT}"
OUT_DIR="${ROOT_DIR}/runs/topo_mvp/v28_isaac/asset_audit"
ALLOW_MISSING="${ALLOW_MISSING:-0}"
ALLOW_UNREADABLE="${ALLOW_UNREADABLE:-0}"
SAFE_CHECK="${SAFE_CHECK:-1}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage)
      STAGE="${2:-${STAGE}}"
      shift 2
      ;;
    --assets_root)
      ASSETS_ROOT="${2:-${ASSETS_ROOT}}"
      shift 2
      ;;
    --out_dir)
      OUT_DIR="${2:-${OUT_DIR}}"
      shift 2
      ;;
    --allow_missing)
      ALLOW_MISSING="${2:-${ALLOW_MISSING}}"
      shift 2
      ;;
    --allow_unreadable)
      ALLOW_UNREADABLE="${2:-${ALLOW_UNREADABLE}}"
      shift 2
      ;;
    --safe_check)
      SAFE_CHECK="${2:-${SAFE_CHECK}}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OUT_DIR}"

bash scripts/isaac/run_with_isaac_python.sh -- \
  -m internnav.sim_backend.isaac_asset_audit_v28 \
  --stage "${STAGE}" \
  --assets_root "${ASSETS_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --allow_missing "${ALLOW_MISSING}" \
  --allow_unreadable "${ALLOW_UNREADABLE}" \
  --safe_check "${SAFE_CHECK}"
