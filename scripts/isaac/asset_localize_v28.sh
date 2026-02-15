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
AUDIT_DIR="${ROOT_DIR}/runs/topo_mvp/v28_isaac/asset_audit"
OUT_ROOT="${ROOT_DIR}/runs/topo_mvp/v28_isaac/assets_local"
ALLOW_OUTSIDE_ROOT="${ALLOW_OUTSIDE_ROOT:-0}"

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
    --out_root)
      OUT_ROOT="${2:-${OUT_ROOT}}"
      shift 2
      ;;
    --allow_outside_root)
      ALLOW_OUTSIDE_ROOT="${2:-${ALLOW_OUTSIDE_ROOT}}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${AUDIT_DIR}" "${OUT_ROOT}"
AUDIT_JSON="${AUDIT_DIR}/asset_audit_report.json"

if [ ! -f "${AUDIT_JSON}" ]; then
  bash scripts/isaac/asset_audit_v28.sh \
    --stage "${STAGE}" \
    --assets_root "${ASSETS_ROOT}" \
    --out_dir "${AUDIT_DIR}" \
    --allow_missing 0 \
    --allow_unreadable 0 \
    --safe_check 1
fi

bash scripts/isaac/run_with_isaac_python.sh -- \
  -m internnav.sim_backend.isaac_asset_localize_v28 \
  --audit_report "${AUDIT_JSON}" \
  --stage "${STAGE}" \
  --assets_root "${ASSETS_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --allow_outside_root "${ALLOW_OUTSIDE_ROOT}"
