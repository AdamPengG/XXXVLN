#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
HINT="${1:-}"
ROOT="${ISAAC_SIM_ROOT:-}"

if [ -z "${HINT}" ]; then
  echo "Usage: bash scripts/isaac/find_usd_asset.sh <usd_hint>" >&2
  exit 2
fi

if [ -z "${ROOT}" ]; then
  ROOT_SPEC="$(bash ${ROOT_DIR}/scripts/isaac/find_isaac_root.sh 2>/dev/null || true)"
  if [ -n "${ROOT_SPEC}" ]; then
    ROOT="$(printf '%s' "${ROOT_SPEC}" | awk -F: '{print $NF}')"
  fi
fi

if [ -z "${ROOT}" ] || [ ! -d "${ROOT}" ]; then
  echo ""; exit 1
fi

FOUND="$(find "${ROOT}" -maxdepth 6 -type f -name "*${HINT}*" 2>/dev/null | head -n 1 || true)"
if [ -n "${FOUND}" ]; then
  echo "${FOUND}"; exit 0
fi
echo ""; exit 1
