#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"

CAPTURE_DIR=""
TAG=""
WRITE_GIF="${WRITE_GIF:-1}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --capture_dir)
      CAPTURE_DIR="${2:-}"
      shift 2
      ;;
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --write_gif)
      WRITE_GIF="${2:-${WRITE_GIF}}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "${CAPTURE_DIR}" ]; then
  echo "Usage: bash scripts/tools/check_capture_motion.sh --capture_dir <dir> [--tag <tag>] [--write_gif 0|1]" >&2
  exit 2
fi

python3 scripts/tools/check_capture_motion.py \
  --capture_dir "${CAPTURE_DIR}" \
  --tag "${TAG}" \
  --write_gif "${WRITE_GIF}"
