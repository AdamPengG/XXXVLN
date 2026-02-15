#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"

# Keep authoring deterministic and fast unless caller overrides.
export ISAAC_CAMERA_PROBE_ON_RESET="${ISAAC_CAMERA_PROBE_ON_RESET:-0}"
export ISAAC_RENDER="${ISAAC_RENDER:-0}"
export ISAAC_RGB_CAPTURE="${ISAAC_RGB_CAPTURE:-0}"

mode=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [ "${args[$i]}" = "--mode" ] && [ $((i+1)) -lt ${#args[@]} ]; then
    mode="${args[$((i+1))]}"
    break
  fi
done

if [ "${mode}" = "record" ]; then
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    bash scripts/isaac/run_with_isaac_python.sh -- \
    scripts/isaac/goal_author_v30.py "$@"
else
  export PYTHONPATH="${ROOT_DIR}/repo/InternNav:${PYTHONPATH:-}"
  python3 scripts/isaac/goal_author_v30.py "$@"
fi
