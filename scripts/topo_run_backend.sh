#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/repo/InternNav:${PYTHONPATH:-}"

backend="habitat"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [ "${args[$i]}" = "--backend" ] && [ $((i+1)) -lt ${#args[@]} ]; then
    backend="${args[$((i+1))]}"
    break
  fi
done

if [ "${backend}" = "isaac" ]; then
  bash scripts/isaac/run_with_isaac_python.sh -- scripts/topo_backend_runner_isaac.py "$@"
else
  python scripts/topo_backend_runner.py "$@"
fi
