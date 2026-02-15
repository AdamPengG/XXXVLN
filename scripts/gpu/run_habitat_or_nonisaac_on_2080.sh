#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "${ROOT_DIR}"

python3 scripts/gpu/gpu_router.py --role habitat --output anchors

pick_json="$(python3 scripts/gpu/gpu_router.py --role habitat --output json --quiet)"
gpu_id="$(python3 - <<'PY' "${pick_json}"
import json, sys
print(int(json.loads(sys.argv[1])["gpu_id"]))
PY
)"
gpu_name="$(python3 - <<'PY' "${pick_json}"
import json, sys
print(str(json.loads(sys.argv[1])["gpu_name"]))
PY
)"

export HABITAT_GPU_ID="${gpu_id}"
export HABITAT_GPU_NAME="${gpu_name}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo "[GPU_PICK] role=habitat gpu_id=${gpu_id} gpu_name=\"${gpu_name}\" reason=wrapper_route_habitat"
echo "[GPU_ROUTE] role=habitat CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [ "${1:-}" = "--" ]; then
  shift
fi
if [ "$#" -gt 0 ]; then
  exec "$@"
fi
