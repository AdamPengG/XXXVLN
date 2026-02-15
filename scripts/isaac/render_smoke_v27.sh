#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "${ROOT_DIR}"

RUN_DIR="${ISAAC_SMOKE_RUN_DIR:-${ROOT_DIR}/runs/topo_mvp/v27_isaac}"
OUT_DIR="${RUN_DIR}/render_smoke"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

ISAAC_CFG="${ISAAC_CFG:-${ROOT_DIR}/configs/isaac_scenes_v27.yaml}"
ISAAC_SCENE_ID="${ISAAC_SCENE_ID:-isaac_office_min}"
ISAAC_RENDERER="${ISAAC_RENDERER:-rtx}"
ISAAC_SMOKE_FRAMES="${ISAAC_SMOKE_FRAMES:-3}"
ISAAC_SMOKE_FAIL_NON_5090="${ISAAC_SMOKE_FAIL_NON_5090:-1}"
ISAAC_ASSETS_ROOT="${ISAAC_ASSETS_ROOT:-/home/peng/isaacsim_assets}"
CACHE_DIR="${ISAAC_CACHE_DIR:-${HOME}/.cache/ov}"
KIT_LOG_DIR="${ISAAC_KIT_LOG_DIR:-${HOME}/.nvidia-omniverse/logs}"
ISAAC_STAGE_USD="${ISAAC_STAGE_USD:-}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage)
      ISAAC_STAGE_USD="${2:-}"
      shift 2
      ;;
    --cfg|--config)
      ISAAC_CFG="${2:-${ISAAC_CFG}}"
      shift 2
      ;;
    --scene)
      ISAAC_SCENE_ID="${2:-${ISAAC_SCENE_ID}}"
      shift 2
      ;;
    --out_dir)
      OUT_DIR="${2:-${OUT_DIR}}"
      RUN_DIR="$(dirname "${OUT_DIR}")"
      LOG_DIR="${RUN_DIR}/logs"
      mkdir -p "${OUT_DIR}" "${LOG_DIR}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${CACHE_DIR}" "${KIT_LOG_DIR}"
touch "${LOG_DIR}/.write_test" && rm -f "${LOG_DIR}/.write_test"
touch "${OUT_DIR}/.write_test" && rm -f "${OUT_DIR}/.write_test"
touch "${CACHE_DIR}/.write_test" && rm -f "${CACHE_DIR}/.write_test"
touch "${KIT_LOG_DIR}/.write_test" && rm -f "${KIT_LOG_DIR}/.write_test"

echo "[ISAAC_RENDER_ENV] root=${ROOT_DIR} isaac_cfg=${ISAAC_CFG} scene=${ISAAC_SCENE_ID} renderer=${ISAAC_RENDERER}"
echo "[ISAAC_RENDER_ENV] isaac_sim_root=${ISAAC_SIM_ROOT:-} assets_root=${ISAAC_ASSETS_ROOT}"
echo "[ISAAC_RENDER_ENV] writable_logs=1 writable_out=1 writable_cache=1 writable_kit_logs=1"
if [ -n "${ISAAC_STAGE_USD}" ]; then
  echo "[ISAAC_RENDER_ENV] stage_override=${ISAAC_STAGE_USD}"
fi

python3 scripts/gpu/gpu_router.py --role isaac --output anchors
pick_json="$(python3 scripts/gpu/gpu_router.py --role isaac --output json --quiet)"
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

export ISAAC_GPU_ID="${gpu_id}"
export ISAAC_GPU_NAME="${gpu_name}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo "[GPU_PICK] role=isaac gpu_id=${gpu_id} gpu_name=\"${gpu_name}\" reason=render_smoke_v27"

if [ "${ISAAC_SMOKE_FAIL_NON_5090}" = "1" ] && [[ "${gpu_name}" != *"5090"* ]]; then
  echo '[ISAAC_RENDER_SMOKE] ok=0 reason=unsupported_or_failed_gpu hint="use ISAAC_GPU_ID=0 (5090)"'
  exit 3
fi

if [ ! -f "${ISAAC_CFG}" ]; then
  echo "[ISAAC_RENDER_SMOKE] ok=0 reason=config_missing hint=${ISAAC_CFG}"
  exit 2
fi

export ISAAC_RENDERER
export ISAAC_RGB_CAPTURE=1
export ISAAC_RENDER=1
export ISAAC_SKIP_WORLD_STEP=0
export ISAAC_MINIMAL=0

bash scripts/isaac/run_with_isaac_python.sh -- \
  -m internnav.sim_backend.isaac_render_smoke \
  --scene_id "${ISAAC_SCENE_ID}" \
  --isaac_config "${ISAAC_CFG}" \
  --out_dir "${OUT_DIR}" \
  --frames "${ISAAC_SMOKE_FRAMES}" \
  --fail_non_5090 "${ISAAC_SMOKE_FAIL_NON_5090}" \
  --stage "${ISAAC_STAGE_USD}"

python3 - "${OUT_DIR}/smoke_meta.json" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print(f"[ISAAC_RENDER_SMOKE] ok=0 reason=missing_smoke_meta hint={p}")
    raise SystemExit(5)
d = json.loads(p.read_text())
ratio = float(d.get("placeholder_ratio", 1.0))
frames = int(d.get("frames", 0))
if frames < 1 or ratio >= 0.95:
    print(
        f"[ISAAC_RENDER_SMOKE] ok=0 reason=placeholder_or_empty "
        f"out_dir={p.parent} frames={frames} placeholder_ratio={ratio:.3f} hint=check_gpu_and_renderer"
    )
    raise SystemExit(4)
print(
    f"[ISAAC_RENDER_SMOKE] ok=1 out_dir={p.parent} frames={frames} "
    f"placeholder_ratio={ratio:.3f}"
)
PY

exit 0
