#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"

RUN_DIR="${ROOT_DIR}/runs/topo_mvp/v33b4_depth_fidelity"
OUT_DIR="${RUN_DIR}/capture"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

ISAAC_CFG="${ISAAC_CFG:-${ROOT_DIR}/configs/isaac_scenes_v29.yaml}"
ISAAC_SCENE_ID="${ISAAC_SCENE_ID:-office_localized}"
ISAAC_STAGE_USD="${ISAAC_STAGE_USD:-}"
FRAMES="${ISAAC_DEPTH_FIDELITY_FRAMES:-10}"
LOG_FILE="${LOG_DIR}/depth_fidelity_v33b4.log"

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
    --frames)
      FRAMES="${2:-${FRAMES}}"
      shift 2
      ;;
    --out_dir)
      OUT_DIR="${2:-${OUT_DIR}}"
      RUN_DIR="$(dirname "${OUT_DIR}")"
      LOG_DIR="${RUN_DIR}/logs"
      LOG_FILE="${LOG_DIR}/depth_fidelity_v33b4.log"
      mkdir -p "${OUT_DIR}" "${LOG_DIR}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "${ISAAC_STAGE_USD}" ]; then
  ISAAC_STAGE_USD="$(python3 - <<'PY' "${ISAAC_CFG}" "${ISAAC_SCENE_ID}"
import json, sys
from pathlib import Path
cfg = Path(sys.argv[1])
scene_id = sys.argv[2]
data = {}
if cfg.exists():
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(cfg.read_text()) or {}
    except Exception:
        try:
            data = json.loads(cfg.read_text())
        except Exception:
            data = {}
for row in data.get('scenes', []) if isinstance(data, dict) else []:
    if str(row.get('scene_id', '')) == scene_id:
        print(str(row.get('usd_path', '')))
        break
PY
)"
fi

if [ ! -f "${ISAAC_CFG}" ]; then
  echo "[ISAAC_DEPTH_FIDELITY] ok=0 reason=config_missing hint=${ISAAC_CFG}" >&2
  exit 2
fi
if [ -n "${ISAAC_STAGE_USD}" ] && [ ! -f "${ISAAC_STAGE_USD}" ]; then
  echo "[ISAAC_DEPTH_FIDELITY] ok=0 reason=stage_missing hint=${ISAAC_STAGE_USD}" >&2
  exit 2
fi

export ISAAC_RENDERER="${ISAAC_RENDERER:-rtx}"
export ISAAC_ENABLE_DEPTH=1
export ISAAC_RGB_CAPTURE=1
export ISAAC_RENDER=1
export ISAAC_SKIP_WORLD_STEP=0
export ISAAC_MINIMAL=0
export ISAAC_CAM_W="${ISAAC_CAM_W:-1280}"
export ISAAC_CAM_H="${ISAAC_CAM_H:-720}"
export ISAAC_CAM_FOV_DEG="${ISAAC_CAM_FOV_DEG:-90}"
export ISAAC_CAM_NEAR="${ISAAC_CAM_NEAR:-0.05}"
export ISAAC_CAM_FAR="${ISAAC_CAM_FAR:-50.0}"
export ISAAC_CAM_AUTO_EXPOSURE="${ISAAC_CAM_AUTO_EXPOSURE:-0}"
export ISAAC_CAM_EXPOSURE="${ISAAC_CAM_EXPOSURE:-1.0}"
export ISAAC_CAM_ASPECT_POLICY="${ISAAC_CAM_ASPECT_POLICY:-locked_w_over_h}"

echo "[ISAAC_DEPTH_RUN] cfg=${ISAAC_CFG} scene=${ISAAC_SCENE_ID} stage=${ISAAC_STAGE_USD} out_dir=${OUT_DIR}" | tee "${LOG_FILE}"

bash scripts/gpu/run_isaac_on_5090.sh -- \
  bash scripts/isaac/run_with_isaac_python.sh -- \
    scripts/isaac/depth_fidelity_v33b4.py \
    --scene_id "${ISAAC_SCENE_ID}" \
    --isaac_config "${ISAAC_CFG}" \
    --out_dir "${OUT_DIR}" \
    --frames "${FRAMES}" \
    --stage "${ISAAC_STAGE_USD}" 2>&1 | tee -a "${LOG_FILE}"

