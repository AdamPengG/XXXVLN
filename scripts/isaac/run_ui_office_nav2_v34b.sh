#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34B_OUT_ROOT:-runs/topo_mvp/v34b_nav2_demo}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/isaac_ui_probe_v34b.log"
: > "${LOG_FILE}"

SCENE_ID="${V34B_SCENE_ID:-office_localized}"
CFG="${V34B_ISAAC_CONFIG:-configs/isaac_scenes_v34b.yaml}"
HEADLESS="${V34B_HEADLESS:-0}"

resolve_stage() {
  local source="official_assets"
  local stage=""
  if [ -n "${ISAAC_STAGE_USD:-}" ]; then
    stage="${ISAAC_STAGE_USD}"
    source="override"
    echo "${stage}|${source}"
    return 0
  fi

  local candidates=()
  if [ -n "${ISAAC_ASSETS_ROOT:-}" ]; then
    candidates+=("${ISAAC_ASSETS_ROOT}")
  fi
  candidates+=("/home/peng/IsaacAssets" "/home/peng/isaacsim_assets")

  for root in "${candidates[@]}"; do
    [ -n "${root}" ] || continue
    for p in \
      "${root}/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd" \
      "${root}/Isaac/Environments/Office/office.usd" \
      "${root}/Office/office.usd"; do
      if [ -f "${p}" ]; then
        stage="${p}"
        echo "${stage}|${source}"
        return 0
      fi
    done
  done

  stage="$(python3 - <<'PY' "${CFG}" "${SCENE_ID}"
import json, sys
from pathlib import Path
cfg = Path(sys.argv[1])
scene_id = sys.argv[2]
if not cfg.is_file():
    print("")
    raise SystemExit(0)
text = cfg.read_text(encoding='utf-8')
obj = None
try:
    import yaml  # type: ignore
    obj = yaml.safe_load(text)
except Exception:
    try:
        obj = json.loads(text)
    except Exception:
        obj = {}
scenes = obj.get('scenes', []) if isinstance(obj, dict) else []
for row in scenes:
    if str(row.get('scene_id', '')) == scene_id:
        print(str(row.get('usd_path', '')))
        break
else:
    print("")
PY
)"
  source="config"
  echo "${stage}|${source}"
}

stage_info="$(resolve_stage)"
STAGE_PATH="${stage_info%%|*}"
STAGE_SOURCE="${stage_info##*|}"
STAGE_EXISTS=0
if [ -n "${STAGE_PATH}" ] && [ -f "${STAGE_PATH}" ]; then
  STAGE_EXISTS=1
fi
export ISAAC_STAGE_USD="${STAGE_PATH}"

echo "[V34B_ISAAC_STAGE] usd=${STAGE_PATH} exists=${STAGE_EXISTS} source=${STAGE_SOURCE}" | tee -a "${LOG_FILE}"
if [ "${STAGE_EXISTS}" != "1" ]; then
  echo "[V34B_ISAAC_UI] ok=0 stage=${STAGE_PATH} robot=unknown ros2_bridge=0 scan=0 odom=0 tf=0 reason=stage_missing" | tee -a "${LOG_FILE}"
  exit 2
fi

export ISAAC_STEP_RENDER=1
export ISAAC_STEP_RENDER_EVERY_N=1
export ISAAC_ENABLE_DEPTH=1
export ISAAC_HEADLESS="${HEADLESS}"

{
  echo "[V34B_UI_LAUNCH] start=$(date -Iseconds) scene=${SCENE_ID} headless=${HEADLESS} cfg=${CFG}"
  python3 scripts/gpu/gpu_router.py --role isaac --output anchors
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    bash scripts/isaac/run_with_isaac_python.sh -- \
      scripts/isaac/ui_office_nav2_probe_v34b.py \
        --scene_id "${SCENE_ID}" \
        --isaac_config "${CFG}" \
        --stage "${STAGE_PATH}" \
        --out_dir "${OUT_ROOT}/ui_probe"
  echo "[V34B_UI_LAUNCH] done log=${LOG_FILE}"
} 2>&1 | tee -a "${LOG_FILE}"
