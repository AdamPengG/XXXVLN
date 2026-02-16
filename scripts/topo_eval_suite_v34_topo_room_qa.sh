#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"

OUT_ROOT="${V34_QA_OUT_ROOT:-runs/topo_mvp/v34_topo_room_qa}"
LOG_DIR="${OUT_ROOT}/logs"
SUITE_LOG="${LOG_DIR}/topo_eval_suite_v34_topo_room_qa.log"
mkdir -p "${LOG_DIR}"
: > "${SUITE_LOG}"

echo "[V34_QA_SUITE] start=$(date -Iseconds) out_root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
python3 scripts/gpu/gpu_router.py --role isaac --output anchors | tee -a "${SUITE_LOG}"

(
  export ISAAC_STEP_RENDER=1
  export ISAAC_STEP_RENDER_EVERY_N=1
  export ISAAC_ENABLE_DEPTH=1
  export ISAAC_CAMERA_PROBE_ON_RESET=0
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    bash scripts/isaac/run_with_isaac_python.sh -- \
      scripts/topo_topomap_room_qa_v34.py \
        --scene_id office_localized \
        --isaac_config configs/isaac_scenes_v29.yaml \
        --out_root "${OUT_ROOT}" \
        --steps 80 \
        --fwd_steps 8 \
        --turn_steps 6 \
        --node_stride 4 \
        --save_frames 20 \
        --overlay_frames 20 \
        --start_offset 0 \
        --seed 0
) 2>&1 | tee -a "${SUITE_LOG}"

(
  bash scripts/tools/check_capture_motion.sh --capture_dir "${OUT_ROOT}/capture" --tag v34_topo_room_qa
) 2>&1 | tee -a "${SUITE_LOG}"

python3 - "${OUT_ROOT}/qa_metrics.json" "${OUT_ROOT}/overlay" <<'PY' | tee -a "${SUITE_LOG}"
import json, pathlib, sys
metrics_p = pathlib.Path(sys.argv[1])
overlay_dir = pathlib.Path(sys.argv[2])
if metrics_p.exists():
    obj = json.loads(metrics_p.read_text(encoding="utf-8"))
else:
    obj = {"ok": 0, "reason": "missing_metrics"}
print(
    f"[V34_SUITE_OK] root={metrics_p.parent} metrics={metrics_p} overlay_dir={overlay_dir} "
    f"ok={int(obj.get('ok', 0))}"
)
PY

echo "[V34_QA_SUITE] done log=${SUITE_LOG}" | tee -a "${SUITE_LOG}"
