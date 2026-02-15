#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"

BASE_CFG="${V33B_BASE_CFG:-configs/isaac_planb_eval_v32.yaml}"
OUT_ROOT="${V33B_OUT_ROOT:-runs/topo_mvp/v33b_costmap_avoid}"
BASELINE_ROOT="${V33B_BASELINE_ROOT:-runs/topo_mvp/v33a_stuck_recovery}"
LOG_DIR="${OUT_ROOT}/logs"
RUNTIME_DIR="${OUT_ROOT}/runtime"
RUNTIME_CFG="${RUNTIME_DIR}/isaac_planb_eval_v33b.yaml"
MASTER_LOG="${LOG_DIR}/topo_eval_suite_v33b_costmap_avoid.log"

mkdir -p "${LOG_DIR}" "${RUNTIME_DIR}"
: > "${MASTER_LOG}"

SMALL_ARG=""
if [ "${1:-}" = "--small" ]; then
  SMALL_ARG="--small"
fi

python3 - "$BASE_CFG" "$RUNTIME_CFG" "$OUT_ROOT" <<'PY'
import json, pathlib, sys
base = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
out_root = sys.argv[3]
text = base.read_text(encoding='utf-8')
try:
    import yaml
    cfg = yaml.safe_load(text)
except Exception:
    cfg = json.loads(text)
if not isinstance(cfg, dict):
    raise SystemExit('invalid_base_cfg')
run = cfg.get('run', {}) if isinstance(cfg.get('run', {}), dict) else {}
run['max_steps'] = int(run.get('max_steps', 80) if int(run.get('max_steps', 0) or 0) > 80 else 80)
cfg['run'] = run
out_cfg = cfg.get('output', {}) if isinstance(cfg.get('output', {}), dict) else {}
out_cfg['root'] = out_root
cfg['output'] = out_cfg
out.parent.mkdir(parents=True, exist_ok=True)
try:
    import yaml
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
except Exception:
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"[V33B_CFG] base={base} runtime={out} max_steps={run['max_steps']} out_root={out_root}")
PY

echo "[V33B_SUITE] start=$(date -Iseconds) cfg=${RUNTIME_CFG} out_root=${OUT_ROOT} baseline=${BASELINE_ROOT} small=${SMALL_ARG:-0}" | tee -a "${MASTER_LOG}"
python3 scripts/gpu/gpu_router.py --role isaac --output anchors | tee -a "${MASTER_LOG}"

if [ ! -f "${BASELINE_ROOT}/summary.json" ] || [ "${V33B_RERUN_BASELINE:-0}" = "1" ]; then
  echo "[V33B_BASELINE] running_v33a=1" | tee -a "${MASTER_LOG}"
  bash scripts/topo_eval_suite_v33a_stuck_recovery.sh ${SMALL_ARG} 2>&1 | tee -a "${MASTER_LOG}"
else
  echo "[V33B_BASELINE] running_v33a=0 using=${BASELINE_ROOT}/summary.json" | tee -a "${MASTER_LOG}"
fi

(
  export ISAAC_ENABLE_DEPTH="${ISAAC_ENABLE_DEPTH:-1}"
  export V33B_ENABLE="${V33B_ENABLE:-1}"
  export V33B_REQUIRE_DEPTH="${V33B_REQUIRE_DEPTH:-1}"
  export V33B_CLEARANCE_M="${V33B_CLEARANCE_M:-0.65}"
  export V33B_MIN_VALID_RATIO="${V33B_MIN_VALID_RATIO:-0.30}"
  export V33B_STRIP_W_FRAC="${V33B_STRIP_W_FRAC:-0.08}"
  export V33B_STRIP_Y0_FRAC="${V33B_STRIP_Y0_FRAC:-0.25}"
  export V33B_STRIP_Y1_FRAC="${V33B_STRIP_Y1_FRAC:-0.85}"
  export V33B_OFFSETS_DEG="${V33B_OFFSETS_DEG:-0,10,20,30,-10,-20,-30}"
  export V33B_BLOCK_PCTL="${V33B_BLOCK_PCTL:-20}"
  export V33B_TURN_STEP_DEG="${V33B_TURN_STEP_DEG:-15}"
  export V33B_MIN_IMPROVE_M="${V33B_MIN_IMPROVE_M:-0.10}"
  export V33B_STEER_HYST_STEPS="${V33B_STEER_HYST_STEPS:-5}"
  export V33B_DOORWAY_TRIGGER="${V33B_DOORWAY_TRIGGER:-1}"
  export V33B_DOORWAY_HYST_STEPS="${V33B_DOORWAY_HYST_STEPS:-8}"
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    python3 scripts/topo_planb_scale_eval_v31.py --config "${RUNTIME_CFG}" ${SMALL_ARG}
) 2>&1 | tee -a "${MASTER_LOG}"

(
  bash scripts/topo_failure_cards_v31.sh \
    "${OUT_ROOT}/failure_cases.jsonl" \
    "${OUT_ROOT}/failure_cards"
) 2>&1 | tee -a "${MASTER_LOG}"

python3 - "${BASELINE_ROOT}/summary.json" "${OUT_ROOT}/summary.json" <<'PY' | tee -a "${MASTER_LOG}"
import json, pathlib, sys
base_p = pathlib.Path(sys.argv[1])
new_p = pathlib.Path(sys.argv[2])
if not base_p.exists() or not new_p.exists():
    raise SystemExit(2)
base = json.loads(base_p.read_text(encoding='utf-8'))
new = json.loads(new_p.read_text(encoding='utf-8'))
base_fail = base.get('by_fail_type', {}) if isinstance(base.get('by_fail_type', {}), dict) else {}
new_fail = new.get('by_fail_type', {}) if isinstance(new.get('by_fail_type', {}), dict) else {}
print(
    "[V33B_COMPARE] "
    f"baseline=v33a success={int(base.get('success', 0))} "
    f"max_steps={int(base_fail.get('max_steps', 0) or 0)} stuck={int(base_fail.get('stuck', 0) or 0)} "
    f"vs v33b success={int(new.get('success', 0))} "
    f"max_steps={int(new_fail.get('max_steps', 0) or 0)} stuck={int(new_fail.get('stuck', 0) or 0)}"
)
PY

echo "[V33B_SUITE_OK] root=${OUT_ROOT} summary=${OUT_ROOT}/summary.json cards=${OUT_ROOT}/failure_cards/index.html" | tee -a "${MASTER_LOG}"
