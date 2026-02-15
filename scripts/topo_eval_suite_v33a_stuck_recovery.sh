#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"

BASE_CFG="${V33A_BASE_CFG:-configs/isaac_planb_eval_v32.yaml}"
OUT_ROOT="${V33A_OUT_ROOT:-runs/topo_mvp/v33a_stuck_recovery}"
LOG_DIR="${OUT_ROOT}/logs"
RUNTIME_DIR="${OUT_ROOT}/runtime"
RUNTIME_CFG="${RUNTIME_DIR}/isaac_planb_eval_v33a.yaml"
MASTER_LOG="${LOG_DIR}/topo_eval_suite_v33a_stuck_recovery.log"

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
print(f"[V33A_CFG] base={base} runtime={out} max_steps={run['max_steps']} out_root={out_root}")
PY


echo "[V33A_SUITE] start=$(date -Iseconds) cfg=${RUNTIME_CFG} out_root=${OUT_ROOT} small=${SMALL_ARG:-0}" | tee -a "${MASTER_LOG}"
python3 scripts/gpu/gpu_router.py --role isaac --output anchors | tee -a "${MASTER_LOG}"

(
  export STUCK_WINDOW="${STUCK_WINDOW:-20}"
  export STUCK_DIST_EPS="${STUCK_DIST_EPS:-0.05}"
  export OSCILLATION_YAW_EPS="${OSCILLATION_YAW_EPS:-10}"
  export OSCILLATION_FLIP_MIN="${OSCILLATION_FLIP_MIN:-4}"
  export OSCILLATION_MIN_TURNS="${OSCILLATION_MIN_TURNS:-6}"
  export STUCK_RECOVERY_SCAN_K="${STUCK_RECOVERY_SCAN_K:-2}"
  export STUCK_MAX_RECOVERIES="${STUCK_MAX_RECOVERIES:-3}"
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    python3 scripts/topo_planb_scale_eval_v31.py --config "${RUNTIME_CFG}" ${SMALL_ARG}
) 2>&1 | tee -a "${MASTER_LOG}"

(
  bash scripts/topo_failure_cards_v31.sh \
    "${OUT_ROOT}/failure_cases.jsonl" \
    "${OUT_ROOT}/failure_cards"
) 2>&1 | tee -a "${MASTER_LOG}"

SUMMARY_JSON="${OUT_ROOT}/summary.json"
CARDS_HTML="${OUT_ROOT}/failure_cards/index.html"
python3 - "$SUMMARY_JSON" <<'PY' | tee -a "${MASTER_LOG}"
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    raise SystemExit(2)
s = json.loads(p.read_text(encoding='utf-8'))
unk = int(s.get('by_fail_type', {}).get('unknown', 0) or 0)
print(f"[V33A_SUITE_CHECK] unknown_failures={unk} by_fail_type={s.get('by_fail_type', {})}")
PY

echo "[V33A_SUITE_OK] root=${OUT_ROOT} summary=${SUMMARY_JSON} cards=${CARDS_HTML}" | tee -a "${MASTER_LOG}"
