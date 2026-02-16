#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"

BASE_CFG="${V33B_BASE_CFG:-configs/isaac_planb_eval_v32.yaml}"
BASELINE_ROOT="${V33B_BASELINE_ROOT:-runs/topo_mvp/v33b_baseline}"
V33B_ROOT="${V33B_OUT_ROOT:-runs/topo_mvp/v33b_costmap_avoid}"
LOG_DIR_BASE="${BASELINE_ROOT}/logs"
LOG_DIR_V33B="${V33B_ROOT}/logs"
RUNTIME_DIR="${V33B_ROOT}/runtime"
RUNTIME_CFG_BASE="${RUNTIME_DIR}/isaac_planb_eval_v33b_baseline.yaml"
RUNTIME_CFG_V33B="${RUNTIME_DIR}/isaac_planb_eval_v33b.yaml"
MASTER_LOG="${LOG_DIR_V33B}/topo_eval_suite_v33b_costmap_avoid.log"
BASELINE_LOG="${LOG_DIR_BASE}/topo_eval_suite_v33b_baseline.log"
V33B_LOG="${LOG_DIR_V33B}/topo_eval_suite_v33b_run.log"

mkdir -p "${LOG_DIR_BASE}" "${LOG_DIR_V33B}" "${RUNTIME_DIR}"
: > "${MASTER_LOG}"
: > "${BASELINE_LOG}"
: > "${V33B_LOG}"

SMALL_ARG=""
if [ "${1:-}" = "--small" ]; then
  SMALL_ARG="--small"
fi

run_capture_motion() {
  local capture_dir="$1"
  local tag="$2"
  if [ -z "${capture_dir}" ] || [ ! -d "${capture_dir}" ]; then
    echo "[CAPTURE_MOTION] frames=0 identical_pairs=0 mean_rgb_diff=0.000000 mean_depth_diff=0.000000 reason=missing_capture_dir tag=${tag}" | tee -a "${MASTER_LOG}"
    return 0
  fi
  (
    bash scripts/tools/check_capture_motion.sh --capture_dir "${capture_dir}" --tag "${tag}"
  ) 2>&1 | tee -a "${MASTER_LOG}"
}

find_representative_capture_dir() {
  local run_root="$1"
  python3 - "$run_root" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
fc = root / "failure_cases.jsonl"
if fc.exists():
    for line in fc.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        dbg = str(rec.get("debug_index", "") or "").strip()
        if not dbg:
            continue
        p = pathlib.Path(dbg)
        if p.exists():
            run_dir = p.parent.parent
            if (run_dir / "frames").exists():
                print(str(run_dir))
                raise SystemExit(0)
            print(str(run_dir))
            raise SystemExit(0)
for p in sorted((root / "debug_runs").glob("*/*")):
    if p.is_dir():
        print(str(p))
        raise SystemExit(0)
print("")
PY
}

python3 - "$BASE_CFG" "$RUNTIME_CFG_BASE" "$RUNTIME_CFG_V33B" "$BASELINE_ROOT" "$V33B_ROOT" "$SMALL_ARG" <<'PY'
import json, pathlib, sys
base = pathlib.Path(sys.argv[1])
out_base = pathlib.Path(sys.argv[2])
out_v33b = pathlib.Path(sys.argv[3])
root_base = sys.argv[4]
root_v33b = sys.argv[5]
small = bool(str(sys.argv[6]).strip())
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
starts = cfg.get('starts', []) if isinstance(cfg.get('starts', []), list) else []
if small and starts:
    starts = sorted(starts, key=lambda s: 0 if str((s or {}).get('id', '')) == 'start_003' else 1)
    cfg['starts'] = starts
cfg_base = json.loads(json.dumps(cfg))
cfg_v33b = json.loads(json.dumps(cfg))
out_cfg_base = cfg_base.get('output', {}) if isinstance(cfg_base.get('output', {}), dict) else {}
out_cfg_base['root'] = root_base
cfg_base['output'] = out_cfg_base
out_cfg_v33b = cfg_v33b.get('output', {}) if isinstance(cfg_v33b.get('output', {}), dict) else {}
out_cfg_v33b['root'] = root_v33b
cfg_v33b['output'] = out_cfg_v33b
for out, payload in [(out_base, cfg_base), (out_v33b, cfg_v33b)]:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding='utf-8')
    except Exception:
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
print(
    f"[V33B_CFG] base={base} baseline_cfg={out_base} v33b_cfg={out_v33b} "
    f"max_steps={run['max_steps']} baseline_root={root_base} v33b_root={root_v33b}"
)
PY

echo "[V33B_SUITE] start=$(date -Iseconds) baseline_cfg=${RUNTIME_CFG_BASE} v33b_cfg=${RUNTIME_CFG_V33B} small=${SMALL_ARG:-0}" | tee -a "${MASTER_LOG}"
python3 scripts/gpu/gpu_router.py --role isaac --output anchors | tee -a "${MASTER_LOG}"

# Depth fidelity gate (must pass before v33b A/B).
(
  export ISAAC_STEP_RENDER="${ISAAC_STEP_RENDER:-1}"
  export ISAAC_STEP_RENDER_EVERY_N="${ISAAC_STEP_RENDER_EVERY_N:-1}"
  bash scripts/isaac/depth_fidelity_v33b4.sh
) 2>&1 | tee -a "${MASTER_LOG}"
run_capture_motion "${ROOT_DIR}/runs/topo_mvp/v33b4_depth_fidelity/capture" "depth_fidelity"

# Baseline (same runtime config, v33b disabled)
(
  export ISAAC_ENABLE_DEPTH="${ISAAC_ENABLE_DEPTH:-1}"
  export ISAAC_STEP_RENDER="${ISAAC_STEP_RENDER:-1}"
  export ISAAC_STEP_RENDER_EVERY_N="${ISAAC_STEP_RENDER_EVERY_N:-1}"
  export V33B_ENABLE=0
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
  export V33B_DEPTH_KEY="${V33B_DEPTH_KEY:-obs.depth}"
  export V33B_STALE_DPOS_EPS="${V33B_STALE_DPOS_EPS:-0.02}"
  export V33B_STALE_DYAW_EPS="${V33B_STALE_DYAW_EPS:-2.0}"
  export V33B_STALE_K="${V33B_STALE_K:-3}"
  export V33B_STATIONARY_K="${V33B_STATIONARY_K:-20}"
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    python3 scripts/topo_planb_scale_eval_v31.py --config "${RUNTIME_CFG_BASE}" ${SMALL_ARG}
) 2>&1 | tee -a "${BASELINE_LOG}" "${MASTER_LOG}"

(
  bash scripts/topo_failure_cards_v31.sh \
    "${BASELINE_ROOT}/failure_cases.jsonl" \
    "${BASELINE_ROOT}/failure_cards"
) 2>&1 | tee -a "${BASELINE_LOG}" "${MASTER_LOG}"
BASELINE_CAPTURE_DIR="$(find_representative_capture_dir "${BASELINE_ROOT}")"
run_capture_motion "${BASELINE_CAPTURE_DIR}" "baseline_repr"

# v33b enabled run (same runtime config)
(
  export ISAAC_ENABLE_DEPTH="${ISAAC_ENABLE_DEPTH:-1}"
  export ISAAC_STEP_RENDER="${ISAAC_STEP_RENDER:-1}"
  export ISAAC_STEP_RENDER_EVERY_N="${ISAAC_STEP_RENDER_EVERY_N:-1}"
  export V33B_ENABLE=1
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
  export V33B_DEPTH_KEY="${V33B_DEPTH_KEY:-obs.depth}"
  export V33B_STALE_DPOS_EPS="${V33B_STALE_DPOS_EPS:-0.02}"
  export V33B_STALE_DYAW_EPS="${V33B_STALE_DYAW_EPS:-2.0}"
  export V33B_STALE_K="${V33B_STALE_K:-3}"
  export V33B_STATIONARY_K="${V33B_STATIONARY_K:-20}"
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    python3 scripts/topo_planb_scale_eval_v31.py --config "${RUNTIME_CFG_V33B}" ${SMALL_ARG}
) 2>&1 | tee -a "${V33B_LOG}" "${MASTER_LOG}"

(
  bash scripts/topo_failure_cards_v31.sh \
    "${V33B_ROOT}/failure_cases.jsonl" \
    "${V33B_ROOT}/failure_cards"
) 2>&1 | tee -a "${V33B_LOG}" "${MASTER_LOG}"
V33B_CAPTURE_DIR="$(find_representative_capture_dir "${V33B_ROOT}")"
run_capture_motion "${V33B_CAPTURE_DIR}" "v33b_repr"

python3 - "${BASELINE_ROOT}/summary.json" "${V33B_ROOT}/summary.json" <<'PY' | tee -a "${MASTER_LOG}"
import json, pathlib, sys
base_p = pathlib.Path(sys.argv[1])
new_p = pathlib.Path(sys.argv[2])
if not base_p.exists() or not new_p.exists():
    raise SystemExit(2)
base = json.loads(base_p.read_text(encoding='utf-8'))
new = json.loads(new_p.read_text(encoding='utf-8'))
def c(obj, key):
    d = obj.get('by_fail_type', {}) if isinstance(obj.get('by_fail_type', {}), dict) else {}
    return int(d.get(key, 0) or 0)
print(
    "[V33B_COMPARE] "
    f"baseline=success:{int(base.get('success', 0))},max_steps:{c(base,'max_steps')},stuck:{c(base,'stuck')},invalid_goal:{c(base,'invalid_goal')} "
    f"vs v33b=success:{int(new.get('success', 0))},max_steps:{c(new,'max_steps')},stuck:{c(new,'stuck')},invalid_goal:{c(new,'invalid_goal')}"
)
depth_key_used = "unset"
v33b = new.get("v33b", {}) if isinstance(new.get("v33b", {}), dict) else {}
dk = v33b.get("depth_key_used", {})
if isinstance(dk, dict) and len(dk) > 0:
    depth_key_used = ",".join(f"{k}:{v}" for k, v in sorted(dk.items()))
print(f"[V33B_DEPTH_KEY] used={depth_key_used}")
PY

DEPTH_KEY_USED="$(python3 - "${V33B_ROOT}/summary.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    print("unset unset unset")
    raise SystemExit(0)
obj = json.loads(p.read_text(encoding='utf-8'))
v = obj.get("v33b", {}) if isinstance(obj.get("v33b", {}), dict) else {}
dk = v.get("depth_key_used", {})
if isinstance(dk, dict) and len(dk) > 0:
    key_s = ",".join(f"{k}:{v}" for k, v in sorted(dk.items()))
else:
    key_s = "unset"
stale_runs = int(v.get("depth_stale_pose_moved_runs", v.get("depth_stale_runs", 0)) or 0)
stationary_runs = int(v.get("robot_stationary_runs", 0) or 0)
print(f"{key_s} {stale_runs} {stationary_runs}")
PY
)"
DEPTH_KEY_VALUE="$(echo "${DEPTH_KEY_USED}" | awk '{print $1}')"
DEPTH_STALE_RUNS="$(echo "${DEPTH_KEY_USED}" | awk '{print $2}')"
ROBOT_STATIONARY_RUNS="$(echo "${DEPTH_KEY_USED}" | awk '{print $3}')"
echo "[V33B_SUITE_OK] baseline_root=${BASELINE_ROOT} v33b_root=${V33B_ROOT} baseline_summary=${BASELINE_ROOT}/summary.json v33b_summary=${V33B_ROOT}/summary.json v33b_cards=${V33B_ROOT}/failure_cards/index.html depth_key_used=${DEPTH_KEY_VALUE} depth_stale_pose_moved_runs=${DEPTH_STALE_RUNS} robot_stationary_runs=${ROBOT_STATIONARY_RUNS}" | tee -a "${MASTER_LOG}"
