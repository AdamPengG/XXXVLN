#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"

OUT_ROOT="${V34A_OUT_ROOT:-runs/topo_mvp/v34a_dual_gpu}"
BASELINE_ROOT="${OUT_ROOT}/baseline"
DUALVLN_ROOT="${OUT_ROOT}/dualvln"
SERVER_DIR="${OUT_ROOT}/server"
LOG_DIR="${OUT_ROOT}/logs"
RUNTIME_DIR="${OUT_ROOT}/runtime"
BASE_CFG="${V34A_BASE_CFG:-configs/isaac_planb_eval_v34a.yaml}"
RUNTIME_CFG_BASE="${RUNTIME_DIR}/isaac_planb_eval_v34a_baseline.yaml"
RUNTIME_CFG_DUAL="${RUNTIME_DIR}/isaac_planb_eval_v34a_dualvln.yaml"
MASTER_LOG="${LOG_DIR}/topo_eval_suite_v34a_dual_gpu_dualvln.log"
BASELINE_LOG="${LOG_DIR}/topo_eval_baseline.log"
DUAL_LOG="${LOG_DIR}/topo_eval_dualvln.log"
SERVER_LOG="${SERVER_DIR}/dualvln_server.log"
SERVER_PID_FILE="${SERVER_DIR}/dualvln_server.pid"
SERVER_URL="${DUALVLN_SERVER_URL:-http://127.0.0.1:18080}"
SERVER_HOST="${DUALVLN_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${DUALVLN_SERVER_PORT:-18080}"
DUALVLN_MODEL_PATH="${DUALVLN_MODEL_PATH:-/home/peng/DualVLN/checkpoints/InternVLA-N1-DualVLN}"
DUALVLN_READY=0
DUALVLN_SKIP_REASON="not_started"
SMALL_ARG=""

if [ "${1:-}" = "--small" ]; then
  SMALL_ARG="--small"
fi

mkdir -p "${SERVER_DIR}" "${LOG_DIR}" "${RUNTIME_DIR}" "${BASELINE_ROOT}" "${DUALVLN_ROOT}"
: > "${MASTER_LOG}"
: > "${BASELINE_LOG}"
: > "${DUAL_LOG}"
: > "${SERVER_LOG}"

cleanup() {
  if [ -f "${SERVER_PID_FILE}" ]; then
    pid="$(cat "${SERVER_PID_FILE}" 2>/dev/null || true)"
    if [ -n "${pid}" ] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT

run_failure_cards_if_exists() {
  local run_root="$1"
  if [ -f "${run_root}/failure_cases.jsonl" ]; then
    (
      bash scripts/topo_failure_cards_v31.sh \
        "${run_root}/failure_cases.jsonl" \
        "${run_root}/failure_cards"
    ) 2>&1 | tee -a "${MASTER_LOG}"
  fi
}

find_capture_dir() {
  local run_root="$1"
  python3 - "$run_root" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
fc = root / "failure_cases.jsonl"
if fc.exists():
    for line in fc.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        dbg = str(rec.get("debug_index", "") or "").strip()
        if dbg:
            p = pathlib.Path(dbg)
            if p.exists():
                run_dir = p.parent.parent
                if run_dir.exists():
                    print(str(run_dir))
                    raise SystemExit(0)
for p in sorted((root / "debug_runs").glob("*/*")):
    if p.is_dir():
        print(str(p))
        raise SystemExit(0)
print("")
PY
}

run_motion_check() {
  local run_root="$1"
  local tag="$2"
  local cap_dir
  cap_dir="$(find_capture_dir "${run_root}")"
  if [ -z "${cap_dir}" ] || [ ! -d "${cap_dir}" ]; then
    echo "[V34A_MOTION] tag=${tag} frames=0 identical_pairs=0 mean_rgb_diff=0.000000 mean_depth_diff=0.000000 reason=missing_capture_dir" | tee -a "${MASTER_LOG}"
    return 0
  fi
  (
    bash scripts/tools/check_capture_motion.sh --capture_dir "${cap_dir}" --tag "${tag}"
  ) 2>&1 | tee -a "${MASTER_LOG}"
  local report_json="${cap_dir}/motion_report.json"
  if [ -f "${report_json}" ]; then
    python3 - "$report_json" "$tag" <<'PY' | tee -a "${MASTER_LOG}"
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); tag = sys.argv[2]
obj = json.loads(p.read_text(encoding='utf-8'))
print(
    f"[V34A_MOTION] tag={tag} frames={int(obj.get('frames', 0))} "
    f"identical_pairs={int(obj.get('identical_pairs', 0))} "
    f"mean_rgb_diff={float(obj.get('mean_rgb_diff', 0.0)):.6f} "
    f"mean_depth_diff={float(obj.get('mean_depth_diff', 0.0)):.6f}"
)
PY
  fi
}

python3 - "$BASE_CFG" "$RUNTIME_CFG_BASE" "$RUNTIME_CFG_DUAL" "$BASELINE_ROOT" "$DUALVLN_ROOT" <<'PY'
import json, pathlib, sys
base = pathlib.Path(sys.argv[1])
out_base = pathlib.Path(sys.argv[2])
out_dual = pathlib.Path(sys.argv[3])
root_base = sys.argv[4]
root_dual = sys.argv[5]
text = base.read_text(encoding='utf-8')
try:
    import yaml
    cfg = yaml.safe_load(text)
except Exception:
    cfg = json.loads(text)
if not isinstance(cfg, dict):
    raise SystemExit("invalid_cfg")
cfg_b = json.loads(json.dumps(cfg))
cfg_d = json.loads(json.dumps(cfg))
run_b = cfg_b.get("run", {}) if isinstance(cfg_b.get("run", {}), dict) else {}
run_d = cfg_d.get("run", {}) if isinstance(cfg_d.get("run", {}), dict) else {}
run_b["max_steps"] = int(run_b.get("max_steps", 80))
run_d["max_steps"] = int(run_d.get("max_steps", 80))
cfg_b["run"] = run_b
cfg_d["run"] = run_d
outcfg_b = cfg_b.get("output", {}) if isinstance(cfg_b.get("output", {}), dict) else {}
outcfg_d = cfg_d.get("output", {}) if isinstance(cfg_d.get("output", {}), dict) else {}
outcfg_b["root"] = root_base
outcfg_d["root"] = root_dual
cfg_b["output"] = outcfg_b
cfg_d["output"] = outcfg_d
for out, payload in [(out_base, cfg_b), (out_dual, cfg_d)]:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding='utf-8')
    except Exception:
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"[V34A_CFG] base={base} baseline_cfg={out_base} dual_cfg={out_dual} baseline_root={root_base} dual_root={root_dual}")
PY

echo "[V34A_SUITE] start=$(date -Iseconds) out_root=${OUT_ROOT} server_url=${SERVER_URL}" | tee -a "${MASTER_LOG}"
echo "[GPU_SPLIT] isaac_gpu=0 dualvln_gpu=1" | tee -a "${MASTER_LOG}"
python3 scripts/gpu/gpu_router.py --role isaac --output anchors | tee -a "${MASTER_LOG}"
python3 scripts/gpu/gpu_router.py --role habitat --output anchors | tee -a "${MASTER_LOG}"

# Start DualVLN server on GPU1/2080Ti.
if [ -z "${DUALVLN_MODEL_PATH}" ]; then
  DUALVLN_SKIP_REASON="missing_DUALVLN_MODEL_PATH"
  echo "[V34A_DUALVLN_SKIP] reason=${DUALVLN_SKIP_REASON}" | tee -a "${MASTER_LOG}"
else
  (
    export CUDA_VISIBLE_DEVICES=1
    export DUALVLN_SERVER_GPU_ID=1
    export DUALVLN_MODEL_PATH="${DUALVLN_MODEL_PATH}"
    export DUALVLN_SERVER_HOST="${SERVER_HOST}"
    export DUALVLN_SERVER_PORT="${SERVER_PORT}"
    python3 scripts/dualvln/dualvln_server_v34a.py \
      --host "${SERVER_HOST}" \
      --port "${SERVER_PORT}" \
      --model_path "${DUALVLN_MODEL_PATH}"
  ) >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  echo "${SERVER_PID}" > "${SERVER_PID_FILE}"
  for _ in $(seq 1 120); do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      break
    fi
    if python3 - "$SERVER_URL" <<'PY'
import json, sys, urllib.request
url = sys.argv[1].rstrip("/") + "/health"
try:
    with urllib.request.urlopen(url, timeout=0.5) as r:
        obj = json.loads(r.read().decode("utf-8"))
    raise SystemExit(0 if int(obj.get("ok", 0)) == 1 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      DUALVLN_READY=1
      DUALVLN_SKIP_REASON=""
      break
    fi
    sleep 0.5
  done
  if [ "${DUALVLN_READY}" -ne 1 ]; then
    DUALVLN_SKIP_REASON="server_not_ready"
    if [ -f "${SERVER_LOG}" ]; then
      DUALVLN_SKIP_REASON="$(grep -E '\[DUALVLN_SERVER_ERR\]' "${SERVER_LOG}" | tail -n1 | sed 's/.*reason=//' || true)"
      DUALVLN_SKIP_REASON="${DUALVLN_SKIP_REASON:-server_not_ready}"
    fi
    echo "[V34A_DUALVLN_SKIP] reason=${DUALVLN_SKIP_REASON}" | tee -a "${MASTER_LOG}"
  fi
fi

# Baseline run (topo policy) on GPU0.
(
  export CUDA_VISIBLE_DEVICES=0
  export ISAAC_GPU_ID=0
  export ISAAC_STEP_RENDER=1
  export PLANB_POLICY=topo
  export DUALVLN_SERVER_GPU_ID=1
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    python3 scripts/topo_planb_scale_eval_v31.py --config "${RUNTIME_CFG_BASE}" ${SMALL_ARG}
) 2>&1 | tee -a "${BASELINE_LOG}" "${MASTER_LOG}"
run_failure_cards_if_exists "${BASELINE_ROOT}"
run_motion_check "${BASELINE_ROOT}" "baseline"

# DualVLN RPC run (same runtime config) on GPU0, only if server is healthy.
if [ "${DUALVLN_READY}" -eq 1 ]; then
  (
    export CUDA_VISIBLE_DEVICES=0
    export ISAAC_GPU_ID=0
    export ISAAC_STEP_RENDER=1
    export PLANB_POLICY=dualvln_rpc
    export DUALVLN_SERVER_URL="${SERVER_URL}"
    export DUALVLN_RPC_TIMEOUT_S="${DUALVLN_RPC_TIMEOUT_S:-5}"
    export DUALVLN_RPC_FALLBACK="${DUALVLN_RPC_FALLBACK:-topo}"
    export DUALVLN_SERVER_GPU_ID=1
    bash scripts/gpu/run_isaac_on_5090.sh -- \
      python3 scripts/topo_planb_scale_eval_v31.py --config "${RUNTIME_CFG_DUAL}" ${SMALL_ARG}
  ) 2>&1 | tee -a "${DUAL_LOG}" "${MASTER_LOG}"
  run_failure_cards_if_exists "${DUALVLN_ROOT}"
  run_motion_check "${DUALVLN_ROOT}" "dualvln"
else
  echo "[V34A_MOTION] tag=dualvln frames=0 identical_pairs=0 mean_rgb_diff=0.000000 mean_depth_diff=0.000000 reason=dualvln_skipped" | tee -a "${MASTER_LOG}"
fi

python3 - "${BASELINE_ROOT}/summary.json" "${DUALVLN_ROOT}/summary.json" "${DUALVLN_READY}" "${DUALVLN_SKIP_REASON}" <<'PY' | tee -a "${MASTER_LOG}"
import json, pathlib, sys
base_p = pathlib.Path(sys.argv[1])
dual_p = pathlib.Path(sys.argv[2])
dual_ready = int(sys.argv[3])
skip_reason = str(sys.argv[4])
if not base_p.exists():
    raise SystemExit(2)
base = json.loads(base_p.read_text(encoding='utf-8'))
def c(obj, key):
    d = obj.get('by_fail_type', {}) if isinstance(obj.get('by_fail_type', {}), dict) else {}
    return int(d.get(key, 0) or 0)
base_s = f"success:{int(base.get('success',0))},max_steps:{c(base,'max_steps')},stuck:{c(base,'stuck')},invalid_goal:{c(base,'invalid_goal')}"
if dual_ready == 1 and dual_p.exists():
    dual = json.loads(dual_p.read_text(encoding='utf-8'))
    dual_s = f"success:{int(dual.get('success',0))},max_steps:{c(dual,'max_steps')},stuck:{c(dual,'stuck')},invalid_goal:{c(dual,'invalid_goal')}"
    print(f"[V34A_COMPARE] baseline={base_s} vs dualvln={dual_s}")
else:
    print(f"[V34A_COMPARE] baseline={base_s} vs dualvln=skipped(reason={skip_reason})")
PY

echo "[V34A_SUITE_OK] root=${OUT_ROOT} baseline=${BASELINE_ROOT}/summary.json dualvln=${DUALVLN_ROOT}/summary.json server_log=${SERVER_LOG}" | tee -a "${MASTER_LOG}"

# Non-regression.
(
  bash scripts/topo_eval_suite_v32_planb_scale_eval.sh --small
) 2>&1 | tee -a "${MASTER_LOG}"

echo "[V34A_SUITE] done log=${MASTER_LOG}" | tee -a "${MASTER_LOG}"
