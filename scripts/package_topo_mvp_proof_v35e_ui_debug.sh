#!/usr/bin/env bash
# package_topo_mvp_proof_v35e_ui_debug.sh — Create evidence zip for v35e.
# Strict ≤ 35 MB. Only keyframes, GIFs, pose traces, logs, source, git meta.
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

TS="$(date +%Y%m%d_%H%M)"

# Find the most recent v35e output dir
OUT_ROOT="${V35E_OUT_ROOT:-}"
if [ -z "${OUT_ROOT}" ]; then
  OUT_ROOT="$(find runs/topo_mvp -maxdepth 1 -type d -name 'v35e_ui_debug_*' 2>/dev/null | sort -r | head -1)"
fi
if [ -z "${OUT_ROOT}" ] || [ ! -d "${OUT_ROOT}" ]; then
  echo "[V35E_EVIDENCE] ok=0 reason=no_v35e_run_dir"
  exit 2
fi
echo "[V35E_EVIDENCE] out_root=${OUT_ROOT}"

ZIP_NAME="v35e_ui_debug_PROOF_${TS}.zip"
ZIP_PATH="/home/peng/DualVLN/forGPT/${ZIP_NAME}"
PROOF_DIR="/tmp/v35e_proof_${TS}"
rm -rf "${PROOF_DIR}"
mkdir -p "${PROOF_DIR}" "$(dirname "${ZIP_PATH}")"

# ── 1) Keyframes (max 12) ──
KF_DIR="${PROOF_DIR}/keyframes"
mkdir -p "${KF_DIR}"
KF_COUNT=0
EVIDENCE="${OUT_ROOT}/evidence"
if [ -d "${EVIDENCE}" ]; then
  for f in $(find "${EVIDENCE}" -name 'key_*.png' | sort | head -12); do
    cp "${f}" "${KF_DIR}/"
    KF_COUNT=$((KF_COUNT + 1))
  done
fi
echo "[V35E_PKG] keyframes=${KF_COUNT}"

# ── 2) Navigation GIFs (2) ──
GIF_DIR="${PROOF_DIR}/gifs"
mkdir -p "${GIF_DIR}"
GIF_COUNT=0
for f in $(find "${EVIDENCE}" -name '*.gif' -not -path '*/axis_sweep/*' 2>/dev/null | sort | head -2); do
  cp "${f}" "${GIF_DIR}/"
  GIF_COUNT=$((GIF_COUNT + 1))
done
echo "[V35E_PKG] nav_gifs=${GIF_COUNT}"

# ── 3) Axis sweep GIFs (3) ──
AXIS_DIR="${PROOF_DIR}/axis_sweep"
mkdir -p "${AXIS_DIR}"
AXIS_GIF_COUNT=0
SWEEP_DIR="${OUT_ROOT}/axis_sweep"
if [ -d "${SWEEP_DIR}" ]; then
  for f in $(find "${SWEEP_DIR}" -name 'axis_*.gif' | sort | head -3); do
    cp "${f}" "${AXIS_DIR}/"
    AXIS_GIF_COUNT=$((AXIS_GIF_COUNT + 1))
  done
fi
echo "[V35E_PKG] axis_gifs=${AXIS_GIF_COUNT}"

# ── 4) Pose trace ──
TRACE_DIR="${PROOF_DIR}/traces"
mkdir -p "${TRACE_DIR}"
TRACE_COUNT=0
for f in "${OUT_ROOT}"/pose_trace_v35e.csv "${OUT_ROOT}"/capture*/pose_trace*.csv; do
  if [ -f "${f}" ]; then
    cp "${f}" "${TRACE_DIR}/"
    TRACE_COUNT=$((TRACE_COUNT + 1))
  fi
done
echo "[V35E_PKG] traces=${TRACE_COUNT}"

# ── 5) Logs (minimal) ──
LOGS_DIR="${PROOF_DIR}/logs"
mkdir -p "${LOGS_DIR}"
LOG_DIR="${OUT_ROOT}/logs"
if [ -d "${LOG_DIR}" ]; then
  # Suite log
  [ -f "${LOG_DIR}/topo_eval_suite_v35e.log" ] && head -500 "${LOG_DIR}/topo_eval_suite_v35e.log" > "${LOGS_DIR}/suite_log_head.txt"
  # Isaac probe log tail
  [ -f "${LOG_DIR}/isaac_probe_v35e.log" ] && tail -200 "${LOG_DIR}/isaac_probe_v35e.log" > "${LOGS_DIR}/isaac_probe_tail.txt"
  # Extract key anchors from suite log
  [ -f "${LOG_DIR}/topo_eval_suite_v35e.log" ] && rg "V35E_|FIDELITY|MOTION_GATE|NAV2_STAGE|CASE_DONE|SUITE_OK|BASIS_SANITY|CAM_ALIGN|AXIS_TEST|AXIS_SWEEP|FAIL_FAST" "${LOG_DIR}/topo_eval_suite_v35e.log" > "${LOGS_DIR}/v35e_anchors.txt" 2>/dev/null || true
fi

# ── 6) Probe report ──
[ -f "${OUT_ROOT}/probe_report_v35e.json" ] && cp "${OUT_ROOT}/probe_report_v35e.json" "${PROOF_DIR}/"

# ── 7) Source snapshots ──
SRC_DIR="${PROOF_DIR}/source"
mkdir -p "${SRC_DIR}"
for f in \
  scripts/isaac/ui_office_topo_nav2_v35e.py \
  scripts/isaac/run_ui_office_topo_nav2_v35e.sh \
  scripts/topo_eval_suite_v35e_ui_debug.sh \
  scripts/package_topo_mvp_proof_v35e_ui_debug.sh; do
  [ -f "${ROOT}/${f}" ] && cp "${ROOT}/${f}" "${SRC_DIR}/"
done

# ── 8) Git metadata ──
META_DIR="${PROOF_DIR}/meta"
mkdir -p "${META_DIR}"
git -C "${ROOT}" log -1 --format='%H %ai %s' > "${META_DIR}/git_commit.txt" 2>/dev/null || true
git -C "${ROOT}" branch --show-current > "${META_DIR}/git_branch.txt" 2>/dev/null || true
git -C "${ROOT}" diff --stat > "${META_DIR}/git_diff_stat.txt" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv > "${META_DIR}/nvidia_smi.csv" 2>/dev/null || true
uname -a > "${META_DIR}/uname.txt" 2>/dev/null || true
date -Iseconds > "${META_DIR}/timestamp.txt"

# ── 9) Axis sweep report ──
REPORT="${PROOF_DIR}/axis_sweep_report.md"
cat > "${REPORT}" <<'REPORT_HEADER'
# v35e Axis Sweep Report

Camera convention: USD (-Z forward, +Y up, +X right).
REPORT_HEADER

if [ -f "${LOGS_DIR}/v35e_anchors.txt" ]; then
  echo "" >> "${REPORT}"
  echo "## Sweep Results" >> "${REPORT}"
  echo '```' >> "${REPORT}"
  grep "AXIS_SWEEP_RESULT\|AXIS_TEST\|AXIS_SWEEP_DEFS" "${LOGS_DIR}/v35e_anchors.txt" >> "${REPORT}" 2>/dev/null || echo "(no axis anchors)" >> "${REPORT}"
  echo '```' >> "${REPORT}"
  echo "" >> "${REPORT}"
  echo "## Alignment Gates" >> "${REPORT}"
  echo '```' >> "${REPORT}"
  grep "BASIS_SANITY\|CAM_ALIGN" "${LOGS_DIR}/v35e_anchors.txt" >> "${REPORT}" 2>/dev/null || echo "(no alignment anchors)" >> "${REPORT}"
  echo '```' >> "${REPORT}"
fi

# ── 10) Zip ──
(cd "${PROOF_DIR}" && zip -r "${ZIP_PATH}" .)
SIZE_BYTES=$(stat -c%s "${ZIP_PATH}" 2>/dev/null || echo 0)
SIZE_MB=$((SIZE_BYTES / 1048576))
SHA256="$(sha256sum "${ZIP_PATH}" | cut -d' ' -f1)"

# Size gate
BAD_COUNT=0
if [ "${SIZE_MB}" -gt 35 ]; then
  BAD_COUNT=1
  echo "[V35E_SIZE_GATE] ok=0 size_mb=${SIZE_MB} limit=35"
fi

echo "[V35E_EVIDENCE] ok=$((1 - BAD_COUNT)) gif_count=$((GIF_COUNT + AXIS_GIF_COUNT)) keyframes=${KF_COUNT} traces=${TRACE_COUNT} size_mb=${SIZE_MB}"
echo "[PROOF_ZIP_OK] zip=${ZIP_PATH} sha256=${SHA256} size_bytes=${SIZE_BYTES} size_mb=${SIZE_MB} bad_count=${BAD_COUNT}"

# Cleanup temp
rm -rf "${PROOF_DIR}"
