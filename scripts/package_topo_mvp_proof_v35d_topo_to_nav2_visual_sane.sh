#!/usr/bin/env bash
# package_topo_mvp_proof_v35d_topo_to_nav2_visual_sane.sh — <=35MB evidence ZIP
# Only: 6 GIFs, 3×2 keyframes (start/mid/last), pose_trace.csv, nav2 feedback, git+nvidia-smi
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

# Find v35d run dir
if [ -n "${V35D_OUT_ROOT:-}" ]; then
  OUT_ROOT="${V35D_OUT_ROOT}"
else
  OUT_ROOT="$(ls -1dt runs/topo_mvp/v35d_topo_to_nav2_* 2>/dev/null | head -1 || true)"
  if [ -z "${OUT_ROOT}" ]; then
    echo "[PROOF_ZIP_OK] ok=0 reason=no_v35d_run_found"
    exit 2
  fi
fi

TS="$(date +%Y%m%d_%H%M)"
ZIP="/home/peng/DualVLN/forGPT/v35d_topo_nav2_visual_sane_PROOF_${TS}.zip"
STAGE_DIR="/tmp/v35d_proof_${TS}"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/evidence"/{logs,roomqa,map,traces,gifs,keyframes,src_snapshot,git,system}

# ── Logs (small text only) ──
for f in "${OUT_ROOT}/logs/"*.log "${OUT_ROOT}/logs/"*.json; do
  [ -f "$f" ] && cp "$f" "${STAGE_DIR}/evidence/logs/" 2>/dev/null || true
done

# ── Topo outputs (JSON only, no images) ──
for f in room_partition.json doorway_edges.json room_graph.json topo_nodes.json staged_goals_*.json staged_result_*.json; do
  cp "${OUT_ROOT}/roomqa/${f}" "${STAGE_DIR}/evidence/roomqa/" 2>/dev/null || true
done

# ── Map (converted to small PNG, no PGM) ──
cp "${OUT_ROOT}/map/office_map.yaml" "${STAGE_DIR}/evidence/map/" 2>/dev/null || true
if [ -f "${OUT_ROOT}/map/office_map.pgm" ]; then
  python3 -c "
from PIL import Image
Image.open('${OUT_ROOT}/map/office_map.pgm').resize((256,256)).save('${STAGE_DIR}/evidence/map/office_map_small.png')
" 2>/dev/null || true
fi

# ── Small traces only: pose_trace.csv + nav2 feedback ──
cp "${OUT_ROOT}/capture/pose_trace.csv" "${STAGE_DIR}/evidence/traces/" 2>/dev/null || true
cp "${OUT_ROOT}/capture/pose_trace_v35d.csv" "${STAGE_DIR}/evidence/traces/" 2>/dev/null || true
# Copy last nav2 feedback per case
for f in "${OUT_ROOT}/roomqa/"staged_result_*.json; do
  [ -f "$f" ] && cp "$f" "${STAGE_DIR}/evidence/traces/" 2>/dev/null || true
done

# ── GIFs (6 total: 3 cases × 2 views) ──
for g in "${OUT_ROOT}/evidence_frames/"*_fp.gif "${OUT_ROOT}/evidence_frames/"*_chase.gif; do
  [ -f "$g" ] && cp "$g" "${STAGE_DIR}/evidence/gifs/" 2>/dev/null || true
done

# ── Keyframes (3 per case×view = 18 max) ──
for k in "${OUT_ROOT}/evidence_frames/"key_*.png; do
  [ -f "$k" ] && cp "$k" "${STAGE_DIR}/evidence/keyframes/" 2>/dev/null || true
done

# ── Source snapshots (5 key scripts) ──
for f in \
  scripts/isaac/ui_office_ros2_bridge_probe_v35d.py \
  scripts/topo_eval_suite_v35d_topo_to_nav2_visual_sane.sh \
  scripts/package_topo_mvp_proof_v35d_topo_to_nav2_visual_sane.sh \
  scripts/tools/make_gif_safe.py \
  scripts/tools/check_capture_motion.py \
; do
  [ -f "$f" ] && cp "$f" "${STAGE_DIR}/evidence/src_snapshot/" 2>/dev/null || true
done

# ── Git metadata ──
{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
} > "${STAGE_DIR}/evidence/git/rev.txt"
git status --porcelain > "${STAGE_DIR}/evidence/git/status_porcelain.txt"
git diff --stat > "${STAGE_DIR}/evidence/git/diff_stat.txt"
git log -n 20 --oneline > "${STAGE_DIR}/evidence/git/log20.txt"

# ── System info ──
nvidia-smi > "${STAGE_DIR}/evidence/system/nvidia_smi.txt" 2>&1 || true

# ── Build ZIP ──
mkdir -p /home/peng/DualVLN/forGPT
(cd "${STAGE_DIR}" && zip -r "${ZIP}" evidence >/dev/null)

# Verify
unzip -t "${ZIP}" >/dev/null
SHA="$(sha256sum "${ZIP}" | awk '{print $1}')"
SIZE="$(stat -c%s "${ZIP}")"
BAD_COUNT="$( (zipinfo -1 "${ZIP}" | grep -E '\.(pt|pth|ckpt|bin|safetensors)$' || true) | wc -l )"
SIZE_MB=$((SIZE / 1048576))
GIF_COUNT="$( (zipinfo -1 "${ZIP}" | grep -c '\.gif$' || true) )"
KF_COUNT="$( (zipinfo -1 "${ZIP}" | grep -c 'key_' || true) )"
TRACE_COUNT="$( (zipinfo -1 "${ZIP}" | grep -c 'pose_trace' || true) )"

echo "[V35D_EVIDENCE] ok=1 gif_count=${GIF_COUNT} keyframes=${KF_COUNT} traces=${TRACE_COUNT} size_mb=${SIZE_MB}"
echo "[PROOF_ZIP_OK] zip=${ZIP} sha256=${SHA} size_bytes=${SIZE} size_mb=${SIZE_MB} bad_count=${BAD_COUNT}"

if [ "${SIZE}" -gt 36700160 ]; then
  echo "[PROOF_ZIP_WARN] size_mb=${SIZE_MB} exceeds 35MB cap!"
fi
