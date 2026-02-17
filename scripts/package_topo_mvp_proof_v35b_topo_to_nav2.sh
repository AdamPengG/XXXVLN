#!/usr/bin/env bash
# package_topo_mvp_proof_v35b_topo_to_nav2.sh — small evidence ZIP (<= 40 MB target, 60 MB cap)
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

# Find the most recent v35b run dir
if [ -n "${V35B_OUT_ROOT:-}" ]; then
  OUT_ROOT="${V35B_OUT_ROOT}"
else
  OUT_ROOT="$(ls -1dt runs/topo_mvp/v35b_topo_to_nav2_* 2>/dev/null | head -1 || true)"
  if [ -z "${OUT_ROOT}" ]; then
    echo "[PROOF_ZIP_OK] ok=0 reason=no_v35b_run_found"
    exit 2
  fi
fi

TS="$(date +%Y%m%d_%H%M)"
ZIP="/home/peng/DualVLN/forGPT/v35b_topo_to_nav2_PROOF_${TS}.zip"
STAGE_DIR="/tmp/v35b_proof_${TS}"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/evidence"/{logs,roomqa,map,capture,evidence_frames,src_snapshot,git,system}

# ── Logs (suite + per-case) ──
cp "${OUT_ROOT}/logs/"*.log "${STAGE_DIR}/evidence/logs/" 2>/dev/null || true
cp "${OUT_ROOT}/logs/"*.json "${STAGE_DIR}/evidence/logs/" 2>/dev/null || true

# ── Roomqa artifacts (JSON + PNG overlays — these are tiny) ──
for f in qa_metrics.json room_partition.json doorway_edges.json room_graph.json topo_nodes.json object_catalog.json; do
  cp "${OUT_ROOT}/roomqa/${f}" "${STAGE_DIR}/evidence/roomqa/" 2>/dev/null || true
done
cp "${OUT_ROOT}/roomqa/"picked_goal_*.json "${STAGE_DIR}/evidence/roomqa/" 2>/dev/null || true
cp "${OUT_ROOT}/roomqa/"staged_goals_*.json "${STAGE_DIR}/evidence/roomqa/" 2>/dev/null || true
cp "${OUT_ROOT}/roomqa/"staged_result_*.json "${STAGE_DIR}/evidence/roomqa/" 2>/dev/null || true
cp "${OUT_ROOT}/roomqa/traj_room_sequence.csv" "${STAGE_DIR}/evidence/roomqa/" 2>/dev/null || true
cp "${OUT_ROOT}/roomqa/"overlay_map_*.png "${STAGE_DIR}/evidence/roomqa/" 2>/dev/null || true

# ── Map PGM/YAML + preview ──
cp "${OUT_ROOT}/map/office_map.yaml" "${STAGE_DIR}/evidence/map/" 2>/dev/null || true
cp "${OUT_ROOT}/map/office_map_preview.png" "${STAGE_DIR}/evidence/map/" 2>/dev/null || true
# PGM can be large; convert to PNG if needed
if [ -f "${OUT_ROOT}/map/office_map.pgm" ]; then
  python3 -c "
from PIL import Image
Image.open('${OUT_ROOT}/map/office_map.pgm').save('${STAGE_DIR}/evidence/map/office_map.png')
" 2>/dev/null || cp "${OUT_ROOT}/map/office_map.pgm" "${STAGE_DIR}/evidence/map/" 2>/dev/null || true
fi

# ── Motion report ──
cp "${OUT_ROOT}/capture/motion_report."* "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true
cp "${OUT_ROOT}/capture/pose_trace.csv" "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true

# ── KEY evidence frames only (not full dump) ──
if [ -d "${OUT_ROOT}/evidence_frames" ]; then
  cp "${OUT_ROOT}/evidence_frames/"key_*.png "${STAGE_DIR}/evidence/evidence_frames/" 2>/dev/null || true
  cp "${OUT_ROOT}/evidence_frames/"*_motion.gif "${STAGE_DIR}/evidence/evidence_frames/" 2>/dev/null || true
fi
# Also grab the existing rgb.gif if evidence_frames GIFs are missing
cp "${OUT_ROOT}/capture/rgb.gif" "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true

# ── Source snapshots ──
for f in \
  scripts/topo_room_from_occ_v35a.py \
  scripts/topo_instruction_to_goal_v35a.py \
  scripts/nav2/send_staged_goals_v35a.py \
  scripts/topo_eval_suite_v35b_topo_to_nav2.sh \
  scripts/package_topo_mvp_proof_v35b_topo_to_nav2.sh \
  scripts/tools/make_gif_safe.py \
  scripts/tools/check_capture_motion.py \
  scripts/nav2/generate_office_map_v34g.py \
  scripts/nav2/bringup_nav2_office_v34i.sh \
  scripts/nav2/ros2_flow_check_v34i.sh \
  scripts/isaac/ui_office_ros2_bridge_probe_v34d.py \
  scripts/tools/check_capture_motion.sh \
; do
  cp "${f}" "${STAGE_DIR}/evidence/src_snapshot/" 2>/dev/null || true
done

# ── Git metadata ──
{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
} > "${STAGE_DIR}/evidence/git/rev.txt"
git status --porcelain > "${STAGE_DIR}/evidence/git/status_porcelain.txt"
git diff --stat > "${STAGE_DIR}/evidence/git/diff_stat.txt"
git log -n 20 --oneline > "${STAGE_DIR}/evidence/git/log20.txt"
git submodule status > "${STAGE_DIR}/evidence/git/submodule_status.txt"
( cd repo/InternNav && git rev-parse HEAD && git rev-parse --abbrev-ref HEAD ) > "${STAGE_DIR}/evidence/git/internnav_head.txt" 2>/dev/null || true

# ── System info ──
nvidia-smi > "${STAGE_DIR}/evidence/system/nvidia_smi.txt" 2>&1 || true

# ── Build ZIP ──
mkdir -p /home/peng/DualVLN/forGPT
(cd "${STAGE_DIR}" && zip -r "${ZIP}" evidence >/dev/null)

# Verify ZIP
unzip -t "${ZIP}" >/dev/null
SHA="$(sha256sum "${ZIP}" | awk '{print $1}')"
SIZE="$(stat -c%s "${ZIP}")"
BAD_COUNT="$( (zipinfo -1 "${ZIP}" | grep -E '\.(pt|pth|ckpt|bin|safetensors)$' || true) | wc -l )"
SIZE_MB=$((SIZE / 1048576))

echo "[PROOF_ZIP_OK] zip=${ZIP} sha256=${SHA} size_bytes=${SIZE} size_mb=${SIZE_MB} bad_count=${BAD_COUNT}"

if [ "${SIZE}" -gt 62914560 ]; then
  echo "[PROOF_ZIP_WARN] size_mb=${SIZE_MB} exceeds 60MB hard cap!"
fi
