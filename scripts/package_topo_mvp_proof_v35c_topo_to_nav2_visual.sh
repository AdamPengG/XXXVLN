#!/usr/bin/env bash
# package_topo_mvp_proof_v35c_topo_to_nav2_visual.sh — small evidence ZIP (<= 60 MB)
# Contains: logs, topo outputs, map, pose trace, 6 dual GIFs, 5 key PNGs per view, git + nvidia-smi
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

# Find the most recent v35c run dir
if [ -n "${V35C_OUT_ROOT:-}" ]; then
  OUT_ROOT="${V35C_OUT_ROOT}"
else
  OUT_ROOT="$(ls -1dt runs/topo_mvp/v35c_topo_to_nav2_* 2>/dev/null | head -1 || true)"
  if [ -z "${OUT_ROOT}" ]; then
    echo "[PROOF_ZIP_OK] ok=0 reason=no_v35c_run_found"
    exit 2
  fi
fi

TS="$(date +%Y%m%d_%H%M)"
ZIP="/home/peng/DualVLN/forGPT/v35c_topo_to_nav2_visual_PROOF_${TS}.zip"
STAGE_DIR="/tmp/v35c_proof_${TS}"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/evidence"/{logs,roomqa,map,capture,gifs,keyframes,src_snapshot,git,system}

# ── Logs ──
cp "${OUT_ROOT}/logs/"*.log "${STAGE_DIR}/evidence/logs/" 2>/dev/null || true
cp "${OUT_ROOT}/logs/"*.json "${STAGE_DIR}/evidence/logs/" 2>/dev/null || true

# ── Topo outputs ──
for f in room_partition.json doorway_edges.json room_graph.json topo_nodes.json topo_map.json object_catalog.json; do
  cp "${OUT_ROOT}/roomqa/${f}" "${STAGE_DIR}/evidence/roomqa/" 2>/dev/null || true
done
cp "${OUT_ROOT}/roomqa/"picked_goal_*.json "${STAGE_DIR}/evidence/roomqa/" 2>/dev/null || true
cp "${OUT_ROOT}/roomqa/"staged_goals_*.json "${STAGE_DIR}/evidence/roomqa/" 2>/dev/null || true
cp "${OUT_ROOT}/roomqa/"staged_result_*.json "${STAGE_DIR}/evidence/roomqa/" 2>/dev/null || true
cp "${OUT_ROOT}/roomqa/traj_room_sequence.csv" "${STAGE_DIR}/evidence/roomqa/" 2>/dev/null || true

# ── Map ──
cp "${OUT_ROOT}/map/office_map.yaml" "${STAGE_DIR}/evidence/map/" 2>/dev/null || true
cp "${OUT_ROOT}/map/office_map_preview.png" "${STAGE_DIR}/evidence/map/" 2>/dev/null || true
if [ -f "${OUT_ROOT}/map/office_map.pgm" ]; then
  python3 -c "
from PIL import Image
Image.open('${OUT_ROOT}/map/office_map.pgm').save('${STAGE_DIR}/evidence/map/office_map.png')
" 2>/dev/null || cp "${OUT_ROOT}/map/office_map.pgm" "${STAGE_DIR}/evidence/map/" 2>/dev/null || true
fi

# ── Pose traces ──
cp "${OUT_ROOT}/capture/pose_trace.csv" "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true
cp "${OUT_ROOT}/capture/pose_trace_v35c.csv" "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true
cp "${OUT_ROOT}/capture/motion_report."* "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true

# ── GIFs (6: 3 cases × 2 views) ──
cp "${OUT_ROOT}/evidence_frames/"*_fp.gif "${STAGE_DIR}/evidence/gifs/" 2>/dev/null || true
cp "${OUT_ROOT}/evidence_frames/"*_chase.gif "${STAGE_DIR}/evidence/gifs/" 2>/dev/null || true
cp "${OUT_ROOT}/capture/rgb.gif" "${STAGE_DIR}/evidence/gifs/" 2>/dev/null || true

# ── Key frames (5 per view, already selected in suite) ──
cp "${OUT_ROOT}/evidence_frames/"key_*.png "${STAGE_DIR}/evidence/keyframes/" 2>/dev/null || true

# ── Source snapshots ──
for f in \
  scripts/isaac/ui_office_ros2_bridge_probe_v35c.py \
  scripts/topo_eval_suite_v35c_topo_to_nav2_visual.sh \
  scripts/package_topo_mvp_proof_v35c_topo_to_nav2_visual.sh \
  scripts/topo_room_from_occ_v35a.py \
  scripts/topo_instruction_to_goal_v35a.py \
  scripts/nav2/send_staged_goals_v35a.py \
  scripts/tools/make_gif_safe.py \
  scripts/tools/check_capture_motion.py \
  scripts/nav2/generate_office_map_v34g.py \
  scripts/nav2/bringup_nav2_office_v34i.sh \
  scripts/nav2/ros2_flow_check_v34i.sh \
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

# Verify
unzip -t "${ZIP}" >/dev/null
SHA="$(sha256sum "${ZIP}" | awk '{print $1}')"
SIZE="$(stat -c%s "${ZIP}")"
BAD_COUNT="$( (zipinfo -1 "${ZIP}" | grep -E '\.(pt|pth|ckpt|bin|safetensors)$' || true) | wc -l )"
SIZE_MB=$((SIZE / 1048576))

echo "[PROOF_ZIP_OK] zip=${ZIP} sha256=${SHA} size_bytes=${SIZE} size_mb=${SIZE_MB} bad_count=${BAD_COUNT}"

if [ "${SIZE}" -gt 62914560 ]; then
  echo "[PROOF_ZIP_WARN] size_mb=${SIZE_MB} exceeds 60MB hard cap!"
fi
