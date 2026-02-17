#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34J_OUT_ROOT:-runs/topo_mvp/v34j_nav2_office_succeeded}"
TS="$(date +%Y%m%d_%H%M)"
ZIP="/home/peng/DualVLN/forGPT/v34j_nav2_office_succeeded_PROOF_${TS}.zip"
STAGE_DIR="/tmp/v34j_nav2_office_succeeded_proof_${TS}"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/evidence"/{logs,map,capture,src_snapshot,git,system}

cp "${OUT_ROOT}/logs/"*.log "${STAGE_DIR}/evidence/logs/" 2>/dev/null || true
cp "${OUT_ROOT}/probe_report_v34d.json" "${STAGE_DIR}/evidence/logs/" 2>/dev/null || true
cp "${OUT_ROOT}/camera_fidelity_v34f.json" "${STAGE_DIR}/evidence/logs/" 2>/dev/null || true
cp "${OUT_ROOT}/goals.json" "${STAGE_DIR}/evidence/logs/" 2>/dev/null || true

cp "${OUT_ROOT}/map/"* "${STAGE_DIR}/evidence/map/" 2>/dev/null || true

cp "${OUT_ROOT}/capture/rgb_"*.png "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true
cp "${OUT_ROOT}/capture/depth_"*.png "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true
cp "${OUT_ROOT}/capture/overlay_rgb_"*.png "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true
cp "${OUT_ROOT}/capture/rgb.gif" "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true
cp "${OUT_ROOT}/capture/pose_trace.csv" "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true
cp "${OUT_ROOT}/capture/motion_report."* "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true

python3 - <<'PY' "${STAGE_DIR}/evidence/capture"
from pathlib import Path
root = Path(__import__("sys").argv[1])
for prefix, keep in (("rgb_", 40), ("depth_", 40), ("overlay_rgb_", 40)):
    files = sorted(root.glob(f"{prefix}*.png"))
    for i, p in enumerate(files):
        if i >= keep:
            p.unlink(missing_ok=True)
PY

cp scripts/topo_eval_suite_v34j_nav2_office_succeeded.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/package_topo_mvp_proof_v34j_nav2_office_succeeded.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/nav2/sample_reachable_goals_v34j.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/nav2/send_goal_v34j.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/nav2/bringup_nav2_office_v34i.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/nav2/ros2_flow_check_v34i.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/nav2/odom_to_tf_broadcaster_v34i.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/ui_office_ros2_bridge_probe_v34d.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/tools/check_capture_motion.py "${STAGE_DIR}/evidence/src_snapshot/"
cp configs/nav2_params_v34j.yaml "${STAGE_DIR}/evidence/src_snapshot/"
cp repo/InternNav/internnav/sim_backend/isaac_backend.py "${STAGE_DIR}/evidence/src_snapshot/"

{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
} > "${STAGE_DIR}/evidence/git/rev.txt"
git status --porcelain > "${STAGE_DIR}/evidence/git/status_porcelain.txt"
git diff --stat > "${STAGE_DIR}/evidence/git/diff_stat.txt"
git diff > "${STAGE_DIR}/evidence/git/diff.patch"
git log -n 20 --oneline > "${STAGE_DIR}/evidence/git/log20.txt"
git submodule status > "${STAGE_DIR}/evidence/git/submodule_status.txt"
( cd repo/InternNav && git rev-parse HEAD && git rev-parse --abbrev-ref HEAD ) > "${STAGE_DIR}/evidence/git/internnav_head.txt" 2>/dev/null || true

nvidia-smi > "${STAGE_DIR}/evidence/system/nvidia_smi.txt" 2>&1 || true

mkdir -p /home/peng/DualVLN/forGPT
(cd "${STAGE_DIR}" && zip -r "${ZIP}" evidence >/dev/null)
unzip -t "${ZIP}" >/dev/null
SHA="$(sha256sum "${ZIP}" | awk '{print $1}')"
SIZE="$(stat -c%s "${ZIP}")"
BAD_COUNT="$( (zipinfo -1 "${ZIP}" | grep -E '\.(pt|pth|ckpt|bin|safetensors)$' || true) | wc -l )"
echo "[PROOF_ZIP_OK] zip=${ZIP} sha256=${SHA} size_bytes=${SIZE} bad_count=${BAD_COUNT}"
