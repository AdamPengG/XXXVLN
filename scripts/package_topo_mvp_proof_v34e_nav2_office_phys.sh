#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34E_OUT_ROOT:-runs/topo_mvp/v34e_nav2_office_phys}"
TS="$(date +%Y%m%d_%H%M)"
ZIP="/home/peng/DualVLN/forGPT/v34e_nav2_office_phys_PROOF_${TS}.zip"
STAGE_DIR="/tmp/v34e_nav2_office_phys_proof_${TS}"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/evidence"/{logs,map,capture,physics,office_phys,src_snapshot,git,system}

cp "${OUT_ROOT}/logs/"*.log "${STAGE_DIR}/evidence/logs/" 2>/dev/null || true
cp "${OUT_ROOT}/collision_audit/"* "${STAGE_DIR}/evidence/logs/" 2>/dev/null || true
cp "${OUT_ROOT}/map/"* "${STAGE_DIR}/evidence/map/" 2>/dev/null || true
cp "${OUT_ROOT}/physics/"* "${STAGE_DIR}/evidence/physics/" 2>/dev/null || true
cp "${OUT_ROOT}/office_phys/"*.json "${STAGE_DIR}/evidence/office_phys/" 2>/dev/null || true
cp "${OUT_ROOT}/office_phys/"*.usd "${STAGE_DIR}/evidence/office_phys/" 2>/dev/null || true

cp "${OUT_ROOT}/capture/rgb_"*.png "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true
cp "${OUT_ROOT}/capture/depth_"*.png "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true
cp "${OUT_ROOT}/capture/rgb.gif" "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true
cp "${OUT_ROOT}/capture/motion_report."* "${STAGE_DIR}/evidence/capture/" 2>/dev/null || true

python3 - <<'PY' "${STAGE_DIR}/evidence/capture"
from pathlib import Path
root = Path(__import__('sys').argv[1])
for prefix in ('rgb_', 'depth_'):
    frames = sorted(root.glob(f'{prefix}*.png'))
    for i, p in enumerate(frames):
        if i >= 80:
            p.unlink(missing_ok=True)
PY

cp scripts/topo_eval_suite_v34e_nav2_office_phys.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/package_topo_mvp_proof_v34e_nav2_office_phys.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/collision_audit_v34e.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/collision_audit_v34e.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/build_office_phys_stage_v34e.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/build_office_phys_stage_v34e.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/phys_spawn_settle_v34e.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/phys_spawn_settle_v34e.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/cmd_vel_drive_smoke_v34e.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/cmd_vel_drive_smoke_v34e.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/ui_office_ros2_bridge_probe_v34d.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/nav2/bringup_nav2_office_v34d.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/nav2/ros2_topic_check_v34d.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/nav2/send_goal_v34d.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/nav2/generate_office_map_v34b.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/nav2/generate_office_map_v34b.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp configs/nav2_params_v34d.yaml "${STAGE_DIR}/evidence/src_snapshot/"

{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
} > "${STAGE_DIR}/evidence/git/rev.txt"
git status --porcelain > "${STAGE_DIR}/evidence/git/status_porcelain.txt"
git diff --stat > "${STAGE_DIR}/evidence/git/diff_stat.txt"
git submodule status > "${STAGE_DIR}/evidence/git/submodule_status.txt"

nvidia-smi > "${STAGE_DIR}/evidence/system/nvidia_smi.txt" 2>&1 || true

mkdir -p /home/peng/DualVLN/forGPT
(cd "${STAGE_DIR}" && zip -r "${ZIP}" evidence >/dev/null)
unzip -t "${ZIP}" >/dev/null
SHA="$(sha256sum "${ZIP}" | awk '{print $1}')"
SIZE="$(stat -c%s "${ZIP}")"
BAD_COUNT="$( (zipinfo -1 "${ZIP}" | grep -E '\.(pt|pth|ckpt|bin|safetensors)$' || true) | wc -l )"
echo "[PROOF_ZIP_OK] zip=${ZIP} sha256=${SHA} size_bytes=${SIZE} bad_count=${BAD_COUNT}"
