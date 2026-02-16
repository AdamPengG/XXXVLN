#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

TS="$(date +%Y%m%d_%H%M)"
OUT_ROOT="${V34C_OUT_ROOT:-runs/topo_mvp/v34c_ros2_bridge}"
ZIP="/home/peng/DualVLN/forGPT/v34c_ros2_nav2_bridge_PROOF_${TS}.zip"
STAGE="/tmp/v34c_ros2_nav2_proof_${TS}"

rm -rf "${STAGE}"
mkdir -p "${STAGE}/evidence/logs" "${STAGE}/evidence/capture" "${STAGE}/evidence/src_snapshot" "${STAGE}/evidence/git"

# Core logs and reports.
cp -r "${OUT_ROOT}/logs" "${STAGE}/evidence/" 2>/dev/null || true
cp "${OUT_ROOT}/bridge_probe_report.json" "${STAGE}/evidence/" 2>/dev/null || true
cp "${OUT_ROOT}/map/office_map.yaml" "${STAGE}/evidence/" 2>/dev/null || true
cp "${OUT_ROOT}/map/office_map.pgm" "${STAGE}/evidence/" 2>/dev/null || true
cp "${OUT_ROOT}/map/office_map_preview.png" "${STAGE}/evidence/" 2>/dev/null || true

# Capture subset.
if [ -d "${OUT_ROOT}/capture" ]; then
  mkdir -p "${STAGE}/evidence/capture"
  cp "${OUT_ROOT}/capture/motion_report.json" "${STAGE}/evidence/capture/" 2>/dev/null || true
  cp "${OUT_ROOT}/capture/motion_report.md" "${STAGE}/evidence/capture/" 2>/dev/null || true
  cp "${OUT_ROOT}/capture/rgb.gif" "${STAGE}/evidence/capture/" 2>/dev/null || true
  python3 - <<'PY' "${OUT_ROOT}/capture" "${STAGE}/evidence/capture"
import shutil, sys
from pathlib import Path
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
imgs = sorted(src.glob('rgb_*.png'))[:40]
for p in imgs:
    shutil.copy2(p, dst / p.name)
PY
fi

# Source snapshot.
cp scripts/nav2/ros2_nav2_env_check_v34c.sh "${STAGE}/evidence/src_snapshot/"
cp scripts/nav2/install_ros2_nav2_v34c.sh "${STAGE}/evidence/src_snapshot/"
cp scripts/isaac/ros2_rclpy_smoke_v34c.sh "${STAGE}/evidence/src_snapshot/"
cp scripts/isaac/run_ui_office_ros2_bridge_v34c.sh "${STAGE}/evidence/src_snapshot/"
cp scripts/isaac/ui_office_ros2_bridge_probe_v34c.py "${STAGE}/evidence/src_snapshot/"
cp scripts/nav2/bringup_nav2_office_v34c.sh "${STAGE}/evidence/src_snapshot/"
cp scripts/nav2/send_goal_v34c.py "${STAGE}/evidence/src_snapshot/"
cp scripts/topo_eval_suite_v34c_ros2_nav2_bridge.sh "${STAGE}/evidence/src_snapshot/"
cp scripts/package_topo_mvp_proof_v34c_ros2_nav2_bridge.sh "${STAGE}/evidence/src_snapshot/"
cp configs/nav2_params_v34c.yaml "${STAGE}/evidence/src_snapshot/"

# Git metadata.
git rev-parse --abbrev-ref HEAD > "${STAGE}/evidence/git/outer_branch.txt"
git rev-parse HEAD > "${STAGE}/evidence/git/outer_head.txt"
git status --porcelain > "${STAGE}/evidence/git/outer_status_porcelain.txt"
git diff --stat > "${STAGE}/evidence/git/outer_diff_stat.txt"
git log -n 20 --oneline > "${STAGE}/evidence/git/outer_log_oneline_20.txt"
git diff > "${STAGE}/evidence/git/outer_diff.patch"
(
  cd repo/InternNav
  git rev-parse --abbrev-ref HEAD > "${STAGE}/evidence/git/submodule_branch.txt"
  git rev-parse HEAD > "${STAGE}/evidence/git/submodule_head.txt"
  git status --porcelain > "${STAGE}/evidence/git/submodule_status_porcelain.txt"
) || true

nvidia-smi > "${STAGE}/evidence/git/nvidia_smi.txt" 2>&1 || true

mkdir -p /home/peng/DualVLN/forGPT
(cd "${STAGE}" && zip -r "${ZIP}" evidence >/dev/null)
unzip -t "${ZIP}" >/dev/null
SHA="$(sha256sum "${ZIP}" | awk '{print $1}')"
SIZE="$(stat -c%s "${ZIP}")"
BAD_COUNT="$( (zipinfo -1 "${ZIP}" | grep -E '(\.pt$|\.pth$|\.ckpt$|\.bin$|\.safetensors$)' || true) | wc -l )"

echo "[PROOF_ZIP_OK] zip=${ZIP} sha256=${SHA} size_bytes=${SIZE} bad_count=${BAD_COUNT}"
