#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

TS="$(date +%Y%m%d_%H%M)"
OUT_ROOT="${V34_QA_OUT_ROOT:-runs/topo_mvp/v34_topo_room_qa}"
ZIP="/home/peng/DualVLN/forGPT/v34_topo_room_qa_PROOF_${TS}.zip"
STAGE="/tmp/v34_topo_room_qa_${TS}"

rm -rf "${STAGE}"
mkdir -p "${STAGE}/evidence/qa" "${STAGE}/evidence/logs" "${STAGE}/evidence/src_snapshot" "${STAGE}/evidence/git" "${STAGE}/evidence/system"

cp "${OUT_ROOT}/qa_metrics.json" "${STAGE}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}/room_partition.json" "${STAGE}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}/doorway_edges.json" "${STAGE}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}/room_graph.json" "${STAGE}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}/topo_map.json" "${STAGE}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}/traj_room_sequence.csv" "${STAGE}/evidence/qa/" 2>/dev/null || true

mkdir -p "${STAGE}/evidence/qa/overlay" "${STAGE}/evidence/qa/capture"
for p in "${OUT_ROOT}"/overlay/overlay_rgb_*.png; do
  [ -f "${p}" ] || continue
  cp "${p}" "${STAGE}/evidence/qa/overlay/"
done
python3 - "${OUT_ROOT}" "${STAGE}/evidence/qa/capture" <<'PY'
from pathlib import Path
import shutil, sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
for p in sorted((src / "capture" / "rgb").glob("rgb_*.png"))[:20]:
    shutil.copy2(p, dst / p.name)
for p in sorted((src / "capture" / "depth_png").glob("depth_*.png"))[:20]:
    shutil.copy2(p, dst / p.name)
for p in sorted((src / "capture" / "depth_npy").glob("depth_*.npy"))[:10]:
    shutil.copy2(p, dst / p.name)
for name in ("motion_report.json", "motion_report.md", "rgb.gif"):
    p = src / "capture" / name
    if p.is_file():
        shutil.copy2(p, dst / p.name)
PY

cp "${OUT_ROOT}"/logs/*.log "${STAGE}/evidence/logs/" 2>/dev/null || true

cp scripts/topo_topomap_room_qa_v34.py "${STAGE}/evidence/src_snapshot/"
cp scripts/topo_eval_suite_v34_topo_room_qa.sh "${STAGE}/evidence/src_snapshot/"
cp scripts/package_topo_mvp_proof_v34_topo_room_qa.sh "${STAGE}/evidence/src_snapshot/"
cp scripts/tools/check_capture_motion.py "${STAGE}/evidence/src_snapshot/"
cp scripts/tools/check_capture_motion.sh "${STAGE}/evidence/src_snapshot/"
cp scripts/isaac/render_smoke_v27.sh "${STAGE}/evidence/src_snapshot/"

git rev-parse --abbrev-ref HEAD > "${STAGE}/evidence/git/outer_branch.txt"
git rev-parse HEAD > "${STAGE}/evidence/git/outer_head.txt"
git status --porcelain > "${STAGE}/evidence/git/outer_status_porcelain.txt"
git diff --stat > "${STAGE}/evidence/git/outer_diff_stat.txt"
git log -n 20 --oneline > "${STAGE}/evidence/git/outer_log_oneline_20.txt"
(
  cd repo/InternNav
  git rev-parse --abbrev-ref HEAD > "${STAGE}/evidence/git/submodule_branch.txt"
  git rev-parse HEAD > "${STAGE}/evidence/git/submodule_head.txt"
) || true

nvidia-smi > "${STAGE}/evidence/system/nvidia_smi.txt" 2>&1 || true

mkdir -p /home/peng/DualVLN/forGPT
(cd "${STAGE}" && zip -r "${ZIP}" evidence >/dev/null)
unzip -t "${ZIP}" >/dev/null
SHA="$(sha256sum "${ZIP}" | awk '{print $1}')"
SIZE="$(stat -c%s "${ZIP}")"
BAD_COUNT="$( (zipinfo -1 "${ZIP}" | grep -E '\.(pt|pth|ckpt|bin|safetensors)$' || true) | wc -l )"
echo "[PROOF_ZIP_OK] zip=${ZIP} sha256=${SHA} size_bytes=${SIZE} bad_count=${BAD_COUNT}"
echo "ZIP=${ZIP}"
