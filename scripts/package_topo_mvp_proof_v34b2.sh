#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

TS="$(date +%Y%m%d_%H%M)"
OUT_ROOT="${V34B_OUT_ROOT:-runs/topo_mvp/v34b_nav2_demo}"
ZIP="/home/peng/DualVLN/forGPT/v34b2_nav2_ui_demo_PROOF_${TS}.zip"
STAGE_DIR="/tmp/v34b2_proof_${TS}"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/evidence"/{logs,map,eval,qa,physics,collision,src_snapshot,git,system}

cp "${OUT_ROOT}/summary.json" "${STAGE_DIR}/evidence/eval/" 2>/dev/null || true
cp "${OUT_ROOT}"/logs/*.log "${STAGE_DIR}/evidence/logs/" 2>/dev/null || true
cp "${OUT_ROOT}"/map/* "${STAGE_DIR}/evidence/map/" 2>/dev/null || true
cp "${OUT_ROOT}"/eval/RESULT_*.json "${STAGE_DIR}/evidence/eval/" 2>/dev/null || true
cp "${OUT_ROOT}"/eval/path_poses_*.json "${STAGE_DIR}/evidence/eval/" 2>/dev/null || true

cp "${OUT_ROOT}"/qa/qa_metrics.json "${STAGE_DIR}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}"/qa/room_partition.json "${STAGE_DIR}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}"/qa/doorway_edges.json "${STAGE_DIR}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}"/qa/room_graph.json "${STAGE_DIR}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}"/qa/topo_map.json "${STAGE_DIR}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}"/qa/traj_room_sequence.csv "${STAGE_DIR}/evidence/qa/" 2>/dev/null || true
mkdir -p "${STAGE_DIR}/evidence/qa/overlay"
for p in "${OUT_ROOT}"/qa/overlay/overlay_rgb_*.png; do
  [ -f "${p}" ] || continue
  cp "${p}" "${STAGE_DIR}/evidence/qa/overlay/"
done

cp "${OUT_ROOT}"/collision_audit/collision_audit.json "${STAGE_DIR}/evidence/collision/" 2>/dev/null || true
cp "${OUT_ROOT}"/collision_audit/collision_audit.md "${STAGE_DIR}/evidence/collision/" 2>/dev/null || true
cp "${OUT_ROOT}"/office_phys/office_phys.usd "${STAGE_DIR}/evidence/collision/" 2>/dev/null || true
cp "${OUT_ROOT}"/office_phys/collision_build.json "${STAGE_DIR}/evidence/collision/" 2>/dev/null || true

cp "${OUT_ROOT}"/physics/physics_drive_result.json "${STAGE_DIR}/evidence/physics/" 2>/dev/null || true
cp "${OUT_ROOT}"/physics/physics_drive.gif "${STAGE_DIR}/evidence/physics/" 2>/dev/null || true
cp "${OUT_ROOT}"/physics/motion_report.json "${STAGE_DIR}/evidence/physics/" 2>/dev/null || true
cp "${OUT_ROOT}"/physics/motion_report.md "${STAGE_DIR}/evidence/physics/" 2>/dev/null || true
mkdir -p "${STAGE_DIR}/evidence/physics/capture"
for p in "${OUT_ROOT}"/physics/capture/rgb_*.png; do
  [ -f "${p}" ] || continue
  cp "${p}" "${STAGE_DIR}/evidence/physics/capture/"
done

# representative debug run artifacts (compact)
python3 - "${OUT_ROOT}/debug_runs" "${STAGE_DIR}/evidence/physics" <<'PY'
from pathlib import Path
import shutil, sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2]) / "debug_repr"
dst.mkdir(parents=True, exist_ok=True)
if not src.exists():
    raise SystemExit(0)
for p in sorted(src.glob('**/rgb_*.png'))[:20]:
    shutil.copy2(p, dst / f"{p.parent.name}_{p.name}")
for p in sorted(src.glob('**/motion_report.*')):
    shutil.copy2(p, dst / f"{p.parent.name}_{p.name}")
PY

cp scripts/topo_eval_suite_v34b_nav2_ui_demo.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/package_topo_mvp_proof_v34b2.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/collision_audit_v34b2.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/collision_audit_v34b2.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/build_office_phys_stage_v34b2.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/build_office_phys_stage_v34b2.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/physics_drive_smoke_v34b.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/nav2/generate_office_map_v34b.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/nav2/generate_office_map_v34b.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/nav2/bringup_nav2_office_v34b.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/run_ui_office_nav2_v34b.sh "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/isaac/ui_office_nav2_probe_v34b.py "${STAGE_DIR}/evidence/src_snapshot/"
cp scripts/topo_topomap_room_qa_v34.py "${STAGE_DIR}/evidence/src_snapshot/"
cp configs/isaac_scenes_v34b.yaml "${STAGE_DIR}/evidence/src_snapshot/"

git rev-parse --abbrev-ref HEAD > "${STAGE_DIR}/evidence/git/outer_branch.txt"
git rev-parse HEAD > "${STAGE_DIR}/evidence/git/outer_head.txt"
git status --porcelain > "${STAGE_DIR}/evidence/git/outer_status_porcelain.txt"
git diff --stat > "${STAGE_DIR}/evidence/git/outer_diff_stat.txt"
git submodule status > "${STAGE_DIR}/evidence/git/submodule_status.txt"

nvidia-smi > "${STAGE_DIR}/evidence/system/nvidia_smi.txt" 2>&1 || true

mkdir -p /home/peng/DualVLN/forGPT
(cd "${STAGE_DIR}" && zip -r "${ZIP}" evidence >/dev/null)
unzip -t "${ZIP}" >/dev/null
SHA="$(sha256sum "${ZIP}" | awk '{print $1}')"
SIZE="$(stat -c%s "${ZIP}")"
BAD_COUNT="$( (zipinfo -1 "${ZIP}" | grep -E '\.(pt|pth|ckpt|bin|safetensors)$' || true) | wc -l )"
echo "[PROOF_ZIP_OK] zip=${ZIP} sha256=${SHA} size_bytes=${SIZE} bad_count=${BAD_COUNT}"

