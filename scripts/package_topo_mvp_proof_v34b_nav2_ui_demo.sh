#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

TS="$(date +%Y%m%d_%H%M)"
OUT_ROOT="${V34B_OUT_ROOT:-runs/topo_mvp/v34b_nav2_demo}"
ZIP="/home/peng/DualVLN/forGPT/v34b_nav2_ui_demo_PROOF_${TS}.zip"
STAGE="/tmp/v34b_nav2_ui_demo_${TS}"

rm -rf "${STAGE}"
mkdir -p "${STAGE}/evidence"/{logs,map,eval,qa,capture,src_snapshot,git,system}

cp "${OUT_ROOT}/summary.json" "${STAGE}/evidence/eval/" 2>/dev/null || true
cp "${OUT_ROOT}"/logs/*.log "${STAGE}/evidence/logs/" 2>/dev/null || true
cp "${OUT_ROOT}"/map/* "${STAGE}/evidence/map/" 2>/dev/null || true

cp "${OUT_ROOT}"/eval/RESULT_*.json "${STAGE}/evidence/eval/" 2>/dev/null || true
cp "${OUT_ROOT}"/eval/path_poses_*.json "${STAGE}/evidence/eval/" 2>/dev/null || true

cp "${OUT_ROOT}"/qa/qa_metrics.json "${STAGE}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}"/qa/room_partition.json "${STAGE}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}"/qa/doorway_edges.json "${STAGE}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}"/qa/room_graph.json "${STAGE}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}"/qa/topo_map.json "${STAGE}/evidence/qa/" 2>/dev/null || true
cp "${OUT_ROOT}"/qa/traj_room_sequence.csv "${STAGE}/evidence/qa/" 2>/dev/null || true

mkdir -p "${STAGE}/evidence/qa/overlay"
for p in "${OUT_ROOT}"/qa/overlay/overlay_rgb_*.png; do
  [ -f "${p}" ] || continue
  cp "${p}" "${STAGE}/evidence/qa/overlay/"
done

python3 - "${OUT_ROOT}/debug_runs" "${STAGE}/evidence/capture" <<'PY'
from pathlib import Path
import shutil, sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
if not src.exists():
    raise SystemExit(0)
# copy up to 20 rgb/depth files total across run dirs
rgb = []
depth = []
for p in sorted(src.glob('**/rgb_*.png')):
    rgb.append(p)
for p in sorted(src.glob('**/depth_*.npy')):
    depth.append(p)
for p in rgb[:20]:
    out = dst / ('rgb_' + p.parent.name + '_' + p.name)
    shutil.copy2(p, out)
for p in depth[:10]:
    out = dst / ('depth_' + p.parent.name + '_' + p.name)
    shutil.copy2(p, out)
for p in sorted(src.glob('**/motion_report.json'))[:6]:
    shutil.copy2(p, dst / ('motion_' + p.parent.name + '.json'))
for p in sorted(src.glob('**/motion_report.md'))[:6]:
    shutil.copy2(p, dst / ('motion_' + p.parent.name + '.md'))
for p in sorted(src.glob('**/rgb.gif'))[:4]:
    shutil.copy2(p, dst / ('gif_' + p.parent.name + '.gif'))
PY

cp scripts/isaac/run_ui_office_nav2_v34b.sh "${STAGE}/evidence/src_snapshot/"
cp scripts/isaac/ui_office_nav2_probe_v34b.py "${STAGE}/evidence/src_snapshot/"
cp scripts/nav2/generate_office_map_v34b.py "${STAGE}/evidence/src_snapshot/"
cp scripts/nav2/generate_office_map_v34b.sh "${STAGE}/evidence/src_snapshot/"
cp scripts/nav2/bringup_nav2_office_v34b.sh "${STAGE}/evidence/src_snapshot/"
cp scripts/nav2/send_goal_v34b.py "${STAGE}/evidence/src_snapshot/"
cp scripts/topo_eval_suite_v34b_nav2_ui_demo.sh "${STAGE}/evidence/src_snapshot/"
cp scripts/package_topo_mvp_proof_v34b_nav2_ui_demo.sh "${STAGE}/evidence/src_snapshot/"
cp scripts/topo_backend_runner.py "${STAGE}/evidence/src_snapshot/"
cp scripts/topo_topomap_room_qa_v34.py "${STAGE}/evidence/src_snapshot/"
cp configs/isaac_scenes_v34b.yaml "${STAGE}/evidence/src_snapshot/"
cp configs/nav2_params_v34b.yaml "${STAGE}/evidence/src_snapshot/"
cp configs/rviz_nav2_v34b.rviz "${STAGE}/evidence/src_snapshot/"

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
