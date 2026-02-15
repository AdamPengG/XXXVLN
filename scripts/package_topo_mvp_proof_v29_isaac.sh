#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
RUN_DIR="${ROOT_DIR}/runs/topo_mvp/v29_isaac"
PROOF_DIR="${ROOT_DIR}/forGPT/v29_isaac_PROOF"
ZIP_PATH="${ROOT_DIR}/forGPT/v29_isaac_PROOF.zip"
README_PATH="${PROOF_DIR}/README_v29.md"

rm -rf "${PROOF_DIR}"
mkdir -p "${PROOF_DIR}"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "${src}" ]; then
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
  fi
}

copy_if_exists "${RUN_DIR}/logs" "${PROOF_DIR}/logs"
copy_if_exists "${RUN_DIR}/camera_fidelity" "${PROOF_DIR}/camera_fidelity"
copy_if_exists "${RUN_DIR}/render_smoke" "${PROOF_DIR}/render_smoke"
copy_if_exists "${RUN_DIR}/debug_runs" "${PROOF_DIR}/debug_runs"
copy_if_exists "${RUN_DIR}/condition_gt_pose" "${PROOF_DIR}/condition_gt_pose"
copy_if_exists "${RUN_DIR}/builds" "${PROOF_DIR}/builds"

mkdir -p "${PROOF_DIR}/proof_src"
copy_if_exists "${ROOT_DIR}/repo/InternNav/internnav/sim_backend/isaac_backend.py" "${PROOF_DIR}/proof_src/isaac_backend.py"
copy_if_exists "${ROOT_DIR}/repo/InternNav/internnav/sim_backend/isaac_camera_config_v29.py" "${PROOF_DIR}/proof_src/isaac_camera_config_v29.py"
copy_if_exists "${ROOT_DIR}/scripts/isaac/camera_fidelity_v29.sh" "${PROOF_DIR}/proof_src/camera_fidelity_v29.sh"
copy_if_exists "${ROOT_DIR}/scripts/isaac/camera_fidelity_v29.py" "${PROOF_DIR}/proof_src/camera_fidelity_v29.py"
copy_if_exists "${ROOT_DIR}/scripts/topo_eval_suite_v29_isaac.sh" "${PROOF_DIR}/proof_src/topo_eval_suite_v29_isaac.sh"
copy_if_exists "${ROOT_DIR}/scripts/package_topo_mvp_proof_v29_isaac.sh" "${PROOF_DIR}/proof_src/package_topo_mvp_proof_v29_isaac.sh"
copy_if_exists "${ROOT_DIR}/scripts/topo_debug_ui_build_report.py" "${PROOF_DIR}/proof_src/topo_debug_ui_build_report.py"
copy_if_exists "${ROOT_DIR}/scripts/topo_backend_runner.py" "${PROOF_DIR}/proof_src/topo_backend_runner.py"
copy_if_exists "${ROOT_DIR}/configs/isaac_scenes_v29.yaml" "${PROOF_DIR}/proof_src/isaac_scenes_v29.yaml"

python3 - <<'PY' "${RUN_DIR}" "${README_PATH}"
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
readme = Path(sys.argv[2])
cam_meta = run_dir / "camera_fidelity" / "camera_fidelity_meta.json"
smoke_meta = run_dir / "render_smoke" / "smoke_meta.json"
cam = json.loads(cam_meta.read_text()) if cam_meta.exists() else {}
smoke = json.loads(smoke_meta.read_text()) if smoke_meta.exists() else {}
camera = cam.get("camera", {}) if isinstance(cam, dict) else {}
rgb_stats = cam.get("rgb_stats", {}) if isinstance(cam, dict) else {}

lines = [
    "# v29 Isaac Camera Fidelity Proof",
    "",
    f"- scene: `{cam.get('scene_id', '')}`",
    f"- stage: `{cam.get('stage_path', '')}`",
    f"- camera_w: {camera.get('camera_w', 'n/a')}",
    f"- camera_h: {camera.get('camera_h', 'n/a')}",
    f"- camera_fov_deg: {camera.get('camera_fov_deg', 'n/a')}",
    f"- auto_exposure: {camera.get('auto_exposure', 'n/a')}",
    f"- exposure: {camera.get('exposure', 'n/a')}",
    f"- renderer_used: {camera.get('renderer_used', cam.get('renderer_used', 'unknown'))}",
    f"- fidelity_placeholder_ratio: {rgb_stats.get('placeholder_ratio', 'n/a')}",
    f"- smoke_placeholder_ratio: {smoke.get('placeholder_ratio', 'n/a')}",
    "",
    "## Reproduce",
    "```bash",
    "cd /home/peng/DualVLN",
    "bash scripts/isaac/camera_fidelity_v29.sh",
    "bash scripts/isaac/render_smoke_v27.sh --stage /home/peng/DualVLN/runs/topo_mvp/v28_isaac/assets_local/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd",
    "bash scripts/topo_eval_suite_v29_isaac.sh --small",
    "bash scripts/package_topo_mvp_proof_v29_isaac.sh",
    "```",
]
readme.write_text("\n".join(lines) + "\n")
PY

(
  cd "${PROOF_DIR}"
  find . -type f | sort | while read -r f; do
    sz=$(stat -c%s "$f" 2>/dev/null || wc -c <"$f")
    sha1=$(sha1sum "$f" | awk '{print $1}')
    printf "%s %s %s\n" "$sha1" "$sz" "$f"
  done
) > "${PROOF_DIR}/MANIFEST.txt"

bad_count=$(
  python3 - <<PY
from pathlib import Path
blocked = (".pt", ".pth", ".ckpt", ".bin", ".safetensors")
root = Path(r"${PROOF_DIR}")
bad = [str(p) for p in root.rglob("*") if p.is_file() and p.name.lower().endswith(blocked)]
print(len(bad))
PY
)
if [ "${bad_count}" != "0" ]; then
  echo "[PROOF_ZIP_BAD] bad_count=${bad_count} zip=${ZIP_PATH}" >&2
  exit 2
fi

rm -f "${ZIP_PATH}"
(
  cd "${PROOF_DIR}"
  zip -r "${ZIP_PATH}" . -x "*.pt" "*.pth" "*.ckpt" "*.bin" "*.safetensors" >/dev/null
)

zip_bad_count=$(
  python3 - <<PY
import zipfile
blocked = (".pt", ".pth", ".ckpt", ".bin", ".safetensors")
with zipfile.ZipFile(r"${ZIP_PATH}") as zf:
    bad = [n for n in zf.namelist() if n.lower().endswith(blocked)]
print(len(bad))
PY
)
if [ "${zip_bad_count}" != "0" ]; then
  echo "[PROOF_ZIP_BAD] bad_count=${zip_bad_count} zip=${ZIP_PATH}" >&2
  exit 2
fi
echo "[PROOF_ZIP_OK] bad_count=0 zip=${ZIP_PATH}"
