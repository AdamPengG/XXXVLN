#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
RUN_DIR="${ROOT_DIR}/runs/topo_mvp/v28_isaac"
PROOF_DIR="${ROOT_DIR}/forGPT/v28_isaac_PROOF"
ZIP_PATH="${ROOT_DIR}/forGPT/v28_isaac_PROOF.zip"
README_PATH="${PROOF_DIR}/README_v28.md"

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
copy_if_exists "${RUN_DIR}/asset_audit" "${PROOF_DIR}/asset_audit"
copy_if_exists "${RUN_DIR}/render_smoke" "${PROOF_DIR}/render_smoke"
copy_if_exists "${RUN_DIR}/debug_runs" "${PROOF_DIR}/debug_runs"
copy_if_exists "${RUN_DIR}/condition_gt_pose" "${PROOF_DIR}/condition_gt_pose"
copy_if_exists "${RUN_DIR}/builds" "${PROOF_DIR}/builds"

mkdir -p "${PROOF_DIR}/assets_local"
copy_if_exists "${RUN_DIR}/assets_local/manifest_copied.json" "${PROOF_DIR}/assets_local/manifest_copied.json"
copy_if_exists "${RUN_DIR}/assets_local/manifest_copied.txt" "${PROOF_DIR}/assets_local/manifest_copied.txt"

mkdir -p "${PROOF_DIR}/proof_src"
copy_if_exists "${ROOT_DIR}/repo/InternNav/internnav/sim_backend/isaac_asset_audit_v28.py" "${PROOF_DIR}/proof_src/isaac_asset_audit_v28.py"
copy_if_exists "${ROOT_DIR}/repo/InternNav/internnav/sim_backend/isaac_asset_localize_v28.py" "${PROOF_DIR}/proof_src/isaac_asset_localize_v28.py"
copy_if_exists "${ROOT_DIR}/repo/InternNav/internnav/sim_backend/isaac_backend.py" "${PROOF_DIR}/proof_src/isaac_backend.py"
copy_if_exists "${ROOT_DIR}/repo/InternNav/internnav/sim_backend/isaac_render_smoke.py" "${PROOF_DIR}/proof_src/isaac_render_smoke.py"
copy_if_exists "${ROOT_DIR}/repo/InternNav/internnav/topo/debug_capture.py" "${PROOF_DIR}/proof_src/debug_capture.py"
copy_if_exists "${ROOT_DIR}/scripts/isaac/asset_audit_v28.sh" "${PROOF_DIR}/proof_src/asset_audit_v28.sh"
copy_if_exists "${ROOT_DIR}/scripts/isaac/asset_localize_v28.sh" "${PROOF_DIR}/proof_src/asset_localize_v28.sh"
copy_if_exists "${ROOT_DIR}/scripts/isaac/render_smoke_v27.sh" "${PROOF_DIR}/proof_src/render_smoke_v27.sh"
copy_if_exists "${ROOT_DIR}/scripts/topo_eval_suite_v28_isaac.sh" "${PROOF_DIR}/proof_src/topo_eval_suite_v28_isaac.sh"
copy_if_exists "${ROOT_DIR}/scripts/package_topo_mvp_proof_v28_isaac.sh" "${PROOF_DIR}/proof_src/package_topo_mvp_proof_v28_isaac.sh"
copy_if_exists "${ROOT_DIR}/scripts/topo_debug_ui_build_report.py" "${PROOF_DIR}/proof_src/topo_debug_ui_build_report.py"
copy_if_exists "${ROOT_DIR}/scripts/gpu/gpu_router.py" "${PROOF_DIR}/proof_src/gpu_router.py"
copy_if_exists "${ROOT_DIR}/configs/isaac_scenes_v28.yaml" "${PROOF_DIR}/proof_src/isaac_scenes_v28.yaml"

audit_json="${RUN_DIR}/asset_audit/asset_audit_report.json"
manifest_json="${RUN_DIR}/assets_local/manifest_copied.json"
smoke_json="${RUN_DIR}/render_smoke/smoke_meta.json"

python3 - <<'PY' "${audit_json}" "${manifest_json}" "${smoke_json}" "${README_PATH}"
import json, sys
from pathlib import Path

audit_p = Path(sys.argv[1])
manifest_p = Path(sys.argv[2])
smoke_p = Path(sys.argv[3])
readme = Path(sys.argv[4])
audit = json.loads(audit_p.read_text()) if audit_p.exists() else {}
manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}
smoke = json.loads(smoke_p.read_text()) if smoke_p.exists() else {}

lines = [
    "# v28 Isaac Asset Self-Contained Proof",
    "",
    f"- stage: `{audit.get('stage', '')}`",
    f"- assets_root: `{audit.get('assets_root', '')}`",
    f"- total deps: {audit.get('total_dependencies', 0)}",
    f"- missing: {audit.get('counts', {}).get('missing', 0)}",
    f"- unreadable: {audit.get('counts', {}).get('unreadable', 0)}",
    f"- outside_root: {audit.get('counts', {}).get('outside_root', 0)}",
    f"- localized stage: `{manifest.get('stage_localized', '')}`",
    f"- files_copied: {manifest.get('files_copied', 0)}",
    f"- smoke placeholder_ratio: {smoke.get('placeholder_ratio', 1.0)}",
    "",
    "## Reproduce",
    "```bash",
    "cd /home/peng/DualVLN",
    "bash scripts/isaac/asset_audit_v28.sh",
    "bash scripts/isaac/asset_localize_v28.sh",
    "bash scripts/topo_eval_suite_v28_isaac.sh --small",
    "bash scripts/package_topo_mvp_proof_v28_isaac.sh",
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
  zip -r "${ZIP_PATH}" . -x "*.pt" "*.pth" "*.ckpt" "*.bin" "*.safetensors"
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
