#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
PROOF_TAG="v24_isaac"
RUN_DIR="${ROOT_DIR}/runs/topo_mvp/v24_isaac"
DOC_DIR="${ROOT_DIR}/forGPT/${PROOF_TAG}"
PROOF_DIR="${ROOT_DIR}/forGPT/topo_mvp_PROOF_${PROOF_TAG}"
ZIP_PATH="${ROOT_DIR}/forGPT/topo_mvp_PROOF_${PROOF_TAG}.zip"

rm -rf "${PROOF_DIR}"
mkdir -p "${PROOF_DIR}/proof_src" "${PROOF_DIR}/forGPT/${PROOF_TAG}" "${DOC_DIR}"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "${src}" ]; then
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
  fi
}

{
  echo "timestamp=$(date -u +%FT%TZ)"
  echo "resolved_root=$(bash ${ROOT_DIR}/scripts/isaac/find_isaac_root.sh 2>/dev/null || true)"
  echo "preflight_start"
  bash "${ROOT_DIR}/scripts/isaac/run_with_isaac_python.sh" -- -c "print('ISAAC_ENV_CHECK_OK')"
  echo "preflight_end"
} > "${DOC_DIR}/ENV_CHECK.txt" 2>&1 || true

copy_if_exists "${RUN_DIR}" "${PROOF_DIR}/runs/v24_isaac"
copy_if_exists "${ROOT_DIR}/configs/isaac_scenes_v24.yaml" "${PROOF_DIR}/configs/isaac_scenes_v24.yaml"
copy_if_exists "${DOC_DIR}/ENV_CHECK.txt" "${PROOF_DIR}/forGPT/${PROOF_TAG}/ENV_CHECK.txt"

for f in \
  repo/InternNav/internnav/sim_backend/base.py \
  repo/InternNav/internnav/sim_backend/habitat_backend.py \
  repo/InternNav/internnav/sim_backend/isaac_backend.py \
  repo/InternNav/internnav/topo/debug_capture.py \
  repo/InternNav/internnav/topo/local_planner.py \
  scripts/topo_backend_runner.py \
  scripts/topo_backend_runner_isaac.py \
  scripts/topo_run_backend.sh \
  scripts/topo_eval_suite_v24_isaac.sh \
  scripts/topo_isaac_sanity_check.py \
  scripts/topo_debug_ui_build_report.py \
  scripts/topo_debug_ui_index.py \
  scripts/topo_debug_ui.sh \
  scripts/isaac/find_isaac_root.sh \
  scripts/isaac/find_usd_asset.sh \
  scripts/isaac/run_with_isaac_python.sh \
  scripts/package_topo_mvp_proof_v24_isaac.sh
 do
  copy_if_exists "${ROOT_DIR}/${f}" "${PROOF_DIR}/proof_src/${f}"
 done

for f in COMMANDS.txt files_changed.txt walkthrough.md DIAGNOSIS.md RESULT_SUMMARY.md ZIP_CHECK.txt; do
  if [ -f "${DOC_DIR}/${f}" ]; then
    copy_if_exists "${DOC_DIR}/${f}" "${PROOF_DIR}/forGPT/${PROOF_TAG}/${f}"
  fi
done

(
  cd "${PROOF_DIR}"
  find . -type f | sort | while read -r f; do
    sz=$(stat -c%s "$f" 2>/dev/null || wc -c <"$f")
    sha1=$(sha1sum "$f" | awk '{print $1}')
    printf "%s %s %s\n" "$sha1" "$sz" "$f"
  done
) > "${PROOF_DIR}/MANIFEST.txt"

bad_count=$(
  python - <<PY
from pathlib import Path
blocked = (".pt", ".pth", ".ckpt", ".bin", ".safetensors")
root = Path(r"${PROOF_DIR}")
bad = [str(p) for p in root.rglob("*") if p.is_file() and p.name.lower().endswith(blocked)]
print(len(bad))
PY
)

echo "[ZIP_CHECK_OK] bad_count=${bad_count} path=${DOC_DIR}/ZIP_CHECK.txt" > "${DOC_DIR}/ZIP_CHECK.txt"
copy_if_exists "${DOC_DIR}/ZIP_CHECK.txt" "${PROOF_DIR}/forGPT/${PROOF_TAG}/ZIP_CHECK.txt"
echo "[ZIP_CHECK_OK] bad_count=${bad_count} path=${DOC_DIR}/ZIP_CHECK.txt"
if [ "${bad_count}" != "0" ]; then
  echo "[PROOF_ZIP_BAD] bad_count=${bad_count} zip=${ZIP_PATH}" >&2
  exit 2
fi

rm -f "${ZIP_PATH}"
(
  cd "${PROOF_DIR}"
  zip -r "${ZIP_PATH}" . \
    -x "*.pt" "*.pth" "*.safetensors" "*.bin" "*.ckpt" "*.pt.*" "*.pth.*" "*.ckpt.*"
)

zip_bad_count=$(
  python - <<PY
import zipfile
z = zipfile.ZipFile(r"${ZIP_PATH}")
blocked = (".pt", ".pth", ".ckpt", ".bin", ".safetensors")
bad = [n for n in z.namelist() if n.lower().endswith(blocked)]
print(len(bad))
PY
)
if [ "${zip_bad_count}" != "0" ]; then
  echo "[PROOF_ZIP_BAD] bad_count=${zip_bad_count} zip=${ZIP_PATH}" >&2
  exit 2
fi
echo "[PROOF_ZIP_OK] bad_count=0 zip=${ZIP_PATH}"
