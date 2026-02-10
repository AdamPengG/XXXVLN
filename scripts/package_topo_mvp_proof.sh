#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
PROOF_TAG="${TOPO_PROOF_TAG:-v18}"
RUN_TAG_DIR="${TOPO_RUN_TAG_DIR:-${PROOF_TAG}}"
RUN_ROOT="${TOPO_RUN_ROOT:-${ROOT_DIR}/runs/topo_mvp}"
RUN_SUITE_DIR="${RUN_ROOT}/${RUN_TAG_DIR}"
DOC_DIR="${ROOT_DIR}/forGPT/${PROOF_TAG}"
PROOF_DIR="${ROOT_DIR}/forGPT/topo_mvp_PROOF_${PROOF_TAG}"
ZIP_PATH="${ROOT_DIR}/forGPT/topo_mvp_PROOF_${PROOF_TAG}.zip"

rm -rf "${PROOF_DIR}"
mkdir -p "${PROOF_DIR}/proof_src" "${PROOF_DIR}/forGPT" "${PROOF_DIR}/runs" "${PROOF_DIR}/logs" "${PROOF_DIR}/configs"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "${src}" ]; then
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
  fi
}

for f in \
  repo/InternNav/internnav/topo/graph.py \
  repo/InternNav/internnav/topo/explore.py \
  repo/InternNav/internnav/topo/relocalize.py \
  repo/InternNav/internnav/topo/retrieve_goal.py \
  repo/InternNav/internnav/topo/room_partition.py \
  repo/InternNav/internnav/topo/room_semantics.py \
  repo/InternNav/internnav/topo/debug_capture.py \
  repo/InternNav/internnav/topo/plan.py \
  repo/InternNav/internnav/topo/controller.py \
  repo/InternNav/internnav/topo/geo_descriptor.py \
  repo/InternNav/internnav/topo/pose_graph.py \
  scripts/topo_explore_build.sh \
  scripts/topo_room_semantics_audit.py \
  scripts/topo_query_navigate.sh \
  scripts/topo_eval_suite.sh \
  scripts/topo_eval_suite_v11.sh \
  scripts/topo_eval_suite_v12.sh \
  scripts/topo_eval_suite_v13.sh \
  scripts/topo_eval_suite_v14.sh \
  scripts/topo_eval_suite_v15.sh \
  scripts/topo_eval_suite_v16.sh \
  scripts/topo_eval_suite_v17.sh \
  scripts/topo_eval_suite_v18.sh \
  scripts/topo_eval_suite_v19.sh \
  scripts/topo_eval_suite_v20.sh \
  scripts/topo_debug_ui_build_report.py \
  scripts/topo_debug_ui_index.py \
  scripts/topo_debug_ui.sh \
  scripts/topo_room_semantics_evidence.py \
  scripts/package_topo_mvp_proof.sh
do
  copy_if_exists "${ROOT_DIR}/${f}" "${PROOF_DIR}/proof_src/$(basename "${f}")"
done

copy_if_exists "${ROOT_DIR}/repo/InternNav/scripts/eval/configs/vln_r2r.yaml" "${PROOF_DIR}/configs/vln_r2r.yaml"

if [ -d "${RUN_SUITE_DIR}" ]; then
  cp -a "${RUN_SUITE_DIR}" "${PROOF_DIR}/runs/"
fi

if [ -d "${DOC_DIR}" ]; then
  mkdir -p "${PROOF_DIR}/forGPT/${PROOF_TAG}"
  cp -a "${DOC_DIR}/." "${PROOF_DIR}/forGPT/${PROOF_TAG}/"
fi

for f in walkthrough.md COMMANDS.txt files_changed.txt RESULT_SUMMARY.md V9_DIAG_REPORT.md DIAGNOSIS.md; do
  if [ -f "${DOC_DIR}/${f}" ]; then
    copy_if_exists "${DOC_DIR}/${f}" "${PROOF_DIR}/forGPT/${f}"
  elif [ -f "${ROOT_DIR}/forGPT/${f}" ]; then
    copy_if_exists "${ROOT_DIR}/forGPT/${f}" "${PROOF_DIR}/forGPT/${f}"
  fi
done

if [ -d "${RUN_SUITE_DIR}/logs" ]; then
  cp -a "${RUN_SUITE_DIR}/logs/." "${PROOF_DIR}/logs/"
fi

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

if [ "${bad_count}" != "0" ]; then
  echo "[PROOF_ZIP_BAD] bad_count=${bad_count} zip=${ZIP_PATH}" >&2
  exit 2
fi
mkdir -p "${DOC_DIR}"
mkdir -p "${PROOF_DIR}/forGPT/${PROOF_TAG}"
echo "[ZIP_CHECK_OK] bad_count=0 path=${DOC_DIR}/ZIP_CHECK.txt" > "${DOC_DIR}/ZIP_CHECK.txt"
cp -a "${DOC_DIR}/ZIP_CHECK.txt" "${PROOF_DIR}/forGPT/${PROOF_TAG}/ZIP_CHECK.txt"
cp -a "${DOC_DIR}/ZIP_CHECK.txt" "${PROOF_DIR}/forGPT/ZIP_CHECK.txt"

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
echo "[ZIP_CHECK_OK] bad_count=0 path=${DOC_DIR}/ZIP_CHECK.txt"
echo "[PROOF_ZIP_OK] bad_count=0 zip=${ZIP_PATH}"
