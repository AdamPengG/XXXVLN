#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
RUN_DIR="${ROOT_DIR}/runs/topo_mvp/v27_isaac"
PROOF_DIR="${ROOT_DIR}/forGPT/v27_isaac_PROOF"
ZIP_PATH="${ROOT_DIR}/forGPT/v27_isaac_PROOF.zip"
DOC_DIR="${ROOT_DIR}/forGPT"
README_PATH="${PROOF_DIR}/README_v27.md"

rm -rf "${PROOF_DIR}"
mkdir -p "${PROOF_DIR}" "${DOC_DIR}"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "${src}" ]; then
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
  fi
}

copy_if_exists "${RUN_DIR}/logs" "${PROOF_DIR}/logs"
copy_if_exists "${RUN_DIR}/render_smoke" "${PROOF_DIR}/render_smoke"
copy_if_exists "${RUN_DIR}/debug_runs" "${PROOF_DIR}/debug_runs"
copy_if_exists "${RUN_DIR}/condition_gt_pose" "${PROOF_DIR}/condition_gt_pose"
copy_if_exists "${RUN_DIR}/builds" "${PROOF_DIR}/builds"
copy_if_exists "${ROOT_DIR}/scripts/isaac/render_smoke_v27.sh" "${PROOF_DIR}/proof_src/render_smoke_v27.sh"
copy_if_exists "${ROOT_DIR}/scripts/topo_eval_suite_v27_isaac.sh" "${PROOF_DIR}/proof_src/topo_eval_suite_v27_isaac.sh"
copy_if_exists "${ROOT_DIR}/scripts/gpu/gpu_router.py" "${PROOF_DIR}/proof_src/gpu_router.py"
copy_if_exists "${ROOT_DIR}/repo/InternNav/internnav/sim_backend/isaac_backend.py" "${PROOF_DIR}/proof_src/isaac_backend.py"
copy_if_exists "${ROOT_DIR}/repo/InternNav/internnav/sim_backend/isaac_render_smoke.py" "${PROOF_DIR}/proof_src/isaac_render_smoke.py"
copy_if_exists "${ROOT_DIR}/repo/InternNav/internnav/topo/debug_capture.py" "${PROOF_DIR}/proof_src/debug_capture.py"
copy_if_exists "${ROOT_DIR}/scripts/topo_debug_ui_build_report.py" "${PROOF_DIR}/proof_src/topo_debug_ui_build_report.py"

gpu_line="$(rg -o '\[GPU_PICK\].*role=isaac.*' -N "${RUN_DIR}/logs/topo_eval_suite_v27_isaac.log" | tail -n 1 || true)"
renderer_line="$(rg -o '\[ISAAC_RENDERER\].*ok=1.*' -N "${RUN_DIR}/logs/topo_eval_suite_v27_isaac.log" | tail -n 1 || true)"
smoke_line="$(rg -o '\[ISAAC_RENDER_SMOKE\].*' -N "${RUN_DIR}/logs/topo_eval_suite_v27_isaac.log" | tail -n 1 || true)"

cat > "${README_PATH}" <<EOF
# v27 Isaac Proof

- GPU route: ${gpu_line}
- Renderer: ${renderer_line}
- Render smoke: ${smoke_line}

## Reproduce
\`\`\`bash
cd /home/peng/DualVLN
bash scripts/isaac/render_smoke_v27.sh
bash scripts/topo_eval_suite_v27_isaac.sh --small
\`\`\`
EOF

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

echo "[ZIP_CHECK_OK] bad_count=${zip_bad_count} path=${DOC_DIR}/v27_isaac_ZIP_CHECK.txt" | tee "${DOC_DIR}/v27_isaac_ZIP_CHECK.txt"
if [ "${zip_bad_count}" != "0" ]; then
  echo "[PROOF_ZIP_BAD] bad_count=${zip_bad_count} zip=${ZIP_PATH}" >&2
  exit 2
fi
echo "[PROOF_ZIP_OK] bad_count=0 zip=${ZIP_PATH}"
