#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

TS="$(date +%Y%m%d_%H%M)"
ZIP="/home/peng/DualVLN/forGPT/v34a_dual_gpu_dualvln_PROOF_${TS}.zip"
STAGE="/tmp/v34a_dual_gpu_proof_${TS}"
OUT_ROOT="runs/topo_mvp/v34a_dual_gpu"
BASE_ROOT="${OUT_ROOT}/baseline"
DUAL_ROOT="${OUT_ROOT}/dualvln"

rm -rf "${STAGE}"
mkdir -p "${STAGE}/evidence/baseline" "${STAGE}/evidence/dualvln" "${STAGE}/evidence/server" "${STAGE}/evidence/src" "${STAGE}/evidence/git"

copy_core_root() {
  local src_root="$1"
  local dst_root="$2"
  mkdir -p "${dst_root}/logs" "${dst_root}/failure_cards" "${dst_root}/captures"
  cp "${src_root}/summary.json" "${dst_root}/" 2>/dev/null || true
  cp "${src_root}/failure_cases.jsonl" "${dst_root}/" 2>/dev/null || true
  cp "${src_root}/runs.csv" "${dst_root}/" 2>/dev/null || true
  cp "${src_root}/index.html" "${dst_root}/" 2>/dev/null || true
  cp "${src_root}/failure_cards/index.html" "${dst_root}/failure_cards/" 2>/dev/null || true
  cp "${src_root}/failure_cards/cards.json" "${dst_root}/failure_cards/" 2>/dev/null || true
  cp "${src_root}/logs"/topo_*.log "${dst_root}/logs/" 2>/dev/null || true
  cp "${src_root}/logs"/*.log "${dst_root}/logs/" 2>/dev/null || true
}

copy_repr_logs() {
  local src_root="$1"
  local dst_root="$2"
  local fc="${src_root}/failure_cases.jsonl"
  [ -f "${fc}" ] || return 0
  python3 - "${fc}" "${dst_root}/logs" <<'PY'
import json, pathlib, shutil, sys
fc = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
rows = []
for line in fc.read_text(encoding='utf-8', errors='ignore').splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except Exception:
        continue
picked = 0
for r in rows:
    if int(r.get("success", 0)) == 1:
        lp = pathlib.Path(str(r.get("log_path", "")))
        if lp.is_file():
            shutil.copy2(lp, out / f"success_{lp.name}")
            picked += 1
            break
for r in rows:
    if int(r.get("success", 0)) == 0:
        lp = pathlib.Path(str(r.get("log_path", "")))
        if lp.is_file():
            shutil.copy2(lp, out / f"failure_{lp.name}")
            picked += 1
            break
PY
}

copy_capture_subset() {
  local src_root="$1"
  local dst_root="$2"
  python3 - "${src_root}" "${dst_root}" <<'PY'
import json, pathlib, shutil, sys
root = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
fc = root / "failure_cases.jsonl"
dbg_dir = None
if fc.exists():
    for line in fc.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        p = pathlib.Path(str(rec.get("debug_index", "") or ""))
        if p.exists():
            dbg_dir = p.parent.parent
            break
if dbg_dir is None:
    for p in sorted((root / "debug_runs").glob("*/*")):
        if p.is_dir():
            dbg_dir = p
            break
if dbg_dir is None:
    raise SystemExit(0)
shutil.copy2(dbg_dir / "trace.jsonl", out / "trace.jsonl") if (dbg_dir / "trace.jsonl").is_file() else None
for name in ("motion_report.json", "motion_report.md", "rgb.gif"):
    p = dbg_dir / name
    if p.is_file():
        shutil.copy2(p, out / name)
report_dir = dbg_dir / "report"
if report_dir.is_dir():
    dst_report = out / "report"
    shutil.copytree(report_dir, dst_report, dirs_exist_ok=True)
frames_dir = dbg_dir / "frames"
if frames_dir.is_dir():
    dst_frames = out / "frames"
    dst_frames.mkdir(parents=True, exist_ok=True)
    imgs = sorted(frames_dir.glob("rgb_*.*"))[:20]
    for p in imgs:
        shutil.copy2(p, dst_frames / p.name)
    depths = sorted(frames_dir.glob("depth_*.*"))[:20]
    for p in depths:
        shutil.copy2(p, dst_frames / p.name)
PY
}

copy_core_root "${BASE_ROOT}" "${STAGE}/evidence/baseline"
copy_core_root "${DUAL_ROOT}" "${STAGE}/evidence/dualvln"
copy_repr_logs "${BASE_ROOT}" "${STAGE}/evidence/baseline"
copy_repr_logs "${DUAL_ROOT}" "${STAGE}/evidence/dualvln"
copy_capture_subset "${BASE_ROOT}" "${STAGE}/evidence/baseline/captures"
copy_capture_subset "${DUAL_ROOT}" "${STAGE}/evidence/dualvln/captures"

cp "${OUT_ROOT}/logs/topo_eval_suite_v34a_dual_gpu_dualvln.log" "${STAGE}/evidence/" 2>/dev/null || true
cp "${OUT_ROOT}/server/dualvln_server.log" "${STAGE}/evidence/server/" 2>/dev/null || true
cp "${OUT_ROOT}/server/dualvln_server.pid" "${STAGE}/evidence/server/" 2>/dev/null || true

cp scripts/topo_backend_runner.py "${STAGE}/evidence/src/"
cp scripts/topo_planb_scale_eval_v31.py "${STAGE}/evidence/src/"
cp scripts/topo_failure_cards_v31.py "${STAGE}/evidence/src/"
cp scripts/topo_eval_suite_v34a_dual_gpu_dualvln.sh "${STAGE}/evidence/src/"
cp scripts/package_topo_mvp_proof_v34a_dual_gpu_dualvln.sh "${STAGE}/evidence/src/"
cp scripts/dualvln/dualvln_server_v34a.py "${STAGE}/evidence/src/"
cp configs/isaac_goal_catalog_v34a.yaml "${STAGE}/evidence/src/"
cp configs/isaac_planb_eval_v34a.yaml "${STAGE}/evidence/src/"

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

mkdir -p /home/peng/DualVLN/forGPT
(cd "${STAGE}" && zip -r "${ZIP}" evidence >/dev/null)
unzip -t "${ZIP}" >/dev/null
SHA="$(sha256sum "${ZIP}" | awk '{print $1}')"
SIZE="$(stat -c%s "${ZIP}")"
BAD_COUNT="$( (zipinfo -1 "${ZIP}" | grep -E '\.(pt|pth|ckpt|bin|safetensors)$' || true) | wc -l )"

echo "[PROOF_ZIP_OK] zip=${ZIP} sha256=${SHA} size_bytes=${SIZE} bad_count=${BAD_COUNT}"
echo "ZIP=${ZIP}"
