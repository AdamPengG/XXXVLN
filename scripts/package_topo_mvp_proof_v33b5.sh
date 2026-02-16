#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "$ROOT"

TS=$(date +%Y%m%d_%H%M)
ZIP="/home/peng/DualVLN/forGPT/v33b5_PROOF_${TS}.zip"
STAGE="/tmp/v33b5_proof_${TS}"
BASE_ROOT="runs/topo_mvp/v33b_baseline"
V33B_ROOT="runs/topo_mvp/v33b_costmap_avoid"
DEPTH_ROOT="runs/topo_mvp/v33b4_depth_fidelity"

rm -rf "$STAGE"
mkdir -p "$STAGE/evidence" "$STAGE/evidence/baseline" "$STAGE/evidence/v33b" "$STAGE/evidence/depth" "$STAGE/evidence/src" "$STAGE/evidence/git"

copy_core() {
  local src_root="$1"
  local dst_root="$2"
  mkdir -p "$dst_root/logs" "$dst_root/failure_cards"
  cp "$src_root/summary.json" "$dst_root/" 2>/dev/null || true
  cp "$src_root/failure_cases.jsonl" "$dst_root/" 2>/dev/null || true
  cp "$src_root/failure_cards/index.html" "$dst_root/failure_cards/" 2>/dev/null || true
  cp "$src_root/failure_cards/cards.json" "$dst_root/failure_cards/" 2>/dev/null || true
  cp "$src_root/logs"/topo_eval_suite_*.log "$dst_root/logs/" 2>/dev/null || true
  cp "$src_root/logs"/topo_planb_scale_eval_v31.log "$dst_root/logs/" 2>/dev/null || true
}

copy_core "$BASE_ROOT" "$STAGE/evidence/baseline"
copy_core "$V33B_ROOT" "$STAGE/evidence/v33b"

copy_representative_logs() {
  local src_root="$1"
  local dst_root="$2"
  local fc="$src_root/failure_cases.jsonl"
  [ -f "$fc" ] || return 0
  python3 - "$fc" "$dst_root/logs" <<'PY'
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
# first failure
for r in rows:
    if int(r.get('success', 0)) == 0:
        lp = pathlib.Path(str(r.get('log_path', '')))
        if lp.is_file():
            shutil.copy2(lp, out / f"failure_{lp.name}")
            break
# first success
for r in rows:
    if int(r.get('success', 0)) == 1:
        lp = pathlib.Path(str(r.get('log_path', '')))
        if lp.is_file():
            shutil.copy2(lp, out / f"success_{lp.name}")
            break
PY
}

copy_representative_logs "$BASE_ROOT" "$STAGE/evidence/baseline"
copy_representative_logs "$V33B_ROOT" "$STAGE/evidence/v33b"

mkdir -p "$STAGE/evidence/depth/capture"
cp "$DEPTH_ROOT/logs/depth_fidelity_v33b4.log" "$STAGE/evidence/depth/" 2>/dev/null || true
cp "$DEPTH_ROOT/capture/depth_fidelity_meta.json" "$STAGE/evidence/depth/" 2>/dev/null || true
cp "$DEPTH_ROOT/capture/motion_report.json" "$STAGE/evidence/depth/" 2>/dev/null || true
cp "$DEPTH_ROOT/capture/motion_report.md" "$STAGE/evidence/depth/" 2>/dev/null || true
cp "$DEPTH_ROOT/capture/rgb.gif" "$STAGE/evidence/depth/" 2>/dev/null || true
for p in "$DEPTH_ROOT"/capture/rgb/*.png; do [ -f "$p" ] && cp "$p" "$STAGE/evidence/depth/capture/"; done
for p in "$DEPTH_ROOT"/capture/depth_png/*.png; do [ -f "$p" ] && cp "$p" "$STAGE/evidence/depth/capture/"; done
python3 - <<'PY' "$DEPTH_ROOT" "$STAGE/evidence/depth/capture"
from pathlib import Path
import shutil, sys
src = Path(sys.argv[1]) / 'capture' / 'depth_npy'
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
for p in sorted(src.glob('*.npy'))[:10]:
    shutil.copy2(p, out / p.name)
PY

cp scripts/topo_backend_runner.py "$STAGE/evidence/src/"
cp scripts/topo_planb_scale_eval_v31.py "$STAGE/evidence/src/"
cp scripts/topo_eval_suite_v33b_costmap_avoid.sh "$STAGE/evidence/src/"
cp scripts/isaac/depth_fidelity_v33b4.py "$STAGE/evidence/src/"
cp scripts/isaac/depth_fidelity_v33b4.sh "$STAGE/evidence/src/"
cp scripts/tools/check_capture_motion.py "$STAGE/evidence/src/"
cp scripts/tools/check_capture_motion.sh "$STAGE/evidence/src/"
cp scripts/package_topo_mvp_proof_v33b5.sh "$STAGE/evidence/src/"

git rev-parse --abbrev-ref HEAD > "$STAGE/evidence/git/branch.txt"
git rev-parse HEAD > "$STAGE/evidence/git/rev_parse_head.txt"
git status --porcelain > "$STAGE/evidence/git/status_porcelain.txt"
git diff --stat > "$STAGE/evidence/git/diff_stat.txt"
git diff > "$STAGE/evidence/git/diff.patch"
git log -n 20 --oneline > "$STAGE/evidence/git/log_oneline_20.txt"

mkdir -p /home/peng/DualVLN/forGPT
(cd "$STAGE" && zip -r "$ZIP" evidence >/dev/null)
unzip -t "$ZIP" >/dev/null
SHA=$(sha256sum "$ZIP" | awk '{print $1}')
SIZE=$(stat -c%s "$ZIP")
BAD_COUNT=$( (zipinfo -1 "$ZIP" | grep -E '\.(pt|pth|ckpt|bin|safetensors)$' || true) | wc -l )

echo "[PROOF_ZIP_OK] zip=$ZIP sha256=$SHA size_bytes=$SIZE bad_count=$BAD_COUNT"
echo "ZIP=$ZIP"
