#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "$ROOT"

TS=$(date +%Y%m%d_%H%M)
ZIP="/home/peng/DualVLN/forGPT/v33b3_PROOF_${TS}.zip"
STAGE="/tmp/v33b3_proof_${TS}"
BASE_ROOT="runs/topo_mvp/v33b_baseline"
V33B_ROOT="runs/topo_mvp/v33b_costmap_avoid"

rm -rf "$STAGE"
mkdir -p "$STAGE/evidence/baseline" "$STAGE/evidence/v33b" "$STAGE/evidence/src" "$STAGE/evidence/git"

copy_core() {
  local src_root="$1"
  local dst_root="$2"
  mkdir -p "$dst_root/logs" "$dst_root/failure_cards"
  cp "$src_root/summary.json" "$dst_root/" 2>/dev/null || true
  cp "$src_root/failure_cases.jsonl" "$dst_root/" 2>/dev/null || true
  cp "$src_root/failure_cards/index.html" "$dst_root/failure_cards/" 2>/dev/null || true
  cp "$src_root/failure_cards/cards.json" "$dst_root/failure_cards/" 2>/dev/null || true
  cp "$src_root/logs"/topo_eval_suite_*.log "$dst_root/logs/" 2>/dev/null || true
}

copy_core "$BASE_ROOT" "$STAGE/evidence/baseline"
copy_core "$V33B_ROOT" "$STAGE/evidence/v33b"

copy_failed_logs() {
  local src_root="$1"
  local dst_root="$2"
  local fc="$src_root/failure_cases.jsonl"
  [ -f "$fc" ] || return 0
  python3 - "$fc" "$dst_root/logs" <<'PY'
import json, pathlib, sys, shutil
fc = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for line in fc.read_text(encoding='utf-8').splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except Exception:
        continue
    fail = str(rec.get('fail_type', ''))
    if fail not in {'stuck','max_steps','unknown'}:
        continue
    lp = pathlib.Path(str(rec.get('log_path','')))
    if lp.is_file():
        shutil.copy2(lp, out / lp.name)
PY
}

copy_failed_logs "$BASE_ROOT" "$STAGE/evidence/baseline"
copy_failed_logs "$V33B_ROOT" "$STAGE/evidence/v33b"

cp scripts/topo_backend_runner.py "$STAGE/evidence/src/"
cp scripts/topo_planb_scale_eval_v31.py "$STAGE/evidence/src/"
cp scripts/topo_eval_suite_v33b_costmap_avoid.sh "$STAGE/evidence/src/"
cp scripts/package_topo_mvp_proof_v33b3.sh "$STAGE/evidence/src/"

git rev-parse HEAD > "$STAGE/evidence/git/rev_parse_head.txt"
git status --porcelain > "$STAGE/evidence/git/status_porcelain.txt"
git diff > "$STAGE/evidence/git/diff.patch"

mkdir -p /home/peng/DualVLN/forGPT
(cd "$STAGE" && zip -r "$ZIP" evidence >/dev/null)
unzip -t "$ZIP" >/dev/null
SHA=$(sha256sum "$ZIP" | awk '{print $1}')
SIZE=$(stat -c%s "$ZIP")
BAD_COUNT=$( (zipinfo -1 "$ZIP" | grep -E '\.(pt|pth|ckpt|bin|safetensors)$' || true) | wc -l)

echo "[PROOF_ZIP_OK] zip=$ZIP sha256=$SHA size_bytes=$SIZE bad_count=$BAD_COUNT"
echo "ZIP=$ZIP"
