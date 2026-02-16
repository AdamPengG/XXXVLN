#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M)"
OUT_ZIP="$ROOT/forGPT/v33c_sensor_fresh_PROOF_${STAMP}.zip"
STAGE_DIR="/tmp/v33c_sensor_fresh_${STAMP}"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/evidence/runs" "$STAGE_DIR/evidence/src" "$STAGE_DIR/evidence/git"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
  fi
}

copy_run_root() {
  local root_rel="$1"
  local root_abs="$ROOT/$root_rel"
  local dst_root="$STAGE_DIR/evidence/runs/$(basename "$root_rel")"
  mkdir -p "$dst_root/logs" "$dst_root/failure_cards"
  copy_if_exists "$root_abs/summary.json" "$dst_root/summary.json"
  copy_if_exists "$root_abs/failure_cases.jsonl" "$dst_root/failure_cases.jsonl"
  copy_if_exists "$root_abs/failure_cards/index.html" "$dst_root/failure_cards/index.html"
  copy_if_exists "$root_abs/failure_cards/cards.json" "$dst_root/failure_cards/cards.json"
  copy_if_exists "$root_abs/logs/topo_eval_suite_v33b_baseline.log" "$dst_root/logs/topo_eval_suite_v33b_baseline.log"
  copy_if_exists "$root_abs/logs/topo_eval_suite_v33b_run.log" "$dst_root/logs/topo_eval_suite_v33b_run.log"
  copy_if_exists "$root_abs/logs/topo_eval_suite_v33b_costmap_avoid.log" "$dst_root/logs/topo_eval_suite_v33b_costmap_avoid.log"
}

copy_run_root "runs/topo_mvp/v33b_baseline"
copy_run_root "runs/topo_mvp/v33b_costmap_avoid"

# Include representative run logs (pose success/failure where available)
copy_if_exists "$ROOT/runs/topo_mvp/v33b_baseline/logs/office_localized__start_003__office_pose_001.log" "$STAGE_DIR/evidence/runs/v33b_baseline/logs/office_localized__start_003__office_pose_001.log"
copy_if_exists "$ROOT/runs/topo_mvp/v33b_costmap_avoid/logs/office_localized__start_003__office_pose_001.log" "$STAGE_DIR/evidence/runs/v33b_costmap_avoid/logs/office_localized__start_003__office_pose_001.log"

STAGE_DIR_ENV="$STAGE_DIR" python3 - <<'PY'
import json
import pathlib
import shutil
import os

root = pathlib.Path('/home/peng/DualVLN')
stage_dir = pathlib.Path(os.environ['STAGE_DIR_ENV'])


def load_first_debug_run(failure_cases: pathlib.Path):
    if not failure_cases.exists():
        return None
    for line in failure_cases.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        dbg = str(rec.get('debug_index', '') or '').strip()
        if dbg:
            p = pathlib.Path(dbg)
            if p.exists():
                return p.parent.parent
    return None


def copy_repr(run_dir: pathlib.Path, out_dir: pathlib.Path, limit: int = 10):
    if run_dir is None or not run_dir.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = run_dir / 'frames'
    if frames.exists():
        rgbs = sorted(list(frames.glob('rgb_*.png')) + list(frames.glob('rgb_*.jpg')))
        for p in rgbs[:limit]:
            shutil.copy2(p, out_dir / p.name)
    for name in ['motion_report.json', 'motion_report.md', 'rgb.gif', 'trace.jsonl', 'meta.json']:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / src.name)

baseline_fc = root / 'runs/topo_mvp/v33b_baseline/failure_cases.jsonl'
v33b_fc = root / 'runs/topo_mvp/v33b_costmap_avoid/failure_cases.jsonl'

baseline_run = load_first_debug_run(baseline_fc)
v33b_run = load_first_debug_run(v33b_fc)

copy_repr(baseline_run, stage_dir / 'evidence/runs/v33b_baseline/repr_capture')
copy_repr(v33b_run, stage_dir / 'evidence/runs/v33b_costmap_avoid/repr_capture')

# Depth fidelity capture + reports
fid = root / 'runs/topo_mvp/v33b4_depth_fidelity/capture'
out_fid = stage_dir / 'evidence/runs/depth_fidelity_capture'
out_fid.mkdir(parents=True, exist_ok=True)
if fid.exists():
    rgbs = sorted(list(fid.glob('rgb_*.png')) + list(fid.glob('rgb_*.jpg')))
    deps = sorted(fid.glob('depth_*.png'))
    npys = sorted(fid.glob('depth_*.npy'))
    for p in rgbs[:10] + deps[:10] + npys[:3]:
        shutil.copy2(p, out_fid / p.name)
    for name in ['motion_report.json', 'motion_report.md', 'rgb.gif']:
        src = fid / name
        if src.exists():
            shutil.copy2(src, out_fid / src.name)
PY

# Source snapshots
copy_if_exists "$ROOT/scripts/topo_backend_runner.py" "$STAGE_DIR/evidence/src/scripts/topo_backend_runner.py"
copy_if_exists "$ROOT/scripts/topo_planb_scale_eval_v31.py" "$STAGE_DIR/evidence/src/scripts/topo_planb_scale_eval_v31.py"
copy_if_exists "$ROOT/scripts/topo_eval_suite_v33b_costmap_avoid.sh" "$STAGE_DIR/evidence/src/scripts/topo_eval_suite_v33b_costmap_avoid.sh"
copy_if_exists "$ROOT/scripts/tools/check_capture_motion.py" "$STAGE_DIR/evidence/src/scripts/tools/check_capture_motion.py"
copy_if_exists "$ROOT/scripts/tools/check_capture_motion.sh" "$STAGE_DIR/evidence/src/scripts/tools/check_capture_motion.sh"
copy_if_exists "$ROOT/repo/InternNav/internnav/sim_backend/isaac_backend.py" "$STAGE_DIR/evidence/src/repo/InternNav/internnav/sim_backend/isaac_backend.py"

# Git metadata (outer + submodule)
{
  echo "branch=$(git branch --show-current)"
  echo "head=$(git rev-parse HEAD)"
  git status --short
} > "$STAGE_DIR/evidence/git/outer_status.txt"

git log -n 20 --oneline > "$STAGE_DIR/evidence/git/outer_log.txt"
git diff --stat > "$STAGE_DIR/evidence/git/outer_diff_stat.txt"
git diff > "$STAGE_DIR/evidence/git/outer_diff.patch"

(
  cd "$ROOT/repo/InternNav"
  {
    echo "branch=$(git branch --show-current)"
    echo "head=$(git rev-parse HEAD)"
    git status --short
  } > "$STAGE_DIR/evidence/git/submodule_status.txt"
  git log -n 20 --oneline > "$STAGE_DIR/evidence/git/submodule_log.txt"
  git diff --stat > "$STAGE_DIR/evidence/git/submodule_diff_stat.txt"
  git diff > "$STAGE_DIR/evidence/git/submodule_diff.patch"
)

rm -f "$OUT_ZIP"
(
  cd "$STAGE_DIR"
  zip -rq "$OUT_ZIP" evidence
)

unzip -t "$OUT_ZIP" >/dev/null
SHA="$(sha256sum "$OUT_ZIP" | awk '{print $1}')"
SIZE="$(stat -c %s "$OUT_ZIP")"
set +e
BAD_COUNT="$(zipinfo -1 "$OUT_ZIP" | grep -Ei '\\.(pt|pth|ckpt|bin|safetensors)$' | wc -l)"
set -e

echo "[PROOF_ZIP_OK] zip=$OUT_ZIP sha256=$SHA size_bytes=$SIZE bad_count=$BAD_COUNT"
echo "ZIP=$OUT_ZIP"
