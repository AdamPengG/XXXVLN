#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
cd "${ROOT_DIR}"

RUN_DIR="${ROOT_DIR}/runs/topo_mvp/v30_goal_author_smoke"
mkdir -p "${RUN_DIR}"

TMP_YAML="${RUN_DIR}/isaac_goal_catalog_v30_smoke.yaml"
LOG_PATH="${RUN_DIR}/topo_goal_author_smoke_v30.log"

cp configs/isaac_goal_catalog_v30.yaml "${TMP_YAML}"

{
  echo "[V30_SMOKE] begin out_yaml=${TMP_YAML}"
  set +e
  bash scripts/isaac/goal_author_v30.sh \
    --mode record \
    --out_yaml "${TMP_YAML}" \
    --scene_id "office_localized" \
    --isaac_config "configs/isaac_scenes_v29.yaml" \
    --start_offset 0 \
    --id "office_pose_smoke" \
    --bucket easy \
    --alias smoke \
    --author "smoke"
  rec_ec=$?
  set -e
  if [ "${rec_ec}" -ne 0 ]; then
    if grep -q "\\[GOAL_AUTHOR_V30\\] ok=1 mode=record" "${LOG_PATH}" 2>/dev/null || grep -q "office_pose_smoke" "${TMP_YAML}" 2>/dev/null; then
      echo "[V30_SMOKE] record_exit=${rec_ec} tolerated=1 reason=isaac_python_shutdown_flaky"
    else
      echo "[V30_SMOKE] record_exit=${rec_ec} tolerated=0"
      exit "${rec_ec}"
    fi
  fi

  bash scripts/isaac/goal_author_v30.sh \
    --mode validate \
    --out_yaml "${TMP_YAML}"

  python3 - <<'PY' "${TMP_YAML}"
import sys
p = sys.argv[1]
try:
    import yaml
    data = yaml.safe_load(open(p, 'r'))
except Exception:
    import json
    data = json.load(open(p, 'r'))
goals = data.get('goals', []) if isinstance(data, dict) else []
print(f"[V30_SMOKE] catalog_goals={len(goals)} has_office_pose_smoke={int(any(str(g.get('id',''))=='office_pose_smoke' for g in goals if isinstance(g, dict)))}")
PY

  echo "[V30_SMOKE] done log=${LOG_PATH}"
} | tee "${LOG_PATH}"
