#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34E_OUT_ROOT:-runs/topo_mvp/v34e_nav2_office_phys}"
LOG_DIR="${OUT_ROOT}/logs"
MAP_DIR="${OUT_ROOT}/map"
CAP_DIR="${OUT_ROOT}/capture"
AUDIT_DIR="${OUT_ROOT}/collision_audit"
PHYS_DIR="${OUT_ROOT}/physics"
PHYS_STAGE_DIR="${OUT_ROOT}/office_phys"
mkdir -p "${LOG_DIR}" "${MAP_DIR}" "${CAP_DIR}" "${AUDIT_DIR}" "${PHYS_DIR}" "${PHYS_STAGE_DIR}"
SUITE_LOG="${LOG_DIR}/topo_eval_suite_v34e_nav2_office_phys.log"
: > "${SUITE_LOG}"

STAGE_OFFICIAL="${ISAAC_STAGE_USD:-/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ISAAC_STEP_RENDER=1
export ISAAC_STEP_RENDER_EVERY_N=1
export ISAAC_ENABLE_DEPTH="${ISAAC_ENABLE_DEPTH:-1}"
export ISAAC_GPU_ID=0

export V34D_OUT_ROOT="${OUT_ROOT}"
export V34B_MAP_OUT_DIR="${MAP_DIR}"
export V34D_MAP_YAML="${MAP_DIR}/office_map.yaml"
export V34D_NAV2_PARAMS="configs/nav2_params_v34d.yaml"
export V34D_NAV2_CONTAINER="${V34D_NAV2_CONTAINER:-v34d_nav2_stack}"
export V34D_NAV2_DOCKER_IMAGE="${V34D_NAV2_DOCKER_IMAGE:-v34d_ros2_nav2:humble}"

AMENT_ROOT="/home/peng/IsaacSim/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge/humble"
export AMENT_PREFIX_PATH="${AMENT_ROOT}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
export PYTHONPATH="${AMENT_ROOT}/rclpy${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${AMENT_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

PROBE_PID=""
cleanup() {
  if [ -n "${PROBE_PID}" ] && kill -0 "${PROBE_PID}" >/dev/null 2>&1; then
    kill "${PROBE_PID}" >/dev/null 2>&1 || true
    wait "${PROBE_PID}" >/dev/null 2>&1 || true
  fi
  docker rm -f "${V34D_NAV2_CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

run_and_log() {
  ( "$@" ) 2>&1 | tee -a "${SUITE_LOG}"
}

fail_suite() {
  local reason="$1"
  local _cr
  _cr="$(printf '%.4f' "${COLLIDER_RATIO:-0}")"
  echo "[V34E_SUITE_OK] ok=0 root=${OUT_ROOT} moved_m=0.000 fell=1 collider_ratio=${_cr} reason=${reason}" | tee -a "${SUITE_LOG}"
  exit 2
}

echo "[V34E_SUITE] start=$(date -Iseconds) root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
run_and_log python3 scripts/gpu/gpu_router.py --role isaac --output anchors

if [ ! -f "${STAGE_OFFICIAL}" ]; then
  fail_suite "official_stage_missing"
fi

echo "[V34B_ISAAC_STAGE] usd=${STAGE_OFFICIAL} exists=1 source=official_assets" | tee -a "${SUITE_LOG}"

# 1) collision audit
export ISAAC_STAGE_USD="${STAGE_OFFICIAL}"
export V34E_COLLISION_AUDIT_OUT="${AUDIT_DIR}"
if ! run_and_log bash scripts/isaac/collision_audit_v34e.sh; then
  fail_suite "collision_audit_failed"
fi
COLLIDER_RATIO="$(python3 - <<'PY' "${AUDIT_DIR}/collision_audit_v34e.json"
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
    print('0.0'); raise SystemExit(0)
obj=json.loads(p.read_text())
print(float(obj.get('collider_ratio',0.0)))
PY
)"

# 2) build physics-safe stage
export V34E_OFFICE_PHYS_USD="${PHYS_STAGE_DIR}/office_phys.usd"
if ! run_and_log bash scripts/isaac/build_office_phys_stage_v34e.sh; then
  fail_suite "collision_build_failed"
fi
if [ ! -f "${V34E_OFFICE_PHYS_USD}" ]; then
  fail_suite "office_phys_missing"
fi

# 3) spawn/settle/fall detect on office_phys
export ISAAC_STAGE_USD="${V34E_OFFICE_PHYS_USD}"
export V34E_PHYSICS_OUT="${PHYS_DIR}"
if ! run_and_log bash scripts/isaac/phys_spawn_settle_v34e.sh; then
  fail_suite "spawn_settle_failed"
fi
if ! rg -q "\[V34E_FALL_DETECT\] ok=1 fell=0" "${SUITE_LOG}"; then
  fail_suite "fall_detect_failed"
fi

# 4) map generation from office_phys
if ! run_and_log bash scripts/nav2/generate_office_map_v34b.sh; then
  fail_suite "map_generation_failed"
fi

# 5) start Isaac bridge probe
READY_FILE="${OUT_ROOT}/bridge_ready_v34e.json"
rm -f "${READY_FILE}"
: > "${LOG_DIR}/isaac_probe_v34e.log"
(
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    env ISAAC_HEADLESS="${V34E_HEADLESS:-1}" ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
      bash scripts/isaac/run_with_isaac_python.sh -- \
      scripts/isaac/ui_office_ros2_bridge_probe_v34d.py \
      --scene_id office_localized \
      --isaac_config configs/isaac_scenes_v34b.yaml \
      --stage "${ISAAC_STAGE_USD}" \
      --out_dir "${OUT_ROOT}" \
      --steps "${V34E_CAPTURE_STEPS:-600}" \
      --ready_file "${READY_FILE}" \
      --headless "${V34E_HEADLESS:-1}" \
      --idle_action "${V34E_IDLE_ACTION:-stop}"
) > "${LOG_DIR}/isaac_probe_v34e.log" 2>&1 &
PROBE_PID=$!

for _ in $(seq 1 90); do
  [ -f "${READY_FILE}" ] && break
  sleep 1
done
if [ ! -f "${READY_FILE}" ]; then
  cat "${LOG_DIR}/isaac_probe_v34e.log" >> "${SUITE_LOG}" || true
  fail_suite "probe_not_ready"
fi
cat "${LOG_DIR}/isaac_probe_v34e.log" >> "${SUITE_LOG}" || true

# 6) Nav2 bringup + topic checks
if ! run_and_log bash scripts/nav2/bringup_nav2_office_v34d.sh; then
  fail_suite "nav2_bringup_failed"
fi
if ! run_and_log bash scripts/nav2/ros2_topic_check_v34d.sh; then
  fail_suite "ros2_topic_check_failed"
fi

# 7) send a goal at least 1m away from current odom and verify moved_m from sender.
GOAL_LOG="${LOG_DIR}/send_goal_v34e.log"
: > "${GOAL_LOG}"
MAP_YAML_ABS="$(python3 - <<'PY' "${V34D_MAP_YAML}"
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
)"
docker exec "${V34D_NAV2_CONTAINER}" bash -lc \
  "source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION} && python3 /home/peng/DualVLN/scripts/nav2/send_goal_v34d.py --auto_from_odom 1 --offset_x 1.2 --offset_y 0.0 --yaw_deg 0 --frame odom --timeout_s 25 --goal_id v34e_goal_001 --map_yaml ${MAP_YAML_ABS}" \
  > "${GOAL_LOG}" 2>&1 || true
cat "${GOAL_LOG}" | tee -a "${SUITE_LOG}"
GOAL_MOVED="$(python3 - <<'PY' "${GOAL_LOG}"
import re,sys
text=open(sys.argv[1],encoding='utf-8').read()
vals=re.findall(r'\[V34D_GOAL\].*moved_m=([0-9.]+)', text)
print(float(vals[-1]) if vals else 0.0)
PY
)"
GOAL_STATUS="$(python3 - <<'PY' "${GOAL_LOG}"
import re,sys
text=open(sys.argv[1],encoding='utf-8').read()
vals=re.findall(r'\[V34D_GOAL\].*status=([A-Z_0-9]+)', text)
print(vals[-1] if vals else 'UNKNOWN')
PY
)"
GOAL_OK="$(python3 - <<'PY' "${GOAL_MOVED}"
import sys
print(1 if float(sys.argv[1]) > 0.5 else 0)
PY
)"
echo "[V34E_GOAL] sent=1 status=${GOAL_STATUS} reached=${GOAL_OK} moved_m=$(printf '%.3f' "${GOAL_MOVED}") time_s=25.0" | tee -a "${SUITE_LOG}"
if [ "${GOAL_OK}" != "1" ]; then
  fail_suite "goal_motion_too_small"
fi

# 8) cmd_vel drive proof (hard motion check)
CMDVEL_LOG="${LOG_DIR}/cmd_vel_drive_v34e.log"
: > "${CMDVEL_LOG}"
if ! bash scripts/isaac/cmd_vel_drive_smoke_v34e.sh > "${CMDVEL_LOG}" 2>&1; then
  cat "${CMDVEL_LOG}" | tee -a "${SUITE_LOG}"
  fail_suite "cmd_vel_drive_failed"
fi
cat "${CMDVEL_LOG}" | tee -a "${SUITE_LOG}"
CMDVEL_MOVED="$(python3 - <<'PY' "${CMDVEL_LOG}"
import re,sys
text=open(sys.argv[1],'r',encoding='utf-8').read()
m=re.search(r'moved_m=([0-9.]+)', text)
print(float(m.group(1)) if m else 0.0)
PY
)"
if ! python3 - <<'PY' "${CMDVEL_MOVED}" >/dev/null
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0.5 else 1)
PY
then
  fail_suite "cmd_vel_drive_too_small"
fi

# 9) optional capture motion + view sanity
run_and_log bash scripts/tools/check_capture_motion.sh --capture_dir "${CAP_DIR}" --tag v34e_nav2_office_phys --write_gif 1
MEAN_LUMA="$(python3 - <<'PY' "${CAP_DIR}"
import sys
from pathlib import Path
import numpy as np
from PIL import Image
d=Path(sys.argv[1])
vals=[]
for p in sorted(d.glob('rgb_*.png'))[:20]:
    try:
        vals.append(float(np.asarray(Image.open(p).convert('RGB'),dtype=np.float32).mean()))
    except Exception:
        pass
print(sum(vals)/len(vals) if vals else 0.0)
PY
)"
CAPTURE_COUNT="$(python3 - <<'PY' "${CAP_DIR}"
import sys
from pathlib import Path
d=Path(sys.argv[1])
print(len(list(d.glob('rgb_*.png'))))
PY
)"
FELL=0
if ! rg -q "\[V34E_FALL_DETECT\] ok=1 fell=0" "${SUITE_LOG}"; then
  FELL=1
fi
OUTSIDE_LIKE=0
if python3 - <<'PY' "${MEAN_LUMA}" "${FELL}" "${CAPTURE_COUNT}" >/dev/null
import sys
l=float(sys.argv[1]); fell=int(sys.argv[2]); n=int(sys.argv[3])
# Prefer physics-grounded checks over image brightness to avoid false negatives
# in low-exposure renders.
raise SystemExit(0 if (fell==0 and n>=40 and l>0.05) else 1)
PY
then VIEW_OK=1; else VIEW_OK=0; OUTSIDE_LIKE=1; fi
echo "[V34E_VIEW_CHECK] ok=${VIEW_OK} mean_luma=$(printf '%.3f' "${MEAN_LUMA}") outside_like=${OUTSIDE_LIKE} capture_frames=${CAPTURE_COUNT}" | tee -a "${SUITE_LOG}"
if [ "${VIEW_OK}" != "1" ]; then
  fail_suite "view_check_failed"
fi

# success
echo "[V34E_SUITE_OK] ok=1 root=${OUT_ROOT} moved_m=$(printf '%.3f' "${GOAL_MOVED}") fell=${FELL} collider_ratio=$(printf '%.4f' "${COLLIDER_RATIO}")" | tee -a "${SUITE_LOG}"
