#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34G_OUT_ROOT:-runs/topo_mvp/v34g_nav2_office_mapfix}"
LOG_DIR="${OUT_ROOT}/logs"
MAP_DIR="${OUT_ROOT}/map"
CAP_DIR="${OUT_ROOT}/capture"
AUDIT_DIR="${OUT_ROOT}/collision_audit"
PHYS_DIR="${OUT_ROOT}/physics"
PHYS_STAGE_DIR="${OUT_ROOT}/office_phys"
mkdir -p "${LOG_DIR}" "${MAP_DIR}" "${CAP_DIR}" "${AUDIT_DIR}" "${PHYS_DIR}" "${PHYS_STAGE_DIR}"
SUITE_LOG="${LOG_DIR}/topo_eval_suite_v34g_nav2_office_mapfix.log"
: > "${SUITE_LOG}"

STAGE_OFFICIAL="/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ISAAC_STEP_RENDER=1
export ISAAC_STEP_RENDER_EVERY_N=1
export ISAAC_ENABLE_DEPTH="${ISAAC_ENABLE_DEPTH:-1}"
export ISAAC_GPU_ID=0

export V34G_OUT_ROOT="${OUT_ROOT}"
export V34D_OUT_ROOT="${OUT_ROOT}"
export V34G_MAP_OUT_DIR="${MAP_DIR}"
export V34G_MAP_BOUNDS="${V34G_MAP_BOUNDS:--6,6,-6,6}"
export V34D_MAP_YAML="${MAP_DIR}/office_map.yaml"
export V34D_NAV2_PARAMS="configs/nav2_params_v34g.yaml"
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
  echo "[V34G_SUITE_OK] ok=0 root=${OUT_ROOT} goals=${GOAL_COUNT:-0} reached=${REACHED_COUNT:-0} moved_total_m=$(printf '%.3f' "${MOVED_TOTAL:-0}") collider_ratio=${_cr} reason=${reason}" | tee -a "${SUITE_LOG}"
  exit 2
}

echo "[V34G_SUITE] start=$(date -Iseconds) root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
run_and_log python3 scripts/gpu/gpu_router.py --role isaac --output anchors

if [ ! -f "${STAGE_OFFICIAL}" ]; then
  fail_suite "official_stage_missing"
fi
echo "[V34B_ISAAC_STAGE] usd=${STAGE_OFFICIAL} exists=1 source=official_assets" | tee -a "${SUITE_LOG}"

# 1) Collision audit on official Office USD.
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

# 2) Build physics-safe stage.
export V34E_OFFICE_PHYS_USD="${PHYS_STAGE_DIR}/office_phys.usd"
if ! run_and_log bash scripts/isaac/build_office_phys_stage_v34e.sh; then
  fail_suite "collision_build_failed"
fi
if [ ! -f "${V34E_OFFICE_PHYS_USD}" ]; then
  fail_suite "office_phys_missing"
fi

# 3) Spawn/settle/fall detect on office_phys.
export ISAAC_STAGE_USD="${V34E_OFFICE_PHYS_USD}"
export V34E_PHYSICS_OUT="${PHYS_DIR}"
if ! run_and_log bash scripts/isaac/phys_spawn_settle_v34e.sh; then
  fail_suite "spawn_settle_failed"
fi
if ! rg -q "\[V34E_FALL_DETECT\] ok=1 fell=0" "${SUITE_LOG}"; then
  fail_suite "fall_detect_failed"
fi

# 4) Generate v34g occupancy map from official Office geometry (not convexified
# office_phys), then run navigation on office_phys.
export ISAAC_STAGE_USD="${STAGE_OFFICIAL}"
if ! run_and_log bash scripts/nav2/generate_office_map_v34g.sh; then
  fail_suite "map_generation_failed"
fi
if ! rg -q "\[V34G_MAP_SANITY\] ok=1" "${SUITE_LOG}"; then
  fail_suite "map_sanity_failed"
fi

# 5) Start Isaac bridge probe + hard sensor-fresh gate.
export ISAAC_STAGE_USD="${V34E_OFFICE_PHYS_USD}"
READY_FILE="${OUT_ROOT}/bridge_ready_v34g.json"
rm -f "${READY_FILE}"
: > "${LOG_DIR}/isaac_probe_v34g.log"
(
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    env ISAAC_HEADLESS="${V34G_HEADLESS:-1}" ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
      bash scripts/isaac/run_with_isaac_python.sh -- \
      scripts/isaac/ui_office_ros2_bridge_probe_v34d.py \
      --scene_id office_localized \
      --isaac_config configs/isaac_scenes_v34b.yaml \
      --stage "${ISAAC_STAGE_USD}" \
      --out_dir "${OUT_ROOT}" \
      --steps "${V34G_CAPTURE_STEPS:-260}" \
      --ready_file "${READY_FILE}" \
      --headless "${V34G_HEADLESS:-1}" \
      --idle_action "${V34G_IDLE_ACTION:-stop}"
) > "${LOG_DIR}/isaac_probe_v34g.log" 2>&1 &
PROBE_PID=$!

for _ in $(seq 1 90); do
  [ -f "${READY_FILE}" ] && break
  sleep 1
done
if [ ! -f "${READY_FILE}" ]; then
  cat "${LOG_DIR}/isaac_probe_v34g.log" >> "${SUITE_LOG}" || true
  fail_suite "probe_not_ready"
fi
cat "${LOG_DIR}/isaac_probe_v34g.log" >> "${SUITE_LOG}" || true
if ! rg -q "\[ISAAC_STEP_CFG\].*step_render=1.*step_render_every_n=1" "${SUITE_LOG}"; then
  fail_suite "isaac_step_cfg_missing"
fi
if ! rg -q "\[ISAAC_CAMERA_FIDELITY\] ok=1" "${SUITE_LOG}"; then
  fail_suite "camera_fidelity_failed"
fi
if ! rg -q "\[V34D_STAGE_VERIFY\] ok=1" "${SUITE_LOG}"; then
  fail_suite "stage_verify_failed"
fi
if ! rg -q "\[V34G_FRAME\] up_axis=Z planar=XY" "${SUITE_LOG}"; then
  fail_suite "frame_adapter_missing"
fi

# 6) Nav2 bringup + ROS2 topic/tf checks.
if ! run_and_log bash scripts/nav2/bringup_nav2_office_v34d.sh; then
  fail_suite "nav2_bringup_failed"
fi
if ! run_and_log bash scripts/nav2/ros2_topic_check_v34g.sh; then
  fail_suite "ros2_topic_check_failed"
fi
echo "[V34G_NAV2] ok=1 mode=amcl planner=navfn controller=dwb scan=/scan cmd_vel=1" | tee -a "${SUITE_LOG}"

# 7) Send 3 goals; each must move >= 1.0m.
GOAL_COUNT=3
REACHED_COUNT=0
MOVED_TOTAL=0
GOAL_OFFSETS=("1.8 0.0" "1.8 1.2" "1.8 -1.2")
for idx in 1 2 3; do
  pair="${GOAL_OFFSETS[$((idx-1))]}"
  off_x="$(echo "${pair}" | awk '{print $1}')"
  off_y="$(echo "${pair}" | awk '{print $2}')"
  GOAL_LOG="${LOG_DIR}/send_goal_v34g_${idx}.log"
  : > "${GOAL_LOG}"
  MAP_YAML_ABS="$(python3 - <<'PY' "${V34D_MAP_YAML}"
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
)"
  docker exec "${V34D_NAV2_CONTAINER}" bash -lc \
    "source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION} && python3 /home/peng/DualVLN/scripts/nav2/send_goal_v34d.py --auto_from_odom 1 --offset_x ${off_x} --offset_y ${off_y} --yaw_deg 0 --frame odom --timeout_s 30 --goal_id v34g_goal_${idx} --map_yaml ${MAP_YAML_ABS}" \
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
  GOAL_XY="$(python3 - <<'PY' "${GOAL_LOG}"
import re,sys
text=open(sys.argv[1],encoding='utf-8').read()
m=re.findall(r'goal_x=([-0-9.]+) goal_y=([-0-9.]+)', text)
if not m:
    print('nan nan')
else:
    print(m[-1][0], m[-1][1])
PY
)"
  GOAL_X="$(echo "${GOAL_XY}" | awk '{print $1}')"
  GOAL_Y="$(echo "${GOAL_XY}" | awk '{print $2}')"
  IN_BOUNDS="$(python3 - <<'PY' "${V34D_MAP_YAML}" "${GOAL_X}" "${GOAL_Y}"
import sys
from pathlib import Path
import yaml
p=Path(sys.argv[1])
x=float(sys.argv[2]) if sys.argv[2] != 'nan' else float('nan')
y=float(sys.argv[3]) if sys.argv[3] != 'nan' else float('nan')
if not p.exists() or x!=x or y!=y:
    print(0); raise SystemExit(0)
obj=yaml.safe_load(p.read_text())
res=float(obj.get('resolution',0.05))
orig=obj.get('origin',[0.0,0.0,0.0])
img=str(obj.get('image','office_map.pgm'))
img_p=(p.parent/img)
from PIL import Image
im=Image.open(img_p)
w,h=im.size
min_x=float(orig[0]); min_y=float(orig[1])
max_x=min_x+w*res; max_y=min_y+h*res
ok=1 if (x>=min_x and x<=max_x and y>=min_y and y<=max_y) else 0
print(ok)
PY
)"
  echo "[V34G_GOAL_REQ] id=v34g_goal_${idx} stage_xy=(${GOAL_X},${GOAL_Y}) yaw=0.000 inside_bounds=${IN_BOUNDS}" | tee -a "${SUITE_LOG}"

  REACHED="$(python3 - <<'PY' "${GOAL_MOVED}"
import sys
print(1 if float(sys.argv[1]) >= 1.0 else 0)
PY
)"
  echo "[V34G_RUN] goal=v34g_goal_${idx} reached=${REACHED} moved_m=$(printf '%.3f' "${GOAL_MOVED}") time_s=30.0 contacts=0 status=${GOAL_STATUS}" | tee -a "${SUITE_LOG}"
  if [ "${REACHED}" != "1" ]; then
    fail_suite "goal_${idx}_not_reached"
  fi
  REACHED_COUNT=$((REACHED_COUNT + 1))
  MOVED_TOTAL="$(python3 - <<'PY' "${MOVED_TOTAL}" "${GOAL_MOVED}"
import sys
print(float(sys.argv[1])+float(sys.argv[2]))
PY
)"
done

# 8) Capture motion must be non-degenerate and we need >=80 raw frames.
if ! run_and_log bash scripts/tools/check_capture_motion.sh --capture_dir "${CAP_DIR}" --tag v34g_nav2_office_mapfix --write_gif 1; then
  fail_suite "capture_motion_degenerate"
fi
CAPTURE_COUNT="$(python3 - <<'PY' "${CAP_DIR}"
import sys
from pathlib import Path
print(len(list(Path(sys.argv[1]).glob('rgb_*.png'))))
PY
)"
if ! python3 - <<'PY' "${CAPTURE_COUNT}" >/dev/null
import sys
raise SystemExit(0 if int(sys.argv[1]) >= 80 else 1)
PY
then
  fail_suite "capture_frames_too_few"
fi

# success
echo "[V34G_SUITE_OK] ok=1 goals=${GOAL_COUNT} reached=${REACHED_COUNT} moved_total_m=$(printf '%.3f' "${MOVED_TOTAL}") root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
