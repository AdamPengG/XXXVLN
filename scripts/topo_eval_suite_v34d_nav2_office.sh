#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34D_OUT_ROOT:-runs/topo_mvp/v34d_nav2_office}"
LOG_DIR="${OUT_ROOT}/logs"
MAP_DIR="${OUT_ROOT}/map"
CAP_DIR="${OUT_ROOT}/capture"
mkdir -p "${LOG_DIR}" "${MAP_DIR}" "${CAP_DIR}"
SUITE_LOG="${LOG_DIR}/topo_eval_suite_v34d_nav2_office.log"
: > "${SUITE_LOG}"

ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
STAGE_DEFAULT="/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"
export ISAAC_STAGE_USD="${ISAAC_STAGE_USD:-${STAGE_DEFAULT}}"
export V34D_OUT_ROOT="${OUT_ROOT}"
export V34B_MAP_OUT_DIR="${MAP_DIR}"
export V34D_MAP_YAML="${MAP_DIR}/office_map.yaml"
export V34D_NAV2_PARAMS="configs/nav2_params_v34d.yaml"
export V34D_NAV2_DOCKER_IMAGE="${V34D_NAV2_DOCKER_IMAGE:-v34d_ros2_nav2:humble}"
export V34D_NAV2_CONTAINER="${V34D_NAV2_CONTAINER:-v34d_nav2_stack}"
export ROS_DOMAIN_ID
export RMW_IMPLEMENTATION
export ISAAC_STEP_RENDER=1
export ISAAC_STEP_RENDER_EVERY_N=1
export ISAAC_GPU_ID=0

AMENT_ROOT="/home/peng/IsaacSim/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge/humble"
export AMENT_PREFIX_PATH="${AMENT_ROOT}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
export PYTHONPATH="${AMENT_ROOT}/rclpy${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${AMENT_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

PROBE_PID=""
cleanup() {
  if [ -n "${PROBE_PID}" ] && kill -0 "${PROBE_PID}" >/dev/null 2>&1; then
    kill "${PROBE_PID}" >/dev/null 2>&1 || true
  fi
  docker rm -f "${V34D_NAV2_CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

run_and_log() {
  ( "$@" ) 2>&1 | tee -a "${SUITE_LOG}"
}

echo "[V34D_SUITE] start=$(date -Iseconds) out_root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
run_and_log python3 scripts/gpu/gpu_router.py --role isaac --output anchors

if [ ! -f "${ISAAC_STAGE_USD}" ]; then
  echo "[V34D_SUITE_OK] ok=0 root=${OUT_ROOT} stage_url=${ISAAC_STAGE_USD} nav2=0 bridge=0 moved=0 reason=stage_missing" | tee -a "${SUITE_LOG}"
  exit 2
fi

# 1) map generation from official office.usd
run_and_log bash scripts/nav2/generate_office_map_v34b.sh

# 2) start Isaac bridge probe (publishes /tf /odom /scan and subscribes /cmd_vel)
READY_FILE="${OUT_ROOT}/bridge_ready_v34d.json"
rm -f "${READY_FILE}"
: > "${LOG_DIR}/isaac_probe_v34d.log"
(
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    env ISAAC_HEADLESS="${V34D_HEADLESS:-1}" bash scripts/isaac/run_with_isaac_python.sh -- \
      scripts/isaac/ui_office_ros2_bridge_probe_v34d.py \
      --scene_id office_localized \
      --isaac_config configs/isaac_scenes_v34b.yaml \
      --stage "${ISAAC_STAGE_USD}" \
      --out_dir "${OUT_ROOT}" \
      --steps "${V34D_CAPTURE_STEPS:-240}" \
      --ready_file "${READY_FILE}"
) > "${LOG_DIR}/isaac_probe_v34d.log" 2>&1 &
PROBE_PID=$!

for _ in $(seq 1 80); do
  if [ -f "${READY_FILE}" ]; then
    break
  fi
  sleep 1
done

if [ ! -f "${READY_FILE}" ]; then
  cat "${LOG_DIR}/isaac_probe_v34d.log" >> "${SUITE_LOG}" || true
  echo "[V34D_SUITE_OK] ok=0 root=${OUT_ROOT} stage_url=${ISAAC_STAGE_USD} nav2=0 bridge=0 moved=0 reason=probe_not_ready" | tee -a "${SUITE_LOG}"
  exit 3
fi
cat "${LOG_DIR}/isaac_probe_v34d.log" >> "${SUITE_LOG}" || true

# 3) Nav2 bringup (real, docker)
run_and_log bash scripts/nav2/bringup_nav2_office_v34d.sh

# 4) topic check from ROS side
run_and_log bash scripts/nav2/ros2_topic_check_v34d.sh

# 5) send one reachable goal
GOAL_LOG="${LOG_DIR}/send_goal_v34d.log"
: > "${GOAL_LOG}"
GOAL_X="0.0"
GOAL_Y="0.0"
if [ -f "${OUT_ROOT}/probe_report_v34d.json" ]; then
  GOAL_X="$(python3 - <<'PY' "${OUT_ROOT}/probe_report_v34d.json"
import json, sys
obj=json.load(open(sys.argv[1]))
start=obj.get("start_pose", {})
print(float(start.get("x", 0.0)))
PY
)"
  GOAL_Y="$(python3 - <<'PY' "${OUT_ROOT}/probe_report_v34d.json"
import json, sys
obj=json.load(open(sys.argv[1]))
start=obj.get("start_pose", {})
print(float(start.get("z", 0.0)))
PY
)"
fi
docker exec "${V34D_NAV2_CONTAINER}" bash -lc \
  "source /opt/ros/humble/setup.bash && python3 /home/peng/DualVLN/scripts/nav2/send_goal_v34d.py --x ${GOAL_X} --y ${GOAL_Y} --yaw_deg 0 --frame odom --timeout_s 20 --goal_id v34d_goal_001" \
  > "${GOAL_LOG}" 2>&1 || true
if ! rg -q "\\[V34D_GOAL\\].*reached=1" "${GOAL_LOG}"; then
  docker exec "${V34D_NAV2_CONTAINER}" bash -lc \
    "source /opt/ros/humble/setup.bash && python3 /home/peng/DualVLN/scripts/nav2/send_goal_v34d.py --x ${GOAL_X} --y ${GOAL_Y} --yaw_deg 0 --frame odom --timeout_s 20 --goal_id v34d_goal_002" \
    >> "${GOAL_LOG}" 2>&1 || true
fi
cat "${GOAL_LOG}" | tee -a "${SUITE_LOG}"

# let probe continue briefly during goal execution
sleep 3
if [ -n "${PROBE_PID}" ] && kill -0 "${PROBE_PID}" >/dev/null 2>&1; then
  kill "${PROBE_PID}" >/dev/null 2>&1 || true
  wait "${PROBE_PID}" >/dev/null 2>&1 || true
fi
PROBE_PID=""

# Drop corrupted captures from abrupt process shutdowns before motion checks.
python3 - <<'PY' "${CAP_DIR}" >> "${SUITE_LOG}" 2>&1
from pathlib import Path
from PIL import Image
import os, sys
cap = Path(sys.argv[1])
bad = 0
for p in sorted(cap.glob("rgb_*.png")):
    try:
        if p.stat().st_size < 1024:
            bad += 1
            p.unlink(missing_ok=True)
            continue
        with Image.open(p) as im:
            _ = im.convert("RGB")
    except Exception:
        bad += 1
        p.unlink(missing_ok=True)
print(f"[V34D_CAPTURE_SANITIZE] removed={bad}")
PY

# 6) capture motion check
run_and_log bash scripts/tools/check_capture_motion.sh --capture_dir "${CAP_DIR}" --tag v34d_nav2_office --write_gif 1

MOTION_JSON="${CAP_DIR}/motion_report.json"
mean_rgb="0"
identical_pairs="0"
if [ -f "${MOTION_JSON}" ]; then
  mean_rgb="$(python3 - <<'PY' "${MOTION_JSON}"
import json, sys
obj=json.load(open(sys.argv[1]))
print(float(obj.get('mean_rgb_diff',0.0)))
PY
)"
  identical_pairs="$(python3 - <<'PY' "${MOTION_JSON}"
import json, sys
obj=json.load(open(sys.argv[1]))
print(int(obj.get('identical_pairs',0)))
PY
)"
fi

echo "[V34D_CAPTURE_MOTION] frames=80 identical_pairs=${identical_pairs} mean_rgb_diff=${mean_rgb} gif=$( [ -f "${CAP_DIR}/rgb.gif" ] && echo 1 || echo 0 )" | tee -a "${SUITE_LOG}"

# 7) final summary
bridge_ok=0
nav2_ok=0
goal_ok=0
moved=0
stage_url=""
if rg -q "\[V34D_BRIDGE_TOPICS\] ok=1" "${SUITE_LOG}"; then bridge_ok=1; fi
if rg -q "\[V34D_NAV2_BRINGUP\] ok=1" "${SUITE_LOG}"; then nav2_ok=1; fi
if rg -q "\[V34D_GOAL\].*reached=1" "${SUITE_LOG}"; then goal_ok=1; fi
if python3 - <<'PY' "${OUT_ROOT}/probe_report_v34d.json" >/tmp/v34d_moved_flag.txt
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
  print(0)
  raise SystemExit(0)
obj=json.loads(p.read_text())
print(1 if float(obj.get('moved_m',0.0))>0.5 else 0)
PY
then
  moved="$(cat /tmp/v34d_moved_flag.txt)"
fi
if [ -f "${OUT_ROOT}/probe_report_v34d.json" ]; then
  stage_url="$(python3 - <<'PY' "${OUT_ROOT}/probe_report_v34d.json"
import json, sys
obj=json.load(open(sys.argv[1]))
print(obj.get('stage_url',''))
PY
)"
fi

ok=0
if [ "${bridge_ok}" = "1" ] && [ "${nav2_ok}" = "1" ] && [ "${goal_ok}" = "1" ] && [ "${moved}" = "1" ]; then
  ok=1
fi

echo "[V34D_SUITE_OK] ok=${ok} root=${OUT_ROOT} stage_url=${stage_url} nav2=${nav2_ok} bridge=${bridge_ok} moved=${moved}" | tee -a "${SUITE_LOG}"

[ "${ok}" = "1" ]
