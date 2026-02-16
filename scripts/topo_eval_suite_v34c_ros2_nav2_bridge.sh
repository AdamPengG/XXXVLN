#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34C_OUT_ROOT:-runs/topo_mvp/v34c_ros2_bridge}"
LOG_DIR="${OUT_ROOT}/logs"
MAP_DIR="${OUT_ROOT}/map"
CAP_DIR="${OUT_ROOT}/capture"
mkdir -p "${LOG_DIR}" "${MAP_DIR}" "${CAP_DIR}"
SUITE_LOG="${LOG_DIR}/topo_eval_suite_v34c_ros2_nav2_bridge.log"
: > "${SUITE_LOG}"

DOCKER_IMAGE="${V34C_NAV2_DOCKER_IMAGE:-v34c_ros2_nav2:humble}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

cleanup_nav2() {
  if [ -f "${LOG_DIR}/nav2_bringup.pid" ]; then
    PID="$(cat "${LOG_DIR}/nav2_bringup.pid" || true)"
    if [ -n "${PID}" ]; then
      kill "${PID}" >/dev/null 2>&1 || true
    fi
    rm -f "${LOG_DIR}/nav2_bringup.pid"
  fi
  if [ -f "${LOG_DIR}/nav2_bringup.container" ]; then
    CNAME="$(cat "${LOG_DIR}/nav2_bringup.container" || true)"
    if [ -n "${CNAME}" ]; then
      docker rm -f "${CNAME}" >/dev/null 2>&1 || true
    fi
    rm -f "${LOG_DIR}/nav2_bringup.container"
  fi
}
trap cleanup_nav2 EXIT

run_and_log() {
  ( "$@" ) 2>&1 | tee -a "${SUITE_LOG}"
}

echo "[V34C_SUITE] start=$(date -Iseconds) out_root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
run_and_log python3 scripts/gpu/gpu_router.py --role isaac --output anchors

export V34C_OUT_ROOT="${OUT_ROOT}"
export V34B_MAP_OUT_DIR="${MAP_DIR}"
export ISAAC_GPU_ID=0
export ISAAC_STEP_RENDER=1
export ISAAC_STEP_RENDER_EVERY_N=1
export V34C_HEADLESS="${V34C_HEADLESS:-1}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}"

run_and_log bash scripts/nav2/ros2_nav2_env_check_v34c.sh
run_and_log bash scripts/nav2/install_ros2_nav2_v34c.sh
run_and_log bash scripts/isaac/ros2_rclpy_smoke_v34c.sh

# Reuse the proven map generator from v34b.
run_and_log bash scripts/gpu/run_isaac_on_5090.sh -- \
  bash scripts/isaac/run_with_isaac_python.sh -- \
    scripts/nav2/generate_office_map_v34b.py \
      --scene_id office_localized \
      --isaac_config configs/isaac_scenes_v34b.yaml \
      --out_dir "${MAP_DIR}"

export V34C_MAP_YAML="${MAP_DIR}/office_map.yaml"
export V34C_NAV2_PARAMS="configs/nav2_params_v34c.yaml"

# Isaac ROS2 bridge probe + capture evidence.
run_and_log bash scripts/isaac/run_ui_office_ros2_bridge_v34c.sh

# Nav2 bringup (host or docker).
export V34C_ROS2_STRATEGY="${V34C_ROS2_STRATEGY:-auto}"
export V34C_NAV2_DOCKER_IMAGE="${DOCKER_IMAGE}"
run_and_log bash scripts/nav2/bringup_nav2_office_v34c.sh || true

NAV2_OK=0
if rg -q "\[V34C_NAV2_BRINGUP\] ok=1" "${SUITE_LOG}"; then
  NAV2_OK=1
fi

# Send one reachable goal.
GOAL_LOG="${LOG_DIR}/send_goal_v34c.log"
: > "${GOAL_LOG}"
GOAL_X="${V34C_GOAL_X:--2.0}"
GOAL_Y="${V34C_GOAL_Y:-0.0}"
GOAL_YAW="${V34C_GOAL_YAW_DEG:-20}"
GOAL_REACHED=0

if [ "${NAV2_OK}" = "1" ]; then
  STRAT="auto"
  if [ -f "${LOG_DIR}/nav2_strategy.txt" ]; then
    STRAT="$(cut -d= -f2- "${LOG_DIR}/nav2_strategy.txt" || echo auto)"
  fi
  if [ "${STRAT}" = "docker" ] && [ -f "${LOG_DIR}/nav2_bringup.container" ]; then
    CNAME="$(cat "${LOG_DIR}/nav2_bringup.container")"
    if [ -n "${CNAME}" ]; then
      docker exec "${CNAME}" bash -lc \
        "source /opt/ros/humble/setup.bash && python3 /home/peng/DualVLN/scripts/nav2/send_goal_v34c.py --x ${GOAL_X} --y ${GOAL_Y} --yaw_deg ${GOAL_YAW} --frame map --timeout_s 25 --goal_id v34c_goal" \
        > "${GOAL_LOG}" 2>&1 || true
    fi
  else
    if [ -f /opt/ros/humble/setup.bash ]; then
      bash -lc "source /opt/ros/humble/setup.bash && python3 scripts/nav2/send_goal_v34c.py --x ${GOAL_X} --y ${GOAL_Y} --yaw_deg ${GOAL_YAW} --frame map --timeout_s 25 --goal_id v34c_goal" > "${GOAL_LOG}" 2>&1 || true
    else
      python3 scripts/nav2/send_goal_v34c.py --x "${GOAL_X}" --y "${GOAL_Y}" --yaw_deg "${GOAL_YAW}" --frame map --timeout_s 25 --goal_id v34c_goal > "${GOAL_LOG}" 2>&1 || true
    fi
  fi
  cat "${GOAL_LOG}" | tee -a "${SUITE_LOG}"
  if rg -q "\[V34C_GOAL\].*reached=1" "${GOAL_LOG}"; then
    GOAL_REACHED=1
  fi
fi

# Motion check on captured frames.
if [ -d "${CAP_DIR}" ]; then
  run_and_log bash scripts/tools/check_capture_motion.sh --capture_dir "${CAP_DIR}" --tag v34c_bridge_probe --write_gif 1
fi

BRIDGE_OK=0
if rg -q "\[V34C_BRIDGE_PROBE\] ok=1" "${SUITE_LOG}"; then
  BRIDGE_OK=1
fi

ODOM_DELTA="$(python3 - <<'PY' "${OUT_ROOT}/bridge_probe_report.json"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print('0.0')
    raise SystemExit(0)
try:
    obj = json.loads(p.read_text(encoding='utf-8'))
except Exception:
    obj = {}
print(float(obj.get('odom_delta_m', 0.0)))
PY
)"

MOTION_DIFF="$(python3 - <<'PY' "${CAP_DIR}/motion_report.json"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print('0.0')
    raise SystemExit(0)
try:
    obj = json.loads(p.read_text(encoding='utf-8'))
except Exception:
    obj = {}
print(float(obj.get('mean_rgb_diff', 0.0)))
PY
)"

MOVED=0
python3 - <<'PY' "${ODOM_DELTA}" "${MOTION_DIFF}" >/tmp/v34c_moved_flag.txt
import sys
odom = float(sys.argv[1])
md = float(sys.argv[2])
print(1 if (odom >= 0.10 and md > 0.0) else 0)
PY
MOVED="$(cat /tmp/v34c_moved_flag.txt)"

OK=0
if [ "${NAV2_OK}" = "1" ] && [ "${BRIDGE_OK}" = "1" ] && [ "${MOVED}" = "1" ]; then
  OK=1
fi

echo "[V34C_SUITE_OK] ok=${OK} nav2=${NAV2_OK} bridge=${BRIDGE_OK} moved=${MOVED} odom_delta_m=${ODOM_DELTA} captures=${CAP_DIR}" | tee -a "${SUITE_LOG}"

if [ "${OK}" != "1" ]; then
  exit 5
fi
