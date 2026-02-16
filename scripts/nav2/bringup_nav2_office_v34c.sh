#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34C_OUT_ROOT:-runs/topo_mvp/v34c_ros2_bridge}"
LOG_DIR="${OUT_ROOT}/logs"
MAP_YAML="${V34C_MAP_YAML:-${OUT_ROOT}/map/office_map.yaml}"
PARAMS="${V34C_NAV2_PARAMS:-configs/nav2_params_v34c.yaml}"
SCAN_TOPIC="${V34C_SCAN_TOPIC:-/scan}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
STRATEGY="${V34C_ROS2_STRATEGY:-auto}"
DOCKER_IMAGE="${V34C_NAV2_DOCKER_IMAGE:-v34c_ros2_nav2:humble}"
CONTAINER_NAME="${V34C_NAV2_CONTAINER:-v34c_nav2_stack}"
READY_TIMEOUT="${V34C_NAV2_READY_TIMEOUT:-40}"
DOCKER_NAV2_MODE="${V34C_DOCKER_NAV2_MODE:-mock}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/nav2_bringup_v34c.log"
: > "${LOG_FILE}"

if [ ! -f "${MAP_YAML}" ]; then
  echo "[V34C_NAV2_BRINGUP] ok=0 reason=map_missing map=${MAP_YAML} frames=\"map,odom,base_link\" scan=${SCAN_TOPIC}" | tee -a "${LOG_FILE}"
  exit 2
fi

if [ "${STRATEGY}" = "auto" ]; then
  if command -v ros2 >/dev/null 2>&1 && python3 - <<'PY' >/dev/null 2>&1
import rclpy  # noqa: F401
PY
  then
    STRATEGY="host"
  else
    STRATEGY="docker"
  fi
fi

echo "strategy=${STRATEGY}" > "${LOG_DIR}/nav2_strategy.txt"

action_ready_host() {
  source /opt/ros/humble/setup.bash >/dev/null 2>&1
  ros2 action list 2>/dev/null | grep -q '/navigate_to_pose'
}

action_ready_docker() {
  docker exec "${CONTAINER_NAME}" bash -lc "source /opt/ros/humble/setup.bash >/dev/null 2>&1 && ros2 action list 2>/dev/null | grep -q '/navigate_to_pose'"
}

if [ "${STRATEGY}" = "host" ]; then
  if ! command -v ros2 >/dev/null 2>&1; then
    echo "[V34C_NAV2_BRINGUP] ok=0 reason=ros2_not_found map=${MAP_YAML} frames=\"map,odom,base_link\" scan=${SCAN_TOPIC}" | tee -a "${LOG_FILE}"
    exit 3
  fi
  if [ ! -f /opt/ros/humble/setup.bash ]; then
    echo "[V34C_NAV2_BRINGUP] ok=0 reason=ros_setup_missing map=${MAP_YAML} frames=\"map,odom,base_link\" scan=${SCAN_TOPIC}" | tee -a "${LOG_FILE}"
    exit 3
  fi
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  CMD=(ros2 launch nav2_bringup navigation_launch.py slam:=False use_sim_time:=True map:="${MAP_YAML}" params_file:="${PARAMS}")
  (
    echo "[V34C_NAV2_CMD] ${CMD[*]}"
    "${CMD[@]}"
  ) >> "${LOG_FILE}" 2>&1 &
  PID=$!
  echo "${PID}" > "${LOG_DIR}/nav2_bringup.pid"

  READY=0
  for _ in $(seq 1 "${READY_TIMEOUT}"); do
    if action_ready_host; then
      READY=1
      break
    fi
    sleep 1
  done
  if [ "${READY}" = "1" ]; then
    echo "[V34C_NAV2_BRINGUP] ok=1 map=${MAP_YAML} frames=\"map,odom,base_link\" scan=${SCAN_TOPIC} controller=nav2_controller planner=navfn strategy=host" | tee -a "${LOG_FILE}"
    exit 0
  fi
  echo "[V34C_NAV2_BRINGUP] ok=0 reason=navigate_to_pose_not_ready map=${MAP_YAML} frames=\"map,odom,base_link\" scan=${SCAN_TOPIC} strategy=host" | tee -a "${LOG_FILE}"
  exit 4
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[V34C_NAV2_BRINGUP] ok=0 reason=docker_not_found map=${MAP_YAML} frames=\"map,odom,base_link\" scan=${SCAN_TOPIC} strategy=docker" | tee -a "${LOG_FILE}"
  exit 4
fi
if ! docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
  echo "[V34C_NAV2_BRINGUP] ok=0 reason=docker_image_missing map=${MAP_YAML} frames=\"map,odom,base_link\" scan=${SCAN_TOPIC} strategy=docker image=${DOCKER_IMAGE}" | tee -a "${LOG_FILE}"
  exit 4
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

if [ "${DOCKER_NAV2_MODE}" = "real" ]; then
  DOCKER_CMD="source /opt/ros/humble/setup.bash && ros2 launch nav2_bringup navigation_launch.py slam:=False use_sim_time:=True map:=${MAP_YAML} params_file:=${PARAMS}"
  CONTROLLER="nav2_controller"
  PLANNER="navfn"
else
  DOCKER_CMD="source /opt/ros/humble/setup.bash && python3 /home/peng/DualVLN/scripts/nav2/mock_nav2_server_v34c.py"
  CONTROLLER="mock_nav2"
  PLANNER="mock"
fi


docker run -d --rm \
  --name "${CONTAINER_NAME}" \
  --network host \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
  -v "/home/peng/DualVLN:/home/peng/DualVLN" \
  -w "/home/peng/DualVLN" \
  "${DOCKER_IMAGE}" \
  bash -lc "${DOCKER_CMD}" \
  >> "${LOG_FILE}" 2>&1

echo "${CONTAINER_NAME}" > "${LOG_DIR}/nav2_bringup.container"

READY=0
for _ in $(seq 1 "${READY_TIMEOUT}"); do
  if action_ready_docker; then
    READY=1
    break
  fi
  sleep 1
done

if [ "${READY}" = "1" ]; then
  echo "[V34C_NAV2_BRINGUP] ok=1 map=${MAP_YAML} frames=\"map,odom,base_link\" scan=${SCAN_TOPIC} controller=${CONTROLLER} planner=${PLANNER} strategy=docker mode=${DOCKER_NAV2_MODE}" | tee -a "${LOG_FILE}"
  exit 0
fi

echo "[V34C_NAV2_BRINGUP] ok=0 reason=navigate_to_pose_not_ready map=${MAP_YAML} frames=\"map,odom,base_link\" scan=${SCAN_TOPIC} strategy=docker mode=${DOCKER_NAV2_MODE}" | tee -a "${LOG_FILE}"
exit 5
