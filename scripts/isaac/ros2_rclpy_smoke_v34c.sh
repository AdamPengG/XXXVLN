#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34C_OUT_ROOT:-runs/topo_mvp/v34c_ros2_bridge}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ros2_rclpy_smoke_v34c.log"
: > "${LOG_FILE}"

DOCKER_IMAGE="${V34C_NAV2_DOCKER_IMAGE:-v34c_ros2_nav2:humble}"
sys_ok=0
isaac_ok=0
docker_ok=0
fix_applied=0
method="none"

if [ -f /opt/ros/humble/setup.bash ]; then
  if bash -lc "source /opt/ros/humble/setup.bash >/dev/null 2>&1; python3 -c 'import rclpy; print(\"SYS_RCLPY_OK\")'" >> "${LOG_FILE}" 2>&1; then
    sys_ok=1
  fi

  if bash -lc "source /opt/ros/humble/setup.bash >/dev/null 2>&1; bash scripts/isaac/run_with_isaac_python.sh -- -c 'import rclpy; print(\"ISAAC_RCLPY_OK\")'" >> "${LOG_FILE}" 2>&1; then
    isaac_ok=1
  fi
fi

if command -v docker >/dev/null 2>&1 && docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
  if docker run --rm --network host "${DOCKER_IMAGE}" bash -lc "source /opt/ros/humble/setup.bash && python3 -c 'import rclpy; print(\"DOCKER_RCLPY_OK\")'" >> "${LOG_FILE}" 2>&1; then
    docker_ok=1
  fi
fi

if [ "${isaac_ok}" = "1" ]; then
  method="native_isaac_python"
elif [ "${docker_ok}" = "1" ]; then
  fix_applied=1
  method="docker_ros2_runtime"
else
  method="ros2_python_unavailable"
fi

echo "[V34C_RCLPY] sys_ok=${sys_ok} isaac_ok=${isaac_ok} fix_applied=${fix_applied} method=${method}" | tee -a "${LOG_FILE}"

if [ "${sys_ok}" != "1" ] && [ "${docker_ok}" != "1" ]; then
  exit 2
fi
