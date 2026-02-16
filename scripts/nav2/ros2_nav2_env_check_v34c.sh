#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34C_OUT_ROOT:-runs/topo_mvp/v34c_ros2_bridge}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ros2_nav2_env_check_v34c.log"
: > "${LOG_FILE}"

os_name="$(. /etc/os-release && echo "${ID:-unknown}-${VERSION_ID:-unknown}")"
sys_py="$(python3 -V 2>&1 || true)"
isaac_py="$(bash scripts/isaac/run_with_isaac_python.sh -- -c 'import sys; print(sys.version.split()[0])' 2>&1 | tail -n 1 || true)"

ros2_ok=0
if command -v ros2 >/dev/null 2>&1; then
  ros2_ok=1
fi

rclpy_sys=0
if python3 - <<'PY' >/dev/null 2>&1
import rclpy  # noqa: F401
PY
then
  rclpy_sys=1
fi

rclpy_isaac=0
if [ -f /opt/ros/humble/setup.bash ]; then
  if bash -lc "source /opt/ros/humble/setup.bash >/dev/null 2>&1; bash scripts/isaac/run_with_isaac_python.sh -- -c 'import rclpy; print(\"ISAAC_RCLPY_OK\")'" >/dev/null 2>&1; then
    rclpy_isaac=1
  fi
fi

strategy="docker"
if [ "${ros2_ok}" = "1" ] && [ "${rclpy_sys}" = "1" ]; then
  strategy="host"
fi

ok=0
if [ "${strategy}" = "host" ]; then
  ok=1
elif command -v docker >/dev/null 2>&1; then
  ok=1
fi

{
  echo "os=${os_name}"
  echo "sys_python=${sys_py}"
  echo "isaac_python=${isaac_py}"
  echo "ros2_ok=${ros2_ok}"
  echo "rclpy_sys=${rclpy_sys}"
  echo "rclpy_isaac=${rclpy_isaac}"
  echo "strategy=${strategy}"
  echo "docker_ok=$(command -v docker >/dev/null 2>&1 && echo 1 || echo 0)"
} >> "${LOG_FILE}"

echo "[V34C_ROS2_ENV] ok=${ok} os=${os_name} ros2=${ros2_ok} rclpy_sys=${rclpy_sys} rclpy_isaac=${rclpy_isaac} strategy=${strategy}" | tee -a "${LOG_FILE}"

if [ "${ok}" != "1" ]; then
  exit 2
fi
