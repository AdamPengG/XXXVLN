#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34I_OUT_ROOT:-runs/topo_mvp/v34i_nav2_office_succeeded}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ros2_flow_check_v34i.log"
: > "${LOG_FILE}"

IMG="${V34D_NAV2_DOCKER_IMAGE:-v34d_ros2_nav2:humble}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[V34I_ROS2_ODOM_FLOW] ok=0 hz=0.000 reason=docker_not_found" | tee -a "${LOG_FILE}"
  echo "[V34I_ROS2_TF_FLOW] ok=0 dynamic=0 delta_m=0.000 reason=docker_not_found" | tee -a "${LOG_FILE}"
  exit 2
fi
if ! docker image inspect "${IMG}" >/dev/null 2>&1; then
  echo "[V34I_ROS2_ODOM_FLOW] ok=0 hz=0.000 reason=docker_image_missing image=${IMG}" | tee -a "${LOG_FILE}"
  echo "[V34I_ROS2_TF_FLOW] ok=0 dynamic=0 delta_m=0.000 reason=docker_image_missing" | tee -a "${LOG_FILE}"
  exit 3
fi

docker_ros() {
  local cmd="$*"
  docker run --rm --network host \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
    -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
    "${IMG}" \
    bash -lc "source /opt/ros/humble/setup.bash && ${cmd}"
}

ODOM_ONCE="${LOG_DIR}/odom_once_v34i.txt"
if ! docker_ros "timeout 8 ros2 topic echo /odom --once" > "${ODOM_ONCE}" 2>> "${LOG_FILE}"; then
  echo "[V34I_ROS2_ODOM_FLOW] ok=0 hz=0.000 reason=odom_echo_failed" | tee -a "${LOG_FILE}"
  echo "[V34I_ROS2_TF_FLOW] ok=0 dynamic=0 delta_m=0.000 reason=odom_echo_failed" | tee -a "${LOG_FILE}"
  exit 4
fi
cat "${ODOM_ONCE}" >> "${LOG_FILE}" 2>/dev/null || true

ODOM_HZ_RAW="${LOG_DIR}/odom_hz_raw_v34i.txt"
docker_ros "timeout 5 ros2 topic hz /odom" > "${ODOM_HZ_RAW}" 2>> "${LOG_FILE}" || true
cat "${ODOM_HZ_RAW}" >> "${LOG_FILE}" 2>/dev/null || true

ODOM_HZ="$(python3 - <<'PY' "${ODOM_HZ_RAW}"
import re,sys
txt=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
m=re.findall(r'average rate:\s*([0-9.]+)', txt)
print(float(m[-1]) if m else 0.0)
PY
)"

ODOM_FLOW_OK="$(python3 - <<'PY' "${ODOM_HZ}"
import sys
hz=float(sys.argv[1])
print(1 if hz >= 5.0 else 0)
PY
)"
echo "[V34I_ROS2_ODOM_FLOW] ok=${ODOM_FLOW_OK} hz=$(printf '%.3f' "${ODOM_HZ}")" | tee -a "${LOG_FILE}"

TF_ECHO="${LOG_DIR}/tf2_echo_odom_base_link_v34i.txt"
docker_ros "timeout 4 ros2 run tf2_ros tf2_echo odom base_link" > "${TF_ECHO}" 2>> "${LOG_FILE}" || true
cat "${TF_ECHO}" >> "${LOG_FILE}" 2>/dev/null || true
TF_LINES="$(python3 - <<'PY' "${TF_ECHO}"
import re,sys
txt=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
n=len(re.findall(r'At time|Translation|rotation', txt))
print(int(n))
PY
)"

ODOM_START="$(python3 - <<'PY' "${ODOM_ONCE}"
import re,sys
txt=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
mx=re.findall(r'x:\s*([-0-9.eE+]+)', txt)
my=re.findall(r'y:\s*([-0-9.eE+]+)', txt)
if len(mx) >= 1 and len(my) >= 1:
    print(float(mx[0]), float(my[0]))
else:
    print("nan nan")
PY
)"

# Drive a short cmd_vel pulse, then sample odom again for dynamic TF validation.
docker_ros "timeout 2.0 ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.20, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'" >> "${LOG_FILE}" 2>&1 || true
sleep 0.5
ODOM_AFTER="${LOG_DIR}/odom_after_v34i.txt"
docker_ros "timeout 8 ros2 topic echo /odom --once" > "${ODOM_AFTER}" 2>> "${LOG_FILE}" || true

DELTA_M="$(python3 - <<'PY' "${ODOM_START}" "${ODOM_AFTER}"
import math,re,sys
s=sys.argv[1].split()
try:
    x0=float(s[0]); y0=float(s[1])
except Exception:
    print(0.0); raise SystemExit(0)
txt=open(sys.argv[2],encoding='utf-8',errors='ignore').read()
mx=re.findall(r'x:\s*([-0-9.eE+]+)', txt)
my=re.findall(r'y:\s*([-0-9.eE+]+)', txt)
if len(mx) < 1 or len(my) < 1:
    print(0.0); raise SystemExit(0)
x1=float(mx[0]); y1=float(my[0])
print(math.hypot(x1-x0,y1-y0))
PY
)"

TF_FLOW_OK="$(python3 - <<'PY' "${DELTA_M}" "${TF_LINES}"
import sys
d=float(sys.argv[1]); lines=int(sys.argv[2])
print(1 if (lines >= 2 and d > 0.05) else 0)
PY
)"
echo "[V34I_ROS2_TF_FLOW] ok=${TF_FLOW_OK} dynamic=${TF_FLOW_OK} delta_m=$(printf '%.3f' "${DELTA_M}")" | tee -a "${LOG_FILE}"

if [ "${ODOM_FLOW_OK}" != "1" ] || [ "${TF_FLOW_OK}" != "1" ]; then
  exit 5
fi
