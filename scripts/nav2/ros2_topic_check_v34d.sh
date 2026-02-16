#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34D_OUT_ROOT:-runs/topo_mvp/v34d_nav2_office}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ros2_topic_check_v34d.log"
: > "${LOG_FILE}"

IMG="${V34D_NAV2_DOCKER_IMAGE:-v34d_ros2_nav2:humble}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[V34D_ROS2_TOPIC_CHECK] ok=0 reason=docker_not_found" | tee -a "${LOG_FILE}"
  exit 2
fi
if ! docker image inspect "${IMG}" >/dev/null 2>&1; then
  echo "[V34D_ROS2_TOPIC_CHECK] ok=0 reason=docker_image_missing image=${IMG}" | tee -a "${LOG_FILE}"
  exit 3
fi

TOPICS_FILE="${LOG_DIR}/ros2_topics_list_v34d.txt"
CMD=(
  docker run --rm --network host
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}"
  -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}"
  "${IMG}"
  bash -lc "source /opt/ros/humble/setup.bash && ros2 topic list"
)
"${CMD[@]}" > "${TOPICS_FILE}" 2>> "${LOG_FILE}" || true
cat "${TOPICS_FILE}" >> "${LOG_FILE}" 2>/dev/null || true

tf=0; odom=0; scan=0; cmd=0; cmd_sub=0
if grep -qx '/tf' "${TOPICS_FILE}"; then tf=1; fi
if grep -qx '/odom' "${TOPICS_FILE}"; then odom=1; fi
if grep -qx '/scan' "${TOPICS_FILE}"; then scan=1; fi
if grep -qx '/cmd_vel' "${TOPICS_FILE}"; then cmd=1; fi

if [ "${scan}" = "1" ]; then
  docker run --rm --network host -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" "${IMG}" \
    bash -lc "source /opt/ros/humble/setup.bash && timeout 8 ros2 topic echo /scan --once" >> "${LOG_FILE}" 2>&1 || true
fi
if [ "${odom}" = "1" ]; then
  docker run --rm --network host -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" "${IMG}" \
    bash -lc "source /opt/ros/humble/setup.bash && timeout 8 ros2 topic echo /odom --once" >> "${LOG_FILE}" 2>&1 || true
fi
if [ "${cmd}" = "1" ]; then
  docker run --rm --network host -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" "${IMG}" \
    bash -lc "source /opt/ros/humble/setup.bash && ros2 topic info /cmd_vel" > "${LOG_DIR}/cmd_vel_info_v34d.txt" 2>> "${LOG_FILE}" || true
  if [ -f "${LOG_DIR}/cmd_vel_info_v34d.txt" ] && grep -Eiq 'subscription count: *[1-9]' "${LOG_DIR}/cmd_vel_info_v34d.txt"; then
    cmd_sub=1
  fi
fi

ok=0
if [ "${tf}" = "1" ] && [ "${odom}" = "1" ] && [ "${scan}" = "1" ] && [ "${cmd_sub}" = "1" ]; then
  ok=1
fi

echo "[V34D_ROS2_TOPIC_CHECK] ok=${ok} tf=${tf} odom=${odom} scan=${scan} cmd_vel_sub=${cmd_sub} domain=${ROS_DOMAIN_ID} rmw=${RMW_IMPLEMENTATION}" | tee -a "${LOG_FILE}"
[ "${ok}" = "1" ]
