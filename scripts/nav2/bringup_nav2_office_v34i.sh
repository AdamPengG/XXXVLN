#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34I_OUT_ROOT:-runs/topo_mvp/v34i_nav2_office_succeeded}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/nav2_bringup_v34i.log"
: > "${LOG_FILE}"

MAP_YAML="${V34D_MAP_YAML:-${OUT_ROOT}/map/office_map.yaml}"
PARAMS="${V34I_NAV2_PARAMS:-${V34D_NAV2_PARAMS:-configs/nav2_params_v34h.yaml}}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
IMG="${V34D_NAV2_DOCKER_IMAGE:-v34d_ros2_nav2:humble}"
NAV2_CONTAINER="${V34D_NAV2_CONTAINER:-v34i_nav2_stack}"
BRIDGE_CONTAINER="${V34I_ODOM_TF_CONTAINER:-v34i_odom_tf_bridge}"
READY_TIMEOUT="${V34D_NAV2_READY_TIMEOUT:-60}"
USE_LOCALIZATION="${V34D_USE_LOCALIZATION:-0}"

if [ ! -f "${MAP_YAML}" ]; then
  echo "[V34I_NAV2_BRINGUP] ok=0 strategy=docker rmw=${RMW_IMPLEMENTATION} domain=${ROS_DOMAIN_ID} reason=map_missing map=${MAP_YAML}" | tee -a "${LOG_FILE}"
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "[V34I_NAV2_BRINGUP] ok=0 strategy=docker rmw=${RMW_IMPLEMENTATION} domain=${ROS_DOMAIN_ID} reason=docker_not_found map=${MAP_YAML}" | tee -a "${LOG_FILE}"
  exit 3
fi
if ! docker image inspect "${IMG}" >/dev/null 2>&1; then
  echo "[V34I_NAV2_BRINGUP] ok=0 strategy=docker rmw=${RMW_IMPLEMENTATION} domain=${ROS_DOMAIN_ID} reason=docker_image_missing image=${IMG}" | tee -a "${LOG_FILE}"
  exit 4
fi

docker rm -f "${NAV2_CONTAINER}" >/dev/null 2>&1 || true
docker rm -f "${BRIDGE_CONTAINER}" >/dev/null 2>&1 || true

if [ "${USE_LOCALIZATION}" = "1" ]; then
  LOC_ARG="use_localization:=True"
else
  LOC_ARG="use_localization:=False"
fi

NAV_CMD="source /opt/ros/humble/setup.bash && ros2 launch nav2_bringup bringup_launch.py slam:=False use_sim_time:=False autostart:=True use_composition:=False ${LOC_ARG} map:=${MAP_YAML} params_file:=${PARAMS}"
docker run -d --rm \
  --name "${NAV2_CONTAINER}" \
  --network host \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
  -v "/home/peng/DualVLN:/home/peng/DualVLN" \
  -w "/home/peng/DualVLN" \
  "${IMG}" \
  bash -lc "${NAV_CMD}" >> "${LOG_FILE}" 2>&1

echo "${NAV2_CONTAINER}" > "${LOG_DIR}/nav2_bringup.container"

ready=0
for _ in $(seq 1 "${READY_TIMEOUT}"); do
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAV2_CONTAINER}$"; then
    break
  fi
  if docker exec "${NAV2_CONTAINER}" bash -lc "source /opt/ros/humble/setup.bash && timeout 2 ros2 action list 2>/dev/null | grep -q '/navigate_to_pose'" >/dev/null 2>&1; then
    if docker exec "${NAV2_CONTAINER}" bash -lc "source /opt/ros/humble/setup.bash && timeout 2 ros2 lifecycle get /bt_navigator 2>/dev/null | grep -qi 'active'" >/dev/null 2>&1; then
      ready=1
      break
    fi
  fi
  sleep 1
done

if [ "${ready}" != "1" ]; then
  docker logs "${NAV2_CONTAINER}" >> "${LOG_FILE}" 2>&1 || true
  echo "[V34I_NAV2_BRINGUP] ok=0 strategy=docker rmw=${RMW_IMPLEMENTATION} domain=${ROS_DOMAIN_ID} reason=nav2_not_ready map=${MAP_YAML}" | tee -a "${LOG_FILE}"
  exit 5
fi

BRIDGE_CMD="source /opt/ros/humble/setup.bash && python3 /home/peng/DualVLN/scripts/nav2/odom_to_tf_broadcaster_v34i.py"
docker run -d --rm \
  --name "${BRIDGE_CONTAINER}" \
  --network host \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
  -v "/home/peng/DualVLN:/home/peng/DualVLN" \
  -w "/home/peng/DualVLN" \
  "${IMG}" \
  bash -lc "${BRIDGE_CMD}" >> "${LOG_FILE}" 2>&1
echo "${BRIDGE_CONTAINER}" > "${LOG_DIR}/odom_tf_bridge.container"

bridge_ready=0
for _ in $(seq 1 15); do
  if ! docker ps --format '{{.Names}}' | grep -q "^${BRIDGE_CONTAINER}$"; then
    break
  fi
  if docker logs "${BRIDGE_CONTAINER}" 2>&1 | grep -q "\[V34I_ODOM_TF_BRIDGE\] ok=1"; then
    bridge_ready=1
    break
  fi
  sleep 1
done

if [ "${bridge_ready}" != "1" ]; then
  docker logs "${BRIDGE_CONTAINER}" >> "${LOG_FILE}" 2>&1 || true
  echo "[V34I_NAV2_BRINGUP] ok=0 strategy=docker rmw=${RMW_IMPLEMENTATION} domain=${ROS_DOMAIN_ID} reason=odom_tf_bridge_not_ready map=${MAP_YAML}" | tee -a "${LOG_FILE}"
  exit 6
fi

echo "[V34I_ODOM_TF_BRIDGE] ok=1 hz=0.000 frames=\"odom->base_link\" dynamic=1" | tee -a "${LOG_FILE}"
echo "[V34I_NAV2_BRINGUP] ok=1 strategy=docker rmw=${RMW_IMPLEMENTATION} domain=${ROS_DOMAIN_ID} map=${MAP_YAML}" | tee -a "${LOG_FILE}"
