#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34D_OUT_ROOT:-runs/topo_mvp/v34d_nav2_office}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/nav2_bringup_v34d.log"
: > "${LOG_FILE}"

MAP_YAML="${V34D_MAP_YAML:-${OUT_ROOT}/map/office_map.yaml}"
PARAMS="${V34D_NAV2_PARAMS:-configs/nav2_params_v34d.yaml}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
IMG="${V34D_NAV2_DOCKER_IMAGE:-v34d_ros2_nav2:humble}"
CONTAINER="${V34D_NAV2_CONTAINER:-v34d_nav2_stack}"
READY_TIMEOUT="${V34D_NAV2_READY_TIMEOUT:-45}"

if [ ! -f "${MAP_YAML}" ]; then
  echo "[V34D_NAV2_BRINGUP] ok=0 reason=map_missing map=${MAP_YAML} frames=\"map,odom,base_link\" planner=none controller=none scan=/scan" | tee -a "${LOG_FILE}"
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "[V34D_NAV2_BRINGUP] ok=0 reason=docker_not_found map=${MAP_YAML} frames=\"map,odom,base_link\" planner=none controller=none scan=/scan" | tee -a "${LOG_FILE}"
  exit 3
fi

if ! docker image inspect "${IMG}" >/dev/null 2>&1; then
  TMP_DIR="$(mktemp -d /tmp/v34d_nav2_img.XXXXXX)"
  BASE_IMG="${V34D_NAV2_BASE_IMAGE:-v34c_ros2_nav2:humble}"
  if ! docker image inspect "${BASE_IMG}" >/dev/null 2>&1; then
    BASE_IMG="ubuntu:22.04"
  fi
  cat > "${TMP_DIR}/Dockerfile" <<'DOCKER'
ARG BASE_IMG=ubuntu:22.04
FROM ${BASE_IMG}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg2 ca-certificates lsb-release && \
    (grep -q "packages.ros.org/ros2" /etc/apt/sources.list.d/ros2.list 2>/dev/null || \
      (curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
       echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list)) && \
    apt-get update && \
    for i in 1 2 3; do apt-get install -y --no-install-recommends \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-rmw-fastrtps-cpp \
    ros-humble-rmw-cyclonedds-cpp \
    python3-colcon-common-extensions \
    && break || { echo "retry $i" && apt-get update; }; done && \
    rm -rf /var/lib/apt/lists/*
SHELL ["/bin/bash", "-lc"]
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
DOCKER
  if ! docker build --build-arg BASE_IMG="${BASE_IMG}" -t "${IMG}" "${TMP_DIR}" >> "${LOG_FILE}" 2>&1; then
    rm -rf "${TMP_DIR}"
    echo "[V34D_NAV2_BRINGUP] ok=0 reason=docker_build_failed map=${MAP_YAML} frames=\"map,odom,base_link\" planner=none controller=none scan=/scan" | tee -a "${LOG_FILE}"
    exit 4
  fi
  rm -rf "${TMP_DIR}"
fi

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

CMD="source /opt/ros/humble/setup.bash && ros2 launch nav2_bringup navigation_launch.py slam:=False use_sim_time:=False autostart:=True use_composition:=False map:=${MAP_YAML} params_file:=${PARAMS}"
docker run -d --rm \
  --name "${CONTAINER}" \
  --network host \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
  -v "/home/peng/DualVLN:/home/peng/DualVLN" \
  -w "/home/peng/DualVLN" \
  "${IMG}" \
  bash -lc "${CMD}" >> "${LOG_FILE}" 2>&1

echo "${CONTAINER}" > "${LOG_DIR}/nav2_bringup.container"

ready=0
for _ in $(seq 1 "${READY_TIMEOUT}"); do
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    break
  fi
  if docker exec "${CONTAINER}" bash -lc "source /opt/ros/humble/setup.bash && timeout 2 ros2 action list 2>/dev/null | grep -q '/navigate_to_pose'" >/dev/null 2>&1; then
    if docker exec "${CONTAINER}" bash -lc "source /opt/ros/humble/setup.bash && timeout 2 ros2 lifecycle get /bt_navigator 2>/dev/null | grep -qi 'active'" >/dev/null 2>&1; then
      ready=1
      break
    fi
  fi
  sleep 1
done

if [ "${ready}" = "1" ]; then
  echo "[V34D_NAV2_BRINGUP] ok=1 mode=amcl map=${MAP_YAML} frames=\"map,odom,base_link\" planner=navfn controller=nav2_controller scan=/scan" | tee -a "${LOG_FILE}"
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  docker logs "${CONTAINER}" >> "${LOG_FILE}" 2>&1 || true
fi
echo "[V34D_NAV2_BRINGUP] ok=0 reason=navigate_to_pose_not_ready map=${MAP_YAML} frames=\"map,odom,base_link\" planner=none controller=none scan=/scan" | tee -a "${LOG_FILE}"
exit 5
