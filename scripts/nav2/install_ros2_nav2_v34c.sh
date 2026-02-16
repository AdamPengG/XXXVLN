#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

OUT_ROOT="${V34C_OUT_ROOT:-runs/topo_mvp/v34c_ros2_bridge}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/install_ros2_nav2_v34c.log"
: > "${LOG_FILE}"

AUTO_INSTALL_HOST="${V34C_AUTO_INSTALL_HOST:-0}"
FORCE_STRATEGY="${V34C_ROS2_STRATEGY:-auto}"
DOCKER_IMAGE="${V34C_NAV2_DOCKER_IMAGE:-v34c_ros2_nav2:humble}"

os_ver="$(. /etc/os-release && echo "${VERSION_ID:-unknown}")"
host_ready=0
if command -v ros2 >/dev/null 2>&1 && python3 - <<'PY' >/dev/null 2>&1
import rclpy  # noqa: F401
PY
then
  host_ready=1
fi

strategy="docker"
if [ "${FORCE_STRATEGY}" = "host" ]; then
  strategy="host"
elif [ "${FORCE_STRATEGY}" = "docker" ]; then
  strategy="docker"
else
  if [ "${host_ready}" = "1" ]; then
    strategy="host"
  else
    strategy="docker"
  fi
fi

if [ "${strategy}" = "host" ]; then
  if [ "${host_ready}" = "1" ]; then
    echo "[V34C_ROS2_INSTALL] ok=1 strategy=host reason=already_available" | tee -a "${LOG_FILE}"
    exit 0
  fi

  if [ "${os_ver}" != "22.04" ] || [ "${AUTO_INSTALL_HOST}" != "1" ]; then
    echo "[V34C_ROS2_INSTALL] ok=0 strategy=host reason=host_ros2_missing hint=set_V34C_ROS2_STRATEGY=docker_or_V34C_AUTO_INSTALL_HOST=1" | tee -a "${LOG_FILE}"
    exit 2
  fi

  {
    sudo apt-get update
    sudo apt-get install -y curl gnupg lsb-release
    if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
      curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
    fi
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
      ros-humble-ros-base \
      ros-humble-nav2-msgs \
      ros-humble-tf-transformations \
      python3-colcon-common-extensions
  } >> "${LOG_FILE}" 2>&1 || {
    echo "[V34C_ROS2_INSTALL] ok=0 strategy=host reason=apt_install_failed hint=use_docker_strategy" | tee -a "${LOG_FILE}"
    exit 3
  }

  echo "[V34C_ROS2_INSTALL] ok=1 strategy=host" | tee -a "${LOG_FILE}"
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[V34C_ROS2_INSTALL] ok=0 strategy=docker reason=docker_not_found" | tee -a "${LOG_FILE}"
  exit 4
fi

if ! docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
  TMP_DIR="$(mktemp -d /tmp/v34c_nav2_img.XXXXXX)"
  cat > "${TMP_DIR}/Dockerfile" <<'DOCKER'
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg2 lsb-release ca-certificates locales && \
    locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-ros-base \
      ros-humble-nav2-msgs \
      ros-humble-tf-transformations \
      python3-colcon-common-extensions && \
    rm -rf /var/lib/apt/lists/*
SHELL ["/bin/bash", "-lc"]
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
DOCKER
  {
    docker build -t "${DOCKER_IMAGE}" "${TMP_DIR}"
  } >> "${LOG_FILE}" 2>&1 || {
    rm -rf "${TMP_DIR}"
    echo "[V34C_ROS2_INSTALL] ok=0 strategy=docker reason=image_build_failed image=${DOCKER_IMAGE}" | tee -a "${LOG_FILE}"
    exit 5
  }
  rm -rf "${TMP_DIR}"
fi

echo "[V34C_ROS2_INSTALL] ok=1 strategy=docker image=${DOCKER_IMAGE}" | tee -a "${LOG_FILE}"
