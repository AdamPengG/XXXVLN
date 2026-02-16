#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

CFG="${V34C_RVIZ_CONFIG:-configs/rviz_nav2_v34b.rviz}"
if [ -f /opt/ros/humble/setup.bash ]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi

if ! command -v rviz2 >/dev/null 2>&1; then
  echo "rviz2 not found" >&2
  exit 2
fi

exec rviz2 -d "${CFG}"
