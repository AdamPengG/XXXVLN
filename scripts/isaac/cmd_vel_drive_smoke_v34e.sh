#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

CONTAINER="${V34D_NAV2_CONTAINER:-v34d_nav2_stack}"
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "[V34E_CMDVEL_DRIVE] ok=0 moved_m=0.000 odom_start=(nan,nan) odom_end=(nan,nan) reason=nav2_container_not_running"
  exit 2
fi

docker exec "${CONTAINER}" bash -lc \
  "source /opt/ros/humble/setup.bash && export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp} ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} && \
   python3 /home/peng/DualVLN/scripts/isaac/cmd_vel_drive_smoke_v34e.py \
    --topic_cmd ${V34E_CMD_TOPIC:-/cmd_vel} \
    --topic_odom ${V34E_ODOM_TOPIC:-/odom} \
    --duration_s ${V34E_CMD_DURATION_S:-3.0} \
    --linear_x ${V34E_CMD_LINEAR_X:-0.35} \
    --angular_z ${V34E_CMD_ANGULAR_Z:-0.0} \
    --min_move_m ${V34E_CMD_MIN_MOVE_M:-0.5} \
    --settle_s ${V34E_CMD_SETTLE_S:-1.5} \
    --tag ${V34E_CMD_TAG:-drive}"
