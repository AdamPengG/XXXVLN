#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/peng/DualVLN"
cd "${ROOT}"

MAP_DIR="${V34B_MAP_OUT_DIR:-runs/topo_mvp/v34b_nav2_demo/map}"
MAP_YAML="${V34B_MAP_YAML:-${MAP_DIR}/office_map.yaml}"
PARAMS="${V34B_NAV2_PARAMS:-configs/nav2_params_v34b.yaml}"
LOG_DIR="${V34B_LOG_DIR:-runs/topo_mvp/v34b_nav2_demo/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/nav2_bringup_v34b.log"
: > "${LOG_FILE}"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[V34B_NAV2_BRINGUP] ok=0 map=${MAP_YAML} frames=\"map,odom,base_link\" controller=none reason=ros2_not_found" | tee -a "${LOG_FILE}"
  exit 0
fi

if [ ! -f "${MAP_YAML}" ]; then
  echo "[V34B_NAV2_BRINGUP] ok=0 map=${MAP_YAML} frames=\"map,odom,base_link\" controller=none reason=map_missing" | tee -a "${LOG_FILE}"
  exit 0
fi

CMD=(ros2 launch nav2_bringup navigation_launch.py slam:=False use_sim_time:=True map:="${MAP_YAML}" params_file:="${PARAMS}")
(
  echo "[V34B_NAV2_BRINGUP_CMD] ${CMD[*]}"
  "${CMD[@]}"
) >> "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${LOG_DIR}/nav2_bringup.pid"

READY=0
for _ in $(seq 1 20); do
  if ros2 action list 2>/dev/null | grep -q "/navigate_to_pose"; then
    READY=1
    break
  fi
  sleep 1
done

if [ "${READY}" = "1" ]; then
  echo "[V34B_NAV2_BRINGUP] ok=1 map=${MAP_YAML} frames=\"map,odom,base_link\" controller=nav2" | tee -a "${LOG_FILE}"
else
  echo "[V34B_NAV2_BRINGUP] ok=0 map=${MAP_YAML} frames=\"map,odom,base_link\" controller=none reason=navigate_to_pose_not_ready" | tee -a "${LOG_FILE}"
fi

exit 0
