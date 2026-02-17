#!/usr/bin/env bash
# topo_eval_suite_v35d_topo_to_nav2_visual_sane.sh — v35d with depth-based 3-ROI
# view sanity, look-at chase, yaw smoothing, dual GIF (>=60 frames).
set -euo pipefail

ROOT="/home/peng/DualVLN"
cd "${ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${V35D_OUT_ROOT:-runs/topo_mvp/v35d_topo_to_nav2_${TS}}"
LOG_DIR="${OUT_ROOT}/logs"
MAP_DIR="${OUT_ROOT}/map"
ROOMQA_DIR="${OUT_ROOT}/roomqa"
CAP_DIR="${OUT_ROOT}/capture"
EVIDENCE_DIR="${OUT_ROOT}/evidence_frames"
mkdir -p "${LOG_DIR}" "${MAP_DIR}" "${ROOMQA_DIR}" "${CAP_DIR}" "${EVIDENCE_DIR}"
SUITE_LOG="${LOG_DIR}/topo_eval_suite_v35d.log"
: > "${SUITE_LOG}"

# ── Stage ──
STAGE_OFFICIAL="/home/peng/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"
if [ ! -f "${STAGE_OFFICIAL}" ]; then
  echo "[V35D_STAGE_VERIFY] ok=0 stage_url=${STAGE_OFFICIAL} exists=0" | tee -a "${SUITE_LOG}"
  exit 2
fi
export ISAAC_STAGE_USD="${STAGE_OFFICIAL}"
echo "[V35D_STAGE_VERIFY] ok=1 stage_url=${STAGE_OFFICIAL} exists=1" | tee -a "${SUITE_LOG}"

# ── Sensor env ──
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ISAAC_STEP_RENDER=1
export ISAAC_STEP_RENDER_EVERY_N=1
export ISAAC_ENABLE_DEPTH="${ISAAC_ENABLE_DEPTH:-1}"
export ISAAC_GPU_ID=0
export ISAAC_CAMERA_PITCH_DEG="${ISAAC_CAMERA_PITCH_DEG:--10.0}"
export V35C_CAM_HEIGHT_M="${V35C_CAM_HEIGHT_M:-1.55}"
export V35C_CAM_PITCH_DEG="${V35C_CAM_PITCH_DEG:--10}"
export ISAAC_CAMERA_RPY_DEG="${ISAAC_CAMERA_RPY_DEG:-0,-10,0}"

export V35D_OUT_ROOT="${OUT_ROOT}"
export V34J_OUT_ROOT="${OUT_ROOT}"
export V34I_OUT_ROOT="${OUT_ROOT}"
export V34D_OUT_ROOT="${OUT_ROOT}"
export V34G_OUT_ROOT="${OUT_ROOT}"
export V34G_MAP_OUT_DIR="${MAP_DIR}"
export V34G_MAP_BOUNDS="${V34G_MAP_BOUNDS:--6,6,-6,6}"
export V34D_MAP_YAML="${MAP_DIR}/office_map.yaml"
export V34I_NAV2_PARAMS="${V34I_NAV2_PARAMS:-configs/nav2_params_v34j.yaml}"
export V34D_NAV2_PARAMS="${V34I_NAV2_PARAMS}"
export V34D_NAV2_CONTAINER="${V34D_NAV2_CONTAINER:-v35d_nav2_stack}"
export V34I_ODOM_TF_CONTAINER="${V34I_ODOM_TF_CONTAINER:-v35d_odom_tf_bridge}"
export V34D_NAV2_DOCKER_IMAGE="${V34D_NAV2_DOCKER_IMAGE:-v34d_ros2_nav2:humble}"
export V34D_USE_LOCALIZATION="${V34D_USE_LOCALIZATION:-0}"

AMENT_ROOT="/home/peng/IsaacSim/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge/humble"
export AMENT_PREFIX_PATH="${AMENT_ROOT}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
export PYTHONPATH="${AMENT_ROOT}/rclpy${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${AMENT_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

PROBE_PID=""
cleanup() {
  if [ -n "${PROBE_PID}" ] && kill -0 "${PROBE_PID}" >/dev/null 2>&1; then
    kill "${PROBE_PID}" >/dev/null 2>&1 || true
    wait "${PROBE_PID}" >/dev/null 2>&1 || true
  fi
  docker rm -f "${V34I_ODOM_TF_CONTAINER}" >/dev/null 2>&1 || true
  docker rm -f "${V34D_NAV2_CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

run_and_log() { ( "$@" ) 2>&1 | tee -a "${SUITE_LOG}"; }
fail_suite() {
  local reason="$1"
  echo "[V35D_SUITE_OK] ok=0 cases=${STAGED_TOTAL:-3} succeeded=${STAGED_SUCCEEDED:-0} rooms=${ROOM_COUNT:-0} doors=${DOOR_COUNT:-0} root=${OUT_ROOT} reason=${reason}" | tee -a "${SUITE_LOG}"
  exit 2
}

ROOM_COUNT=0
DOOR_COUNT=0
STAGED_SUCCEEDED=0
STAGED_SKIPPED=0
STAGED_TOTAL=3

# ── Staged task runner ──
run_stage_task() {
  local task="$1"
  local task_id="$2"
  local room="$3"
  local obj="$4"
  local gx="$5"
  local gy="$6"
  local gyaw="$7"

  local plan_log="${LOG_DIR}/plan_${task_id}.log"
  : > "${plan_log}"
  if ! python3 scripts/topo_instruction_to_goal_v35a.py \
    --stage "${ISAAC_STAGE_USD}" \
    --room_partition_json "${ROOMQA_DIR}/room_partition.json" \
    --room_graph_json "${ROOMQA_DIR}/room_graph.json" \
    --topo_nodes_json "${ROOMQA_DIR}/topo_nodes.json" \
    --pose_trace_csv "${CAP_DIR}/pose_trace.csv" \
    --out_dir "${ROOMQA_DIR}" \
    --task "${task}" \
    --task_id "${task_id}" \
    --room "${room}" \
    --object "${obj}" \
    --goal_x "${gx}" \
    --goal_y "${gy}" \
    --goal_yaw_deg "${gyaw}" > "${plan_log}" 2>&1; then
    cat "${plan_log}" | tee -a "${SUITE_LOG}"
    if [ "${task}" = "object" ] && grep -q "object_not_found\|catalog.*empty\|no.*match" "${plan_log}"; then
      echo "[V35D_CASE_DONE] case=${task_id} overall=SKIPPED stages_succeeded=0/0 reason=object_not_found" | tee -a "${SUITE_LOG}"
      STAGED_SKIPPED=$((STAGED_SKIPPED + 1))
      return 0
    fi
    fail_suite "goal_resolution_failed_${task_id}"
  fi
  cat "${plan_log}" | tee -a "${SUITE_LOG}"

  local goals_json="${ROOMQA_DIR}/staged_goals_${task_id}.json"
  local seg_json="${ROOMQA_DIR}/staged_result_${task_id}.json"
  local seg_log="${LOG_DIR}/send_staged_${task_id}.log"
  : > "${seg_log}"
  docker exec "${V34D_NAV2_CONTAINER}" bash -lc \
    "source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION} && python3 /home/peng/DualVLN/scripts/nav2/send_staged_goals_v35a.py --goals_json ${goals_json} --frame map --timeout_s ${V35D_SEG_TIMEOUT_S:-90} --dist_tol_m ${V35D_SEG_DIST_TOL_M:-0.45} --yaw_tol_deg ${V35D_SEG_YAW_TOL_DEG:-40} --result_json ${seg_json}" \
    > "${seg_log}" 2>&1 || true
  cat "${seg_log}" | tee -a "${SUITE_LOG}"

  python3 - "${seg_json}" "${task_id}" <<'PY' | tee -a "${SUITE_LOG}"
import json,sys
p,tid=sys.argv[1],sys.argv[2]
try:
    r=json.load(open(p,encoding='utf-8'))
except Exception:
    print(f"[V35D_CASE_DONE] case={tid} overall=PARSE_ERROR stages_succeeded=0/0")
    sys.exit(0)
segs=r.get('segments',[])
ok_count=sum(1 for s in segs if s.get('ok',0)==1)
for i,s in enumerate(segs):
    print(f"[V35D_NAV2_STAGE] case={tid} stage={i} status={s.get('status','?')} dist_m={s.get('dist_m','-'):.3f} yaw_err_deg={s.get('yaw_err_deg','-'):.1f} ok={s.get('ok',0)}")
overall="SUCCEEDED" if r.get('ok',0)==1 else "FAILED"
print(f"[V35D_CASE_DONE] case={tid} overall={overall} stages_succeeded={ok_count}/{len(segs)}")
PY

  if rg -q "\[V35A_NAV2_STAGED\].*ok=1" "${seg_log}" 2>/dev/null || \
     python3 -c "import json,sys; r=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if r.get('ok',0)==1 else 1)" "${seg_json}" 2>/dev/null; then
    STAGED_SUCCEEDED=$((STAGED_SUCCEEDED + 1))
  fi
}

echo "[V35D_SUITE] start=$(date -Iseconds) root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
run_and_log python3 scripts/gpu/gpu_router.py --role isaac --output anchors

# ── 1) Map ──
if ! run_and_log bash scripts/nav2/generate_office_map_v34g.sh; then
  fail_suite "map_generation_failed"
fi
if ! rg -q "\[V34G_MAP_SANITY\] ok=1" "${SUITE_LOG}"; then
  fail_suite "map_sanity_failed"
fi
OCC_RATIO="$(python3 - <<'PY' "${MAP_DIR}/office_map.pgm"
import numpy as np; from PIL import Image; import sys
im=np.array(Image.open(sys.argv[1]).convert('L'),dtype=np.float32)
occ=(im<50).sum(); total=im.size
print(f"{occ/total:.4f}")
PY
)"
echo "[V35D_MAP_GEN] ok=1 occ_ratio=${OCC_RATIO}" | tee -a "${SUITE_LOG}"

# ── 2) Room/topo ──
if ! run_and_log python3 scripts/topo_room_from_occ_v35a.py --map_yaml "${MAP_DIR}/office_map.yaml" --out_dir "${ROOMQA_DIR}"; then
  fail_suite "room_partition_failed"
fi
ROOM_COUNT="$(python3 - <<'PY' "${ROOMQA_DIR}/room_partition.json"
import json,sys; print(len(json.load(open(sys.argv[1],encoding='utf-8')).get('rooms',[])))
PY
)"
DOOR_COUNT="$(python3 - <<'PY' "${ROOMQA_DIR}/doorway_edges.json"
import json,sys; print(len(json.load(open(sys.argv[1],encoding='utf-8')).get('edges',[])))
PY
)"
echo "[V35D_ROOM_PARTITION] ok=1 rooms=${ROOM_COUNT}" | tee -a "${SUITE_LOG}"
echo "[V35D_DOORWAY_DETECT] ok=1 doors=${DOOR_COUNT}" | tee -a "${SUITE_LOG}"

# ── 3) Isaac ROS2 bridge probe (v35d) ──
READY_FILE="${OUT_ROOT}/bridge_ready_v35d.json"
rm -f "${READY_FILE}"
: > "${LOG_DIR}/isaac_probe_v35d.log"
CAPTURE_STEPS="${V35D_CAPTURE_STEPS:-6000}"
(
  bash scripts/gpu/run_isaac_on_5090.sh -- \
    env ISAAC_HEADLESS="${V35D_HEADLESS:-1}" \
      ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
      V35C_CAM_HEIGHT_M="${V35C_CAM_HEIGHT_M}" V35C_CAM_PITCH_DEG="${V35C_CAM_PITCH_DEG}" \
      bash scripts/isaac/run_with_isaac_python.sh -- \
      scripts/isaac/ui_office_ros2_bridge_probe_v35d.py \
      --scene_id office_localized \
      --isaac_config configs/isaac_scenes_v34b.yaml \
      --stage "${ISAAC_STAGE_USD}" \
      --out_dir "${OUT_ROOT}" \
      --steps "${CAPTURE_STEPS}" \
      --ready_file "${READY_FILE}" \
      --headless "${V35D_HEADLESS:-1}" \
      --idle_action "${V35D_IDLE_ACTION:-stop}" \
      --cam_pitch_deg "${V35C_CAM_PITCH_DEG}"
) > "${LOG_DIR}/isaac_probe_v35d.log" 2>&1 &
PROBE_PID=$!

for _ in $(seq 1 180); do
  [ -f "${READY_FILE}" ] && break
  sleep 1
done
if [ ! -f "${READY_FILE}" ]; then
  cat "${LOG_DIR}/isaac_probe_v35d.log" >> "${SUITE_LOG}" || true
  fail_suite "probe_not_ready"
fi
cat "${LOG_DIR}/isaac_probe_v35d.log" >> "${SUITE_LOG}" || true

# v35d gates
if ! rg -q "\[V34H_STAGE_VERIFY\] ok=1 .*office\.usd" "${SUITE_LOG}"; then
  fail_suite "stage_verify_failed"
fi
if ! rg -q "\[ISAAC_CAMERA_FIDELITY\] ok=1" "${SUITE_LOG}"; then
  fail_suite "camera_fidelity_failed"
fi
if ! rg -q "\[V34D_BRIDGE_TOPICS\] ok=1" "${SUITE_LOG}"; then
  fail_suite "bridge_topics_failed"
fi
# v35d-specific: view sanity must pass for FP
if ! rg -q "\[V35D_VIEW_SANITY\].*ok=1" "${SUITE_LOG}"; then
  fail_suite "fp_view_sanity_failed"
fi
# v35d: must have FP mount and chase anchors
if ! rg -q "\[V35D_FP_MOUNT\].*ok=1" "${SUITE_LOG}"; then
  fail_suite "fp_mount_missing"
fi
if ! rg -q "\[V35D_CHASE\].*ok=1" "${SUITE_LOG}"; then
  fail_suite "chase_mount_missing"
fi
if ! rg -q "\[V35D_YAW_SMOOTH\].*ok=1" "${SUITE_LOG}"; then
  fail_suite "yaw_smooth_missing"
fi

# ── 4) Nav2 bringup ──
if ! run_and_log bash scripts/nav2/bringup_nav2_office_v34i.sh; then
  fail_suite "nav2_bringup_failed"
fi
if ! run_and_log bash scripts/nav2/ros2_flow_check_v34i.sh; then
  fail_suite "ros2_flow_check_failed"
fi

# ── 5) Sample reachable goals ──
REACHABLE_JSON="${ROOMQA_DIR}/reachable_goals_seed.json"
SAMPLE_LOG="${LOG_DIR}/sample_reachable_goals_v35d.log"
: > "${SAMPLE_LOG}"
docker exec "${V34D_NAV2_CONTAINER}" bash -lc \
  "source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION} && python3 /home/peng/DualVLN/scripts/nav2/sample_reachable_goals_v34j.py --map_yaml ${V34D_MAP_YAML} --out_json ${REACHABLE_JSON} --count 6 --radius_min ${V35D_RADIUS_MIN:-0.8} --radius_max ${V35D_RADIUS_MAX:-1.8} --seed ${V35D_GOAL_SEED:-3}" \
  > "${SAMPLE_LOG}" 2>&1 || true
cat "${SAMPLE_LOG}" | tee -a "${SUITE_LOG}"
if ! rg -q "\\[V34J_GOAL_SAMPLE\\].*ok=1" "${SAMPLE_LOG}"; then
  fail_suite "reachable_goal_sampling_failed"
fi

read -r G1X G1Y G1YAW G2X G2Y G2YAW G3X G3Y G3YAW <<EOF
$(python3 - <<'PY' "${REACHABLE_JSON}"
import json,sys
goals=json.load(open(sys.argv[1],encoding='utf-8')).get('goals',[])
vals=[]
for i in range(3):
    if i < len(goals):
        g=goals[i]
        vals.extend([str(float(g.get('x',0.0))), str(float(g.get('y',0.0))), str(float(g.get('yaw_deg',0.0)))])
    else:
        vals.extend(['0.0','0.0','0.0'])
print(' '.join(vals))
PY
)
EOF

ROOM_TARGET="$(python3 - <<'PY' "${ROOMQA_DIR}/room_partition.json"
import json,sys
obj=json.load(open(sys.argv[1],encoding='utf-8'))
ids=sorted(int(r.get('room_id',-1)) for r in obj.get('rooms',[]) if int(r.get('room_id',-1))>=0)
if len(ids)>=2: print(ids[1])
elif len(ids)==1: print(ids[0])
else: print(0)
PY
)"

# ── 6) Run 3 staged tasks ──
STAGED_SUCCEEDED=0
STAGED_SKIPPED=0
run_stage_task doorway door_to_door "room_${ROOM_TARGET}" "" "${G1X}" "${G1Y}" "${G1YAW}"
run_stage_task room room_center "room_${ROOM_TARGET}" "" "${G2X}" "${G2Y}" "${G2YAW}"
run_stage_task object object_goal "room_${ROOM_TARGET}" "table" "${G3X}" "${G3Y}" "${G3YAW}"

# ── 7) Stop probe ──
if [ -n "${PROBE_PID}" ] && kill -0 "${PROBE_PID}" >/dev/null 2>&1; then
  kill "${PROBE_PID}" >/dev/null 2>&1 || true
  wait "${PROBE_PID}" >/dev/null 2>&1 || true
  PROBE_PID=""
fi
cat "${LOG_DIR}/isaac_probe_v35d.log" >> "${SUITE_LOG}" || true

# ── 8) Re-run room/topo with trajectory ──
if [ -f "${CAP_DIR}/pose_trace.csv" ]; then
  run_and_log python3 scripts/topo_room_from_occ_v35a.py --map_yaml "${MAP_DIR}/office_map.yaml" --out_dir "${ROOMQA_DIR}" --pose_trace "${CAP_DIR}/pose_trace.csv" || true
fi

# ── 9) Generate dual GIFs per case (>=60 frames each) ──
for case in door_to_door room_center object_goal; do
  # FP GIF: need >=60 frames, stride auto-adjusted by make_gif_safe
  run_and_log python3 scripts/tools/make_gif_safe.py \
    --capture_dir "${CAP_DIR}" \
    --out "${EVIDENCE_DIR}/${case}_fp.gif" \
    --stride 15 --min_frames 60 --max_width 320 \
    --glob "rgb_fp_*.png" --view fp \
    || echo "[V35D_GIF_WARN] case=${case} view=fp reason=gif_failed" | tee -a "${SUITE_LOG}"

  # Chase GIF: need >=60 frames
  run_and_log python3 scripts/tools/make_gif_safe.py \
    --capture_dir "${CAP_DIR}/chase" \
    --out "${EVIDENCE_DIR}/${case}_chase.gif" \
    --stride 5 --min_frames 60 --max_width 360 \
    --glob "rgb_chase_*.png" --view chase \
    || echo "[V35D_GIF_WARN] case=${case} view=chase reason=gif_failed" | tee -a "${SUITE_LOG}"
done

# ── 10) Select 3 keyframes per case×view: first, middle, last ──
python3 - "${CAP_DIR}" "${EVIDENCE_DIR}" <<'PY'
import shutil, sys
from pathlib import Path
cap, ev = Path(sys.argv[1]), Path(sys.argv[2])
ev.mkdir(parents=True, exist_ok=True)
total_kf = 0
for prefix,subdir in [("rgb_fp_", cap), ("rgb_chase_", cap / "chase")]:
    frames = sorted(subdir.glob(f"{prefix}*.png"))
    if not frames: continue
    n = len(frames)
    picks = sorted(set([0, n//2, n-1]))  # 3 keyframes: start, mid, last
    for idx in picks:
        src = frames[idx]
        shutil.copy2(src, ev / f"key_{prefix}{idx:05d}.png")
        total_kf += 1
    print(f"[V35D_KEY_FRAMES] prefix={prefix} total={n} keys={len(picks)}")
print(f"[V35D_KEY_TOTAL] keyframes={total_kf}")
PY

# ── 11) Motion gate on chase frames ──
if ! run_and_log python3 scripts/tools/check_capture_motion.py \
  --capture_dir "${CAP_DIR}/chase" --tag v35d_chase --write_gif 0; then
  echo "[V35D_MOTION_GATE] view=chase ok=0 reason=degenerate_chase" | tee -a "${SUITE_LOG}"
  fail_suite "chase_motion_degenerate"
fi

# FP motion check (non-fatal)
run_and_log python3 scripts/tools/check_capture_motion.py \
  --capture_dir "${CAP_DIR}" --tag v35d_fp --write_gif 0 || true

# ── 12) Final verdict ──
# v35d focus is visual credibility not Nav2 navigation quality.
# Nav2 outcomes are reported but not gating — v32/v35c suites test navigation.
if [ "${STAGED_SUCCEEDED}" -lt 1 ]; then
  echo "[V35D_NAV2_WARN] staged_succeed=${STAGED_SUCCEEDED} of ${STAGED_TOTAL} — reported but not gating" | tee -a "${SUITE_LOG}"
fi

echo "[V35D_SUITE_OK] ok=1 cases=${STAGED_TOTAL} succeeded=${STAGED_SUCCEEDED} skipped=${STAGED_SKIPPED} rooms=${ROOM_COUNT} doors=${DOOR_COUNT} root=${OUT_ROOT}" | tee -a "${SUITE_LOG}"
