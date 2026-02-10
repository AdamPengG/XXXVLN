#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/DualVLN"
RUN_DIR="${ROOT_DIR}/runs/topo_mvp/v20"
BUILD_DIR="${RUN_DIR}/builds"
LOG_DIR="${RUN_DIR}/logs"
BUILD_LOG="${LOG_DIR}/topo_build_v20.log"
EVIDENCE_DIR="${RUN_DIR}/evidence"
GT_AUDIT_DIR="${RUN_DIR}/gt_audit"

mkdir -p "${RUN_DIR}" "${BUILD_DIR}" "${LOG_DIR}" "${EVIDENCE_DIR}" "${GT_AUDIT_DIR}"
: > "${BUILD_LOG}"

export CUDA_VISIBLE_DEVICES=0
export VLN_SPLIT=train
export PYTHONHASHSEED=0

SCENES_CSV="${TOPO_SCENES:-8WUmhLawc2A,JeFG25nYj2p,r47D5H71a5s,ur6pFq6Qu1A,Vvot9Ly1tCj}"
QUERIES_CSV="${TOPO_QUERIES:-kitchen table|bedroom|bathroom sink|living room sofa|front door}"
START_OFFSETS_CSV="${TOPO_START_OFFSETS:-8}"

BUILD_STEPS="${TOPO_BUILD_STEPS:-190}"
SAVE_EVERY_K="${TOPO_SAVE_EVERY_K:-8}"
MAX_EPISODES="${TOPO_MAX_EPISODES:-0}"
MIN_NODES="${TOPO_MIN_NODES:-8}"
MAX_RETRY="${TOPO_BUILD_MAX_RETRY:-3}"
RETRY_STEP_DELTA="${TOPO_BUILD_RETRY_STEP_DELTA:-60}"
RETRY_OFFSET_STRIDE="${TOPO_BUILD_RETRY_OFFSET_STRIDE:-11}"
SKIP_BUILD="${TOPO_SKIP_BUILD:-0}"

MAX_STEPS="${TOPO_MAX_STEPS:-120}"
NODE_REACH_THRESH="${TOPO_NODE_REACH_THRESH:-0.80}"
KIDNAP_MIN_NODE_DIST="${TOPO_KIDNAP_MIN_NODE_DIST:-0.5}"
KIDNAP_MAX_NODE_DIST="${TOPO_KIDNAP_MAX_NODE_DIST:-3.5}"
RUN_FOLLOWER_BASELINE="${TOPO_RUN_FOLLOWER_BASELINE:-0}"

STALL_WINDOW="${TOPO_STALL_WINDOW:-8}"
STALL_MIN_DELTA="${TOPO_STALL_MIN_DELTA:-0.08}"
RECOVERY_MODE="${TOPO_RECOVERY_MODE:-hybrid}"
RECOVERY_TURN_STEPS="${TOPO_RECOVERY_TURN_STEPS:-3}"
RECOVERY_FOLLOWER_STEPS="${TOPO_RECOVERY_FOLLOWER_STEPS:-8}"
RECOVERY_COOLDOWN="${TOPO_RECOVERY_COOLDOWN:-5}"
STOP_GUARD_DIST="${TOPO_STOP_GUARD_DIST:-0.40}"
STOP_NEAR_M="${TOPO_STOP_NEAR_M:-1.80}"
STOP_NEAR_HOLD_STEPS="${TOPO_STOP_NEAR_HOLD_STEPS:-3}"
STOP_NEAR_GOAL_PROB="${TOPO_STOP_NEAR_GOAL_PROB:-0.05}"
STOP_NEAR_GOAL_SCORE_MIN="${TOPO_STOP_NEAR_GOAL_SCORE_MIN:-0.05}"
STOP_NEAR_LOW_CONF_GUARD="${TOPO_STOP_NEAR_LOW_CONF_GUARD:-0.18}"
STOP_NEAR_LOC_CONF_MIN="${TOPO_STOP_NEAR_LOC_CONF_MIN:-0.30}"
RECOVERY_GOAL_LOCK_AFTER="${TOPO_RECOVERY_GOAL_LOCK_AFTER:-4}"
RECOVERY_GOAL_LOCK_MIN_DIST="${TOPO_RECOVERY_GOAL_LOCK_MIN_DIST:-3.5}"
RECOVERY_GOAL_BURST_AFTER="${TOPO_RECOVERY_GOAL_BURST_AFTER:-4}"
RECOVERY_GOAL_BURST_STEPS="${TOPO_RECOVERY_GOAL_BURST_STEPS:-12}"

OBJECT_TOPK="${TOPO_OBJECT_TOPK:-8}"
ROOM_MIN="${TOPO_ROOM_MIN:-0.24}"
ROOM_QUERY_TOPK="${TOPO_ROOM_QUERY_TOPK:-5}"
GOAL_SCORE_THRESH="${TOPO_GOAL_SCORE_THRESH:-0.05}"
FILL_STEPS="${TOPO_FILL_STEPS:-60}"
FILL_MIN_NODE_SPACING="${TOPO_FILL_MIN_NODE_SPACING:-0.35}"
FILL_LOW_CONF_PATIENCE="${TOPO_FILL_LOW_CONF_PATIENCE:-6}"
FILL_RETRY_NAV_EXTRA_STEPS="${TOPO_FILL_RETRY_NAV_EXTRA_STEPS:-30}"
FILL_ROTATE_STEPS="${TOPO_FILL_ROTATE_STEPS:-8}"
FILL_FORWARD_STEPS="${TOPO_FILL_FORWARD_STEPS:-12}"

LOC_ALPHA="${TOPO_LOC_ALPHA:-0.60}"
LOC_PRIOR_ALPHA="${TOPO_LOC_PRIOR_ALPHA:-0.74}"
LOC_PRIOR_SIGMA="${TOPO_LOC_PRIOR_SIGMA:-1.85}"
LOC_CONF_MARGIN_CENTER="${TOPO_LOC_CONF_MARGIN_CENTER:-0.018}"
LOC_CONF_GAIN="${TOPO_LOC_CONF_GAIN:-14.0}"
BELIEF_TEMP="${TOPO_BELIEF_TEMP:-0.28}"
BELIEF_STAY_PROB="${TOPO_BELIEF_STAY_PROB:-0.68}"
BELIEF_MIX="${TOPO_BELIEF_MIX:-0.30}"
BELIEF_ENTROPY_FILL_THRESH="${TOPO_BELIEF_ENTROPY_FILL_THRESH:-0.98}"
BELIEF_TOPK_PRINT="${TOPO_BELIEF_TOPK_PRINT:-4}"
BELIEF_MAX_ENTROPY_USE="${TOPO_BELIEF_MAX_ENTROPY_USE:-0.88}"
BELIEF_HOP_RADIUS="${TOPO_BELIEF_HOP_RADIUS:-2}"
BELIEF_GEO_RADIUS="${TOPO_BELIEF_GEO_RADIUS:-3.5}"
BELIEF_NONCAND_PENALTY="${TOPO_BELIEF_NONCAND_PENALTY:-1.00}"
BELIEF_RESET_LOW_CONF="${TOPO_BELIEF_RESET_LOW_CONF:-0.18}"
BELIEF_RESET_HIGH_ENTROPY="${TOPO_BELIEF_RESET_HIGH_ENTROPY:-0.98}"
DOORWAY_BURST_STEPS="${TOPO_DOORWAY_BURST_STEPS:-0}"

MAX_NODES="${TOPO_MAX_NODES:-30}"
ODOM_NOISE_TRANS="${TOPO_ODOM_NOISE_TRANS:-0.02}"
ODOM_NOISE_YAW="${TOPO_ODOM_NOISE_YAW:-1.0}"

LC_SIM_THRESH="${TOPO_LC_SIM_THRESH:-0.92}"
LC_GEO_THRESH="${TOPO_LC_GEO_THRESH:-2.2}"
LC_GEO_SIM_THRESH="${TOPO_LC_GEO_SIM_THRESH:-0.92}"
LC_MIN_HOPS="${TOPO_LC_MIN_HOPS:-3}"
LC_REL_TRANS_ERR_MAX="${TOPO_LC_REL_TRANS_ERR_MAX:-1.15}"
LC_REL_YAW_ERR_DEG_MAX="${TOPO_LC_REL_YAW_ERR_DEG_MAX:-42.0}"
LC_WEIGHT_MIN="${TOPO_LC_WEIGHT_MIN:-0.8}"
LC_WEIGHT_MAX="${TOPO_LC_WEIGHT_MAX:-2.4}"
PG_LOOP_ROBUST="${TOPO_PG_LOOP_ROBUST:-huber}"
PG_LOOP_ROBUST_SCALE="${TOPO_PG_LOOP_ROBUST_SCALE:-0.65}"
ROOM_SPLIT_EDGE_DIST="${TOPO_ROOM_SPLIT_EDGE_DIST:-1.45}"
ROOM_SPLIT_YAW_DEG="${TOPO_ROOM_SPLIT_YAW_DEG:-62.0}"
ROOM_MIN_SPLIT_SPAN="${TOPO_ROOM_MIN_SPLIT_SPAN:-1.2}"
ROOM_MIN_SIDE_NODES="${TOPO_ROOM_MIN_SIDE_NODES:-3}"
ROOM_CLIP_AGG_K="${TOPO_ROOM_CLIP_AGG_K:-4}"
ROOM_CLIP_MARGIN_TH="${TOPO_ROOM_CLIP_MARGIN_TH:-0.04}"
ROOM_CLIP_MIN_SCORE="${TOPO_ROOM_CLIP_MIN_SCORE:-0.18}"
ROOM_SEG_METHOD="${TOPO_ROOM_SEG_METHOD:-hydralite}"
ROOM_SEG_MODE="${TOPO_ROOM_SEG_MODE:-EDGE_FILTRATION}"
ROOM_SEG_Q_MIN="${TOPO_ROOM_SEG_Q_MIN:-0.10}"
ROOM_SEG_Q_MAX="${TOPO_ROOM_SEG_Q_MAX:-0.80}"
ROOM_SEG_Q_POINTS="${TOPO_ROOM_SEG_Q_POINTS:-25}"
ROOM_SEG_MIN_COMPONENT_NODES="${TOPO_ROOM_SEG_MIN_COMPONENT_NODES:-2}"
EDGE_CLEARANCE_SAMPLES="${TOPO_EDGE_CLEARANCE_SAMPLES:-12}"
DILATION_MODE="${TOPO_DILATION_MODE:-PLATEAU}"
MIN_DILATION_M="${TOPO_MIN_DILATION_M:-0.5}"
MAX_DILATION_M="${TOPO_MAX_DILATION_M:-1.2}"
EVIDENCE_INPUT_BUILD_ROOT="${TOPO_ROOM_EVIDENCE_INPUT_BUILD_ROOT:-${BUILD_DIR}}"

IFS=',' read -r -a scenes <<< "${SCENES_CSV}"
IFS='|' read -r -a queries <<< "${QUERIES_CSV}"
IFS=',' read -r -a start_offsets <<< "${START_OFFSETS_CSV}"

if [ "${#queries[@]}" -ne 5 ]; then
  echo "Expected exactly 5 queries, got ${#queries[@]}." >&2
  exit 2
fi

build_scene_once() {
  local scene_idx="$1"
  local scene="$2"
  local scene_build="${BUILD_DIR}/${scene}"
  rm -rf "${scene_build}"
  mkdir -p "${scene_build}"

  local build_ok=0
  local build_steps_curr="${BUILD_STEPS}"
  local nodes=0
  local edges=0

  for try_idx in $(seq 1 "${MAX_RETRY}"); do
    local build_offset=$(( (scene_idx * 37 + try_idx * RETRY_OFFSET_STRIDE) % 279 ))
    local build_seed=$((123 + scene_idx * 97 + try_idx))
    echo "[TOPO_BUILD_RUN] scene=${scene} try=${try_idx}/${MAX_RETRY} steps=${build_steps_curr} start_offset=${build_offset}" | tee -a "${BUILD_LOG}"
    (
      cd "${ROOT_DIR}"
      TOPO_OUT_DIR="${scene_build}" \
      TOPO_SCENE_ID="${scene}" \
      TOPO_SEED="${build_seed}" \
      TOPO_BUILD_STEPS="${build_steps_curr}" \
      TOPO_SAVE_EVERY_K="${SAVE_EVERY_K}" \
      TOPO_MAX_EPISODES="${MAX_EPISODES}" \
      TOPO_START_EPISODE_OFFSET="${build_offset}" \
      TOPO_MAX_NODES="${MAX_NODES}" \
      TOPO_USE_EST_POSE="0" \
      TOPO_ODOM_NOISE_TRANS="0.0" \
      TOPO_ODOM_NOISE_YAW="0.0" \
      TOPO_LC_SIM_THRESH="${LC_SIM_THRESH}" \
      TOPO_LC_GEO_THRESH="${LC_GEO_THRESH}" \
      TOPO_LC_GEO_SIM_THRESH="${LC_GEO_SIM_THRESH}" \
      TOPO_LC_MIN_HOPS="${LC_MIN_HOPS}" \
      TOPO_LC_REL_TRANS_ERR_MAX="${LC_REL_TRANS_ERR_MAX}" \
      TOPO_LC_REL_YAW_ERR_DEG_MAX="${LC_REL_YAW_ERR_DEG_MAX}" \
      TOPO_LC_WEIGHT_MIN="${LC_WEIGHT_MIN}" \
      TOPO_LC_WEIGHT_MAX="${LC_WEIGHT_MAX}" \
      TOPO_PG_LOOP_ROBUST="${PG_LOOP_ROBUST}" \
      TOPO_PG_LOOP_ROBUST_SCALE="${PG_LOOP_ROBUST_SCALE}" \
      TOPO_ROOM_SPLIT_EDGE_DIST="${ROOM_SPLIT_EDGE_DIST}" \
      TOPO_ROOM_SPLIT_YAW_DEG="${ROOM_SPLIT_YAW_DEG}" \
      TOPO_ROOM_MIN_SPLIT_SPAN="${ROOM_MIN_SPLIT_SPAN}" \
      TOPO_ROOM_MIN_SIDE_NODES="${ROOM_MIN_SIDE_NODES}" \
      TOPO_ROOM_CLIP_AGG_K="${ROOM_CLIP_AGG_K}" \
      TOPO_ROOM_CLIP_MARGIN_TH="${ROOM_CLIP_MARGIN_TH}" \
      TOPO_ROOM_CLIP_MIN_SCORE="${ROOM_CLIP_MIN_SCORE}" \
      TOPO_ROOM_SEG_METHOD="${ROOM_SEG_METHOD}" \
      TOPO_ROOM_SEG_MODE="${ROOM_SEG_MODE}" \
      TOPO_ROOM_SEG_Q_MIN="${ROOM_SEG_Q_MIN}" \
      TOPO_ROOM_SEG_Q_MAX="${ROOM_SEG_Q_MAX}" \
      TOPO_ROOM_SEG_Q_POINTS="${ROOM_SEG_Q_POINTS}" \
      TOPO_ROOM_SEG_MIN_COMPONENT_NODES="${ROOM_SEG_MIN_COMPONENT_NODES}" \
      TOPO_EDGE_CLEARANCE_SAMPLES="${EDGE_CLEARANCE_SAMPLES}" \
      TOPO_DILATION_MODE="${DILATION_MODE}" \
      TOPO_MIN_DILATION_M="${MIN_DILATION_M}" \
      TOPO_MAX_DILATION_M="${MAX_DILATION_M}" \
      TOPO_GT_ROOM_AUDIT="1" \
      TOPO_GT_AUDIT_OUT_ROOT="${GT_AUDIT_DIR}" \
      bash scripts/topo_explore_build.sh
    ) >> "${BUILD_LOG}" 2>&1

    read -r nodes edges <<EOF
$(SCENE_OUT="${scene_build}" python - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ["SCENE_OUT"]) / "graph.json"
if not p.exists():
    print("0 0")
else:
    g = json.loads(p.read_text())
    print(len(g.get("nodes", [])), len(g.get("edges", [])))
PY
)
EOF

    if [ "${nodes}" -ge "${MIN_NODES}" ]; then
      build_ok=1
      break
    fi
    if [ "${try_idx}" -lt "${MAX_RETRY}" ]; then
      local next_try=$((try_idx + 1))
      echo "[TOPO_BUILD_RETRY] scene=${scene} try=${next_try} reason=low_nodes nodes=${nodes} steps=${build_steps_curr}" | tee -a "${BUILD_LOG}"
      build_steps_curr=$((build_steps_curr + RETRY_STEP_DELTA))
    fi
  done

  if [ "${build_ok}" -ne 1 ]; then
    echo "[TOPO_BUILD_FAIL] scene=${scene} reason=low_nodes after_tries=${MAX_RETRY}" | tee -a "${BUILD_LOG}"
    return 1
  fi

  (
    cd "${ROOT_DIR}"
    python scripts/topo_room_semantics_audit.py \
      --scene_id "${scene}" \
      --room_graph "${scene_build}/room_graph.json" \
      --object_graph "${scene_build}/object_graph.json" \
      --node_meta "${scene_build}/node_meta.json" \
      --out_json "${scene_build}/room_semantics_audit.json"
  ) >> "${BUILD_LOG}" 2>&1
  return 0
}

run_condition() {
  local cond="$1"
  local use_est_pose="$2"
  local odom_trans="$3"
  local odom_yaw="$4"

  local cond_dir="${RUN_DIR}/condition_${cond}"
  local cond_log="${LOG_DIR}/topo_eval_suite_v20_${cond}.log"
  rm -rf "${cond_dir}"
  mkdir -p "${cond_dir}"
  : > "${cond_log}"

  for scene_idx in "${!scenes[@]}"; do
    local scene="${scenes[$scene_idx]}"
    local scene_build="${BUILD_DIR}/${scene}"
    local scene_out="${cond_dir}/${scene}"
    mkdir -p "${scene_out}"
    local loop_edges=0
    local pg_after=0.0
    local pg_before=0.0
    read -r loop_edges pg_before pg_after <<EOF
$(SCENE_BUILD="${scene_build}" python - <<'PY'
import json, os
from pathlib import Path
g = json.loads((Path(os.environ["SCENE_BUILD"]) / "graph.json").read_text())
m = g.get("meta", {})
print(
    int(float(m.get("loop_closure_edges", 0))),
    float(m.get("pose_graph_residual_before", 0.0)),
    float(m.get("pose_graph_residual_after", 0.0)),
)
PY
)
EOF

    for q_idx in "${!queries[@]}"; do
      local query="${queries[$q_idx]}"
      for s_idx in "${!start_offsets[@]}"; do
        local start_offset="${start_offsets[$s_idx]}"
        local run_id="${scene}_q${q_idx}_s${s_idx}_bearing"
        local kidnap_seed=$((100000 + scene_idx * 1000 + q_idx * 100 + s_idx))
        echo "[TOPO_EVAL_RUN] condition=${cond} scene=${scene} run_id=${run_id} query=\"${query}\" start_offset=${start_offset} mode=bearing" | tee -a "${cond_log}"
        (
          cd "${ROOT_DIR}"
          TOPO_CONTROLLER=bearing \
          TOPO_RUN_CONDITION="${cond}" \
          TOPO_DEBUG_ROOT="${RUN_DIR}/debug_runs" \
          TOPO_BELIEF_MAX_ENTROPY_USE="${BELIEF_MAX_ENTROPY_USE}" \
          TOPO_STOP_NEAR_LOC_CONF_MIN="${STOP_NEAR_LOC_CONF_MIN}" \
          TOPO_RECOVERY_GOAL_LOCK_AFTER="${RECOVERY_GOAL_LOCK_AFTER}" \
          TOPO_RECOVERY_GOAL_LOCK_MIN_DIST="${RECOVERY_GOAL_LOCK_MIN_DIST}" \
          TOPO_RECOVERY_GOAL_BURST_AFTER="${RECOVERY_GOAL_BURST_AFTER}" \
          TOPO_RECOVERY_GOAL_BURST_STEPS="${RECOVERY_GOAL_BURST_STEPS}" \
          TOPO_ENABLE_RELLOC_SCAN=0 \
          TOPO_ENABLE_DOORWAY_BURST=0 \
          TOPO_ENABLE_WATCHDOG_FILL=0 \
          python -m internnav.topo.relocalize \
            --config_path "repo/InternNav/scripts/eval/configs/vln_r2r.yaml" \
            --graph_json "${scene_build}/graph.json" \
            --embeds_npy "${scene_build}/node_embeds.npy" \
            --clip_embeds_npy "${scene_build}/node_embeds_clip.npy" \
            --geo_embeds_npy "${scene_build}/node_geo.npy" \
            --object_graph_json "${scene_build}/object_graph.json" \
            --object_nodes_json "${scene_build}/object_nodes.json" \
            --node_meta "${scene_build}/node_meta.json" \
            --query "${query}" \
            --out_dir "${scene_out}" \
            --max_steps "${MAX_STEPS}" \
            --model_path "checkpoints/InternVLA-N1-DualVLN" \
            --pooling "mean" \
            --clip_model_name "openai/clip-vit-base-patch32" \
            --controller_mode "bearing" \
            --vote_window 5 \
            --relocalize_every 5 \
            --node_reach_thresh "${NODE_REACH_THRESH}" \
            --start_episode_offset "${start_offset}" \
            --vision_prompt "" \
            --goal_topk 5 \
            --loc_conf_thresh 0.34 \
            --kidnap_start 1 \
            --kidnap_seed "${kidnap_seed}" \
            --kidnap_min_nearest_node_dist "${KIDNAP_MIN_NODE_DIST}" \
            --kidnap_max_nearest_node_dist "${KIDNAP_MAX_NODE_DIST}" \
            --use_two_stage_goal 1 \
            --room_shortlist_k 12 \
            --object_topk "${OBJECT_TOPK}" \
            --room_min "${ROOM_MIN}" \
            --room_query_topk "${ROOM_QUERY_TOPK}" \
            --goal_score_thresh "${GOAL_SCORE_THRESH}" \
            --fill_steps "${FILL_STEPS}" \
            --fill_min_node_spacing "${FILL_MIN_NODE_SPACING}" \
            --fill_relocalize_low_conf_patience "${FILL_LOW_CONF_PATIENCE}" \
            --fill_retry_nav_extra_steps "${FILL_RETRY_NAV_EXTRA_STEPS}" \
            --fill_rotate_steps "${FILL_ROTATE_STEPS}" \
            --fill_forward_steps "${FILL_FORWARD_STEPS}" \
            --loc_fuse_alpha "${LOC_ALPHA}" \
            --loc_prior_alpha "${LOC_PRIOR_ALPHA}" \
            --loc_prior_sigma "${LOC_PRIOR_SIGMA}" \
            --loc_conf_margin_center "${LOC_CONF_MARGIN_CENTER}" \
            --loc_conf_gain "${LOC_CONF_GAIN}" \
            --belief_temp "${BELIEF_TEMP}" \
            --belief_stay_prob "${BELIEF_STAY_PROB}" \
            --belief_mix "${BELIEF_MIX}" \
            --belief_entropy_fill_thresh "${BELIEF_ENTROPY_FILL_THRESH}" \
            --belief_topk_print "${BELIEF_TOPK_PRINT}" \
            --belief_hop_radius "${BELIEF_HOP_RADIUS}" \
            --belief_geo_radius "${BELIEF_GEO_RADIUS}" \
            --belief_noncand_penalty "${BELIEF_NONCAND_PENALTY}" \
            --belief_reset_low_conf "${BELIEF_RESET_LOW_CONF}" \
            --belief_reset_high_entropy "${BELIEF_RESET_HIGH_ENTROPY}" \
            --geo_bins_r 20 \
            --geo_bins_theta 60 \
            --geo_max_depth 5.0 \
            --max_nodes "${MAX_NODES}" \
            --stall_window "${STALL_WINDOW}" \
            --stall_min_delta "${STALL_MIN_DELTA}" \
            --recovery_mode "${RECOVERY_MODE}" \
            --recovery_turn_steps "${RECOVERY_TURN_STEPS}" \
            --recovery_follower_steps "${RECOVERY_FOLLOWER_STEPS}" \
            --recovery_cooldown "${RECOVERY_COOLDOWN}" \
            --stop_guard_dist "${STOP_GUARD_DIST}" \
            --stop_near_m "${STOP_NEAR_M}" \
            --stop_near_hold_steps "${STOP_NEAR_HOLD_STEPS}" \
            --stop_near_goal_prob "${STOP_NEAR_GOAL_PROB}" \
            --stop_near_goal_score_min "${STOP_NEAR_GOAL_SCORE_MIN}" \
            --stop_near_low_conf_guard "${STOP_NEAR_LOW_CONF_GUARD}" \
            --doorway_burst_steps "${DOORWAY_BURST_STEPS}" \
            --use_est_pose "${use_est_pose}" \
            --odom_noise_trans "${odom_trans}" \
            --odom_noise_yaw_deg "${odom_yaw}" \
            --run_id "${run_id}"
        ) >> "${cond_log}" 2>&1
      done
    done

    if [ "${RUN_FOLLOWER_BASELINE}" = "1" ]; then
      local baseline_id="${scene}_baseline_follower"
      echo "[TOPO_EVAL_RUN] condition=${cond} scene=${scene} run_id=${baseline_id} query=\"${queries[0]}\" start_offset=${start_offsets[0]} mode=follower" | tee -a "${cond_log}"
      (
        cd "${ROOT_DIR}"
        TOPO_CONTROLLER=follower \
        TOPO_RUN_CONDITION="${cond}" \
        TOPO_DEBUG_ROOT="${RUN_DIR}/debug_runs" \
        TOPO_BELIEF_MAX_ENTROPY_USE="${BELIEF_MAX_ENTROPY_USE}" \
        TOPO_STOP_NEAR_LOC_CONF_MIN="${STOP_NEAR_LOC_CONF_MIN}" \
        TOPO_RECOVERY_GOAL_LOCK_AFTER="${RECOVERY_GOAL_LOCK_AFTER}" \
        TOPO_RECOVERY_GOAL_LOCK_MIN_DIST="${RECOVERY_GOAL_LOCK_MIN_DIST}" \
        TOPO_RECOVERY_GOAL_BURST_AFTER="${RECOVERY_GOAL_BURST_AFTER}" \
        TOPO_RECOVERY_GOAL_BURST_STEPS="${RECOVERY_GOAL_BURST_STEPS}" \
        TOPO_ENABLE_RELLOC_SCAN=0 \
        TOPO_ENABLE_DOORWAY_BURST=0 \
        TOPO_ENABLE_WATCHDOG_FILL=0 \
        python -m internnav.topo.relocalize \
          --config_path "repo/InternNav/scripts/eval/configs/vln_r2r.yaml" \
          --graph_json "${scene_build}/graph.json" \
          --embeds_npy "${scene_build}/node_embeds.npy" \
          --clip_embeds_npy "${scene_build}/node_embeds_clip.npy" \
          --geo_embeds_npy "${scene_build}/node_geo.npy" \
          --object_graph_json "${scene_build}/object_graph.json" \
          --object_nodes_json "${scene_build}/object_nodes.json" \
          --node_meta "${scene_build}/node_meta.json" \
          --query "${queries[0]}" \
          --out_dir "${scene_out}" \
          --max_steps "${MAX_STEPS}" \
          --model_path "checkpoints/InternVLA-N1-DualVLN" \
          --pooling "mean" \
          --clip_model_name "openai/clip-vit-base-patch32" \
          --controller_mode "follower" \
          --vote_window 5 \
          --relocalize_every 5 \
          --node_reach_thresh "${NODE_REACH_THRESH}" \
          --start_episode_offset "${start_offsets[0]}" \
          --loc_conf_thresh 0.34 \
          --kidnap_start 1 \
          --kidnap_seed "$((200000 + scene_idx * 17))" \
          --kidnap_min_nearest_node_dist "${KIDNAP_MIN_NODE_DIST}" \
          --kidnap_max_nearest_node_dist "${KIDNAP_MAX_NODE_DIST}" \
          --use_two_stage_goal 1 \
          --room_shortlist_k 12 \
          --object_topk "${OBJECT_TOPK}" \
          --room_min "${ROOM_MIN}" \
          --room_query_topk "${ROOM_QUERY_TOPK}" \
          --loc_fuse_alpha "${LOC_ALPHA}" \
          --loc_prior_alpha "${LOC_PRIOR_ALPHA}" \
          --loc_prior_sigma "${LOC_PRIOR_SIGMA}" \
          --loc_conf_margin_center "${LOC_CONF_MARGIN_CENTER}" \
          --loc_conf_gain "${LOC_CONF_GAIN}" \
          --belief_temp "${BELIEF_TEMP}" \
          --belief_stay_prob "${BELIEF_STAY_PROB}" \
          --belief_mix "${BELIEF_MIX}" \
          --belief_entropy_fill_thresh "${BELIEF_ENTROPY_FILL_THRESH}" \
          --belief_topk_print "${BELIEF_TOPK_PRINT}" \
          --belief_hop_radius "${BELIEF_HOP_RADIUS}" \
          --belief_geo_radius "${BELIEF_GEO_RADIUS}" \
          --belief_noncand_penalty "${BELIEF_NONCAND_PENALTY}" \
          --belief_reset_low_conf "${BELIEF_RESET_LOW_CONF}" \
          --belief_reset_high_entropy "${BELIEF_RESET_HIGH_ENTROPY}" \
          --geo_bins_r 20 \
          --geo_bins_theta 60 \
          --geo_max_depth 5.0 \
          --max_nodes "${MAX_NODES}" \
          --stop_guard_dist "${STOP_GUARD_DIST}" \
          --stop_near_m "${STOP_NEAR_M}" \
          --stop_near_hold_steps "${STOP_NEAR_HOLD_STEPS}" \
          --stop_near_goal_prob "${STOP_NEAR_GOAL_PROB}" \
          --stop_near_goal_score_min "${STOP_NEAR_GOAL_SCORE_MIN}" \
          --stop_near_low_conf_guard "${STOP_NEAR_LOW_CONF_GUARD}" \
          --doorway_burst_steps "${DOORWAY_BURST_STEPS}" \
          --use_est_pose "${use_est_pose}" \
          --odom_noise_trans "${odom_trans}" \
          --odom_noise_yaw_deg "${odom_yaw}" \
          --run_id "${baseline_id}"
      ) >> "${cond_log}" 2>&1
    fi

    SCENE_OUT="${scene_out}" SCENE_ID="${scene}" LOOP_EDGES="${loop_edges}" PG_BEFORE="${pg_before}" PG_AFTER="${pg_after}" python - <<'PY' >> "${cond_log}" 2>&1
import collections
import json
import os
from pathlib import Path

scene_out = Path(os.environ["SCENE_OUT"])
scene = os.environ["SCENE_ID"]
files = sorted(scene_out.glob(f"RESULT_{scene}_q*_s*_bearing.json"))
runs = len(files)
success = 0
steps = 0
loc_conf = 0.0
goal_score = 0.0
fill_triggers = 0
early_stop = 0
failed_max_steps = 0
terminated_max_steps = 0
rooms_visited_hist = collections.Counter()
for f in files:
    d = json.loads(f.read_text())
    succ = bool(d.get("success", False))
    success += int(succ)
    steps += int(d.get("steps_used", 0))
    loc_conf += float(d.get("avg_loc_conf", 0.0))
    goal_score += float(d.get("goal_score", 0.0))
    fill_triggers += int(d.get("fill_triggers", 0))
    rooms_visited_hist[int(d.get("rooms_visited_count", 0))] += 1
    fr = d.get("fail_reason", None)
    terminated_by = str(d.get("terminated_by", ""))
    if fr == "early_stop_before_node":
        early_stop += 1
    if terminated_by == "max_steps":
        terminated_max_steps += 1
        if not succ:
            failed_max_steps += 1

room_graph_path = scene_out.parent.parent / "builds" / scene / "room_graph.json"
audit_path = scene_out.parent.parent / "builds" / scene / "room_semantics_audit.json"
gt_audit_path = scene_out.parent.parent / "gt_audit" / f"{scene}_gt_confusion.json"
rooms = 0
unique_room_labels = 0
unique_non_unknown_labels = 0
obj_label_ratio = 0.0
gt_top1_agreement = 0.0
gt_available = 0
doorway_edges_count = 0
doorway_nodes_count = 0
delta_star = -1.0
k_curve_preview = []
if room_graph_path.exists():
    rg = json.loads(room_graph_path.read_text())
    rooms = len(rg.get("rooms", {}))
    room_rows = list(rg.get("rooms", {}).values())
    labels = [str(x.get("label", "unknown")) for x in room_rows]
    unique_room_labels = len(set(labels))
    unique_non_unknown_labels = len(
        set(x for x in labels if x not in {"unknown", "unknown_clip", ""})
    )
    if len(room_rows) > 0:
        obj_label_ratio = float(
            sum(1 for x in room_rows if str(x.get("label_source", "")) == "objects")
        ) / float(len(room_rows))
    seg = rg.get("segmentation_debug", {})
    doorway = rg.get("doorway_debug", {})
    delta_star = float(seg.get("delta_star", -1.0))
    doorway_edges_count = int(doorway.get("doorway_edges_count", seg.get("doorway_edges_count", 0)))
    doorway_nodes_count = int(doorway.get("doorway_nodes_count", seg.get("doorway_nodes_count", 0)))
    deltas = [float(x) for x in seg.get("deltas", [])]
    ks = [int(x) for x in seg.get("Ks", [])]
    k_curve_preview = [[round(d, 4), int(k)] for d, k in list(zip(deltas, ks))[:12]]
if audit_path.exists():
    audit = json.loads(audit_path.read_text())
    if unique_room_labels <= 0:
        unique_room_labels = int(audit.get("unique_labels", 0))
    if obj_label_ratio <= 0.0:
        obj_label_ratio = float(audit.get("object_label_ratio", 0.0))
if gt_audit_path.exists():
    gt_data = json.loads(gt_audit_path.read_text())
    gt_available = int(gt_data.get("gt_available", 0))
    gt_top1_agreement = float(gt_data.get("top1_agreement", 0.0))

summary = {
    "scene_id": scene,
    "bearing_runs": runs,
    "bearing_success": success,
    "avg_steps": (steps / runs) if runs else 0.0,
    "avg_loc_conf": (loc_conf / runs) if runs else 0.0,
    "avg_goal_score": (goal_score / runs) if runs else 0.0,
    "fill_triggers": int(fill_triggers),
    "early_stop": int(early_stop),
    "failed_max_steps": int(failed_max_steps),
    "terminated_max_steps": int(terminated_max_steps),
    "loops": int(float(os.environ["LOOP_EDGES"])),
    "pg_before_resid": float(os.environ["PG_BEFORE"]),
    "pg_after_resid": float(os.environ["PG_AFTER"]),
    "rooms": int(rooms),
    "unique_room_labels": int(unique_room_labels),
    "unique_non_unknown_labels": int(unique_non_unknown_labels),
    "obj_label_ratio": float(obj_label_ratio),
    "gt_available": int(gt_available),
    "gt_top1_agreement": float(gt_top1_agreement),
    "delta_star": float(delta_star),
    "doorway_edges_count": int(doorway_edges_count),
    "doorway_nodes_count": int(doorway_nodes_count),
    "k_curve_preview": k_curve_preview,
    "rooms_visited_hist": {str(k): int(v) for k, v in sorted(rooms_visited_hist.items())},
}
(scene_out / "scene_summary.json").write_text(json.dumps(summary, indent=2))
print(
    f"[TOPO_SCENE_SUMMARY_V20] scene={scene} runs={summary['bearing_runs']} success={summary['bearing_success']} "
    f"avg_steps={summary['avg_steps']:.2f} avg_loc_conf={summary['avg_loc_conf']:.3f} "
    f"avg_goal_score={summary['avg_goal_score']:.3f} rooms={summary['rooms']} "
    f"unique_labels={summary['unique_room_labels']} non_unknown={summary['unique_non_unknown_labels']} "
    f"obj_label_ratio={summary['obj_label_ratio']:.3f} doorway_edges={summary['doorway_edges_count']} "
    f"doorway_nodes={summary['doorway_nodes_count']} loops={summary['loops']}",
    flush=True,
)
PY
  done

  COND_DIR="${cond_dir}" COND="${cond}" python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

cond_dir = Path(os.environ["COND_DIR"])
cond = os.environ["COND"]
scene_summaries = []
failure = Counter()
rooms_visited_hist = Counter()
failure_rows = []
total_runs = 0
total_success = 0
sum_steps = 0.0
sum_loc_conf = 0.0
sum_goal_score = 0.0
sum_early = 0
sum_failed_max_steps = 0
sum_terminated_max_steps = 0
sum_loops = 0
sum_pg_after = 0.0
sum_pg_count = 0
room_ok_scenes = 0
sum_unique_room_labels = 0.0
sum_unique_non_unknown_labels = 0.0
sum_obj_label_ratio = 0.0
sum_gt_top1 = 0.0
sum_gt_count = 0
sum_rooms_per_scene = 0.0
sum_doorway_edges = 0.0
sum_doorway_nodes = 0.0
ge2_rooms = 0

for p in sorted(cond_dir.glob("*/scene_summary.json")):
    s = json.loads(p.read_text())
    scene_summaries.append(s)
    r = int(s.get("bearing_runs", 0))
    total_runs += r
    total_success += int(s.get("bearing_success", 0))
    sum_steps += float(s.get("avg_steps", 0.0)) * r
    sum_loc_conf += float(s.get("avg_loc_conf", 0.0)) * r
    sum_goal_score += float(s.get("avg_goal_score", 0.0)) * r
    sum_early += int(s.get("early_stop", 0))
    sum_failed_max_steps += int(s.get("failed_max_steps", 0))
    sum_terminated_max_steps += int(s.get("terminated_max_steps", 0))
    sum_loops += int(s.get("loops", 0))
    sum_pg_after += float(s.get("pg_after_resid", 0.0))
    sum_pg_count += 1
    sum_unique_room_labels += float(s.get("unique_room_labels", 0.0))
    if int(s.get("rooms", 0)) >= 2:
        sum_unique_non_unknown_labels += float(s.get("unique_non_unknown_labels", 0.0))
    sum_obj_label_ratio += float(s.get("obj_label_ratio", 0.0))
    if int(s.get("gt_available", 0)) == 1:
        sum_gt_top1 += float(s.get("gt_top1_agreement", 0.0))
        sum_gt_count += 1
    if int(s.get("rooms", 0)) >= 2:
        room_ok_scenes += 1
    sum_rooms_per_scene += float(s.get("rooms", 0.0))
    sum_doorway_edges += float(s.get("doorway_edges_count", 0.0))
    sum_doorway_nodes += float(s.get("doorway_nodes_count", 0.0))

for rf in sorted(cond_dir.glob("*/RESULT_*_bearing.json")):
    d = json.loads(rf.read_text())
    run_id = rf.stem.replace("RESULT_", "")
    succ = bool(d.get("success", False))
    fail_reason_raw = d.get("fail_reason", None)
    fail_reason = str(fail_reason_raw) if fail_reason_raw is not None else ""
    terminated_by = str(d.get("terminated_by", ""))
    success_by = d.get("success_by", None)
    if not succ:
        failure[fail_reason if fail_reason else "unknown"] += 1
    rooms_visited = int(d.get("rooms_visited_count", 0))
    rooms_visited_hist[rooms_visited] += 1
    if rooms_visited >= 2:
        ge2_rooms += 1
    failure_rows.append(
        {
            "run_id": run_id,
            "success": int(succ),
            "fail_reason": fail_reason if fail_reason else None,
            "terminated_by": terminated_by if terminated_by else None,
            "success_by": success_by,
            "steps_used": int(d.get("steps_used", 0)),
            "final_dist_to_goal_node": float(d.get("final_dist_to_goal_node", 0.0)),
            "avg_loc_conf": float(d.get("avg_loc_conf", 0.0)),
            "goal_score": float(d.get("goal_score", 0.0)),
            "watchdog_trigger_count": int(d.get("watchdog_trigger_count", 0)),
            "relocalize_reset_count": int(d.get("relocalize_reset_count", 0)),
            "doorway_burst_count": int(d.get("doorway_burst_count", 0)),
            "near_goal_stop_count": int(d.get("near_goal_stop_count", 0)),
            "belief_entropy_summary": d.get("belief_entropy_summary", {}),
            "room_id_seq": d.get("room_id_seq", []),
            "rooms_visited_count": rooms_visited,
            "room_transitions": d.get("room_transitions", []),
            "path_nodes": d.get("path_nodes", []),
            "path_room_seq": d.get("path_room_seq", []),
            "action_histogram": d.get("action_histogram", {}),
        }
    )
    print(
        f"[TOPO_RUN_ROOM_SEQ] run={run_id} rooms_visited={rooms_visited} "
        f"seq={d.get('room_id_seq', [])} transitions={int(d.get('room_transitions_count', 0))}",
        flush=True,
    )

failure_jsonl = cond_dir / "failure_cases.jsonl"
with failure_jsonl.open("w") as f:
    for row in failure_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "condition": cond,
    "scenes": len(scene_summaries),
    "builds": len(scene_summaries),
    "runs": int(total_runs),
    "success": int(total_success),
    "avg_steps": float(sum_steps / max(1, total_runs)),
    "avg_loc_conf": float(sum_loc_conf / max(1, total_runs)),
    "avg_goal_score": float(sum_goal_score / max(1, total_runs)),
    "early_stop": int(sum_early),
    "failed_max_steps": int(sum_failed_max_steps),
    "terminated_max_steps": int(sum_terminated_max_steps),
    "loops": int(sum_loops),
    "pg_after_resid": float(sum_pg_after / max(1, sum_pg_count)),
    "rooms_ge_2_scenes": int(room_ok_scenes),
    "avg_rooms_per_scene": float(sum_rooms_per_scene / max(1, len(scene_summaries))),
    "avg_doorway_edges_per_scene": float(sum_doorway_edges / max(1, len(scene_summaries))),
    "avg_doorway_nodes_per_scene": float(sum_doorway_nodes / max(1, len(scene_summaries))),
    "unique_room_labels_avg": float(sum_unique_room_labels / max(1, len(scene_summaries))),
    "unique_non_unknown_labels_avg": float(sum_unique_non_unknown_labels / max(1, room_ok_scenes)),
    "obj_label_ratio": float(sum_obj_label_ratio / max(1, len(scene_summaries))),
    "gt_top1_agreement_avg": float(sum_gt_top1 / max(1, sum_gt_count)),
    "gt_top1_agreement_count": int(sum_gt_count),
    "rooms_visited_hist": {str(k): int(v) for k, v in sorted(rooms_visited_hist.items())},
    "room_traversal_ge2": int(ge2_rooms),
    "room_traversal_ge2_pct": float(100.0 * ge2_rooms / max(1, total_runs)),
    "scene_summaries": scene_summaries,
    "failure_taxonomy": dict(failure),
}
(cond_dir.parent / f"condition_{cond}_summary.json").write_text(json.dumps(summary, indent=2))
(cond_dir.parent / f"condition_{cond}_failure_taxonomy.json").write_text(json.dumps(dict(failure), indent=2))
print(
    f"[TOPO_ROOM_TRAVERSAL_SUMMARY] condition={cond} runs={total_runs} ge2_rooms={ge2_rooms} "
    f"pct={summary['room_traversal_ge2_pct']:.2f} hist={summary['rooms_visited_hist']}",
    flush=True,
)
print(
    f"[TOPO_EVAL_SUMMARY_V20] condition={summary['condition']} scenes={summary['scenes']} builds={summary['builds']} "
    f"runs={summary['runs']} success={summary['success']} avg_steps={summary['avg_steps']:.2f} "
    f"avg_loc_conf={summary['avg_loc_conf']:.3f} avg_goal_score={summary['avg_goal_score']:.3f} "
    f"early_stop={summary['early_stop']} failed_max_steps={summary['failed_max_steps']} "
    f"terminated_max_steps={summary['terminated_max_steps']} loops={summary['loops']} "
    f"pg_after_resid={summary['pg_after_resid']:.6f} avg_rooms_per_scene={summary['avg_rooms_per_scene']:.3f} "
    f"unique_non_unknown_labels_avg={summary['unique_non_unknown_labels_avg']:.3f} "
    f"obj_label_ratio={summary['obj_label_ratio']:.3f}",
    flush=True,
)
PY
}

if [ "${SKIP_BUILD}" != "1" ]; then
  for scene_idx in "${!scenes[@]}"; do
    scene="${scenes[$scene_idx]}"
    build_scene_once "${scene_idx}" "${scene}"
  done
else
  for scene in "${scenes[@]}"; do
    if [ ! -f "${BUILD_DIR}/${scene}/graph.json" ]; then
      echo "Missing build artifact for scene=${scene}: ${BUILD_DIR}/${scene}/graph.json" >&2
      exit 2
    fi
    if [ ! -f "${BUILD_DIR}/${scene}/room_semantics_audit.json" ]; then
      (
        cd "${ROOT_DIR}"
        python scripts/topo_room_semantics_audit.py \
          --scene_id "${scene}" \
          --room_graph "${BUILD_DIR}/${scene}/room_graph.json" \
          --object_graph "${BUILD_DIR}/${scene}/object_graph.json" \
          --node_meta "${BUILD_DIR}/${scene}/node_meta.json" \
          --out_json "${BUILD_DIR}/${scene}/room_semantics_audit.json"
      ) >> "${BUILD_LOG}" 2>&1
    fi
  done
  echo "[TOPO_BUILD_REUSE] using existing builds at ${BUILD_DIR}"
fi

run_condition "gt_pose" "0" "0.0" "0.0"
run_condition "est_pose_noise" "1" "${ODOM_NOISE_TRANS}" "${ODOM_NOISE_YAW}"

(
  cd "${ROOT_DIR}"
  python scripts/topo_room_semantics_evidence.py \
    --input_build_root "${EVIDENCE_INPUT_BUILD_ROOT}" \
    --failure_root "${RUN_DIR}" \
    --output_json "${EVIDENCE_DIR}/room_semantics_evidence.json"
) >> "${EVIDENCE_DIR}/topo_room_semantics_evidence.log" 2>&1

python - <<'PY'
import json
from pathlib import Path

root = Path("/home/peng/DualVLN/runs/topo_mvp/v20")
gt = json.loads((root / "condition_gt_pose_summary.json").read_text())
est = json.loads((root / "condition_est_pose_noise_summary.json").read_text())
summary = {
    "conditions": {
        "gt_pose": gt,
        "est_pose_noise": est,
    },
    "metrics": {
        "gt_pose_success": int(gt.get("success", 0)),
        "est_pose_noise_success": int(est.get("success", 0)),
        "gt_pose_runs": int(gt.get("runs", 0)),
        "est_pose_noise_runs": int(est.get("runs", 0)),
        "gt_pose_unique_non_unknown_labels_avg": float(gt.get("unique_non_unknown_labels_avg", 0.0)),
        "est_pose_noise_unique_non_unknown_labels_avg": float(est.get("unique_non_unknown_labels_avg", 0.0)),
        "gt_pose_obj_label_ratio": float(gt.get("obj_label_ratio", 0.0)),
        "est_pose_noise_obj_label_ratio": float(est.get("obj_label_ratio", 0.0)),
        "gt_pose_gt_top1_agreement_avg": float(gt.get("gt_top1_agreement_avg", 0.0)),
        "est_pose_noise_gt_top1_agreement_avg": float(est.get("gt_top1_agreement_avg", 0.0)),
    },
    "targets": {
        "gt_pose_success_min": 23,
        "est_pose_noise_success_min": 23,
        "room_semantics_unique_labels_min": 2,
        "room_semantics_obj_label_ratio_min": 0.9,
        "gt_top1_agreement_min": 0.5,
    },
}
(root / "v20_summary.json").write_text(json.dumps(summary, indent=2))
combined_failure = {
    "gt_pose": gt.get("failure_taxonomy", {}),
    "est_pose_noise": est.get("failure_taxonomy", {}),
}
(root / "v20_failure_taxonomy.json").write_text(json.dumps(combined_failure, indent=2))
PY

if [ "${TOPO_DEBUG_CAPTURE:-0}" = "1" ]; then
  DEBUG_ROOT="${RUN_DIR}/debug_runs"
  mkdir -p "${DEBUG_ROOT}"
  while IFS= read -r meta_file; do
    run_dir="$(dirname "${meta_file}")"
    (
      cd "${ROOT_DIR}"
      python scripts/topo_debug_ui_build_report.py \
        --run_dir "${run_dir}" \
        --build_root "${BUILD_DIR}"
    ) >> "${LOG_DIR}/topo_eval_suite_v20.log" 2>&1 || true
  done < <(find "${DEBUG_ROOT}" -type f -name "meta.json" | sort)
  (
    cd "${ROOT_DIR}"
    python scripts/topo_debug_ui_index.py --root "${DEBUG_ROOT}"
  ) >> "${LOG_DIR}/topo_eval_suite_v20.log" 2>&1
  echo "[TOPO_DEBUG_INDEX_OK] path=${DEBUG_ROOT}/index.html"
fi

echo "[V20] wrote ${RUN_DIR}/v20_summary.json"
