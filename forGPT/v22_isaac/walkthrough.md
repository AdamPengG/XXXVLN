# Topo MVP v22 Isaac Bring-up

## Scope
- Keep Habitat v20/v21 code path unchanged by default.
- Add strict real Isaac launch path (`backend=isaac`) with fail-fast when omni is unavailable unless `ISAAC_BACKEND_ALLOW_FALLBACK=1`.
- Reuse v20 debug capture + report UI for Isaac runs.

## Key Anchors Observed
- `[ISAAC_ENV_RESOLVED] root=docker-image:nvcr.io/nvidia/isaac-sim:5.1.0:/isaac-sim`
- `[ISAAC_ENV_OK] root=docker-image:nvcr.io/nvidia/isaac-sim:5.1.0:/isaac-sim msg=OMNI_OK`
- `[ISAAC_BACKEND] mode=real_isaac scene=isaac_office_min root=/isaac-sim`
- `[TOPO_EVAL_SUMMARY_V22_ISAAC] backend=real_isaac condition=gt_pose runs=1 success=0 avg_steps=120.00 avg_loc_conf=0.754`
- `[TOPO_EVAL_SUMMARY_V22_ISAAC] backend=real_isaac condition=est_pose_noise runs=1 success=0 avg_steps=120.00 avg_loc_conf=0.754`
- `[TOPO_DEBUG_INDEX_OK] path=/home/peng/DualVLN/runs/topo_mvp/v22_isaac/debug_runs/index.html`

## Notes
- v22 objective was real backend bring-up + debug observability, not semantic success target.
- Debug reports are generated per captured failing run under:
  - `runs/topo_mvp/v22_isaac/debug_runs/gt_pose/*/report/index.html`
  - `runs/topo_mvp/v22_isaac/debug_runs/est_pose_noise/*/report/index.html`

## ZIP Hygiene
- Packaging script runs extension filter and writes zip check.
- Expected anchor at package time:
  - `[ZIP_CHECK_OK] bad_count=0 path=forGPT/v22_isaac/ZIP_CHECK.txt`
  - `[PROOF_ZIP_OK] bad_count=0 zip=/home/peng/DualVLN/forGPT/topo_mvp_PROOF_v22_isaac.zip`
