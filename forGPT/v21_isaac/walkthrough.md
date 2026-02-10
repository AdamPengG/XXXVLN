# Topo MVP v21 Isaac Backend Bring-up

## Scope
- Added simulator backend abstraction (`SimBackend`) with Habitat adapter and Isaac adapter.
- Preserved existing Habitat v20 pipeline behavior; no changes to ActionHead/BC training chain.
- Added `backend=isaac` runner path and Isaac suite script.
- Reused v20 debug capture + UI flow for Isaac runs.

## Important runtime note
- Current environment has no Isaac Python package (`omni` not importable).
- `IsaacSimBackend` auto-falls back to Habitat while preserving backend selection path and output schema.
- Logs include backend mode anchor: `mode=fallback_habitat`.

## v21 run outputs
- `runs/topo_mvp/v21_isaac/v21_isaac_summary.json`
- `runs/topo_mvp/v21_isaac/debug_runs/index.html`
- `runs/topo_mvp/v21_isaac/debug_runs/<condition>/<run_id>/report/index.html`

## Key anchors
- `[TOPO_EVAL_SUMMARY_V21_ISAAC] condition=gt_pose runs=2 success=1 ...`
- `[TOPO_EVAL_SUMMARY_V21_ISAAC] condition=est_pose_noise runs=2 success=1 ...`
- `[TOPO_DEBUG_INDEX_OK] path=/home/peng/DualVLN/runs/topo_mvp/v21_isaac/debug_runs/index.html`

## UI
- `python3 -m http.server 8000 --directory runs/topo_mvp/v21_isaac/debug_runs`
- open `http://127.0.0.1:8000/index.html`

## Packaging
- Script: `scripts/package_topo_mvp_proof_isaac.sh`
- Output zip: `/home/peng/DualVLN/forGPT/topo_mvp_PROOF_v21_isaac.zip`
- Weight exclusion enforced via ZIP checks.
