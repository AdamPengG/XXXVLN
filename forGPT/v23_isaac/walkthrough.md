# Topo MVP v23 Isaac bring-up

## What ran
- `bash scripts/topo_eval_suite_v23_isaac.sh`
- `bash scripts/package_topo_mvp_proof_v23_isaac.sh`

## Key behavior
- Isaac sanity check runs first and gates the suite.
- Debug capture forced stride=1 with placeholders for missing RGB.
- UI reports are generated under `runs/topo_mvp/v23_isaac/debug_runs/`.

## Anchors (from logs)
- `[ISAAC_ENV_OK] ...`
- `[ISAAC_BACKEND] mode=real_isaac ...`
- `[ISAAC_SANITY] ...`
- `[TOPO_DEBUG_CAPTURE_CFG] stride=1 placeholder=1`
- `[TOPO_DEBUG_INDEX_OK] path=...`

## Notes
- Isaac bring-up currently uses a minimal synthetic topo graph for fast validation.
