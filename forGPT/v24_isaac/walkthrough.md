# v24 Isaac bring-up walkthrough

Commands:
- `bash scripts/topo_eval_suite_v24_isaac.sh`
- `bash scripts/package_topo_mvp_proof_v24_isaac.sh`

Expected anchors in logs:
- `[ISAAC_ENV_OK] ...`
- `[ISAAC_BACKEND] mode=real_isaac ...`
- `[ISAAC_SANITY] ...`
- `[ISAAC_LOCAL_PLANNER] ... ok=1 ...`
- `[ISAAC_ORACLE] mode=grid_astar available=1 success=1 ...`
- `[TOPO_EVAL_SUMMARY_V24_ISAAC] ... success=1 ...`
- `[TOPO_DEBUG_CAPTURE_CFG] stride=1 placeholder=1`
- `[TOPO_DEBUG_INDEX_OK] ...`

Notes:
- Suite uses `TOPO_DEBUG_ORACLE_ON_FAIL=2` to force oracle probe on successes.
- `ISAAC_MINIMAL=1` + `ISAAC_SKIP_WORLD_STEP=1` keep Isaac initialization fast while still running under Isaac python.
