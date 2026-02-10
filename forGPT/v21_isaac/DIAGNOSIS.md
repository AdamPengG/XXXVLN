# v21 Isaac Backend Diagnosis

## What worked
- Backend abstraction module compiles and is wired.
- Isaac suite runs end-to-end in this environment via fallback mode.
- Failure-only debug capture produces per-run traces and frame strips.
- Offline report/index HTML now generates correctly after fixing graph edge parsing (`u/v` fallback).

## What failed initially and fix
- Initial UI report build failed because build graph edge schema used `u/v` while report builder expected `source/target`.
- Fixed in `scripts/topo_debug_ui_build_report.py` by supporting both schemas.
- Initial index generation call used wrong flag (`--debug_root`); fixed to `--root` in `scripts/topo_eval_suite_v21_isaac.sh`.

## Isaac-specific status
- `omni.isaac` is not installed in current runtime.
- `IsaacSimBackend` enters explicit fallback mode to Habitat and logs reason.
- This preserves API and debug/report flow now; real Isaac stage/robot binding can be enabled later without changing runner contract.
