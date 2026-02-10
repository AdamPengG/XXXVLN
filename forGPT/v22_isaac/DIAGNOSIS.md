# v22 Diagnosis

## Why v21 failed real Isaac requirement
- `backend=isaac` still executed under system/conda python, so `omni` import failed.
- Backend class silently fell back to Habitat by default.

## v22 fixes
- Added explicit Isaac environment discovery + wrapper execution via Isaac `python.sh`.
- Added strict behavior: fallback disabled by default (`ISAAC_BACKEND_ALLOW_FALLBACK=0`).
- Added explicit failure anchor `[ISAAC_BACKEND_ERROR]` for missing omni.
- Added real Isaac backend mode anchor `[ISAAC_BACKEND] mode=real_isaac ...`.

## Runtime outcome
- Real Isaac path is now active and observable in logs.
- Current tiny bring-up run (1 run per condition) reached max steps; debug UI artifacts exist for forensics.
