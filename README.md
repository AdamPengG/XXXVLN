# DualVLN TopoGraph (NO-TRAINING) + Isaac Backend

This repo contains the NO-TRAINING TopoGraph navigation pipeline with Habitat and Isaac backends. ActionHead/BC training is intentionally excluded from this branch.

## Habitat (v20) quick run
```bash
bash scripts/topo_eval_suite_v20.sh
```
Debug UI:
```bash
python3 -m http.server 8000 --directory runs/topo_mvp/v20/debug_runs
```
Open: `http://127.0.0.1:8000/index.html`

## Isaac (v24) quick run
```bash
bash scripts/topo_eval_suite_v24_isaac.sh
```
Debug UI:
```bash
python3 -m http.server 8000 --directory runs/topo_mvp/v24_isaac/debug_runs
```
Open: `http://127.0.0.1:8000/index.html`

## Notes
- This branch excludes training, checkpoints, and proof zip artifacts.
- Isaac uses a grid A* local planner + waypoint-follow controller for bring-up.
