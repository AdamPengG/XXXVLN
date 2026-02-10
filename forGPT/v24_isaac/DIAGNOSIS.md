# v24 Isaac diagnosis

Root cause (v22 failure):
- Coordinate/units ambiguity in bearing/turn thresholds and lack of a local waypoint planner under Isaac led to drift and no-goal behavior.
- Debug capture in v23 showed missing per-step frames and uncertain motion alignment.

v24 fixes:
- Added grid A* local planner + waypoint-follow controller for Isaac; logs `[ISAAC_LOCAL_PLANNER]` and `[ISAAC_ORACLE]`.
- Enforced radian units and explicit frame anchors `[TOPO_CTRL_UNITS]` + `[TOPO_CTRL_FRAME]`.
- Debug capture now stride=1 with placeholders to keep UI intact.

Evidence:
- `[ISAAC_SANITY]` shows forward mean and yaw deltas with high heading alignment.
- `[ISAAC_ORACLE]` confirms A* path is available and reachable.
