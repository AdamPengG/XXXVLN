# V23 Isaac Diagnosis

## V22 failure root cause (based on v23 sanity check)
- Kinematics are consistent: `forward_step_mean=0.125m` and `heading_alignment_cos=1.0` indicate forward motion matches yaw heading and unit scale is correct.
- Turn increments are stable: `yaw_left_mean≈+0.131rad`, `yaw_right_mean≈-0.131rad`, consistent with configured `turn_rate_degps=15` and `dt_action=0.5`.
- Conclusion: v22 failures are not from unit/coordinate sign mistakes. Primary cause is navigation/goal mismatch in the Isaac bring-up (synthetic topo graph vs real/sim pose), not controller kinematics.

## What v23 adds
- Isaac sanity gate to detect unit/coordinate issues early.
- Controller units and frame anchors for auditability.
- Debug capture stride=1 with placeholder frames to ensure UI continuity.
