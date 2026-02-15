import math
from typing import Tuple

import numpy as np

try:
    import quaternion
except Exception:  # pragma: no cover - optional dependency
    quaternion = None

try:
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
except Exception:  # Habitat may not be available in some environments
    ShortestPathFollower = None


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def bearing_distance(
    curr_pos: np.ndarray, curr_rot: np.ndarray, target_pos: np.ndarray
) -> Tuple[float, float]:
    delta = np.array(target_pos, dtype=np.float32) - np.array(curr_pos, dtype=np.float32)
    delta[1] = 0.0
    dist = float(np.linalg.norm(delta[[0, 2]]))
    if dist < 1e-8:
        return 0.0, 0.0

    if hasattr(curr_rot, "w") and hasattr(curr_rot, "x"):
        qarr = np.array([float(curr_rot.w), float(curr_rot.x), float(curr_rot.y), float(curr_rot.z)], dtype=np.float32)
    else:
        qarr = np.array(curr_rot, dtype=np.float32).reshape(-1)
        if qarr.shape[0] < 4:
            qarr = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            qarr = qarr[:4]

    if quaternion is not None:
        try:
            q = np.quaternion(float(qarr[0]), float(qarr[1]), float(qarr[2]), float(qarr[3]))
            rot = quaternion.as_rotation_matrix(q)
        except Exception:
            rot = None
    else:
        rot = None
    if rot is None:
        w, x, y, z = [float(v) for v in qarr]
        n = math.sqrt(max(1e-12, w * w + x * x + y * y + z * z))
        w, x, y, z = w / n, x / n, y / n, z / n
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    forward = rot @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    right = rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    f_comp = float(np.dot(delta, forward))
    r_comp = float(np.dot(delta, right))
    bearing = _wrap_angle(math.atan2(r_comp, f_comp))
    return bearing, dist


def bearing_distance_to_action(
    bearing: float, dist: float, stop_thresh: float = 0.3, turn_thresh: float = 0.35
) -> Tuple[int, str]:
    if dist <= stop_thresh:
        return 0, "STOP"
    if abs(bearing) > turn_thresh:
        if bearing > 0:
            return 3, "RIGHT"
        return 2, "LEFT"
    return 1, "FORWARD"


def map_habitat_action(action) -> int:
    if isinstance(action, int):
        if action == 0:
            return 0
        if action == 1:
            return 1
        if action == 2:
            return 2
        if action == 3:
            return 3
        return 0
    act_name = action.name if hasattr(action, "name") else str(action)
    act_name = act_name.upper()
    if "STOP" in act_name:
        return 0
    if "FORWARD" in act_name:
        return 1
    if "LEFT" in act_name:
        return 2
    if "RIGHT" in act_name:
        return 3
    return 0


class FollowerController:
    def __init__(self, sim, step_size: float = 0.25, stop_on_goal: bool = False):
        if ShortestPathFollower is None:
            raise RuntimeError("ShortestPathFollower is unavailable (habitat not importable).")
        self.follower = ShortestPathFollower(sim, step_size, stop_on_goal)

    def act(self, target_pos: np.ndarray) -> int:
        act = self.follower.get_next_action(target_pos)
        return map_habitat_action(act)


class BearingController:
    def __init__(
        self,
        stop_thresh: float = 0.3,
        turn_high: float = 0.45,
        turn_low: float = 0.25,
        stuck_move_thresh: float = 0.02,
        stuck_forward_limit: int = 3,
        recovery_turn_steps: int = 2,
    ):
        self.stop_thresh = stop_thresh
        self.turn_high = turn_high
        self.turn_low = turn_low
        self.turning = False
        self.stuck_move_thresh = stuck_move_thresh
        self.stuck_forward_limit = stuck_forward_limit
        self.recovery_turn_steps = recovery_turn_steps
        self.force_turn_steps = 0
        self.forward_stuck_count = 0
        self.turn_dir = 0
        self.consecutive_turns = 0
        self.max_consecutive_turns = 6
        self.force_forward_steps = 0

    def decide(self, bearing: float, dist: float) -> Tuple[int, str]:
        if dist <= self.stop_thresh:
            return 0, "STOP"
        if self.force_forward_steps > 0:
            self.force_forward_steps -= 1
            return 1, "FORWARD"
        if self.force_turn_steps > 0:
            self.force_turn_steps -= 1
            return 2, "LEFT"
        if self.turning:
            if abs(bearing) < self.turn_low:
                self.turning = False
                self.turn_dir = 0
                self.consecutive_turns = 0
            else:
                self.consecutive_turns += 1
                if self.consecutive_turns >= self.max_consecutive_turns:
                    self.turning = False
                    self.turn_dir = 0
                    self.consecutive_turns = 0
                    self.force_forward_steps = 1
                    return 1, "FORWARD"
                return self.turn_dir, "LEFT" if self.turn_dir == 2 else "RIGHT"
        else:
            if abs(bearing) > self.turn_high:
                self.turning = True
                # Habitat VLN yaw convention: negative bearing should turn LEFT.
                self.turn_dir = 2 if bearing < 0 else 3
                self.consecutive_turns = 1
                return self.turn_dir, "LEFT" if self.turn_dir == 2 else "RIGHT"
        return 1, "FORWARD"

    def update_after_step(self, action: int, moved_dist: float) -> None:
        if action == 1:
            if moved_dist < self.stuck_move_thresh:
                self.forward_stuck_count += 1
            else:
                self.forward_stuck_count = 0
        else:
            self.forward_stuck_count = 0
        if self.forward_stuck_count >= self.stuck_forward_limit:
            self.force_turn_steps = self.recovery_turn_steps
            self.forward_stuck_count = 0

    def stuck_state(self) -> int:
        return int(self.force_turn_steps > 0 or self.force_forward_steps > 0)
