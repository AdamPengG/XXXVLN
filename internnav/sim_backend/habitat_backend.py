from __future__ import annotations

import math
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import quaternion
from habitat_sim import ShortestPath

from internnav.configs.evaluator import EnvCfg
from internnav.env.habitat_env import HabitatEnv
from internnav.sim_backend.base import Obs, Pose, SimBackend
from internnav.topo.controller import FollowerController


def _agent_yaw_from_quat(quat_obj) -> float:
    rot = quaternion.as_rotation_matrix(quat_obj)
    forward = rot @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return float(math.atan2(float(forward[0]), float(-forward[2])))


def _extract_collision_flag(info: Optional[Dict]) -> Optional[bool]:
    if not isinstance(info, dict):
        return None
    if "collided" in info:
        try:
            return bool(info.get("collided"))
        except Exception:
            return None
    col = info.get("collisions")
    if isinstance(col, dict) and "is_collision" in col:
        try:
            return bool(col.get("is_collision"))
        except Exception:
            return None
    return None


def _geodesic_distance(sim, start_pos: np.ndarray, goal_pos: np.ndarray) -> float:
    try:
        sp = ShortestPath()
        sp.requested_start = np.asarray(start_pos, dtype=np.float32)
        sp.requested_end = np.asarray(goal_pos, dtype=np.float32)
        ok = bool(sim.pathfinder.find_path(sp))
        if not ok:
            return float("inf")
        return float(sp.geodesic_distance)
    except Exception:
        return float("inf")


class HabitatSimBackend(SimBackend):
    def __init__(
        self,
        config_path: str,
        out_dir: str,
    ) -> None:
        from habitat_baselines.config.default import get_config as get_habitat_config

        habitat_config = get_habitat_config(config_path)
        env_cfg = EnvCfg(
            env_type="habitat",
            env_settings={
                "habitat_config": habitat_config,
                "output_path": out_dir,
            },
        )
        self._env = HabitatEnv(env_cfg)
        self._last_obs_raw: Optional[Dict[str, object]] = None
        self._last_info: Dict[str, object] = {}
        self._last_scene_id: str = ""
        self._goal_geodesic: Optional[float] = None

    def _set_scene(self, scene_id: str, start_offset: int = 0) -> None:
        if scene_id == self._last_scene_id and len(self._env.episodes) > 0:
            return
        filtered = [ep for ep in self._env.episodes if str(scene_id) in str(ep.scene_id)]
        self._env.episodes = filtered
        if len(filtered) > 0:
            self._env._current_episode_index = int(start_offset) % len(filtered)
        else:
            self._env._current_episode_index = 0
        self._env.is_running = len(filtered) > 0
        self._last_scene_id = str(scene_id)

    def _make_obs(self, raw_obs: Dict[str, object], info: Optional[Dict[str, object]]) -> Obs:
        state = self._env._env.sim.get_agent_state()
        rgb = np.asarray(raw_obs.get("rgb"), dtype=np.uint8)
        depth = raw_obs.get("depth")
        if depth is not None:
            depth = np.asarray(depth, dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
        pose = Pose(
            x=float(state.position[0]),
            y=float(state.position[1]),
            z=float(state.position[2]),
            yaw=float(_agent_yaw_from_quat(state.rotation)),
        )
        collision = _extract_collision_flag(info)
        return Obs(
            rgb=rgb,
            depth=depth,
            pose=pose,
            collision=collision,
            timestamp=time.time(),
            info=dict(info or {}),
        )

    def reset(self, scene_id: str, start_spec: Dict[str, object]) -> Obs:
        start_offset = int(start_spec.get("start_offset", 0))
        self._set_scene(scene_id=str(scene_id), start_offset=start_offset)
        obs = self._env.reset()
        if obs is None or not self._env.is_running:
            raise RuntimeError(f"No episode available for scene_id={scene_id}")

        kidnap_start = bool(int(start_spec.get("kidnap_start", 0)))
        if kidnap_start:
            sim = self._env._env.sim
            seed = int(start_spec.get("kidnap_seed", 7))
            rng = np.random.default_rng(seed)
            min_dist = float(start_spec.get("kidnap_min_nearest_node_dist", 0.5))
            max_dist = float(start_spec.get("kidnap_max_nearest_node_dist", 4.0))
            near_nodes = np.asarray(start_spec.get("reference_nodes_xyz", []), dtype=np.float32)
            placed = False
            for _ in range(80):
                p = np.array(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
                if near_nodes.ndim == 2 and near_nodes.shape[0] > 0:
                    d = np.linalg.norm(near_nodes - p[None, :], axis=1)
                    nn = float(np.min(d))
                    if nn < min_dist or nn > max_dist:
                        continue
                yaw = float(rng.uniform(-math.pi, math.pi))
                rot = np.quaternion(float(math.cos(yaw / 2.0)), 0.0, float(math.sin(yaw / 2.0)), 0.0)
                sim.set_agent_state(p, rot, reset_sensors=False)
                try:
                    obs = sim.get_sensor_observations()
                except Exception:
                    pass
                placed = True
                break
            if not placed:
                random.seed(seed)

        self._last_obs_raw = dict(obs)
        self._last_info = {}
        self._goal_geodesic = None
        return self._make_obs(self._last_obs_raw, self._last_info)

    def step(self, action_discrete: int) -> Tuple[Obs, bool, Dict[str, object]]:
        obs, _reward, done, info = self._env.step(int(action_discrete))
        if obs is None:
            obs = self._last_obs_raw or {}
        self._last_obs_raw = dict(obs)
        self._last_info = dict(info or {})
        if isinstance(info, dict) and "distance_to_goal" in info:
            try:
                self._goal_geodesic = float(info["distance_to_goal"])
            except Exception:
                self._goal_geodesic = None
        return self._make_obs(self._last_obs_raw, self._last_info), bool(done), self._last_info

    def close(self) -> None:
        self._env.close()

    def get_pose(self) -> Pose:
        state = self._env._env.sim.get_agent_state()
        return Pose(
            x=float(state.position[0]),
            y=float(state.position[1]),
            z=float(state.position[2]),
            yaw=float(_agent_yaw_from_quat(state.rotation)),
        )

    def get_rgb(self) -> np.ndarray:
        if self._last_obs_raw is None:
            raise RuntimeError("No observation available yet.")
        return np.asarray(self._last_obs_raw.get("rgb"), dtype=np.uint8)

    def get_depth(self) -> Optional[np.ndarray]:
        if self._last_obs_raw is None:
            return None
        depth = self._last_obs_raw.get("depth")
        if depth is None:
            return None
        arr = np.asarray(depth, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr

    def get_collision_flag(self) -> Optional[bool]:
        return _extract_collision_flag(self._last_info)

    def get_goal_geodesic(self) -> Optional[float]:
        return self._goal_geodesic

    def oracle_probe(
        self,
        goal_position: np.ndarray,
        max_steps: int,
        success_dist: float,
    ) -> Dict[str, object]:
        sim = self._env._env.sim
        result = {
            "oracle_available": 0,
            "oracle_success": 0,
            "oracle_geodesic_start": float("inf"),
            "oracle_geodesic_final": float("inf"),
            "oracle_steps_used": 0,
        }
        try:
            backup = sim.get_agent_state()
            start = np.asarray(backup.position, dtype=np.float32)
            goal = np.asarray(goal_position, dtype=np.float32)
            result["oracle_geodesic_start"] = _geodesic_distance(sim, start, goal)
            follower = FollowerController(sim)
            steps = 0
            while steps < int(max_steps):
                st = sim.get_agent_state()
                cur = np.asarray(st.position, dtype=np.float32)
                if float(np.linalg.norm(cur - goal)) <= float(success_dist):
                    result["oracle_success"] = 1
                    break
                act = int(follower.act(goal))
                if act == 0:
                    break
                sim.step(act)
                steps += 1
            st2 = sim.get_agent_state()
            fin = np.asarray(st2.position, dtype=np.float32)
            if float(np.linalg.norm(fin - goal)) <= float(success_dist):
                result["oracle_success"] = 1
            result["oracle_geodesic_final"] = _geodesic_distance(sim, fin, goal)
            result["oracle_steps_used"] = int(steps)
            result["oracle_available"] = 1
            sim.set_agent_state(backup.position, backup.rotation, reset_sensors=False)
        except Exception:
            result["oracle_available"] = 0
        return result

