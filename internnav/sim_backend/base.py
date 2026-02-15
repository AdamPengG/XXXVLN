from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass
class Obs:
    rgb: np.ndarray
    depth: Optional[np.ndarray]
    pose: Pose
    collision: Optional[bool]
    timestamp: float
    info: Dict[str, object] = field(default_factory=dict)


class SimBackend(ABC):
    """Unified simulator backend API for topo navigation."""

    @abstractmethod
    def reset(self, scene_id: str, start_spec: Dict[str, object]) -> Obs:
        raise NotImplementedError

    @abstractmethod
    def step(self, action_discrete: int) -> Tuple[Obs, bool, Dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_pose(self) -> Pose:
        raise NotImplementedError

    @abstractmethod
    def get_rgb(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_depth(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def get_collision_flag(self) -> Optional[bool]:
        raise NotImplementedError

    @abstractmethod
    def get_goal_geodesic(self) -> Optional[float]:
        raise NotImplementedError

    def oracle_probe(
        self,
        goal_position: np.ndarray,
        max_steps: int,
        success_dist: float,
    ) -> Dict[str, object]:
        """Optional oracle probe. Backends can override."""
        return {
            "oracle_available": 0,
            "oracle_success": 0,
            "oracle_geodesic_start": None,
            "oracle_geodesic_final": None,
            "oracle_steps_used": 0,
        }

