import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class PoseNode:
    node_id: int
    x: float
    z: float
    yaw: float
    gt_x: float
    gt_z: float
    gt_yaw: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PoseEdge:
    i: int
    j: int
    dx: float
    dz: float
    dyaw: float
    edge_type: str
    weight: float = 1.0
    sim: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class PoseGraphLite:
    def __init__(self, loop_robust: str = "huber", loop_robust_scale: float = 0.75):
        self.nodes: Dict[int, PoseNode] = {}
        self.edges: List[PoseEdge] = []
        self._optimized_poses: Optional[Dict[int, Tuple[float, float, float]]] = None
        self.loop_robust = str(loop_robust).strip().lower()
        self.loop_robust_scale = max(1e-6, float(loop_robust_scale))

    def _loop_robust_multiplier(self, edge_residual: np.ndarray) -> float:
        if self.loop_robust in ("none", "off", "disable", "disabled"):
            return 1.0
        sq_norm = float(np.dot(edge_residual, edge_residual))
        if sq_norm <= 0.0:
            return 1.0
        scale = max(1e-6, float(self.loop_robust_scale))
        if self.loop_robust == "cauchy":
            # Cauchy IRLS weight in sqrt form.
            return float(1.0 / math.sqrt(1.0 + sq_norm / (scale * scale)))
        # Default: Huber (piecewise-quadratic/linear).
        r = math.sqrt(sq_norm)
        if r <= scale:
            return 1.0
        return float(scale / max(scale, r))

    def add_node(
        self,
        node_id: int,
        x: float,
        z: float,
        yaw: float,
        gt_x: float,
        gt_z: float,
        gt_yaw: float,
    ) -> None:
        self.nodes[int(node_id)] = PoseNode(
            node_id=int(node_id),
            x=float(x),
            z=float(z),
            yaw=wrap_angle(float(yaw)),
            gt_x=float(gt_x),
            gt_z=float(gt_z),
            gt_yaw=wrap_angle(float(gt_yaw)),
        )

    def add_edge(
        self,
        i: int,
        j: int,
        dx: float,
        dz: float,
        dyaw: float,
        edge_type: str = "sequential",
        weight: float = 1.0,
        sim: float = 0.0,
    ) -> None:
        self.edges.append(
            PoseEdge(
                i=int(i),
                j=int(j),
                dx=float(dx),
                dz=float(dz),
                dyaw=wrap_angle(float(dyaw)),
                edge_type=str(edge_type),
                weight=float(weight),
                sim=float(sim),
            )
        )

    def _sorted_ids(self) -> List[int]:
        return sorted(self.nodes.keys())

    def get_pose(self, node_id: int, optimized: bool = True) -> Tuple[float, float, float]:
        if optimized and self._optimized_poses is not None and int(node_id) in self._optimized_poses:
            return self._optimized_poses[int(node_id)]
        n = self.nodes[int(node_id)]
        return float(n.x), float(n.z), float(n.yaw)

    def _state_to_pose_map(self, x: np.ndarray, ids: List[int]) -> Dict[int, Tuple[float, float, float]]:
        anchor_id = ids[0]
        anchor = self.nodes[anchor_id]
        out = {int(anchor_id): (float(anchor.x), float(anchor.z), float(anchor.yaw))}
        k = 0
        for nid in ids[1:]:
            out[int(nid)] = (
                float(x[k]),
                float(x[k + 1]),
                wrap_angle(float(x[k + 2])),
            )
            k += 3
        return out

    def _residual_vector(self, x: np.ndarray, ids: List[int]) -> np.ndarray:
        poses = self._state_to_pose_map(x, ids)
        res = []
        for e in self.edges:
            if e.i not in poses or e.j not in poses:
                continue
            xi, zi, yawi = poses[e.i]
            xj, zj, yawj = poses[e.j]
            w = max(1e-6, float(e.weight))
            # Keep residual simple and stable: world-frame delta + yaw delta.
            rx = (xj - xi) - float(e.dx)
            rz = (zj - zi) - float(e.dz)
            ry = wrap_angle((yawj - yawi) - float(e.dyaw))
            yaw_scale = 0.20
            s = math.sqrt(w)
            edge_res = np.array([s * rx, s * rz, s * yaw_scale * ry], dtype=np.float64)
            if str(e.edge_type).lower() == "loop":
                edge_res *= float(self._loop_robust_multiplier(edge_res))
            res.extend(edge_res.tolist())
        # Add weak priors so optimization is well-conditioned.
        prior_w = 0.02
        s_prior = math.sqrt(prior_w)
        for nid in ids[1:]:
            x0, z0, yaw0 = self.nodes[nid].x, self.nodes[nid].z, self.nodes[nid].yaw
            xx, zz, yy = poses[nid]
            res.extend([s_prior * (xx - x0), s_prior * (zz - z0), s_prior * 0.2 * wrap_angle(yy - yaw0)])
        return np.array(res, dtype=np.float64)

    def residual_rms(self, optimized: bool = False) -> float:
        ids = self._sorted_ids()
        if len(ids) <= 1:
            return 0.0
        if optimized and self._optimized_poses is not None:
            vals = []
            for nid in ids[1:]:
                x, z, yaw = self._optimized_poses[nid]
                vals.extend([x, z, yaw])
            xvec = np.array(vals, dtype=np.float64)
        else:
            vals = []
            for nid in ids[1:]:
                n = self.nodes[nid]
                vals.extend([n.x, n.z, n.yaw])
            xvec = np.array(vals, dtype=np.float64)
        r = self._residual_vector(xvec, ids)
        if r.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(r * r)))

    def optimize(self, max_nfev: int = 100) -> Tuple[float, float]:
        ids = self._sorted_ids()
        if len(ids) <= 1 or len(self.edges) == 0:
            before = self.residual_rms(optimized=False)
            self._optimized_poses = {
                int(nid): (float(self.nodes[nid].x), float(self.nodes[nid].z), float(self.nodes[nid].yaw))
                for nid in ids
            }
            return before, before

        x0 = []
        for nid in ids[1:]:
            n = self.nodes[nid]
            x0.extend([n.x, n.z, n.yaw])
        x0 = np.array(x0, dtype=np.float64)
        before = self.residual_rms(optimized=False)

        if least_squares is not None:
            sol = least_squares(
                fun=lambda xx: self._residual_vector(xx, ids),
                x0=x0,
                max_nfev=int(max_nfev),
                method="trf",
            )
            x_opt = sol.x
        else:
            # Fallback: no optimization library available.
            x_opt = x0

        self._optimized_poses = self._state_to_pose_map(x_opt, ids)
        after = self.residual_rms(optimized=True)
        return before, after

    def to_dict(self, optimized: bool = False) -> Dict:
        nodes = []
        for nid in self._sorted_ids():
            n = self.nodes[nid]
            if optimized and self._optimized_poses is not None and nid in self._optimized_poses:
                x, z, yaw = self._optimized_poses[nid]
            else:
                x, z, yaw = n.x, n.z, n.yaw
            row = n.to_dict()
            row["x"] = float(x)
            row["z"] = float(z)
            row["yaw"] = float(yaw)
            nodes.append(row)
        return {
            "nodes": nodes,
            "edges": [e.to_dict() for e in self.edges],
            "meta": {
                "loop_robust": str(self.loop_robust),
                "loop_robust_scale": float(self.loop_robust_scale),
            },
        }
