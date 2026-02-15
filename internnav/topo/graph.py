import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TopoNode:
    node_id: int
    step_idx: int
    position: List[float]
    rotation: List[float]
    embed_idx: int
    timestamp: float
    rgb_path: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class TopoGraph:
    def __init__(self):
        self.nodes: Dict[int, TopoNode] = {}
        self.edges: Dict[int, Dict[int, float]] = {}
        self.edge_types: Dict[Tuple[int, int], str] = {}
        self.meta: Dict[str, str] = {}

    def add_node(
        self,
        node_id: int,
        step_idx: int,
        position: List[float],
        rotation: List[float],
        embed_idx: int,
        rgb_path: Optional[str] = None,
    ) -> TopoNode:
        node = TopoNode(
            node_id=node_id,
            step_idx=step_idx,
            position=position,
            rotation=rotation,
            embed_idx=embed_idx,
            timestamp=time.time(),
            rgb_path=rgb_path,
        )
        self.nodes[node_id] = node
        if node_id not in self.edges:
            self.edges[node_id] = {}
        return node

    def add_edge(
        self,
        a: int,
        b: int,
        weight: float,
        bidir: bool = True,
        edge_type: str = "temporal",
    ) -> None:
        self.edges.setdefault(a, {})[b] = float(weight)
        if bidir:
            self.edges.setdefault(b, {})[a] = float(weight)
        key = (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))
        if key not in self.edge_types:
            self.edge_types[key] = str(edge_type)

    def to_dict(self) -> Dict:
        return {
            "meta": self.meta,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [
                {
                    "u": u,
                    "v": v,
                    "w": w,
                    "edge_type": self.edge_types.get((u, v), "temporal"),
                }
                for u, nbrs in self.edges.items()
                for v, w in nbrs.items()
                if u <= v
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TopoGraph":
        g = cls()
        g.meta = data.get("meta", {})
        for n in data.get("nodes", []):
            node = TopoNode(**n)
            g.nodes[node.node_id] = node
            g.edges.setdefault(node.node_id, {})
        for e in data.get("edges", []):
            g.add_edge(
                int(e["u"]),
                int(e["v"]),
                float(e["w"]),
                bidir=True,
                edge_type=str(e.get("edge_type", "temporal")),
            )
        return g

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "TopoGraph":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def save_embeddings(path: str, embeds: List[np.ndarray]) -> None:
    if not embeds:
        arr = np.zeros((0, 1), dtype=np.float16)
    else:
        arr = np.stack(embeds, axis=0)
    np.save(path, arr)


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


def l2_distance(a: List[float], b: List[float]) -> float:
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a_np - b_np))
