from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def build_occupancy_grid(
    bounds: List[float],
    obstacles: List[Dict[str, float]],
    meters_per_cell: float = 0.25,
) -> Tuple[np.ndarray, Dict[str, float]]:
    xmin, xmax, zmin, zmax = [float(v) for v in bounds]
    mpc = float(max(0.05, meters_per_cell))
    width = int(math.ceil((xmax - xmin) / mpc)) + 1
    height = int(math.ceil((zmax - zmin) / mpc)) + 1
    grid = np.zeros((height, width), dtype=np.uint8)

    for o in obstacles:
        ox = float(o.get("x", 0.0))
        oz = float(o.get("z", 0.0))
        r = float(o.get("r", o.get("radius", 0.5)))
        minx = int(_clamp((ox - r - xmin) / mpc, 0, width - 1))
        maxx = int(_clamp((ox + r - xmin) / mpc, 0, width - 1))
        minz = int(_clamp((oz - r - zmin) / mpc, 0, height - 1))
        maxz = int(_clamp((oz + r - zmin) / mpc, 0, height - 1))
        for iz in range(minz, maxz + 1):
            for ix in range(minx, maxx + 1):
                x = xmin + ix * mpc
                z = zmin + iz * mpc
                if (x - ox) ** 2 + (z - oz) ** 2 <= r ** 2:
                    grid[iz, ix] = 1
    meta = {
        "xmin": xmin,
        "xmax": xmax,
        "zmin": zmin,
        "zmax": zmax,
        "meters_per_cell": mpc,
    }
    return grid, meta


def _to_idx(x: float, z: float, meta: Dict[str, float]) -> Tuple[int, int]:
    mpc = float(meta["meters_per_cell"])
    ix = int(round((x - float(meta["xmin"])) / mpc))
    iz = int(round((z - float(meta["zmin"])) / mpc))
    return ix, iz


def _to_pos(ix: int, iz: int, meta: Dict[str, float]) -> Tuple[float, float]:
    mpc = float(meta["meters_per_cell"])
    x = float(meta["xmin"]) + ix * mpc
    z = float(meta["zmin"]) + iz * mpc
    return x, z


def astar_path(
    grid: np.ndarray,
    meta: Dict[str, float],
    start: np.ndarray,
    goal: np.ndarray,
    max_nodes: int = 20000,
) -> Optional[List[Tuple[float, float]]]:
    import heapq

    h, w = grid.shape
    sx, sz = float(start[0]), float(start[2])
    gx, gz = float(goal[0]), float(goal[2])
    s_idx = _to_idx(sx, sz, meta)
    g_idx = _to_idx(gx, gz, meta)
    if s_idx == g_idx:
        return [(sx, sz), (gx, gz)]

    def in_bounds(ix: int, iz: int) -> bool:
        return 0 <= ix < w and 0 <= iz < h

    def is_free(ix: int, iz: int) -> bool:
        return in_bounds(ix, iz) and grid[iz, ix] == 0

    if not is_free(*s_idx) or not is_free(*g_idx):
        return None

    def heur(ix: int, iz: int) -> float:
        x, z = _to_pos(ix, iz, meta)
        return math.hypot(x - gx, z - gz)

    open_heap = []
    heapq.heappush(open_heap, (heur(*s_idx), 0.0, s_idx))
    came_from = {}
    gscore = {s_idx: 0.0}
    visited = set()
    steps = 0
    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]
    while open_heap and steps < int(max_nodes):
        _, gcur, (ix, iz) = heapq.heappop(open_heap)
        if (ix, iz) in visited:
            continue
        visited.add((ix, iz))
        if (ix, iz) == g_idx:
            path = [(ix, iz)]
            cur = (ix, iz)
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            out = [(_to_pos(ix, iz, meta)) for ix, iz in path]
            return out
        for dx, dz, cost in neighbors:
            nix = ix + dx
            niz = iz + dz
            if not is_free(nix, niz):
                continue
            ng = gcur + cost * float(meta["meters_per_cell"])
            if ng < gscore.get((nix, niz), 1e12):
                gscore[(nix, niz)] = ng
                came_from[(nix, niz)] = (ix, iz)
                heapq.heappush(open_heap, (ng + heur(nix, niz), ng, (nix, niz)))
        steps += 1
    return None


def plan_grid_path(
    bounds: List[float],
    obstacles: List[Dict[str, float]],
    start: np.ndarray,
    goal: np.ndarray,
    meters_per_cell: float = 0.25,
    max_nodes: int = 20000,
) -> Tuple[bool, List[Tuple[float, float]], Dict[str, float]]:
    grid, meta = build_occupancy_grid(bounds=bounds, obstacles=obstacles, meters_per_cell=meters_per_cell)
    path = astar_path(grid=grid, meta=meta, start=start, goal=goal, max_nodes=max_nodes)
    if path is None:
        return False, [], meta
    return True, path, meta
