import heapq
from typing import Dict, List


def dijkstra_path(edges: Dict[int, Dict[int, float]], start: int, goal: int) -> List[int]:
    if start == goal:
        return [start]

    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v, w in edges.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if goal not in dist:
        return [start]

    path = [goal]
    cur = goal
    while cur in prev:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path
