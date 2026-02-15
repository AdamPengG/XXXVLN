"""v25 Goal Resolver — maps free-text language queries to catalog goal poses.

No external dependencies beyond PyYAML (already required by the project).
Deterministic matching: exact goal_id → exact alias → substring → token overlap.
"""
from __future__ import annotations

import json
import re
import string
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GoalEntry:
    goal_id: str
    goal_type: str  # "pose" | "object" | "room"
    pose: Dict[str, float]  # {x, y, z} minimum; yaw optional
    language_aliases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class GoalMatch:
    goal: GoalEntry
    score: float
    matched_alias: str
    match_method: str  # "exact_id" | "exact_alias" | "substring" | "token_overlap"


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[" + re.escape(string.punctuation) + r"]")
_MULTI_WS = re.compile(r"\s+")


def _normalise(text: str) -> str:
    t = text.lower().strip()
    t = _PUNCT_RE.sub(" ", t)
    return _MULTI_WS.sub(" ", t).strip()


def _tokenise(text: str) -> List[str]:
    return _normalise(text).split()


# ---------------------------------------------------------------------------
# Scoring helpers (pure Python, no deps)
# ---------------------------------------------------------------------------

def _token_overlap(query_tokens: List[str], alias_tokens: List[str]) -> float:
    """Overlap coefficient: |intersection| / min(|A|, |B|)."""
    if not query_tokens or not alias_tokens:
        return 0.0
    q_set = set(query_tokens)
    a_set = set(alias_tokens)
    inter = len(q_set & a_set)
    denom = min(len(q_set), len(a_set))
    return float(inter) / float(max(1, denom))


def _score_alias(query_norm: str, query_tokens: List[str],
                 alias: str, goal_id: str) -> Tuple[float, str]:
    """Return (score, match_method) for one alias against the query."""
    alias_norm = _normalise(alias)

    # 1) exact goal_id match (perfect)
    if query_norm == _normalise(goal_id):
        return 1.0, "exact_id"

    # 2) exact alias match
    if query_norm == alias_norm:
        return 0.95, "exact_alias"

    # 3) substring / contains
    if alias_norm in query_norm or query_norm in alias_norm:
        longer = max(len(query_norm), len(alias_norm))
        shorter = min(len(query_norm), len(alias_norm))
        return 0.60 + 0.30 * (shorter / max(1, longer)), "substring"

    # 4) token overlap
    alias_tokens = _tokenise(alias)
    overlap = _token_overlap(query_tokens, alias_tokens)
    if overlap > 0.0:
        return min(0.55, overlap * 0.70), "token_overlap"

    return 0.0, "token_overlap"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class GoalCatalog:
    """Load and query the goal catalog YAML for a given scene."""

    def __init__(self, catalog_path: str, scene_id: str) -> None:
        self.catalog_path = str(catalog_path)
        self.scene_id = str(scene_id)
        self._goals: List[GoalEntry] = []
        self._load()

    # -- loading ---------------------------------------------------------

    def _load(self) -> None:
        path = Path(self.catalog_path)
        if not path.is_file():
            raise FileNotFoundError(f"Goal catalog not found: {self.catalog_path}")
        raw: Dict[str, Any] = {}
        try:
            import yaml  # type: ignore
            raw = yaml.safe_load(path.read_text()) or {}
        except Exception:
            import json as _json
            raw = _json.loads(path.read_text())
        scenes_block = raw.get("scenes", {})
        if isinstance(scenes_block, list):
            for entry in scenes_block:
                if str(entry.get("scene_id", "")) == self.scene_id:
                    goals_raw = entry.get("goals", [])
                    break
            else:
                goals_raw = []
        elif isinstance(scenes_block, dict):
            goals_raw = scenes_block.get(self.scene_id, {}).get("goals", [])
        else:
            goals_raw = []

        for g in goals_raw:
            pose = g.get("pose", {})
            # Ensure y is present
            if "y" not in pose:
                pose["y"] = 0.0
            self._goals.append(GoalEntry(
                goal_id=str(g["goal_id"]),
                goal_type=str(g.get("goal_type", "pose")),
                pose={k: float(v) for k, v in pose.items()},
                language_aliases=[str(a) for a in g.get("language_aliases", g.get("aliases", []))],
                tags=[str(t) for t in g.get("tags", [])],
            ))

    # -- query -----------------------------------------------------------

    def list_goals(self) -> List[GoalEntry]:
        return list(self._goals)

    def resolve(self, query: str, topk: int = 5,
                min_score: float = 0.10) -> Tuple[Optional[GoalMatch], List[GoalMatch]]:
        """Resolve a free-text query to the best matching goal.

        Returns:
            (best_match_or_None, topk_list)
        """
        query_norm = _normalise(query)
        query_tokens = _tokenise(query)

        candidates: List[GoalMatch] = []
        for goal in self._goals:
            best_score = 0.0
            best_alias = ""
            best_method = "token_overlap"

            # Check goal_id itself
            s, m = _score_alias(query_norm, query_tokens, goal.goal_id, goal.goal_id)
            if s > best_score:
                best_score, best_alias, best_method = s, goal.goal_id, m

            for alias in goal.language_aliases:
                s, m = _score_alias(query_norm, query_tokens, alias, goal.goal_id)
                if s > best_score:
                    best_score, best_alias, best_method = s, alias, m

            if best_score > 0.0:
                candidates.append(GoalMatch(
                    goal=goal,
                    score=best_score,
                    matched_alias=best_alias,
                    match_method=best_method,
                ))

        # Sort by score descending, then by goal_id for determinism
        candidates.sort(key=lambda m: (-m.score, m.goal.goal_id))
        top = candidates[:max(1, int(topk))]

        if not top or top[0].score < min_score:
            best_s = top[0].score if top else 0.0
            print(
                f'[TOPO_GOAL_RESOLVE_FAIL] query="{query}" reason=low_confidence '
                f'min_score={min_score:.3f} best_score={best_s:.3f}',
                flush=True,
            )
            return None, top

        best = top[0]
        print(
            f'[TOPO_GOAL_RESOLVE] query="{query}" hit={best.goal.goal_id} '
            f'type={best.goal.goal_type} score={best.score:.3f} '
            f'method={best.match_method}',
            flush=True,
        )
        topk_str = [(m.goal.goal_id, round(m.score, 3)) for m in top]
        print(f'[TOPO_GOAL_RESOLVE_TOPK] {topk_str}', flush=True)
        return best, top

    def to_query_json(self, match: GoalMatch) -> str:
        """Convert a resolved match to the JSON query format the runner expects."""
        pose = match.goal.pose
        value: Dict[str, float] = {"x": pose["x"], "z": pose.get("z", 0.0)}
        if "y" in pose:
            value["y"] = pose["y"]
        if "yaw" in pose:
            value["yaw"] = pose["yaw"]
        return json.dumps({"type": "pose", "value": value})


# ---------------------------------------------------------------------------
# CLI entry-point (used by eval suite)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="v25 goal resolver CLI")
    ap.add_argument("--catalog", type=str, required=True, help="Path to goal catalog YAML")
    ap.add_argument("--scene_id", type=str, required=True, help="Scene ID to look up")
    ap.add_argument("--query", type=str, default="", help="Free-text query to resolve")
    ap.add_argument("--goal_id", type=str, default="", help="Direct goal_id lookup (bypasses text match)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min_score", type=float, default=0.10)
    ap.add_argument("--list_goals", action="store_true", help="List all goals and exit")
    args = ap.parse_args()

    catalog = GoalCatalog(catalog_path=args.catalog, scene_id=args.scene_id)

    if args.list_goals:
        for g in catalog.list_goals():
            print(json.dumps(asdict(g)))
        return

    query = args.goal_id if args.goal_id else args.query
    if not query:
        print("[TOPO_GOAL_RESOLVE_FAIL] query=\"\" reason=empty_query", flush=True)
        raise SystemExit(1)

    best, top = catalog.resolve(query, topk=args.topk, min_score=args.min_score)
    if best is None:
        raise SystemExit(1)

    # Output the pose JSON to stdout (last line) for piping
    print(catalog.to_query_json(best))


if __name__ == "__main__":
    main()
