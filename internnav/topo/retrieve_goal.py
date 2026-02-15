from typing import Dict, List, Tuple

import numpy as np


def retrieve_goal_node(
    text_embed: np.ndarray, node_embeds: np.ndarray, idx_to_node: Dict[int, int], topk: int = 5
) -> Tuple[int, float]:
    text = text_embed.astype(np.float32)
    text = text / (np.linalg.norm(text) + 1e-6)
    embeds = node_embeds.astype(np.float32)
    embeds = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-6)
    sims = embeds.dot(text)
    best_idx = int(np.argmax(sims))
    return idx_to_node[best_idx], float(sims[best_idx])


def retrieve_goal_topk(
    text_embed: np.ndarray, node_embeds: np.ndarray, idx_to_node: Dict[int, int], topk: int = 5
) -> List[Tuple[int, float]]:
    text = text_embed.astype(np.float32)
    text = text / (np.linalg.norm(text) + 1e-6)
    embeds = node_embeds.astype(np.float32)
    embeds = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-6)
    sims = embeds.dot(text)
    k = min(int(topk), sims.shape[0])
    if k <= 0:
        return []
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(idx_to_node[int(i)], float(sims[int(i)])) for i in top_idx]
