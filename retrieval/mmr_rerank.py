"""
Classical (non-differentiable) MMR re-ranking, applied at inference time
(Section 3.6, step 3) on top of a cosine-similarity shortlist.

    MMR(x) = lambda * sim(q, x) - (1 - lambda) * max_{y in S} sim(x, y)   (Eq. 6)
"""
from typing import List
import numpy as np


def mmr_rerank(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    candidate_ids: List[int],
    k: int,
    lam: float = 0.7,
) -> List[int]:
    """
    query_emb:       (d,)  L2-normalised query embedding.
    candidate_embs:  (N, d) L2-normalised shortlist embeddings.
    candidate_ids:   list of length N, external ids matching the rows.
    k:               cohort size K.
    lam:             relevance/diversity trade-off.

    Returns the list of selected external ids, in selection order.
    """
    n = candidate_embs.shape[0]
    k = min(k, n)
    sim_to_query = candidate_embs @ query_emb                    # (N,)
    sim_matrix = candidate_embs @ candidate_embs.T                # (N, N)

    selected = []
    remaining = list(range(n))

    # first pick: most relevant to the query
    first = int(np.argmax(sim_to_query))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        best_score, best_idx = -np.inf, None
        for j in remaining:
            redundancy = np.max(sim_matrix[j, selected])
            score = lam * sim_to_query[j] - (1 - lam) * redundancy
            if score > best_score:
                best_score, best_idx = score, j
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidate_ids[i] for i in selected]


def mmr_score(
    query_emb: np.ndarray,
    selected_embs: np.ndarray,
    lam: float = 0.7,
) -> float:
    """Average MMR objective value (Eq. 6) over a selected cohort, used as
    the 'MMR-Score' evaluation metric (Section 4.2, item 3)."""
    k = selected_embs.shape[0]
    if k == 0:
        return 0.0
    sim_to_query = selected_embs @ query_emb
    sim_matrix = selected_embs @ selected_embs.T
    scores = []
    for i in range(k):
        others = [j for j in range(k) if j != i]
        redundancy = np.max(sim_matrix[i, others]) if others else 0.0
        scores.append(lam * sim_to_query[i] - (1 - lam) * redundancy)
    return float(np.mean(scores))
