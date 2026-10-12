"""
Determinantal Point Process (DPP) based post-hoc re-ranking baseline
(Kulesza and Taskar, 2012), used as one of the diversity baselines in
Section 4.3. We use the fast greedy MAP inference algorithm of
Chen, Zhang and Zhou (NeurIPS 2018), which incrementally updates a Cholesky
factor instead of re-evaluating determinants from scratch (O(K^2 N)).

Kernel construction: L_ij = q_i * q_j * sim(z_i, z_j), where q_i is a
"quality" term (here, cosine similarity to the query, clipped to be
non-negative) and sim is cosine similarity between candidates -- this is
the standard quality x similarity decomposition of a DPP kernel.
"""
from typing import List
import numpy as np


def dpp_greedy_map(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    candidate_ids: List[int],
    k: int,
    quality_weight: float = 1.0,
    eps: float = 1e-10,
) -> List[int]:
    n = candidate_embs.shape[0]
    k = min(k, n)

    quality = np.clip(candidate_embs @ query_emb, 1e-3, None) ** quality_weight
    sim = candidate_embs @ candidate_embs.T
    L = np.outer(quality, quality) * sim
    L = L + eps * np.eye(n)  # numerical stability

    # --- Fast greedy MAP (Chen et al., 2018) ---
    c = np.zeros((k, n))
    d2 = np.diag(L).copy()
    selected = []
    for t in range(k):
        j = int(np.argmax(d2))
        if d2[j] < eps:
            break
        selected.append(j)
        if t == k - 1:
            break
        e_num = L[j, :] - c[:t, j].T @ c[:t, :]
        e = e_num / np.sqrt(max(d2[j], eps))
        c[t, :] = e
        d2 = d2 - e ** 2
        d2[j] = -np.inf  # never reselect

    return [candidate_ids[i] for i in selected]
