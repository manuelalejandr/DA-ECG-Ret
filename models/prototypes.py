"""
Component A (the core fix): pseudo-cluster-gated redundancy.

Motivation (see README "Bug fixes" / the sweep results discussion): the
original retrieval-aware loss penalises a candidate for resembling ANY
other point in the batch, with no notion of class. This treats "correctly
clustering with same-phenotype recordings" (good -- exactly what gives
good Precision@K/AUROC) the same as "genuine redundant near-duplicate"
(bad), and the sweep showed this actively hurts multi-class datasets as
beta increases. The contrastive-learning literature has a name for this:
the "false negative" problem (Chuang et al., 2020, "Debiased Contrastive
Learning"; Khosla et al., 2020, "Supervised Contrastive Learning"), and the
standard self-supervised fix when no true labels are available is to
derive PSEUDO-labels via online clustering (Caron et al., 2020, "SwAV";
Li et al., 2020, "Prototypical Contrastive Learning") and use them to
avoid penalising same-cluster pairs.

Simplification vs. SwAV/PCL: rather than maintaining a persistent,
separately-trained set of learnable prototypes (which needs its own
swapped-prediction loss and is tuned for large-batch, large-dataset
regimes), we use a lightweight, PARAMETER-FREE per-batch soft k-means run
under torch.no_grad() on the current projected embeddings. This is cheap,
requires no extra loss term, and is well suited to the small batch sizes
that several datasets in this project are limited to (e.g. TwoLeadECG's
23-series training split). The resulting soft assignment Q is DETACHED
before being used as a multiplicative gate inside the retrieval-aware
loss (see losses/mmr_loss.py), so gradients still flow only through the
relevance/redundancy similarity terms, not through the clustering itself.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def batch_soft_kmeans(z: torch.Tensor, k: int, n_iters: int = 5,
                       temperature: float = 0.1, seed: int = 0) -> torch.Tensor:
    """
    z: (N, d) L2-normalised embeddings.
    Returns: (N, k) soft cluster-assignment distribution (rows sum to 1),
    fully detached from the autograd graph.
    """
    n = z.shape[0]
    k = max(1, min(k, n))
    gen = torch.Generator(device=z.device).manual_seed(seed)
    init_idx = torch.randperm(n, generator=gen, device=z.device)[:k]
    centroids = z[init_idx].clone()  # (k, d)

    for _ in range(n_iters):
        sims = z @ centroids.T                      # (N, k) cosine sim (both L2-normalised)
        assign = sims.argmax(dim=1)                  # hard assignment for stable centroid updates
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=z.device)
        new_centroids.index_add_(0, assign, z)
        counts.index_add_(0, assign, torch.ones(n, device=z.device))
        empty = counts == 0
        counts = counts.clamp_min(1.0)
        new_centroids = new_centroids / counts.unsqueeze(1)
        # re-seed any empty cluster with a random point to avoid degenerate collapse
        if empty.any():
            replace_idx = torch.randperm(n, generator=gen, device=z.device)[: int(empty.sum())]
            new_centroids[empty] = z[replace_idx]
        centroids = F.normalize(new_centroids, dim=1)

    final_sims = z @ centroids.T
    Q = F.softmax(final_sims / temperature, dim=1)   # (N, k) SOFT assignment for the gate
    return Q


def cluster_gate(Q: torch.Tensor, min_gate: float = 0.05) -> torch.Tensor:
    """
    Q: (N, k) soft cluster assignment (rows sum to 1).
    Returns: (N, N) gate in [min_gate, 1], where gate[j, k] = 1 - <Q_j, Q_k>
    is close to 0 when j and k are likely the same pseudo-cluster (so their
    mutual redundancy contribution is suppressed) and close to 1 when they
    likely belong to different clusters (so genuine cross-cluster
    redundancy is still penalised at full strength).
    """
    affinity = Q @ Q.T                     # (N, N), in [0, 1]
    gate = (1.0 - affinity).clamp(min=min_gate, max=1.0)
    return gate
