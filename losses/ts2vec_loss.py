"""
Simplified re-implementation of TS2Vec's hierarchical contrastive loss
(Yue et al., "TS2Vec: Towards Universal Representation of Time Series",
AAAI 2022), adapted to share this project's encoder backbone.

Original idea: obtain per-timestamp representations for two augmented
views of the same series (via random cropping / timestamp masking), then
contrast them at TWO levels simultaneously:
  - "temporal" contrastive loss: within one instance, timestamp t of view A
    should match timestamp t of view B, against other timestamps as
    negatives;
  - "instance" contrastive loss: at a fixed timestamp t, instance i of view
    A should match instance i of view B, against other instances (at the
    same timestamp) as negatives;
and REPEAT this at multiple temporal resolutions by max-pooling the
sequences by a factor of 2 between levels (hierarchical), summing the
losses.

Simplification vs. the original paper/codebase: we use our existing
augmentation pipeline (which already preserves sequence length via
crop+interpolate, keeping the two views timestamp-aligned) instead of
TS2Vec's specific random-cropping-with-overlap scheme, and cap the number
of hierarchical levels for efficiency on short UCR series. This keeps the
core dual-level contrastive mechanism faithful while reusing the shared
encoder/tokenizer architecture used by every other method in this codebase
for a controlled comparison.
"""
import torch
import torch.nn.functional as F


def _batched_nce(z: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """z: (N, 2M, d) L2-normalised. For each of the N independent groups,
    treat row i and row i+M (mod 2M) as a positive pair and every other row
    in the SAME group as a negative. Returns the mean NCE loss over N and 2M.
    Fully vectorised (batched matmul), no Python-level loop over N.

    `tau` (temperature) matters a lot here: cosine similarities are bounded
    in [-1, 1], so WITHOUT dividing by a small tau the softmax over
    candidates stays almost uniform even after training pulls positives
    together — this was a real bug in the first full run (no temperature
    was applied at all), producing a loss that barely moved and near-chance
    downstream AUROC. Dividing by `tau` (shared with the other methods'
    cfg.tau, default 0.07) sharpens the softmax so the loss actually
    provides a useful gradient, exactly as it does for SimCLR/InfoNCE.
    """
    N, twoM, d = z.shape
    M = twoM // 2
    sim = torch.bmm(z, z.transpose(1, 2)) / tau         # (N, 2M, 2M)
    mask = torch.eye(twoM, dtype=torch.bool, device=z.device).unsqueeze(0).expand(N, -1, -1)
    sim = sim.masked_fill(mask, float("-inf"))
    pos_idx = (torch.arange(twoM, device=z.device) + M) % twoM   # (2M,)
    pos_idx = pos_idx.unsqueeze(0).expand(N, -1)                 # (N, 2M)
    log_prob = F.log_softmax(sim, dim=-1)
    loss = -log_prob.gather(-1, pos_idx.unsqueeze(-1)).squeeze(-1)  # (N, 2M)
    return loss.mean()


def _instance_contrastive(h1: torch.Tensor, h2: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """h1, h2: (B, L, d). For each timestamp (N=L groups), contrast the 2B
    instance-views at that timestamp."""
    B, L, d = h1.shape
    h1n = F.normalize(h1, dim=-1).transpose(0, 1)   # (L, B, d)
    h2n = F.normalize(h2, dim=-1).transpose(0, 1)   # (L, B, d)
    z = torch.cat([h1n, h2n], dim=1)                # (L, 2B, d)
    return _batched_nce(z, tau=tau)


def _temporal_contrastive(h1: torch.Tensor, h2: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """h1, h2: (B, L, d). For each instance (N=B groups), contrast the 2L
    timestamp-views within that instance."""
    B, L, d = h1.shape
    if L < 2:
        return torch.tensor(0.0, device=h1.device)
    h1n = F.normalize(h1, dim=-1)   # (B, L, d)
    h2n = F.normalize(h2, dim=-1)   # (B, L, d)
    z = torch.cat([h1n, h2n], dim=1)  # (B, 2L, d)
    return _batched_nce(z, tau=tau)


def hierarchical_contrastive_loss(h1: torch.Tensor, h2: torch.Tensor,
                                   max_levels: int = 3, tau: float = 0.07) -> torch.Tensor:
    """
    h1, h2: (B, L, d) per-timestamp representations of the two views.
    Applies instance + temporal contrastive losses at up to `max_levels`
    temporal resolutions (max-pooling by factor 2 between levels), as in
    TS2Vec's hierarchical loss. `tau` is the same softmax temperature used
    by every other contrastive method in this codebase (see _batched_nce).
    """
    total = 0.0
    n_levels = 0
    cur1, cur2 = h1, h2
    for level in range(max_levels):
        if cur1.shape[1] < 2:
            break
        total = total + _instance_contrastive(cur1, cur2, tau=tau) + _temporal_contrastive(cur1, cur2, tau=tau)
        n_levels += 1
        # max-pool along time by factor 2 for the next (coarser) level
        cur1 = F.max_pool1d(cur1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        cur2 = F.max_pool1d(cur2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    return total / max(n_levels, 1)
