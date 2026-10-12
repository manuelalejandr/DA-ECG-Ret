"""
InfoNCE / NT-Xent loss (Eq. 5), used both as the standalone SimCLR baseline
loss and as the first term of the combined DA-ECG-Ret objective (Eq. 12).
"""
import torch
import torch.nn.functional as F


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    z1, z2: (B, d) embeddings of the two augmented views (already projected
    by g_phi). Positives are (z1_i, z2_i); all other 2B-2 embeddings in the
    batch act as negatives, following SimCLR (Chen et al., 2020).
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                 # (2B, d)
    z = F.normalize(z, dim=1)
    sim = z @ z.T / tau                             # (2B, 2B) cosine sims / tau

    # mask out self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # positive index for row i (i<B -> i+B, i>=B -> i-B)
    pos_idx = torch.arange(2 * B, device=z.device)
    pos_idx = (pos_idx + B) % (2 * B)

    log_prob = F.log_softmax(sim, dim=1)
    loss = -log_prob[torch.arange(2 * B, device=z.device), pos_idx]
    return loss.mean()
