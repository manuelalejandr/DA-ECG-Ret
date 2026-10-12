"""
Simplified re-implementation of SoftCLT (Lee, Kim, Cho, Kim, Kim, Park,
"Soft Contrastive Learning for Time Series", ICLR 2024).

Core idea: standard instance-wise contrastive learning (SimCLR/InfoNCE)
treats every other instance in the batch as an equally "hard" negative
(target probability 0), even though some negatives are, in raw signal
space, much closer to the anchor than others (e.g. two recordings of a
similar rhythm) and probably should not be pushed apart as aggressively as
a truly dissimilar pair. SoftCLT replaces the hard 0/1 target matrix with a
SOFT target derived from a similarity kernel over the raw (un-augmented)
instances, so nearby instances get partial positive credit instead of
being fully treated as negatives.

We use Euclidean distance between z-normalised raw signals as the
similarity kernel (paper's official implementation uses DTW; the two ECG
series compared here are already length-matched within a dataset -- since
every experiment here trains on ONE UCR/PTB-XL/MIT-BIH problem at a time
with a fixed series length -- so plain Euclidean distance is a fast,
length-compatible proxy for the shape-similarity DTW is meant to capture,
documented here as a simplification for computational efficiency).
"""
import torch
import torch.nn.functional as F


def instance_similarity_kernel(raw: torch.Tensor, tau_inst: float) -> torch.Tensor:
    """
    raw: (B, 1, T) raw (un-augmented) signals.
    Returns: (B, B) soft target matrix in [0, 1], diagonal = 1 (self).
    """
    B = raw.shape[0]
    flat = raw.view(B, -1)
    # pairwise Euclidean distance
    dist = torch.cdist(flat, flat, p=2)               # (B, B)
    dist = dist / (flat.shape[1] ** 0.5)               # normalise by length
    sim = torch.exp(-dist / max(tau_inst, 1e-6))       # (B, B), in (0, 1]
    return sim


def soft_instance_contrastive_loss(
    z1: torch.Tensor, z2: torch.Tensor, raw: torch.Tensor,
    tau_emb: float, tau_inst: float, alpha: float,
) -> torch.Tensor:
    """
    z1, z2: (B, d) projected embeddings of the two augmented views.
    raw:    (B, 1, T) original signals (pre-augmentation), used only to
            build the soft target kernel.
    tau_emb: embedding-space softmax temperature (as in standard InfoNCE).
    tau_inst: instance-kernel temperature (bandwidth of the soft targets).
    alpha:  blend factor in [0, 1] between the hard one-hot target (alpha=0,
            recovers vanilla InfoNCE) and the fully soft target (alpha=1).
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                     # (2B, d)
    z = F.normalize(z, dim=1)
    sim = z @ z.T / tau_emb
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))
    log_prob = F.log_softmax(sim, dim=1)                # (2B, 2B)

    # hard target: one-hot on the true positive view
    pos_idx = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    hard_target = F.one_hot(pos_idx, num_classes=2 * B).float()

    # soft target: instance-similarity kernel tiled to the 2B x 2B view space,
    # renormalised (row-wise) to sum to 1 over the non-self entries.
    inst_sim = instance_similarity_kernel(raw, tau_inst)          # (B, B)
    soft_full = inst_sim.repeat(2, 2)                              # (2B, 2B)
    soft_full = soft_full.masked_fill(mask, 0.0)
    soft_full = soft_full / (soft_full.sum(dim=1, keepdim=True) + 1e-8)

    target = (1 - alpha) * hard_target + alpha * soft_full
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # log_prob is -inf on the masked diagonal (self-similarity excluded from
    # the softmax); target is exactly 0 there by construction, but
    # 0 * (-inf) = nan in IEEE754, so we zero out log_prob on the diagonal
    # BEFORE multiplying rather than after (masking the product would be too
    # late to prevent the nan from ever being created).
    safe_log_prob = log_prob.masked_fill(mask, 0.0)
    loss = -(target * safe_log_prob).sum(dim=1).mean()
    return loss
