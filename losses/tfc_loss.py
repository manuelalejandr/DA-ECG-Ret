"""
Simplified re-implementation of TF-C (Time-Frequency Consistency)
(Zhang, Yan, Zhang, Franks, Liu, Chen, "Self-Supervised Contrastive
Pre-Training for Time Series via Time-Frequency Consistency", NeurIPS 2022).

Core idea: learn a time-domain encoder and a frequency-domain encoder such
that (a) each domain is internally contrastive (InfoNCE on augmented
views), and (b) the time and frequency embeddings of the SAME instance are
pulled together in a shared space while being pushed apart from other
instances' embeddings (the "consistency" objective) -- encouraging the
model to capture information that agrees across both representations of
the signal.

Simplification vs. the original paper: the official TF-C additionally
uses a learned time-frequency "translation" via random frequency masks and
a dedicated distance-metric consistency term; here we use plain FFT
magnitude as the frequency view, simple frequency-domain augmentation
(random bin masking / jitter), and a symmetric InfoNCE-style consistency
loss between the two domains, which captures the same self-supervised
signal (agreement between time- and frequency-domain views of one
instance) in a form consistent with this codebase's shared-encoder design.
"""
import random
import torch
import torch.nn.functional as F

from losses.infonce import info_nce_loss


class FreqAugment:
    """Simple frequency-domain augmentation: random magnitude jitter plus
    random bin masking on the rFFT magnitude spectrum."""

    def __init__(self, mask_frac=0.1, jitter_sigma=0.05):
        self.mask_frac = mask_frac
        self.jitter_sigma = jitter_sigma

    def __call__(self, mag: torch.Tensor) -> torch.Tensor:
        # mag: (1, F)
        mag = mag + torch.randn_like(mag) * self.jitter_sigma * mag.std()
        F_bins = mag.shape[-1]
        n_mask = max(1, int(F_bins * self.mask_frac))
        idx = torch.randperm(F_bins)[:n_mask]
        mag = mag.clone()
        mag[..., idx] = 0.0
        return mag


def to_freq_magnitude(x: torch.Tensor) -> torch.Tensor:
    """x: (B, 1, T) raw time-domain signal -> (B, 1, F) rFFT magnitude,
    log1p-compressed and z-normalised per instance."""
    spec = torch.fft.rfft(x, dim=-1)
    mag = spec.abs()
    mag = torch.log1p(mag)
    mu = mag.mean(dim=-1, keepdim=True)
    sigma = mag.std(dim=-1, keepdim=True) + 1e-8
    return (mag - mu) / sigma


def consistency_loss(z_time: torch.Tensor, z_freq: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Symmetric cross-domain InfoNCE: positive pair = (time_i, freq_i);
    negatives = all other instances in the batch, in both directions."""
    zt = F.normalize(z_time, dim=1)
    zf = F.normalize(z_freq, dim=1)
    logits_t2f = zt @ zf.T / tau
    logits_f2t = logits_t2f.T
    targets = torch.arange(zt.shape[0], device=zt.device)
    loss_t2f = F.cross_entropy(logits_t2f, targets)
    loss_f2t = F.cross_entropy(logits_f2t, targets)
    return 0.5 * (loss_t2f + loss_f2t)


def tfc_loss(
    time_v1: torch.Tensor, time_v2: torch.Tensor,
    freq_v1: torch.Tensor, freq_v2: torch.Tensor,
    time_repr_1: torch.Tensor, freq_repr_1: torch.Tensor,
    tau: float, consistency_weight: float,
):
    """
    time_v1, time_v2:  (B, d) projected time-domain embeddings of the two
                        time-augmented views (for the within-domain InfoNCE).
    freq_v1, freq_v2:  (B, d) projected frequency-domain embeddings of the
                        two frequency-augmented views.
    time_repr_1, freq_repr_1: (B, d) projected embeddings of view 1 in each
                        domain, used for the cross-domain consistency term.
    """
    l_time = info_nce_loss(time_v1, time_v2, tau=tau)
    l_freq = info_nce_loss(freq_v1, freq_v2, tau=tau)
    l_consist = consistency_loss(time_repr_1, freq_repr_1, tau=tau)
    total = l_time + l_freq + consistency_weight * l_consist
    return total, {
        "time": l_time.item(), "freq": l_freq.item(),
        "consistency": l_consist.item(), "total": total.item(),
    }
