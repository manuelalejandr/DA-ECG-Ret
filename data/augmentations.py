"""
Augmentation pipeline T(x) used to build the two stochastic views
v1, v2 ~ T(x) in Section 3.3 of the draft.

Since we operate on a single (univariate) lead, "random channel masking"
from the multi-lead literature is replaced by a functionally analogous
"random segment masking" (zeroing short temporal chunks), which plays the
same role of forcing the encoder to not rely on any single local pattern.
"""
import random
import torch
import torch.nn.functional as F


class Jitter:
    """Additive Gaussian temporal noise."""

    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.sigma


class AmplitudeScale:
    """Random amplitude scaling in [low, high]."""

    def __init__(self, low=0.8, high=1.2):
        self.low, self.high = low, high

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = random.uniform(self.low, self.high)
        return x * scale


class SegmentMasking:
    """Random masking of short temporal segments (univariate analogue of
    'random channel masking')."""

    def __init__(self, n_segments=3, max_len_frac=0.05):
        self.n_segments = n_segments
        self.max_len_frac = max_len_frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        max_len = max(1, int(T * self.max_len_frac))
        x = x.clone()
        for _ in range(self.n_segments):
            length = random.randint(1, max_len)
            start = random.randint(0, T - length)
            x[..., start : start + length] = 0.0
        return x


class RandomCropResize:
    """Random temporal crop followed by resizing back to the original
    length (linear interpolation)."""

    def __init__(self, min_frac=0.8):
        self.min_frac = min_frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        frac = random.uniform(self.min_frac, 1.0)
        crop_len = max(8, int(T * frac))
        start = random.randint(0, T - crop_len)
        cropped = x[..., start : start + crop_len]
        cropped = cropped.unsqueeze(0)  # (1,1,T)
        resized = F.interpolate(cropped, size=T, mode="linear", align_corners=False)
        return resized.squeeze(0)


class BaselineWander:
    """Low-frequency sinusoidal baseline wander perturbation."""

    def __init__(self, max_amp=0.15, max_freq_hz=0.5, sampling_rate=100):
        self.max_amp = max_amp
        self.max_freq_hz = max_freq_hz
        self.fs = sampling_rate

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        t = torch.arange(T, dtype=torch.float32) / self.fs
        freq = random.uniform(0.05, self.max_freq_hz)
        amp = random.uniform(0.0, self.max_amp)
        phase = random.uniform(0, 2 * 3.14159265)
        wander = amp * torch.sin(2 * 3.14159265 * freq * t + phase)
        return x + wander.unsqueeze(0)


class Compose:
    def __init__(self, transforms, apply_prob=0.8):
        self.transforms = transforms
        self.apply_prob = apply_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            if random.random() < self.apply_prob:
                x = t(x)
        return x


def default_ecg_augmentation(sampling_rate: int = 100) -> Compose:
    """Matches the augmentation family listed in Section 3.3 of the draft."""
    return Compose(
        [
            Jitter(sigma=0.03),
            AmplitudeScale(0.8, 1.2),
            SegmentMasking(n_segments=3, max_len_frac=0.05),
            RandomCropResize(min_frac=0.85),
            BaselineWander(max_amp=0.1, sampling_rate=sampling_rate),
        ],
        apply_prob=0.8,
    )
