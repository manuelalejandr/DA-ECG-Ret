"""
Extra augmentation pipelines used only by tune_simclr.py (the default pipeline in
data/augmentations.py is unchanged, so every earlier result stays reproducible).

  strong_ecg_augmentation   stronger parameters than the default pipeline
  PerLead                   applies a (single-channel) pipeline INDEPENDENTLY to each lead,
                            so that the 12 leads of a view do not receive identical
                            time-warping / masking / wander (12-lead only)
  LeadDropWrapper           ContrastiveWrapper that additionally zeroes a random subset of
                            leads in the second view ("lead dropout", 12-lead only)
"""
import torch

from data.augmentations import (AmplitudeScale, BaselineWander, Compose, Jitter, RandomCropResize,
                                SegmentMasking)
from data.common import ContrastiveWrapper


def strong_ecg_augmentation(sampling_rate: int = 100) -> Compose:
    return Compose(
        [
            Jitter(sigma=0.06),
            AmplitudeScale(0.6, 1.4),
            SegmentMasking(n_segments=5, max_len_frac=0.10),
            RandomCropResize(min_frac=0.6),
            BaselineWander(max_amp=0.25, sampling_rate=sampling_rate),
        ],
        apply_prob=0.9,
    )


class PerLead:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.aug(x[c:c + 1].clone()) for c in range(x.shape[0])], dim=0)


class LeadDropWrapper(ContrastiveWrapper):
    def __init__(self, base_dataset, augment, p: float = 0.5, min_keep: int = 3):
        super().__init__(base_dataset, augment)
        self.p, self.min_keep = p, min_keep

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        v = item["view2"]
        C = v.shape[0]
        keep = torch.rand(C) > self.p
        if int(keep.sum()) < self.min_keep:
            keep[torch.randperm(C)[: self.min_keep]] = True
        item["view2"] = v * keep.to(v.dtype)[:, None]
        return item
