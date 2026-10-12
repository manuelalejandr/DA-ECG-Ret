"""
Synthetic univariate ECG-like signal generator.

This is NOT a physiological simulator: it exists purely so the full
pipeline (training + retrieval evaluation + plotting) can be smoke-tested
in seconds, on a laptop or in this sandbox, before running the real
experiments on PTB-XL / MIMIC-IV-ECG. It fabricates 5 synthetic "diagnostic
superclasses" with distinct waveform morphology and heart-rate ranges so
that Precision@K / Diversity@K / AUROC are non-trivial and sanity-checkable.

Usage in real experiments: replace this module with data/ptbxl_dataset.py.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def _make_beat(t, hr_bpm, morphology, noise=0.02):
    """Generate a toy periodic waveform with class-specific morphology."""
    f = hr_bpm / 60.0
    base = np.sin(2 * np.pi * f * t)
    if morphology == "NORM":
        sig = base + 0.3 * np.sin(2 * np.pi * 2 * f * t)
    elif morphology == "MI":
        # depressed / inverted secondary wave
        sig = base - 0.5 * np.sin(2 * np.pi * 2 * f * t + 0.5)
    elif morphology == "STTC":
        sig = base + 0.4 * np.sin(2 * np.pi * 3 * f * t)
    elif morphology == "CD":
        # irregular rhythm: modulate frequency slightly
        drift = 1 + 0.1 * np.sin(2 * np.pi * 0.2 * t)
        sig = np.sin(2 * np.pi * f * drift * t)
    elif morphology == "HYP":
        sig = 1.4 * base + 0.2 * np.sin(2 * np.pi * 4 * f * t)
    else:
        sig = base
    sig = sig + np.random.normal(0, noise, size=t.shape)
    return sig.astype(np.float32)


class SyntheticECG(Dataset):
    def __init__(self, n_samples=2000, signal_length=1000, sampling_rate=100,
                 n_patients=400, seed=0):
        rng = np.random.RandomState(seed)
        self.signal_length = signal_length
        self.fs = sampling_rate
        self.t = np.arange(signal_length) / sampling_rate

        self.labels = []
        self.patient_ids = []
        self.hrs = []
        self.morphs = []
        for i in range(n_samples):
            # allow multi-label (comorbidity) with small probability
            primary = rng.choice(SUPERCLASSES)
            labels = {primary}
            if rng.rand() < 0.15:
                labels.add(rng.choice(SUPERCLASSES))
            self.labels.append(sorted(labels))
            self.patient_ids.append(int(rng.randint(0, n_patients)))
            hr_range = {
                "NORM": (60, 90), "MI": (50, 70), "STTC": (70, 100),
                "CD": (40, 60), "HYP": (80, 110),
            }[primary]
            self.hrs.append(rng.uniform(*hr_range))
            self.morphs.append(primary)
        self.seed = seed
        self.classes = SUPERCLASSES

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.seed * 100003 + idx)
        np.random.seed(rng.randint(0, 2**31 - 1))
        sig = _make_beat(self.t, self.hrs[idx], self.morphs[idx])
        mu, sigma = sig.mean(), sig.std() + 1e-8
        sig = (sig - mu) / sigma
        x = torch.from_numpy(sig).unsqueeze(0)
        label = torch.zeros(len(SUPERCLASSES))
        for sc in self.labels[idx]:
            label[SUPERCLASSES.index(sc)] = 1.0
        return {
            "signal": x,
            "label": label,
            "patient_id": self.patient_ids[idx],
            "ecg_id": idx,
        }


class ContrastiveSynthetic(Dataset):
    def __init__(self, base: SyntheticECG, augment):
        self.base = base
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        x = item["signal"]
        v1 = self.augment(x.clone())
        v2 = self.augment(x.clone())
        return {
            "view1": v1, "view2": v2, "label": item["label"],
            "patient_id": item["patient_id"], "ecg_id": item["ecg_id"],
        }
