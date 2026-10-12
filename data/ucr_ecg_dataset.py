"""
Loader for the free, univariate ECG problems in the UCR/UEA time series
classification archive (https://www.timeseriesclassification.com/dataset.php),
fetched through the aeon toolkit (https://www.aeon-toolkit.org).

aeon's `load_classification(name, split=...)` downloads (and locally caches)
the requested problem the first time it is called, returning:
    X: np.ndarray of shape (n_instances, n_channels, series_length)
    y: np.ndarray of shape (n_instances,) with string/int class labels

All datasets in config.UCR_ECG_DATASETS are univariate (n_channels == 1),
matching the "ECG as a univariate time series" framing of the project.
Series length varies PER DATASET (e.g. TwoLeadECG: 82, CinCECGTorso: 1639),
so each UCR dataset is treated as an independent experiment -- exactly as is
standard practice when benchmarking on the UCR/UEA archive -- and results
are aggregated across datasets afterwards (see aggregate.py).

Note: downloading requires outbound internet access to the archive's Zenodo
mirror. This is expected to work from Colab or any local machine with
normal internet access; it will fail in network-sandboxed environments.
"""
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from aeon.datasets import load_classification
    _HAS_AEON = True
except ImportError:
    _HAS_AEON = False


def _znorm(x: np.ndarray) -> np.ndarray:
    mu, sigma = x.mean(), x.std() + 1e-8
    return (x - mu) / sigma


class UCRECGDataset(Dataset):
    """Wraps a single UCR/UEA univariate ECG problem/split.

    Each item's label is returned as a one-hot vector over the SORTED set
    of classes present in the TRAIN split (fixed at construction / passed
    in via `classes`, so that train/test one-hot encodings agree).
    """

    def __init__(self, name: str, split: str, classes: List[str] = None):
        if not _HAS_AEON:
            raise ImportError("aeon is required for UCR/UEA datasets: pip install aeon")
        self.name = name
        self.split = split
        X, y = load_classification(name, split=split)
        # X: (n, 1, T) for univariate problems
        assert X.shape[1] == 1, f"{name} is not univariate (n_channels={X.shape[1]})"
        self.X = X.astype(np.float32)
        self.y_raw = np.asarray(y)
        self.classes = classes if classes is not None else sorted(set(self.y_raw.tolist()))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.signal_length = self.X.shape[-1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sig = _znorm(self.X[idx, 0, :])
        x = torch.from_numpy(sig).unsqueeze(0).float()  # (1, T)
        label = torch.zeros(len(self.classes), dtype=torch.float32)
        cls = self.y_raw[idx]
        if cls in self.class_to_idx:  # a test-split label unseen in train is left as all-zero
            label[self.class_to_idx[cls]] = 1.0
        return {
            "signal": x,
            "label": label,
            "patient_id": idx,   # UCR archive has no patient identifiers
            "ecg_id": idx,
        }


def load_ucr_ecg_splits(name: str) -> Tuple[UCRECGDataset, UCRECGDataset]:
    """Returns (train_dataset, test_dataset) with a shared, consistent
    one-hot label space derived from the train split's classes."""
    train_ds = UCRECGDataset(name, split="train")
    test_ds = UCRECGDataset(name, split="test", classes=train_ds.classes)
    return train_ds, test_ds
