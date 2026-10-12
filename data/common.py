"""
Generic dataset plumbing shared by every SSL method and every data source
(synthetic / PTB-XL / UCR-UEA ECG / MIT-BIH). This replaces the
dataset-specific Contrastive* wrapper classes with a single generic one, and
centralises "given --dataset X, give me (train_base, test_base, n_classes,
signal_length)" so train.py / evaluate.py don't special-case each source.
"""
from typing import Tuple
import torch
from torch.utils.data import Dataset

from data.physiological_features import extract_physio_features


class ContrastiveWrapper(Dataset):
    """Wraps ANY base dataset whose __getitem__ returns a dict with a
    'signal' tensor (1, T), producing two augmented views plus the raw
    signal (some SSL methods, e.g. TF-C and SoftCLT, need the original
    un-augmented signal too) and a small vector of unsupervised
    physiological/morphology descriptors (Component B of DA-ECG-Ret v2,
    see data/physiological_features.py) used as an auxiliary regression
    target -- computed once per access, no labels involved."""

    def __init__(self, base_dataset: Dataset, augment):
        self.base = base_dataset
        self.augment = augment
        # The physiological descriptors are a deterministic function of the
        # UNAUGMENTED signal, so they only need computing once per sample
        # (they cost ~10 ms each; recomputing every epoch dominated training
        # time on PTB-XL). Lazy cache -> identical values, far faster epochs
        # 2..N. Assumes num_workers=0 (the project default); with workers the
        # cache is simply per-worker and still correct.
        self._phys_cache = {}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        x = item["signal"]
        v1 = self.augment(x.clone())
        v2 = self.augment(x.clone())
        phys = self._phys_cache.get(idx)
        if phys is None:
            phys = extract_physio_features(x.squeeze(0).numpy())
            self._phys_cache[idx] = phys
        return {
            "view1": v1,
            "view2": v2,
            "raw": x,
            "phys": torch.from_numpy(phys),
            "label": item["label"],
            "patient_id": item["patient_id"],
            "ecg_id": item["ecg_id"],
            **({"label_code": item["label_code"]} if "label_code" in item else {}),
        }


def get_dataset_splits(args, cfg) -> Tuple[Dataset, Dataset, int, int]:
    """Returns (train_base, test_base, n_classes, signal_length) for the
    dataset requested by `args`/`cfg`. `test_base` doubles as both the
    query pool and the retrieval corpus in evaluate.py (see its docstring
    for why this does not leak labels used during SSL pre-training, which
    is unsupervised)."""
    dataset = args.dataset

    if dataset == "synthetic":
        from data.synthetic_ecg import SyntheticECG, SUPERCLASSES
        train_base = SyntheticECG(n_samples=getattr(args, "n_synthetic_train", 1200), seed=1)
        test_base = SyntheticECG(n_samples=getattr(args, "n_synthetic_val", 400), seed=2)
        return train_base, test_base, len(SUPERCLASSES), cfg.signal_length

    elif dataset == "ptbxl":
        from data.ptbxl_dataset import (
            load_ptbxl_metadata, PTBXLUnivariate, patient_wise_split, get_ptbxl_hierarchy,
        )
        meta = load_ptbxl_metadata(cfg.data_root)
        hier = get_ptbxl_hierarchy(meta, cfg.data_root)
        train_meta, val_meta, test_meta = patient_wise_split(meta)
        train_base = PTBXLUnivariate(cfg.data_root, train_meta, cfg.sampling_rate,
                                      cfg.lead_index, cfg.signal_length, hier=hier)
        test_base = PTBXLUnivariate(cfg.data_root, test_meta, cfg.sampling_rate,
                                     cfg.lead_index, cfg.signal_length, hier=hier)
        test_base.hier_matrices = train_base.hier_matrices = hier
        return train_base, test_base, len(cfg.superclasses), cfg.signal_length

    elif dataset == "ucr":
        from data.ucr_ecg_dataset import load_ucr_ecg_splits
        train_base, test_base = load_ucr_ecg_splits(cfg.ucr_name)
        return train_base, test_base, len(train_base.classes), train_base.signal_length

    elif dataset == "mitbih":
        from data.mitbih_dataset import load_mitbih_splits
        train_base, test_base = load_mitbih_splits(cfg.mitbih_records,
                                                     cache_dir=cfg.data_root)
        return train_base, test_base, len(train_base.classes), train_base.signal_length

    else:
        raise ValueError(f"Unknown dataset '{dataset}'")


def effective_batch_size(requested: int, n_train: int) -> int:
    """Small UCR splits (e.g. TwoLeadECG has only 23 training series) can be
    smaller than the requested batch size; cap it so training doesn't crash,
    while warning the caller."""
    bs = min(requested, max(4, n_train // 2))
    if bs < requested:
        print(f"[warn] reduced batch_size {requested} -> {bs} "
              f"(train split has only {n_train} samples)")
    return bs
