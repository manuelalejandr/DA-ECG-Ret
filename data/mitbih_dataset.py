"""
MIT-BIH Arrhythmia Database (Moody & Mark, 2001) — open-access on
PhysioNet (no credentialing required, unlike MIMIC).

We stream each record directly from PhysioNet using wfdb's `pn_dir`
mechanism (no need to bulk-download the whole database), extract the
primary lead (MLII when present, else the first available lead — keeping
the univariate framing), segment individual heartbeats around annotated
R-peaks, and map the original beat annotation symbols to the standard AAMI
5-class scheme used throughout the arrhythmia-classification literature:

    N (normal), S (supraventricular ectopic), V (ventricular ectopic),
    F (fusion), Q (unknown / paced)

Segmented beats are cached to a local .npz file after the first extraction
so repeated runs (e.g. across several training epochs / methods) do not
re-hit the network or re-parse annotations every time.
"""
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import wfdb
    _HAS_WFDB = True
except ImportError:
    _HAS_WFDB = False

from config import MITBIH_WINDOW_BEFORE, MITBIH_WINDOW_AFTER, AAMI_CLASSES

# Standard AAMI mapping (de Chazal et al., 2004) from raw MIT-BIH beat
# annotation symbols to the 5 superclasses.
_AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "P": "Q", "f": "Q", "U": "Q", "/": "Q",
}


def _extract_record_beats(record_name: str, pn_dir: str = "mitdb"):
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    ann = wfdb.rdann(record_name, "atr", pn_dir=pn_dir)

    sig_names = record.sig_name
    lead_idx = sig_names.index("MLII") if "MLII" in sig_names else 0
    signal = record.p_signal[:, lead_idx].astype(np.float32)

    beats, labels = [], []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol not in _AAMI_MAP:
            continue
        start, end = sample - MITBIH_WINDOW_BEFORE, sample + MITBIH_WINDOW_AFTER
        if start < 0 or end > len(signal):
            continue
        beats.append(signal[start:end])
        labels.append(_AAMI_MAP[symbol])

    if not beats:
        return np.zeros((0, MITBIH_WINDOW_BEFORE + MITBIH_WINDOW_AFTER), dtype=np.float32), []
    return np.stack(beats, axis=0), labels


def build_mitbih_cache(records: List[str], cache_path: str, pn_dir: str = "mitdb"):
    if not _HAS_WFDB:
        raise ImportError("wfdb is required for MIT-BIH: pip install wfdb")
    all_beats, all_labels, all_record_ids = [], [], []
    for rec in records:
        print(f"[mitbih] streaming record {rec} from PhysioNet ({pn_dir})...")
        beats, labels = _extract_record_beats(rec, pn_dir=pn_dir)
        all_beats.append(beats)
        all_labels.extend(labels)
        all_record_ids.extend([rec] * len(labels))
    X = np.concatenate(all_beats, axis=0) if all_beats else np.zeros((0, 1))
    y = np.array(all_labels)
    record_ids = np.array(all_record_ids)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, record_ids=record_ids)
    print(f"[mitbih] cached {X.shape[0]} beats from {len(records)} records to {cache_path}")


class MITBIHBeats(Dataset):
    """Beat-level MIT-BIH dataset with AAMI 5-class labels.

    `split` selects a deterministic 80/20 RECORD-WISE split (not beat-wise,
    to avoid leaking beats from the same patient/record across train/test).
    """

    def __init__(self, cache_path: str, records: List[str], split: str = "train",
                 test_frac: float = 0.2, seed: int = 0):
        if not os.path.exists(cache_path):
            build_mitbih_cache(records, cache_path)
        data = np.load(cache_path, allow_pickle=True)
        X, y, record_ids = data["X"], data["y"], data["record_ids"]

        rng = np.random.RandomState(seed)
        unique_records = sorted(set(records))
        rng.shuffle(unique_records)
        n_test = max(1, int(len(unique_records) * test_frac))
        test_records = set(unique_records[:n_test])
        train_records = set(unique_records[n_test:])
        keep_records = test_records if split == "test" else train_records

        mask = np.isin(record_ids, list(keep_records))
        self.X = X[mask].astype(np.float32)
        self.y_raw = y[mask]
        self.classes = AAMI_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.signal_length = self.X.shape[-1] if self.X.shape[0] > 0 else (
            None
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sig = self.X[idx]
        mu, sigma = sig.mean(), sig.std() + 1e-8
        sig = (sig - mu) / sigma
        x = torch.from_numpy(sig).unsqueeze(0).float()
        label = torch.zeros(len(self.classes), dtype=torch.float32)
        label[self.class_to_idx[self.y_raw[idx]]] = 1.0
        return {
            "signal": x,
            "label": label,
            "patient_id": idx,
            "ecg_id": idx,
        }


def load_mitbih_splits(records: List[str], cache_dir: str = "./mitbih_cache"
                        ) -> Tuple[MITBIHBeats, MITBIHBeats]:
    cache_path = os.path.join(cache_dir, "mitbih_beats.npz")
    train_ds = MITBIHBeats(cache_path, records, split="train")
    test_ds = MITBIHBeats(cache_path, records, split="test")
    return train_ds, test_ds
