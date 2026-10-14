"""
PTB-XL dataset utilities.

Download (run once, ~1.7 GB for the 100 Hz version):

    wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P ./ptbxl_data
    # then point Config.data_root to .../ptbxl_data/physionet.org/files/ptb-xl/1.0.3

We treat each ECG as a UNIVARIATE time series by keeping a single lead
(default: lead II, the classical rhythm-strip lead), matching the framing
in the draft ("si tomamos los ECG como series de tiempo univariadas").

Diagnostic superclass aggregation follows the official PTB-XL example code
(Wagner et al., 2020): each SCP code maps to one of {NORM, MI, STTC, CD, HYP}
via scp_statements.csv, and a recording's label is the SET of superclasses
present (a recording can have more than one).
"""
import ast
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import wfdb
    _HAS_WFDB = True
except ImportError:
    _HAS_WFDB = False


SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def _aggregate_diagnostic(scp_codes: dict, agg_df: pd.DataFrame) -> List[str]:
    labels = set()
    for code in scp_codes.keys():
        if code in agg_df.index:
            sc = agg_df.loc[code, "diagnostic_class"]
            if isinstance(sc, str) and sc in SUPERCLASSES:
                labels.add(sc)
    return sorted(labels)


def load_ptbxl_metadata(data_root: str) -> pd.DataFrame:
    """Load ptbxl_database.csv + scp_statements.csv and attach diagnostic
    labels at THREE granularities (list columns):
        'diagnostic_superclass' : 5  coarse classes (NORM, MI, STTC, CD, HYP)
        'diagnostic_subclass'   : ~23 diagnostic subclasses
        'diagnostic_code'       : ~44 diagnostic SCP statements
    Rows are kept iff they have >=1 superclass (unchanged from earlier
    versions, so the train/test splits are IDENTICAL to previous runs)."""
    db_path = os.path.join(data_root, "ptbxl_database.csv")
    scp_path = os.path.join(data_root, "scp_statements.csv")
    df = pd.read_csv(db_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    agg_df = pd.read_csv(scp_path, index_col=0)
    agg_df = agg_df[agg_df["diagnostic"] == 1]

    df["diagnostic_superclass"] = df["scp_codes"].apply(
        lambda codes: _aggregate_diagnostic(codes, agg_df)
    )
    df["diagnostic_subclass"] = df["scp_codes"].apply(
        lambda codes: sorted({agg_df.loc[c, "diagnostic_subclass"] for c in codes
                              if c in agg_df.index
                              and isinstance(agg_df.loc[c, "diagnostic_subclass"], str)
                              and agg_df.loc[c, "diagnostic_class"] in SUPERCLASSES})
    )
    df["diagnostic_code"] = df["scp_codes"].apply(
        lambda codes: sorted({c for c in codes if c in agg_df.index
                              and agg_df.loc[c, "diagnostic_class"] in SUPERCLASSES})
    )
    df = df[df["diagnostic_superclass"].map(len) > 0].copy()
    return df


def get_ptbxl_hierarchy(meta: pd.DataFrame, data_root: str) -> dict:
    """Label vocabularies + parent matrices, computed over the FULL metadata
    (so train and test share identical indices). Returns
        {"super": [...], "sub": [...], "code": [...],
         "sub2super": (n_sub x n_super) 0/1,  "code2sub": (n_code x n_sub),
         "code2super": (n_code x n_super)}"""
    agg_df = pd.read_csv(os.path.join(data_root, "scp_statements.csv"), index_col=0)
    agg_df = agg_df[agg_df["diagnostic"] == 1]
    sup = list(SUPERCLASSES)
    sub = sorted({x for l in meta["diagnostic_subclass"] for x in l})
    code = sorted({x for l in meta["diagnostic_code"] for x in l})
    sub2super = np.zeros((len(sub), len(sup)), dtype=np.float32)
    code2sub = np.zeros((len(code), len(sub)), dtype=np.float32)
    code2super = np.zeros((len(code), len(sup)), dtype=np.float32)
    for ci, c in enumerate(code):
        sc, sb = agg_df.loc[c, "diagnostic_class"], agg_df.loc[c, "diagnostic_subclass"]
        code2super[ci, sup.index(sc)] = 1.0
        code2sub[ci, sub.index(sb)] = 1.0
        sub2super[sub.index(sb), sup.index(sc)] = 1.0
    return {"super": sup, "sub": sub, "code": code, "sub2super": sub2super,
            "code2sub": code2sub, "code2super": code2super}


class PTBXLUnivariate(Dataset):
    """Returns single-lead, fixed-length ECG signals as float32 tensors of
    shape (1, T), plus a multi-hot label vector over SUPERCLASSES and the
    patient_id (useful for the CLOCS-style baseline / patient-level splits).
    """

    def __init__(
        self,
        data_root: str,
        metadata: pd.DataFrame,
        sampling_rate: int = 100,
        lead_index: int = 1,
        signal_length: int = 1000,
        transform=None,
        hier: Optional[dict] = None,
    ):
        self.hier = hier   # if given, items also carry label_sub / label_code
        self.data_root = data_root
        self.meta = metadata.reset_index()
        self.sampling_rate = sampling_rate
        self.lead_index = lead_index
        self.signal_length = signal_length
        self.transform = transform
        self.filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        self.classes = SUPERCLASSES

        if not _HAS_WFDB:
            raise ImportError(
                "wfdb is required to read PTB-XL waveforms: pip install wfdb"
            )
        self._load_signals_into_memory()

    # ------------------------------------------------------------------
    # One-time in-memory cache (also makes 100-epoch training fast: the
    # per-item wfdb parse of ~17k files per epoch was the main bottleneck).
    # Records that cannot be read (e.g. a .dat file truncated by an
    # interrupted download -> "cannot reshape array of size N into shape
    # (12)") are DROPPED with a loud warning instead of crashing training.
    # Re-run the wget command (it resumes with -c) and delete the cache
    # file to recover them; check_ptbxl.py lists exactly which ones.
    # ------------------------------------------------------------------
    def _cache_path(self) -> str:
        return os.path.join(
            self.data_root,
            f"_cache_sr{self.sampling_rate}_lead{'all' if self.lead_index < 0 else self.lead_index}_len{self.signal_length}.npz",
        )

    def _read_signal_from_disk(self, rec_rel: str) -> np.ndarray:
        sig, _ = wfdb.rdsamp(os.path.join(self.data_root, rec_rel))
        if self.lead_index < 0:                       # all 12 leads -> (12, T), z-normalised per lead
            x = sig.astype(np.float32).T
            if x.shape[1] > self.signal_length:
                x = x[:, : self.signal_length]
            elif x.shape[1] < self.signal_length:
                x = np.pad(x, ((0, 0), (0, self.signal_length - x.shape[1])))
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True) + 1e-8
            return (x - mu) / sd
        lead = sig[:, self.lead_index].astype(np.float32)
        if len(lead) > self.signal_length:
            lead = lead[: self.signal_length]
        elif len(lead) < self.signal_length:
            lead = np.pad(lead, (0, self.signal_length - len(lead)))
        mu, sigma = lead.mean(), lead.std() + 1e-8
        return (lead - mu) / sigma

    def _load_signals_into_memory(self):
        cache_path = self._cache_path()
        cached = {}
        bad_cached = set()
        if os.path.exists(cache_path):
            z = np.load(cache_path)
            # IMPORTANT: read each array ONCE. NpzFile re-reads (and re-allocates)
            # the whole array on every z["signals"] access, and each row view pins
            # its entire base array -> indexing z["signals"][k] inside a loop used
            # ~N x 85 MB of RAM and crashed with out-of-memory.
            ids_c, sigs_c, bad_c = z["ids"], z["signals"], z["bad_ids"]
            cached = {int(i): sigs_c[k] for k, i in enumerate(ids_c)}
            bad_cached = set(int(i) for i in bad_c)

        ecg_ids = [int(e) for e in self.meta["ecg_id"].tolist()]
        new_signals, new_bad = {}, set()
        for n, (eid, rel) in enumerate(zip(ecg_ids, self.meta[self.filename_col].tolist())):
            if eid in cached or eid in bad_cached:
                continue
            try:
                new_signals[eid] = self._read_signal_from_disk(rel)
            except Exception as e:  # truncated / corrupt / missing file
                new_bad.add(eid)
                print(f"[PTB-XL][WARN] unreadable record ecg_id={eid} ({rel}): "
                      f"{type(e).__name__}: {e}")
            if (n + 1) % 2000 == 0:
                print(f"[PTB-XL] read {n+1}/{len(ecg_ids)} records ...")

        if new_signals or new_bad:
            cached.update(new_signals)
            bad_cached |= new_bad
            ids = np.array(sorted(cached.keys()), dtype=np.int64)
            sigs = np.stack([cached[int(i)] for i in ids]) if len(ids) else np.zeros((0, self.signal_length), np.float32)
            np.savez(cache_path, ids=ids, signals=sigs,
                     bad_ids=np.array(sorted(bad_cached), dtype=np.int64))

        keep = [i for i, eid in enumerate(ecg_ids) if eid in cached]
        n_drop = len(ecg_ids) - len(keep)
        if n_drop:
            print(f"[PTB-XL][WARN] dropped {n_drop}/{len(ecg_ids)} unreadable records from this split. "
                  f"Re-run the wget command to complete the download, delete {cache_path}, "
                  f"and re-run for the exact full benchmark (python check_ptbxl.py lists them).")
            self.meta = self.meta.iloc[keep].reset_index(drop=True)
            ecg_ids = [ecg_ids[i] for i in keep]
        self._signals = np.stack([cached[e] for e in ecg_ids]) if ecg_ids else np.zeros((0, self.signal_length), np.float32)

    def __len__(self):
        return len(self.meta)

    def _read_signal(self, idx: int) -> np.ndarray:
        return self._signals[idx]

    def __getitem__(self, idx: int):
        lead = self._read_signal(idx)
        x = torch.from_numpy(lead)
        if x.dim() == 1:
            x = x.unsqueeze(0)                   # (1, T); all-lead mode is already (C, T)
        if self.transform is not None:
            x = self.transform(x)
        row = self.meta.iloc[idx]
        label = torch.zeros(len(SUPERCLASSES), dtype=torch.float32)
        for sc in row["diagnostic_superclass"]:
            label[SUPERCLASSES.index(sc)] = 1.0
        out = {
            "signal": x,
            "label": label,
            "patient_id": int(row["patient_id"]),
            "ecg_id": int(row["ecg_id"]),
        }
        if self.hier is not None:
            for key, col, voc in (("label_sub", "diagnostic_subclass", self.hier["sub"]),
                                  ("label_code", "diagnostic_code", self.hier["code"])):
                v = torch.zeros(len(voc), dtype=torch.float32)
                for item in row[col]:
                    v[voc.index(item)] = 1.0
                out[key] = v
        return out


class ContrastivePTBXL(Dataset):
    """Wraps PTBXLUnivariate to emit two augmented views per sample, as
    required by InfoNCE / SimCLR-style training (Eq. 3-4 in the draft)."""

    def __init__(self, base_dataset: PTBXLUnivariate, augment):
        self.base = base_dataset
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        x = item["signal"]
        v1 = self.augment(x.clone())
        v2 = self.augment(x.clone())
        return {
            "view1": v1,
            "view2": v2,
            "label": item["label"],
            "patient_id": item["patient_id"],
            "ecg_id": item["ecg_id"],
        }


def patient_wise_split(meta: pd.DataFrame, val_folds=(9,), test_folds=(10,)):
    """PTB-XL ships a recommended 10-fold column ('strat_fold') that is
    already patient-stratified; we use it directly to avoid patient leakage
    between train / val / test."""
    train = meta[~meta["strat_fold"].isin(list(val_folds) + list(test_folds))]
    val = meta[meta["strat_fold"].isin(val_folds)]
    test = meta[meta["strat_fold"].isin(test_folds)]
    return train, val, test
