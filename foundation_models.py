"""
Frozen public ECG foundation models as extra encoders (A6). NO training: embeddings only.
They go in a SEPARATE table because they are pre-trained on millions of 12-lead ECGs (so they
are not comparable with the controlled single-lead encoders that share one backbone and budget).

    python foundation_models.py --model hubert_ecg --data_root <.../ptb-xl/1.0.3> \
        --ref_embed bench_outputs/embeds/ptbxl/simclr_seed0.npz --out bench_outputs_fm \
        --hf_name Edoardo-Coppola/hubert-ecg-small
    python foundation_models.py --model ecg_fm --data_root ... --ref_embed ... --out bench_outputs_fm \
        --ckpt /path/mimic_iv_ecg_physionet_pretrained.pt          # needs `pip install fairseq-signals`

Writes <out>/embeds/ptbxl_fm/<model>_seed0.npz with the SAME keys/row order as --ref_embed
(labels, patients and hierarchy are copied from it), so ptbxl_eval_v2.py (--tag ptbxl_fm) and
ptbxl_benchmark.py (--run_tag _fm --skip_embed, for the linear-probe AUROC) work unchanged.

Preprocessing follows each model's own documentation (summarised in --help of each wrapper below):
  hubert_ecg : 12 leads, 5 s windows, band-pass 0.05-47 Hz, rescaled to [-1,1], leads flattened
               into one sequence; embedding = mean of last_hidden_state; two 5-s windows averaged.
  ecg_fm     : 12 leads x 2500 samples (5 s at 500 Hz), per-lead z-score; embedding = mean over time
               of the encoder features.
  dummy      : per-lead mean/std/diff statistics (pipeline test only).
PTB-XL records are 10 s; native 100 Hz records are resampled (scipy.signal.resample_poly) to the
model rate unless --native_500 (reads records500). Verify the preprocessing against the model's
repository before reporting; the linear-probe AUROC of this script's output is the sanity check.
"""
import argparse
import os
import time

import numpy as np
import wfdb
from scipy.signal import firwin, filtfilt, resample_poly

from data.ptbxl_dataset import load_ptbxl_metadata, patient_wise_split


def read_12(root, rel, fs_native, fs_target):
    sig, _ = wfdb.rdsamp(os.path.join(root, rel))
    x = sig.astype(np.float32).T                      # (12, T)
    if fs_native != fs_target:
        from math import gcd
        g = gcd(fs_native, fs_target)
        x = resample_poly(x, fs_target // g, fs_native // g, axis=1).astype(np.float32)
    return np.nan_to_num(x)


# ------------------------------------------------------------------------------ wrappers
class Dummy:
    fs, win = 100, 1000
    def __init__(self, a): pass
    def embed(self, x):                               # x: (12, T)
        return np.concatenate([x.mean(1), x.std(1), np.abs(np.diff(x, axis=1)).mean(1)])


class HuBERTECG:
    fs, win = 100, 500                                # 5 s at 100 Hz  (see module docstring)
    def __init__(self, a):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.m = AutoModel.from_pretrained(a.hf_name, trust_remote_code=True).to(self.dev).eval()
        self.fs, self.win = a.fs or self.fs, 5 * (a.fs or self.fs)
        self.fir = firwin(101, [0.05, 47.0], pass_zero=False, fs=self.fs)

    def embed(self, x):
        torch = self.torch
        outs = []
        for s in range(0, x.shape[1] - self.win + 1, self.win):
            w = filtfilt(self.fir, [1.0], x[:, s:s + self.win], axis=1).astype(np.float32)
            w = w / (np.abs(w).max() + 1e-8)          # rescale to [-1, 1]
            inp = torch.from_numpy(w.reshape(1, -1)).to(self.dev)   # leads flattened into one sequence
            with torch.no_grad():
                h = self.m(inp).last_hidden_state     # (1, T', D)
            outs.append(h.mean(1)[0].cpu().numpy())
        return np.mean(outs, axis=0)


class ECGFM:
    fs, win = 500, 2500
    def __init__(self, a):
        import torch
        from fairseq_signals.models import build_model_from_checkpoint
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = build_model_from_checkpoint(checkpoint_path=a.ckpt).to(self.dev).eval()

    def embed(self, x):
        torch = self.torch
        outs = []
        for s in range(0, x.shape[1] - self.win + 1, self.win):
            w = x[:, s:s + self.win]
            w = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-8)
            inp = torch.from_numpy(w[None].astype(np.float32)).to(self.dev)
            with torch.no_grad():
                out = self.m(source=inp, padding_mask=None, mask=False, features_only=True)
            f = out["x"] if isinstance(out, dict) else out
            outs.append(f.mean(1)[0].cpu().numpy() if f.dim() == 3 else f[0].cpu().numpy())
        return np.mean(outs, axis=0)


MODELS = {"dummy": Dummy, "hubert_ecg": HuBERTECG, "ecg_fm": ECGFM}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ref_embed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf_name", default="Edoardo-Coppola/hubert-ecg-small")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--fs", type=int, default=None, help="override the model's input rate")
    ap.add_argument("--native_500", action="store_true", help="read records500 instead of resampling 100 Hz")
    ap.add_argument("--max_records", type=int, default=None, help="debug")
    a = ap.parse_args()

    ref = np.load(a.ref_embed, allow_pickle=False)
    train, _, test = patient_wise_split(load_ptbxl_metadata(a.data_root))
    assert len(train) == len(ref["tr_emb"]) and len(test) == len(ref["te_emb"]), \
        f"row mismatch: meta {len(train)}/{len(test)} vs ref {len(ref['tr_emb'])}/{len(ref['te_emb'])}"
    assert np.all(test.patient_id.astype(int).values == np.asarray(ref["te_patients"]).astype(int)), "test order differs"

    enc = MODELS[a.model](a)
    fs_t = getattr(enc, "fs", 100)
    col, fs_n = ("filename_hr", 500) if a.native_500 else ("filename_lr", 100)

    def run(df, name):
        embs, t0 = [], time.time()
        n = len(df) if a.max_records is None else min(a.max_records, len(df))
        for i in range(n):
            embs.append(enc.embed(read_12(a.data_root, df.iloc[i][col], fs_n, fs_t)))
            if i % 500 == 0:
                print(f"  {name} {i}/{n}  {time.time() - t0:.0f}s", flush=True)
        return np.stack(embs).astype(np.float32)

    tr, te = run(train, "train"), run(test, "test")
    payload = {k: ref[k] for k in ref.files if k not in ("tr_emb", "te_emb")}
    if a.max_records:                                 # debug: truncate labels consistently
        for k in list(payload):
            if k.startswith("tr_"): payload[k] = payload[k][: len(tr)]
            if k.startswith("te_"): payload[k] = payload[k][: len(te)]
    payload.update(tr_emb=tr, te_emb=te)
    d = os.path.join(a.out, "embeds", "ptbxl_fm")
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"{a.model}_seed0.npz"), **payload)
    print("saved", d, tr.shape, te.shape)


if __name__ == "__main__":
    main()
