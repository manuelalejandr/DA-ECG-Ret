"""
Test of the explanation "TF-C wins on the UCR sets because they are short, univariate and
strongly periodic" (B5). For every secondary dataset we compute a per-series periodicity index

    PI(x) = max_{lag in [L/50, L/2]} autocorr(x, lag)          (0 = aperiodic, 1 = periodic)

(and, as a second index, 1 - spectral flatness), average it over the test split, and correlate the
dataset means with the per-dataset advantage of TF-C over SimCLR (P@10 and AUROC).

    python periodicity_index.py --secondary_csv ./secondary_raw.csv --out ./eval_v9/periodicity.csv

secondary_csv: the CSV of the secondary run (columns dataset, method, seed, ranker, K, metric, value).
UCR datasets are loaded through the repository loader (needs `aeon`); MIT-BIH is skipped here
unless --mitbih_root is given.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def periodicity(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    L = len(x)
    if x.std() < 1e-8:
        return 0.0
    f = np.fft.rfft(x, 2 * L)
    ac = np.fft.irfft(f * np.conj(f))[:L]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = max(2, L // 50), max(3, L // 2)
    return float(ac[lo:hi].max())


def spectral_flatness(x):
    p = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64) - np.mean(x))) ** 2 + 1e-12
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secondary_csv", required=True)
    ap.add_argument("--out", default="./periodicity.csv")
    ap.add_argument("--max_series", type=int, default=2000)
    a = ap.parse_args()

    from data.ucr_ecg_dataset import load_ucr_ecg_splits
    raw = pd.read_csv(a.secondary_csv)
    raw = raw[raw.seed == 0]
    rows = []
    for ds in sorted(raw.dataset.unique()):
        name = ds.replace("ucr_", "", 1)
        if "mitbih" in ds.lower():
            continue
        try:
            _, test = load_ucr_ecg_splits(name)
        except Exception as e:  # dataset-name mismatch: skip loudly
            print(f"[skip] {ds}: {e}")
            continue
        n = min(len(test), a.max_series)
        xs = [np.asarray(test.X[i, 0]).reshape(-1) for i in range(n)]
        pi = np.mean([periodicity(x) for x in xs])
        sf = np.mean([1 - spectral_flatness(x) for x in xs])
        d = raw[raw.dataset == ds]
        def val(m, met):
            sel = (d.method == m) & (d.metric == met)
            if met == "P":
                sel &= (d.ranker == "nn") & (d.K == 10)
            v = d[sel].value
            return float(v.iloc[0]) if len(v) else np.nan
        rows.append(dict(dataset=ds, series_len=len(xs[0]), periodicity=pi, one_minus_flatness=sf,
                         dP_tfc_simclr=val("tfc", "P") - val("simclr", "P"),
                         dAUROC_tfc_simclr=val("tfc", "AUROC_super") - val("simclr", "AUROC_super")))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    for idx in ("periodicity", "one_minus_flatness", "series_len"):
        for tgt in ("dP_tfc_simclr", "dAUROC_tfc_simclr"):
            ok = df[[idx, tgt]].dropna()
            if len(ok) >= 4:
                r, p = spearmanr(ok[idx], ok[tgt])
                print(f"Spearman({idx}, {tgt}) = {r:+.2f} (p={p:.3f}, n={len(ok)})")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
