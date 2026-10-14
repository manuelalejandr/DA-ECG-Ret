"""
Compare input variants / pretrained models in ONE table (A1 12-lead, A6 foundation models).

    python analyze_variants.py --out tab_variants.tex \
        --cfg "Lead II, 100 Hz"=eval_v9/ptbxl__corpus-train_excl-patient_N200:bench_outputs/benchmark_raw.csv \
        --cfg "12 leads, 100 Hz"=eval_v9/ptbxl_12lead__corpus-train_excl-patient_N200:eval_v9/raw_12lead/benchmark_raw.csv

Each --cfg is  label=<setting_dir>[:<raw_csv with AUROC>].  Rows: method x variant;
columns: P@10, alpha-nDCG@10 (|S_q|>=2), RelDiv@10 (|S_q|>=2), AUROC super / SCP.
Mean +- s.d. over seeds. A random-cohort row is added from the first config.
"""
import argparse
import os

import numpy as np
import pandas as pd

from analyze_v2 import NAMES, SUP, pm, table, val

NAMES = dict(NAMES, hubert_ecg="HuBERT-ECG (frozen)", ecg_fm="ECG-FM (frozen)", st_mem="ST-MEM (frozen)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", action="append", required=True)
    ap.add_argument("--out", default="tab_variants.tex")
    ap.add_argument("--primary", default="sq2_sub")
    ap.add_argument("--methods", nargs="*", default=None)
    a = ap.parse_args()
    rows, first = [], True
    for spec in a.cfg:
        label, rest = spec.split("=", 1)
        sdir, _, raw = rest.partition(":")
        df = pd.read_csv(os.path.join(sdir, "benchmark_v2.csv"))
        au = pd.read_csv(raw) if raw and os.path.exists(raw) else None
        if au is not None:
            au = au[au.dataset.astype(str).str.startswith("ptbxl")]
        meths = [m for m in (a.methods or list(NAMES)) if m in set(df.method)]
        rows.append(r"\midrule" if not first else "")
        rows.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{label}}}}} \\")
        for m in meths:
            def aur(met):
                if au is None:
                    return "--"
                v = au[(au.method == m) & (au.metric == met)].sort_values("seed").value.values
                return pm(v)
            rows.append(f"{NAMES[m]} & {pm(val(df, m, 'nn', 'P'))} & {pm(val(df, m, 'nn', 'aNDCG_sub', a.primary))} & "
                        f"{pm(val(df, m, 'nn', 'RelDiv_sub', a.primary))} & {aur('AUROC_super')} & {aur('AUROC_code')} \\\\")
        if first:
            m0 = meths[0]
            rnd = [val(df, m0, "rand", "P"), val(df, m0, "rand", "aNDCG_sub", a.primary), val(df, m0, "rand", "RelDiv_sub", a.primary)]
            rnd_row = "Random cohort & " + " & ".join(f"${np.mean(r):.3f}$" for r in rnd) + r" & -- & -- \\"
        first = False
    rows = [r for r in rows if r] + [r"\midrule", rnd_row]
    table(a.out, "lccccc", r"Encoder & P@10 & $\alpha$-nDCG@10 & RelDiv@10 & AUROC$_{\text{sup}}$ & AUROC$_{\text{SCP}}$", rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
