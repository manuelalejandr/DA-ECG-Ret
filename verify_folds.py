"""
Verify that PTB-XL folds are patient-disjoint and report the |S_q| distribution inputs.

    python verify_folds.py --data_root <.../ptb-xl/1.0.3> --out ./eval_v9/folds_check.txt

Writes a short text report (used for the sentence "folds are patient-disjoint" in the paper).
"""
import argparse
import itertools
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="./folds_check.txt")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.data_root, "ptbxl_database.csv"), index_col="ecg_id")
    lines = [f"records: {len(df)}  patients: {df.patient_id.nunique()}"]
    fold_pat = {f: set(g.patient_id.unique()) for f, g in df.groupby("strat_fold")}
    for f in sorted(fold_pat):
        lines.append(f"fold {f}: {int((df.strat_fold == f).sum())} records, {len(fold_pat[f])} patients")
    bad = 0
    for f1, f2 in itertools.combinations(sorted(fold_pat), 2):
        k = len(fold_pat[f1] & fold_pat[f2])
        if k:
            bad += 1
            lines.append(f"OVERLAP folds {f1}-{f2}: {k} patients")
    lines.append("RESULT: folds are patient-disjoint" if bad == 0 else f"RESULT: {bad} fold pairs share patients")
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
