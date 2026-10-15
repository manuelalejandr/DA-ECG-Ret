"""
Patient-cluster bootstrap (95% CI) and seed s.d. for paired differences, from ptbxl_eval_v2.py output.

    python bootstrap_v2.py --setting eval_v9/ptbxl__corpus-train_excl-patient_N200 --out paper_v9

Writes <out>/tab_bootstrap.tex (encoders vs SimCLR; re-rankers vs NN) and prints the same numbers.
Test-set sampling: queries are resampled by PATIENT (2000 resamples, same resamples for both arms
of a pair), on per-query metrics averaged over the training seeds. Training randomness: s.d. over
seeds of the paired difference of the full-set means (reported separately, never pooled).
All metrics (alpha-nDCG and P@10) use the same population: queries with |S_q|>=2 (subclass level).
"""
import argparse
import glob
import os

import numpy as np

SSL = ["simclr", "ts2vec", "tfc", "softclt", "da_ecg_ret"]            # simclr_tuned is a row of its own, not part of the mean
NAMES = {"simclr": "SimCLR", "simclr_tuned": "SimCLR (tuned on fold 9)", "ts2vec": "TS2Vec", "tfc": "TF-C", "softclt": "SoftCLT", "da_ecg_ret": "DA-ECG-Ret",
         "supervised_fine": "Supervised (SCP)", "supervised_super": "Supervised (superclass)"}
B = 2000


def load(sdir):
    d = {}
    for f in sorted(glob.glob(os.path.join(sdir, "*_seed*.npz"))):
        base = os.path.basename(f)[:-4]
        m, s = base.rsplit("_seed", 1)
        d.setdefault(m, {})[int(s)] = np.load(f)
    return d


def arr(z, ranker, metric, K=10):
    return z[f"{ranker}__{K}__{metric}"].astype(float)


def per_query(d, m, ranker, metric, seeds=None):
    """(n_queries, n_seeds)"""
    ss = sorted(d[m]) if seeds is None else seeds
    return np.stack([arr(d[m][s], ranker, metric) for s in ss], 1)


class Boot:
    def __init__(self, patients, rng):
        up, inv = np.unique(patients, return_inverse=True)
        self.inv, self.n = inv, len(up)
        self.W = rng.multinomial(self.n, np.ones(self.n) / self.n, size=B).astype(float)   # (B, n_patients)

    def ci(self, x, mask):
        """x (n_queries,), mask bool. CI of the mean over masked queries."""
        ok = mask & np.isfinite(x)
        s = np.bincount(self.inv[ok], weights=x[ok], minlength=self.n)
        c = np.bincount(self.inv[ok], minlength=self.n).astype(float)
        num, den = self.W @ s, self.W @ c
        v = num / np.maximum(den, 1)
        return np.percentile(v, [2.5, 97.5])


def fmt(mean, lo, hi, sd, d=4):
    return f"${mean:+.{d}f}$ & $[{lo:+.{d}f},\\,{hi:+.{d}f}]$ & ${sd:.{d}f}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--out", default="paper_v9")
    a = ap.parse_args()
    d = load(a.setting)
    first = d[SSL[0]][min(d[SSL[0]])]
    patients = first["patient"]
    sq2 = first["struct__0__nsub_sub"] >= 2
    boot = Boot(patients, np.random.RandomState(0))
    allq = np.ones(len(patients), bool)
    rows_a, rows_b, out = [], [], []

    def pair(x, y, mask):
        """x,y (n_q, n_seeds) paired by seed -> mean diff, CI (bootstrap on seed-averaged), seed sd"""
        diff_q = (x - y).mean(1)
        mean = np.nanmean(diff_q[mask])
        lo, hi = boot.ci(diff_q, mask)
        per_seed = np.nanmean((x - y)[mask], 0)
        sd = per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan
        return mean, lo, hi, sd

    # ---- A: encoders vs SimCLR (NN)
    for m in [x for x in NAMES if x in d and x != "simclr"]:
        seeds = sorted(set(d[m]) & set(d["simclr"]))
        ra = pair(per_query(d, m, "nn", "aNDCG_sub", seeds), per_query(d, "simclr", "nn", "aNDCG_sub", seeds), sq2)
        rp = pair(per_query(d, m, "nn", "P", seeds), per_query(d, "simclr", "nn", "P", seeds), sq2)
        rows_a.append(f"{NAMES[m]} & {fmt(*ra)} & {fmt(*rp, d=3)} \\\\")
        out.append(f"{NAMES[m]} vs SimCLR: dANDCG {ra[0]:+.4f} [{ra[1]:+.4f},{ra[2]:+.4f}] sd {ra[3]:.4f} | dP {rp[0]:+.3f} [{rp[1]:+.3f},{rp[2]:+.3f}]")

    # ---- B: re-rankers vs NN, mean over the five self-supervised encoders (same seeds)
    ssl = [m for m in SSL if m in d]
    seeds = sorted(set.intersection(*[set(d[m]) for m in ssl]))
    def stack(ranker, metric):
        return np.mean([per_query(d, m, ranker, metric, seeds) for m in ssl], 0)
    labels = {"mmr@0.3": r"MMR $\lambda=0.3$", "mmr@0.5": r"MMR $\lambda=0.5$", "mmr@0.7": r"MMR $\lambda=0.7$",
              "dpp": r"DPP $\theta=1$", "xquad@0.5": r"xQuAD (pseudo-intents) $\lambda=0.5$",
              "xquad_cl_sub@0.5": r"xQuAD (corpus labels, predicted family) $\lambda=0.5$"}
    nn_a, nn_p = stack("nn", "aNDCG_sub"), stack("nn", "P")
    for r, lab in labels.items():
        if f"{r}__10__aNDCG_sub" not in first.files:
            continue
        ra = pair(stack(r, "aNDCG_sub"), nn_a, sq2)
        rp = pair(stack(r, "P"), nn_p, sq2)
        rows_b.append(f"{lab} & {fmt(*ra)} & {fmt(*rp, d=3)} \\\\")
        out.append(f"{r} vs NN (mean of SSL): dANDCG {ra[0]:+.4f} [{ra[1]:+.4f},{ra[2]:+.4f}] sd {ra[3]:.4f} | dP {rp[0]:+.3f} [{rp[1]:+.3f},{rp[2]:+.3f}]")

    hdr = r"& \multicolumn{3}{c}{$\Delta\alpha$-nDCG@10 ($|S_q|\ge2$)} & \multicolumn{3}{c}{$\Delta$P@10 ($|S_q|\ge2$)} \\" + "\n" + \
          r"& mean & 95\% CI & seed s.d. & mean & 95\% CI & seed s.d. \\"
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "tab_bootstrap.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n" + hdr + "\n\\midrule\n")
        f.write("\\multicolumn{7}{l}{\\textit{Encoder minus SimCLR (NN retrieval)}} \\\\\n" + "\n".join(rows_a))
        f.write("\n\\midrule\n\\multicolumn{7}{l}{\\textit{Re-ranker minus NN (mean over the five self-supervised encoders)}} \\\\\n" + "\n".join(rows_b))
        f.write("\n\\bottomrule\n\\end{tabular}\n")
    print("\n".join(out))
    print("wrote", os.path.join(a.out, "tab_bootstrap.tex"))


if __name__ == "__main__":
    main()
