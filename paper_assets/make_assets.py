"""
Generates every number-bearing asset of paper_taskbench.tex from the benchmark
output, so the manuscript can never drift from the data:

    tab_main.tex      main encoder table (PTB-XL, NN retrieval, K=10)
    tab_rerank.tex    effect of MMR / DPP re-ranking relative to plain NN
    tab_coupling.tex  Spearman coupling of each diversity metric with precision
    numbers.tex       \\newcommand macros for numbers quoted in the prose
    figures/fig_coupling.png

Usage:
    python make_assets.py --bench <dir containing benchmark_raw.csv> \\
                          [--exclude clocs_lite] [--K 10]
Re-run after every new benchmark zip.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {
    "supervised_fine": r"Supervised (SCP codes)$^{\ddagger}$",
    "supervised_super": r"Supervised (superclass)$^{\ddagger}$",
    "simclr": "SimCLR",
    "ts2vec": "TS2Vec",
    "tfc": "TF-C",
    "softclt": "SoftCLT",
    "da_ecg_ret": "DA-ECG-Ret",
    "clocs_lite": "CLOCS-lite",
}
ORDER = ["supervised_fine", "supervised_super", "simclr", "ts2vec", "tfc", "softclt", "da_ecg_ret", "clocs_lite"]


def write_tab(path, colspec, header, rows):
    """Full tabular (\\input inside a tabular breaks booktabs rules)."""
    with open(path, "w") as f:
        f.write("\\begin{tabular}{" + colspec + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n")
        f.write("\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def fmt(m, s=None, d=3):
    if s is None or np.isnan(s):
        return f"{m:.{d}f}"
    return f"{m:.{d}f} \\pm {s:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--exclude", nargs="*", default=["clocs_lite"])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(a.bench, "benchmark_raw.csv"))
    df = df[~df.method.isin(a.exclude)]
    K = a.K
    seeds_of = df.groupby("method").seed.nunique().to_dict()

    def val(method, ranker, metric):
        sub = df[(df.method == method) & (df.ranker == ranker) & (df.K == K) & (df.metric == metric)]
        return sub.sort_values("seed").value.values

    def au(method, metric):
        sub = df[(df.method == method) & (df.ranker.isna()) & (df.metric == metric)]
        return sub.sort_values("seed").value.values

    methods = [m for m in ORDER if m in seeds_of]
    num = {}

    # ---------------- main table ----------------
    rows = []
    for m in methods:
        cells = []
        for kind, ranker, metric in [("r", "nn", "P"), ("r", "nn", "aNDCG_sub"), ("r", "nn", "aNDCG_code"),
                                     ("a", None, "AUROC_super")]:
            v = val(m, ranker, metric) if kind == "r" else au(m, metric)
            if len(v) == 0:
                cells.append("--")
            else:
                cells.append("$" + fmt(v.mean(), v.std(ddof=1) if len(v) > 1 else None) + "$")
        rows.append(f"{NAMES[m]} & {seeds_of[m]} & " + " & ".join(cells) + r" \\")
    m0 = methods[0]
    rnd = {k: val(m0, "rand", k).mean() for k in ["P", "aNDCG_sub", "aNDCG_code"]}
    rows.append(r"\midrule")
    rows.append(f"Random cohort & -- & ${rnd['P']:.3f}$ & ${rnd['aNDCG_sub']:.3f}$ & ${rnd['aNDCG_code']:.3f}$ & -- \\\\")
    write_tab(os.path.join(a.out, "tab_main.tex"), "lccccc",
              r"Method & Seeds & P@10 & $\alpha$-nDCG (sub.) & $\alpha$-nDCG (SCP) & AUROC", rows)
    for m in methods:
        for met in ["P", "aNDCG_sub", "aNDCG_code"]:
            v = val(m, "nn", met)
            num[f"{m}_{met}"] = v.mean()
        v = au(m, "AUROC_super")
        if len(v):
            num[f"{m}_AUROC"] = v.mean()
    num["rand_P"], num["rand_aNDCG_sub"] = rnd["P"], rnd["aNDCG_sub"]

    # ---------------- re-ranker effect ----------------
    rer = [r for r in sorted(df.ranker.dropna().unique(), key=lambda x: (x != "nn", x)) if r not in ("nn", "rand")]
    lab = {"dpp": "DPP"}
    rows = []
    allrows = []
    for r in rer:
        dP, dRD, dA = [], [], []
        for m in methods:
            for s in sorted(df[df.method == m].seed.unique()):
                def g(rk, met):
                    x = df[(df.method == m) & (df.seed == s) & (df.ranker == rk) & (df.K == K) & (df.metric == met)]
                    return x.value.values[0] if len(x) else np.nan
                dP.append(g(r, "P") - g("nn", "P"))
                dRD.append(g(r, "RelDiv_sub") - g("nn", "RelDiv_sub"))
                dA.append(g(r, "aNDCG_sub") - g("nn", "aNDCG_sub"))
        dP, dRD, dA = map(np.array, (dP, dRD, dA))
        name = lab.get(r, f"MMR $\\lambda={r.split('@')[1]}$" if r.startswith("mmr@") else r)
        rows.append(f"{name} & ${dP.mean():+.3f}$ & ${dRD.mean():+.3f}$ & ${dA.mean():+.4f}$ & $[{dA.min():+.4f},\\,{dA.max():+.4f}]$ \\\\")
        allrows.append((r, dP.mean(), dRD.mean(), dA.mean(), dA.min(), dA.max(), len(dA)))
        num[f"d_{r}"] = dA.mean()
        num[f"dmin_{r}"], num[f"dmax_{r}"] = dA.min(), dA.max()
        num[f"dP_{r}"], num[f"dRD_{r}"] = dP.mean(), dRD.mean()
    write_tab(os.path.join(a.out, "tab_rerank.tex"), "lcccc",
              r"Re-ranker & $\Delta$P@10 & $\Delta$RelDiv & $\Delta\alpha$-nDCG & range over runs", rows)
    n_runs = allrows[0][-1] if allrows else 0

    # ---------------- coupling ----------------
    piv = df[(df.K == K) & df.ranker.notna() & (df.ranker != "rand")].pivot_table(
        index=["method", "seed", "ranker"], columns="metric", values="value").reset_index()
    rows = []
    mets = [("LegacyDiv", "Legacy label-coverage diversity"), ("RelDiv_sub", r"RelDiv@10 (subclass)"),
            ("aNDCG_sub", r"$\alpha$-nDCG@10 (subclass)"), ("aNDCG_code", r"$\alpha$-nDCG@10 (SCP code)")]
    for key, nm in mets:
        r_all = spearmanr(piv["P"], piv[key])[0]
        sub = piv[piv.ranker == "nn"]
        r_enc = spearmanr(sub["P"], sub[key])[0] if len(sub) > 3 else np.nan
        rows.append(f"{nm} & ${r_enc:+.2f}$ & ${r_all:+.2f}$ \\\\")
        num[f"rho_all_{key}"], num[f"rho_enc_{key}"] = r_all, r_enc
    write_tab(os.path.join(a.out, "tab_coupling.tex"), "lcc",
              r"Diversity measure & $\rho$ (encoders) & $\rho$ (all points)", rows)
    num["n_points_all"], num["n_points_enc"] = len(piv), int((piv.ranker == "nn").sum())

    # ---------------- coupling figure ----------------
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.7), sharex=True)
    kind = piv["ranker"].map(lambda r: "NN" if r == "nn" else ("DPP" if r == "dpp" else "MMR"))
    colors = {"NN": "#1f77b4", "MMR": "#ff7f0e", "DPP": "#2ca02c"}
    for ax, key, ttl in [(axs[0], "LegacyDiv", "Legacy label-coverage diversity"),
                         (axs[1], "aNDCG_sub", r"$\alpha$-nDCG@10 (subclass)")]:
        for k_ in ["NN", "MMR", "DPP"]:
            s = piv[kind == k_]
            ax.scatter(s["P"], s[key], s=14, alpha=0.75, color=colors[k_], label=k_)
        ax.set_xlabel("Precision@10 (shares a superclass)")
        ax.set_ylabel(ttl)
        ax.set_title(f"Spearman $\\rho$ with precision = {spearmanr(piv['P'], piv[key])[0]:+.2f}", fontsize=10)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "figures", "fig_coupling.png"), dpi=300)
    plt.close(fig)

    # ---------------- macros ----------------
    def macro(name, v, d=3, sign=False):
        s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
        return f"\\newcommand{{\\{name}}}{{{s.replace('-', '$-$')}}}"
    lines = [
        macro("rhoLegacyAll", num["rho_all_LegacyDiv"], 2, True),
        macro("rhoANDCGAll", num["rho_all_aNDCG_sub"], 2, True),
        macro("rhoRelDivAll", num["rho_all_RelDiv_sub"], 2, True),
        macro("rhoLegacyEnc", num["rho_enc_LegacyDiv"], 2, True),
        macro("rhoANDCGEnc", num["rho_enc_aNDCG_sub"], 2, True),
        macro("nPointsAll", num["n_points_all"], 0),
        macro("randP", num["rand_P"]), macro("randANDCG", num["rand_aNDCG_sub"]),
        macro("nRuns", n_runs, 0),
    ]
    for r, tag in [("mmr@0.7", "MMRseven"), ("mmr@0.5", "MMRfive"), ("mmr@0.3", "MMRthree"), ("dpp", "DPP")]:
        if f"d_{r}" in num:
            lines += [macro(f"d{tag}", num[f"d_{r}"], 4, True), macro(f"d{tag}min", num[f"dmin_{r}"], 4, True),
                      macro(f"d{tag}max", num[f"dmax_{r}"], 4, True), macro(f"dP{tag}", num[f"dP_{r}"], 3, True),
                      macro(f"dRD{tag}", num[f"dRD_{r}"], 3, True)]
    for m, tag in [("simclr", "Simclr"), ("da_ecg_ret", "Da"), ("ts2vec", "Tsvec"), ("tfc", "Tfc"), ("softclt", "Soft"),
                   ("supervised_fine", "SupFine"), ("supervised_super", "SupSuper")]:
        for met, mt in [("P", "P"), ("aNDCG_sub", "ANDCG"), ("AUROC", "AUROC")]:
            if f"{m}_{met}" in num:
                lines.append(macro(f"v{tag}{mt}", num[f"{m}_{met}"], 3))
    # DA-ECG-Ret minus SimCLR, per seed (paired by seed)
    if "da_ecg_ret" in seeds_of and "simclr" in seeds_of:
        for met, tag in [("aNDCG_sub", "ANDCG"), ("P", "P")]:
            a_, b_ = val("da_ecg_ret", "nn", met), val("simclr", "nn", met)
            n = min(len(a_), len(b_))
            d = a_[:n] - b_[:n]
            lines += [macro(f"dDaSim{tag}", d.mean(), 4, True), macro(f"dDaSim{tag}min", d.min(), 4, True),
                      macro(f"dDaSim{tag}max", d.max(), 4, True)]
        d = au("da_ecg_ret", "AUROC_super")[:n] - au("simclr", "AUROC_super")[:n]
        lines += [macro("dDaSimAUROC", d.mean(), 4, True), macro("dDaSimAUROCmin", d.min(), 4, True),
                  macro("dDaSimAUROCmax", d.max(), 4, True), macro("nSeedsDaSim", n, 0)]
    open(os.path.join(a.out, "numbers.tex"), "w").write("\n".join(lines) + "\n")
    print("wrote tab_main.tex tab_rerank.tex tab_coupling.tex numbers.tex figures/fig_coupling.png")
    print(open(os.path.join(a.out, "tab_main.tex")).read())
    print(open(os.path.join(a.out, "tab_rerank.tex")).read())
    print(open(os.path.join(a.out, "tab_coupling.tex")).read())


if __name__ == "__main__":
    main()
