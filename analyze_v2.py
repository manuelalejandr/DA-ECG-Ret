"""
Analysis of ptbxl_eval_v2.py outputs -> LaTeX tables, figures and number macros for the paper.

    python analyze_v2.py --main  eval_v2/corpus-train_excl-patient_N200 \
                         --test  eval_v2/corpus-test_excl-patient_N200 \
                         --leak  eval_v2/corpus-test_excl-patient_N200 eval_v2/corpus-test_excl-self_N200 \
                                 eval_v2/corpus-test_excl-none_N200 \
                         --raw   bench_outputs/benchmark_raw.csv   (AUROC at three levels, optional) \
                         --out   paper_v2

--main is the setting reported in the main text (queries = fold 10, corpus = folds 1-8).
Everything reported with seeds is mean +- s.d. over training seeds. Strata:
    all      every query
    sq2_sub  queries whose family has >= 2 subclasses (drops almost all NORM queries)
    super_X  queries having superclass X
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"supervised_fine": r"Supervised (SCP)$^{\ddagger}$", "supervised_super": r"Supervised (superclass)$^{\ddagger}$",
         "simclr": "SimCLR", "simclr_tuned": "SimCLR (tuned on fold 9)", "ts2vec": "TS2Vec", "tfc": "TF-C", "softclt": "SoftCLT", "da_ecg_ret": "DA-ECG-Ret"}
ORDER = ["supervised_fine", "supervised_super", "simclr", "simclr_tuned", "ts2vec", "tfc", "softclt", "da_ecg_ret"]
SUP = ("supervised_fine", "supervised_super")
K = 10
REAL_RANKERS_PREFIX = ("nn", "mmr@", "dpp", "xquad@")          # realisable (no labels at test time)


def pm(x, d=3):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "--"
    return f"${x.mean():.{d}f}$" if len(x) == 1 else f"${x.mean():.{d}f} \\pm {x.std(ddof=1):.{d}f}$"


def table(path, colspec, header, rows):
    with open(path, "w") as f:
        f.write("\\begin{tabular}{" + colspec + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n")
        f.write("\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def load(setting_dir):
    return pd.read_csv(os.path.join(setting_dir, "benchmark_v2.csv"))


def val(df, method, ranker, metric, stratum="all", k=K):
    s = df[(df.method == method) & (df.ranker == ranker) & (df.K == k) & (df.metric == metric) & (df.stratum == stratum)]
    return s.sort_values("seed").value.values


def methods_in(df):
    return [m for m in ORDER if m in set(df.method)]


def partial_corr(x, y, z):
    """Spearman partial correlation of x,y controlling for z: Pearson partial correlation of the
    ranks (residuals of the rank of x and of y on the rank of z)."""
    x, y, z = (rankdata(v).astype(float) for v in (x, y, z))
    def resid(a, b):
        A = np.c_[np.ones_like(b), b]
        beta = np.linalg.lstsq(A, a, rcond=None)[0]
        return a - A @ beta
    rx, ry = resid(x, z), resid(y, z)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--leak", nargs="*", default=[])
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default="paper_v2")
    ap.add_argument("--primary", default="sq2_sub", help="stratum used for the primary alpha-nDCG column")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    df = load(a.main)
    meths = methods_in(df)                                   # every row shown in the tables
    core = [m for m in meths if m != "simclr_tuned"]         # 7 benchmark methods: used for every aggregate
    ssl = [m for m in core if m not in SUP]                  # the five self-supervised encoders
    macros = {}

    # ------------------------------------------------------------------ main table
    au = None
    if a.raw and os.path.exists(a.raw):
        raw = pd.read_csv(a.raw)
        raw = raw[raw.dataset.astype(str).str.startswith("ptbxl")]
        raw = raw[raw.metric.isin(["AUROC_super", "AUROC_sub", "AUROC_code"])]
        au = raw
    rows = []
    for m in meths:
        cells = [pm(val(df, m, "nn", "P")),
                 pm(val(df, m, "nn", "P", a.primary)),
                 pm(val(df, m, "nn", "aNDCG_sub", "all")),
                 pm(val(df, m, "nn", "aNDCG_sub", a.primary)),
                 pm(val(df, m, "nn", "RelDiv_sub", a.primary)),
                 pm(val(df, m, "nn", "aNDCG_code", "sq2_code"))]
        for met in ("AUROC_super", "AUROC_sub", "AUROC_code"):
            cells.append(pm(au[(au.method == m) & (au.metric == met)].sort_values("seed").value.values) if au is not None else "--")
        nseed = df[df.method == m].seed.nunique()
        rows.append(f"{NAMES[m]} & {nseed} & " + " & ".join(cells) + r" \\")
    m0 = meths[0]
    rnd = [val(df, m0, "rand", "P"), val(df, m0, "rand", "P", a.primary), val(df, m0, "rand", "aNDCG_sub", "all"), val(df, m0, "rand", "aNDCG_sub", a.primary),
           val(df, m0, "rand", "RelDiv_sub", a.primary), val(df, m0, "rand", "aNDCG_code", "sq2_code")]
    rows.append(r"\midrule")
    rows.append("Random cohort & -- & " + " & ".join(f"${np.mean(r):.3f}$" for r in rnd) + r" & -- & -- & -- \\")
    table(os.path.join(a.out, "tab_main_v2.tex"), "l" + "c" * 10,
          r"Method & Seeds & P@10 (all) & P@10 ($|S_q|\!\ge\!2$) & $\alpha$-nDCG (all) & $\alpha$-nDCG ($|S_q|\!\ge\!2$) & RelDiv ($|S_q|\!\ge\!2$) & $\alpha$-nDCG SCP ($|S_q|\!\ge\!2$) & AUROC$_{\text{sup}}$ & AUROC$_{\text{sub}}$ & AUROC$_{\text{SCP}}$",
          rows)

    # ------------------------------------------------------------------ strata
    srows = []
    first = sorted(glob.glob(os.path.join(a.main, "*_seed*.npz")))
    S = {"n": {}, "p1": {}, "msub": {}, "mcode": {}}
    acc = {}
    for f in first:
        z = np.load(f)
        coarse = z["_coarse"]
        vs = [str(x) for x in z["_vocab_super"]] if "_vocab_super" in z.files else [f"C{j}" for j in range(coarse.shape[1])]
        nsub = z["struct__0__nsub_sub"]
        ncode = z["struct__0__nsub_code"] if "struct__0__nsub_code" in z.files else None
        for j, nm in enumerate(vs):
            msk = coarse[:, j] > 0
            acc.setdefault(nm, dict(n=msk.sum(), p1=(nsub[msk] == 1).mean(), msub=nsub[msk].mean(),
                                    mcode=ncode[msk].mean() if ncode is not None else np.nan))
        break
    sup_names = list(acc)
    if first:
        z0 = np.load(first[0])
        macros["nQueries"] = len(z0["struct__0__nsub_sub"])
        macros["nSq2sub"] = int((z0["struct__0__nsub_sub"] >= 2).sum())
        if "struct__0__nsub_code" in z0.files:
            macros["nSq2code"] = int((z0["struct__0__nsub_code"] >= 2).sum())
    for nm in sup_names:
        stratum = f"super_{nm}"
        nn = np.concatenate([val(df, m, "nn", "aNDCG_sub", stratum) for m in ssl])
        mm = np.concatenate([val(df, m, "mmr@0.7", "aNDCG_sub", stratum) for m in ssl])
        rdn = np.concatenate([val(df, m, "nn", "RelDiv_sub", stratum) for m in ssl])
        orc = np.concatenate([val(df, m, "oracle_sub", "aNDCG_sub", stratum) for m in ssl])
        rnd_ = np.concatenate([val(df, m, "rand", "aNDCG_sub", stratum) for m in ssl])
        srows.append(f"{nm} & {acc[nm]['n']} & ${100 * acc[nm]['p1']:.0f}\\%$ & ${acc[nm]['msub']:.1f}$ & ${acc[nm]['mcode']:.1f}$ & "
                     f"${nn.mean():.3f}$ & ${rdn.mean():.3f}$ & ${(mm - nn).mean():+.4f}$ & ${orc.mean():.3f}$ & ${rnd_.mean():.3f}$ \\\\")
        macros[f"strat{nm}NN"] = nn.mean()
    if srows:
      table(os.path.join(a.out, "tab_strata.tex"), "lrcccccccc",
          r"Superclass & $n_q$ & $|S_q|\!=\!1$ & mean $|S_q|$ (sub.) & mean $|S_q|$ (SCP) & $\alpha$-nDCG NN & RelDiv NN & $\Delta$MMR$_{0.7}$ & Oracle & Random", srows)

    # ------------------------------------------------------------------ re-rankers (realisable) vs NN
    rer = sorted({r for r in df.ranker.unique() if r.startswith(("mmr@", "dpp", "xquad@"))},
                 key=lambda r: (r.split("@")[0], float(r.split("@")[1]) if "@" in r else 1.0))
    def lab(r):
        if r.startswith("mmr@"): return f"MMR $\\lambda={r.split('@')[1]}$"
        if r == "dpp": return r"DPP $\theta=1$"
        if r.startswith("dpp@"): return f"DPP $\\theta={r.split('@')[1]}$"
        return f"xQuAD (pseudo-intents) $\\lambda={r.split('@')[1]}$"
    rrows = []
    for r in rer:
        dP, dR, dA, dPall = [], [], [], []
        for m in core:
            for s in sorted(df[df.method == m].seed.unique()):
                def g(rk, met, st):
                    x = df[(df.method == m) & (df.seed == s) & (df.ranker == rk) & (df.K == K) & (df.metric == met) & (df.stratum == st)]
                    return x.value.values[0] if len(x) else np.nan
                dP.append(g(r, "P", a.primary) - g("nn", "P", a.primary))
                dPall.append(g(r, "P", "all") - g("nn", "P", "all"))
                dR.append(g(r, "RelDiv_sub", a.primary) - g("nn", "RelDiv_sub", a.primary))
                dA.append(g(r, "aNDCG_sub", a.primary) - g("nn", "aNDCG_sub", a.primary))
        dP, dR, dA, dPall = map(np.array, (dP, dR, dA, dPall))
        rrows.append(f"{lab(r)} & ${dP.mean():+.3f}$ & ${dPall.mean():+.3f}$ & ${dR.mean():+.3f}$ & ${dA.mean():+.4f}$ & $[{dA.min():+.4f},\\,{dA.max():+.4f}]$ \\\\")
        macros[f"dP_{r}"] = dP.mean(); macros[f"dPall_{r}"] = dPall.mean()
        macros[f"dA_{r}"] = dA.mean(); macros[f"dAmax_{r}"] = dA.max(); macros[f"dAmin_{r}"] = dA.min()
    table(os.path.join(a.out, "tab_rerank_v2.tex"), "lccccc",
          r"Re-ranker & $\Delta$P@10 ($|S_q|\!\ge\!2$) & $\Delta$P@10 (all queries) & $\Delta$RelDiv & $\Delta\alpha$-nDCG & range over runs", rrows)

    # ------------------------------------------------------------------ oracle / shortlist
    orows = []
    for m in meths:
        cells = []
        sl = df[(df.method == m) & (df.ranker == "shortlist") & (df.metric == "SLrecall_sub") & (df.stratum == a.primary)]
        cells.append(pm(sl.sort_values("seed").value.values))
        slp = df[(df.method == m) & (df.ranker == "shortlist") & (df.metric == "SLP") & (df.stratum == "all")]
        cells.append(pm(slp.sort_values("seed").value.values))
        for rk in ["nn", "mmr@0.7", "dpp", "xquad@0.5", "xquad_cl_sub@0.5", "xquad_or_sub@0.5", "oracle_sub"]:
            v_ = val(df, m, rk, "aNDCG_sub", a.primary)
            cells.append(pm(v_) if len(v_) else "--")
        orows.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
    table(os.path.join(a.out, "tab_oracle.tex"), "lccccccccc",
          r"Encoder & SL recall & SL P & NN & MMR$_{0.7}$ & DPP & xQuAD & xQuAD$^{\mathrm{cl}}$ & xQuAD$^{\mathrm{or}}$ & Oracle", orows)
    # scenario B: baseline summary and accuracy of the family probe (on queries with |S_q|>=2)
    prow = []
    for m in meths:
        f_ = os.path.join(a.main, f"{m}_seed0.npz")
        if not os.path.exists(f_):
            continue
        z_ = np.load(f_)
        if "struct__0__fam_exact" not in z_.files:
            continue
        ns_ = z_["struct__0__nsub_sub"] >= 2
        ex, hit = z_["struct__0__fam_exact"][ns_].mean(), z_["struct__0__fam_hit"][ns_].mean()
        tp, npd, ntr = (z_[f"struct__0__fam_{k}"][ns_].sum() for k in ("tp", "npred", "ntrue"))
        f1 = 2 * tp / (npd + ntr)
        cl = val(df, m, "xquad_cl_sub@0.5", "aNDCG_sub", a.primary)
        clw = val(df, m, "xquad_clw_sub@0.5", "aNDCG_sub", a.primary)
        cP = val(df, m, "xquad_cl_sub@0.5", "P", a.primary)
        nP = val(df, m, "nn", "P", a.primary)
        prow.append(f"{NAMES[m]} & ${ex:.3f}$ & ${hit:.3f}$ & ${f1:.3f}$ & {pm(cl)} & {pm(clw)} & ${(cP - nP).mean():+.3f}$ \\\\")
        macros[f"famExact_{m}"] = ex; macros[f"famF1_{m}"] = f1
    if prow:
        table(os.path.join(a.out, "tab_probe.tex"), "lcccccc",
              r"Encoder & family exact match & family hit & family micro-F1 & $\alpha$-nDCG xQuAD$^{\mathrm{cl}}$ & $\alpha$-nDCG xQuAD$^{\mathrm{cl}}$ (prob.-weighted) & $\Delta$P@10", prow)
        clv = [val(df, m, "xquad_cl_sub@0.5", "aNDCG_sub", a.primary).mean() for m in ssl if len(val(df, m, "xquad_cl_sub@0.5", "aNDCG_sub", a.primary))]
        if clv:
            macros["clSSL"] = float(np.mean(clv))
    nn_ssl = np.mean([val(df, m, "nn", "aNDCG_sub", a.primary).mean() for m in ssl])
    or_ssl = np.mean([val(df, m, "oracle_sub", "aNDCG_sub", a.primary).mean() for m in ssl])
    macros["oracleSSL"], macros["nnSSL"] = or_ssl, nn_ssl
    macros["slrecallSSL"] = np.mean([df[(df.method == m) & (df.ranker == "shortlist") & (df.metric == "SLrecall_sub") & (df.stratum == a.primary)].value.mean() for m in ssl])

    # ------------------------------------------------------------------ alpha sensitivity
    alphas = [0, 0.25, 0.5, 0.75, 0.9]
    def aval(m, al, st=a.primary):
        met = "aNDCG_sub" if al == 0.5 else f"aNDCG{al:g}_sub"
        return val(df, m, "nn", met, st).mean()
    arows, taus = [], []
    base = [aval(m, 0.5) for m in core]
    for m in core:
        arows.append(f"{NAMES[m]} & " + " & ".join(f"${aval(m, al):.3f}$" for al in alphas) + r" \\")
    for al in alphas:
        t = kendalltau(base, [aval(m, al) for m in core])[0]
        taus.append(t)
    arows.append(r"\midrule")
    arows.append(r"Kendall $\tau$ with $\alpha=0.5$ ranking & " + " & ".join(f"${t:.2f}$" for t in taus) + r" \\")
    table(os.path.join(a.out, "tab_alpha.tex"), "lccccc", r"Encoder (NN) & $\alpha=0$ & $0.25$ & $0.5$ & $0.75$ & $0.9$", arows)
    macros["tauAlphaMin"] = float(np.nanmin(taus))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for m in ssl:
        ax.plot(alphas, [aval(m, al) for al in alphas], marker="o", ms=3, lw=1.2, label=NAMES[m])
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\alpha$-nDCG@10 (subclass)")
    ax.legend(frameon=False, fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "figures", "fig_alpha.png"), dpi=300); plt.close(fig)

    # ------------------------------------------------------------------ metric redundancy
    pts = df[(df.K == K) & (df.stratum == a.primary) & df.method.isin(core) & df.ranker.apply(lambda r: r.startswith(REAL_RANKERS_PREFIX))]
    pv = pts[pts.metric.isin(["P", "LegacyDiv", "RelDiv_sub", "Srecall_sub", "aNDCG_sub", "nDCGIA_sub", "ERRIA_sub"])].pivot_table(
        index=["method", "seed", "ranker"], columns="metric", values="value").dropna()
    mets = [("LegacyDiv", "Legacy"), ("RelDiv_sub", "RelDiv"), ("Srecall_sub", "S-recall"),
            ("nDCGIA_sub", "nDCG-IA"), ("ERRIA_sub", "ERR-IA"), ("aNDCG_sub", r"$\alpha$-nDCG")]
    crow = []
    for k_, nm in mets:
        rho_p = spearmanr(pv["P"], pv[k_])[0]
        rho_r = spearmanr(pv["RelDiv_sub"], pv[k_])[0] if k_ != "RelDiv_sub" else np.nan
        pc = partial_corr(pv[k_].values.astype(float), pv["RelDiv_sub"].values.astype(float), pv["P"].values.astype(float)) if k_ != "RelDiv_sub" else np.nan
        crow.append(f"{nm} & ${rho_p:+.2f}$ & " + (f"${rho_r:+.2f}$ & ${pc:+.2f}$" if k_ != "RelDiv_sub" else "-- & --") + r" \\")
        macros[f"rhoP_{k_}"] = rho_p
        if k_ != "RelDiv_sub":
            macros[f"pcorr_{k_}"] = pc
    table(os.path.join(a.out, "tab_metriccorr.tex"), "lccc",
          r"Measure & $\rho$ with P@10 & $\rho$ with RelDiv & partial corr.\ with RelDiv given P@10", crow)
    macros["nPointsRealisable"] = len(pv)

    # ------------------------------------------------------------------ leakage ablation
    if a.leak:
        lrows = []
        for d in a.leak:
            dl = load(d)
            lab_ = d.split("excl-")[1].split("_")[0]
            Ps = np.mean([val(dl, m, "nn", "P").mean() for m in ssl])
            ssl_l = [m for m in ssl if m in set(dl.method)]
            Ps = np.mean([val(dl, m, "nn", "P").mean() for m in ssl_l])
            As = np.mean([val(dl, m, "nn", "aNDCG_sub", "all").mean() for m in ssl_l])
            Rs = np.mean([val(dl, m, "nn", "RelDiv_sub", "all").mean() for m in ssl_l])
            lrows.append((lab_, Ps, As, Rs))
            macros[f"leakP_{lab_}"] = Ps; macros[f"leakA_{lab_}"] = As
        nice = {"patient": "query and same patient excluded (ours)", "self": "only the query itself excluded", "none": "no exclusion (query can retrieve itself)"}
        table(os.path.join(a.out, "tab_leak.tex"), "lccc", r"Exclusion rule & P@10 & $\alpha$-nDCG@10 & RelDiv@10",
              [f"{nice[l]} & ${p:.3f}$ & ${an:.3f}$ & ${r:.3f}$ \\\\" for l, p, an, r in lrows])

    # ------------------------------------------------------------------ test-vs-test comparison
    if a.test:
        dt = load(a.test)
        trow = []
        for m in meths:
            trow.append(f"{NAMES[m]} & {pm(val(df, m, 'nn', 'P', a.primary))} & {pm(val(dt, m, 'nn', 'P', a.primary))} & "
                        f"{pm(val(df, m, 'nn', 'aNDCG_sub', a.primary))} & {pm(val(dt, m, 'nn', 'aNDCG_sub', a.primary))} \\\\")
        table(os.path.join(a.out, "tab_corpus.tex"), "lcccc",
              r"Encoder & P@10 (corpus: folds 1--8) & P@10 (corpus: fold 10) & $\alpha$-nDCG (folds 1--8) & $\alpha$-nDCG (fold 10)", trow)

    # ------------------------------------------------------------------ K=5, 20 (supplementary)
    for kk in (5, 20):
        krow = []
        for m in meths:
            krow.append(f"{NAMES[m]} & {pm(val(df, m, 'nn', 'P', a.primary, kk))} & {pm(val(df, m, 'nn', 'aNDCG_sub', a.primary, kk))} & "
                        f"{pm(val(df, m, 'nn', 'RelDiv_sub', a.primary, kk))} \\\\")
        table(os.path.join(a.out, f"tab_K{kk}.tex"), "lccc", r"Encoder & P@%d ($|S_q|\!\ge\!2$) & $\alpha$-nDCG@%d ($|S_q|\!\ge\!2$) & RelDiv@%d ($|S_q|\!\ge\!2$)" % (kk, kk, kk), krow)

    # ------------------------------------------------------------------ frontier figure (publication style)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    cols = dict(zip(core, plt.cm.tab10(np.arange(len(core)))))
    lam = [0.9, 0.7, 0.5, 0.3, 0.1, 0.0]
    for m in core:
        xs, ys = [], []
        for r in ["nn"] + [f"mmr@{l}" for l in lam]:
            xs.append(val(df, m, r, "P", a.primary).mean()); ys.append(val(df, m, r, "RelDiv_sub", a.primary).mean())
        ax.plot(xs, ys, "-o", ms=2.5, lw=1, color=cols[m], label=NAMES[m].replace("$^{\\ddagger}$", "*"))
        sd = val(df, m, "nn", "P", a.primary).std(ddof=1) if len(val(df, m, "nn", "P", a.primary)) > 1 else 0
        sdy = val(df, m, "nn", "RelDiv_sub", a.primary).std(ddof=1) if len(val(df, m, "nn", "P", a.primary)) > 1 else 0
        ax.errorbar(xs[0], ys[0], xerr=sd, yerr=sdy, color=cols[m], capsize=2, lw=1)
    ax.scatter([np.mean(rnd[1])], [np.mean(rnd[4])], marker="x", color="k", zorder=5, label="Random")
    ax.set_xlabel(r"Precision@10 ($|S_q|\geq2$)"); ax.set_ylabel(r"RelDiv@10 ($|S_q|\geq2$)")
    ax.legend(frameon=False, fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "figures", "fig_frontier_v2.png"), dpi=300); plt.close(fig)

    # ------------------------------------------------------------------ macros
    def macro_name(k_):
        out = "vtwo"
        for ch in k_:
            if ch.isalpha(): out += ch
            elif ch.isdigit(): out += "ABCDEFGHIJ"[int(ch)]
            elif ch == "@": out += "At"
            elif ch == ".": out += "P"
            elif ch == "_": out += "X"
        return out
    with open(os.path.join(a.out, "numbers_ej10.tex"), "w") as f:
        for k_, v in macros.items():
            f.write(f"\\newcommand{{\\{macro_name(k_)}}}{{{int(v) if float(v).is_integer() and abs(v) >= 1 else format(float(v), '.4f')}}}\n")
    print("wrote tables/figures/macros to", a.out)
    for k_, v in macros.items():
        print(f"  {k_}: {v:.4f}")


if __name__ == "__main__":
    main()
