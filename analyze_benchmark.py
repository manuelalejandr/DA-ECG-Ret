"""
Statistics + figures for the diversity-aware retrieval benchmark.

Reads <out>/metrics/<tag>/*.npz (per-query metrics written by
ptbxl_benchmark.py) and produces, in <out>/analysis/<tag>/:

  main_table.csv        metric means with 95% CIs per method (NN, MMR@0.7, DPP; K=10)
  paired_vs_simclr.csv  paired differences vs SimCLR with 95% CIs
  reranker_effect.csv   what MMR / DPP do to each encoder (vs plain NN)
  metric_coupling.csv   Spearman(P@K, each diversity metric): the construct-validity test
  fig_frontier_*.png    precision-vs-diversity trade-off per method (lambda sweep)
  fig_forest_*.png      paired differences vs SimCLR (forest plot)

Uncertainty (two sources, reported separately):
  * TEST-SET sampling: cluster bootstrap over PATIENTS (2000 resamples); the
    per-query metric is first averaged over training seeds. Paired
    differences reuse the same resamples for both methods.
  * TRAINING randomness: std across seeds of the full-test-set mean
    ('seed_std'; needs >=2 seeds, meaningful from ~3).

Usage:  python analyze_benchmark.py --out ./bench_outputs --datasets ptbxl
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = 2000
HEADLINE = ["P", "Pfine_sub", "RelDiv_sub", "aNDCG_sub", "Pfine_code", "RelDiv_code", "aNDCG_code", "LegacyDiv"]
RANKERS_MAIN = ["nn", "mmr@0.7", "dpp"]


def fn_to_key(k):
    return k.replace("__", "|")


def load_runs(out, tag):
    runs = {}
    for f in sorted(glob.glob(os.path.join(out, "metrics", tag, "*_seed*.npz"))):
        base = os.path.basename(f)[:-4]
        method, seed = base.rsplit("_seed", 1)
        z = np.load(f)
        runs[(method, int(seed))] = {fn_to_key(k): z[k] for k in z.files}
    if not runs:
        raise SystemExit(f"no metrics found for {tag} under {out}/metrics")
    ref = next(iter(runs.values()))
    for r in runs.values():
        assert np.array_equal(r["query_idx"], ref["query_idx"]), "runs evaluated on different queries"
    return runs, ref["patient"]


class Boot:
    """Patient-cluster bootstrap on per-query vectors."""

    def __init__(self, patients, seed=0):
        _, inv = np.unique(patients, return_inverse=True)
        self.inv, self.np_ = inv, inv.max() + 1
        rng = np.random.RandomState(seed)
        draws = rng.randint(0, self.np_, size=(B, self.np_))
        self.W = np.stack([np.bincount(d, minlength=self.np_) for d in draws]).astype(np.float64)  # (B, P)

    def _agg(self, v):
        ok = np.isfinite(v)
        s = np.bincount(self.inv[ok], weights=v[ok], minlength=self.np_)
        c = np.bincount(self.inv[ok], minlength=self.np_).astype(float)
        return s, c

    def mean_ci(self, v):
        s, c = self._agg(v)
        est = s.sum() / max(c.sum(), 1)
        bs = (self.W @ s) / np.maximum(self.W @ c, 1)
        return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    def diff_ci(self, v1, v2):
        # paired: only queries valid for both
        ok = np.isfinite(v1) & np.isfinite(v2)
        d = np.where(ok, v1 - v2, np.nan)
        est, lo, hi = self.mean_ci(d)
        s, c = self._agg(d)
        bs = (self.W @ s) / np.maximum(self.W @ c, 1)
        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        return est, lo, hi, min(1.0, p)


def seed_avg(runs, method, key):
    arrs = [r[key] for (m, s), r in runs.items() if m == method and key in r]
    return np.nanmean(np.stack(arrs), axis=0) if arrs else None


def seed_means(runs, method, key):
    return [float(np.nanmean(r[key])) for (m, s), r in runs.items() if m == method and key in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./bench_outputs")
    ap.add_argument("--datasets", nargs="+", default=["ptbxl"])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--reference", default="simclr")
    args = ap.parse_args()

    for tag in args.datasets:
        runs, patients = load_runs(args.out, tag)
        methods = sorted({m for m, _ in runs})
        seeds = sorted({s for _, s in runs})
        print(f"\n===== {tag}: methods={methods} seeds={seeds} queries={len(patients)} =====")
        boot = Boot(patients)
        ad = os.path.join(args.out, "analysis", tag)
        os.makedirs(ad, exist_ok=True)
        K = args.K
        sample = next(iter(runs.values()))
        metrics = [m for m in HEADLINE if f"nn|{K}|{m}" in sample]

        # ---------------- main table ----------------
        rows = []
        for m in methods:
            for r in RANKERS_MAIN + ["rand"]:
                if m != methods[0] and r == "rand":
                    continue
                for met in metrics:
                    key = f"{r}|{K}|{met}"
                    v = seed_avg(runs, m, key)
                    if v is None:
                        continue
                    est, lo, hi = boot.mean_ci(v)
                    sm = seed_means(runs, m, key)
                    rows.append(dict(method=m, ranker=r, K=K, metric=met, mean=est, ci_lo=lo, ci_hi=hi,
                                     seed_std=float(np.std(sm, ddof=1)) if len(sm) > 1 else np.nan,
                                     n_seeds=len(sm)))
        main_t = pd.DataFrame(rows)
        main_t.to_csv(os.path.join(ad, "main_table.csv"), index=False)

        # ---------------- paired vs reference ----------------
        rows = []
        if args.reference in methods:
            for m in methods:
                if m == args.reference:
                    continue
                for r in RANKERS_MAIN:
                    for met in metrics:
                        key = f"{r}|{K}|{met}"
                        a, b = seed_avg(runs, m, key), seed_avg(runs, args.reference, key)
                        if a is None or b is None:
                            continue
                        d, lo, hi, p = boot.diff_ci(a, b)
                        rows.append(dict(method=m, vs=args.reference, ranker=r, K=K, metric=met,
                                         diff=d, ci_lo=lo, ci_hi=hi, p_boot=p, excludes_0=bool(lo > 0 or hi < 0)))
        paired = pd.DataFrame(rows)
        paired.to_csv(os.path.join(ad, "paired_vs_simclr.csv"), index=False)

        # ---------------- reranker effect ----------------
        rows = []
        rerankers = sorted({k.split("|")[0] for k in sample if "|" in k} - {"nn", "rand"})
        for m in methods:
            for rr in rerankers:
                for met in metrics:
                    a, b = seed_avg(runs, m, f"{rr}|{K}|{met}"), seed_avg(runs, m, f"nn|{K}|{met}")
                    if a is None or b is None:
                        continue
                    d, lo, hi, p = boot.diff_ci(a, b)
                    rows.append(dict(method=m, ranker=rr, K=K, metric=met, diff_vs_nn=d, ci_lo=lo, ci_hi=hi))
        pd.DataFrame(rows).to_csv(os.path.join(ad, "reranker_effect.csv"), index=False)

        # ---------------- metric coupling (construct validity) ----------------
        rows = []
        pts = []   # (method, seed, ranker) -> metric means
        for (m, s), r in runs.items():
            for rk in sorted({k.split("|")[0] for k in r if "|" in k} - {"rand"}):
                d = {"method": m, "seed": s, "ranker": rk}
                for met in ["P"] + metrics:
                    k_ = f"{rk}|{K}|{met}"
                    if k_ in r:
                        d[met] = float(np.nanmean(r[k_]))
                pts.append(d)
        pts = pd.DataFrame(pts)
        for scope, sub in [("encoders (NN only)", pts[pts.ranker == "nn"]),
                           ("all encoder x re-ranker points", pts)]:
            for met in metrics:
                if met == "P" or met not in sub or len(sub) < 4:
                    continue
                rho = spearmanr(sub["P"], sub[met], nan_policy="omit")[0]
                rows.append(dict(scope=scope, metric=met, spearman_with_P=rho, n_points=len(sub)))
        coup = pd.DataFrame(rows)
        coup.to_csv(os.path.join(ad, "metric_coupling.csv"), index=False)

        # ---------------- figures ----------------
        for lvl in ["sub", "code"]:
            if f"nn|{K}|RelDiv_{lvl}" not in sample:
                continue
            fig, ax = plt.subplots(figsize=(6.4, 5))
            cm = plt.get_cmap("tab10")
            lam_r = sorted([r for r in rerankers if r.startswith("mmr@")], key=lambda x: float(x[4:]))
            for i, m in enumerate(methods):
                xs, ys = [], []
                for r in ["nn"] + lam_r[::-1]:
                    a, b = seed_avg(runs, m, f"{r}|{K}|P"), seed_avg(runs, m, f"{r}|{K}|RelDiv_{lvl}")
                    if a is None:
                        continue
                    xs.append(np.nanmean(a)); ys.append(np.nanmean(b))
                ax.plot(xs, ys, "-o", ms=3.5, lw=1.2, color=cm(i), label=m)
            rnd = [np.nanmean(seed_avg(runs, methods[0], f"rand|{K}|{k}")) for k in ("P", f"RelDiv_{lvl}")]
            ax.scatter(*rnd, marker="x", color="k", label="random", zorder=5)
            ax.set_xlabel(f"Precision@{K} (shares a superclass)")
            ax.set_ylabel(f"RelDiv@{K} ({lvl}-level, within query family)")
            ax.set_title(f"{tag}: precision / diversity trade-off\n(NN -> MMR lambda 0.9..0.0; each line one encoder)")
            ax.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(ad, f"fig_frontier_{lvl}.png"), dpi=300)
            plt.close(fig)

        if len(paired):
            for lvl in ["sub", "code"]:
                met = f"aNDCG_{lvl}"
                sub = paired[(paired.metric == met)]
                if sub.empty:
                    continue
                fig, ax = plt.subplots(figsize=(6.4, 0.35 * len(sub) + 1.5))
                labels = [f"{r.method} [{r.ranker}]" for r in sub.itertuples()]
                y = np.arange(len(sub))[::-1]
                ax.errorbar(sub["diff"], y, xerr=[sub["diff"] - sub["ci_lo"], sub["ci_hi"] - sub["diff"]],
                            fmt="o", color="C0", capsize=2)
                ax.axvline(0, color="gray", lw=0.8)
                ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
                ax.set_xlabel(f"alpha-nDCG@{K} ({lvl}) difference vs {args.reference} (95% patient-bootstrap CI)")
                fig.tight_layout()
                fig.savefig(os.path.join(ad, f"fig_forest_{lvl}.png"), dpi=300)
                plt.close(fig)

        # ---------------- console summary ----------------
        pd.set_option("display.width", 200)
        print("\n-- Main table (NN, K=%d), mean [95%% CI] --" % K)
        nn = main_t[main_t.ranker == "nn"]
        for met in metrics:
            sub = nn[nn.metric == met].sort_values("mean", ascending=False)
            print(f"  {met}: " + " | ".join(f"{r.method} {r.mean:.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}]" for r in sub.itertuples()))
        print("\n-- Metric coupling with Precision (Spearman) --")
        print(coup.to_string(index=False))
        if len(paired):
            sig = paired[paired.excludes_0 & paired.metric.str.startswith("aNDCG")]
            print(f"\n-- Paired vs {args.reference}: aNDCG differences whose CI excludes 0: {len(sig)} of "
                  f"{(paired.metric.str.startswith('aNDCG')).sum()} --")
            if len(sig):
                print(sig[["method", "ranker", "metric", "diff", "ci_lo", "ci_hi"]].to_string(index=False))
        print(f"\nSaved tables and figures to {ad}")


if __name__ == "__main__":
    main()
