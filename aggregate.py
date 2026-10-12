"""
Cross-dataset aggregation and statistical testing.

Running on a SINGLE dataset only tells you whether DA-ECG-Ret beats the
baselines on that one problem, which could easily be a fluke of that
dataset's class structure. Since we now train/evaluate on several
independent, unrelated ECG problems (the UCR/UEA archive is specifically
designed to allow this kind of aggregate benchmarking), we can instead ask
the much stronger question: "does DA-ECG-Ret win MORE OFTEN THAN CHANCE,
across datasets?" This script answers that with the two standard tools
from the time-series-classification benchmarking literature:

  1. Average rank per method (per metric), across datasets — the same
     summary used to build critical-difference diagrams (Demsar, 2006).
  2. Wilcoxon signed-rank test, DA-ECG-Ret vs. each baseline, on paired
     per-dataset metric values — tests whether the *median* difference
     across datasets is significantly different from zero (non-parametric,
     appropriate for a handful of datasets where a t-test's normality
     assumption is shaky).

Run after evaluate.py has produced outputs/<dataset_tag>/results/retrieval_results.csv
for every dataset you trained on:

    python aggregate.py --output_dir ./outputs --dataset_tags ucr_ECG200 ucr_ECG5000 ucr_TwoLeadECG mitbih

Interpretation caveat (be explicit about this in the paper): Wilcoxon with
fewer than ~6-8 datasets has low power, so a non-significant p-value with
only 3-4 datasets means "inconclusive", not "no effect". Run on as many of
the 7 UCR ECG problems + MIT-BIH as you can (8 datasets total) to get a
meaningful test; PTB-XL/synthetic can be reported separately as additional
single-dataset evidence rather than folded into this test.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_LABELS = {
    "simclr": "SimCLR", "clocs_lite": "CLOCS-lite", "ts2vec": "TS2Vec",
    "tfc": "TF-C", "softclt": "SoftCLT", "da_ecg_ret": "DA-ECG-Ret (proposed)",
}
BASELINES = ["simclr", "clocs_lite", "ts2vec", "tfc", "softclt"]
PROPOSED = "da_ecg_ret"
METRICS = ["Precision@K", "NormalizedDiversity@K", "MMR-Score", "AUROC_macro"]


def load_all_results(output_dir: str, dataset_tags):
    frames = []
    for tag in dataset_tags:
        path = os.path.join(output_dir, tag, "results", "retrieval_results.csv")
        if not os.path.exists(path):
            print(f"[skip] no results for dataset '{tag}' at {path}")
            continue
        df = pd.read_csv(path)
        df["dataset"] = tag
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No per-dataset results found. Run evaluate.py first.")
    return pd.concat(frames, ignore_index=True)


def per_dataset_method_summary(df: pd.DataFrame, rerank: str = "nn") -> pd.DataFrame:
    """One row per (dataset, method): metrics averaged over K, restricted to
    a re-ranking strategy available for ALL methods (NN by default, since
    only simclr/da_ecg_ret currently support mmr)."""
    sub = df[df["rerank"] == rerank]
    agg = sub.groupby(["dataset", "method"])[METRICS].mean().reset_index()
    return agg


def average_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    """For each metric, rank methods WITHIN each dataset (1 = best), then
    average the rank across datasets."""
    rank_frames = []
    for metric in METRICS:
        pivot = summary.pivot(index="dataset", columns="method", values=metric)
        ranks = pivot.rank(axis=1, ascending=False, method="average")  # higher metric = better -> rank 1
        mean_rank = ranks.mean(axis=0).rename(metric)
        rank_frames.append(mean_rank)
    return pd.concat(rank_frames, axis=1)


def wilcoxon_tests(summary: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test: DA-ECG-Ret vs. each baseline, per
    metric, using one paired sample per dataset."""
    rows = []
    for metric in METRICS:
        pivot = summary.pivot(index="dataset", columns="method", values=metric)
        if PROPOSED not in pivot.columns:
            continue
        proposed_vals = pivot[PROPOSED]
        for base in BASELINES:
            if base not in pivot.columns:
                continue
            paired = pivot[[PROPOSED, base]].dropna()
            if len(paired) < 2:
                rows.append({"metric": metric, "baseline": base, "n_datasets": len(paired),
                             "mean_diff": np.nan, "wilcoxon_p": np.nan,
                             "note": "too few paired datasets for a test"})
                continue
            diff = paired[PROPOSED] - paired[base]
            try:
                stat, p = wilcoxon(diff)
            except ValueError:
                # all differences are zero, or n too small for the normal approx
                stat, p = np.nan, np.nan
            rows.append({
                "metric": metric, "baseline": base, "n_datasets": len(paired),
                "mean_diff": float(diff.mean()), "wilcoxon_p": p,
                "note": "significant at 0.05" if (p is not None and not np.isnan(p) and p < 0.05) else "",
            })
    return pd.DataFrame(rows)


def plot_average_ranks(ranks: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, len(ranks.columns), figsize=(4.5 * len(ranks.columns), 4), squeeze=False)
    axes = axes[0]
    for ax, metric in zip(axes, ranks.columns):
        vals = ranks[metric].reindex([m for m in METHOD_LABELS if m in ranks.index])
        colors = ["#55A868" if m == PROPOSED else "#4C72B0" for m in vals.index]
        ax.bar([METHOD_LABELS[m] for m in vals.index], vals.values, color=colors)
        ax.set_title(metric)
        ax.set_ylabel("Average rank (1 = best)")
        ax.invert_yaxis()
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.suptitle("Average rank across datasets (lower is better)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--dataset_tags", nargs="+", required=True,
                         help="e.g. ucr_ECG200 ucr_ECG5000 ucr_TwoLeadECG mitbih")
    parser.add_argument("--rerank", type=str, default="nn",
                         help="re-ranking strategy to compare across ALL methods (default: nn, "
                              "since only simclr/da_ecg_ret currently support mmr)")
    args = parser.parse_args()

    df = load_all_results(args.output_dir, args.dataset_tags)
    summary = per_dataset_method_summary(df, rerank=args.rerank)

    out_dir = os.path.join(args.output_dir, "summary")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "per_dataset_method_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    ranks = average_ranks(summary)
    ranks_path = os.path.join(out_dir, "average_ranks.csv")
    ranks.to_csv(ranks_path)
    print(f"Saved {ranks_path}")
    print("\nAverage ranks (lower = better):")
    print(ranks.to_string())

    tests = wilcoxon_tests(summary)
    tests_path = os.path.join(out_dir, "wilcoxon_tests.csv")
    tests.to_csv(tests_path, index=False)
    print(f"\nSaved {tests_path}")
    print("\nWilcoxon signed-rank tests (DA-ECG-Ret vs. each baseline):")
    print(tests.to_string(index=False))

    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    plot_average_ranks(ranks, os.path.join(out_dir, "figures", "average_rank_bar.png"))


if __name__ == "__main__":
    main()
