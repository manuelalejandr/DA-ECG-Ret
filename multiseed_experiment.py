"""
Multi-seed confirmatory experiment.

The full 8-dataset benchmark trained each (dataset, method) pair with a
SINGLE fixed seed. A single run cannot distinguish "DA-ECG-Ret is really a
bit better" from "this particular random initialization happened to land
well" — exactly the caveat flagged after reviewing the v2 full-benchmark
results (6/8 datasets won on Pareto-frontier AUC, but a paired Wilcoxon
across only 8 DATASETS gave p=0.25, and we have no within-dataset variance
estimate at all).

This script re-trains SimCLR and DA-ECG-Ret with several random seeds per
dataset, giving:
  1. A per-dataset mean +/- std for each metric, so a "6/8 datasets won" can
     be checked against "was the win bigger than the run-to-run noise?"
  2. A per-dataset paired test across seeds (t-test; Wilcoxon needs >= 6
     seeds to ever reach p<0.05 two-sided, so with the default 5 seeds a
     paired t-test is used and reported alongside the raw Wilcoxon p for
     transparency).
  3. A LESS NOISY input to the across-dataset test: instead of one sample
     per dataset (the original benchmark), we use the per-dataset MEAN
     across seeds, so the across-dataset Wilcoxon test is no longer
     contaminated by single-run noise on either side.

Usage:
    python multiseed_experiment.py \\
        --datasets ucr:ECG200 ucr:ECG5000 ucr:ECGFiveDays ucr:TwoLeadECG \\
                   ucr:CinCECGTorso ucr:NonInvasiveFetalECGThorax1 \\
                   ucr:NonInvasiveFetalECGThorax2 mitbih \\
        --seeds 0 1 2 3 4 --epochs 100
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from device_utils import resolve_device
from data.augmentations import default_ecg_augmentation
from data.common import ContrastiveWrapper, get_dataset_splits, effective_batch_size
from train import set_seed, train_modules_in_memory
from experiment_utils import evaluate_embedding, parse_dataset_spec, dataset_tag, make_args_namespace

METRICS = ["precision_at_k", "normalized_diversity_at_k", "auroc_macro", "pareto_auc"]


def run(args):
    device = resolve_device(verbose=True)
    rows = []

    for spec in args.datasets:
        kind, name = parse_dataset_spec(spec)
        tag = dataset_tag(kind, name)
        ns = make_args_namespace(kind, name, args.data_root)

        cfg = Config()
        cfg.dataset = kind
        cfg.ucr_name = ns.ucr_name
        cfg.data_root = args.data_root
        cfg.encoder_name = args.encoder
        cfg.epochs = args.epochs
        cfg.batch_size = args.batch_size

        print(f"\n{'='*70}\nDataset: {tag}\n{'='*70}")
        train_base, test_base, n_classes, _ = get_dataset_splits(ns, cfg)
        augment = default_ecg_augmentation(cfg.sampling_rate)
        train_ds = ContrastiveWrapper(train_base, augment)
        bs = effective_batch_size(cfg.batch_size, len(train_ds))
        cfg.batch_size = bs

        for seed in args.seeds:
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                       num_workers=cfg.num_workers, drop_last=True)
            for method in args.methods:
                method_seed = cfg.seed + sum(ord(c) for c in method) + seed * 1000
                set_seed(method_seed)
                t0 = time.time()
                modules, _ = train_modules_in_memory(
                    method, cfg, train_loader, device, verbose_prefix=f"{tag}/seed{seed}/{method}"
                )
                metrics = evaluate_embedding(
                    modules["embedding_model"], train_base, test_base, n_classes, cfg, device,
                    k=args.k, lambda_sweep=list(cfg.lambda_sweep),
                )
                metrics.update({"dataset": tag, "seed": seed, "method": method,
                                 "train_time_sec": time.time() - t0})
                rows.append(metrics)
                print(f"  seed={seed} {method:12s} "
                      f"prec={metrics['precision_at_k']:.4f} "
                      f"div={metrics['normalized_diversity_at_k']:.4f} "
                      f"auroc={metrics['auroc_macro']:.4f} "
                      f"pareto_auc={metrics['pareto_auc']:.5f} "
                      f"({metrics['train_time_sec']:.1f}s)")

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, methods=("simclr", "da_ecg_ret")):
    """Per-dataset mean/std + paired test across seeds, for each metric."""
    rows = []
    for ds, sub in df.groupby("dataset"):
        pivot = {m: sub[sub["method"] == m].sort_values("seed") for m in methods}
        for metric in METRICS:
            vals = {m: pivot[m][metric].values for m in methods}
            a, b = vals[methods[1]], vals[methods[0]]  # da_ecg_ret, simclr
            diff = a - b
            row = {
                "dataset": ds, "metric": metric,
                f"{methods[0]}_mean": b.mean(), f"{methods[0]}_std": b.std(),
                f"{methods[1]}_mean": a.mean(), f"{methods[1]}_std": a.std(),
                "mean_diff": diff.mean(),
            }
            n = len(diff)
            if n >= 2 and np.std(diff) > 1e-12:
                t_stat, t_p = ttest_rel(a, b)
                row["ttest_p"] = t_p
            else:
                row["ttest_p"] = float("nan")
            if n >= 6:
                try:
                    _, w_p = wilcoxon(diff)
                    row["wilcoxon_p"] = w_p
                except ValueError:
                    row["wilcoxon_p"] = float("nan")
            else:
                row["wilcoxon_p"] = float("nan")  # not enough seeds for Wilcoxon to ever be significant
            rows.append(row)
    return pd.DataFrame(rows)


def cross_dataset_test(df: pd.DataFrame, methods=("simclr", "da_ecg_ret")):
    """Uses the PER-DATASET MEAN (across seeds) as the paired sample for
    the across-dataset test — much less noisy than the single-seed values
    used in the original 8-dataset benchmark's aggregate.py."""
    means = df.groupby(["dataset", "method"])[METRICS].mean().reset_index()
    rows = []
    for metric in METRICS:
        pivot = means.pivot(index="dataset", columns="method", values=metric)
        if methods[0] not in pivot.columns or methods[1] not in pivot.columns:
            continue
        diff = pivot[methods[1]] - pivot[methods[0]]
        n = len(diff)
        try:
            stat, p = wilcoxon(diff)
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append({
            "metric": metric, "n_datasets": n, "mean_diff": diff.mean(),
            "n_datasets_won": int((diff > 0).sum()), "wilcoxon_p": p,
        })
    return pd.DataFrame(rows)


def plot_seed_variance(df: pd.DataFrame, out_dir: str, methods=("simclr", "da_ecg_ret")):
    os.makedirs(out_dir, exist_ok=True)
    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 4), squeeze=False)
    axes = axes[0]
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        for i, m in enumerate(methods):
            vals = sub[sub["method"] == m]["pareto_auc"].values
            ax.scatter([i] * len(vals), vals, alpha=0.7)
            ax.errorbar([i], [vals.mean()], yerr=[vals.std()], fmt="o", color="black", capsize=4)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        ax.set_title(ds, fontsize=9)
    axes[0].set_ylabel("Pareto-frontier AUC")
    fig.suptitle("Per-seed Pareto-AUC: individual runs (dots) + mean/std (black)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(out_dir, "seed_variance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                         default=["ucr:ECG200", "ucr:ECG5000", "ucr:ECGFiveDays", "ucr:TwoLeadECG",
                                  "ucr:CinCECGTorso", "ucr:NonInvasiveFetalECGThorax1",
                                  "ucr:NonInvasiveFetalECGThorax2", "mitbih"])
    parser.add_argument("--data_root", type=str, default="./ptbxl_data")
    parser.add_argument("--encoder", choices=["transformer", "resnet18", "fcn"], default="transformer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", default=["simclr", "da_ecg_ret"])
    parser.add_argument("--output_dir", type=str, default="./multiseed_outputs")
    args = parser.parse_args()

    t0 = time.time()
    df = run(args)
    os.makedirs(args.output_dir, exist_ok=True)

    raw_path = os.path.join(args.output_dir, "multiseed_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nSaved {raw_path}")

    summary = summarize(df, methods=tuple(args.methods))
    summary_path = os.path.join(args.output_dir, "multiseed_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")
    print("\n=== Per-dataset summary (mean +/- std across seeds, paired test) ===")
    print(summary.to_string(index=False))

    cross = cross_dataset_test(df, methods=tuple(args.methods))
    cross_path = os.path.join(args.output_dir, "multiseed_cross_dataset_test.csv")
    cross.to_csv(cross_path, index=False)
    print(f"\nSaved {cross_path}")
    print("\n=== Across-dataset test (using per-dataset seed-mean as the paired sample) ===")
    print(cross.to_string(index=False))

    plot_seed_variance(df, os.path.join(args.output_dir, "figures"), methods=tuple(args.methods))

    print(f"\nTotal time: {(time.time() - t0)/60:.1f} min")

    import zipfile
    zip_path = os.path.join(os.path.dirname(os.path.normpath(args.output_dir)) or ".",
                             "multiseed_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(args.output_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                zf.write(full_path, os.path.relpath(full_path, os.path.dirname(args.output_dir)))
    print(f"Packaged multi-seed results into: {zip_path}")


if __name__ == "__main__":
    main()
