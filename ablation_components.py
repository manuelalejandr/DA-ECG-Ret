"""
Component ablation study.

Trains 5 variants of DA-ECG-Ret per dataset (per seed):
    base            : Component A off, Component B off (i.e. the bug-fixed
                      v1 loss + beta warm-up curriculum, but no
                      pseudo-cluster gate and no physiological auxiliary
                      task)
    A_only          : pseudo-cluster gate ON,  physio task OFF
    B_only          : pseudo-cluster gate OFF, physio task ON, beta UNCHANGED
                      (cfg.beta, default 1.0) -- L_ret is still fully active
                      here, just ungated. NOT a clean "physio-only" control.
    phys_only_beta0 : pseudo-cluster gate OFF, physio task ON, beta=0.0 --
                      L_ret contributes NOTHING to the gradient. This is the
                      clean isolation of Component B that the co-author
                      review (Sept 2026) pointed out was missing: without
                      it, no result can be attributed to "diversity-aware
                      retrieval" as opposed to "SimCLR + an auxiliary task".
    A_and_B         : both ON, beta UNCHANGED (the current DA-ECG-Ret v2
                      default)

plus one SimCLR reference per (dataset, seed), all trained IN-MEMORY (no
checkpoint I/O) so the whole sweep is fast. This answers "which component
is responsible for the AUROC gain / the Pareto-frontier gain we saw in the
full benchmark?" — the full benchmark only ever ran the full A_and_B
configuration, so we cannot currently tell whether the physiological
auxiliary task (Component B) or the cluster-gated redundancy term
(Component A) — or their interaction — is doing the work. Compare
phys_only_beta0 against B_only to further isolate whether an ACTIVE (if
ungated) L_ret matters even alongside Component B.

Usage:
    python ablation_components.py \\
        --datasets ucr:ECG200 ucr:ECG5000 ucr:ECGFiveDays ucr:TwoLeadECG \\
                   ucr:CinCECGTorso ucr:NonInvasiveFetalECGThorax1 \\
                   ucr:NonInvasiveFetalECGThorax2 mitbih \\
        --seeds 0 1 2 --epochs 100
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
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

VARIANTS = {
    "base": dict(use_cluster_gate=False, use_physio_aux_task=False),
    "A_only": dict(use_cluster_gate=True, use_physio_aux_task=False),
    "B_only": dict(use_cluster_gate=False, use_physio_aux_task=True),
    "A_and_B": dict(use_cluster_gate=True, use_physio_aux_task=True),
    # NEW (post co-author review, Sept 2026): the one condition the previous
    # ablation never tested. All four variants above still train with the
    # default cfg.beta (i.e. L_ret's retrieval-aware pressure is ALWAYS
    # present, even in "B_only"), so no variant isolated "the physiological
    # auxiliary task alone, with ZERO retrieval-aware pressure" from "the
    # physiological auxiliary task plus a still-active but ungated L_ret".
    # phys_only_beta0 sets beta=0.0 explicitly (L_ret contributes nothing to
    # the gradient at all -- InfoNCE + gamma_physio * L_phys only) so the
    # paper can honestly attribute any gain to Component B in isolation.
    "phys_only_beta0": dict(use_cluster_gate=False, use_physio_aux_task=True, beta=0.0),
}
METRICS = ["precision_at_k", "normalized_diversity_at_k", "auroc_macro", "pareto_auc"]


def run(args):
    device = resolve_device(verbose=True)
    rows = []

    for spec in args.datasets:
        kind, name = parse_dataset_spec(spec)
        tag = dataset_tag(kind, name)
        ns = make_args_namespace(kind, name, args.data_root)

        base_cfg = Config()
        base_cfg.dataset = kind
        base_cfg.ucr_name = ns.ucr_name
        base_cfg.data_root = args.data_root
        base_cfg.encoder_name = args.encoder
        base_cfg.epochs = args.epochs
        base_cfg.batch_size = args.batch_size

        print(f"\n{'='*70}\nDataset: {tag}\n{'='*70}")
        train_base, test_base, n_classes, _ = get_dataset_splits(ns, base_cfg)
        augment = default_ecg_augmentation(base_cfg.sampling_rate)
        train_ds = ContrastiveWrapper(train_base, augment)
        bs = effective_batch_size(base_cfg.batch_size, len(train_ds))
        base_cfg.batch_size = bs

        for seed in args.seeds:
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                       num_workers=base_cfg.num_workers, drop_last=True)

            # one SimCLR reference per (dataset, seed), shared across all 4 variants
            simclr_seed = base_cfg.seed + sum(ord(c) for c in "simclr") + seed * 1000
            set_seed(simclr_seed)
            simclr_modules, _ = train_modules_in_memory(
                "simclr", base_cfg, train_loader, device, verbose_prefix=f"{tag}/seed{seed}/simclr(ref)"
            )
            simclr_metrics = evaluate_embedding(
                simclr_modules["embedding_model"], train_base, test_base, n_classes, base_cfg, device,
                k=args.k, lambda_sweep=list(base_cfg.lambda_sweep),
            )
            rows.append({**simclr_metrics, "dataset": tag, "seed": seed, "variant": "simclr"})

            for variant_name, flags in VARIANTS.items():
                cfg = Config()
                cfg.__dict__.update(base_cfg.__dict__)
                cfg.use_cluster_gate = flags["use_cluster_gate"]
                cfg.use_physio_aux_task = flags["use_physio_aux_task"]
                if "beta" in flags:
                    cfg.beta = flags["beta"]

                var_seed = base_cfg.seed + sum(ord(c) for c in "da_ecg_ret") + seed * 1000
                set_seed(var_seed)
                train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                           num_workers=cfg.num_workers, drop_last=True)
                modules, _ = train_modules_in_memory(
                    "da_ecg_ret", cfg, train_loader, device,
                    verbose_prefix=f"{tag}/seed{seed}/{variant_name}",
                )
                metrics = evaluate_embedding(
                    modules["embedding_model"], train_base, test_base, n_classes, cfg, device,
                    k=args.k, lambda_sweep=list(cfg.lambda_sweep),
                )
                metrics["pareto_auc_gap_vs_simclr"] = metrics["pareto_auc"] - simclr_metrics["pareto_auc"]
                metrics.update({"dataset": tag, "seed": seed, "variant": variant_name})
                rows.append(metrics)
                print(f"  seed={seed} {variant_name:10s} "
                      f"prec={metrics['precision_at_k']:.4f} "
                      f"div={metrics['normalized_diversity_at_k']:.4f} "
                      f"auroc={metrics['auroc_macro']:.4f} "
                      f"pareto_auc={metrics['pareto_auc']:.5f} "
                      f"(gap vs simclr: {metrics['pareto_auc_gap_vs_simclr']:+.5f})")

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    """Mean +/- std across seeds, per dataset, per variant (including the
    simclr reference), for every metric."""
    agg = df.groupby(["dataset", "variant"])[
        [m for m in METRICS if m in df.columns] + (["pareto_auc_gap_vs_simclr"] if "pareto_auc_gap_vs_simclr" in df.columns else [])
    ].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def plot_component_contributions(summary: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    all_variants = ["simclr", "base", "A_only", "B_only", "phys_only_beta0", "A_and_B"]
    present = set(summary["variant"].unique())
    variants = [v for v in all_variants if v in present]
    colors = {"simclr": "#999999", "base": "#4C72B0", "A_only": "#DD8452",
              "B_only": "#55A868", "phys_only_beta0": "#8172B2", "A_and_B": "#C44E52"}

    for metric in ["auroc_macro", "pareto_auc"]:
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        if mean_col not in summary.columns:
            continue
        datasets = summary["dataset"].unique()
        fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(datasets) + 2), 5))
        n_variants = len(variants)
        width = 0.8 / n_variants
        center = (n_variants - 1) / 2
        x = np.arange(len(datasets))
        for i, variant in enumerate(variants):
            sub = summary[summary["variant"] == variant].set_index("dataset")
            sub = sub.reindex(datasets)
            means = sub[mean_col].values
            stds = sub[std_col].values
            ax.bar(x + (i - center) * width, means, width, yerr=stds, capsize=3,
                   label=variant, color=colors.get(variant))
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f"Component ablation: {metric}\n(mean +/- std across seeds)", fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        path = os.path.join(out_dir, f"ablation_{metric}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def main():
    global VARIANTS
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output_dir", type=str, default="./ablation_outputs")
    parser.add_argument("--variants", type=str, nargs="+", default=None,
                         choices=list(VARIANTS.keys()),
                         help="Subset of VARIANTS to train (default: all 5). "
                              "E.g. --variants base B_only phys_only_beta0 A_and_B "
                              "to skip A_only, which two independent runs have already "
                              "shown adds nothing -- saves ~1/5 of the training time.")
    args = parser.parse_args()

    if args.variants is not None:
        VARIANTS = {k: v for k, v in VARIANTS.items() if k in args.variants}
        print(f"Restricting to variants: {list(VARIANTS.keys())}")

    t0 = time.time()
    df = run(args)
    os.makedirs(args.output_dir, exist_ok=True)

    raw_path = os.path.join(args.output_dir, "ablation_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nSaved {raw_path}")

    summary = summarize(df)
    summary_path = os.path.join(args.output_dir, "ablation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")
    print("\n=== Component ablation summary (mean +/- std across seeds) ===")
    print(summary.to_string(index=False))

    # cross-dataset average: which variant wins most often on each metric?
    print("\n=== Cross-dataset average (mean over datasets of the per-dataset seed-mean) ===")
    per_ds_mean = df.groupby(["dataset", "variant"])[METRICS].mean().reset_index()
    overall = per_ds_mean.groupby("variant")[METRICS].mean()
    print(overall.to_string())

    plot_component_contributions(summary, os.path.join(args.output_dir, "figures"))

    print(f"\nTotal time: {(time.time() - t0)/60:.1f} min")

    import zipfile
    zip_path = os.path.join(os.path.dirname(os.path.normpath(args.output_dir)) or ".",
                             "ablation_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(args.output_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                zf.write(full_path, os.path.relpath(full_path, os.path.dirname(args.output_dir)))
    print(f"Packaged ablation results into: {zip_path}")


if __name__ == "__main__":
    main()
