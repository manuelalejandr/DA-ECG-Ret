"""
Diagnostic hyperparameter sweep for DA-ECG-Ret, built after the first full
benchmark run showed DA-ECG-Ret statistically indistinguishable from plain
SimCLR (see the "Results discussion" in the project README / your
conversation with Claude). Two real bugs were found and fixed in the
process (see losses/mmr_loss.py and losses/ts2vec_loss.py); this script
answers the remaining open question empirically: **at what beta/lambda
does the (now-fixed) retrieval-aware loss actually start to move the
Pareto frontier**, and does the fix alone already help at the original
beta=0.5?

For each dataset, this script:
  1. Trains ONE SimCLR reference encoder (the Pareto-frontier baseline).
  2. Trains DA-ECG-Ret across a grid of (beta, lambda, exclude_self_from_redundancy)
     — everything in-memory (no checkpoint files, no subprocess overhead),
     reusing train.py's train_modules_in_memory().
  3. For each combo, computes the same Precision/Diversity Pareto frontier
     (MMR lambda in cfg.lambda_sweep) used in evaluate.py, and its AUC
     relative to the SimCLR reference (positive = DA-ECG-Ret's embedding
     space supports a better trade-off than SimCLR's).
  4. Saves a CSV of every combo's result + summary plots.

This is intentionally run at FEWER epochs than the full benchmark (default
40) and on a SMALL set of fast, representative datasets, since it's a
coarse search meant to find a promising region, not the final numbers for
the paper — re-run the full run_all.py at 100 epochs with the winning
(beta, lambda, exclude_self_from_redundancy) once you've picked one.

Usage (defaults are a reasonable ~20-40 minute sweep on an M-series Mac):
    python sweep_hparams.py
    python sweep_hparams.py --datasets ucr:ECG5000 ucr:TwoLeadECG --epochs 30
    python sweep_hparams.py --betas 0.5 1 2 5 10 20 40 --lambdas 0.7
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from device_utils import resolve_device
from data.augmentations import default_ecg_augmentation
from data.common import ContrastiveWrapper, get_dataset_splits, effective_batch_size
from train import set_seed, train_modules_in_memory
from retrieval.metrics import extract_embeddings
from retrieval.pareto import trade_off_frontier, frontier_auc
from sklearn.neighbors import NearestNeighbors


def parse_dataset_spec(spec: str):
    if ":" in spec:
        kind, name = spec.split(":", 1)
        return kind, name
    return spec, None


def dataset_tag(kind: str, name: str) -> str:
    return f"ucr_{name}" if kind == "ucr" else kind


def make_args_namespace(kind: str, name: str, data_root: str):
    class NS:
        pass
    ns = NS()
    ns.dataset = kind
    ns.ucr_name = name if name else "ECG200"
    ns.data_root = data_root
    ns.n_synthetic_train = 1200
    ns.n_synthetic_val = 400
    return ns


def compute_pareto_auc(embedding_model, test_base, n_classes, cfg, device):
    loader = DataLoader(test_base, batch_size=64, shuffle=False)
    data = extract_embeddings(embedding_model, loader, device)
    embs, labels = data["embeddings"], data["labels"]
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    n_neighbors = min(cfg.top_n_candidates, len(embs))
    nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(embs)

    def shortlist_fn(q_emb_raw):
        _, idx = nn_index.kneighbors(q_emb_raw.reshape(1, -1))
        idx = idx[0]
        return list(idx), embs_n[idx]

    k = cfg.cohort_sizes[len(cfg.cohort_sizes) // 2]
    frontier = trade_off_frontier(embs, labels, embs_n, labels, shortlist_fn,
                                   k, n_classes, list(cfg.lambda_sweep))
    return frontier_auc(frontier), frontier


def run_sweep(args):
    device = resolve_device(verbose=True)
    all_rows = []
    frontier_rows = []

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
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                   num_workers=cfg.num_workers, drop_last=True)

        # --- reference SimCLR encoder for this dataset ---
        set_seed(cfg.seed)
        simclr_modules, _ = train_modules_in_memory("simclr", cfg, train_loader, device,
                                                      verbose_prefix=f"{tag}/simclr(ref)")
        ref_auc, ref_frontier = compute_pareto_auc(simclr_modules["embedding_model"],
                                                    test_base, n_classes, cfg, device)
        print(f"  SimCLR reference Pareto AUC = {ref_auc:.5f}")
        for pt in ref_frontier:
            frontier_rows.append({"dataset": tag, "config": "simclr (reference)", **pt})

        # --- DA-ECG-Ret grid ---
        combos = list(itertools.product(args.betas, args.lambdas, args.redundancy_variants))
        for beta, lam, exclude_self in combos:
            cfg.beta, cfg.lam, cfg.alpha = beta, lam, args.alpha
            cfg.exclude_self_from_redundancy = bool(exclude_self)
            set_seed(cfg.seed + 1)  # distinct from simclr's seed, fixed across the grid
            tag_str = f"beta={beta}_lam={lam}_fix={int(exclude_self)}"
            modules, history = train_modules_in_memory("da_ecg_ret", cfg, train_loader, device,
                                                         verbose_prefix=f"{tag}/{tag_str}")
            auc, frontier = compute_pareto_auc(modules["embedding_model"], test_base,
                                                n_classes, cfg, device)
            final = history[-1]
            row = {
                "dataset": tag, "beta": beta, "lambda": lam,
                "exclude_self_from_redundancy": exclude_self,
                "auc": auc, "auc_gap_vs_simclr": auc - ref_auc,
                "final_infonce_loss": final.get("infonce"), "final_ret_loss": final.get("ret"),
            }
            all_rows.append(row)
            for pt in frontier:
                frontier_rows.append({"dataset": tag, "config": tag_str, **pt})
            print(f"  [{tag_str}] AUC={auc:.5f}  (gap vs SimCLR: {auc - ref_auc:+.5f})")

    return pd.DataFrame(all_rows), pd.DataFrame(frontier_rows)


def make_plots(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False)
    axes = axes[0]
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        for fix_val, label, style in [(True, "with fix", "-o"), (False, "without fix (original)", "--x")]:
            s2 = sub[sub["exclude_self_from_redundancy"] == fix_val]
            if s2.empty:
                continue
            for lam, s3 in s2.groupby("lambda"):
                s3 = s3.sort_values("beta")
                ax.plot(s3["beta"], s3["auc_gap_vs_simclr"], style,
                        label=f"{label}, lambda={lam}")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xscale("symlog")
        ax.set_xlabel("beta")
        ax.set_ylabel("Pareto AUC gap vs. SimCLR (higher = better)")
        ax.set_title(ds)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("DA-ECG-Ret vs. SimCLR: Pareto-frontier AUC gap", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "sweep_auc_gap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                         default=["ucr:ECG5000", "ucr:TwoLeadECG", "ucr:NonInvasiveFetalECGThorax1"])
    parser.add_argument("--data_root", type=str, default="./ptbxl_data")
    parser.add_argument("--encoder", choices=["transformer", "resnet18", "fcn"], default="transformer")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 1, 2, 5, 10, 20])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.7])
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--redundancy_variants", type=int, nargs="+", default=[1, 0],
                         help="1=with fix (default/recommended), 0=original buggy behaviour, "
                              "for a direct before/after comparison")
    parser.add_argument("--output_dir", type=str, default="./sweep_outputs")
    args = parser.parse_args()

    t0 = time.time()
    df, frontier_df = run_sweep(args)
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    frontier_csv = os.path.join(args.output_dir, "sweep_frontiers.csv")
    frontier_df.to_csv(frontier_csv, index=False)
    print(f"\nSaved {csv_path}\nSaved {frontier_csv}")

    make_plots(df, os.path.join(args.output_dir, "figures"))

    print(f"\nTotal sweep time: {(time.time() - t0)/60:.1f} min")
    print("\n=== Best config per dataset (by auc_gap_vs_simclr) ===")
    for ds, sub in df.groupby("dataset"):
        best = sub.loc[sub["auc_gap_vs_simclr"].idxmax()]
        print(f"{ds}: beta={best['beta']}, lambda={best['lambda']}, "
              f"fix={bool(best['exclude_self_from_redundancy'])}, "
              f"gap={best['auc_gap_vs_simclr']:+.5f}")

    print("\nAlso zipping sweep outputs...")
    import zipfile
    zip_path = os.path.join(os.path.dirname(os.path.normpath(args.output_dir)) or ".",
                             "sweep_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(args.output_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                zf.write(full_path, os.path.relpath(full_path, os.path.dirname(args.output_dir)))
    print(f"Packaged sweep results into: {zip_path}")


if __name__ == "__main__":
    main()
