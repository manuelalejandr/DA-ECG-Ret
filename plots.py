"""
Generates all per-dataset figures for the manuscript at 300 dpi, English
labels. Run once per dataset (see run_all.py, which calls this
automatically for every dataset in --datasets). For the cross-dataset
summary figures (average rank, Wilcoxon test), see aggregate.py.

Figures produced (into <output_dir>/<dataset_tag>/figures/):
  1. training_loss_curves.png       - loss curves for every trained method
  2. precision_at_k.png             - Precision@K vs K, all methods x reranks
  3. diversity_at_k.png             - NormalizedDiversity@K vs K
  4. precision_diversity_tradeoff.png - single-point scatter, all methods
  5. pareto_frontier.png            - full lambda-sweep frontier (SimCLR vs DA-ECG-Ret)
  6. embedding_tsne.png             - 2D t-SNE, SimCLR vs DA-ECG-Ret, colored by class
  7. auroc_bar.png                  - downstream linear-probe AUROC, all methods
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from config import Config, SSL_METHODS
from device_utils import resolve_device
from data.common import get_dataset_splits
from ssl_methods.registry import get_method
from retrieval.metrics import extract_embeddings

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
})

METHOD_LABELS = {
    "simclr": "SimCLR",
    "clocs_lite": "CLOCS-lite",
    "ts2vec": "TS2Vec",
    "tfc": "TF-C",
    "softclt": "SoftCLT",
    "da_ecg_ret": "DA-ECG-Ret (proposed)",
}
RERANK_LABELS = {"nn": "NN", "mmr": "MMR", "dpp": "DPP"}
COLORS = {
    "simclr": "#4C72B0", "clocs_lite": "#DD8452", "ts2vec": "#8172B2",
    "tfc": "#937860", "softclt": "#CCB974", "da_ecg_ret": "#55A868",
}
MARKERS = {"simclr": "o", "clocs_lite": "s", "ts2vec": "^",
           "tfc": "D", "softclt": "v", "da_ecg_ret": "P"}


def dataset_tag_from_args(args, cfg):
    return args.dataset if args.dataset != "ucr" else f"ucr_{cfg.ucr_name}"


def plot_training_curves(base_dir):
    hist_dir = os.path.join(base_dir, "history")
    fig, ax = plt.subplots(figsize=(6, 4))
    found = False
    for method in SSL_METHODS:
        path = os.path.join(hist_dir, f"history_{method}.json")
        if not os.path.exists(path):
            continue
        found = True
        with open(path) as f:
            hist = json.load(f)
        epochs = [h["epoch"] + 1 for h in hist]
        total = [h["total"] for h in hist]
        ax.plot(epochs, total, label=METHOD_LABELS[method], color=COLORS.get(method))
    if not found:
        plt.close(fig)
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss curves")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(base_dir, "figures", "training_loss_curves.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def _combo_key(row):
    return f"{METHOD_LABELS.get(row['method'], row['method'])} + {RERANK_LABELS.get(row['rerank'], row['rerank'])}"


def plot_precision_and_diversity(base_dir, results_csv):
    df = pd.read_csv(results_csv)
    df["combo"] = df.apply(_combo_key, axis=1)

    for metric, fname, ylabel, title in [
        ("Precision@K", "precision_at_k.png", "Precision@K", "Retrieval precision vs. cohort size"),
        ("NormalizedDiversity@K", "diversity_at_k.png", "Normalized Diversity@K (fraction of classes covered)",
         "Retrieved diagnostic diversity vs. cohort size"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for combo, sub in df.groupby("combo"):
            sub = sub.sort_values("K")
            ax.plot(sub["K"], sub[metric], marker="o", label=combo)
        ax.set_xlabel("Cohort size K")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        out_path = os.path.join(base_dir, "figures", fname)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_precision_diversity_tradeoff(base_dir, results_csv):
    df = pd.read_csv(results_csv)
    df["combo"] = df.apply(_combo_key, axis=1)
    agg = df.groupby("combo").agg({"Precision@K": "mean", "NormalizedDiversity@K": "mean"}).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]
    for i, (_, row) in enumerate(agg.iterrows()):
        ax.scatter(row["NormalizedDiversity@K"], row["Precision@K"], s=90,
                   marker=markers[i % len(markers)], label=row["combo"],
                   edgecolors="black", linewidths=0.5)
    ax.set_xlabel("Mean Normalized Diversity@K")
    ax.set_ylabel("Mean Precision@K")
    ax.set_title("Precision-diversity trade-off across methods")
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    out_path = os.path.join(base_dir, "figures", "precision_diversity_tradeoff.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_pareto_frontier(base_dir, pareto_csv):
    if not os.path.exists(pareto_csv):
        return
    df = pd.read_csv(pareto_csv)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("NormalizedDiversity@K")
        auc = sub["auc"].iloc[0]
        ax.plot(sub["NormalizedDiversity@K"], sub["Precision@K"], marker="o",
                color=COLORS.get(method), label=f"{METHOD_LABELS.get(method, method)} (AUC={auc:.3f})")
    ax.set_xlabel("Normalized Diversity@K (MMR lambda sweep)")
    ax.set_ylabel("Precision@K")
    ax.set_title("Precision-diversity Pareto frontier (MMR $\\lambda \\in [0,1]$)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(base_dir, "figures", "pareto_frontier.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_auroc_bar(base_dir, results_csv):
    df = pd.read_csv(results_csv)
    present = [m for m in SSL_METHODS if m in df["method"].unique()]
    agg = df.groupby("method")["AUROC_macro"].mean().reindex(present)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar([METHOD_LABELS[m] for m in agg.index], agg.values,
                  color=[COLORS[m] for m in agg.index])
    ax.set_ylabel("Macro AUROC (linear probe)")
    ax.set_title("Downstream classification quality of frozen embeddings")
    ax.set_ylim(max(0.0, agg.values.min() - 0.1), 1.0)
    for b, v in zip(bars, agg.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    out_path = os.path.join(base_dir, "figures", "auroc_bar.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_embedding_geometry(base_dir, cfg, args, device, n_points=600):
    """t-SNE contrasting SimCLR vs. DA-ECG-Ret latent geometry, colored by
    class label -- illustrates the Discussion claim that the retrieval-aware
    loss reshapes the latent manifold."""
    train_base, test_base, n_classes, _ = get_dataset_splits(args, cfg)
    n_points = min(n_points, len(test_base))
    subset_idx = np.random.RandomState(99).choice(len(test_base), n_points, replace=False)
    subset = torch.utils.data.Subset(test_base, subset_idx.tolist())
    loader = DataLoader(subset, batch_size=64, shuffle=False)

    dataset_tag = dataset_tag_from_args(args, cfg)
    ckpt_dir = os.path.join(cfg.output_dir, dataset_tag, "checkpoints")

    class_names = getattr(test_base, "classes", None) or [f"class_{i}" for i in range(n_classes)]

    methods_present = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, method_name in zip(axes, ["simclr", "da_ecg_ret"]):
        ckpt = os.path.join(ckpt_dir, f"encoder_{method_name}.pt")
        if not os.path.exists(ckpt):
            ax.set_visible(False)
            continue
        methods_present.append(method_name)
        method = get_method(method_name)
        modules = method.build_modules(cfg, device)
        state = torch.load(ckpt, map_location=device)
        modules["embedding_model"].load_state_dict(state["embedding_model"])
        modules["embedding_model"].eval()

        data = extract_embeddings(modules["embedding_model"], loader, device)
        embs, labels = data["embeddings"], data["labels"]
        perplexity = min(30, max(5, embs.shape[0] // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0, init="pca")
        coords = tsne.fit_transform(embs)

        primary = labels.argmax(axis=1)
        for c in range(n_classes):
            mask = primary == c
            if mask.sum() == 0:
                continue
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            ax.scatter(coords[mask, 0], coords[mask, 1], s=10, label=str(name), alpha=0.7)
        ax.set_title(METHOD_LABELS[method_name])
        ax.set_xticks([]); ax.set_yticks([])

    if methods_present:
        vis_ax = axes[0] if axes[0].get_visible() else axes[1]
        handles, labs = vis_ax.get_legend_handles_labels()
        fig.legend(handles, labs, loc="lower center", ncol=min(len(labs), 6),
                   frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=8)
        fig.suptitle("Latent geometry: SimCLR vs. DA-ECG-Ret (t-SNE, colored by class)")
        fig.tight_layout()
        out_path = os.path.join(base_dir, "figures", "embedding_tsne.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "ptbxl", "ucr", "mitbih"], default="synthetic")
    parser.add_argument("--ucr_name", type=str, default="ECG200")
    parser.add_argument("--data_root", type=str, default="./ptbxl_data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--encoder", choices=["transformer", "resnet18", "fcn"], default="transformer")
    args = parser.parse_args()

    cfg = Config()
    cfg.dataset = args.dataset
    cfg.ucr_name = args.ucr_name
    cfg.data_root = args.data_root
    cfg.encoder_name = args.encoder
    cfg.output_dir = args.output_dir
    device = resolve_device(verbose=True)

    dataset_tag = dataset_tag_from_args(args, cfg)
    base_dir = os.path.join(cfg.output_dir, dataset_tag)
    os.makedirs(os.path.join(base_dir, "figures"), exist_ok=True)
    results_csv = os.path.join(base_dir, "results", "retrieval_results.csv")
    pareto_csv = os.path.join(base_dir, "results", "pareto_frontier.csv")

    plot_training_curves(base_dir)
    if os.path.exists(results_csv):
        plot_precision_and_diversity(base_dir, results_csv)
        plot_precision_diversity_tradeoff(base_dir, results_csv)
        plot_auroc_bar(base_dir, results_csv)
    plot_pareto_frontier(base_dir, pareto_csv)
    plot_embedding_geometry(base_dir, cfg, args, device)


if __name__ == "__main__":
    main()
