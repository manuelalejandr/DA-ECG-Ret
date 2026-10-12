"""
Retrieval evaluation — generic over ssl_methods/registry.py and
data/common.py::get_dataset_splits.

For every trained encoder we evaluate the re-ranking strategies that are
meaningful for it:

    <method> + NN                   (pure similarity)
    <method> + MMR                  (classical MMR re-ranking, Eq. 6)
    simclr   + DPP                  (DPP re-ranking; reported once, on the
                                      strongest single-objective baseline,
                                      as the diversity-baseline reference)

"simclr + MMR" is the critical inference-only ablation that isolates the
contribution of the retrieval-aware TRAINING loss (L_ret): it uses the same
re-ranking step as "da_ecg_ret + MMR" but on embeddings trained without
L_ret.

Metrics per query/K: Precision@K, Diversity@K, NormalizedDiversity@K,
MMR-Score, Intra-list Distance (ILD). Plus, once per method: downstream
linear-probe AUROC, and (for methods re-ranked with MMR) a full
precision/diversity Pareto frontier across a lambda sweep with its AUC
(see retrieval/pareto.py and the README section on metric validity).

NOTE on train/test split re-use: SSL pre-training is unsupervised (uses no
labels), so evaluating retrieval and the linear probe on the SAME held-out
test split that also serves as the retrieval corpus does not leak label
information into representation learning; the linear probe itself is
fitted on the disjoint TRAIN split and only evaluated on test queries.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors

from config import Config, SSL_METHODS
from device_utils import resolve_device
from data.common import get_dataset_splits
from ssl_methods.registry import get_method
from retrieval.mmr_rerank import mmr_rerank, mmr_score
from retrieval.dpp_rerank import dpp_greedy_map
from retrieval.pareto import trade_off_frontier, frontier_auc
from retrieval.metrics import (
    precision_at_k, diagnostic_diversity_at_k, normalized_diversity_at_k,
    intra_list_distance, extract_embeddings, linear_probe_auroc,
)


def dataset_tag_from_args(args, cfg):
    return args.dataset if args.dataset != "ucr" else f"ucr_{cfg.ucr_name}"


def load_embedding_model(ckpt_path, method_name, cfg, device):
    method = get_method(method_name)
    modules = method.build_modules(cfg, device)
    state = torch.load(ckpt_path, map_location=device)
    modules["embedding_model"].load_state_dict(state["embedding_model"])
    modules["embedding_model"].eval()
    return modules["embedding_model"]


def evaluate_method(embedding_model, method_name, rerank_strategies, train_base,
                     test_base, n_classes, cfg, device, run_pareto: bool):
    loader_kwargs = dict(batch_size=64, shuffle=False)
    train_loader = DataLoader(train_base, **loader_kwargs)
    test_loader = DataLoader(test_base, **loader_kwargs)

    train_data = extract_embeddings(embedding_model, train_loader, device)
    test_data = extract_embeddings(embedding_model, test_loader, device)

    auroc = linear_probe_auroc(
        train_data["embeddings"], train_data["labels"],
        test_data["embeddings"], test_data["labels"], device="cpu",
    )

    # queries and corpus are both the test split (standard transductive
    # retrieval evaluation protocol; SSL pre-training used no labels)
    corpus_embs = test_data["embeddings"]
    corpus_embs_n = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-8)
    corpus_labels = test_data["labels"]
    query_embs = test_data["embeddings"]
    query_labels = test_data["labels"]

    n_neighbors = min(cfg.top_n_candidates, len(corpus_embs))
    nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(corpus_embs)

    def shortlist_fn(q_emb_raw):
        _, idx = nn_index.kneighbors(q_emb_raw.reshape(1, -1))
        idx = idx[0]
        return list(idx), corpus_embs_n[idx]

    rows = []
    for rerank in rerank_strategies:
        for k in cfg.cohort_sizes:
            precisions, diversities, norm_diversities, mmrs, ilds = [], [], [], [], []
            for qi in range(len(query_embs)):
                q_emb_raw = query_embs[qi]
                q_label = query_labels[qi]
                q_emb_n = q_emb_raw / (np.linalg.norm(q_emb_raw) + 1e-8)

                shortlist_ids, shortlist_embs_n = shortlist_fn(q_emb_raw)

                if rerank == "nn":
                    selected = shortlist_ids[:k]
                elif rerank == "mmr":
                    selected = mmr_rerank(q_emb_n, shortlist_embs_n, shortlist_ids, k,
                                           lam=cfg.mmr_lambda_inference)
                elif rerank == "dpp":
                    selected = dpp_greedy_map(q_emb_n, shortlist_embs_n, shortlist_ids, k)
                else:
                    raise ValueError(rerank)

                sel_embs_n = corpus_embs_n[selected]
                sel_labels = [corpus_labels[i] for i in selected]

                precisions.append(precision_at_k(q_label, sel_labels))
                diversities.append(diagnostic_diversity_at_k(sel_labels))
                norm_diversities.append(normalized_diversity_at_k(sel_labels, n_classes, k=k))
                mmrs.append(mmr_score(q_emb_n, sel_embs_n, lam=cfg.mmr_lambda_inference))
                ilds.append(intra_list_distance(corpus_embs[selected]))

            rows.append({
                "method": method_name,
                "rerank": rerank,
                "K": k,
                "Precision@K": float(np.mean(precisions)),
                "Diversity@K": float(np.mean(diversities)),
                "NormalizedDiversity@K": float(np.mean(norm_diversities)),
                "MMR-Score": float(np.mean(mmrs)),
                "ILD": float(np.mean(ilds)),
                "AUROC_macro": auroc.get("macro", float("nan")),
            })

    pareto_rows = []
    if run_pareto:
        k_for_pareto = cfg.cohort_sizes[len(cfg.cohort_sizes) // 2]  # a representative K
        frontier = trade_off_frontier(
            query_embs, query_labels, corpus_embs_n, corpus_labels,
            shortlist_fn, k_for_pareto, n_classes, list(cfg.lambda_sweep),
        )
        auc = frontier_auc(frontier)
        for pt in frontier:
            pareto_rows.append({"method": method_name, "K": k_for_pareto,
                                 "auc": auc, **pt})

    return rows, pareto_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "ptbxl", "ucr", "mitbih"], default="synthetic")
    parser.add_argument("--ucr_name", type=str, default="ECG200")
    parser.add_argument("--data_root", type=str, default="./ptbxl_data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--encoder", choices=["transformer", "resnet18", "fcn"], default="transformer")
    parser.add_argument("--methods", nargs="+", default=SSL_METHODS)
    args = parser.parse_args()

    cfg = Config()
    cfg.dataset = args.dataset
    cfg.ucr_name = args.ucr_name
    cfg.data_root = args.data_root
    cfg.encoder_name = args.encoder
    cfg.output_dir = args.output_dir
    device = resolve_device(verbose=True)

    train_base, test_base, n_classes, _ = get_dataset_splits(args, cfg)
    dataset_tag = dataset_tag_from_args(args, cfg)
    ckpt_dir = os.path.join(cfg.output_dir, dataset_tag, "checkpoints")

    method_rerank_map = {
        "simclr": ["nn", "mmr", "dpp"],   # "simclr+mmr" = critical inference-only ablation
        "clocs_lite": ["nn"],
        "ts2vec": ["nn"],
        "tfc": ["nn"],
        "softclt": ["nn"],
        "da_ecg_ret": ["nn", "mmr"],       # "da_ecg_ret+mmr" = full proposed method
    }
    pareto_methods = {"simclr", "da_ecg_ret"}  # methods for which we sweep MMR's lambda

    all_rows, all_pareto = [], []
    for method_name in args.methods:
        ckpt = os.path.join(ckpt_dir, f"encoder_{method_name}.pt")
        if not os.path.exists(ckpt):
            print(f"[skip] no checkpoint for {method_name} at {ckpt}")
            continue
        embedding_model = load_embedding_model(ckpt, method_name, cfg, device)
        reranks = method_rerank_map.get(method_name, ["nn"])
        rows, pareto_rows = evaluate_method(
            embedding_model, method_name, reranks, train_base, test_base,
            n_classes, cfg, device, run_pareto=method_name in pareto_methods,
        )
        all_rows.extend(rows)
        all_pareto.extend(pareto_rows)

    for row in all_rows:
        row["dataset"] = dataset_tag
    for row in all_pareto:
        row["dataset"] = dataset_tag

    out_dir = os.path.join(cfg.output_dir, dataset_tag, "results")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, "retrieval_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")
    print(df.to_string(index=False))

    if all_pareto:
        pareto_df = pd.DataFrame(all_pareto)
        pareto_csv = os.path.join(out_dir, "pareto_frontier.csv")
        pareto_df.to_csv(pareto_csv, index=False)
        print(f"Saved Pareto-frontier data to {pareto_csv}")


if __name__ == "__main__":
    main()
