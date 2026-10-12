"""
Shared helpers for multiseed_experiment.py and ablation_components.py —
both need "train a method, then compute Precision@K / NormalizedDiversity@K
/ AUROC / Pareto-frontier-AUC for it" many times over, so that logic lives
here once instead of being duplicated.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors

from retrieval.metrics import (
    extract_embeddings, linear_probe_auroc, precision_at_k, normalized_diversity_at_k,
)
from retrieval.pareto import trade_off_frontier, frontier_auc


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


def evaluate_embedding(embedding_model, train_base, test_base, n_classes, cfg, device,
                        k: int = 10, lambda_sweep=None):
    """
    Trains a linear probe (on train_base) and computes retrieval metrics on
    test_base (transductive: queries = corpus = test_base, consistent with
    evaluate.py). Returns a dict with:
        precision_at_k, normalized_diversity_at_k, auroc_macro
    and, if `lambda_sweep` is given (a list of MMR lambda values), also
    `pareto_auc` — the area under the Precision-vs-NormalizedDiversity
    curve as lambda is swept (see retrieval/pareto.py).
    """
    train_loader = DataLoader(train_base, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_base, batch_size=64, shuffle=False)
    train_data = extract_embeddings(embedding_model, train_loader, device)
    test_data = extract_embeddings(embedding_model, test_loader, device)

    auroc = linear_probe_auroc(
        train_data["embeddings"], train_data["labels"],
        test_data["embeddings"], test_data["labels"], device="cpu",
    )

    embs = test_data["embeddings"]
    labels = test_data["labels"]
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    n_neighbors = min(cfg.top_n_candidates, len(embs))
    nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(embs)

    def shortlist_fn(q_emb_raw):
        _, idx = nn_index.kneighbors(q_emb_raw.reshape(1, -1))
        idx = idx[0]
        return list(idx), embs_n[idx]

    precisions, diversities = [], []
    for qi in range(len(embs)):
        q_label = labels[qi]
        shortlist_ids, _ = shortlist_fn(embs[qi])
        selected = shortlist_ids[:k]
        sel_labels = [labels[i] for i in selected]
        precisions.append(precision_at_k(q_label, sel_labels))
        diversities.append(normalized_diversity_at_k(sel_labels, n_classes, k=k))

    result = {
        "precision_at_k": float(np.mean(precisions)),
        "normalized_diversity_at_k": float(np.mean(diversities)),
        "auroc_macro": auroc.get("macro", float("nan")),
    }

    if lambda_sweep is not None:
        frontier = trade_off_frontier(embs, labels, embs_n, labels, shortlist_fn,
                                       k, n_classes, list(lambda_sweep))
        result["pareto_auc"] = frontier_auc(frontier)

    return result
