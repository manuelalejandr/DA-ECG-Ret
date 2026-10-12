"""
Evaluation metrics from Section 4.2:
  1. Precision@K
  2. Diagnostic Diversity@K
  3. MMR-Score (see retrieval/mmr_rerank.py::mmr_score)
  4. AUROC on downstream linear classification (frozen encoder)
  5. Intra-list Distance (ILD)
"""
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def precision_at_k(query_labels: np.ndarray, retrieved_labels: List[np.ndarray]) -> float:
    """Fraction of retrieved items sharing >=1 diagnostic superclass with
    the query (multi-hot label vectors)."""
    if len(retrieved_labels) == 0:
        return 0.0
    hits = [int(np.any(np.logical_and(query_labels, lab))) for lab in retrieved_labels]
    return float(np.mean(hits))


def diagnostic_diversity_at_k(retrieved_labels: List[np.ndarray]) -> int:
    """Number of distinct diagnostic superclasses present in the retrieved
    cohort (union over the multi-hot vectors)."""
    if len(retrieved_labels) == 0:
        return 0
    union = np.any(np.stack(retrieved_labels, axis=0), axis=0)
    return int(union.sum())


def normalized_diversity_at_k(retrieved_labels: List[np.ndarray], n_classes: int,
                               k: int = None) -> float:
    """Diversity@K divided by the ACHIEVABLE ceiling min(K, n_classes),
    i.e. the fraction of the label space that a cohort of size K could
    possibly cover.

    Rationale: raw Diversity@K is not comparable across datasets with very
    different class cardinalities (e.g. a 2-class UCR problem saturates at
    2 regardless of method, while a 42-class problem has much more room to
    show a difference). Normalizing removes this ceiling-effect confound.

    FIX (post co-author review, Sept 2026): the original version divided
    by n_classes alone. That overcorrects whenever K < n_classes: with
    K=5 and n_classes=42 (e.g. NonInvasiveFetalECGThorax1/2), a cohort of
    5 items can cover AT MOST 5 distinct classes, so the theoretical
    ceiling of the raw metric is 5/42 = 0.119, never 1.0 -- every method
    looks artificially "low-diversity" on high-cardinality datasets purely
    from this ceiling, not from any real difference in behaviour.
    Dividing by min(K, n_classes) instead makes 1.0 the true achievable
    ceiling regardless of K vs. n_classes, restoring comparability. `k` is
    optional and defaults to n_classes-only normalisation (the old
    behaviour) for backward compatibility with callers that don't have K
    on hand; all call sites in this codebase now pass it explicitly.
    """
    if n_classes == 0:
        return 0.0
    ceiling = min(k, n_classes) if k is not None else n_classes
    if ceiling == 0:
        return 0.0
    return diagnostic_diversity_at_k(retrieved_labels) / ceiling


def intra_list_distance(embs: np.ndarray) -> float:
    """Mean pairwise Euclidean distance within the retrieved cohort
    (geometric diversity proxy)."""
    k = embs.shape[0]
    if k < 2:
        return 0.0
    dists = []
    for i in range(k):
        for j in range(i + 1, k):
            dists.append(np.linalg.norm(embs[i] - embs[j]))
    return float(np.mean(dists))


@torch.no_grad()
def extract_embeddings(encoder: nn.Module, loader, device) -> Dict[str, np.ndarray]:
    encoder.eval()
    all_embs, all_labels, all_ids, all_patients = [], [], [], []
    extra = {"label_sub": [], "label_code": []}
    for batch in loader:
        x = batch["signal"].to(device)
        z = encoder(x).cpu().numpy()
        all_embs.append(z)
        all_labels.append(batch["label"].numpy())
        for k_ in extra:
            if k_ in batch:
                extra[k_].append(batch[k_].numpy())
        all_ids.extend(batch["ecg_id"].tolist() if torch.is_tensor(batch["ecg_id"]) else batch["ecg_id"])
        all_patients.extend(batch["patient_id"].tolist() if torch.is_tensor(batch["patient_id"]) else batch["patient_id"])
    out = {
        "embeddings": np.concatenate(all_embs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "ids": np.array(all_ids),
        "patients": np.array(all_patients),
    }
    if extra["label_sub"]:
        out["labels_sub"] = np.concatenate(extra["label_sub"], axis=0)
    if extra["label_code"]:
        out["labels_code"] = np.concatenate(extra["label_code"], axis=0)
    return out


def linear_probe_auroc(
    train_embs: np.ndarray, train_labels: np.ndarray,
    test_embs: np.ndarray, test_labels: np.ndarray,
    device="cpu", epochs=100, lr=1e-2,
) -> Dict[str, float]:
    """Trains a linear classifier on frozen embeddings (Section 4.2, item 4)
    and reports per-class + macro AUROC."""
    import torch.nn.functional as F

    n_classes = train_labels.shape[1]
    Xtr = torch.tensor(train_embs, dtype=torch.float32, device=device)
    Ytr = torch.tensor(train_labels, dtype=torch.float32, device=device)
    Xte = torch.tensor(test_embs, dtype=torch.float32, device=device)

    clf = nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        opt.zero_grad()
        logits = clf(Xtr)
        loss = F.binary_cross_entropy_with_logits(logits, Ytr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs = torch.sigmoid(clf(Xte)).cpu().numpy()

    per_class = {}
    valid_aucs = []
    for c in range(n_classes):
        y_true = test_labels[:, c]
        if len(np.unique(y_true)) < 2:
            continue  # AUROC undefined for a single-class column
        auc = roc_auc_score(y_true, probs[:, c])
        per_class[f"class_{c}"] = auc
        valid_aucs.append(auc)

    per_class["macro"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return per_class
