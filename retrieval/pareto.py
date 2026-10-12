"""
Pareto-frontier analysis for the precision/diversity trade-off.

Motivation (see README, "Is Diversity@K actually measuring what we want?"):
a single (Precision@K, Diversity@K) point per method conflates two
questions: (1) does the method's *embedding space* support a better
trade-off at all, and (2) did we happen to pick a re-ranking operating
point (lambda) that flatters one method over another? Reporting the FULL
trade-off curve as lambda sweeps from 0 (pure diversity) to 1 (pure
relevance), and the area under that curve, answers (1) directly and makes
(2) impossible to game.
"""
from typing import List, Dict
import numpy as np

from retrieval.mmr_rerank import mmr_rerank
from retrieval.metrics import precision_at_k, normalized_diversity_at_k


def _trapz(ys, xs):
    """Trapezoidal integration that works across numpy versions: numpy>=2.0
    renamed/removed np.trapz (-> np.trapezoid) on some builds, so we fall
    back to a manual implementation rather than pin a numpy version."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(ys, xs))
    if hasattr(np, "trapz"):
        return float(np.trapz(ys, xs))
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    return float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2.0))


def trade_off_frontier(
    query_embs: np.ndarray, query_labels: np.ndarray,
    corpus_embs_n: np.ndarray, corpus_labels: np.ndarray,
    shortlist_fn, k: int, n_classes: int, lambda_values: List[float],
) -> List[Dict]:
    """
    shortlist_fn(query_emb_raw) -> (shortlist_ids, shortlist_embs_n) giving
    the stage-1 cosine shortlist for one query (see evaluate.py).

    Returns a list of dicts, one per lambda value, each with the mean
    Precision@K and mean NormalizedDiversity@K over all queries at that
    operating point.
    """
    rows = []
    for lam in lambda_values:
        precisions, diversities = [], []
        for qi in range(len(query_embs)):
            q_emb_raw = query_embs[qi]
            q_label = query_labels[qi]
            q_emb_n = q_emb_raw / (np.linalg.norm(q_emb_raw) + 1e-8)

            shortlist_ids, shortlist_embs_n = shortlist_fn(q_emb_raw)
            selected = mmr_rerank(q_emb_n, shortlist_embs_n, shortlist_ids, k, lam=lam)
            sel_labels = [corpus_labels[i] for i in selected]

            precisions.append(precision_at_k(q_label, sel_labels))
            diversities.append(normalized_diversity_at_k(sel_labels, n_classes, k=k))
        rows.append({
            "lambda": lam,
            "Precision@K": float(np.mean(precisions)),
            "NormalizedDiversity@K": float(np.mean(diversities)),
        })
    return rows


def frontier_auc(frontier: List[Dict]) -> float:
    """Area under the (NormalizedDiversity@K, Precision@K) curve, over the
    FIXED domain x in [0, 1], via the trapezoidal rule. Higher = better
    overall trade-off.

    FIX (post co-author review, Sept 2026): the original version integrated
    only over the x-range the swept lambda values happened to VISIT for
    that particular method, i.e. [min(NormalizedDiversity@K),
    max(NormalizedDiversity@K)]. That is not a fair basis for comparing
    two methods' areas: a method whose frontier happens to span a wider
    diversity range gets more area "for free," independent of whether it
    trades precision for diversity any better at a MATCHED diversity
    level, which was exactly the complaint that the reported ~0.14 area
    over a ~0.2-wide range (TwoLeadECG, Fig. 3) was not comparable to
    another method's area computed over a different-width range.

    This is a hypervolume-style fix with a fixed reference point: we
    extend each method's own frontier flat to the full [0, 1] domain
    before integrating -- holding Precision@K constant at its value at
    the lowest observed diversity for any x below that point, and at its
    value at the highest observed diversity for any x above it (the
    standard "achieved-so-far" extension used for interpolated
    precision-recall curves in IR). Every method's AUC is now the area
    under a curve defined over the SAME domain, so two methods with
    identical precision-at-matched-diversity behaviour get identical AUC
    regardless of which lambda values happened to be swept or what range
    of diversity values they produced."""
    pts = sorted(frontier, key=lambda r: r["NormalizedDiversity@K"])
    xs = [p["NormalizedDiversity@K"] for p in pts]
    ys = [p["Precision@K"] for p in pts]
    if len(xs) < 2:
        return 0.0
    # Extend to the fixed [0, 1] domain by holding the boundary values flat.
    xs_ext = [0.0] + xs + [1.0]
    ys_ext = [ys[0]] + ys + [ys[-1]]
    return _trapz(ys_ext, xs_ext)
