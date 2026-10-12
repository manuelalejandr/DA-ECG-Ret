"""
Diversity-aware retrieval evaluation (numpy only; no torch).

WHY THIS EXISTS (construct validity)
------------------------------------
The original "Diversity@K" counted how many distinct diagnostic classes a
retrieved cohort contains. Across methods it was almost perfectly
ANTI-correlated with Precision@K (mean Spearman -0.90 over 9 datasets): a
worse retriever returns more wrong-class neighbours and therefore looks
"more diverse". That metric cannot support a claim about diversity-aware
retrieval.

The task is therefore re-defined with a LABEL HIERARCHY (PTB-XL provides
superclass > subclass > SCP statement):

    relevant     : a retrieved ECG is relevant to the query iff it shares a
                   COARSE diagnosis with it (e.g. same superclass).
    diversity    : among RELEVANT results, cover as many different FINE
                   diagnoses of the query's diagnostic family as possible
                   (e.g. different subclasses / SCP statements inside the
                   same superclass). Irrelevant results never earn
                   diversity credit, so diversity can no longer be bought
                   by being wrong.

Metrics (per query, then averaged), for each (coarse -> fine) pair:
    P@K        fraction of the cohort that is relevant (shares a coarse label)
    P_fine@K   fraction sharing >=1 FINE label with the query (strict)
    RelDiv@K   #distinct fine labels (inside the query's family) covered by
               the cohort / min(K, #such fine labels available in the corpus)
    aNDCG@K    alpha-nDCG (Clarke et al., SIGIR 2008), alpha=0.5: each fine
               label of the query's family is a "subtopic"; redundancy of a
               subtopic is discounted by (1-alpha)^count. Normalised by a
               greedy ideal ranking over the whole corpus (the standard
               approximation; exact ideal is NP-hard).
    LegacyDiv@K  the old distinct-coarse-classes / min(K, C) metric, kept
               ONLY to show its coupling with P@K.

Protocol hygiene (new): the query itself AND every other recording of the
same patient are removed from the candidate corpus. (The earlier
evaluation let a query retrieve itself as its own top-1 neighbour and, on
PTB-XL, other ECGs of the same patient.)

Candidate generation = top-N cosine shortlist; re-rankers = NN, MMR
(lambda sweep) and DPP, all applied to the SAME shortlist for every
encoder, so encoder effects and re-ranker effects can be separated.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

from retrieval.dpp_rerank import dpp_greedy_map

ALPHA = 0.5
MMR_LAMBDAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)


# --------------------------------------------------------------------------
# candidate generation
# --------------------------------------------------------------------------
def _normalize(e):
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)


def shortlists(embs: np.ndarray, patients: np.ndarray, n_shortlist: int,
               query_idx: np.ndarray, block: int = 512) -> np.ndarray:
    """For each query index, the top-`n_shortlist` corpus indices by cosine
    similarity, EXCLUDING the query itself and all same-patient items.
    Returns (n_queries, n_shortlist) int array sorted by similarity (desc).
    Rows may be padded with -1 if the corpus is too small."""
    En = _normalize(embs.astype(np.float32))
    n = len(En)
    N = min(n_shortlist, n - 1)
    out = np.full((len(query_idx), n_shortlist), -1, dtype=np.int64)
    for b in range(0, len(query_idx), block):
        qi = query_idx[b:b + block]
        S = En[qi] @ En.T                                       # (b, n)
        same = patients[qi][:, None] == patients[None, :]       # includes self
        S[same] = -np.inf
        idx = np.argpartition(-S, N - 1, axis=1)[:, :N]
        sims = np.take_along_axis(S, idx, axis=1)
        order = np.argsort(-sims, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        sims = np.take_along_axis(sims, order, axis=1)
        idx = np.where(np.isfinite(sims), idx, -1)
        out[b:b + len(qi), :N] = idx
    return out


# --------------------------------------------------------------------------
# re-rankers (prefix property: cohort of size K = first K of the K_max list)
# --------------------------------------------------------------------------
def mmr_select(sim_q: np.ndarray, sim_mat: np.ndarray, k: int, lam: float) -> List[int]:
    """Vectorised, behaviour-identical version of retrieval.mmr_rerank."""
    n = len(sim_q)
    k = min(k, n)
    first = int(np.argmax(sim_q))
    sel = [first]
    max_sim = sim_mat[first].copy()
    avail = np.ones(n, dtype=bool)
    avail[first] = False
    while len(sel) < k:
        score = lam * sim_q - (1 - lam) * max_sim
        score[~avail] = -np.inf
        j = int(np.argmax(score))
        sel.append(j)
        avail[j] = False
        max_sim = np.maximum(max_sim, sim_mat[j])
    return sel


def rank_lists(En: np.ndarray, qi: int, cand: np.ndarray, k_max: int,
               rerankers: List[str], rand_pool: Optional[np.ndarray] = None,
               rng: Optional[np.random.RandomState] = None) -> Dict[str, np.ndarray]:
    """cand: shortlist corpus indices (no -1). Returns {ranker: corpus idx array (<=k_max)}."""
    cE = En[cand]
    q = En[qi]
    sim_q = cE @ q
    sim_mat = cE @ cE.T
    res = {}
    for r in rerankers:
        if r == "nn":
            res[r] = cand[:k_max]
        elif r.startswith("mmr@"):
            lam = float(r.split("@")[1])
            res[r] = cand[mmr_select(sim_q, sim_mat, k_max, lam)]
        elif r == "dpp":
            ids = dpp_greedy_map(q, cE, list(range(len(cand))), k_max)
            res[r] = cand[ids]
        elif r == "rand":
            # lower-bound reference: k_max random corpus items (no similarity at all)
            res[r] = rng.choice(rand_pool, size=min(k_max, len(rand_pool)), replace=False)
        else:
            raise ValueError(r)
    return res


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _ideal_dcg(M: np.ndarray, k_max: int, alpha: float) -> np.ndarray:
    """Greedy ideal alpha-DCG@r for r=1..k_max. M: (n_cand, n_subtopics) bool."""
    out = np.zeros(k_max)
    if M.size == 0 or M.shape[1] == 0:
        return out
    Mf = M.astype(np.float32)
    counts = np.zeros(M.shape[1], dtype=np.float32)
    used = np.zeros(M.shape[0], dtype=bool)
    acc = 0.0
    for r in range(k_max):
        w = (1.0 - alpha) ** counts
        gain = Mf @ w
        gain[used] = -1.0
        j = int(np.argmax(gain))
        g = gain[j]
        if g <= 0:
            out[r:] = acc
            return out
        used[j] = True
        counts += Mf[j]
        acc += g / np.log2(r + 2)
        out[r] = acc
    return out


def evaluate_queries(
    embs: np.ndarray,
    patients: np.ndarray,
    coarse: np.ndarray,                       # (n, C) multi-hot, relevance level
    fine_levels: Dict[str, Tuple[np.ndarray, np.ndarray]],
    # {"sub": (fine multi-hot (n,F), parent (F,C)), "code": (...)}
    rerankers: Optional[List[str]] = None,
    ks: Tuple[int, ...] = (5, 10, 20),
    n_shortlist: int = 200,
    alpha: float = ALPHA,
    max_queries: Optional[int] = None,
    seed: int = 0,
    fine_pairs: Optional[Dict[str, np.ndarray]] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Returns {"<ranker>|<K>|<metric>": (n_queries_used,) array, ..., "query_idx": ..., "patient": ...}.
    Metrics not defined for a query (no fine subtopics available) are NaN.
    `fine_levels` may be empty (single-level datasets): then only P@K and
    LegacyDiv@K are produced."""
    rerankers = rerankers or (["nn"] + [f"mmr@{l}" for l in MMR_LAMBDAS] + ["dpp", "rand"])
    n = len(embs)
    rng = np.random.RandomState(seed)
    qidx = np.arange(n)
    if max_queries is not None and max_queries < n:
        qidx = np.sort(rng.choice(n, max_queries, replace=False))
    k_max = max(ks)
    En = _normalize(embs.astype(np.float32))
    SL = shortlists(embs, patients, n_shortlist, qidx)

    coarse_b = coarse > 0
    C = coarse.shape[1]
    fl = {name: (f > 0, par > 0) for name, (f, par) in fine_levels.items()}

    out: Dict[str, List[float]] = {}

    def put(r, k, m, v):
        out.setdefault(f"{r}|{k}|{m}", []).append(v)

    # per-query ideal alpha-DCG, cached by fine level
    for qn, qi in enumerate(qidx):
        cand = SL[qn][SL[qn] >= 0]
        if len(cand) == 0:
            for r in rerankers:
                for k in ks:
                    for m in ["P", "LegacyDiv"] + [f"{a}_{nm}" for nm in fl for a in ("Pfine", "RelDiv", "aNDCG")]:
                        put(r, k, m, np.nan)
            continue
        # corpus-wide candidate set (excl. self & same patient) for ideal / availability
        corpus_mask = patients != patients[qi]
        lists = rank_lists(En, qi, cand, k_max, rerankers,
                           rand_pool=np.flatnonzero(corpus_mask), rng=rng)

        ideals = {}
        S_q = {}
        for name, (Fb, par) in fl.items():
            sq = (par.astype(np.float32) @ coarse_b[qi].astype(np.float32)) > 0   # (F,)
            S_q[name] = sq
            Mc = Fb[corpus_mask][:, sq]
            ideals[name] = (_ideal_dcg(Mc, k_max, alpha), int(Mc.any(axis=0).sum()))

        for r in rerankers:
            sel = lists[r]
            for k in ks:
                s = sel[:k]
                rel = (coarse_b[s] & coarse_b[qi]).any(axis=1)
                put(r, k, "P", float(rel.mean()))
                cl = coarse_b[s].any(axis=0).sum()
                put(r, k, "LegacyDiv", float(cl) / min(k, C))
                for name, (Fb, par) in fl.items():
                    sq = S_q[name]
                    idcg_arr, n_avail = ideals[name]
                    if sq.sum() == 0 or n_avail == 0:
                        for m in ("Pfine", "RelDiv", "aNDCG"):
                            put(r, k, f"{m}_{name}", np.nan)
                        continue
                    Fs = Fb[s]
                    put(r, k, f"Pfine_{name}", float((Fs & Fb[qi]).any(axis=1).mean()))
                    H = Fs[:, sq]                                  # (k, |S_q|)
                    cov = H.any(axis=0).sum()
                    put(r, k, f"RelDiv_{name}", float(cov) / max(1, min(k, n_avail)))
                    # alpha-DCG
                    counts = np.zeros(H.shape[1])
                    dcg = 0.0
                    for rank in range(len(s)):
                        dcg += float((H[rank] * (1 - alpha) ** counts).sum()) / np.log2(rank + 2)
                        counts += H[rank]
                    idcg = idcg_arr[min(k, len(idcg_arr)) - 1]
                    put(r, k, f"aNDCG_{name}", dcg / idcg if idcg > 0 else np.nan)
        if verbose and (qn + 1) % 500 == 0:
            print(f"    queries {qn+1}/{len(qidx)}")

    res = {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}
    res["query_idx"] = qidx
    res["patient"] = patients[qidx]
    return res
