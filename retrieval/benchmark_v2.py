"""
Extended diversity-aware retrieval evaluation (numpy only).

Adds to retrieval/benchmark_eval.py (which is kept unchanged so earlier numbers
remain reproducible):

  * a SEPARATE corpus (e.g. PTB-XL folds 1-8) distinct from the query set
    (fold 10), or the original test-vs-test setting (`same_set=True`);
  * three exclusion modes for the leakage ablation: "patient" (default: drop the
    query and every record of its patient), "self" (drop only the query) and
    "none" (the query may retrieve itself);
  * more metrics: alpha-nDCG for several alpha, S-recall@K (Zhai et al., 2003),
    nDCG-IA and ERR-IA (Agrawal et al., 2009; uniform intent distribution over the
    query's subtopics that exist in the corpus);
  * more re-rankers: DPP quality-exponent sweep, xQuAD with pseudo-intents
    (k-means on the shortlist), xQuAD with oracle intents, and an ORACLE ranking
    (greedy alpha-DCG maximisation with the true labels, restricted to the same
    shortlist) that bounds what ANY re-ranker could gain;
  * shortlist diagnostics: fraction of the query's available subtopics that
    appear in the N-candidate shortlist;
  * per-query structure (|S_q|, available subtopics) so results can be
    stratified (e.g. dropping queries with |S_q| = 1, or per superclass).

Output keys: "<ranker>|<K>|<metric>" -> (n_queries,) array (NaN if undefined),
plus "struct|0|<name>" arrays, "query_idx" and "patient".
"""
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from retrieval.dpp_rerank import dpp_greedy_map
from retrieval.benchmark_eval import (_normalize, mmr_select, _ideal_dcg)

MMR_LAMBDAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
DPP_THETAS = (0.25, 0.5, 2.0, 4.0)          # plus theta = 1 under the name "dpp"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 0.9)
MAIN_ALPHA = 0.5
XQUAD_LAMBDAS = (0.5,)
N_PSEUDO_INTENTS = 5


def default_rerankers(fine_names: Sequence[str]) -> List[str]:
    r = ["nn"] + [f"mmr@{l}" for l in MMR_LAMBDAS] + ["dpp"] + [f"dpp@{t}" for t in DPP_THETAS]
    r += [f"xquad@{l}" for l in XQUAD_LAMBDAS]
    for nm in fine_names:
        r += [f"xquad_or_{nm}@{l}" for l in XQUAD_LAMBDAS] + [f"oracle_{nm}"]
    return r + ["rand"]


# ----------------------------------------------------------------------------
# shortlists
# ----------------------------------------------------------------------------
def shortlists_v2(q_emb, c_emb, q_pat, c_pat, same_set: bool, exclusion: str,
                  n_shortlist: int, query_idx: np.ndarray, block: int = 256) -> np.ndarray:
    """Top-`n_shortlist` corpus indices per query by cosine similarity.
    exclusion: 'patient' | 'self' | 'none'. With same_set=True query i is corpus item i."""
    Q = _normalize(q_emb.astype(np.float32))
    D = Q if same_set else _normalize(c_emb.astype(np.float32))
    n = len(D)
    N = min(n_shortlist, n - (1 if (same_set and exclusion != "none") else 0))
    out = np.full((len(query_idx), n_shortlist), -1, dtype=np.int64)
    for b in range(0, len(query_idx), block):
        qi = query_idx[b:b + block]
        S = Q[qi] @ D.T
        if exclusion == "patient" and q_pat is not None and c_pat is not None:
            S[q_pat[qi][:, None] == c_pat[None, :]] = -np.inf
        elif exclusion in ("patient", "self") and same_set:
            S[np.arange(len(qi)), qi] = -np.inf
        idx = np.argpartition(-S, N - 1, axis=1)[:, :N]
        sims = np.take_along_axis(S, idx, axis=1)
        order = np.argsort(-sims, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        sims = np.take_along_axis(sims, order, axis=1)
        out[b:b + len(qi), :N] = np.where(np.isfinite(sims), idx, -1)
    return out


# ----------------------------------------------------------------------------
# re-rankers that need intents
# ----------------------------------------------------------------------------
def _kmeans(X: np.ndarray, k: int, rng: np.random.RandomState, iters: int = 15):
    n = len(X)
    k = min(k, n)
    cent = X[rng.choice(n, k, replace=False)].copy()
    for _ in range(iters):
        lab = np.argmax(X @ cent.T, axis=1)            # cosine (X unit norm)
        for j in range(k):
            m = lab == j
            if m.any():
                c = X[m].mean(0)
                cent[j] = c / (np.linalg.norm(c) + 1e-8)
    return cent, np.argmax(X @ cent.T, axis=1)


def xquad_select(rel: np.ndarray, P_di: np.ndarray, P_i: np.ndarray, k: int, lam: float) -> List[int]:
    """xQuAD (Santos et al., WWW 2010).
    rel   (n,)   P(d|q)               relevance in [0,1]
    P_di  (n,m)  P(d|q,i)             coverage of intent i by candidate d
    P_i   (m,)   P(i|q)               intent distribution
    score(d) = (1-lam) rel(d) + lam * sum_i P_i P_di[d,i] prod_{j in S} (1 - P_di[j,i])"""
    n = len(rel)
    k = min(k, n)
    novelty = np.ones(P_di.shape[1])
    avail = np.ones(n, dtype=bool)
    sel = []
    for _ in range(k):
        score = (1 - lam) * rel + lam * (P_di @ (P_i * novelty))
        score[~avail] = -np.inf
        j = int(np.argmax(score))
        sel.append(j)
        avail[j] = False
        novelty = novelty * (1 - P_di[j])
    return sel


def oracle_select(M: np.ndarray, k: int, alpha: float) -> List[int]:
    """Greedy alpha-DCG maximisation with TRUE subtopics inside the shortlist.
    M: (n, m) bool. Ties and zero-gain positions are filled by candidate order (NN)."""
    n = M.shape[0]
    k = min(k, n)
    Mf = M.astype(np.float32)
    counts = np.zeros(M.shape[1], dtype=np.float32)
    used = np.zeros(n, dtype=bool)
    sel = []
    for r in range(k):
        gain = Mf @ ((1.0 - alpha) ** counts) if M.shape[1] else np.zeros(n)
        gain = np.where(used, -1.0, gain)
        # tie-break toward the better (earlier) NN rank
        gain = gain - 1e-6 * np.arange(n)
        j = int(np.argmax(gain))
        sel.append(j)
        used[j] = True
        if M.shape[1]:
            counts += Mf[j]
    return sel


# ----------------------------------------------------------------------------
# metric helpers for one cohort
# ----------------------------------------------------------------------------
def _adcg(H: np.ndarray, alpha: float) -> float:
    counts = np.zeros(H.shape[1])
    dcg = 0.0
    for r in range(len(H)):
        dcg += float((H[r] * (1 - alpha) ** counts).sum()) / np.log2(r + 2)
        counts += H[r]
    return dcg


def _ndcg_ia(H: np.ndarray, n_i: np.ndarray, k: int) -> float:
    disc = 1.0 / np.log2(np.arange(2, len(H) + 2))
    dcg_i = (H * disc[:, None]).sum(0)
    cum = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))])
    idcg_i = cum[np.minimum(n_i, k)]
    return float(np.mean(dcg_i / idcg_i))


def _err_ia(H: np.ndarray) -> float:
    R = 0.5 * H.astype(np.float64)                      # binary grades, g_max = 1
    surv = np.cumprod(np.vstack([np.ones((1, H.shape[1])), 1 - R[:-1]]), axis=0)
    ranks = (1.0 / np.arange(1, len(H) + 1))[:, None]
    return float(np.mean((ranks * R * surv).sum(0)))


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
def evaluate_v2(
    q_emb: np.ndarray, q_pat: Optional[np.ndarray], q_coarse: np.ndarray,
    q_fine: Dict[str, Tuple[np.ndarray, np.ndarray]],      # name -> (multi-hot (nq,F), parent (F,C))
    c_emb: Optional[np.ndarray] = None, c_pat: Optional[np.ndarray] = None,
    c_coarse: Optional[np.ndarray] = None, c_fine: Optional[Dict[str, np.ndarray]] = None,
    same_set: bool = True, exclusion: str = "patient",
    rerankers: Optional[List[str]] = None, ks: Tuple[int, ...] = (5, 10, 20),
    n_shortlist: int = 200, alphas: Sequence[float] = ALPHAS,
    max_queries: Optional[int] = None, seed: int = 0, verbose: bool = True,
) -> Dict[str, np.ndarray]:
    assert exclusion in ("patient", "self", "none")
    if same_set:
        c_emb, c_pat, c_coarse = q_emb, q_pat, q_coarse
        c_fine = {nm: f for nm, (f, _) in q_fine.items()}
    names = list(q_fine)
    rerankers = rerankers or default_rerankers(names)
    rng = np.random.RandomState(seed)
    nq = len(q_emb)
    qidx = np.arange(nq)
    if max_queries is not None and max_queries < nq:
        qidx = np.sort(rng.choice(nq, max_queries, replace=False))
    k_max = max(ks)
    k_main = 10 if 10 in ks else k_max
    SL = shortlists_v2(q_emb, c_emb, q_pat, c_pat, same_set, exclusion, n_shortlist, qidx)

    Qn = _normalize(q_emb.astype(np.float32))
    Dn = Qn if same_set else _normalize(c_emb.astype(np.float32))
    qc = q_coarse > 0
    cc = c_coarse > 0
    C = q_coarse.shape[1]
    fl = {nm: (q_fine[nm][0] > 0, q_fine[nm][1] > 0, c_fine[nm] > 0) for nm in names}

    out: Dict[str, List[float]] = {}

    def put(r, k, m, v):
        out.setdefault(f"{r}|{k}|{m}", []).append(v)

    metric_names = ["P", "LegacyDiv"]
    for nm in names:
        metric_names += [f"{a}_{nm}" for a in ("Pfine", "RelDiv", "Srecall", "nDCGIA", "ERRIA", "aNDCG")]
        metric_names += [f"aNDCG{a:g}_{nm}" for a in alphas if a != MAIN_ALPHA]

    for qn, qi in enumerate(qidx):
        cand = SL[qn][SL[qn] >= 0]
        if len(cand) == 0:
            for r in rerankers:
                for k in ks:
                    for m in metric_names:
                        put(r, k, m, np.nan)
            continue
        # corpus mask (what the ideal ranking / availability may use)
        if exclusion == "patient" and q_pat is not None and c_pat is not None:
            cmask = c_pat != q_pat[qi]
        elif exclusion in ("patient", "self") and same_set:
            cmask = np.ones(len(Dn), dtype=bool)
            cmask[qi] = False
        else:
            cmask = np.ones(len(Dn), dtype=bool)

        cE = Dn[cand]
        qv = Qn[qi]
        sim_q = cE @ qv
        sim_mat = cE @ cE.T

        # per-level query structure
        sq, Mc, n_avail_vec, idcgs, M_cand = {}, {}, {}, {}, {}
        for nm, (Fq, par, Fc) in fl.items():
            s = (par.astype(np.float32) @ qc[qi].astype(np.float32)) > 0
            sq[nm] = s
            Mcorp = Fc[cmask][:, s]
            n_i = Mcorp.sum(0)
            keep = n_i > 0
            Mc[nm] = Mcorp[:, keep]
            n_avail_vec[nm] = n_i[keep]
            idcgs[nm] = {a: _ideal_dcg(Mc[nm], k_max, a) for a in alphas}
            M_cand[nm] = Fc[cand][:, s][:, keep]
            put("struct", 0, f"nsub_{nm}", float(s.sum()))
            put("struct", 0, f"navail_{nm}", float(keep.sum()))
            tot = int(keep.sum())
            put("shortlist", len(cand), f"SLrecall_{nm}",
                float(M_cand[nm].any(0).sum()) / tot if tot else np.nan)
        relc = (cc[cand] & qc[qi]).any(axis=1)
        put("shortlist", len(cand), "SLP", float(relc.mean()))

        # rankings
        lists = {}
        for r in rerankers:
            if r == "nn":
                lists[r] = cand[:k_max]
            elif r.startswith("mmr@"):
                lists[r] = cand[mmr_select(sim_q, sim_mat, k_max, float(r.split("@")[1]))]
            elif r == "dpp" or r.startswith("dpp@"):
                th = 1.0 if r == "dpp" else float(r.split("@")[1])
                ids = dpp_greedy_map(qv, cE, list(range(len(cand))), k_max, quality_weight=th)
                lists[r] = cand[ids]
            elif r.startswith("xquad@"):
                lam = float(r.split("@")[1])
                cent, lab = _kmeans(cE, N_PSEUDO_INTENTS, rng)
                memb = np.exp((cE @ cent.T) / 0.1)
                memb = memb / memb.sum(1, keepdims=True)
                P_i = np.bincount(lab, minlength=len(cent)).astype(np.float64)
                P_i = P_i / P_i.sum()
                rel = np.clip(sim_q, 0, None)
                rel = rel / (rel.max() + 1e-8)
                lists[r] = cand[xquad_select(rel, memb, P_i, k_max, lam)]
            elif r.startswith("xquad_or_"):
                nm, lam = r[len("xquad_or_"):].split("@")
                M = M_cand[nm].astype(np.float64)
                rel = np.clip(sim_q, 0, None)
                rel = rel / (rel.max() + 1e-8)
                if M.shape[1] == 0:
                    lists[r] = cand[:k_max]
                else:
                    lists[r] = cand[xquad_select(rel, M, np.ones(M.shape[1]) / M.shape[1], k_max, float(lam))]
            elif r.startswith("oracle_"):
                nm = r[len("oracle_"):]
                lists[r] = cand[oracle_select(M_cand[nm], k_max, MAIN_ALPHA)]
            elif r == "rand":
                lists[r] = rng.choice(np.flatnonzero(cmask), size=min(k_max, int(cmask.sum())), replace=False)
            else:
                raise ValueError(r)

        # metrics
        for r in rerankers:
            sel = lists[r]
            for k in ks:
                s = sel[:k]
                rel = (cc[s] & qc[qi]).any(axis=1)
                put(r, k, "P", float(rel.mean()))
                put(r, k, "LegacyDiv", float(cc[s].any(axis=0).sum()) / min(k, C))
                for nm, (Fq, par, Fc) in fl.items():
                    nav = Mc[nm].shape[1]
                    if sq[nm].sum() == 0 or nav == 0:
                        for m in metric_names:
                            if m.endswith(f"_{nm}"):
                                put(r, k, m, np.nan)
                        continue
                    put(r, k, f"Pfine_{nm}", float((Fc[s] & Fq[qi]).any(axis=1).mean()))
                    Hk = Fc[s][:, sq[nm]]
                    n_i_full = Fc[cmask][:, sq[nm]].sum(0)
                    Hk = Hk[:, n_i_full > 0].astype(np.float64)
                    cov = Hk.any(axis=0).sum()
                    put(r, k, f"RelDiv_{nm}", float(cov) / max(1, min(k, nav)))
                    put(r, k, f"Srecall_{nm}", float(cov) / nav)
                    put(r, k, f"nDCGIA_{nm}", _ndcg_ia(Hk, n_avail_vec[nm], k))
                    put(r, k, f"ERRIA_{nm}", _err_ia(Hk))
                    for a in alphas:
                        idcg = idcgs[nm][a][min(k, k_max) - 1]
                        v = _adcg(Hk, a) / idcg if idcg > 0 else np.nan
                        put(r, k, f"aNDCG_{nm}" if a == MAIN_ALPHA else f"aNDCG{a:g}_{nm}", v)
        if verbose and (qn + 1) % 500 == 0:
            print(f"    queries {qn + 1}/{len(qidx)}", flush=True)

    res = {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}
    res["query_idx"] = qidx
    if q_pat is not None:
        res["patient"] = q_pat[qidx]
    return res
