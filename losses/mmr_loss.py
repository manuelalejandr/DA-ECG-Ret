"""
Differentiable relaxation of Maximal Marginal Relevance and the
retrieval-aware training loss (Section 3.4).

Classical MMR (Eq. 6):
    MMR(x) = lambda * sim(q, x) - (1 - lambda) * max_{y in S} sim(x, y)

This is non-differentiable because of the max over the already-selected set
S. Section 3.4.2 replaces it with a smooth surrogate evaluated over the
whole mini-batch (playing the role of the candidate pool):

    r_j = sim(z_q, z_j)                                             (7)
    d_j = (1/alpha) * log sum_k exp(alpha * sim(z_j, z_k))          (8)   -- log-sum-exp softmax of similarity to the batch, -> max as alpha -> inf
    s_j = lambda * r_j - (1 - lambda) * d_j                          (9)
    p_j = softmax_j(s_j)                                            (10)
    L_ret = - log p(z+_q | z_q)                                     (11)

Intuition: d_j penalises candidates that are highly similar to *many* other
points in the batch (i.e. redundant / "generic" regions of the embedding
space), independent of the query. The model is therefore pushed to make
the true positive stand out in relevance while not living in an
over-crowded region of the latent space.
"""
import torch
import torch.nn.functional as F


def smooth_redundancy(z_candidates: torch.Tensor, alpha: float,
                       exclude_self: bool = True, gate: torch.Tensor = None,
                       twin_index: torch.Tensor = None) -> torch.Tensor:
    """
    d_j = (1/alpha) * logsumexp_k( alpha * sim(z_j, z_k) + log(gate_jk) )   for every j.

    z_candidates: (N, d) L2-normalised embeddings (the candidate pool C_q).
    exclude_self: if True (default, and the FIX applied after diagnosing the
        v1 results — see README "Bug fixes after the first full run"),
        excludes k==j from the logsumexp. This matters a lot: cosine
        self-similarity sim(z_j, z_j) is always exactly 1, the maximum
        possible value, so if it is included it DOMINATES the logsumexp for
        every single candidate regardless of how crowded its neighbourhood
        actually is. Excluding the self-term restores a genuine
        "similarity to the REST of the batch" signal.
    twin_index: optional (N,) long tensor giving, for every row j, the index
        of j's OTHER augmented view in the same pool (its "twin"). FIX
        (post co-author review, Sept 2026): when the candidate pool is the
        concatenation of both augmented views (as in combined_loss below),
        excluding only k==j is *not enough* — candidate j's nearest
        neighbour in the pool is almost always its own twin, because
        InfoNCE pulls the two views of the same instance together BY
        DESIGN. Left unexcluded, the twin's near-1 similarity dominates the
        logsumexp almost as much as the literal self-term did before the
        v1->v2 fix, so d_j re-saturates in an attenuated form and the
        redundancy signal is muted for every candidate, not just some.
        Excluding both k==j and k==twin(j) restores "similarity to the rest
        of the batch, excluding the two positions that are guaranteed to be
        near-duplicates of j by construction of the training objective
        itself." Pass None if the pool does not have this twin structure
        (e.g. a plain inference-time candidate pool with one embedding per
        instance).
    gate: optional (N, N) multiplicative weight in [0, 1], DA-ECG-Ret v2's
        Component A (see models/prototypes.py::cluster_gate). gate[j, k]
        close to 0 suppresses the redundancy contribution from candidates
        likely sharing j's pseudo-cluster (probably the same clinical
        phenotype, which SHOULD be free to cluster tightly — this is
        exactly the "false negative" fix from the debiased/supervised
        contrastive learning literature, adapted to the label-free setting
        via on-the-fly clustering instead of true labels). gate[j, k] close
        to 1 keeps full redundancy pressure against genuinely different
        pseudo-clusters. Pass None to recover the ungated v1 behaviour.
    """
    sim = z_candidates @ z_candidates.T                 # (N, N) cosine sims
    if exclude_self:
        n = sim.shape[0]
        mask = torch.eye(n, dtype=torch.bool, device=sim.device)
        if twin_index is not None:
            rows = torch.arange(n, device=sim.device)
            mask[rows, twin_index] = True
        sim = sim.masked_fill(mask, float("-inf"))
    logits = alpha * sim
    if gate is not None:
        logits = logits + torch.log(gate.clamp_min(1e-6))
    d = torch.logsumexp(logits, dim=1) / alpha          # (N,)
    return d


def retrieval_aware_loss(
    z_query: torch.Tensor,
    z_candidates: torch.Tensor,
    positive_index: torch.Tensor,
    lam: float,
    alpha: float,
    exclude_self_index: torch.Tensor = None,
    exclude_self_from_redundancy: bool = True,
    gate: torch.Tensor = None,
    twin_index: torch.Tensor = None,
    tau_ret: float = 1.0,
) -> torch.Tensor:
    """
    Computes L_ret = -log p(z+_q | z_q) averaged over the batch of queries.

    z_query:        (B, d) raw (unnormalised) query embeddings.
    z_candidates:   (N, d) raw candidate pool (typically both augmented
                    views concatenated, N = 2B).
    positive_index: (B,) long tensor; positive_index[i] is the row in
                    z_candidates that is the true positive for query i.
    lam, alpha:     as in Eq. 8-9.
    exclude_self_index: optional (B,) long tensor giving, for each query,
                    the row in z_candidates equal to the query itself
                    (to exclude a trivial "retrieve yourself" solution).
                    Pass None if the query is not present in the pool.
    exclude_self_from_redundancy: see smooth_redundancy() docstring — the
                    bug fix that turned out to matter most in practice.
    gate:           optional (N, N) pseudo-cluster gate, see
                    smooth_redundancy() docstring (Component A, v2).
    twin_index:     optional (N,) long tensor, see smooth_redundancy()
                    docstring (twin-exclusion fix, v2.1).
    tau_ret:        temperature dividing the score s before the softmax
                    (Eq. 10) -- FIX (post co-author review, Sept 2026).
                    r and d both live in roughly [-1, 1] (cosine sims /
                    logsumexp of cosine sims), so with the pool size used
                    in practice (N ~ 2B, B up to a few hundred) the
                    un-scaled softmax over s is nearly uniform: unlike
                    InfoNCE, which divides sim/tau with tau=0.07 to produce
                    a sharp, high-gradient distribution, Eq. 5 previously
                    had NO temperature at all (tau_ret == 1.0 is the old,
                    unscaled behaviour, kept as the default so existing
                    configs stay reproducible until deliberately swept).
                    Passing tau_ret < 1 sharpens p_j and restores a
                    gradient magnitude comparable to InfoNCE's.
    """
    zq = F.normalize(z_query, dim=1)
    zc = F.normalize(z_candidates, dim=1)

    r = zq @ zc.T                                  # (B, N) relevance, Eq. 7
    d = smooth_redundancy(zc, alpha, exclude_self=exclude_self_from_redundancy,
                           gate=gate, twin_index=twin_index)  # (N,) redundancy, Eq. 8
    s = lam * r - (1 - lam) * d.unsqueeze(0)        # (B, N) Eq. 9
    s = s / tau_ret                                  # temperature scaling, Eq. 9'

    if exclude_self_index is not None:
        s = s.clone()
        s.scatter_(1, exclude_self_index.unsqueeze(1), float("-inf"))

    log_p = F.log_softmax(s, dim=1)                 # Eq. 10 (log domain)
    loss = -log_p.gather(1, positive_index.unsqueeze(1)).squeeze(1)  # Eq. 11
    return loss.mean()


def combined_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    tau: float,
    lam: float,
    alpha: float,
    beta: float,
    exclude_self_from_redundancy: bool = True,
    gate: torch.Tensor = None,
    tau_ret: float = 1.0,
    exclude_twin_from_redundancy: bool = True,
):
    """
    Full DA-ECG-Ret objective (Eq. 12): L = L_InfoNCE + beta * L_ret.

    z1, z2: (B, d) projected embeddings of the two augmented views.
    Candidate pool for the retrieval-aware term is the concatenation of
    both views (N = 2B), each view acting once as query set. Note the pool
    has a fixed twin structure by construction: pool row i (< B) is view1
    of instance i, and pool row i+B is view2 of the SAME instance i --
    exactly the pair InfoNCE pulls together. twin_index encodes this and is
    passed to smooth_redundancy() (via retrieval_aware_loss) so the
    redundancy term does not re-saturate on these known near-duplicates
    (see mmr_loss.py::smooth_redundancy docstring, "twin-exclusion fix").
    gate: optional (2B, 2B) pseudo-cluster gate shared by both the view1-
    and view2-as-query passes (computed once from the pooled embeddings by
    the caller — see ssl_methods/da_ecg_ret.py).
    tau_ret: temperature for Eq. 10's softmax; see retrieval_aware_loss()
    docstring. Default 1.0 reproduces the pre-fix (unscaled) behaviour.
    """
    from .infonce import info_nce_loss

    B = z1.shape[0]
    l_infonce = info_nce_loss(z1, z2, tau=tau)

    z_pool = torch.cat([z1, z2], dim=0)             # (2B, d)
    idx = torch.arange(B, device=z1.device)
    twin_index = None
    if exclude_twin_from_redundancy:
        twin_index = torch.cat([idx + B, idx], dim=0)   # (2B,): twin(i)=i+B, twin(i+B)=i

    # queries = view1, positives = view2 (offset by B in the pool)
    pos_idx_1 = idx + B
    self_idx_1 = idx
    l_ret_1 = retrieval_aware_loss(z1, z_pool, pos_idx_1, lam, alpha, self_idx_1,
                                   exclude_self_from_redundancy, gate=gate,
                                   twin_index=twin_index, tau_ret=tau_ret)

    # queries = view2, positives = view1
    pos_idx_2 = idx
    self_idx_2 = idx + B
    l_ret_2 = retrieval_aware_loss(z2, z_pool, pos_idx_2, lam, alpha, self_idx_2,
                                   exclude_self_from_redundancy, gate=gate,
                                   twin_index=twin_index, tau_ret=tau_ret)

    l_ret = 0.5 * (l_ret_1 + l_ret_2)
    total = l_infonce + beta * l_ret
    return total, {"infonce": l_infonce.item(), "ret": l_ret.item(), "total": total.item()}
