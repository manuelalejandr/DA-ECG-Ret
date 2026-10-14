"""
Qualitative figure (B3): top-5 retrieved traces for one MI and one CD query (NN retrieval).
Queries are drawn at random (fixed seed) among test queries of the given superclass that carry at least one subclass label,
so the figure is not cherry-picked.

    python qualitative_cases.py --data_root <.../ptb-xl/1.0.3> \
        --embed ./bench_outputs/embeds/ptbxl/da_ecg_ret_seed0.npz \
        --out ./figures/fig_qualitative.png --classes MI CD --seed 0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wfdb

from data.ptbxl_dataset import load_ptbxl_metadata, patient_wise_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--embed", required=True)
    ap.add_argument("--out", default="./figures/fig_qualitative.png")
    ap.add_argument("--classes", nargs="+", default=["MI", "CD"])
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    z = np.load(a.embed, allow_pickle=False)
    emb = z["te_emb"].astype(np.float64)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    pat = z["te_patients"]
    sup = [str(s) for s in z["vocab_super"]]
    sub = [str(s) for s in z["vocab_sub"]]
    lab_sup, lab_sub, sub2sup = z["te_lab"] > 0, z["te_lab_sub"] > 0, z["sub2super"] > 0
    # sub2super has shape (n_sub, n_super) (get_ptbxl_hierarchy)

    _, _, test = patient_wise_split(load_ptbxl_metadata(a.data_root))
    assert len(test) == len(emb), f"metadata ({len(test)}) and embeddings ({len(emb)}) disagree"
    assert np.all(test.patient_id.astype(int).values == np.asarray(pat).astype(int)), "row order differs"

    rng = np.random.RandomState(a.seed)
    sim = emb @ emb.T
    fig, axes = plt.subplots(len(a.classes), a.k + 1, figsize=(3.0 * (a.k + 1), 2.1 * len(a.classes)), squeeze=False)
    for r, cname in enumerate(a.classes):
        ci = sup.index(cname)
        # subtopics of the query's family
        S = np.where(sub2sup[:, ci])[0]
        cand = np.where(lab_sup[:, ci] & (lab_sub[:, S].sum(1) >= 1))[0]
        q = int(rng.choice(cand))
        s = sim[q].copy()
        s[pat == pat[q]] = -np.inf
        top = np.argsort(-s)[: a.k]
        for c, idx in enumerate([q] + list(top)):
            sig, _ = wfdb.rdsamp(os.path.join(a.data_root, test.iloc[idx]["filename_lr"]))
            x = sig[:, a.lead]
            ax = axes[r, c]
            ax.plot(x, lw=0.7, color="C3" if c == 0 else "k")
            ax.set_xticks([]); ax.set_yticks([])
            subs = [sub[j] for j in np.where(lab_sub[idx])[0]]
            sups = [sup[j] for j in np.where(lab_sup[idx])[0]]
            rel = (lab_sup[idx] & lab_sup[q]).any()
            tag = "query" if c == 0 else f"#{c}" + ("" if rel else " (not relevant)")
            ax.set_title(f"{tag}\n{'/'.join(sups)}: {','.join(subs)}", fontsize=7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    plt.savefig(a.out, dpi=300)
    print("saved", a.out)


if __name__ == "__main__":
    main()
