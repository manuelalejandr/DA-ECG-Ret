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
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wfdb

from data.ptbxl_dataset import load_ptbxl_metadata, patient_wise_split


def make_title(c, sups, subs, rel):
    """Short multi-line title: rank tag, superclasses, subclasses (wrapped)."""
    tag = "query" if c == 0 else f"#{c}" + ("" if rel else " not relevant")
    return tag + "\n" + "/".join(sups) + "\n" + textwrap.fill(", ".join(subs), width=15, break_long_words=False, break_on_hyphens=False)


def plot_panels(panels, out, fontsize=9.5):
    """panels: list of rows of (signal, title, is_query, relevant). Large fonts so that the titles stay
    legible when the figure is scaled to the text width (about 6 in)."""
    nr, nc = len(panels), len(panels[0])
    fig, axes = plt.subplots(nr, nc, figsize=(1.75 * nc, 2.05 * nr), squeeze=False)
    for r, row in enumerate(panels):
        for c, (x, title, isq, rel) in enumerate(row):
            ax = axes[r, c]
            ax.plot(x, lw=0.6, color="C3" if isq else "k")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=fontsize, linespacing=1.15, color="k" if (isq or rel) else "C3")
    plt.tight_layout(w_pad=0.4, h_pad=0.8)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close(fig)


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
    panels = []   # rows of (signal, title, is_query, relevant)
    for r, cname in enumerate(a.classes):
        ci = sup.index(cname)
        S = np.where(sub2sup[:, ci])[0]          # subtopics of the query's family
        cand = np.where(lab_sup[:, ci] & (lab_sub[:, S].sum(1) >= 1))[0]
        q = int(rng.choice(cand))
        s = sim[q].copy()
        s[pat == pat[q]] = -np.inf
        top = np.argsort(-s)[: a.k]
        row = []
        for c, idx in enumerate([q] + list(top)):
            sig, _ = wfdb.rdsamp(os.path.join(a.data_root, test.iloc[idx]["filename_lr"]))
            subs = [sub[j] for j in np.where(lab_sub[idx])[0]]
            sups = [sup[j] for j in np.where(lab_sup[idx])[0]]
            rel = bool((lab_sup[idx] & lab_sup[q]).any())
            row.append((sig[:, a.lead], make_title(c, sups, subs, rel), c == 0, rel))
        panels.append(row)
    plot_panels(panels, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
