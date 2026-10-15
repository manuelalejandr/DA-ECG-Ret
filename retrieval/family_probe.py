"""
Superclass ("family") probe for scenario B (labelled corpus, unlabelled query).

A linear classifier on the frozen embeddings, trained with the corpus labels only
(PTB-XL folds 1-8), predicts the query's superclass set. Recipe as in
retrieval.metrics.linear_probe_auroc (linear layer, BCE, Adam lr 1e-2, weight decay 1e-4,
100 full-batch epochs) with one addition: embeddings are standardised with the
corpus mean / s.d., so that encoders with different embedding scales are treated alike.

Decision rule: a class is predicted when its probability is >= 0.5; if no class reaches
0.5, the most probable class is used (top-1).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fit_predict_proba(tr_emb, tr_lab, te_emb, epochs=100, lr=1e-2, wd=1e-4, seed=0, device="cpu"):
    torch.manual_seed(seed)
    mu = tr_emb.mean(0, keepdims=True)
    sd = tr_emb.std(0, keepdims=True) + 1e-6
    Xtr = torch.tensor((tr_emb - mu) / sd, dtype=torch.float32, device=device)
    Ytr = torch.tensor((tr_lab > 0).astype(np.float32), device=device)
    Xte = torch.tensor((te_emb - mu) / sd, dtype=torch.float32, device=device)
    clf = nn.Linear(Xtr.shape[1], Ytr.shape[1]).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(clf(Xtr), Ytr).backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(clf(Xte)).cpu().numpy()


def decide(prob, thr=0.5):
    pred = prob >= thr
    none = ~pred.any(1)
    pred[none, prob[none].argmax(1)] = True
    return pred
