"""
DA-ECG-Ret v2 (proposed method): InfoNCE + a pseudo-cluster-gated,
curriculum-weighted retrieval-aware MMR loss (Component A) + a
physiologically-grounded auxiliary regression pretext task (Component B).

This redesign directly targets the failure mode diagnosed after the first
full benchmark run and the beta/lambda sweep (see README, "Bug fixes after
the first full run" and "Redesign: DA-ECG-Ret v2"): the v1 retrieval-aware
loss penalised ANY batch-wide redundancy with no notion of class, which
actively fought against the class-discriminative clustering that InfoNCE
needs to do well -- worse on higher-class-cardinality datasets. Component A
fixes the MECHANISM (only penalise redundancy against likely-different
pseudo-clusters); Component B adds a complementary, fully self-supervised
signal (classical ECG morphology/rhythm descriptors) that gives the
encoder something concrete and clinically motivated to represent, instead
of relying purely on augmentation-invariance.
"""
from models.encoder import build_encoder
from models.heads import ProjectionHead, PhysioRegressionHead
from models.prototypes import batch_soft_kmeans, cluster_gate
from losses.mmr_loss import combined_loss
import torch
import torch.nn.functional as F


def build_modules(cfg, device):
    encoder = build_encoder(cfg.encoder_name, cfg).to(device)
    proj_head = ProjectionHead(cfg.embed_dim, cfg.proj_dim, cfg.proj_dim).to(device)
    modules = {"embedding_model": encoder, "proj_head": proj_head}
    if cfg.use_physio_aux_task:
        modules["phys_head"] = PhysioRegressionHead(
            cfg.embed_dim, cfg.embed_dim, cfg.n_physio_features
        ).to(device)
    return modules


def _effective_beta(cfg, epoch, total_epochs):
    """Linear warm-up: beta ramps 0 -> cfg.beta over the first
    `beta_warmup_frac` of training, then holds. Lets InfoNCE establish
    reasonable class clustering before the retrieval-aware term starts
    exerting pressure, rather than the two objectives fighting from
    epoch 0 (a plausible contributor to L_ret's near-flat loss curves
    observed in the first full run)."""
    if epoch is None or total_epochs is None or total_epochs <= 0:
        return cfg.beta
    warmup_epochs = max(1, int(cfg.beta_warmup_frac * total_epochs))
    frac = min(1.0, (epoch + 1) / warmup_epochs)
    return cfg.beta * frac


def compute_loss(batch, modules, cfg, device, epoch=None, total_epochs=None):
    v1 = batch["view1"].to(device)
    v2 = batch["view2"].to(device)
    h1, h2 = modules["embedding_model"](v1), modules["embedding_model"](v2)
    z1, z2 = modules["proj_head"](h1), modules["proj_head"](h2)

    gate = None
    if cfg.use_cluster_gate:
        pool_z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
        Q = batch_soft_kmeans(
            pool_z, k=cfg.n_pseudo_clusters, n_iters=cfg.kmeans_iters,
            temperature=cfg.kmeans_temperature,
        )
        gate = cluster_gate(Q)

    eff_beta = _effective_beta(cfg, epoch, total_epochs)
    loss, logs = combined_loss(
        z1, z2, cfg.tau, cfg.lam, cfg.alpha, eff_beta,
        exclude_self_from_redundancy=cfg.exclude_self_from_redundancy,
        gate=gate,
        tau_ret=getattr(cfg, "tau_ret", 1.0),
        exclude_twin_from_redundancy=getattr(cfg, "exclude_twin_from_redundancy", True),
    )
    logs["effective_beta"] = eff_beta

    if cfg.use_physio_aux_task and "phys_head" in modules:
        phys_target = batch["phys"].to(device).float()      # (B, n_features)
        # regress from BOTH views (their augmentations still share the same
        # underlying morphology/rhythm target)
        pred1 = modules["phys_head"](h1)
        pred2 = modules["phys_head"](h2)
        # standardize the target using batch statistics (see README) so the
        # regression head isn't fighting an arbitrary raw feature scale
        mu = phys_target.mean(dim=0, keepdim=True)
        sigma = phys_target.std(dim=0, keepdim=True).clamp_min(1e-4)
        target_n = (phys_target - mu) / sigma
        l_phys = 0.5 * (F.mse_loss(pred1, target_n) + F.mse_loss(pred2, target_n))
        loss = loss + cfg.gamma_physio * l_phys
        logs["physio"] = l_phys.item()
        logs["total"] = loss.item()

    return loss, logs
