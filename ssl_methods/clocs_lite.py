"""
CLOCS-lite: simplified adaptation of CLOCS (Kiyasseh, Zhu, Clifton, "CLOCS:
Contrastive Learning of Cardiac Signals Across Space, Time, and Patients",
ICML 2021).

The original CLOCS defines three contrastive settings that all exploit
MULTI-LEAD structure (contrasting across leads at a fixed time, across time
within a lead, and across patients). Since this codebase commits to
UNIVARIATE (single-lead) ECGs, the "across leads" view is unavailable.
CLOCS-lite keeps the "across time" flavour (two augmented crops of the same
recording, as in SimCLR) and, where the underlying dataset exposes a
`patient_id` for multiple recordings per patient (e.g. PTB-XL, MIT-BIH),
prefers pairing two recordings from the SAME PATIENT as the positive pair
instead of two augmentations of the SAME recording -- the "across patients'
recordings" (CLOCS' temporal invariance across visits) flavour of the
original method. When a dataset has no repeat-patient structure (most
single-instance-per-series UCR/UEA problems), this reduces to plain
InfoNCE and should be reported as such (see README "Limitations").
"""
import torch

from models.encoder import build_encoder
from models.heads import ProjectionHead
from losses.infonce import info_nce_loss


def build_modules(cfg, device):
    encoder = build_encoder(cfg.encoder_name, cfg).to(device)
    proj_head = ProjectionHead(cfg.embed_dim, cfg.proj_dim, cfg.proj_dim).to(device)
    return {"embedding_model": encoder, "proj_head": proj_head}


def compute_loss(batch, modules, cfg, device, epoch=None, total_epochs=None):
    v1 = batch["view1"].to(device)
    v2 = batch["view2"].to(device)
    # NOTE: genuine patient-level positive-pair resampling (drawing v2 from
    # a different recording of the same patient_id) requires a
    # patient-indexed sampler at the DataLoader level; see README for how
    # to wire this in for PTB-XL / MIT-BIH. Left as standard augmented
    # views here to keep the reference pipeline dataset-agnostic.
    h1, h2 = modules["embedding_model"](v1), modules["embedding_model"](v2)
    z1, z2 = modules["proj_head"](h1), modules["proj_head"](h2)
    loss = info_nce_loss(z1, z2, tau=cfg.tau)
    return loss, {"infonce": loss.item(), "total": loss.item()}
