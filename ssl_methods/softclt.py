"""SoftCLT baseline (Lee et al., ICLR 2024) — see losses/softclt_loss.py
for the soft instance-wise contrastive loss and simplifications made."""
from models.encoder import build_encoder
from models.heads import ProjectionHead
from losses.softclt_loss import soft_instance_contrastive_loss


def build_modules(cfg, device):
    encoder = build_encoder(cfg.encoder_name, cfg).to(device)
    proj_head = ProjectionHead(cfg.embed_dim, cfg.proj_dim, cfg.proj_dim).to(device)
    return {"embedding_model": encoder, "proj_head": proj_head}


def compute_loss(batch, modules, cfg, device, epoch=None, total_epochs=None):
    v1 = batch["view1"].to(device)
    v2 = batch["view2"].to(device)
    raw = batch["raw"].to(device)
    h1, h2 = modules["embedding_model"](v1), modules["embedding_model"](v2)
    z1, z2 = modules["proj_head"](h1), modules["proj_head"](h2)
    loss = soft_instance_contrastive_loss(
        z1, z2, raw, tau_emb=cfg.tau, tau_inst=cfg.softclt_tau_inst, alpha=cfg.softclt_alpha,
    )
    return loss, {"soft_infonce": loss.item(), "total": loss.item()}
