"""SimCLR baseline (Chen et al., ICML 2020): InfoNCE on two augmented
views, using this project's shared encoder backbone."""
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
    h1, h2 = modules["embedding_model"](v1), modules["embedding_model"](v2)
    z1, z2 = modules["proj_head"](h1), modules["proj_head"](h2)
    loss = info_nce_loss(z1, z2, tau=cfg.tau)
    return loss, {"infonce": loss.item(), "total": loss.item()}
