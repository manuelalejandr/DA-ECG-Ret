"""TS2Vec-lite baseline (Yue et al., AAAI 2022) — see
losses/ts2vec_loss.py for the hierarchical contrastive loss and a note on
simplifications made to reuse this project's shared encoder."""
from models.encoder import build_encoder
from losses.ts2vec_loss import hierarchical_contrastive_loss


def build_modules(cfg, device):
    encoder = build_encoder(cfg.encoder_name, cfg).to(device)
    return {"embedding_model": encoder}


def compute_loss(batch, modules, cfg, device, epoch=None, total_epochs=None):
    v1 = batch["view1"].to(device)
    v2 = batch["view2"].to(device)
    h1_seq = modules["embedding_model"](v1, return_seq=True)   # (B, L, d)
    h2_seq = modules["embedding_model"](v2, return_seq=True)
    loss = hierarchical_contrastive_loss(h1_seq, h2_seq, max_levels=3, tau=cfg.tau)
    return loss, {"hierarchical": loss.item(), "total": loss.item()}
