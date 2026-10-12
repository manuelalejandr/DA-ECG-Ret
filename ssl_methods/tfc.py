"""TF-C baseline (Zhang et al., NeurIPS 2022) — see losses/tfc_loss.py for
the time-frequency consistency loss and simplifications made."""
import torch

from models.encoder import build_encoder
from models.heads import ProjectionHead
from models.tfc_model import TFCEmbeddingWrapper
from losses.tfc_loss import to_freq_magnitude, FreqAugment, tfc_loss


def build_modules(cfg, device):
    time_encoder = build_encoder(cfg.encoder_name, cfg).to(device)
    freq_encoder = build_encoder(cfg.encoder_name, cfg).to(device)
    time_proj = ProjectionHead(cfg.embed_dim, cfg.proj_dim, cfg.proj_dim).to(device)
    freq_proj = ProjectionHead(cfg.embed_dim, cfg.proj_dim, cfg.proj_dim).to(device)
    embedding_model = TFCEmbeddingWrapper(time_encoder, freq_encoder).to(device)
    return {
        "embedding_model": embedding_model,
        "time_encoder": time_encoder,
        "freq_encoder": freq_encoder,
        "time_proj": time_proj,
        "freq_proj": freq_proj,
        "_freq_augment": FreqAugment(),
    }


def compute_loss(batch, modules, cfg, device, epoch=None, total_epochs=None):
    v1 = batch["view1"].to(device)
    v2 = batch["view2"].to(device)
    raw = batch["raw"].to(device)
    freq_aug = modules["_freq_augment"]

    base_mag = to_freq_magnitude(raw)                       # (B, 1, F)
    freq_v1 = torch.stack([freq_aug(m) for m in base_mag])
    freq_v2 = torch.stack([freq_aug(m) for m in base_mag])

    time_h1 = modules["time_encoder"](v1)
    time_h2 = modules["time_encoder"](v2)
    freq_h1 = modules["freq_encoder"](freq_v1)
    freq_h2 = modules["freq_encoder"](freq_v2)

    time_z1, time_z2 = modules["time_proj"](time_h1), modules["time_proj"](time_h2)
    freq_z1, freq_z2 = modules["freq_proj"](freq_h1), modules["freq_proj"](freq_h2)

    loss, logs = tfc_loss(
        time_z1, time_z2, freq_z1, freq_z2, time_z1, freq_z1,
        tau=cfg.tau, consistency_weight=cfg.tfc_consistency_weight,
    )
    return loss, logs
