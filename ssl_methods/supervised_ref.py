"""SUPERVISED REFERENCE encoders (NOT self-supervised). They exist only to give
the benchmark an empirical upper reference: "how good can the embedding space
get when diagnostic labels ARE available?". They use the same encoder
backbone, the same augmentations (view1) and the same number of epochs as
the SSL methods, plus a linear classification head trained with BCE on

    supervised_fine  -> the ~44 SCP diagnostic statements (finest level)
    supervised_super -> the 5 diagnostic superclasses (coarsest level)

Comparing the two also shows what coarse supervision does to within-family
diversity (it pulls every recording of a superclass together, which should
hurt subtype coverage). Only meaningful on PTB-XL (needs label_code).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import build_encoder


def _build(cfg, device, n_out):
    encoder = build_encoder(cfg.encoder_name, cfg).to(device)
    head = nn.Linear(cfg.embed_dim, n_out).to(device)
    return {"embedding_model": encoder, "head": head}


def _loss(batch, modules, device, key):
    x = batch["view1"].to(device)
    y = batch[key].to(device).float()
    logits = modules["head"](modules["embedding_model"](x))
    loss = F.binary_cross_entropy_with_logits(logits, y)
    return loss, {"bce": loss.item(), "total": loss.item()}


class _Fine:
    @staticmethod
    def build_modules(cfg, device):
        return _build(cfg, device, cfg.n_label_code)

    @staticmethod
    def compute_loss(batch, modules, cfg, device, epoch=None, total_epochs=None):
        if "label_code" not in batch:
            raise RuntimeError("supervised_fine needs PTB-XL label_code (use --datasets ptbxl)")
        return _loss(batch, modules, device, "label_code")


class _Super:
    @staticmethod
    def build_modules(cfg, device):
        return _build(cfg, device, cfg.n_label_super)

    @staticmethod
    def compute_loss(batch, modules, cfg, device, epoch=None, total_epochs=None):
        return _loss(batch, modules, device, "label")


supervised_fine = _Fine
supervised_super = _Super
