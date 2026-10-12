"""
One-shot diagnostic: on a single real batch from a chosen dataset, computes
the gradient norm (w.r.t. the encoder's parameters) contributed by
L_infonce vs. L_ret separately, both WITH and WITHOUT the
exclude_self_from_redundancy fix (see losses/mmr_loss.py). This directly
answers "is L_ret actually able to push the encoder anywhere, and did the
fix help?" without waiting for a full training run.

Usage:
    python diagnose_gradients.py --dataset ucr --ucr_name ECG5000
    python diagnose_gradients.py --dataset mitbih
"""
import argparse

import torch
from torch.utils.data import DataLoader

from config import Config
from device_utils import resolve_device
from data.augmentations import default_ecg_augmentation
from data.common import ContrastiveWrapper, get_dataset_splits, effective_batch_size
from models.encoder import build_encoder
from models.heads import ProjectionHead
from losses.infonce import info_nce_loss
from losses.mmr_loss import combined_loss, retrieval_aware_loss


def grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.pow(2).sum().item()
    return total ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "ptbxl", "ucr", "mitbih"], default="ucr")
    parser.add_argument("--ucr_name", type=str, default="ECG5000")
    parser.add_argument("--data_root", type=str, default="./ptbxl_data")
    parser.add_argument("--encoder", choices=["transformer", "resnet18", "fcn"], default="transformer")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    cfg.dataset = args.dataset
    cfg.ucr_name = args.ucr_name
    cfg.data_root = args.data_root
    cfg.encoder_name = args.encoder
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    device = resolve_device(verbose=True)

    train_base, _, _, _ = get_dataset_splits(args, cfg)
    augment = default_ecg_augmentation(cfg.sampling_rate)
    ds = ContrastiveWrapper(train_base, augment)
    bs = effective_batch_size(cfg.batch_size, len(ds))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    batch = next(iter(loader))

    torch.manual_seed(0)
    encoder = build_encoder(cfg.encoder_name, cfg).to(device)
    proj_head = ProjectionHead(cfg.embed_dim, cfg.proj_dim, cfg.proj_dim).to(device)
    params = list(encoder.parameters()) + list(proj_head.parameters())

    v1 = batch["view1"].to(device)
    v2 = batch["view2"].to(device)
    h1, h2 = encoder(v1), encoder(v2)
    z1, z2 = proj_head(h1), proj_head(h2)

    l_infonce = info_nce_loss(z1, z2, tau=cfg.tau)
    g_infonce = grad_norm(l_infonce, params)

    print(f"\nDataset: {args.ucr_name if args.dataset == 'ucr' else args.dataset} "
          f"(batch_size={bs}, beta={args.beta}, lambda={args.lam}, alpha={args.alpha})\n")
    print(f"||grad L_infonce|| = {g_infonce:.6f}\n")

    from losses.mmr_loss import smooth_redundancy
    zc = torch.nn.functional.normalize(torch.cat([z1, z2], dim=0), dim=1)
    for exclude_self, label in [(False, "WITHOUT fix (original, buggy)"),
                                 (True, "WITH fix (self-similarity excluded)")]:
        d = smooth_redundancy(zc, args.alpha, exclude_self=exclude_self)
        d_range = (d.max() - d.min()).item()

        B = z1.shape[0]
        z_pool = torch.cat([z1, z2], dim=0)
        idx = torch.arange(B, device=device)
        l_ret_1 = retrieval_aware_loss(z1, z_pool, idx + B, args.lam, args.alpha, idx, exclude_self)
        l_ret_2 = retrieval_aware_loss(z2, z_pool, idx, args.lam, args.alpha, idx + B, exclude_self)
        l_ret = 0.5 * (l_ret_1 + l_ret_2)
        g_ret = grad_norm(l_ret, params)

        print(f"[{label}]")
        print(f"  redundancy term d_j: min={d.min().item():.4f} max={d.max().item():.4f} "
              f"range={d_range:.4f}  <-- THE key diagnostic (see note below)")
        print(f"  ||grad L_ret|| = {g_ret:.6f} (effective ratio at beta={args.beta}: "
              f"{args.beta * g_ret / max(g_infonce, 1e-12):.4f})\n")

    print("Note: the self-similarity term sim(z_j, z_j) is a CONSTANT (=1 exactly, always, "
          "post-normalization), so it contributes essentially ZERO direct gradient -- "
          "||grad L_ret|| alone can look similar with/without the fix and is NOT the right "
          "signal for this specific bug. What the bug actually breaks is the DYNAMIC RANGE "
          "of d_j across candidates (the 'range' printed above): when it collapses toward ~0 "
          "(WITHOUT the fix), every candidate gets an almost identical redundancy penalty, so "
          "the softmax in Eq. 9-10 barely responds to it regardless of beta -- L_ret degenerates "
          "into a noisy near-duplicate of InfoNCE. A range close to the WITH-fix number confirms "
          "the fix restored a genuine, discriminative diversity signal.")


if __name__ == "__main__":
    main()
