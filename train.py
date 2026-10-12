"""
Training entry point — generic over any method in ssl_methods/registry.py
and any dataset in data/common.py::get_dataset_splits.

Usage:
  python train.py --method da_ecg_ret --dataset synthetic --epochs 20
  python train.py --method ts2vec     --dataset ucr --ucr_name ECG200
  python train.py --method tfc        --dataset ucr --ucr_name TwoLeadECG
  python train.py --method softclt    --dataset mitbih
  python train.py --method simclr     --dataset ptbxl --data_root ./ptbxl_data
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config, SSL_METHODS
from device_utils import resolve_device
from data.augmentations import default_ecg_augmentation
from data.common import ContrastiveWrapper, get_dataset_splits, effective_batch_size
from ssl_methods.registry import get_method


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def build_dataloaders(args, cfg):
    train_base, _, n_classes, signal_length = get_dataset_splits(args, cfg)
    augment = default_ecg_augmentation(cfg.sampling_rate)
    train_ds = ContrastiveWrapper(train_base, augment)

    bs = effective_batch_size(cfg.batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                               num_workers=cfg.num_workers, drop_last=True)
    return train_loader, n_classes, signal_length


def train_modules_in_memory(method_name: str, cfg, train_loader, device, verbose_prefix=None):
    """Runs the training loop and returns (modules, history) WITHOUT touching
    disk. Factored out of train_one_method so other scripts (e.g.
    sweep_hparams.py) can train many configurations back-to-back in one
    process without checkpoint-path collisions or repeated dataset loading."""
    method = get_method(method_name)
    modules = method.build_modules(cfg, device)

    seen_ids = set()
    trainable = []
    for m in modules.values():
        if not hasattr(m, "parameters"):
            continue
        for p in m.parameters():
            if id(p) not in seen_ids:
                seen_ids.add(id(p))
                trainable.append(p)
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    tag = verbose_prefix or method_name
    history = []
    for epoch in range(cfg.epochs):
        for m in modules.values():
            if hasattr(m, "train"):
                m.train()
        t0 = time.time()
        agg_logs = {}
        n_batches = 0

        for batch in train_loader:
            loss, logs = method.compute_loss(batch, modules, cfg, device,
                                              epoch=epoch, total_epochs=cfg.epochs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in logs.items():
                agg_logs[k] = agg_logs.get(k, 0.0) + v
            n_batches += 1

        sched.step()
        n_batches = max(1, n_batches)
        agg_logs = {k: v / n_batches for k, v in agg_logs.items()}
        agg_logs["epoch"] = epoch
        agg_logs["time_sec"] = time.time() - t0
        history.append(agg_logs)
        total_str = agg_logs.get("total", float("nan"))
        print(f"[{tag}] epoch {epoch+1}/{cfg.epochs} total={total_str:.4f} "
              f"({agg_logs['time_sec']:.1f}s)")

    for m in modules.values():
        if hasattr(m, "eval"):
            m.eval()
    return modules, history


def train_one_method(args, cfg, method_name: str):
    device = resolve_device(verbose=True)
    # Different methods get different (but still reproducible) seeds. This
    # matters most for clocs_lite, whose training loop is currently
    # identical to simclr's (see ssl_methods/clocs_lite.py docstring) --
    # with a SHARED seed the two runs were bit-for-bit identical in the
    # first full run, silently reducing the "5 baselines" to 4 distinct
    # ones. Offsetting the seed per method name keeps everything
    # reproducible while giving each method its own random initialization.
    method_seed = cfg.seed + sum(ord(c) for c in method_name)
    set_seed(method_seed)

    train_loader, n_classes, signal_length = build_dataloaders(args, cfg)
    modules, history = train_modules_in_memory(method_name, cfg, train_loader, device)

    dataset_tag = args.dataset if args.dataset != "ucr" else f"ucr_{cfg.ucr_name}"
    out_dir = os.path.join(cfg.output_dir, dataset_tag, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"encoder_{method_name}.pt")
    torch.save({
        "embedding_model": modules["embedding_model"].state_dict(),
        "method": method_name,
        "cfg": cfg.__dict__,
        "n_classes": n_classes,
        "signal_length": signal_length,
    }, ckpt_path)

    hist_dir = os.path.join(cfg.output_dir, dataset_tag, "history")
    os.makedirs(hist_dir, exist_ok=True)
    with open(os.path.join(hist_dir, f"history_{method_name}.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=SSL_METHODS, required=True)
    parser.add_argument("--dataset", choices=["synthetic", "ptbxl", "ucr", "mitbih"], default="synthetic")
    parser.add_argument("--ucr_name", type=str, default="ECG200")
    parser.add_argument("--data_root", type=str, default="./ptbxl_data")
    parser.add_argument("--encoder", choices=["transformer", "resnet18", "fcn"], default="transformer")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--exclude_self_from_redundancy", type=int, choices=[0, 1], default=None,
                         help="1 (default/fixed) or 0 (original buggy behaviour, for comparison only)")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--n_synthetic_train", type=int, default=1600)
    parser.add_argument("--n_synthetic_val", type=int, default=400)
    args = parser.parse_args()

    cfg = Config()
    cfg.dataset = args.dataset
    cfg.ucr_name = args.ucr_name
    cfg.data_root = args.data_root
    cfg.encoder_name = args.encoder
    cfg.output_dir = args.output_dir
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.beta is not None: cfg.beta = args.beta
    if args.lam is not None: cfg.lam = args.lam
    if args.alpha is not None: cfg.alpha = args.alpha
    if args.exclude_self_from_redundancy is not None:
        cfg.exclude_self_from_redundancy = bool(args.exclude_self_from_redundancy)

    train_one_method(args, cfg, args.method)


if __name__ == "__main__":
    main()
