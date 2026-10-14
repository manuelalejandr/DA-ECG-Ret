"""
Diversity-aware ECG retrieval BENCHMARK (task + rigorous evaluation).

Two resumable stages (every (dataset, method, seed) is a separate file, so a
crash or Ctrl-C never loses finished work -- just re-run the same command):

  EMBED     train each SSL method (or reuse an existing checkpoint) and save
            train/test embeddings + labels at every available granularity.
            Expensive (training). Output: <out>/embeds/<tag>/<method>_seed<s>.npz
  EVALUATE  compute the diversity-aware retrieval metrics (retrieval/benchmark_eval.py)
            from the saved embeddings. Cheap and re-runnable: if a metric is
            changed later, only this stage is repeated -- NO retraining.
            Output: <out>/metrics/<tag>/... and <out>/benchmark_raw.csv

Typical use (PTB-XL, the main clinical dataset):

  # (A) FREE first look: reuse the PTB-XL checkpoints from the earlier run_all.py
  #     as seed 0 (no training at all), evaluate with the new metrics:
  python ptbxl_benchmark.py --datasets ptbxl --seeds 0 \\
      --data_root ./ptbxl_data/physionet.org/files/ptb-xl/1.0.3 \\
      --reuse_checkpoints ./outputs --out ./bench_outputs

  # (B) add seeds (this is the expensive part: one training run per method per seed):
  python ptbxl_benchmark.py --datasets ptbxl --seeds 0 1 2 3 4 \\
      --data_root ./ptbxl_data/physionet.org/files/ptb-xl/1.0.3 \\
      --reuse_checkpoints ./outputs --out ./bench_outputs

Single-label datasets (UCR, MIT-BIH) have no label hierarchy, so only
P@K / AUROC / the legacy metric are produced for them (they serve as
representation-quality sanity checks, not as diversity benchmarks).
"""
import argparse
import glob
import json
import os
import time
import zipfile

import numpy as np
import pandas as pd

from config import Config, SSL_METHODS
from experiment_utils import parse_dataset_spec, dataset_tag, make_args_namespace


def run_tag(spec_tag, args):
    """Directory tag for embeddings/metrics; a non-empty --run_tag keeps variants
    (e.g. 12 leads, 500 Hz) from overwriting the default lead-II 100 Hz results."""
    return spec_tag + (args.run_tag or "")


def _key_to_fn(k):
    return k.replace("|", "__")


def _fn_to_key(k):
    return k.replace("__", "|")


# ----------------------------------------------------------------------------
# EMBED
# ----------------------------------------------------------------------------
def stage_embed(args):
    import torch
    from torch.utils.data import DataLoader
    from device_utils import resolve_device
    from data.augmentations import default_ecg_augmentation
    from data.common import ContrastiveWrapper, get_dataset_splits, effective_batch_size
    from retrieval.metrics import extract_embeddings
    from train import set_seed, train_modules_in_memory
    from evaluate import load_embedding_model

    device = resolve_device(verbose=True)
    for spec in args.datasets:
        kind, name = parse_dataset_spec(spec)
        tag = run_tag(dataset_tag(kind, name), args)
        ns = make_args_namespace(kind, name, args.data_root)
        cfg = Config()
        cfg.dataset, cfg.ucr_name, cfg.data_root = kind, ns.ucr_name, args.data_root
        if kind == "ptbxl":
            cfg.lead_index, cfg.sampling_rate, cfg.signal_length = args.lead_index, args.sampling_rate, args.signal_length
            cfg.in_channels = 12 if args.lead_index < 0 else 1
            print(f"[ptbxl] lead_index={cfg.lead_index} (in_channels={cfg.in_channels}), "
                  f"{cfg.sampling_rate} Hz, length {cfg.signal_length}")
        cfg.encoder_name, cfg.epochs, cfg.batch_size = args.encoder, args.epochs, args.batch_size

        print(f"\n{'=' * 70}\nEMBED  dataset={tag}\n{'=' * 70}")
        train_base, test_base, n_classes, _ = get_dataset_splits(ns, cfg)
        hier = getattr(train_base, "hier_matrices", None)
        if hier is not None:
            cfg.n_label_code, cfg.n_label_super = len(hier["code"]), len(hier["super"])
        train_ds = ContrastiveWrapper(train_base, default_ecg_augmentation(cfg.sampling_rate))
        bs = effective_batch_size(cfg.batch_size, len(train_ds))
        cfg.batch_size = bs

        emb_dir = os.path.join(args.out, "embeds", tag)
        os.makedirs(emb_dir, exist_ok=True)

        for seed in args.seeds:
            for method in args.methods:
                path = os.path.join(emb_dir, f"{method}_seed{seed}.npz")
                if os.path.exists(path):
                    print(f"[skip] {path} exists")
                    continue
                t0 = time.time()
                model = None
                if seed == 0 and args.reuse_checkpoints and not args.run_tag:
                    ckpt = os.path.join(args.reuse_checkpoints, tag, "checkpoints", f"encoder_{method}.pt")
                    if os.path.exists(ckpt):
                        print(f"[reuse] {ckpt}")
                        model = load_embedding_model(ckpt, method, cfg, device)
                if model is None:
                    mseed = cfg.seed + sum(ord(c) for c in method) + seed * 1000
                    set_seed(mseed)
                    loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                        num_workers=cfg.num_workers, drop_last=True)
                    modules, hist = train_modules_in_memory(
                        method, cfg, loader, device, verbose_prefix=f"{tag}/{method}/seed{seed}")
                    model = modules["embedding_model"]
                tr = extract_embeddings(model, DataLoader(train_base, batch_size=64, shuffle=False), device)
                te = extract_embeddings(model, DataLoader(test_base, batch_size=64, shuffle=False), device)
                payload = dict(tr_emb=tr["embeddings"], te_emb=te["embeddings"],
                               tr_lab=tr["labels"], te_lab=te["labels"], te_patients=te["patients"])
                if "labels_sub" in te:
                    payload.update(tr_lab_sub=tr["labels_sub"], te_lab_sub=te["labels_sub"])
                if "labels_code" in te:
                    payload.update(tr_lab_code=tr["labels_code"], te_lab_code=te["labels_code"])
                hier = getattr(test_base, "hier_matrices", None)
                if hier is not None:
                    payload.update(sub2super=hier["sub2super"], code2super=hier["code2super"],
                                   code2sub=hier["code2sub"],
                                   vocab_sub=np.array(hier["sub"]), vocab_code=np.array(hier["code"]),
                                   vocab_super=np.array(hier["super"]))
                np.savez_compressed(path, **payload)
                print(f"[saved] {path}  ({(time.time() - t0) / 60:.1f} min)")


# ----------------------------------------------------------------------------
# EVALUATE
# ----------------------------------------------------------------------------
def stage_evaluate(args):
    from retrieval.benchmark_eval import evaluate_queries
    from retrieval.metrics import linear_probe_auroc

    for spec in args.datasets:
        kind, name = parse_dataset_spec(spec)
        tag = run_tag(dataset_tag(kind, name), args)
        met_dir = os.path.join(args.out, "metrics", tag)
        os.makedirs(met_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(args.out, "embeds", tag, "*_seed*.npz")))
        print(f"\nEVALUATE dataset={tag}: {len(files)} embedding files")
        for f in files:
            base = os.path.basename(f)[:-4]
            out_npz = os.path.join(met_dir, base + ".npz")
            out_json = os.path.join(met_dir, base + ".json")
            if os.path.exists(out_npz) and os.path.exists(out_json) and not args.force_eval:
                continue
            t0 = time.time()
            z = np.load(f, allow_pickle=False)
            fine = {}
            if "te_lab_sub" in z:
                fine["sub"] = (z["te_lab_sub"], z["sub2super"])
            if "te_lab_code" in z:
                fine["code"] = (z["te_lab_code"], z["code2super"])
            res = evaluate_queries(z["te_emb"], z["te_patients"], z["te_lab"], fine,
                                   ks=tuple(args.ks), max_queries=args.max_queries)
            np.savez_compressed(out_npz, **{_key_to_fn(k): v for k, v in res.items()})

            au = {"AUROC_super": linear_probe_auroc(z["tr_emb"], z["tr_lab"], z["te_emb"], z["te_lab"])["macro"]}
            if "tr_lab_sub" in z:
                au["AUROC_sub"] = linear_probe_auroc(z["tr_emb"], z["tr_lab_sub"], z["te_emb"], z["te_lab_sub"])["macro"]
            if "tr_lab_code" in z:
                au["AUROC_code"] = linear_probe_auroc(z["tr_emb"], z["tr_lab_code"], z["te_emb"], z["te_lab_code"])["macro"]
            json.dump(au, open(out_json, "w"))
            print(f"  [{base}] evaluated in {time.time() - t0:.0f}s  AUROC={au}")

    # rebuild the flat CSV from everything on disk (idempotent)
    rows = []
    for spec in args.datasets:
        kind, name = parse_dataset_spec(spec)
        tag = run_tag(dataset_tag(kind, name), args)
        for npz in sorted(glob.glob(os.path.join(args.out, "metrics", tag, "*_seed*.npz"))):
            base = os.path.basename(npz)[:-4]
            method, seed = base.rsplit("_seed", 1)
            z = np.load(npz)
            for k in z.files:
                if k in ("query_idx", "patient"):
                    continue
                ranker, K, metric = _fn_to_key(k).split("|")
                v = z[k]
                rows.append(dict(dataset=tag, method=method, seed=int(seed), ranker=ranker,
                                 K=int(K), metric=metric, value=float(np.nanmean(v)),
                                 n_valid=int(np.isfinite(v).sum())))
            for m, val in json.load(open(npz[:-4] + ".json")).items():
                rows.append(dict(dataset=tag, method=method, seed=int(seed), ranker="", K=0,
                                 metric=m, value=float(val), n_valid=0))
    df = pd.DataFrame(rows)
    csv = os.path.join(args.out, "benchmark_raw.csv")
    df.to_csv(csv, index=False)
    print(f"\nSaved {csv} ({len(df)} rows)")


def zip_results(out_dir):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    zpath = os.path.join(os.path.dirname(os.path.normpath(out_dir)) or ".",
                         f"{os.path.basename(os.path.normpath(out_dir))}_results_{stamp}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(out_dir):
            if "embeds" in dirs:           # embeddings stay on disk (big); metrics are what we need
                dirs.remove("embeds")
            for fn in files:
                p = os.path.join(root, fn)
                zf.write(p, os.path.relpath(p, os.path.dirname(os.path.normpath(out_dir))))
    return zpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ptbxl"])
    ap.add_argument("--data_root", type=str, default="./ptbxl_data")
    ap.add_argument("--out", type=str, default="./bench_outputs")
    ap.add_argument("--methods", nargs="+", default=SSL_METHODS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--encoder", choices=["transformer", "resnet18", "fcn"], default="transformer")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--max_queries", type=int, default=None,
                    help="subsample queries (use for big datasets such as MIT-BIH); default: all")
    ap.add_argument("--reuse_checkpoints", type=str, default=None,
                    help="run_all.py output dir whose <tag>/checkpoints/encoder_<method>.pt are "
                         "reused as seed 0 (skips that training run)")
    ap.add_argument("--lead_index", type=int, default=1,
                    help="PTB-XL lead: 1 = lead II (default), -1 = all 12 leads")
    ap.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])
    ap.add_argument("--signal_length", type=int, default=1000, help="samples per record (1000 at 100 Hz, 5000 at 500 Hz)")
    ap.add_argument("--run_tag", type=str, default="",
                    help="suffix for output dirs of a variant, e.g. _12lead or _500hz (no checkpoint reuse)")
    ap.add_argument("--skip_embed", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--force_eval", action="store_true", help="recompute metrics even if present")
    ap.add_argument("--no_zip", action="store_true")
    args = ap.parse_args()

    if not args.skip_embed:
        stage_embed(args)
    if not args.skip_eval:
        stage_evaluate(args)
    if not args.no_zip:
        print(f"\nPackaged: {zip_results(args.out)}  (embeddings excluded; send this zip back)")


if __name__ == "__main__":
    main()
