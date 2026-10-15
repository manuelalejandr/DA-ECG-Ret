"""
Tune SimCLR on the VALIDATION fold (PTB-XL fold 9) and retrain the best configuration with several
seeds, so that the benchmark also contains a SimCLR that is not handicapped by untuned settings.

Selection criterion: macro AUROC of a linear probe (superclass labels) trained on folds 1-8
embeddings and evaluated on fold 9. Retrieval metrics and fold 10 are NOT used for selection.

Stage 1, grid (one seed, shortened schedule):
    python tune_simclr.py grid  --leads II --data_root $DATA --out ./tune_out
    python tune_simclr.py grid  --leads 12 --data_root $DATA --out ./tune_out
Stage 2, retrain the best configuration (full schedule) and save embeddings in the benchmark format:
    python tune_simclr.py final --leads II --data_root $DATA --out ./tune_out --bench_out ./bench_outputs --tag ptbxl --seeds 0 1 2
    python tune_simclr.py final --leads 12 --data_root $DATA --out ./tune_out --bench_out ./bench_outputs_12lead --tag ptbxl_12lead --seeds 0 1 2
Summary:
    python tune_simclr.py show --out ./tune_out

Grids. Lead II: tau {0.1,0.2,0.5} x batch {128,256} x augmentation {default,strong} (12 runs).
12 leads: tau {0.1,0.2,0.5} x augmentation {default,strong,strong_indep,strong_leaddrop}, batch 128 (12 runs).
Each run is resumable: finished configurations are read from <out>/grid_<leads>.json.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np


def grid_for(leads):
    if leads == "II":
        return [dict(tau=t, batch=b, aug=a) for t, b, a in itertools.product((0.1, 0.2, 0.5), (128, 256), ("default", "strong"))]
    return [dict(tau=t, batch=128, aug=a) for t, a in itertools.product((0.1, 0.2, 0.5), ("default", "strong", "strong_indep", "strong_leaddrop"))]


def key(c):
    return f"tau{c['tau']}_b{c['batch']}_{c['aug']}"


def make_loader_ds(base, aug_name, sr):
    from data.aug_extra import LeadDropWrapper, PerLead, strong_ecg_augmentation
    from data.augmentations import default_ecg_augmentation
    from data.common import ContrastiveWrapper
    if aug_name == "default":
        return ContrastiveWrapper(base, default_ecg_augmentation(sr))
    if aug_name == "strong":
        return ContrastiveWrapper(base, strong_ecg_augmentation(sr))
    if aug_name == "strong_indep":
        return ContrastiveWrapper(base, PerLead(strong_ecg_augmentation(sr)))
    if aug_name == "strong_leaddrop":
        return LeadDropWrapper(base, PerLead(strong_ecg_augmentation(sr)), p=0.5, min_keep=3)
    raise ValueError(aug_name)


class _Fake:
    """Random stand-in dataset for --smoke (checks the plumbing, no PTB-XL needed)."""
    def __init__(self, n, C, T, seed):
        r = np.random.RandomState(seed)
        self.x = r.randn(n, C, T).astype(np.float32)
        self.y = (r.rand(n, 5) < 0.3).astype(np.float32); self.y[:, 0] = np.maximum(self.y[:, 0], 1 - self.y.max(1))
        self.ys = (r.rand(n, 9) < 0.2).astype(np.float32); self.yc = (r.rand(n, 14) < 0.15).astype(np.float32)
        self.p = np.arange(n)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        import torch
        return dict(signal=torch.from_numpy(self.x[i]), label=torch.from_numpy(self.y[i]), label_sub=torch.from_numpy(self.ys[i]),
                    label_code=torch.from_numpy(self.yc[i]), patient_id=int(self.p[i]), ecg_id=int(i))


def build_data(args, cfg):
    if args.smoke:
        T = cfg.signal_length
        return _Fake(256, cfg.in_channels, T, 0), _Fake(64, cfg.in_channels, T, 1), _Fake(64, cfg.in_channels, T, 2), None
    from data.ptbxl_dataset import PTBXLUnivariate, get_ptbxl_hierarchy, load_ptbxl_metadata, patient_wise_split
    meta = load_ptbxl_metadata(args.data_root)
    hier = get_ptbxl_hierarchy(meta, args.data_root)
    tr, va, te = patient_wise_split(meta)
    mk = lambda m: PTBXLUnivariate(args.data_root, m, cfg.sampling_rate, cfg.lead_index, cfg.signal_length, hier=hier)
    tb, vb, eb = mk(tr), mk(va), mk(te)
    tb.hier_matrices = vb.hier_matrices = eb.hier_matrices = hier
    return tb, vb, eb, hier


def base_cfg(args):
    from config import Config
    cfg = Config()
    cfg.dataset, cfg.data_root = "ptbxl", args.data_root
    cfg.lead_index = 1 if args.leads == "II" else -1
    cfg.in_channels = 1 if args.leads == "II" else 12
    cfg.sampling_rate, cfg.signal_length = 100, (64 if args.smoke else 1000)
    cfg.encoder_name = "transformer"
    return cfg


def train_simclr(args, cfg, train_base, c, epochs, seed):
    import torch
    from torch.utils.data import DataLoader
    from device_utils import resolve_device
    from train import set_seed, train_modules_in_memory
    device = resolve_device(verbose=False)
    cfg.tau, cfg.epochs, cfg.batch_size = c["tau"], epochs, c["batch"]
    ds = make_loader_ds(train_base, c["aug"], cfg.sampling_rate)
    set_seed(cfg.seed + sum(ord(ch) for ch in "simclr_tuned") + seed * 1000)
    loader = DataLoader(ds, batch_size=c["batch"], shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    modules, hist = train_modules_in_memory("simclr", cfg, loader, device, verbose_prefix=f"{key(c)}/s{seed}")
    return modules["embedding_model"], hist, device


def stage_grid(args):
    from torch.utils.data import DataLoader
    from retrieval.metrics import extract_embeddings, linear_probe_auroc
    cfg = base_cfg(args)
    tb, vb, _, _ = build_data(args, cfg)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"grid_{args.leads}.json")
    done = json.load(open(path)) if os.path.exists(path) else []
    done_keys = {r["key"] for r in done}
    todo = [c for c in grid_for(args.leads) if key(c) not in done_keys]
    if args.max_runs:
        todo = todo[: args.max_runs]
    print(f"{len(done)} runs done, {len(todo)} to run (epochs={args.epochs})")
    for c in todo:
        t0 = time.time()
        model, hist, device = train_simclr(args, cfg, tb, c, args.epochs, seed=0)
        tr = extract_embeddings(model, DataLoader(tb, batch_size=64, shuffle=False), device)
        va = extract_embeddings(model, DataLoader(vb, batch_size=64, shuffle=False), device)
        r = dict(key=key(c), **c, epochs=args.epochs, final_loss=float(hist[-1]["total"]), minutes=(time.time() - t0) / 60,
                 auroc_super=float(linear_probe_auroc(tr["embeddings"], tr["labels"], va["embeddings"], va["labels"])["macro"]))
        if "labels_sub" in tr:
            r["auroc_sub"] = float(linear_probe_auroc(tr["embeddings"], tr["labels_sub"], va["embeddings"], va["labels_sub"])["macro"])
        if "labels_code" in tr:
            r["auroc_code"] = float(linear_probe_auroc(tr["embeddings"], tr["labels_code"], va["embeddings"], va["labels_code"])["macro"])
        done.append(r)
        json.dump(done, open(path, "w"), indent=1)
        print(f"[grid] {r['key']}: AUROC_super(fold 9)={r['auroc_super']:.4f} loss={r['final_loss']:.4f} ({r['minutes']:.1f} min)", flush=True)
    stage_show(args)


def stage_show(args):
    for leads in ("II", "12"):
        path = os.path.join(args.out, f"grid_{leads}.json")
        if not os.path.exists(path):
            continue
        rows = sorted(json.load(open(path)), key=lambda r: -r["auroc_super"])
        print(f"\nGrid, leads={leads} (validation fold 9, best first)")
        for r in rows:
            print(f"  {r['key']:34s} AUROC_super={r['auroc_super']:.4f}  AUROC_sub={r.get('auroc_sub', float('nan')):.4f}  "
                  f"AUROC_SCP={r.get('auroc_code', float('nan')):.4f}  loss={r['final_loss']:.4f}")


def stage_final(args):
    from torch.utils.data import DataLoader
    from retrieval.metrics import extract_embeddings
    cfg = base_cfg(args)
    rows = json.load(open(os.path.join(args.out, f"grid_{args.leads}.json")))
    best = max(rows, key=lambda r: r["auroc_super"])
    c = {k: best[k] for k in ("tau", "batch", "aug")}
    print(f"best configuration (fold 9 AUROC_super={best['auroc_super']:.4f}): {c}")
    tb, _, eb, hier = build_data(args, cfg)
    edir = os.path.join(args.bench_out, "embeds", args.tag)
    os.makedirs(edir, exist_ok=True)
    for s in args.seeds:
        path = os.path.join(edir, f"simclr_tuned_seed{s}.npz")
        if os.path.exists(path):
            print("[skip]", path)
            continue
        model, _, device = train_simclr(args, cfg, tb, c, args.final_epochs, seed=s)
        tr = extract_embeddings(model, DataLoader(tb, batch_size=64, shuffle=False), device)
        te = extract_embeddings(model, DataLoader(eb, batch_size=64, shuffle=False), device)
        payload = dict(tr_emb=tr["embeddings"], te_emb=te["embeddings"], tr_lab=tr["labels"], te_lab=te["labels"], te_patients=te["patients"])
        if "labels_sub" in te:
            payload.update(tr_lab_sub=tr["labels_sub"], te_lab_sub=te["labels_sub"])
        if "labels_code" in te:
            payload.update(tr_lab_code=tr["labels_code"], te_lab_code=te["labels_code"])
        if hier is not None:
            payload.update(sub2super=hier["sub2super"], code2super=hier["code2super"], code2sub=hier["code2sub"],
                           vocab_sub=np.array(hier["sub"]), vocab_code=np.array(hier["code"]), vocab_super=np.array(hier["super"]))
        np.savez_compressed(path, **payload)
        json.dump(dict(best_config=c, fold9_auroc_super=best["auroc_super"], epochs=args.final_epochs),
                  open(os.path.join(args.out, f"final_{args.leads}.json"), "w"), indent=1)
        print("[saved]", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["grid", "final", "show"])
    ap.add_argument("--leads", choices=["II", "12"], default="II")
    ap.add_argument("--data_root", default="")
    ap.add_argument("--out", default="./tune_out")
    ap.add_argument("--epochs", type=int, default=30, help="grid schedule (shortened)")
    ap.add_argument("--final_epochs", type=int, default=100)
    ap.add_argument("--max_runs", type=int, default=None)
    ap.add_argument("--bench_out", default="./bench_outputs")
    ap.add_argument("--tag", default="ptbxl")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true", help="random data, 1 epoch: checks that the code runs")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = args.final_epochs = 1
    {"grid": stage_grid, "final": stage_final, "show": stage_show}[args.stage](args)


if __name__ == "__main__":
    main()
