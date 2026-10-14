"""
Extended evaluation of the SAVED PTB-XL embeddings (no training; numpy only).

Reads  <bench_dir>/embeds/ptbxl/<method>_seed<s>.npz  (written by ptbxl_benchmark.py)
and writes, for one evaluation setting, to <out_dir>/<setting>/:
    <method>_seed<s>.npz   per-query metric arrays (for bootstrap / stratification)
    benchmark_v2.csv       flat table with strata  (dataset, method, seed, ranker, K,
                           metric, stratum, value, n)

A setting is (corpus, exclusion, n_shortlist):
    corpus    test   queries = corpus = fold 10   (original setting)
              train  queries = fold 10, corpus = folds 1-8 (the "database" of past ECGs)
    exclusion patient | self | none  (leakage ablation; only meaningful for corpus=test)

Examples (see README):
    python ptbxl_eval_v2.py --bench_dir ./bench_outputs --out ./eval_v2 --corpus train
    python ptbxl_eval_v2.py --bench_dir ./bench_outputs --out ./eval_v2 --corpus test --exclusion self
    python ptbxl_eval_v2.py --bench_dir ./bench_outputs --out ./eval_v2 --corpus test --exclusion none
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from retrieval.benchmark_v2 import evaluate_v2, default_rerankers

STRATA_SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]


def _key_to_fn(k):
    return k.replace("|", "__")


def _fn_to_key(k):
    return k.replace("__", "|")


def evaluate_file(path, args):
    z = np.load(path, allow_pickle=False)
    fine_q, fine_c = {}, {}
    if "te_lab_sub" in z:
        fine_q["sub"] = (z["te_lab_sub"], z["sub2super"])
        fine_c["sub"] = z["tr_lab_sub"] if args.corpus == "train" else z["te_lab_sub"]
    if "te_lab_code" in z:
        fine_q["code"] = (z["te_lab_code"], z["code2super"])
        fine_c["code"] = z["tr_lab_code"] if args.corpus == "train" else z["te_lab_code"]
    same = args.corpus == "test"
    kw = dict(q_emb=z["te_emb"], q_pat=z["te_patients"], q_coarse=z["te_lab"], q_fine=fine_q,
              same_set=same, exclusion=args.exclusion, ks=tuple(args.ks),
              n_shortlist=args.n_shortlist, max_queries=args.max_queries,
              rerankers=default_rerankers(list(fine_q)) if args.rerankers is None else args.rerankers)
    if not same:
        kw.update(c_emb=z["tr_emb"], c_pat=None, c_coarse=z["tr_lab"], c_fine=fine_c)
    return evaluate_v2(**kw), z


def aggregate(setting_dir, tag="ptbxl"):
    rows = []
    for npz in sorted(glob.glob(os.path.join(setting_dir, "*_seed*.npz"))):
        base = os.path.basename(npz)[:-4]
        method, seed = base.rsplit("_seed", 1)
        d = np.load(npz, allow_pickle=False)
        keys = {_fn_to_key(k): d[k] for k in d.files}
        coarse = d["_coarse"]
        masks = {"all": np.ones(len(coarse), dtype=bool)}
        for j, nm in enumerate(STRATA_SUPER[: coarse.shape[1]] if "_vocab_super" not in d.files
                               else [str(x) for x in d["_vocab_super"]]):
            masks[f"super_{nm}"] = coarse[:, j] > 0
        for lvl in ("sub", "code"):
            if f"struct|0|nsub_{lvl}" in keys:
                masks[f"sq2_{lvl}"] = keys[f"struct|0|nsub_{lvl}"] >= 2
        for k, v in keys.items():
            if k in ("query_idx", "patient") or k.startswith("_"):
                continue
            ranker, K, metric = k.split("|")
            lvl = "code" if metric.endswith("_code") else "sub"
            for sname, m in masks.items():
                if sname.startswith("sq2_") and not sname.endswith(lvl):
                    continue
                if ranker in ("struct",) and sname != "all":
                    continue
                vv = v[m] if len(v) == len(m) else v
                rows.append(dict(dataset=tag, method=method, seed=int(seed), ranker=ranker, K=int(K),
                                 metric=metric, stratum=sname, value=float(np.nanmean(vv)) if len(vv) else np.nan,
                                 n=int(np.isfinite(vv).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(setting_dir, "benchmark_v2.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_dir", required=True, help="dir containing embeds/ptbxl/")
    ap.add_argument("--out", default="./eval_v2")
    ap.add_argument("--corpus", choices=["test", "train"], default="train")
    ap.add_argument("--exclusion", choices=["patient", "self", "none"], default="patient")
    ap.add_argument("--n_shortlist", type=int, default=200)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--rerankers", nargs="+", default=None)
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--tag", default="ptbxl", help="embeds sub-dir (e.g. ptbxl_12lead)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    setting = f"{args.tag}__corpus-{args.corpus}_excl-{args.exclusion}_N{args.n_shortlist}"
    sdir = os.path.join(args.out, setting)
    os.makedirs(sdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.bench_dir, "embeds", args.tag, "*_seed*.npz")))
    if args.methods:
        files = [f for f in files if os.path.basename(f).rsplit("_seed", 1)[0] in args.methods]
    print(f"setting {setting}: {len(files)} embedding files")
    for f in files:
        base = os.path.basename(f)[:-4]
        out = os.path.join(sdir, base + ".npz")
        if os.path.exists(out) and not args.force:
            continue
        t0 = time.time()
        res, z = evaluate_file(f, args)
        payload = {_key_to_fn(k): v for k, v in res.items()}
        payload["_coarse"] = z["te_lab"][res["query_idx"]]
        if "vocab_super" in z.files:
            payload["_vocab_super"] = z["vocab_super"]
        np.savez_compressed(out, **payload)
        print(f"  [{base}] {time.time() - t0:.0f}s", flush=True)
    df = aggregate(sdir, args.tag)
    print(f"wrote {os.path.join(sdir, 'benchmark_v2.csv')} ({len(df)} rows)")


if __name__ == "__main__":
    main()
