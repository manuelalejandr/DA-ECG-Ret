# Diversity-aware ECG retrieval: task, hierarchical evaluation and benchmark

Code, metrics and results for the paper *Diversity-aware ECG retrieval for
differential diagnosis: task definition, hierarchical evaluation and a benchmark
of self-supervised encoders* (under review).

**Task.** Given a query ECG, retrieve a cohort that is (i) *relevant* (shares the
query's coarse diagnostic family) and (ii) *diverse* (covers the distinct
fine-grained diagnoses inside that family). To our knowledge no benchmark existed
for this. This repository provides the task definition, validated metrics, a
leakage-aware protocol and baseline results for existing methods that were *not*
designed for it (self-supervised encoders + MMR / DPP re-ranking), so that other
methods can be compared on the same footing.

## What is here

| Path | Content |
|---|---|
| `retrieval/benchmark_eval.py` | The benchmark metrics (numpy only): P@K, Pfine@K, RelDiv@K, α-nDCG@K, legacy diversity; NN / MMR / DPP / random rankers; query and same-patient exclusion. |
| `ptbxl_benchmark.py` | Resumable benchmark runner (stage 1: train/reuse encoders and save embeddings; stage 2: evaluate). |
| `analyze_benchmark.py` | Tables, paired tests, patient-level bootstrap CIs and figures from the runner's output. |
| `ssl_methods/`, `losses/`, `models/` | Encoders: SimCLR, TS2Vec, TF-C, SoftCLT (controlled re-implementations), DA-ECG-Ret, supervised references. |
| `data/` | PTB-XL (with SCP hierarchy), UCR/UEA, MIT-BIH loaders. |
| `run_all.py` | Trains every method on a dataset (produces the seed-0 checkpoints). |
| `check_ptbxl.py` | Detects truncated PTB-XL files. |
| `paper_assets/` | Scripts that generate the paper's tables and numbers from the result CSVs. |
| `results/` | Raw result tables used in the paper (`ptbxl/`, `secondary/`) and the analysis outputs. |
| `docs/` | Legacy README and development log of the earlier method-centred version. |

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.10. GPU or Apple-silicon (MPS) recommended; CPU works but is slow.

## Data (all open access)

```bash
# PTB-XL 1.0.3 (about 3 GB)
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P ./ptbxl_data
python check_ptbxl.py --data_root ./ptbxl_data/physionet.org/files/ptb-xl/1.0.3   # detects truncated files
```

UCR/UEA datasets are downloaded by `aeon`; MIT-BIH is read through `wfdb`.
No data are redistributed here.

## Reproduce the PTB-XL benchmark

```bash
DATA=./ptbxl_data/physionet.org/files/ptb-xl/1.0.3

# 1) three training seeds for the self-supervised methods
python ptbxl_benchmark.py --datasets ptbxl --seeds 0 1 2 \
  --methods simclr ts2vec tfc softclt da_ecg_ret \
  --data_root $DATA --out ./bench_outputs

# 2) supervised reference encoders (one seed)
python ptbxl_benchmark.py --datasets ptbxl --seeds 0 \
  --methods supervised_fine supervised_super \
  --data_root $DATA --out ./bench_outputs

# 3) tables, paired tests, bootstrap CIs, figures
python analyze_benchmark.py --out ./bench_outputs --datasets ptbxl
```

Each (dataset, method, seed) is a separate file, so an interrupted run resumes by
repeating the same command. Metrics can be recomputed from the saved embeddings
without retraining (`--skip_embed --force_eval`). To reuse checkpoints trained by
`run_all.py`, add `--reuse_checkpoints <run_all output dir>` (used for seed 0 in
the paper). Protocol: official PTB-XL folds 1–8 train / fold 10 test, lead II,
100 Hz, 100 epochs, shortlist N = 200, K = 10 (also 5 and 20).

## Secondary datasets (flat labels: P@K and AUROC only)

```bash
python ptbxl_benchmark.py --seeds 0 --methods simclr ts2vec tfc softclt da_ecg_ret \
  --datasets ucr:ECG200 ucr:ECG5000 ucr:ECGFiveDays ucr:TwoLeadECG \
             ucr:CinCECGTorso ucr:NonInvasiveFetalECGThorax1 ucr:NonInvasiveFetalECGThorax2 \
  --data_root ./ptbxl_data --out ./bench_secondary
python ptbxl_benchmark.py --seeds 0 --methods simclr ts2vec tfc softclt da_ecg_ret \
  --datasets mitbih --max_queries 5000 --data_root ./ptbxl_data --out ./bench_secondary
```

## Regenerate the paper's tables

```bash
cd paper_assets
python make_assets.py --bench ../results/ptbxl      # tab_main, tab_rerank, tab_coupling, numbers.tex, figure
python make_secondary.py                            # tab_secondary.tex (reads secondary_raw.csv)
```

## Adding your own method

Register a new encoder in `ssl_methods/registry.py` (it must expose an
embedding model), then run `ptbxl_benchmark.py --methods <name>`. Because
embeddings are saved per (dataset, method, seed), a new method is evaluated with
exactly the same queries, shortlists and metrics as the baselines.

## Notes and limitations

- Encoders are controlled re-implementations sharing one backbone, not ports of
  the authors' code; absolute numbers may change with tuning.
- The label hierarchy is PTB-XL's (superclass > subclass > SCP statement).
  Its use as a proxy for clinical need has not been validated with clinicians.
- Results at the secondary datasets use one training seed.

## License and citation

MIT (see `LICENSE`). Please cite the paper (see `CITATION.cff`).
