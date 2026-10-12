# DA-ECG-Ret: Diversity-Aware Self-Supervised ECG Retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Official implementation and full experimental pipeline for the paper
**"Diversity-Aware Self-Supervised ECG Retrieval"** (submitted to *Expert
Systems with Applications*).

Retrieval of clinically similar electrocardiograms underlies case-based
reasoning and cohort discovery in cardiology, but standard similarity-based
retrieval tends to return redundant, near-duplicate recordings instead of a
physiologically diverse cohort. **DA-ECG-Ret** is a self-supervised
framework that combines contrastive pre-training with a differentiable,
pseudo-cluster-gated relaxation of Maximal Marginal Relevance and a
physiologically-grounded auxiliary pretext task, benchmarked against five
contemporary self-supervised baselines on eight freely available,
univariate ECG datasets.

This repository contains everything needed to reproduce every number and
figure in the paper: data loading, training, evaluation, the diagnostic
hyperparameter sweep, the multi-seed confirmatory analysis, and the
component ablation study.

## Table of contents
- [Key results](#key-results)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Reproducing the paper's results](#reproducing-the-papers-results)
- [Datasets](#datasets)
- [Method summary](#method-summary)
- [Repository structure](#repository-structure)
- [Development log: bugs found, sweeps, and the v2 redesign](#development-log-bugs-found-sweeps-and-the-v2-redesign)
- [Citation](#citation)
- [License](#license)

## Key results

Averaged over five random seeds and eight datasets, relative to a
plain-contrastive (SimCLR) baseline sharing the exact same encoder:

| Metric | Datasets won (of 8) | Mean difference | Wilcoxon *p* |
|---|---|---|---|
| Downstream AUROC (linear probe) | 7/8 | +0.010 | 0.055 |
| Precision-diversity Pareto-frontier area | 7/8 | +0.009 | 0.109 |
| Precision@K | 5/8 | +0.008 | 0.195 |
| NormalizedDiversity@K (plain retrieval, no re-ranking) | 3/8 | −0.009 | 0.195 |

**Read this honestly, not optimistically**: with only eight independent
datasets, neither headline result reaches the conventional 0.05
significance threshold — both are directionally consistent and worth
reporting, but not proof. A controlled ablation further shows that the
**physiological auxiliary task** (Component B) drives the large majority
of this improvement; the pseudo-cluster-gated diversity loss (Component A),
although theoretically motivated, shows no additional measurable benefit
in the regime tested. See the [paper](#citation) and the
[development log](#development-log-bugs-found-sweeps-and-the-v2-redesign)
below for the full, transparent story of how we got here, including two
real implementation bugs we found and fixed along the way.

## Installation

Tested on Apple Silicon (MPS) and CPU; CUDA is auto-detected if available.

```bash
git clone https://github.com/YOUR-USERNAME/da-ecg-ret.git
cd da-ecg-ret
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

No manual dataset download is required: the seven UCR/UEA problems are
fetched automatically via [`aeon`](https://github.com/aeon-toolkit/aeon) on
first use, and MIT-BIH Arrhythmia beats are streamed directly from
[PhysioNet](https://physionet.org/) via `wfdb`. Nothing is redistributed in
this repository.

## Quickstart

A synthetic-data smoke test that exercises the full pipeline (all 6
methods → evaluation → figures → results zip) in a couple of minutes, no
download required:

```bash
python run_all.py --datasets synthetic --epochs 10 --output_dir ./outputs
```

## Reproducing the paper's results

```bash
# 1. Full 8-dataset, 6-method benchmark (Table 3 / Fig. 1 in the paper)
python run_all.py \
    --datasets ucr:ECG200 ucr:ECG5000 ucr:ECGFiveDays ucr:TwoLeadECG \
               ucr:CinCECGTorso ucr:NonInvasiveFetalECGThorax1 \
               ucr:NonInvasiveFetalECGThorax2 mitbih \
    --epochs 100 --batch_size 128 --output_dir ./outputs

# 2. Multi-seed confirmatory analysis (Table 5 / Fig. 2)
python multiseed_experiment.py --seeds 0 1 2 3 4 --epochs 100

# 3. Component ablation: Component A vs. B vs. A+B vs. phys-only-beta0
#    (Table 5 / Figs. 4-5) -- includes the v2.1 twin-exclusion and
#    tau_ret fixes (see "Development log" below) and the new
#    phys_only_beta0 variant, which isolates Component B under beta=0.
#    This has NOT been run since the v2.1 fixes; running it is the
#    immediate next step to check whether Component A behaves
#    differently once the twin/temperature bugs are gone.
python ablation_components.py --seeds 0 1 2 --epochs 100
```

Two things worth sweeping before trusting the v2.1 numbers: `tau_ret`
(`config.py`, default `0.1`, currently unswept) and whether re-fixing
Component A changes its ranking now that its gradient is no longer
muted by the twin/temperature bugs. A quick way to check both together:

```bash
python sweep_hparams.py --datasets ucr:ECG5000 ucr:TwoLeadECG --betas 0.5 1 2 5 10
# (extend sweep_hparams.py with a --tau_rets flag, or edit config.py's
# default directly, to sweep tau_ret specifically)
```

Each script packages its own results into a timestamped `.zip` automatically
(model checkpoints excluded by default to keep it small; pass
`--include_checkpoints_in_zip` to `run_all.py` if you want them). Two
diagnostic-only scripts are also included:

```bash
python diagnose_gradients.py --dataset ucr --ucr_name ECG5000   # is L_ret's gradient alive?
python sweep_hparams.py --datasets ucr:ECG5000 ucr:TwoLeadECG --betas 0.5 1 2 5 10
```

`notebooks/DA_ECG_Ret_Colab.ipynb` runs the same pipeline on Google Colab.

## Datasets

All datasets are freely available with no credentialing requirement.

| Dataset | Classes | Length | Source |
|---|---|---|---|
| ECG200 | 2 | 96 | UCR/UEA archive |
| ECG5000 | 5 | 140 | UCR/UEA archive |
| ECGFiveDays | 2 | 136 | UCR/UEA archive |
| TwoLeadECG | 2 | 82 | UCR/UEA archive (23 training series) |
| CinCECGTorso | 4 | 1639 | UCR/UEA archive |
| NonInvasiveFetalECGThorax1/2 | 42 | 750 | UCR/UEA archive |
| MIT-BIH Arrhythmia | 5 (AAMI) | 250 (per beat) | PhysioNet, streamed via `wfdb` |

## Method summary

DA-ECG-Ret trains a shared Transformer encoder (1-D conv tokenizer + 4
attention layers) with three objectives:

1. **InfoNCE** contrastive loss over augmented views (jitter, scaling,
   segment masking, crop-resize, baseline wander).
2. **Retrieval-aware loss** — a differentiable relaxation of Maximal
   Marginal Relevance, whose redundancy term is gated by an on-the-fly,
   parameter-free soft *k*-means over the mini-batch (**Component A**),
   so that legitimate same-phenotype clustering is not penalised as if it
   were redundancy.
3. **Physiological auxiliary task** (**Component B**) — a small MLP head
   regresses six classical, fully unsupervised waveform descriptors
   (dominant frequency, spectral entropy, sample entropy, autocorrelation
   periodicity, zero-crossing rate, kurtosis) from the encoder
   representation.

At inference, embeddings are re-ranked with classical MMR or a fast greedy
DPP-MAP re-ranker to produce the final retrieved cohort. Five baselines —
SimCLR, a simplified CLOCS, TS2Vec, TF-C, and SoftCLT — share the identical
encoder for a controlled comparison. Full mathematical detail is in the
paper; see [`ssl_methods/`](ssl_methods/) and [`losses/`](losses/) for the
corresponding code.

## Repository structure

```
da-ecg-ret/
├── config.py                     # all hyperparameters, dataset/method registries
├── train.py                      # generic trainer over ssl_methods/registry.py
├── evaluate.py                   # retrieval eval: NN/MMR/DPP + Pareto-frontier analysis
├── plots.py                      # per-dataset figures (300 dpi)
├── aggregate.py                  # cross-dataset average ranks + Wilcoxon tests
├── run_all.py                    # orchestrates train → evaluate → plots → zip
├── diagnose_gradients.py          # one-shot: is L_ret's gradient alive?
├── sweep_hparams.py               # coarse beta/lambda diagnostic search
├── multiseed_experiment.py        # confirmatory: SimCLR vs. DA-ECG-Ret, several seeds
├── ablation_components.py         # confirmatory: isolates Component A vs. B vs. A+B
├── experiment_utils.py            # shared evaluation helper
├── package_results.py             # standalone results-zipping utility
├── device_utils.py                # CUDA > MPS (Apple Silicon) > CPU auto-detection
├── data/                          # dataset loaders, augmentations, physiological features
├── models/                        # encoder, projection/regression heads, prototypes
├── losses/                        # InfoNCE, MMR/retrieval-aware loss, TS2Vec/TF-C/SoftCLT losses
├── ssl_methods/                   # per-method build/train registry (the 6 compared methods)
├── retrieval/                     # MMR/DPP re-ranking, metrics, Pareto-frontier analysis
└── notebooks/                     # Colab notebook
```

## Development log: bugs found, sweeps, and the v2 redesign

<details>
<summary><strong>Click to expand the full, honest development history</strong> (two real bugs found and fixed, a hyperparameter sweep that argued against the obvious fix, and the reasoning behind the Component A/B redesign)</summary>

### Bug fixes discovered during development

1. **Redundancy-term self-similarity saturation.** The retrieval-aware
   loss's smoothed redundancy term originally included each candidate's
   similarity to *itself* in its log-sum-exp. Since cosine self-similarity
   is always exactly 1 — the maximum possible value — it dominated the
   sum for every candidate regardless of true crowding, collapsing the
   term's dynamic range to near-zero (measured: ~0.0009 with the bug vs.
   ~0.40 once fixed, on the same toy batch). Fixed by excluding `k == j`
   from the sum (`losses/mmr_loss.py::smooth_redundancy`,
   `exclude_self_from_redundancy=True` by default).
2. **Missing temperature in the TS2Vec baseline.** Its contrastive softmax
   summed raw cosine similarities (bounded in [-1, 1]) with no temperature
   scaling, keeping the softmax near-uniform throughout training and
   producing a flat loss curve and near-chance downstream AUROC. Fixed by
   applying the same `tau` used by every other method
   (`losses/ts2vec_loss.py`).
3. **(Minor) SimCLR and CLOCS-lite were bit-for-bit identical** in an
   early run, since both reduce to the same InfoNCE loop and shared a
   fixed seed. Fixed by offsetting the seed per method name (`train.py`).

### The beta/lambda sweep argued against "just weight it more"

A first fix (excluding self-similarity, above) restored the redundancy
term's dynamic range, but a follow-up sweep over `beta` (the retrieval-aware
loss's weight) showed that simply increasing it did **not** help — and
actively hurt on the highest-class-cardinality dataset tested
(`NonInvasiveFetalECGThorax1`, 42 classes), where the Pareto-frontier gap
vs. SimCLR degraded monotonically from +0.0019 at `beta=0.5` to −0.0110 at
`beta=20`. The mechanism: the redundancy term penalises a candidate for
resembling *any* other point in the batch, with no notion of class — an
instance of the well-documented "false-negative problem" in contrastive
learning. Pushing harder on an unconditional anti-redundancy pressure
fights the class-discriminative clustering that gives good retrieval
precision, and the damage scales with how many classes a dataset has.

### The v2 redesign: Components A and B

Rather than continuing to tune a loss with the wrong mechanism, we
redesigned it:

- **Component A** (`models/prototypes.py`): a lightweight, parameter-free
  per-batch soft *k*-means (5 Lloyd iterations, no gradient tracking)
  produces a pseudo-cluster assignment used to *gate* the redundancy term
  — pairs likely sharing a pseudo-cluster contribute little to the
  penalty, while genuinely cross-cluster pairs are penalised at full
  strength. A simplified, small-batch-friendly stand-in for SwAV/PCL's
  persistently learned prototypes.
- **Component B** (`data/physiological_features.py`,
  `models/heads.py::PhysioRegressionHead`): a fully unsupervised auxiliary
  regression task predicting six classical waveform descriptors from the
  encoder representation, in the spirit of recent ECG-specific
  self-supervised work that injects domain knowledge as an auxiliary
  pretext task.

A re-run of the sweep with both components active showed the redesign
working as intended — the previously-degrading `NonInvasiveFetalECGThorax1`
result flipped from negative to positive at most tested `beta` values, and
`TwoLeadECG`'s gap roughly doubled at its peak. `config.py`'s default
`beta` was updated from 0.5 to **1.0** based on this evidence.

### The ablation: which component actually did the work?

A controlled ablation (`ablation_components.py`, 4 variants × 3 seeds × 8
datasets, all compared against a shared SimCLR reference) revealed that
**Component B is responsible for the large majority of the measured
improvement**: the `B_only` and `A_and_B` (full) variants are statistically
indistinguishable from each other on both AUROC and Pareto-frontier area,
while `A_only` and the bug-fixed-but-otherwise-plain `base` variant trail
close behind SimCLR. Component A, despite being carefully implemented and
verified to behave correctly at the mechanism level (a toy two-cluster test
confirmed the intended within-cluster suppression / cross-cluster full
penalty behaviour), shows no additional measurable benefit in this data
regime — plausibly because several datasets' training splits are too
small for a per-batch clustering step to form stable pseudo-clusters. We
report this as an honest negative result rather than omitting it.

We consider a persistent-prototype variant of Component A (maintained
across batches via a memory bank, rather than re-clustered from scratch
each step) the most promising next step if a stronger diversity-specific
mechanism is needed.

### v2.1: three more bugs found in an external review (Sept 2026)

A co-author's critical review of the manuscript caught three additional
problems in `L_ret` that the ablation above could not distinguish from a
genuine "Component A doesn't help" result, because both explanations
(bug vs. no real effect) predict the same symptom. All three are now
fixed; a fourth item (below) is a missing experimental condition, also
added.

1. **Twin saturation.** `smooth_redundancy`'s `exclude_self` only masked
   the literal diagonal (`k == j`). But the candidate pool is the
   concatenation of both augmented views, so every candidate has an
   "augmented twin" at a *different* pool index that InfoNCE pulls
   towards similarity ≈1 by design. Left unexcluded, this twin dominated
   the log-sum-exp almost as much as the self-term used to, re-saturating
   `d_j` in an attenuated form — the same bug as item 1 above, reappearing
   one level down. Fixed via `twin_index` in `smooth_redundancy` /
   `retrieval_aware_loss` / `combined_loss` (`losses/mmr_loss.py`). A
   quick numeric check (`alpha=10`, near-duplicate synthetic twins):
   redundancy `d` had mean 0.989 / std 0.008 before the fix (saturated,
   almost no information) vs. mean 0.659 / std 0.081 after (genuine
   spread). This also directly answers a related concern the reviewer
   raised — "is the positive penalised for resembling the query?" — since
   the positive candidate in `L_ret` *is* the query's twin by
   construction; excluding the twin removes exactly that penalty.
2. **No temperature in the retrieval softmax (Eq. 5 / Eq. 10 in the
   paper).** `s = lam*r - (1-lam)*d` went straight into a softmax with no
   division by any `tau`, unlike InfoNCE (`tau=0.07`). With cosine-scale
   scores and a pool of a few hundred candidates, this made `p_j` close to
   uniform: a numeric check gave mean entropy 4.14 nats vs. 4.16 nats for
   a uniform distribution over 64 candidates — the loss was carrying
   almost no gradient. Fixed via a new `tau_ret` parameter
   (`config.py::Config.tau_ret`, default **0.1, unswept** — sweep this
   before trusting it; `tau_ret=0.1` in the same check sharpened mean
   entropy to 1.29 nats).
3. **No clean isolation of Component B.** All four ablation variants
   (`base`, `A_only`, `B_only`, `A_and_B`) trained with the same default
   `beta` (1.0) — so even `B_only` still had a fully active, merely
   ungated `L_ret`. No variant tested "the physiological auxiliary task
   with *zero* retrieval-aware pressure," so the ablation could not
   cleanly attribute gains to Component B in isolation from Component A's
   underlying mechanism. Fixed by adding a fifth variant,
   `phys_only_beta0` (`ablation_components.py`, `beta=0.0` forced), which
   isolates exactly this condition.

None of these fixes have been re-run through the full ablation/multi-seed
pipeline yet (see "Reproducing the paper's results" below for the updated
command); until that re-run, we cannot say whether the corrected Component
A behaves differently, so **Table 5 / Figs. 4-5 in the current paper draft
still reflect the pre-fix code** and are annotated in the manuscript text
accordingly (Section 5.3).

</details>

## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff);
update once published) and/or this repository:

```bibtex
@article{goyo2026diversity,
  title   = {Diversity-Aware Self-Supervised {ECG} Retrieval},
  author  = {Goyo, Manuel Alejandro and Montoya, Fernando and Hidalgo, Mauricio},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Submitted; update volume/pages/DOI upon acceptance}
}
```

## License

Released under the [MIT License](LICENSE).
