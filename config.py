"""
Global configuration for DA-ECG-Ret experiments.

Values follow the "Implementation Details (Tentative)" section of the draft
(batch size, temperature, lambda, alpha, beta). Override any of these from
the command line in train.py / run_all.py.
"""
from dataclasses import dataclass, field
from typing import List


# All free, univariate ECG problems in the UCR/UEA time series archive
# (https://www.timeseriesclassification.com/dataset.php), fetched through
# aeon (https://www.aeon-toolkit.org). Chosen by filtering the archive's
# dataset list for "ECG" in the name; CinCECGTorso is included because it is
# also ECG-derived even though its name doesn't say so directly.
UCR_ECG_DATASETS = [
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "TwoLeadECG",
    "CinCECGTorso",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
]

# MIT-BIH Arrhythmia Database (Moody & Mark, 2001) - open-access on
# PhysioNet, no credentialing required. We stream records directly via
# wfdb's pn_dir mechanism (no bulk download needed) and segment individual
# heartbeats around annotated R-peaks, labelled with the standard AAMI
# 5-class scheme (N, S, V, F, Q). A modest default record subset is used to
# keep first-run latency reasonable; pass more records for the full study.
MITBIH_DEFAULT_RECORDS = [
    "100", "101", "103", "105", "106", "108", "109", "111", "112", "113",
    "114", "115", "116", "117", "118", "119", "121", "122", "123", "124",
]
MITBIH_WINDOW_BEFORE = 90   # samples before R-peak (fs=360 Hz -> 0.25 s)
MITBIH_WINDOW_AFTER = 160   # samples after R-peak (-> ~0.44 s)
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

# SSL methods available (see ssl_methods/). "da_ecg_ret" is the proposed
# method; the other five are baselines.
SSL_METHODS = ["simclr", "clocs_lite", "ts2vec", "tfc", "softclt", "da_ecg_ret"]
BASELINE_METHODS = ["simclr", "clocs_lite", "ts2vec", "tfc", "softclt"]


@dataclass
class Config:
    # ---- data ----
    dataset: str = "synthetic"                # synthetic | ptbxl | ucr | mitbih
    ucr_name: str = "ECG200"                  # which UCR/UEA problem when dataset=="ucr"
    data_root: str = "./ptbxl_data"          # where ptbxl_database.csv etc. live
    sampling_rate: int = 100                  # 100 Hz records (filename_lr)
    lead_index: int = 1                       # 0=I, 1=II, ... standard 12-lead order; -1 = ALL 12 leads
    in_channels: int = 1                      # encoder input channels (1 for a single lead, 12 for lead_index=-1)
    signal_length: int = 1000                 # 10 s @ 100 Hz (PTB-XL only; UCR/MIT-BIH infer their own)
    superclasses: List[str] = field(
        default_factory=lambda: ["NORM", "MI", "STTC", "CD", "HYP"]
    )
    mitbih_records: List[str] = field(default_factory=lambda: list(MITBIH_DEFAULT_RECORDS))

    # ---- encoder ----
    encoder_name: str = "transformer"   # "transformer" | "resnet18" | "fcn"
                                         # "fcn": classic 4-layer FCN (Wang et al. 2017; 16/32/32/64
                                         # filters, kernels 8/5/3/3), the small, well-known backbone
                                         # from the authors' prior semi-supervised TSC work -- use
                                         # embed_dim=64 to match it exactly (no extra projection layer)
    embed_dim: int = 128        # encoder output dim (d)
    proj_dim: int = 128         # projection head output dim (contrastive space)
    conv_channels: int = 64
    n_transformer_layers: int = 4
    n_heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.1

    # ---- contrastive / retrieval-aware training ----
    batch_size: int = 256       # paper suggests 512; lowered as a practical default
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-6
    tau: float = 0.07           # InfoNCE temperature
    alpha: float = 10.0         # smooth-max sharpness (Eq. 8)
    lam: float = 0.7            # relevance/diversity trade-off (Eq. 9), lambda in paper
    beta: float = 1.0           # weight of retrieval-aware loss (Eq. 12) — raised from 0.5 to 1.0
                                 # after the v2 (Components A+B) sweep showed beta=1.0 is the best
                                 # or near-best operating point on 2 of 3 tested datasets, with the
                                 # third (ECG5000) showing only noise-level differences across beta
    exclude_self_from_redundancy: bool = True   # bug fix, see losses/mmr_loss.py::smooth_redundancy
    exclude_twin_from_redundancy: bool = True   # v2.1 fix: also exclude each candidate's augmented
                                                 # twin from its redundancy term (see mmr_loss.py)
    tau_ret: float = 0.1                        # temperature for Eq. 10's softmax (v2.1 fix). Old
                                                 # behaviour (no temperature) = 1.0; 0.1 is a first,
                                                 # UNSWEPT default matching the InfoNCE-adjacent
                                                 # scale (tau=0.07) -- sweep before trusting it

    # ---- DA-ECG-Ret v2: Component A (pseudo-cluster gating) ----
    use_cluster_gate: bool = True        # gate L_ret's redundancy term by pseudo-cluster (see models/prototypes.py)
    n_pseudo_clusters: int = 8           # k for the per-batch soft k-means
    kmeans_iters: int = 5
    kmeans_temperature: float = 0.1

    # ---- DA-ECG-Ret v2: Component B (physiological auxiliary pretext task) ----
    use_physio_aux_task: bool = True
    n_physio_features: int = 6           # data/physiological_features.py::N_PHYSIO_FEATURES
    gamma_physio: float = 0.1            # weight of the auxiliary regression loss

    # ---- DA-ECG-Ret v2: beta warm-up curriculum ----
    beta_warmup_frac: float = 0.2        # fraction of training over which beta ramps 0 -> cfg.beta

    # ---- supervised reference encoders (ssl_methods/supervised_ref.py) ----
    n_label_code: int = 44    # overwritten from the PTB-XL hierarchy by ptbxl_benchmark.py
    n_label_super: int = 5

    # ---- retrieval / evaluation ----
    top_n_candidates: int = 200   # stage-1 cosine shortlist before re-ranking
    cohort_sizes: tuple = (5, 10, 20)  # K values reported
    mmr_lambda_inference: float = 0.7  # lambda used by classical MMR at test time
    lambda_sweep: tuple = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)  # for Pareto-frontier analysis

    # ---- TF-C specific ----
    tfc_consistency_weight: float = 1.0

    # ---- SoftCLT specific ----
    softclt_tau_inst: float = 1.0     # softness of the instance-similarity kernel
    softclt_alpha: float = 0.5        # blend between hard and soft targets (0=hard InfoNCE, 1=fully soft)

    # ---- misc ----
    seed: int = 42
    device: str = "auto"        # resolved via device_utils.resolve_device(); this field is
                                 # informational only and not read directly by train/evaluate/plots
    num_workers: int = 0        # 0 is safest on macOS (spawn-based multiprocessing);
                                 # raise to 2-4 on Linux/Colab if I/O-bound
    output_dir: str = "./outputs"
    dpi: int = 300
