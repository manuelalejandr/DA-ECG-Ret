"""
End-to-end orchestrator over MULTIPLE datasets and ALL SSL methods:
  1. trains SimCLR, CLOCS-lite, TS2Vec, TF-C, SoftCLT and DA-ECG-Ret,
  2. runs the full retrieval evaluation (NN / MMR / DPP re-ranking) +
     Pareto-frontier analysis, per dataset,
  3. generates every per-dataset figure at 300 dpi,
  4. aggregates across datasets (average ranks + Wilcoxon signed-rank
     tests vs. every baseline).

Dataset specs (space-separated, passed to --datasets):
    synthetic                 - fast smoke test, no download
    ptbxl                     - requires --data_root pointing at PTB-XL
    ucr:ECG200                - any UCR/UEA "ucr_name" (auto-downloaded by aeon)
    mitbih                    - streams from PhysioNet via wfdb

Quick smoke test (synthetic data, a few minutes on CPU):
    python run_all.py --datasets synthetic --epochs 5

Full multi-dataset benchmark (recommended for the paper):
    python run_all.py \\
        --datasets ucr:ECG200 ucr:ECG5000 ucr:ECGFiveDays ucr:TwoLeadECG \\
                   ucr:CinCECGTorso ucr:NonInvasiveFetalECGThorax1 \\
                   ucr:NonInvasiveFetalECGThorax2 mitbih \\
        --epochs 100
"""
import argparse
import os
import subprocess
import sys
import time
import zipfile

from config import SSL_METHODS


def zip_results(output_dir: str, zip_path: str, include_checkpoints: bool = False):
    """Zips output_dir into zip_path. By default SKIPS the 'checkpoints/'
    subfolders (model weights) to keep the archive small and focused on
    what's needed to discuss results (CSVs, JSON histories, 300 dpi
    figures) -- the checkpoints remain untouched on disk at --output_dir
    for your own later use, just not included in the file you send back."""
    output_dir = os.path.normpath(output_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            if not include_checkpoints and "checkpoints" in dirs:
                dirs.remove("checkpoints")
            for fname in files:
                full_path = os.path.join(root, fname)
                arcname = os.path.relpath(full_path, os.path.dirname(output_dir))
                zf.write(full_path, arcname)
    return zip_path


def run(cmd):
    print(f"\n$ {' '.join(cmd)}\n" + "-" * 60)
    subprocess.run(cmd, check=True)


def parse_dataset_spec(spec: str):
    """'ucr:ECG200' -> ('ucr', 'ECG200'); 'mitbih' -> ('mitbih', None)."""
    if ":" in spec:
        kind, name = spec.split(":", 1)
        return kind, name
    return spec, None


def dataset_tag(kind: str, name: str) -> str:
    return f"ucr_{name}" if kind == "ucr" else kind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["synthetic"])
    parser.add_argument("--data_root", type=str, default="./ptbxl_data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--encoder", choices=["transformer", "resnet18", "fcn"], default="transformer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_synthetic_train", type=int, default=1200)
    parser.add_argument("--n_synthetic_val", type=int, default=300)
    parser.add_argument("--methods", nargs="+", default=SSL_METHODS)
    parser.add_argument("--beta", type=float, default=None,
                         help="override DA-ECG-Ret's beta (default: config.py's, currently 1.0)")
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_aggregate", action="store_true")
    parser.add_argument("--no_zip", action="store_true",
                         help="skip creating the final results .zip (on by default)")
    parser.add_argument("--zip_path", type=str, default=None,
                         help="where to write the results zip (default: alongside --output_dir, "
                              "named <output_dir>_results_<timestamp>.zip)")
    parser.add_argument("--include_checkpoints_in_zip", action="store_true",
                         help="also include model checkpoints (.pt files) in the results zip; "
                              "off by default to keep the archive small (checkpoints stay on "
                              "disk under --output_dir either way)")
    args = parser.parse_args()

    py = sys.executable
    tags = []

    for spec in args.datasets:
        kind, name = parse_dataset_spec(spec)
        tag = dataset_tag(kind, name)
        tags.append(tag)
        print(f"\n{'='*70}\nDataset: {spec}  (output tag: {tag})\n{'='*70}")

        common = ["--dataset", kind, "--data_root", args.data_root,
                  "--encoder", args.encoder, "--output_dir", args.output_dir]
        if kind == "ucr":
            common += ["--ucr_name", name]

        if not args.skip_train:
            for method in args.methods:
                cmd = [py, "train.py", "--method", method, "--epochs", str(args.epochs),
                       "--batch_size", str(args.batch_size)] + common
                if args.beta is not None:
                    cmd += ["--beta", str(args.beta)]
                if args.lam is not None:
                    cmd += ["--lam", str(args.lam)]
                if kind == "synthetic":
                    cmd += ["--n_synthetic_train", str(args.n_synthetic_train),
                            "--n_synthetic_val", str(args.n_synthetic_val)]
                run(cmd)

        run([py, "evaluate.py", "--methods"] + args.methods + common)
        run([py, "plots.py"] + common)

    if not args.skip_aggregate and len(tags) > 1:
        run([py, "aggregate.py", "--output_dir", args.output_dir, "--dataset_tags"] + tags)

    print("\nAll done. Per-dataset outputs under:")
    for tag in tags:
        print(f"  - {args.output_dir}/{tag}/results/retrieval_results.csv")
        print(f"  - {args.output_dir}/{tag}/figures/*.png (300 dpi)")
    if len(tags) > 1:
        print(f"  - {args.output_dir}/summary/*.csv  (cross-dataset average ranks + Wilcoxon tests)")
        print(f"  - {args.output_dir}/summary/figures/average_rank_bar.png")

    if not args.no_zip:
        zip_path = args.zip_path
        if zip_path is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_name = os.path.basename(os.path.normpath(args.output_dir))
            zip_path = os.path.join(os.path.dirname(os.path.normpath(args.output_dir)) or ".",
                                     f"{out_name}_results_{stamp}.zip")
        archive_path = zip_results(args.output_dir, zip_path,
                                    include_checkpoints=args.include_checkpoints_in_zip)
        size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        print(f"\nPackaged all results into: {archive_path}  ({size_mb:.1f} MB)")
        if not args.include_checkpoints_in_zip:
            print("(model checkpoints excluded to keep this small; pass "
                  "--include_checkpoints_in_zip to include them; they remain on disk regardless)")
        print("Send this single .zip back for the results discussion.")


if __name__ == "__main__":
    main()
