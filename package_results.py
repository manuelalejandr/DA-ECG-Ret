"""
Standalone convenience script: zips an existing --output_dir into a single
.zip file, without re-running training/evaluation. Useful if you added a
dataset later (re-ran aggregate.py) or just want a fresh archive to send.

By default, model checkpoints (.pt files, one per method per dataset) are
EXCLUDED from the archive to keep it small and focused on what's needed to
discuss results -- CSVs, JSON training histories, and 300 dpi figures.
Checkpoints stay untouched on disk at --output_dir either way; pass
--include_checkpoints if you specifically want them zipped too (e.g. to
move the whole experiment to another machine).

`run_all.py` already does this automatically at the end of a full run
(unless --no_zip is passed); use this script when you want to re-package
without re-running everything.

Usage:
    python package_results.py --output_dir ./outputs
    python package_results.py --output_dir ./outputs --zip_path ./my_results.zip
    python package_results.py --output_dir ./outputs --include_checkpoints
"""
import argparse
import os
import time
import zipfile


def zip_results(output_dir: str, zip_path: str, include_checkpoints: bool = False):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--zip_path", type=str, default=None)
    parser.add_argument("--include_checkpoints", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"No such directory: {args.output_dir}")

    zip_path = args.zip_path
    if zip_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_name = os.path.basename(os.path.normpath(args.output_dir))
        zip_path = os.path.join(os.path.dirname(os.path.normpath(args.output_dir)) or ".",
                                 f"{out_name}_results_{stamp}.zip")

    archive_path = zip_results(args.output_dir, zip_path,
                                include_checkpoints=args.include_checkpoints)
    size_mb = os.path.getsize(archive_path) / (1024 * 1024)
    print(f"Packaged {args.output_dir} into: {archive_path}  ({size_mb:.1f} MB)")
    if not args.include_checkpoints:
        print("(model checkpoints excluded; pass --include_checkpoints to include them)")


if __name__ == "__main__":
    main()
