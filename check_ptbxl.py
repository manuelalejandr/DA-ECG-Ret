"""
Verifies a PTB-XL download (100 Hz version) by checking every .dat/.hea file.
A 100 Hz, 10 s, 12-lead record has a .dat of exactly 24000 bytes; anything
else is a truncated/corrupt download (the cause of
"cannot reshape array of size N into shape (12)").

Usage:
    python check_ptbxl.py --data_root ./ptbxl_data/physionet.org/files/ptb-xl/1.0.3
    python check_ptbxl.py --data_root ... --delete    # delete the bad files
Then re-run the same wget command (it resumes/refills missing files):
    wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P ./ptbxl_data
"""
import argparse, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--delete", action="store_true")
a = ap.parse_args()

df = pd.read_csv(os.path.join(a.data_root, "ptbxl_database.csv"))
bad = []
for rel in df["filename_lr"]:
    dat, hea = os.path.join(a.data_root, rel + ".dat"), os.path.join(a.data_root, rel + ".hea")
    if not os.path.exists(hea) or not os.path.exists(dat) or os.path.getsize(dat) != 24000:
        bad.append((rel, os.path.getsize(dat) if os.path.exists(dat) else None))
print(f"{len(df)} records checked, {len(bad)} bad")
for rel, sz in bad[:50]:
    print("  BAD:", rel, "size:", sz)
if a.delete:
    for rel, _ in bad:
        for ext in (".dat", ".hea"):
            f = os.path.join(a.data_root, rel + ext)
            if os.path.exists(f):
                os.remove(f)
    print("Deleted bad files. Now re-run the wget command, then delete any _cache_*.npz in data_root.")
