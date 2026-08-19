import pandas as pd
import glob
import os
import gzip

chunks_dir = "/scratch/alice/b/bg205/01_03_AURUM/smoking_chunks"
pattern = os.path.join(chunks_dir, "Aurum_Clinical_SmokingStatus_task*.txt.gz")
out_path = os.path.join(chunks_dir, "Aurum_smoking_records_all.txt.gz")

files = sorted(glob.glob(pattern))
if not files:
    raise FileNotFoundError(f"No chunk files found matching {pattern}")

# Memory-friendly streaming merge
with gzip.open(out_path, "wt") as w:
    wrote_header = False
    for f in files:
        for chunk in pd.read_csv(f, sep="\t", dtype=str, chunksize=500_000):
            chunk.to_csv(w, sep="\t", index=False, header=not wrote_header)
            wrote_header = True

print(f"Merged {len(files)} files -> {out_path}")
