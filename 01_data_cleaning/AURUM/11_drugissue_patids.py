"""
Extract unique patids from filtered Observation files and save to a single file.
Run once before the DrugIssue SLURM array job.
"""
import pandas as pd
import glob
import os

FILTERED_OBS_DIR = "/scratch/alice/b/bg205/01_03_AURUM/filtered_aurum_chunks"
OUT_PATH = "/scratch/alice/b/bg205/01_03_AURUM/cohort_patids.txt"

pattern = os.path.join(FILTERED_OBS_DIR, "Cleaned_AURUM_Observation_*.txt")
files = sorted(glob.glob(pattern))
assert len(files) > 0, f"No filtered Observation files found in {FILTERED_OBS_DIR}"

patids = set()
for f in files:
    chunk = pd.read_csv(f, sep='\t', dtype=str, usecols=['patid'])
    patids.update(chunk['patid'].dropna().unique())
    print(f"  {os.path.basename(f)}: +{chunk['patid'].nunique():,} patids (running total: {len(patids):,})")

print(f"\nTotal unique cohort patids: {len(patids):,}")

pd.Series(sorted(patids), name='patid').to_csv(OUT_PATH, index=False)
print(f"Saved to: {OUT_PATH}")
