"""
Filter raw DrugIssue ZIPs to keep only rows for cohort patients.
Usage: python filter_drugissue.py <SLURM_ARRAY_TASK_ID>
"""
import pandas as pd
import glob
import os
import sys
import time
import zipfile
import warnings

warnings.simplefilter(action='ignore')

# ── Paths ──
COHORT_PATID_PATH = "/scratch/alice/b/bg205/01_03_AURUM/cohort_patids.txt"
DRUGISSUE_GLOB = "/scratch/alice/b/bg205/01_03_AURUM/DrugIssue/*.zip"
OUTPUT_DIR = "/scratch/alice/b/bg205/01_03_AURUM/filtered_aurum_drugissue_chunks"

# ── Helpers ──
def read_zip_all_txt(zip_path):
    frames = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        members = [n for n in z.namelist() if n.lower().endswith(".txt")]
        print(f"DBG| [ZIP] {os.path.basename(zip_path)}: {len(members)} TXT member(s)")
        for name in members:
            with z.open(name) as f:
                try:
                    df = pd.read_csv(f, sep='\t', dtype=str, low_memory=False)
                    frames.append(df)
                except Exception as e:
                    print(f"  Skipping {name} (read error: {e})")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

# ── Main ──
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: Usage: python filter_drugissue.py <SLURM_ARRAY_TASK_ID>")
        sys.exit(1)

    task_id = int(sys.argv[1])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load cohort patids
    cohort = set(pd.read_csv(COHORT_PATID_PATH, dtype=str)['patid'].dropna().unique())
    print(f"Loaded {len(cohort):,} cohort patids")

    # Get DrugIssue files
    all_files = sorted(glob.glob(DRUGISSUE_GLOB))
    print(f"Total DrugIssue ZIPs: {len(all_files)}")

    if task_id >= len(all_files):
        print(f"Task ID {task_id} exceeds file count ({len(all_files)}). Exiting.")
        sys.exit(0)

    zipf = all_files[task_id]
    print(f"\n{'='*60}\nTask {task_id}: {os.path.basename(zipf)}\n{'='*60}")

    start = time.perf_counter()
    df = read_zip_all_txt(zipf)
    print(f"Rows before filter: {len(df):,}")

    if df.empty:
        print("Empty ZIP, skipping.")
        sys.exit(0)

    before_rows = len(df)
    before_patids = df['patid'].nunique()
    df = df[df['patid'].astype(str).isin(cohort)]

    print(f"Rows after filter:  {len(df):,} (dropped {before_rows - len(df):,})")
    print(f"Patids: {before_patids:,} → {df['patid'].nunique():,}")

    out_path = os.path.join(OUTPUT_DIR, f"Cleaned_AURUM_DrugIssue_{task_id}.txt")
    df.to_csv(out_path, sep='\t', index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(f"Done in {round((time.perf_counter() - start)/60, 2)} mins")
