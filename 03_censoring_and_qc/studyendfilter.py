import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
STUDY_END_DATE = "2021-03-31"
study_end = pd.Timestamp(STUDY_END_DATE)

MASTER_FILE = "/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/Combined_GOLD_Aurum_with_meds_comorbidities.txt"
MASTER_SEP = "\t"

FILES_TO_FILTER = {
    "Medications (Aurum)": {
        "path": "/scratch/alice/b/bg205/01_03_AURUM/medication_output/AURUM_medication_wide.csv",
        "sep": ","
    },
    "Medications (GOLD)": {
        "path": "/scratch/alice/b/bg205/28_02_GOLD/medication_output/GOLD_medication_wide.csv",
        "sep": ","
    },
    "Comorbidities": {
        "path": "/scratch/alice/b/bg205/16_02_26/comorbidityV2/final_combined_comorbidities_fixed.txt",
        "sep": "\t"
    },
}

# ============================================================
# STEP 1: Filter master cohort by study end date
# ============================================================

print(f"Step 1: Filtering master cohort to indexdate <= {STUDY_END_DATE}\n")

master = pd.read_csv(MASTER_FILE, sep=MASTER_SEP, low_memory=False)

master["patid"] = master["patid"].astype(str)
master["indexdate"] = pd.to_datetime(master["indexdate"], errors="coerce")

n_before = len(master)
n_gold_before = (master["database"] == "GOLD").sum()

master_filtered = master[master["indexdate"] <= study_end].copy()

n_after = len(master_filtered)
n_removed = n_before - n_after
n_gold_after = (master_filtered["database"] == "GOLD").sum()

print(f"  Before:  {n_before:,}")
print(f"  After:   {n_after:,}")
print(f"  Removed: {n_removed:,}")

removed_by_db = master.loc[master["indexdate"] > study_end, "database"].value_counts()

print("\n  Removed by database:")
print(removed_by_db.to_string())

print(f"\n  GOLD patients removed: {n_gold_before - n_gold_after:,}")

max_date = master_filtered["indexdate"].max()
print(f"\n  ✓ Max indexdate after filter: {max_date}")

assert max_date <= study_end, f"ERROR: Max date {max_date} > {STUDY_END_DATE}"

master_out = MASTER_FILE.replace(".txt", "_studyend_clean.txt")
master_filtered.to_csv(master_out, sep=MASTER_SEP, index=False)

print(f"  Saved → {master_out}\n")

# ============================================================
# STEP 2: Build valid patid set
# ============================================================

valid_patids = set(master_filtered["patid"].astype(str))

print(f"Step 2: Valid patid set built: {len(valid_patids):,} patients\n")

# ============================================================
# STEP 3: Filter each downstream file
# ============================================================

print("Step 3: Filtering downstream files...\n")

for label, info in FILES_TO_FILTER.items():
    filepath = info["path"]
    sep = info["sep"]

    print(f"  --- {label} ---")

    try:
        df = pd.read_csv(filepath, sep=sep, low_memory=False)
    except FileNotFoundError:
        print(f"  File not found: {filepath} — skipping\n")
        continue

    df["patid"] = df["patid"].astype(str)

    n_before_file = len(df)
    df_filtered = df[df["patid"].isin(valid_patids)].copy()
    n_after_file = len(df_filtered)

    print(f"  Before:  {n_before_file:,} rows")
    print(f"  After:   {n_after_file:,} rows")
    print(f"  Removed: {n_before_file - n_after_file:,} rows")

    if n_after_file == len(valid_patids):
        print("  ✓ Row count matches filtered cohort")
    else:
        print(
            f"  ⚠ Row count ({n_after_file:,}) differs from cohort ({len(valid_patids):,}) "
            f"— check if file has multiple rows per patient or missing patients"
        )

    out_path = Path(filepath)
    out_path = out_path.with_name(out_path.stem + "_studyend" + out_path.suffix)

    df_filtered.to_csv(out_path, sep=sep, index=False)

    print(f"  Saved → {out_path}\n")

print(f"{'='*70}")
print("  DONE")
print(f"{'='*70}")
print(f"  Study end date: {STUDY_END_DATE}")
print(f"  Patients removed: {n_removed:,}")
print("  Removed by database:")
print(removed_by_db.to_string())
print(f"  Filtered cohort: {n_after:,} patients")
