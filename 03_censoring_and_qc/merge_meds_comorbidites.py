#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

# =============================================================================
# PATHS
# =============================================================================

BASE_FILE = "/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/Combined_GOLD_Aurum_recoded.txt"

# Medication wide files
AURUM_MED_FILE = "/scratch/alice/b/bg205/01_03_AURUM/medication_output/AURUM_medication_wide.csv"
GOLD_MED_FILE  = "/scratch/alice/b/bg205/28_02_GOLD/medication_output/GOLD_medication_wide.csv"

# Comorbidity file
COMORB_FILE = "/scratch/alice/b/bg205/16_02_26/comorbidityV2/final_combined_comorbidities_fixed.txt"

OUTPUT_FILE = "/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/Combined_GOLD_Aurum_with_meds_comorbidities.txt"

# =============================================================================
# HELPER
# =============================================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# 1. LOAD BASE MASTER DATASET
# =============================================================================

print_section("LOADING BASE MASTER DATASET")

df = pd.read_csv(BASE_FILE, sep="\t", low_memory=False)

print(f"Base shape: {df.shape}")
print("\nRows by database:")
print(df["database"].value_counts())

# Safety check
dup_base = df.duplicated(subset=["database", "patid"]).sum()
print(f"\nDuplicate database+patid in base: {dup_base:,}")

if dup_base > 0:
    raise ValueError("Base file has duplicate database+patid rows. Stop and investigate.")


# =============================================================================
# 2. LOAD AND COMBINE MEDICATION WIDE FILES
# =============================================================================

print_section("LOADING MEDICATION FILES")

aurum_med = pd.read_csv(AURUM_MED_FILE, low_memory=False)
gold_med = pd.read_csv(GOLD_MED_FILE, low_memory=False)

aurum_med["database"] = "AURUM"
gold_med["database"] = "GOLD"

print(f"Aurum medication shape: {aurum_med.shape}")
print(f"GOLD medication shape: {gold_med.shape}")

# Make sure patid is consistent type
df["patid"] = df["patid"].astype(str)
aurum_med["patid"] = aurum_med["patid"].astype(str)
gold_med["patid"] = gold_med["patid"].astype(str)

# Combine medication files
med = pd.concat([gold_med, aurum_med], ignore_index=True, sort=False)

print(f"Combined medication shape: {med.shape}")

# Check duplicates
dup_med = med.duplicated(subset=["database", "patid"]).sum()
print(f"Duplicate database+patid in medication file: {dup_med:,}")

if dup_med > 0:
    raise ValueError("Medication file has duplicate database+patid rows. Stop and investigate.")

# Identify medication columns
med_key_cols = ["database", "patid"]
med_cols = [c for c in med.columns if c not in med_key_cols]

print(f"\nMedication columns to merge: {med_cols}")

# Rename medication columns to avoid confusion
rename_med_cols = {c: f"med_{c}" for c in med_cols if not c.startswith("med_")}
med = med.rename(columns=rename_med_cols)

med_cols_renamed = [c for c in med.columns if c not in med_key_cols]

# Merge
print("\nMerging medication data onto base...")
before_rows = len(df)

df = df.merge(med, on=["database", "patid"], how="left")

after_rows = len(df)
print(f"Rows before: {before_rows:,}")
print(f"Rows after:  {after_rows:,}")

if before_rows != after_rows:
    raise ValueError("Row count changed after medication merge. Stop and investigate.")

# Fill missing medication indicators with 0
for col in med_cols_renamed:
    df[col] = df[col].fillna(0).astype("int8")

print("\nMedication prevalence after merge:")
for col in med_cols_renamed:
    print(f"  {col}: {df[col].sum():,} ({df[col].mean()*100:.2f}%)")


# =============================================================================
# 3. LOAD COMORBIDITY FILE
# =============================================================================

print_section("LOADING COMORBIDITY FILE")

comorb = pd.read_csv(COMORB_FILE, sep="\t", low_memory=False)

print(f"Comorbidity shape: {comorb.shape}")
print(f"Comorbidity columns: {comorb.columns.tolist()}")

# Make sure patid is consistent type
comorb["patid"] = comorb["patid"].astype(str)

# If comorbidity file does not have database column, try to infer/check
if "database" not in comorb.columns:
    raise ValueError(
        "Comorbidity file does not contain a 'database' column. "
        "You need database = GOLD/AURUM before merging, otherwise patids may collide."
    )

# Check duplicates
dup_comorb = comorb.duplicated(subset=["database", "patid"]).sum()
print(f"\nDuplicate database+patid in comorbidity file: {dup_comorb:,}")

if dup_comorb > 0:
    raise ValueError("Comorbidity file has duplicate database+patid rows. Stop and investigate.")

# Decide columns to merge
comorb_key_cols = ["database", "patid"]

# Do not bring duplicate baseline columns if present
exclude_cols = set(comorb_key_cols + [
    "indexdate",
    "gender",
    "yob",
    "mob",
    "e2019_imd_10",
    "gen_ethnicity"
])

comorb_cols = [c for c in comorb.columns if c not in exclude_cols]

print(f"\nComorbidity columns to merge: {comorb_cols}")

# Rename comorbidity columns to make them clear
rename_comorb_cols = {}

for c in comorb_cols:
    if c.endswith("_first_date"):
        rename_comorb_cols[c] = f"comorb_{c}"
    elif c.endswith("_flag"):
        rename_comorb_cols[c] = f"comorb_{c}"
    elif c.lower() in ["ckd", "htn", "cvd", "cancer", "mi", "hf", "stroke", "chd"]:
        rename_comorb_cols[c] = f"comorb_{c}"
    else:
        rename_comorb_cols[c] = f"comorb_{c}"

comorb = comorb[comorb_key_cols + comorb_cols].rename(columns=rename_comorb_cols)

comorb_cols_renamed = [c for c in comorb.columns if c not in comorb_key_cols]

# Merge
print("\nMerging comorbidity data onto base...")
before_rows = len(df)

df = df.merge(comorb, on=["database", "patid"], how="left")

after_rows = len(df)
print(f"Rows before: {before_rows:,}")
print(f"Rows after:  {after_rows:,}")

if before_rows != after_rows:
    raise ValueError("Row count changed after comorbidity merge. Stop and investigate.")


# =============================================================================
# 4. FILL COMORBIDITY FLAGS ONLY
# =============================================================================

print_section("CLEANING COMORBIDITY COLUMNS")

# Fill binary flag columns with 0.
# Do NOT fill date columns with 0.
for col in comorb_cols_renamed:
    if "date" not in col.lower():
        unique_vals = set(df[col].dropna().unique())

        # Fill only if it looks binary/numeric
        if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
            df[col] = df[col].fillna(0).astype("int8")

print("\nComorbidity prevalence after merge:")
for col in comorb_cols_renamed:
    if "date" not in col.lower():
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"  {col}: {df[col].sum():,} ({df[col].mean()*100:.2f}%)")


# =============================================================================
# 5. FINAL QC
# =============================================================================

print_section("FINAL QC")

print(f"Final shape: {df.shape}")

print("\nRows by database:")
print(df["database"].value_counts())

print("\nDuplicate database+patid:")
print(df.duplicated(subset=["database", "patid"]).sum())

print("\nMissing medication columns:")
print(df[med_cols_renamed].isna().sum())

print("\nIMD quintile distribution:")
print(df["imd_quintile"].value_counts(dropna=False).sort_index())

print("\nEthnicity distribution:")
print(df["gen_ethnicity"].value_counts(dropna=False))

# Optional: show newly added columns
new_cols = med_cols_renamed + comorb_cols_renamed
print(f"\nNumber of newly added columns: {len(new_cols)}")
print(new_cols)


# =============================================================================
# 6. SAVE FINAL DATASET
# =============================================================================

print_section("SAVING FINAL DATASET")

df.to_csv(OUTPUT_FILE, sep="\t", index=False)

print(f"Saved final dataset to:")
print(OUTPUT_FILE)
