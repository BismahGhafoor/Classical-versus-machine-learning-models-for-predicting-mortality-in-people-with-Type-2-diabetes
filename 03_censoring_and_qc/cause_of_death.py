#!/usr/bin/env python3
"""
================================================================================
Cause-of-Death Extraction from ONS Death File
================================================================================
Reads the ONS death registration file (with ICD-10 codes), classifies
underlying cause of death into CVD / Cancer / CKD, and merges flags
into the final cleaned dataset.

ICD-10 definitions match the comorbidity extraction script exactly:
  CVD:    I*  (excluding I10-I15 hypertension)
  Cancer: C*
  CKD:    N18.3, N18.4, N18.5, N18.6, N19, Z99.2

Outputs:
  - cod_classified.txt          (one row per death, with all flags)
  - cod_qc_summary.txt          (prevalence and top codes)
  - Final dataset with cod_* columns appended
================================================================================
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURATION — edit paths here
# ============================================================

# ONS death file (the one you showed me)
#ONS_DEATH_FILE = "/scratch/alice/b/bg205/HES_linked/GOLD/death_patient_23_002869_DM.txt"
# ^ ADJUST THIS — find the actual path. You may have separate GOLD/Aurum files.
#   If so, set both:
ONS_DEATH_FILE_GOLD  = "/scratch/alice/b/bg205/HES_linked/GOLD/death_patient_23_002869_DM.txt"
ONS_DEATH_FILE_AURUM = "/scratch/alice/b/bg205/HES_linked/Aurum/death_patient_23_002869_DM.txt"
# Set to None if you only have one file:
# ONS_DEATH_FILE_AURUM = None

# Final cleaned dataset to merge into
FINAL_DATASET = "/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/Combined_GOLD_Aurum_with_meds_comorbidities_studyend_clean.txt"

# Output directory
OUTPUT_DIR = "/scratch/alice/b/bg205/16_02_26/CLEANED_DATA"

# Output filename (the updated dataset)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Combined_GOLD_Aurum_with_meds_comorbidities_studyend_cod.txt")

# ============================================================
# ICD-10 DEFINITIONS — identical to comorbidity script
# ============================================================

def classify_underlying_cause(icd_raw):
    """
    Classify a single ICD-10 code into cause-of-death categories.
    Returns a dict of binary flags.
    
    Uses EXACTLY the same definitions as the comorbidity extraction:
      CVD:    I* EXCLUDING I10-I15 (hypertension)
      Cancer: C*
      CKD:    N18.3, N18.4, N18.5, N18.6, N19, Z99.2
    """
    flags = {
        'cod_cvd': 0,
        'cod_cancer': 0,
        'cod_ckd': 0,
    }
    
    if pd.isna(icd_raw) or str(icd_raw).strip() == '':
        return flags
    
    icd = str(icd_raw).strip().upper()
    icd_nodot = icd.replace('.', '')
    
    # --- CVD: I* excluding I10-I15 (hypertension) ---
    # This matches your comorbidity script's HES_CONDITIONS["cvd"]
    if icd.startswith('I'):
        htn_prefixes = ['I10', 'I11', 'I12', 'I13', 'I14', 'I15']
        if not any(icd.startswith(p) for p in htn_prefixes):
            flags['cod_cvd'] = 1
    
    # --- Cancer: C* ---
    if icd.startswith('C'):
        flags['cod_cancer'] = 1
    
    # --- CKD: N18.3, N18.4, N18.5, N18.6, N19, Z99.2 ---
    # This matches your comorbidity script's HES_CONDITIONS["ckd"]
    ckd_codes_nodot = ['N183', 'N184', 'N185', 'N186', 'N19', 'Z992']
    if any(icd_nodot.startswith(c) for c in ckd_codes_nodot):
        flags['cod_ckd'] = 1
    
    return flags


# ============================================================
# STEP 1: LOAD AND CLASSIFY ONS DEATH FILE(S)
# ============================================================

print("=" * 60)
print("CAUSE-OF-DEATH EXTRACTION FROM ONS")
print("=" * 60)

death_frames = []

for label, filepath in [("GOLD", ONS_DEATH_FILE_GOLD), ("AURUM", ONS_DEATH_FILE_AURUM)]:
    if filepath is None or not os.path.exists(filepath):
        print(f"\n  {label}: File not found or not set — skipping")
        print(f"    Path: {filepath}")
        continue
    
    print(f"\n  Loading {label} ONS death file: {filepath}")
    
    # Try tab-separated first, fall back to comma
    try:
        df = pd.read_csv(filepath, sep='\t', dtype=str, low_memory=False)
        if len(df.columns) < 3:
            df = pd.read_csv(filepath, sep=',', dtype=str, low_memory=False)
    except:
        df = pd.read_csv(filepath, sep=',', dtype=str, low_memory=False)
    
    print(f"    Rows: {len(df):,}")
    print(f"    Columns: {df.columns.tolist()}")
    
    # Standardise column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Check required columns exist
    if 'cause' not in df.columns:
        # Sometimes the underlying cause column has a different name
        possible = [c for c in df.columns if 'cause' in c.lower() and c != 'cause']
        print(f"    WARNING: 'cause' column not found. Available cause columns: {possible}")
        print(f"    All columns: {df.columns.tolist()}")
        continue
    
    if 'patid' not in df.columns:
        print(f"    WARNING: 'patid' column not found")
        continue
    
    df['patid'] = pd.to_numeric(df['patid'], errors='coerce')
    df = df.dropna(subset=['patid'])
    df['patid'] = df['patid'].astype(np.int64)
    
    # Parse date of death
    if 'dod' in df.columns:
        df['dod_ons'] = pd.to_datetime(df['dod'], dayfirst=True, errors='coerce')
    
    # Classify the UNDERLYING cause of death
    print(f"    Classifying underlying cause of death...")
    cause_flags = df['cause'].apply(classify_underlying_cause).apply(pd.Series)
    df = pd.concat([df, cause_flags], axis=1)
    
    df['_source'] = label
    death_frames.append(df)
    
    # Quick summary
    print(f"    CVD deaths:    {df['cod_cvd'].sum():,} ({df['cod_cvd'].mean()*100:.1f}%)")
    print(f"    Cancer deaths: {df['cod_cancer'].sum():,} ({df['cod_cancer'].mean()*100:.1f}%)")
    print(f"    CKD deaths:    {df['cod_ckd'].sum():,} ({df['cod_ckd'].mean()*100:.1f}%)")
    other = len(df) - df['cod_cvd'].sum() - df['cod_cancer'].sum() - df['cod_ckd'].sum()
    # Note: a death can't be both CVD and cancer from underlying cause, but
    # theoretically CKD codes don't overlap with I* or C*, so no double counting
    print(f"    Other cause:   {other:,} ({other/len(df)*100:.1f}%)")
    
    # Top underlying causes
    print(f"\n    Top 15 underlying causes:")
    top_causes = df['cause'].value_counts().head(15)
    for code, count in top_causes.items():
        pct = count / len(df) * 100
        print(f"      {code}: {count:,} ({pct:.1f}%)")

if not death_frames:
    raise FileNotFoundError(
        "No ONS death files found. Check the file paths in CONFIGURATION."
    )

# Combine GOLD + Aurum death records
cod_all = pd.concat(death_frames, ignore_index=True)
print(f"\n  Combined ONS death records: {len(cod_all):,}")
print(f"  Unique patients: {cod_all['patid'].nunique():,}")

# Handle duplicate patids (shouldn't happen, but just in case)
n_dup = cod_all.duplicated(subset=['patid'], keep=False).sum()
if n_dup > 0:
    print(f"  WARNING: {n_dup} duplicate patids found — keeping first occurrence")
    cod_all = cod_all.drop_duplicates(subset=['patid'], keep='first')


# ============================================================
# STEP 2: QC AND SAVE CLASSIFIED DEATHS
# ============================================================

print("\n" + "=" * 60)
print("QC SUMMARY")
print("=" * 60)

# Save the full classified file for reference
cod_output = cod_all[['patid', 'cause', 'cod_cvd', 'cod_cancer', 'cod_ckd', '_source']].copy()

# Add contributory cause columns if they exist
contrib_cols = [c for c in cod_all.columns if c.startswith('cause') and c != 'cause']
for col in sorted(contrib_cols):
    if col in cod_all.columns:
        cod_output[col] = cod_all[col]

cod_output_path = os.path.join(OUTPUT_DIR, "cod_classified.txt")
cod_output.to_csv(cod_output_path, sep='\t', index=False)
print(f"  Classified deaths saved to: {cod_output_path}")

# QC summary
qc_rows = []
for cause_col, cause_label in [('cod_cvd', 'CVD'), ('cod_cancer', 'Cancer'), ('cod_ckd', 'CKD')]:
    n = int(cod_all[cause_col].sum())
    pct = cod_all[cause_col].mean() * 100
    
    # Most common specific codes within this category
    if n > 0:
        mask = cod_all[cause_col] == 1
        top = cod_all.loc[mask, 'cause'].value_counts().head(5)
        top_str = '; '.join([f"{code} (n={count})" for code, count in top.items()])
    else:
        top_str = ''
    
    qc_rows.append({
        'cause_category': cause_label,
        'n_deaths': n,
        'pct_of_all_deaths': round(pct, 1),
        'top_icd10_codes': top_str
    })

# "Other" category
n_other = len(cod_all) - cod_all[['cod_cvd', 'cod_cancer', 'cod_ckd']].max(axis=1).sum()
qc_rows.append({
    'cause_category': 'Other',
    'n_deaths': int(n_other),
    'pct_of_all_deaths': round(n_other / len(cod_all) * 100, 1),
    'top_icd10_codes': ''
})

qc_df = pd.DataFrame(qc_rows)
qc_path = os.path.join(OUTPUT_DIR, "cod_qc_summary.txt")
qc_df.to_csv(qc_path, sep='\t', index=False)
print(f"  QC summary saved to: {qc_path}")
print(f"\n{qc_df.to_string(index=False)}")


# ============================================================
# STEP 3: MERGE INTO FINAL DATASET
# ============================================================

print("\n" + "=" * 60)
print("MERGING INTO FINAL DATASET")
print("=" * 60)

print(f"  Loading: {FINAL_DATASET}")
final = pd.read_csv(FINAL_DATASET, sep='\t', low_memory=False)
print(f"  Rows: {len(final):,}")
print(f"  Columns: {len(final.columns)}")

# Ensure patid types match
final['patid'] = pd.to_numeric(final['patid'], errors='coerce')

# Check patid dtype consistency
print(f"\n  Final dataset patid dtype: {final['patid'].dtype}")
print(f"  COD patid dtype: {cod_all['patid'].dtype}")
print(f"  Final dataset patid sample: {final['patid'].head(3).tolist()}")
print(f"  COD patid sample: {cod_all['patid'].head(3).tolist()}")

# Prepare merge columns — only the flags we need
cod_merge = cod_all[['patid', 'cod_cvd', 'cod_cancer', 'cod_ckd']].copy()

# Check for existing cod columns in final dataset (don't overwrite)
existing_cod = [c for c in final.columns if c.startswith('cod_')]
if existing_cod:
    print(f"\n  WARNING: Final dataset already has cod columns: {existing_cod}")
    print(f"  These will be REPLACED with new values")
    final = final.drop(columns=existing_cod)

# Merge
final = final.merge(cod_merge, on='patid', how='left')

# Fill NaN with 0 (patients who didn't die, or died but not in ONS file)
for col in ['cod_cvd', 'cod_cancer', 'cod_ckd']:
    final[col] = final[col].fillna(0).astype(int)

# ============================================================
# STEP 4: SANITY CHECKS
# ============================================================

print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

# Check 1: cod flags should only be 1 for patients who actually died
death_col = None
for candidate in ['death_ons', 'dod_ons']:
    if candidate in final.columns:
        death_col = candidate
        break

if death_col:
    if final[death_col].dtype == object or final[death_col].dtype == 'string':
        # It's a date column — not-null means died
        died_mask = final[death_col].notna() & (final[death_col].astype(str).str.strip() != '')
    else:
        # It's binary
        died_mask = final[death_col].fillna(0).astype(int) == 1
    
    n_died = died_mask.sum()
    
    for col in ['cod_cvd', 'cod_cancer', 'cod_ckd']:
        n_flagged = final[col].sum()
        # How many have a cod flag but didn't die?
        n_alive_with_flag = ((final[col] == 1) & (~died_mask)).sum()
        
        print(f"  {col}: {n_flagged:,} flagged, {n_alive_with_flag} flagged but not dead")
        
        if n_alive_with_flag > 0:
            print(f"    WARNING: {n_alive_with_flag} patients have {col}=1 but no death recorded")
            print(f"    This likely means patid matched ONS but death not in main dataset")
            print(f"    Setting these to 0...")
            final.loc[(final[col] == 1) & (~died_mask), col] = 0
    
    # Summary
    n_with_any_cod = ((final['cod_cvd'] + final['cod_cancer'] + final['cod_ckd']) > 0).sum()
    print(f"\n  Patients who died: {n_died:,}")
    print(f"  Patients with cause-of-death flag: {n_with_any_cod:,}")
    print(f"  Match rate: {n_with_any_cod/n_died*100:.1f}%" if n_died > 0 else "  No deaths")
    print(f"  Deaths without CVD/Cancer/CKD cause: {n_died - n_with_any_cod:,}")

else:
    print("  WARNING: Could not find death indicator column (death_ons or dod_ons)")
    print("  Skipping alive-with-flag check")

# Check 2: Overall distribution
print(f"\n  Final cause-of-death distribution:")
print(f"    cod_cvd:    {final['cod_cvd'].sum():,} ({final['cod_cvd'].mean()*100:.2f}%)")
print(f"    cod_cancer: {final['cod_cancer'].sum():,} ({final['cod_cancer'].mean()*100:.2f}%)")
print(f"    cod_ckd:    {final['cod_ckd'].sum():,} ({final['cod_ckd'].mean()*100:.2f}%)")

# Check 3: No overlap between CVD and Cancer (should be impossible from underlying cause)
overlap_cvd_cancer = ((final['cod_cvd'] == 1) & (final['cod_cancer'] == 1)).sum()
if overlap_cvd_cancer > 0:
    print(f"\n  WARNING: {overlap_cvd_cancer} patients flagged as BOTH cvd AND cancer")
    print(f"  This shouldn't happen with underlying cause — check the ICD codes")

# ============================================================
# STEP 5: SAVE
# ============================================================

print("\n" + "=" * 60)
print("SAVING")
print("=" * 60)

final.to_csv(OUTPUT_FILE, sep='\t', index=False)
print(f"  Saved to: {OUTPUT_FILE}")
print(f"  Rows: {len(final):,}")
print(f"  Columns: {len(final.columns)}")
print(f"  New columns added: cod_cvd, cod_cancer, cod_ckd")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
