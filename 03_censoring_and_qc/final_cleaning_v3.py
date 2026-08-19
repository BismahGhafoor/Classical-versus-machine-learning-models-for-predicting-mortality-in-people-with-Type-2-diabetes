import pandas as pd
import zipfile
import os

# =============================================================================
# PATHS
# =============================================================================
# GOLD
gold_patient_zip = (
    "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/"
    "FZ_GOLD_All_Extract_Patient_001.zip"
)
gold_practice_zip = (
    "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/"
    "FZ_GOLD_All_Extract_Practice_001.zip"
)
gold_cleaned_data = (
    "/scratch/alice/b/bg205/28_02_GOLD/extracted_lab_data_1YEAR.txt"
)
gold_output_path = (
    "/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/GOLD_with_censoring.txt"
)

# Aurum

aurum_patient_dir = (
    "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/Aurum/Patient"
)
aurum_practice_dir = (
    "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/Aurum/Practice"
)
aurum_cleaned_data = (
    "/scratch/alice/b/bg205/01_03_AURUM/FINAL_Aurum_with_Tests_1year.txt"
)
aurum_output_path = (
    "/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/Aurum_with_censoring.txt"
)

# =============================================================================
# 1. PROCESS GOLD
# =============================================================================
print("="*60)
print("PROCESSING GOLD")
print("="*60)

# Read GOLD patient file from zip
print("\nReading GOLD patient file...")
with zipfile.ZipFile(gold_patient_zip, 'r') as z:
    file_list = z.namelist()
    txt_file = [f for f in file_list if f.endswith('.txt')][0]
    print(f"Reading: {txt_file}")
    
    with z.open(txt_file) as f:
        gold_patient = pd.read_csv(f, sep='\t', low_memory=False)

print(f"Shape: {gold_patient.shape}")
print(f"Columns: {gold_patient.columns.tolist()}")

# Extract only what we need: patid, regend, tod
cols_needed = ['patid']
if 'regend' in gold_patient.columns:
    cols_needed.append('regend')
if 'tod' in gold_patient.columns:
    cols_needed.append('tod')

print(f"\nExtracting columns: {cols_needed}")
gold_patient_subset = gold_patient[cols_needed].copy()

# Parse dates
for col in ['regend', 'tod']:
    if col in gold_patient_subset.columns:
        gold_patient_subset[col] = pd.to_datetime(
            gold_patient_subset[col], dayfirst=True, errors='coerce'
        )
        print(f"  {col}: {gold_patient_subset[col].notna().sum():,} non-null")

# -----------------------------------------------------------------------------
# Read GOLD practice file from zip (for lcd)
# -----------------------------------------------------------------------------
print("\nReading GOLD practice file...")
with zipfile.ZipFile(gold_practice_zip, 'r') as z:
    file_list = z.namelist()
    txt_file = [f for f in file_list if f.endswith('.txt')][0]
    print(f"Reading: {txt_file}")
    
    with z.open(txt_file) as f:
        gold_practice = pd.read_csv(f, sep='\t', low_memory=False)

print(f"Shape: {gold_practice.shape}")
print(f"Columns: {gold_practice.columns.tolist()}")

# Extract pracid and lcd
cols_needed_practice = ['pracid']
if 'lcd' in gold_practice.columns:
    cols_needed_practice.append('lcd')

print(f"\nExtracting columns: {cols_needed_practice}")
gold_practice_subset = gold_practice[cols_needed_practice].copy()

# Parse lcd date
if 'lcd' in gold_practice_subset.columns:
    gold_practice_subset['lcd'] = pd.to_datetime(
        gold_practice_subset['lcd'], dayfirst=True, errors='coerce'
    )
    print(f"  lcd: {gold_practice_subset['lcd'].notna().sum():,} non-null")

# Read cleaned GOLD data
print(f"\nReading cleaned GOLD data...")
gold_cleaned = pd.read_csv(gold_cleaned_data, sep='\t', low_memory=False)
print(f"Shape: {gold_cleaned.shape}")

# Derive pracid from patid (last 3 digits)
if 'pracid' not in gold_cleaned.columns:
    gold_cleaned['pracid'] = gold_cleaned['patid'].astype(str).str[-5:].astype(int)
    print(f"Derived pracid from patid (last 5 digits)")

# Merge patient data
print(f"\nMerging patient data...")
gold_merged = gold_cleaned.merge(gold_patient_subset, on='patid', how='left')
print(f"Merged shape: {gold_merged.shape}")

# Check patient merge
for col in ['regend', 'tod']:
    if col in gold_merged.columns:
        n = gold_merged[col].notna().sum()
        print(f"  {col} matched: {n:,} / {len(gold_merged):,} ({n/len(gold_merged)*100:.1f}%)")

# Merge practice data (for lcd)
print(f"\nMerging practice data (lcd)...")
gold_merged = gold_merged.merge(gold_practice_subset, on='pracid', how='left')
print(f"Merged shape: {gold_merged.shape}")

# Check practice merge
if 'lcd' in gold_merged.columns:
    n = gold_merged['lcd'].notna().sum()
    print(f"  lcd matched: {n:,} / {len(gold_merged):,} ({n/len(gold_merged)*100:.1f}%)")

# Save
print(f"\nSaving to: {gold_output_path}")
gold_merged.to_csv(gold_output_path, sep='\t', index=False)
print("GOLD complete!")

# =============================================================================
# PROCESS AURUM
# =============================================================================
print("="*60)
print("PROCESSING AURUM")
print("="*60)

# -----------------------------------------------------------------------------
# 1. Read all Aurum PATIENT files
# -----------------------------------------------------------------------------
print("\nReading Aurum patient files...")
aurum_patient_list = []
zip_files = sorted([f for f in os.listdir(aurum_patient_dir) if f.endswith('.zip')])
print(f"Found {len(zip_files)} patient zip files")

for zip_name in zip_files:
    zip_path = os.path.join(aurum_patient_dir, zip_name)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            txt_files = [f for f in z.namelist() if f.endswith('.txt')]
            
            if txt_files:
                with z.open(txt_files[0]) as f:
                    chunk = pd.read_csv(f, sep='\t', low_memory=False)
                    aurum_patient_list.append(chunk)
                    print(f"  {zip_name}: {len(chunk):,} rows")
    except Exception as e:
        print(f"  {zip_name}: Error - {e}")

# Concatenate patient data
print(f"\nConcatenating patient data...")
aurum_patient = pd.concat(aurum_patient_list, ignore_index=True)
print(f"Total patient shape: {aurum_patient.shape}")
print(f"Patient columns: {aurum_patient.columns.tolist()}")

# Extract only patid and regenddate
cols_needed_patient = ['patid']
if 'regenddate' in aurum_patient.columns:
    cols_needed_patient.append('regenddate')

print(f"\nExtracting patient columns: {cols_needed_patient}")
aurum_patient_subset = aurum_patient[cols_needed_patient].copy()

# Parse date
if 'regenddate' in aurum_patient_subset.columns:
    aurum_patient_subset['regenddate'] = pd.to_datetime(
        aurum_patient_subset['regenddate'], dayfirst=True, errors='coerce'
    )
    print(f"  regenddate: {aurum_patient_subset['regenddate'].notna().sum():,} non-null")

# -----------------------------------------------------------------------------
# 2. Read all Aurum PRACTICE files (for lcd)
# -----------------------------------------------------------------------------
print("\n" + "-"*40)
print("Reading Aurum practice files...")
aurum_practice_list = []
practice_zip_files = sorted([f for f in os.listdir(aurum_practice_dir) if f.endswith('.zip')])
print(f"Found {len(practice_zip_files)} practice zip files")

for zip_name in practice_zip_files:
    zip_path = os.path.join(aurum_practice_dir, zip_name)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            txt_files = [f for f in z.namelist() if f.endswith('.txt')]
            
            if txt_files:
                with z.open(txt_files[0]) as f:
                    chunk = pd.read_csv(f, sep='\t', low_memory=False)
                    aurum_practice_list.append(chunk)
                    print(f"  {zip_name}: {len(chunk):,} rows")
    except Exception as e:
        print(f"  {zip_name}: Error - {e}")

# Concatenate practice data
print(f"\nConcatenating practice data...")
aurum_practice = pd.concat(aurum_practice_list, ignore_index=True)
print(f"Total practice shape: {aurum_practice.shape}")
print(f"Practice columns: {aurum_practice.columns.tolist()}")

# Remove duplicates (same practice may appear in multiple files)
aurum_practice = aurum_practice.drop_duplicates(subset=['pracid'])
print(f"After deduplication: {len(aurum_practice):,} unique practices")

# Extract pracid and lcd
cols_needed_practice = ['pracid']
if 'lcd' in aurum_practice.columns:
    cols_needed_practice.append('lcd')

print(f"\nExtracting practice columns: {cols_needed_practice}")
aurum_practice_subset = aurum_practice[cols_needed_practice].copy()

# Parse lcd date
if 'lcd' in aurum_practice_subset.columns:
    aurum_practice_subset['lcd'] = pd.to_datetime(
        aurum_practice_subset['lcd'], dayfirst=True, errors='coerce'
    )
    print(f"  lcd: {aurum_practice_subset['lcd'].notna().sum():,} non-null")

# -----------------------------------------------------------------------------
# 3. Read cleaned Aurum data and merge
# -----------------------------------------------------------------------------
print("\n" + "-"*40)
print(f"Reading cleaned Aurum data...")
aurum_cleaned = pd.read_csv(aurum_cleaned_data, sep='\t', low_memory=False)
print(f"Shape: {aurum_cleaned.shape}")

# Derive pracid from patid (last 5 digits)
if 'pracid' not in aurum_cleaned.columns:
    aurum_cleaned['pracid'] = aurum_cleaned['patid'].astype(str).str[-5:].astype(int)
    print(f"Derived pracid from patid (last 5 digits)")

# Merge patient data (regenddate)
print(f"\nMerging patient data...")
aurum_merged = aurum_cleaned.merge(aurum_patient_subset, on='patid', how='left')
print(f"Merged shape: {aurum_merged.shape}")

if 'regenddate' in aurum_merged.columns:
    n = aurum_merged['regenddate'].notna().sum()
    print(f"  regenddate matched: {n:,} / {len(aurum_merged):,} ({n/len(aurum_merged)*100:.1f}%)")

# Merge practice data (lcd)
print(f"\nMerging practice data (lcd)...")
aurum_merged = aurum_merged.merge(aurum_practice_subset, on='pracid', how='left')
print(f"Merged shape: {aurum_merged.shape}")

if 'lcd' in aurum_merged.columns:
    n = aurum_merged['lcd'].notna().sum()
    print(f"  lcd matched: {n:,} / {len(aurum_merged):,} ({n/len(aurum_merged)*100:.1f}%)")

# -----------------------------------------------------------------------------
# 4. Save
# -----------------------------------------------------------------------------
print(f"\nSaving to: {aurum_output_path}")
aurum_merged.to_csv(aurum_output_path, sep='\t', index=False)
print("Aurum complete!")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
GOLD:
  - Rows: {len(gold_merged):,}
  - Columns added: regend, tod, lcd, pracid
  - Saved to: {gold_output_path}
  
AURUM:
  - Rows: {len(aurum_merged):,}
  - Columns added: regenddate, lcd, pracid
  - Unique practices: {aurum_merged['pracid'].nunique():,}
  - regenddate non-null: {aurum_merged['regenddate'].notna().sum():,} ({aurum_merged['regenddate'].notna().mean()*100:.1f}%)
  - lcd non-null: {aurum_merged['lcd'].notna().sum():,} ({aurum_merged['lcd'].notna().mean()*100:.1f}%)
  - Saved to: {aurum_output_path}
""")
