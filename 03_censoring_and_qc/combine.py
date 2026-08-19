import pandas as pd

# =============================================================================
# PATHS
# =============================================================================
gold_path = "/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/GOLD_with_censoring.txt"
aurum_path = "/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/Aurum_with_censoring.txt"
combined_output_path = "/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/Combined_GOLD_Aurum.txt"

# =============================================================================
# 1. READ DATA
# =============================================================================
print("="*60)
print("COMBINING GOLD AND AURUM")
print("="*60)

print("\nReading GOLD data...")
gold = pd.read_csv(gold_path, sep="\t", low_memory=False, quoting=3)
print(f"  Shape: {gold.shape}")

print("\nReading Aurum data...")
aurum = pd.read_csv(aurum_path, sep="\t", low_memory=False, quoting=3)
print(f"  Shape: {aurum.shape}")

# =============================================================================
# 2. ADD DATABASE COLUMN
# =============================================================================
print("\nAdding database identifier column...")
gold["database"] = "GOLD"
aurum["database"] = "AURUM"

# =============================================================================
# 3. HARMONIZE COLUMN NAMES
# =============================================================================
print("\nHarmonizing column names...")

# regenddate -> regend
if "regenddate" in aurum.columns and "regend" not in aurum.columns:
    aurum = aurum.rename(columns={"regenddate": "regend"})
    print("  Renamed 'regenddate' -> 'regend' in Aurum")

# triglycerides -> trigly (and date)
# (Align Aurum to GOLD naming)
if "triglycerides" in aurum.columns and "trigly" not in aurum.columns:
    aurum = aurum.rename(columns={"triglycerides": "trigly"})
    print("  Renamed 'triglycerides' -> 'trigly' in Aurum")

if "triglycerides_date" in aurum.columns and "trigly_date" not in aurum.columns:
    aurum = aurum.rename(columns={"triglycerides_date": "trigly_date"})
    print("  Renamed 'triglycerides_date' -> 'trigly_date' in Aurum")

# (Optional safety: if GOLD had triglycerides naming instead, unify the other way)
if "triglycerides" in gold.columns and "trigly" not in gold.columns:
    gold = gold.rename(columns={"triglycerides": "trigly"})
    print("  Renamed 'triglycerides' -> 'trigly' in GOLD")

if "triglycerides_date" in gold.columns and "trigly_date" not in gold.columns:
    gold = gold.rename(columns={"triglycerides_date": "trigly_date"})
    print("  Renamed 'triglycerides_date' -> 'trigly_date' in GOLD")

# Check unique columns
gold_only_cols = set(gold.columns) - set(aurum.columns)
aurum_only_cols = set(aurum.columns) - set(gold.columns)

if gold_only_cols:
    print(f"  Columns only in GOLD: {sorted(gold_only_cols)}")
if aurum_only_cols:
    print(f"  Columns only in Aurum: {sorted(aurum_only_cols)}")

# =============================================================================
# 4. COMBINE DATASETS
# =============================================================================
print("\nCombining datasets...")
combined = pd.concat([gold, aurum], ignore_index=True, sort=False)
print(f"  Combined shape: {combined.shape}")

# =============================================================================
# 5. SUMMARY STATS
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\nRows by database:")
print(combined["database"].value_counts().to_string())

print("\nNon-null counts for censoring columns:")
for col in ["regend", "tod", "lcd"]:
    if col in combined.columns:
        by_db = combined.groupby("database")[col].apply(lambda x: x.notna().sum())
        for db, count in by_db.items():
            total = (combined["database"] == db).sum()
            print(f"  {col} - {db}: {count:,} / {total:,} ({count/total*100:.1f}%)")

# =============================================================================
# 6. SAVE
# =============================================================================
print(f"\nSaving to: {combined_output_path}")
combined.to_csv(combined_output_path, sep="\t", index=False)
print("Done!")

print(f"""
=============================================================
OUTPUT
=============================================================
  File: {combined_output_path}
  Total rows: {len(combined):,}
  GOLD rows: {len(gold):,}
  Aurum rows: {len(aurum):,}
""")

