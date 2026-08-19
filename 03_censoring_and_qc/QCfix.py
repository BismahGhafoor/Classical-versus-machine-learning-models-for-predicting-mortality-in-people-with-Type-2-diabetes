#!/usr/bin/env python3
"""
fix_qc_issues.py — Fix the 3 QC failures + gender harmonisation
"""
import pandas as pd
import numpy as np

FILE = "/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/Combined_GOLD_Aurum_with_meds_comorbidities_studyend_clean.txt"
SEP  = "\t"

print("Loading...")
df = pd.read_csv(FILE, sep=SEP, low_memory=False)
df["indexdate"]    = pd.to_datetime(df["indexdate"], errors="coerce")
df["censor_date"]  = pd.to_datetime(df["censor_date"], errors="coerce")
df["dod_ons"]      = pd.to_datetime(df["dod_ons"], errors="coerce")
print(f"  Loaded: {len(df):,} rows\n")

# ============================================================
# FIX 1: Harmonise gender (Male/Female → M/F, or vice versa)
# ============================================================
print("Fix 1: Harmonising gender...")
before = df["gender"].value_counts().to_dict()
df["gender"] = df["gender"].replace({"Male": "M", "Female": "F"})
after = df["gender"].value_counts().to_dict()
print(f"  Before: {before}")
print(f"  After:  {after}\n")

# ============================================================
# FIX 2: Remove patients with death before indexdate
# ============================================================
print("Fix 2: Removing patients with death_date < indexdate...")
death_before_index = df["dod_ons"].notna() & (df["dod_ons"] < df["indexdate"])
n_remove = death_before_index.sum()
df = df[~death_before_index].copy()
print(f"  Removed: {n_remove:,} patients\n")

# ============================================================
# FIX 3: Flag implausible YOB (before 1900)
# ============================================================
print("Fix 3: Checking implausible YOB...")
df["yob"] = pd.to_numeric(df["yob"], errors="coerce")
pre1900 = (df["yob"] < 1900).sum()
print(f"  Patients with YOB < 1900: {pre1900:,}")
# Remove them — can't have a valid age
df = df[df["yob"] >= 1900].copy()
print(f"  Removed: {pre1900:,} patients\n")

# ============================================================
# SUMMARY
# ============================================================
n_final = len(df)
n_gold  = (df["database"].str.upper() == "GOLD").sum()
n_aurum = (df["database"].str.upper() == "AURUM").sum()

print(f"{'='*60}")
print(f"  FINAL COHORT")
print(f"{'='*60}")
print(f"  Total:  {n_final:,}")
print(f"  GOLD:   {n_gold:,}")
print(f"  AURUM:  {n_aurum:,}")
print(f"  Gender: {df['gender'].value_counts().to_dict()}")
print(f"  YOB:    {df['yob'].min():.0f} – {df['yob'].max():.0f}")
print(f"  Max indexdate: {df['indexdate'].max()}")

# Check the 609 issue is resolved
bad_censor = df["censor_date"].notna() & (df["censor_date"] < df["indexdate"])
bad_death  = df["dod_ons"].notna() & (df["dod_ons"] < df["indexdate"])
print(f"  Censor < index violations: {bad_censor.sum()}")
print(f"  Death < index violations:  {bad_death.sum()}")

# Save
out = FILE.replace("_studyend.txt", "_studyend_clean.txt")
df.to_csv(out, sep=SEP, index=False)
print(f"\n  Saved → {out}")
print(f"  Done.")
