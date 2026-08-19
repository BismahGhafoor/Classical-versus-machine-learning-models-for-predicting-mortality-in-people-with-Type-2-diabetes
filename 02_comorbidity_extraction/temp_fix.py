#!/usr/bin/env python3
"""
Quick fix: apply MIN_EVENT_DATE = 1900-01-01 to the existing
final_combined_comorbidities.txt without re-running the full extraction.
"""
import pandas as pd
import numpy as np

INPUT_FILE  = "/scratch/alice/b/bg205/16_02_26/comorbidityV2/final_combined_comorbidities.txt"
OUTPUT_FILE = "/scratch/alice/b/bg205/16_02_26/comorbidityV2/final_combined_comorbidities_fixed.txt"

MIN_EVENT_DATE = pd.Timestamp("1900-01-01")

print("Loading...")
df = pd.read_csv(INPUT_FILE, sep="\t", low_memory=False)
print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")

# Parse indexdate
df["indexdate"] = pd.to_datetime(df["indexdate"], errors="coerce")

# Find all first_date columns
first_date_cols = [c for c in df.columns if c.endswith("_first_date")]
print(f"  First-date columns found: {first_date_cols}")

for col in first_date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

    # Count pre-1900
    pre1900 = (df[col] < MIN_EVENT_DATE).sum()
    print(f"  {col}: {pre1900:,} pre-1900 dates → setting to NaT")

    # Fix
    df.loc[df[col] < MIN_EVENT_DATE, col] = pd.NaT

    # Derive the condition name
    cond = col.replace("_first_date", "")
    bin_col  = f"{cond}_bin"
    days_col = f"{cond}_duration_days"
    years_col = f"{cond}_duration_years"

    # Recalculate binary flag (patient might lose their only record)
    if bin_col in df.columns:
        df[bin_col] = df[col].notna().astype(int)

    # Recalculate duration
    if days_col in df.columns:
        df[days_col] = np.where(
            df[col].notna(),
            (df["indexdate"] - df[col]).dt.days,
            np.nan,
        )
    if years_col in df.columns:
        df[years_col] = df[days_col] / 365.25 if days_col in df.columns else np.nan

# --- QC ---
print(f"\n  QC after fix:")
for col in first_date_cols:
    cond = col.replace("_first_date", "")
    bin_col = f"{cond}_bin"
    if bin_col in df.columns:
        n = int(df[bin_col].sum())
        pct = 100 * n / len(df)
        fmin = df[col].min()
        fmax = df[col].max()
        print(f"    {cond:<25} n={n:>8,} ({pct:5.1f}%)  dates: [{fmin} – {fmax}]")

# Check for negative durations
print(f"\n  Negative-duration check:")
any_neg = False
for col in first_date_cols:
    cond = col.replace("_first_date", "")
    days_col = f"{cond}_duration_days"
    if days_col in df.columns:
        neg = (df[days_col] < 0).sum()
        if neg > 0:
            print(f"    WARNING — {cond}: {neg:,} negative durations!")
            any_neg = True
if not any_neg:
    print(f"    PASS")

# Save
df.to_csv(OUTPUT_FILE, sep="\t", index=False)
print(f"\n  Saved → {OUTPUT_FILE}")
print("  Done.")
