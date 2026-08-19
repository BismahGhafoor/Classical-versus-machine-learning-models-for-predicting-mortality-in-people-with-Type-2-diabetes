#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AURUM: Build baseline risk factors (smoking + BMI + BP) at INDEXDATE
- Smoking: most recent record within 1 year on/before indexdate (CPRD priority, HES fallback)
- BMI:
    1) recorded BMI within 1 year on/before indexdate, filter 10–80
    2) fallback calculated BMI from weight+height on same obsdate within 1 year, filter 10–80
- BP: paired SBP+DBP on same obsdate within 1 year on/before indexdate, filter SBP 40–250, DBP 20–200
- Dates outside [MIN_DATE, MAX_DATE] are dropped (prevents 1860/2099 junk)
- Uses STUDY_END only to cap MAX_DATE if you want; censoring is NOT created here (that belongs to outcome script).
Outputs: /scratch/alice/b/bg205/DataCleaning_Aurum_v2/Enriched_Aurum_with_Biomarkers.txt
"""

import os
import glob
import time
import platform
import sys
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np

# =============================================================================
# Logging helpers
# =============================================================================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def dbg(tag: str, msg: str) -> None:
    print(f"DBG| [{tag}] {msg}", flush=True)

def dbg_date_col(df: pd.DataFrame, col: str, tag: str) -> None:
    s = pd.to_datetime(df[col], errors="coerce")
    missing = s.isna().sum()
    valid = s.dropna()
    if len(valid) == 0:
        dbg(tag, f"  - {col}: missing={missing} (ALL missing, no valid dates)")
        return
    years = valid.dt.year
    dbg(tag, f"  - {col}: missing={missing}")
    dbg(tag, f"    year: min={int(years.min())} p50={int(years.median())} max={int(years.max())} "
             f">2025={int((years > 2025).sum())} <1900={int((years < 1900).sum())} ==9999={int((years == 9999).sum())}")

def dbg_df(df: pd.DataFrame, tag: str, patid_col: str = "patid") -> None:
    n_patids = df[patid_col].nunique() if patid_col in df.columns else "N/A"
    dbg(tag, f"rows={len(df):,}  patids={n_patids:,}")

def dbg_value_stats(series: pd.Series, tag: str, label: str) -> None:
    """Print numeric value distribution stats."""
    v = pd.to_numeric(series, errors="coerce")
    valid = v.dropna()
    if len(valid) == 0:
        dbg(tag, f"{label}: ALL missing/non-numeric")
        return
    dbg(tag, f"{label}: n={len(valid):,}  missing={v.isna().sum():,}  "
             f"min={valid.min():.2f}  p25={valid.quantile(0.25):.2f}  "
             f"median={valid.median():.2f}  p75={valid.quantile(0.75):.2f}  "
             f"max={valid.max():.2f}")

# --------------------------
# Paths
# --------------------------
ROOT = "/scratch/alice/b/bg205/01_03_AURUM"

BIOMARKER_CHUNK_DIR = os.path.join(ROOT, "biomarker_chunks")
SMOKING_FILE = os.path.join(ROOT, "smoking_chunks", "Aurum_smoking_records_all.txt.gz")
DEMOG_FILE = os.path.join(ROOT, "Enriched_baseline_with_demographics.txt")
HES_FILE = os.path.join(ROOT, "linkage", "hes_diagnosis_hosp_23_002869_DM.txt")

CODES_DIR = "/scratch/alice/b/bg205/DataCleaning_Aurum_v2/Codes"
SMOKING_CODES_DIR = os.path.join(CODES_DIR, "smoking_CSV_exports")

OUT_FILE = os.path.join(ROOT, "Enriched_Aurum_with_Biomarkers_3year.txt")

# --------------------------
# Parameters (your rules)
# --------------------------
WINDOW_DAYS = 1095   # 3 year lookback
# WINDOW_DAYS = 365   # 1 year lookback
STUDY_END = pd.Timestamp("2021-03-29")  # used only as max date cap
MIN_DATE = pd.Timestamp("1900-01-01")
MAX_DATE = pd.Timestamp("2024-12-31")  # change if you intentionally want later values

BMI_MIN, BMI_MAX = 10.0, 80.0
SBP_MIN, SBP_MAX = 40.0, 250.0
DBP_MIN, DBP_MAX = 20.0, 200.0

HES_CHUNKSIZE = 500_000
SMOKE_CHUNKSIZE = 2_000_000
OBS_CHUNKSIZE = 300_000

t0 = time.time()
log(f"Host: {platform.node()}")
log(f"CWD: {os.getcwd()}")
log(f"Python: {sys.executable} {platform.python_version()}")
dbg("CONFIG", f"WINDOW_DAYS={WINDOW_DAYS}  STUDY_END={STUDY_END}  MIN_DATE={MIN_DATE}  MAX_DATE={MAX_DATE}")
dbg("CONFIG", f"BMI range=[{BMI_MIN}, {BMI_MAX}]  SBP range=[{SBP_MIN}, {SBP_MAX}]  DBP range=[{DBP_MIN}, {DBP_MAX}]")
dbg("CONFIG", f"SMOKE_CHUNKSIZE={SMOKE_CHUNKSIZE}  HES_CHUNKSIZE={HES_CHUNKSIZE}  OBS_CHUNKSIZE={OBS_CHUNKSIZE}")

# --------------------------
# Helpers
# --------------------------
def load_codes(filename: str, base_dir: str) -> set:
    """Load medcodeids from CPRD-exported CSVs that may have 2 header rows."""
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Code file not found: {path}")

    for skip in (2, 0):
        df = pd.read_csv(path, skiprows=skip, dtype=str)
        cols = df.columns.str.lower().str.strip()
        medcode_cols = [c for c in cols if "medcode" in c]
        if medcode_cols:
            colname = df.columns[cols.get_loc(medcode_cols[0])]
            codes_set = set(df[colname].dropna().astype(str).str.strip())
            dbg("LOAD_CODES", f"{filename}: skiprows={skip}  unique_medcodes={len(codes_set)}")
            return codes_set
    raise ValueError(f"No column containing 'medcode' in {filename}")

def parse_date(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def enforce_date_bounds(df, date_col):
    before = len(df)
    dt = df[date_col]
    df = df[(dt.notna()) & (dt >= MIN_DATE) & (dt <= MAX_DATE)]
    after = len(df)
    if before - after > 0:
        dbg("DATE_BOUNDS", f"  {date_col}: dropped {before - after:,} rows outside [{MIN_DATE.date()}, {MAX_DATE.date()}]  "
                           f"kept={after:,}")
    return df

def best_within_lookback(current_best: pd.DataFrame, new_rows: pd.DataFrame, date_col: str, value_cols: list):
    """
    Keep most recent record on/before indexdate within lookback window.
    Comparison key: (patid, indexdate). Ranking: smallest days_before.
    """
    if new_rows.empty:
        return current_best

    df = new_rows.dropna(subset=["patid", "indexdate", date_col]).copy()
    df = df[df[date_col] <= df["indexdate"]]
    df["days_before"] = (df["indexdate"] - df[date_col]).dt.days
    df = df[(df["days_before"] >= 0) & (df["days_before"] <= WINDOW_DAYS)]
    if df.empty:
        return current_best

    keep_cols = ["patid", "indexdate", date_col, "days_before"] + value_cols
    df = df[keep_cols]

    if current_best is None or current_best.empty:
        return df.sort_values(["patid", "indexdate", "days_before"]).drop_duplicates(["patid", "indexdate"], keep="first")

    comb = pd.concat([current_best, df], ignore_index=True)
    comb = comb.sort_values(["patid", "indexdate", "days_before"]).drop_duplicates(["patid", "indexdate"], keep="first")
    return comb

def list_chunk_files(prefix: str) -> list:
    pat_tsv = os.path.join(BIOMARKER_CHUNK_DIR, f"{prefix}_chunk_*.txt.gz")
    files = sorted(glob.glob(pat_tsv))
    if not files:
        raise FileNotFoundError(f"No files found for {pat_tsv}")
    dbg("CHUNK_FILES", f"{prefix}: found {len(files)} files  "
                       f"first={os.path.basename(files[0])}  last={os.path.basename(files[-1])}")
    return files

# --------------------------
# Load baseline/demographics
# --------------------------
log("Loading demographics/baseline...")
demog = pd.read_csv(DEMOG_FILE, sep="\t", dtype=str)
demog["patid"] = demog["patid"].astype(str)
demog["indexdate"] = pd.to_datetime(demog["indexdate"], errors="coerce", dayfirst=False)
if demog["indexdate"].isna().any():
    raise ValueError(f"Found missing indexdate in demog: {demog['indexdate'].isna().sum()}")

dbg_df(demog, "DEMOG_LOADED")
dbg("DEMOG_LOADED", f"columns={list(demog.columns)}")
dbg_date_col(demog, "indexdate", "DEMOG_LOADED")

# We do NOT create censoring/outcome here; those are handled in outcome script.
for c in ["dod_ons", "death_ons", "censor_date"]:
    if c in demog.columns:
        if c in ["dod_ons", "censor_date"]:
            demog[c] = pd.to_datetime(demog[c], errors="coerce", dayfirst=False)

index_map = demog[["patid", "indexdate"]].drop_duplicates("patid")
log(f"Cohort size: {len(index_map):,}")
dbg("INDEX_MAP", f"rows={len(index_map):,}  unique_patids={index_map['patid'].nunique():,}")
dbg_date_col(index_map, "indexdate", "INDEX_MAP")

# Check for duplicate patids in demog
dup_patids = demog["patid"].duplicated().sum()
if dup_patids > 0:
    dbg("DEMOG_WARNING", f"DUPLICATE patids in demog: {dup_patids:,}")
else:
    dbg("DEMOG_OK", "no duplicate patids in demog (good)")

# ===========================================================================
# SMOKING
# ===========================================================================
log("Building smoking status (CPRD priority, 1y pre-index)...")
smoke_start = time.time()

current_codes = load_codes("Current_smoker.csv", SMOKING_CODES_DIR)
ex_codes      = load_codes("Ex-smoker.csv",      SMOKING_CODES_DIR)
never_codes   = load_codes("Never_smoked.csv",   SMOKING_CODES_DIR)

# Check overlap between smoking code sets
for name_a, set_a, name_b, set_b in [
    ("Current", current_codes, "Ex", ex_codes),
    ("Current", current_codes, "Never", never_codes),
    ("Ex", ex_codes, "Never", never_codes),
]:
    overlap = set_a & set_b
    if overlap:
        dbg("SMOKE_CODES_WARNING", f"OVERLAP {name_a} vs {name_b}: {len(overlap)} shared medcodes  examples={sorted(overlap)[:5]}")
    else:
        dbg("SMOKE_CODES_OK", f"no overlap {name_a} vs {name_b}")

dbg("SMOKE_CODES", f"Current={len(current_codes)}  Ex={len(ex_codes)}  Never={len(never_codes)}  "
                    f"total_unique={len(current_codes | ex_codes | never_codes)}")

if not os.path.exists(SMOKING_FILE):
    raise FileNotFoundError(f"Smoking file not found: {SMOKING_FILE}")
dbg("SMOKE_FILE", f"file={SMOKING_FILE}  size_bytes={os.path.getsize(SMOKING_FILE):,}")

smoke_best = None
smoke_chunks_read = 0
smoke_rows_read = 0
smoke_rows_merged = 0
smoke_rows_date_ok = 0
smoke_rows_classified = 0
smoke_status_counts = Counter()

for chunk in pd.read_csv(SMOKING_FILE, sep="\t", dtype=str, compression="gzip", chunksize=SMOKE_CHUNKSIZE):
    smoke_chunks_read += 1
    smoke_rows_read += len(chunk)

    chunk["patid"] = chunk["patid"].astype(str)
    chunk = chunk.merge(index_map, on="patid", how="inner")
    if chunk.empty:
        continue
    smoke_rows_merged += len(chunk)

    chunk["smoking_date"] = parse_date(chunk["obsdate"])
    chunk = chunk.drop(columns=["obsdate"], errors="ignore")
    chunk = enforce_date_bounds(chunk, "smoking_date")
    if chunk.empty:
        continue
    smoke_rows_date_ok += len(chunk)

    # ----- FIXED smoking status assignment (NO numpy where) -----
    mc = chunk["medcodeid"].astype(str).str.strip()
    chunk["smoking_status"] = pd.NA
    chunk.loc[mc.isin(current_codes), "smoking_status"] = "Yes"
    chunk.loc[mc.isin(ex_codes),      "smoking_status"] = "Ex"
    chunk.loc[mc.isin(never_codes),   "smoking_status"] = "No"

    unclassified = chunk["smoking_status"].isna().sum()
    if unclassified > 0:
        dbg("SMOKE_UNCLASSIFIED", f"chunk {smoke_chunks_read}: {unclassified:,} rows with medcode not in any smoking set")

    chunk = chunk.dropna(subset=["smoking_status"])
    smoke_rows_classified += len(chunk)
    # count statuses in this chunk
    for status in chunk["smoking_status"].value_counts().items():
        smoke_status_counts[status[0]] += status[1]
    # ------------------------------------------------------------

    smoke_best = best_within_lookback(
        smoke_best,
        chunk[["patid", "indexdate", "smoking_date", "smoking_status"]],
        date_col="smoking_date",
        value_cols=["smoking_status"]
    )

    if smoke_chunks_read % 5 == 0:
        dbg("SMOKE_PROGRESS", f"chunks={smoke_chunks_read}  rows_read={smoke_rows_read:,}  "
                              f"merged={smoke_rows_merged:,}  date_ok={smoke_rows_date_ok:,}  "
                              f"classified={smoke_rows_classified:,}  "
                              f"best_size={0 if smoke_best is None else len(smoke_best):,}")

cprd_smoke_count = 0 if smoke_best is None else len(smoke_best)
log(f"CPRD smoking best rows: {cprd_smoke_count:,}")
dbg("SMOKE_CPRD_DONE", f"chunks_read={smoke_chunks_read}  total_rows_read={smoke_rows_read:,}")
dbg("SMOKE_CPRD_DONE", f"merged_to_cohort={smoke_rows_merged:,}  date_valid={smoke_rows_date_ok:,}  "
                        f"classified={smoke_rows_classified:,}")
dbg("SMOKE_CPRD_DONE", f"status distribution across ALL chunks: {dict(smoke_status_counts)}")

if smoke_best is not None and not smoke_best.empty:
    dbg("SMOKE_CPRD_BEST", f"smoking_status value_counts:\n{smoke_best['smoking_status'].value_counts(dropna=False).to_string()}")
    dbg_date_col(smoke_best, "smoking_date", "SMOKE_CPRD_BEST")
    dbg("SMOKE_CPRD_BEST", f"days_before stats: min={smoke_best['days_before'].min()}  "
                            f"median={int(smoke_best['days_before'].median())}  "
                            f"max={smoke_best['days_before'].max()}")
    dbg("SMOKE_CPRD_BEST", f"coverage: {cprd_smoke_count:,}/{len(index_map):,} = "
                            f"{100*cprd_smoke_count/len(index_map):.2f}%")

# HES fallback: only for those missing CPRD smoking
log("Adding HES fallback smoking (current smoker ICDs)...")
if smoke_best is None:
    smoke_best = pd.DataFrame(columns=["patid","indexdate","smoking_date","days_before","smoking_status"])

have_smoke = set(smoke_best["patid"].astype(str).tolist())
need_smoke = len(index_map) - len(have_smoke)
dbg("SMOKE_HES_START", f"patients with CPRD smoking={len(have_smoke):,}  "
                        f"patients needing HES fallback={need_smoke:,}")

tmp_head = pd.read_csv(HES_FILE, sep="\t", dtype=str, nrows=5)
dbg("HES_FILE", f"columns={list(tmp_head.columns)}")

icd_col = None
for cand in ["ICD", "diag_icd10", "diagcode", "icd", "icd10"]:
    if cand in tmp_head.columns:
        icd_col = cand
        break
if icd_col is None:
    raise KeyError("No ICD column found in HES file. Expected one of: ICD, diag_icd10, diagcode, icd, icd10")
dbg("HES_FILE", f"using ICD column: '{icd_col}'")

def is_current_icd(x: str) -> bool:
    if pd.isna(x):
        return False
    x = str(x).strip().upper()
    return x.startswith("F17") or x == "Z72.0" or x == "T65.2"

hes_best = None
hes_chunks_read = 0
hes_rows_read = 0
hes_rows_filtered = 0
hes_rows_icd_match = 0

for chunk in pd.read_csv(HES_FILE, sep="\t", dtype=str, chunksize=HES_CHUNKSIZE):
    hes_chunks_read += 1
    hes_rows_read += len(chunk)

    chunk["patid"] = chunk["patid"].astype(str)
    chunk = chunk[~chunk["patid"].isin(have_smoke)]
    if chunk.empty:
        continue
    chunk = chunk.merge(index_map, on="patid", how="inner")
    if chunk.empty:
        continue
    hes_rows_filtered += len(chunk)

    if "admidate" not in chunk.columns:
        dbg("HES_WARNING", f"chunk {hes_chunks_read}: no 'admidate' column, skipping")
        continue
    chunk["smoking_date"] = parse_date(chunk["admidate"])
    chunk = enforce_date_bounds(chunk, "smoking_date")
    if chunk.empty:
        continue

    icd = chunk[icd_col].astype(str)
    keep = icd.map(is_current_icd)
    chunk = chunk[keep]
    if chunk.empty:
        continue
    hes_rows_icd_match += len(chunk)

    chunk["smoking_status"] = "Yes"
    hes_best = best_within_lookback(
        hes_best,
        chunk[["patid","indexdate","smoking_date","smoking_status"]],
        date_col="smoking_date",
        value_cols=["smoking_status"]
    )

    if hes_chunks_read % 10 == 0:
        dbg("SMOKE_HES_PROGRESS", f"chunks={hes_chunks_read}  rows_read={hes_rows_read:,}  "
                                   f"filtered={hes_rows_filtered:,}  icd_match={hes_rows_icd_match:,}  "
                                   f"best_size={0 if hes_best is None else len(hes_best):,}")

hes_smoke_count = 0 if hes_best is None else len(hes_best)
dbg("SMOKE_HES_DONE", f"chunks_read={hes_chunks_read}  total_rows={hes_rows_read:,}  "
                       f"filtered_to_cohort={hes_rows_filtered:,}  icd_match={hes_rows_icd_match:,}")
dbg("SMOKE_HES_DONE", f"HES smoking fallback rows: {hes_smoke_count:,}")

if hes_best is not None and not hes_best.empty:
    hes_best = hes_best.drop(columns=["days_before"], errors="ignore")
    smoke_best2 = smoke_best.drop(columns=["days_before"], errors="ignore")
    smoke_best = pd.concat([smoke_best2, hes_best], ignore_index=True)
    smoke_best = smoke_best.drop_duplicates(["patid","indexdate"], keep="first")
else:
    smoke_best = smoke_best.drop(columns=["days_before"], errors="ignore")

log(f"Final smoking rows: {len(smoke_best):,}")
dbg("SMOKE_FINAL", f"smoking_status value_counts:\n{smoke_best['smoking_status'].value_counts(dropna=False).to_string()}")
dbg("SMOKE_FINAL", f"source: CPRD={cprd_smoke_count:,}  HES_fallback={hes_smoke_count:,}  "
                    f"total={len(smoke_best):,}")
dbg("SMOKE_FINAL", f"coverage: {len(smoke_best):,}/{len(index_map):,} = "
                    f"{100*len(smoke_best)/len(index_map):.2f}%")
if not smoke_best.empty:
    dbg_date_col(smoke_best, "smoking_date", "SMOKE_FINAL")

smoke_elapsed = time.time() - smoke_start
log(f"Smoking section elapsed: {smoke_elapsed/60:.1f} min")

demog = demog.merge(smoke_best, on=["patid","indexdate"], how="left")
dbg("SMOKE_MERGE", f"demog rows after merge: {len(demog):,}  (should be {len(index_map):,})")
dbg("SMOKE_MERGE", f"smoking_status in demog:\n{demog['smoking_status'].value_counts(dropna=False).to_string()}")

# ===========================================================================
# BMI (recorded first, then calculated fallback using recent weight + any prior height)
# ===========================================================================
log("Building BMI (recorded priority, 3y pre-index)...")
bmi_start = time.time()

# --------------------------
# Helper for height: most recent valid height on/before indexdate, no lookback restriction
# --------------------------
def most_recent_before_index(current_best: pd.DataFrame, new_rows: pd.DataFrame, date_col: str, value_cols: list):
    """
    Keep most recent valid record on/before indexdate, with NO lookback restriction.
    Used for height.
    """
    if new_rows.empty:
        return current_best

    df = new_rows.dropna(subset=["patid", "indexdate", date_col]).copy()
    df = df[df[date_col] <= df["indexdate"]]
    if df.empty:
        return current_best

    df["days_before"] = (df["indexdate"] - df[date_col]).dt.days
    keep_cols = ["patid", "indexdate", date_col, "days_before"] + value_cols
    df = df[keep_cols]

    if current_best is None or current_best.empty:
        return df.sort_values(["patid", "indexdate", "days_before"]).drop_duplicates(["patid", "indexdate"], keep="first")

    comb = pd.concat([current_best, df], ignore_index=True)
    comb = comb.sort_values(["patid", "indexdate", "days_before"]).drop_duplicates(["patid", "indexdate"], keep="first")
    return comb

# --------------------------
# 1) Recorded BMI within lookback
# --------------------------
bmi_best = None
bmi_chunks_read = 0
bmi_rows_read = 0
bmi_rows_merged = 0
bmi_rows_valid = 0
bmi_rows_in_range = 0

bmi_files = list_chunk_files("bmi")
dbg("BMI_FILES", f"total files={len(bmi_files)}")

for fi, fp in enumerate(bmi_files, start=1):
    for chunk in pd.read_csv(fp, sep="\t", dtype=str, compression="gzip", chunksize=OBS_CHUNKSIZE):
        bmi_chunks_read += 1
        bmi_rows_read += len(chunk)

        chunk["patid"] = chunk["patid"].astype(str)
        chunk = chunk.merge(index_map, on="patid", how="inner")
        if chunk.empty:
            continue
        bmi_rows_merged += len(chunk)

        chunk["obsdate"] = parse_date(chunk["obsdate"])
        chunk = enforce_date_bounds(chunk, "obsdate")
        if chunk.empty:
            continue

        chunk["bmi"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk.dropna(subset=["bmi"])
        bmi_rows_valid += len(chunk)

        chunk = chunk[(chunk["bmi"] >= BMI_MIN) & (chunk["bmi"] <= BMI_MAX)]
        if chunk.empty:
            continue
        bmi_rows_in_range += len(chunk)

        bmi_best = best_within_lookback(
            bmi_best,
            chunk[["patid", "indexdate", "obsdate", "bmi"]],
            date_col="obsdate",
            value_cols=["bmi"]
        )

    if fi % 50 == 0:
        dbg("BMI_REC_PROGRESS", f"files={fi}/{len(bmi_files)}  chunks={bmi_chunks_read}  "
                                 f"rows_read={bmi_rows_read:,}  merged={bmi_rows_merged:,}  "
                                 f"valid={bmi_rows_valid:,}  in_range={bmi_rows_in_range:,}  "
                                 f"best_size={0 if bmi_best is None else len(bmi_best):,}")

if bmi_best is None:
    bmi_best = pd.DataFrame(columns=["patid", "indexdate", "obsdate", "days_before", "bmi"])

recorded_bmi_count = len(bmi_best)
bmi_best = bmi_best.rename(columns={"obsdate": "bmi_date"}).drop(columns=["days_before"], errors="ignore")
bmi_best["bmi_source"] = "recorded"

log(f"Recorded BMI rows: {recorded_bmi_count:,}")
dbg("BMI_RECORDED_DONE", f"chunks={bmi_chunks_read}  rows_read={bmi_rows_read:,}  "
                          f"merged={bmi_rows_merged:,}  valid_numeric={bmi_rows_valid:,}  "
                          f"in_range=[{BMI_MIN},{BMI_MAX}]={bmi_rows_in_range:,}")
dbg("BMI_RECORDED_DONE", f"coverage: {recorded_bmi_count:,}/{len(index_map):,} = "
                          f"{100*recorded_bmi_count/len(index_map):.2f}%")
if recorded_bmi_count > 0:
    dbg_value_stats(bmi_best["bmi"], "BMI_RECORDED_BEST", "bmi")
    dbg_date_col(bmi_best, "bmi_date", "BMI_RECORDED_BEST")

# --------------------------
# 2) Most recent valid weight within lookback
# --------------------------
log("Building calculated BMI fallback using recent weight + most recent historical height...")

weight_files = list_chunk_files("weight")
height_files = list_chunk_files("height")

dbg("BMI_CALC", f"weight files={len(weight_files)}  height files={len(height_files)}")

weight_best = None
calc_weight_rows = 0

for fi, fp in enumerate(weight_files, start=1):
    for chunk in pd.read_csv(fp, sep="\t", dtype=str, compression="gzip", chunksize=OBS_CHUNKSIZE):
        chunk["patid"] = chunk["patid"].astype(str)
        chunk = chunk.merge(index_map, on="patid", how="inner")
        if chunk.empty:
            continue

        chunk["obsdate"] = parse_date(chunk["obsdate"])
        chunk = enforce_date_bounds(chunk, "obsdate")
        if chunk.empty:
            continue

        chunk["weight_kg"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk.dropna(subset=["weight_kg"])
        chunk = chunk[(chunk["weight_kg"] > 0) & (chunk["weight_kg"] < 500)]
        if chunk.empty:
            continue

        calc_weight_rows += len(chunk)

        weight_best = best_within_lookback(
            weight_best,
            chunk[["patid", "indexdate", "obsdate", "weight_kg"]],
            date_col="obsdate",
            value_cols=["weight_kg"]
        )

if weight_best is None:
    weight_best = pd.DataFrame(columns=["patid", "indexdate", "obsdate", "days_before", "weight_kg"])

weight_best = weight_best.rename(columns={"obsdate": "weight_date"}).drop(columns=["days_before"], errors="ignore")

# --------------------------
# 3) Most recent valid height on/before indexdate (no lookback restriction)
# --------------------------
height_best = None
calc_height_rows = 0

for fi, fp in enumerate(height_files, start=1):
    for chunk in pd.read_csv(fp, sep="\t", dtype=str, compression="gzip", chunksize=OBS_CHUNKSIZE):
        chunk["patid"] = chunk["patid"].astype(str)
        chunk = chunk.merge(index_map, on="patid", how="inner")
        if chunk.empty:
            continue

        chunk["obsdate"] = parse_date(chunk["obsdate"])
        chunk = enforce_date_bounds(chunk, "obsdate")
        if chunk.empty:
            continue

        chunk["height_raw"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk.dropna(subset=["height_raw"])
        if chunk.empty:
            continue

        calc_height_rows += len(chunk)

        chunk["height_m"] = np.where(chunk["height_raw"] > 10, chunk["height_raw"] / 100.0, chunk["height_raw"])
        chunk = chunk[(chunk["height_m"] >= 0.5) & (chunk["height_m"] <= 2.5)]
        if chunk.empty:
            continue

        height_best = most_recent_before_index(
            height_best,
            chunk[["patid", "indexdate", "obsdate", "height_m"]],
            date_col="obsdate",
            value_cols=["height_m"]
        )

if height_best is None:
    height_best = pd.DataFrame(columns=["patid", "indexdate", "obsdate", "days_before", "height_m"])

height_best = height_best.rename(columns={"obsdate": "height_date"}).drop(columns=["days_before"], errors="ignore")

dbg("BMI_CALC_DONE", f"valid weight rows seen={calc_weight_rows:,}  valid height rows seen={calc_height_rows:,}  "
                      f"weight_best={len(weight_best):,}  height_best={len(height_best):,}")

# --------------------------
# 4) Calculated BMI fallback: recent weight within lookback + most recent height ever before indexdate
# --------------------------
calc_best = weight_best.merge(
    height_best,
    on=["patid", "indexdate"],
    how="inner"
)

dbg("BMI_CALC_FALLBACK", f"merged weight+height candidates={len(calc_best):,}")

if not calc_best.empty:
    calc_best["bmi"] = calc_best["weight_kg"] / (calc_best["height_m"] ** 2)
    calc_best = calc_best[(calc_best["bmi"] >= BMI_MIN) & (calc_best["bmi"] <= BMI_MAX)].copy()
    calc_best["bmi_date"] = calc_best["weight_date"]   # anchor calculated BMI to weight date
    calc_best["bmi_source"] = "calculated"

calc_candidate_count = len(calc_best)

# remove those who already have recorded BMI
if not calc_best.empty:
    rec_keys = set(bmi_best["patid"].astype(str) + "_" + bmi_best["indexdate"].astype(str))
    calc_best["key"] = calc_best["patid"].astype(str) + "_" + calc_best["indexdate"].astype(str)
    overlap_count = calc_best["key"].isin(rec_keys).sum()
    calc_best = calc_best[~calc_best["key"].isin(rec_keys)].drop(columns=["key"])
else:
    overlap_count = 0

calculated_bmi_count = len(calc_best)

dbg("BMI_CALC_FALLBACK", f"calculated BMI candidates={calc_candidate_count:,}  "
                          f"already_have_recorded={overlap_count:,}  "
                          f"new_fallback={calculated_bmi_count:,}")

if calculated_bmi_count > 0:
    dbg_value_stats(calc_best["bmi"], "BMI_CALC_BEST", "calculated_bmi")
    dbg_date_col(calc_best, "bmi_date", "BMI_CALC_BEST")

# --------------------------
# 5) Final BMI
# --------------------------
if recorded_bmi_count > 0:
    bmi_best = bmi_best[["patid", "indexdate", "bmi_date", "bmi", "bmi_source"]]

if calculated_bmi_count > 0:
    calc_best = calc_best[["patid", "indexdate", "bmi_date", "bmi", "bmi_source"]]

bmi_final = pd.concat(
    [
        bmi_best if recorded_bmi_count > 0 else pd.DataFrame(columns=["patid", "indexdate", "bmi_date", "bmi", "bmi_source"]),
        calc_best if calculated_bmi_count > 0 else pd.DataFrame(columns=["patid", "indexdate", "bmi_date", "bmi", "bmi_source"])
    ],
    ignore_index=True
).drop_duplicates(["patid", "indexdate"], keep="first")

log(f"Final BMI rows: {len(bmi_final):,}")
dbg("BMI_FINAL", f"recorded={recorded_bmi_count:,}  calculated_fallback={calculated_bmi_count:,}  "
                  f"total={len(bmi_final):,}")
dbg("BMI_FINAL", f"coverage: {len(bmi_final):,}/{len(index_map):,} = "
                  f"{100*len(bmi_final)/len(index_map):.2f}%")
if len(bmi_final) > 0:
    dbg("BMI_FINAL", f"bmi_source counts:\n{bmi_final['bmi_source'].value_counts(dropna=False).to_string()}")
    dbg_value_stats(bmi_final["bmi"], "BMI_FINAL", "bmi")
    dbg_date_col(bmi_final, "bmi_date", "BMI_FINAL")

bmi_elapsed = time.time() - bmi_start
log(f"BMI section elapsed: {bmi_elapsed/60:.1f} min")

demog = demog.merge(
    bmi_final[["patid", "indexdate", "bmi_date", "bmi", "bmi_source"]] if not bmi_final.empty
    else pd.DataFrame(columns=["patid", "indexdate", "bmi_date", "bmi", "bmi_source"]),
    on=["patid", "indexdate"], how="left"
)
dbg("BMI_MERGE", f"demog rows after merge: {len(demog):,}  (should be {len(index_map):,})")
dbg("BMI_MERGE", f"bmi_source value_counts:\n{demog['bmi_source'].value_counts(dropna=False).to_string()}")

# ===========================================================================
# Blood pressure — OLD paired SBP+DBP (commented out)
# ===========================================================================
"""
log("Building blood pressure (paired, 1y pre-index)...")
bp_start = time.time()

sbp_files = list_chunk_files("sbp")
dbp_files = list_chunk_files("dbp")
dbg("BP_FILES", f"sbp files={len(sbp_files)}  dbp files={len(dbp_files)}")

bp_best = None
bp_sbp_rows = 0
bp_dbp_rows = 0
bp_joined_rows = 0

for fi, (sf, dfp) in enumerate(zip(sbp_files, dbp_files), start=1):
    s_list = []
    for schunk in pd.read_csv(sf, sep="\t", dtype=str, compression="gzip", chunksize=OBS_CHUNKSIZE):
        schunk["patid"] = schunk["patid"].astype(str)
        schunk = schunk.merge(index_map, on="patid", how="inner")
        if schunk.empty:
            continue
        schunk["obsdate"] = parse_date(schunk["obsdate"])
        schunk = enforce_date_bounds(schunk, "obsdate")
        if schunk.empty:
            continue
        schunk["systolic"] = pd.to_numeric(schunk["value"], errors="coerce")
        schunk = schunk.dropna(subset=["systolic"])
        schunk = schunk[schunk["systolic"].between(SBP_MIN, SBP_MAX)]
        if schunk.empty:
            continue
        s_list.append(schunk[["patid","indexdate","obsdate","systolic"]])
    if not s_list:
        continue
    s_all = pd.concat(s_list, ignore_index=True)
    bp_sbp_rows += len(s_all)

    for dchunk in pd.read_csv(dfp, sep="\t", dtype=str, compression="gzip", chunksize=OBS_CHUNKSIZE):
        dchunk["patid"] = dchunk["patid"].astype(str)
        dchunk = dchunk.merge(index_map, on="patid", how="inner")
        if dchunk.empty:
            continue
        dchunk["obsdate"] = parse_date(dchunk["obsdate"])
        dchunk = enforce_date_bounds(dchunk, "obsdate")
        if dchunk.empty:
            continue
        dchunk["diastolic"] = pd.to_numeric(dchunk["value"], errors="coerce")
        dchunk = dchunk.dropna(subset=["diastolic"])
        dchunk = dchunk[dchunk["diastolic"].between(DBP_MIN, DBP_MAX)]
        if dchunk.empty:
            continue
        bp_dbp_rows += len(dchunk)

        joined = s_all.merge(
            dchunk[["patid","indexdate","obsdate","diastolic"]],
            on=["patid","indexdate","obsdate"],
            how="inner"
        )
        if joined.empty:
            continue
        bp_joined_rows += len(joined)

        bp_best = best_within_lookback(
            bp_best,
            joined[["patid","indexdate","obsdate","systolic","diastolic"]],
            date_col="obsdate",
            value_cols=["systolic","diastolic"]
        )

    if fi % 50 == 0:
        dbg("BP_PROGRESS", f"file_pairs={fi}/{min(len(sbp_files),len(dbp_files))}  "
                            f"sbp_rows={bp_sbp_rows:,}  dbp_rows={bp_dbp_rows:,}  "
                            f"joined={bp_joined_rows:,}  "
                            f"best_size={0 if bp_best is None else len(bp_best):,}")

bp_best = pd.DataFrame() if bp_best is None else bp_best
if not bp_best.empty:
    bp_best = bp_best.rename(columns={"obsdate":"bp_date"}).drop(columns=["days_before"], errors="ignore")

log(f"Final BP rows: {len(bp_best):,}")
dbg("BP_DONE", f"sbp_rows_valid={bp_sbp_rows:,}  dbp_rows_valid={bp_dbp_rows:,}  "
                f"paired_same_date={bp_joined_rows:,}")
dbg("BP_DONE", f"final best rows={len(bp_best):,}")
dbg("BP_DONE", f"coverage: {len(bp_best):,}/{len(index_map):,} = "
                f"{100*len(bp_best)/max(len(index_map),1):.2f}%")

if not bp_best.empty:
    dbg_value_stats(bp_best["systolic"], "BP_FINAL", "systolic")
    dbg_value_stats(bp_best["diastolic"], "BP_FINAL", "diastolic")
    dbg_date_col(bp_best, "bp_date", "BP_FINAL")

    sbp_lt_dbp = (bp_best["systolic"] < bp_best["diastolic"]).sum()
    if sbp_lt_dbp > 0:
        dbg("BP_WARNING", f"SBP < DBP in {sbp_lt_dbp:,} rows ({100*sbp_lt_dbp/len(bp_best):.2f}%) — check data quality")
    else:
        dbg("BP_OK", "all rows have SBP >= DBP (good)")

bp_elapsed = time.time() - bp_start
log(f"BP section elapsed: {bp_elapsed/60:.1f} min")

demog = demog.merge(
    bp_best[["patid","indexdate","bp_date","systolic","diastolic"]] if not bp_best.empty
    else pd.DataFrame(columns=["patid","indexdate","bp_date","systolic","diastolic"]),
    on=["patid","indexdate"], how="left"
)
dbg("BP_MERGE", f"demog rows after merge: {len(demog):,}  (should be {len(index_map):,})")
"""

# ===========================================================================
# Blood pressure — NEW SBP only (no pairing with DBP)
# ===========================================================================
log("Building blood pressure (SBP only, 1y pre-index)...")
bp_start = time.time()

sbp_files = list_chunk_files("sbp")
dbg("BP_FILES", f"sbp files={len(sbp_files)}")

bp_best = None
bp_sbp_rows = 0
bp_sbp_in_range = 0

for fi, sf in enumerate(sbp_files, start=1):
    for schunk in pd.read_csv(sf, sep="\t", dtype=str, compression="gzip", chunksize=OBS_CHUNKSIZE):
        schunk["patid"] = schunk["patid"].astype(str)
        schunk = schunk.merge(index_map, on="patid", how="inner")
        if schunk.empty:
            continue
        bp_sbp_rows += len(schunk)

        schunk["obsdate"] = parse_date(schunk["obsdate"])
        schunk = enforce_date_bounds(schunk, "obsdate")
        if schunk.empty:
            continue
        schunk["systolic"] = pd.to_numeric(schunk["value"], errors="coerce")
        schunk = schunk.dropna(subset=["systolic"])
        schunk = schunk[schunk["systolic"].between(SBP_MIN, SBP_MAX)]
        if schunk.empty:
            continue
        bp_sbp_in_range += len(schunk)

        bp_best = best_within_lookback(
            bp_best,
            schunk[["patid", "indexdate", "obsdate", "systolic"]],
            date_col="obsdate",
            value_cols=["systolic"]
        )

    if fi % 50 == 0:
        dbg("BP_PROGRESS", f"files={fi}/{len(sbp_files)}  "
                            f"sbp_rows={bp_sbp_rows:,}  in_range={bp_sbp_in_range:,}  "
                            f"best_size={0 if bp_best is None else len(bp_best):,}")

bp_best = pd.DataFrame() if bp_best is None else bp_best
if not bp_best.empty:
    bp_best = bp_best.rename(columns={"obsdate": "bp_date"}).drop(columns=["days_before"], errors="ignore")

log(f"Final BP rows: {len(bp_best):,}")
dbg("BP_DONE", f"sbp_rows_merged={bp_sbp_rows:,}  sbp_in_range={bp_sbp_in_range:,}")
dbg("BP_DONE", f"final best rows={len(bp_best):,}")
dbg("BP_DONE", f"coverage: {len(bp_best):,}/{len(index_map):,} = "
                f"{100*len(bp_best)/max(len(index_map),1):.2f}%")

if not bp_best.empty:
    dbg_value_stats(bp_best["systolic"], "BP_FINAL", "systolic")
    dbg_date_col(bp_best, "bp_date", "BP_FINAL")

bp_elapsed = time.time() - bp_start
log(f"BP section elapsed: {bp_elapsed/60:.1f} min")

demog = demog.merge(
    bp_best[["patid", "indexdate", "bp_date", "systolic"]] if not bp_best.empty
    else pd.DataFrame(columns=["patid", "indexdate", "bp_date", "systolic"]),
    on=["patid", "indexdate"], how="left"
)
dbg("BP_MERGE", f"demog rows after merge: {len(demog):,}  (should be {len(index_map):,})")
# ===========================================================================
# Export
# ===========================================================================
log("Exporting final dataset...")

final_cols = [
    "patid","indexdate","diabetes_type","gender","yob","gen_ethnicity","e2019_imd_10",
    "dod_ons","death_ons","censor_date",
    "smoking_date","smoking_status",
    "bmi_date","bmi","bmi_source",
    "bp_date","systolic"
]

for c in final_cols:
    if c not in demog.columns:
        dbg("EXPORT_WARNING", f"column '{c}' not in demog — creating as NA")
        demog[c] = pd.NA

demog = demog[final_cols]

dbg_df(demog, "EXPORT_FINAL")
dbg("EXPORT_FINAL", f"columns={list(demog.columns)}")

# Final missingness report
dbg("EXPORT_FINAL", "--- MISSINGNESS REPORT ---")
for col in demog.columns:
    n_miss = demog[col].isna().sum()
    pct = 100 * n_miss / len(demog)
    dbg("EXPORT_FINAL", f"  {col}: missing={n_miss:,} ({pct:.2f}%)")

# Date diagnostics on all date columns
for col in ["indexdate", "dod_ons", "censor_date", "smoking_date", "bmi_date", "bp_date"]:
    if col in demog.columns:
        dbg_date_col(demog, col, "EXPORT_FINAL")

# Value distributions
#for col in ["bmi", "systolic", "diastolic"]:
for col in ["bmi", "systolic"]:
    if col in demog.columns:
        dbg_value_stats(demog[col], "EXPORT_FINAL", col)

# Categorical distributions
for col in ["smoking_status", "gender", "gen_ethnicity", "e2019_imd_10", "death_ons"]:
    if col in demog.columns:
        dbg("EXPORT_FINAL", f"{col} value_counts:\n{demog[col].value_counts(dropna=False).to_string()}")

# Coverage summary
dbg("EXPORT_FINAL", "--- COVERAGE SUMMARY ---")
total = len(demog)
for col, label in [("smoking_status", "Smoking"), ("bmi", "BMI"), ("systolic", "BP")]:
    if col in demog.columns:
        n = demog[col].notna().sum()
        dbg("EXPORT_FINAL", f"  {label}: {n:,}/{total:,} ({100*n/total:.2f}%)")

# Row count check
if len(demog) != len(index_map):
    dbg("EXPORT_WARNING", f"FINAL ROW COUNT {len(demog):,} != COHORT SIZE {len(index_map):,}")
else:
    dbg("EXPORT_OK", f"final row count matches cohort size: {len(demog):,}")

demog.to_csv(OUT_FILE, sep="\t", index=False)
dbg("OUTPUT", f"file={OUT_FILE}  size_bytes={os.path.getsize(OUT_FILE):,}")

total_elapsed = time.time() - t0
log(f"Saved: {OUT_FILE}")
log(f"TOTAL runtime: {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} hours)")
log("Done.")
