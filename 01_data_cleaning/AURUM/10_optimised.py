#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AURUM: Add labs/tests at indexdate (1-year lookback, on/before indexdate)
- HDL (0–40 mmol/L), LDL (0–20 mmol/L), Triglycerides (0–40 mmol/L), HbA1c (% 2–20)
- Total cholesterol:
    * Prefer Friedewald-calculated TC = HDL + LDL + (TG/2.2) when HDL+LDL+TG exist on SAME obsdate
    * Otherwise use recorded TC (0–20 mmol/L)
- Unit handling:
    * Lipids: convert mg/dL -> mmol/L (LDL/HDL/TC factor 0.02586; TG factor 0.01129)
    * HbA1c: IFCC mmol/mol -> % via 0.09148*x + 2.152 (and infer when unit missing)
- Streams chunk files (does NOT load everything at once)
Output: FINAL_Aurum_with_Tests.txt (TSV)
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
CHUNK_DIR = os.path.join(ROOT, "test_chunks")
ENRICHED_FILE = os.path.join(ROOT, "Enriched_Aurum_with_Biomarkers_3year.txt")
OUT_FILE = os.path.join(ROOT, "FINAL_Aurum_with_Tests_1year.txt")

# --------------------------
# Parameters
# --------------------------
#WINDOW_DAYS = 1095
WINDOW_DAYS = 365
MIN_DATE = pd.Timestamp("1900-01-01")
MAX_DATE = pd.Timestamp("2024-12-31")  # cap junk 2099; adjust if needed

# Filters (your rules)
HDL_MIN, HDL_MAX = 0.0, 10.0
TG_MIN,  TG_MAX  = 0.0, 40.0
LDL_MIN, LDL_MAX = 0.0, 20.0
TC_MIN,  TC_MAX  = 0.0, 20.0
HBA1C_MIN, HBA1C_MAX = 2.0, 20.0   # %

READ_CHUNKSIZE = 500_000

t0 = time.time()
log(f"Host: {platform.node()}")
log(f"CWD: {os.getcwd()}")
log(f"Python: {sys.executable} {platform.python_version()}")
dbg("CONFIG", f"ROOT={ROOT}")
dbg("CONFIG", f"CHUNK_DIR={CHUNK_DIR}")
dbg("CONFIG", f"ENRICHED_FILE={ENRICHED_FILE}")
dbg("CONFIG", f"OUT_FILE={OUT_FILE}")
dbg("CONFIG", f"WINDOW_DAYS={WINDOW_DAYS}  MIN_DATE={MIN_DATE}  MAX_DATE={MAX_DATE}")
dbg("CONFIG", f"HDL=[{HDL_MIN},{HDL_MAX}]  LDL=[{LDL_MIN},{LDL_MAX}]  TG=[{TG_MIN},{TG_MAX}]  "
              f"TC=[{TC_MIN},{TC_MAX}]  HbA1c=[{HBA1C_MIN},{HBA1C_MAX}]")
dbg("CONFIG", f"READ_CHUNKSIZE={READ_CHUNKSIZE}")

# --------------------------
# Helper utilities
# --------------------------
def list_chunks(prefix: str) -> list:
    files = sorted(glob.glob(os.path.join(CHUNK_DIR, f"{prefix}_chunk_*.txt.gz")))
    if not files:
        raise FileNotFoundError(f"No chunk files found for prefix '{prefix}' in {CHUNK_DIR}")
    dbg("CHUNK_FILES", f"{prefix}: found {len(files)} files  "
                       f"first={os.path.basename(files[0])}  last={os.path.basename(files[-1])}")
    return files

def as_lower_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def enforce_date_bounds(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    before = len(df)
    dt = df[date_col]
    df = df[(dt.notna()) & (dt >= MIN_DATE) & (dt <= MAX_DATE)]
    after = len(df)
    dropped = before - after
    if dropped > 0:
        dbg("DATE_BOUNDS", f"  {date_col}: dropped {dropped:,} rows outside [{MIN_DATE.date()}, {MAX_DATE.date()}]  kept={after:,}")
    return df

def within_lookback_on_or_before(df: pd.DataFrame, obs_col: str) -> pd.DataFrame:
    df = df.dropna(subset=["patid", "indexdate", obs_col]).copy()
    df = df[df[obs_col] <= df["indexdate"]]
    df["days_before"] = (df["indexdate"] - df[obs_col]).dt.days
    df = df[(df["days_before"] >= 0) & (df["days_before"] <= WINDOW_DAYS)]
    return df

def update_best(best: pd.DataFrame, new_rows: pd.DataFrame, obs_col: str, value_cols: list) -> pd.DataFrame:
    if new_rows is None or new_rows.empty:
        return best

    keep = ["patid", "indexdate", obs_col, "days_before"] + value_cols
    new_rows = new_rows[keep]

    if best is None or best.empty:
        out = new_rows.sort_values(["patid", "indexdate", "days_before"]).drop_duplicates(["patid", "indexdate"], keep="first")
        return out

    comb = pd.concat([best, new_rows], ignore_index=True)
    comb = comb.sort_values(["patid", "indexdate", "days_before"]).drop_duplicates(["patid", "indexdate"], keep="first")
    return comb

def standardize_lipids_to_mmol(df: pd.DataFrame, varname: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["unit"] = out["unit"] if "unit" in out.columns else pd.NA
    unit = as_lower_str(out["unit"])

    mgdl_syn = {"mg/dl", "mgdl", "mg per dl", "mg%/dl", "mg%dl"}
    factor = 0.01129 if varname in {"triglycerides", "tg"} else 0.02586

    is_mgdl = unit.isin(mgdl_syn)
    n_mgdl = int(is_mgdl.sum())
    if n_mgdl > 0:
        dbg(f"UNIT_CONV_{varname.upper()}", f"converting {n_mgdl:,} rows from mg/dL -> mmol/L (factor={factor})")
    out.loc[is_mgdl, "value"] = out.loc[is_mgdl, "value"] * factor

    missingish = (unit.isna()) | (unit == "nan") | (unit == "")
    looks_mgdl = missingish & (out["value"] > 40)
    n_inferred = int(looks_mgdl.sum())
    if n_inferred > 0:
        dbg(f"UNIT_CONV_{varname.upper()}", f"inferred {n_inferred:,} rows as mg/dL (value>40, unit missing) -> converting")
    out.loc[looks_mgdl, "value"] = out.loc[looks_mgdl, "value"] * factor

    return out

def standardize_hba1c_to_percent(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["unit"] = out["unit"] if "unit" in out.columns else pd.NA
    unit = as_lower_str(out["unit"])

    is_ifcc = unit.isin({"mmol/mol", "mmol per mol", "ifcc"})
    n_ifcc = int(is_ifcc.sum())
    if n_ifcc > 0:
        dbg("UNIT_CONV_HBA1C", f"converting {n_ifcc:,} rows from mmol/mol -> % (IFCC formula)")
    out.loc[is_ifcc, "value"] = 0.09148 * out.loc[is_ifcc, "value"] + 2.152

    missingish = (unit.isna()) | (unit == "nan") | (unit == "")
    infer_pct = missingish & out["value"].between(2, 20, inclusive="both")
    infer_ifcc = missingish & (~out["value"].between(2, 20, inclusive="both")) & out["value"].notna()

    n_infer_pct = int(infer_pct.sum())
    n_infer_ifcc = int(infer_ifcc.sum())
    if n_infer_pct > 0:
        dbg("UNIT_CONV_HBA1C", f"inferred {n_infer_pct:,} rows as already % (value 2-20, unit missing)")
    if n_infer_ifcc > 0:
        dbg("UNIT_CONV_HBA1C", f"inferred {n_infer_ifcc:,} rows as mmol/mol (value outside 2-20, unit missing) -> converting")

    out.loc[infer_ifcc, "value"] = 0.09148 * out.loc[infer_ifcc, "value"] + 2.152

    return out

def stream_var_best(prefix: str, index_map: pd.DataFrame, varname: str, vmin: float, vmax: float, out_col: str, date_col: str):
    """
    Stream chunks for a single lab and keep best within 1y pre-index.
    Returns: best_df with columns patid,indexdate,<date_col>,<out_col>
    """
    var_start = time.time()
    files = list_chunks(prefix)
    best = None

    total_chunks = 0
    total_rows_read = 0
    total_rows_merged = 0
    total_rows_date_ok = 0
    total_rows_valid = 0
    total_rows_in_range = 0
    total_rows_in_window = 0
    unit_counts = Counter()

    log(f"[{varname}] Streaming {len(files)} chunk files...")

    for i, fp in enumerate(files, 1):
        for chunk in pd.read_csv(fp, sep="\t", dtype=str, compression="gzip", chunksize=READ_CHUNKSIZE):
            total_chunks += 1
            total_rows_read += len(chunk)

            chunk["patid"] = chunk["patid"].astype(str)
            chunk = chunk.merge(index_map, on="patid", how="inner")
            if chunk.empty:
                continue
            total_rows_merged += len(chunk)

            chunk["obsdate"] = parse_date(chunk["obsdate"])
            chunk = enforce_date_bounds(chunk, "obsdate")
            if chunk.empty:
                continue
            total_rows_date_ok += len(chunk)

            chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
            chunk = chunk.dropna(subset=["value"])
            total_rows_valid += len(chunk)

            chunk = chunk[["patid", "indexdate", "obsdate", "value", "unit"] if "unit" in chunk.columns else ["patid","indexdate","obsdate","value"]]
            if "unit" not in chunk.columns:
                chunk["unit"] = pd.NA

            # Track units
            for u in chunk["unit"].value_counts(dropna=False).items():
                unit_counts[str(u[0])] += u[1]

            # unit standardization for lipids only
            if varname.lower() in {"hdl","ldl","triglycerides","tot_chol"}:
                chunk = standardize_lipids_to_mmol(chunk, varname.lower())

            chunk = within_lookback_on_or_before(chunk, "obsdate")
            if chunk.empty:
                continue
            total_rows_in_window += len(chunk)

            chunk = chunk[(chunk["value"] >= vmin) & (chunk["value"] <= vmax)]
            if chunk.empty:
                continue
            total_rows_in_range += len(chunk)

            chunk = chunk.rename(columns={"obsdate": date_col, "value": out_col})
            best = update_best(best, chunk, obs_col=date_col, value_cols=[out_col])

        if i % 50 == 0:
            dbg(f"{varname.upper()}_PROGRESS", f"files={i}/{len(files)}  chunks={total_chunks}  "
                                                f"rows_read={total_rows_read:,}  merged={total_rows_merged:,}  "
                                                f"in_window={total_rows_in_window:,}  in_range={total_rows_in_range:,}  "
                                                f"best_size={0 if best is None else len(best):,}")

    if best is None:
        best = pd.DataFrame(columns=["patid","indexdate",date_col,"days_before",out_col])

    best_count = len(best)
    best = best.drop(columns=["days_before"], errors="ignore")

    var_elapsed = time.time() - var_start
    log(f"[{varname}] best rows: {best_count:,}  elapsed: {var_elapsed/60:.1f} min")

    dbg(f"{varname.upper()}_DONE", f"chunks={total_chunks}  rows_read={total_rows_read:,}  "
                                    f"merged_to_cohort={total_rows_merged:,}  date_valid={total_rows_date_ok:,}  "
                                    f"numeric_valid={total_rows_valid:,}")
    dbg(f"{varname.upper()}_DONE", f"in_lookback_window={total_rows_in_window:,}  "
                                    f"in_range=[{vmin},{vmax}]={total_rows_in_range:,}  "
                                    f"best_rows={best_count:,}")
    dbg(f"{varname.upper()}_DONE", f"coverage: {best_count:,}/{len(index_map):,} = "
                                    f"{100*best_count/len(index_map):.2f}%")

    # Unit distribution
    if unit_counts:
        top_units = sorted(unit_counts.items(), key=lambda x: -x[1])[:10]
        dbg(f"{varname.upper()}_UNITS", f"unit distribution (top 10): {top_units}")

    # Value stats on best
    if best_count > 0:
        dbg_value_stats(best[out_col], f"{varname.upper()}_BEST", out_col)
        dbg_date_col(best, date_col, f"{varname.upper()}_BEST")

    return best

# --------------------------
# Load cohort
# --------------------------
log("Loading enriched cohort...")
if not os.path.exists(ENRICHED_FILE):
    raise FileNotFoundError(f"Enriched file not found: {ENRICHED_FILE}")
dbg("COHORT_FILE", f"file={ENRICHED_FILE}  size_bytes={os.path.getsize(ENRICHED_FILE):,}")

df = pd.read_csv(ENRICHED_FILE, sep="\t", dtype=str)
df["patid"] = df["patid"].astype(str)
df["indexdate"] = pd.to_datetime(df["indexdate"], errors="coerce", dayfirst=False)
if df["indexdate"].isna().any():
    raise ValueError(f"Missing indexdate in enriched cohort: {df['indexdate'].isna().sum()}")

dbg_df(df, "COHORT_LOADED")
dbg("COHORT_LOADED", f"columns={list(df.columns)}")
dbg_date_col(df, "indexdate", "COHORT_LOADED")

# Check for duplicate patids
dup_patids = df["patid"].duplicated().sum()
if dup_patids > 0:
    dbg("COHORT_WARNING", f"DUPLICATE patids: {dup_patids:,}")
else:
    dbg("COHORT_OK", "no duplicate patids (good)")

index_map = df[["patid","indexdate"]].drop_duplicates("patid")
log(f"Cohort size: {len(index_map):,}")

# Check what columns already exist from biomarker script
#existing_biomarker_cols = [c for c in ["smoking_status", "bmi", "systolic", "diastolic"] if c in df.columns]
existing_biomarker_cols = [c for c in ["smoking_status", "bmi", "systolic"] if c in df.columns]
dbg("COHORT_EXISTING", f"biomarker columns already present: {existing_biomarker_cols}")
for col in existing_biomarker_cols:
    n = df[col].notna().sum()
    dbg("COHORT_EXISTING", f"  {col}: non-missing={n:,}/{len(df):,} ({100*n/len(df):.2f}%)")

# ===========================================================================
# 1) Stream best HDL/LDL/TG
# ===========================================================================
log("=" * 60)
log("SECTION 1: HDL, LDL, Triglycerides")
log("=" * 60)

hdl_best = stream_var_best("hdl", index_map, "HDL", HDL_MIN, HDL_MAX, out_col="hdl", date_col="hdl_date")
ldl_best = stream_var_best("ldl", index_map, "LDL", LDL_MIN, LDL_MAX, out_col="ldl", date_col="ldl_date")
tg_best  = stream_var_best("triglycerides", index_map, "Triglycerides", TG_MIN, TG_MAX, out_col="triglycerides", date_col="triglycerides_date")

# Merge into main df
pre_merge = len(df)
df = df.merge(hdl_best, on=["patid","indexdate"], how="left")
dbg("MERGE_HDL", f"rows before={pre_merge:,}  after={len(df):,}")
df = df.merge(ldl_best, on=["patid","indexdate"], how="left")
dbg("MERGE_LDL", f"rows after={len(df):,}")
df = df.merge(tg_best,  on=["patid","indexdate"], how="left")
dbg("MERGE_TG", f"rows after={len(df):,}")

if len(df) != pre_merge:
    dbg("MERGE_WARNING", f"ROW COUNT CHANGED after lipid merges! was={pre_merge:,}  now={len(df):,}")

# ===========================================================================
# 2) Total cholesterol
# ===========================================================================
log("=" * 60)
log("SECTION 2: Total Cholesterol (Friedewald priority, recorded fallback)")
log("=" * 60)
tc_start = time.time()

log("[Total Cholesterol] Computing Friedewald when trio same-day, else recorded TC...")

hdl_files = list_chunks("hdl")
ldl_files = list_chunks("ldl")
tg_files  = list_chunks("triglycerides")

dbg("TC_CALC", f"hdl_files={len(hdl_files)}  ldl_files={len(ldl_files)}  tg_files={len(tg_files)}")
if not (len(hdl_files) == len(ldl_files) == len(tg_files)):
    dbg("TC_CALC_WARNING", f"file counts differ! hdl={len(hdl_files)} ldl={len(ldl_files)} tg={len(tg_files)}")

tc_calc_best = None
tc_trio_total_joined = 0
tc_trio_total_valid = 0
tc_trio_empty_indices = 0

log(f"[Total Cholesterol] Streaming trio same-day across {len(hdl_files)} chunk indices...")

for i, (hf, lf, tf) in enumerate(zip(hdl_files, ldl_files, tg_files), 1):

    def load_one(fp, vname, vmin, vmax):
        parts = []
        for ch in pd.read_csv(fp, sep="\t", dtype=str, compression="gzip", chunksize=READ_CHUNKSIZE):
            ch["patid"] = ch["patid"].astype(str)
            ch = ch.merge(index_map, on="patid", how="inner")
            if ch.empty:
                continue
            ch["obsdate"] = parse_date(ch["obsdate"])
            ch = enforce_date_bounds(ch, "obsdate")
            if ch.empty:
                continue
            ch["value"] = pd.to_numeric(ch["value"], errors="coerce")
            ch = ch.dropna(subset=["value"])
            if "unit" not in ch.columns:
                ch["unit"] = pd.NA
            ch = ch[["patid","indexdate","obsdate","value","unit"]]
            ch = standardize_lipids_to_mmol(ch, vname)
            ch = within_lookback_on_or_before(ch, "obsdate")
            if ch.empty:
                continue
            ch = ch[(ch["value"] >= vmin) & (ch["value"] <= vmax)]
            if ch.empty:
                continue
            parts.append(ch[["patid","indexdate","obsdate","value"]])
        if not parts:
            return pd.DataFrame(columns=["patid","indexdate","obsdate","value"])
        return pd.concat(parts, ignore_index=True)

    h = load_one(hf, "hdl", HDL_MIN, HDL_MAX).rename(columns={"value":"hdl"})
    l = load_one(lf, "ldl", LDL_MIN, LDL_MAX).rename(columns={"value":"ldl"})
    t = load_one(tf, "triglycerides", TG_MIN, TG_MAX).rename(columns={"value":"triglycerides"})

    if h.empty or l.empty or t.empty:
        tc_trio_empty_indices += 1
        if i % 50 == 0:
            dbg("TC_CALC_PROGRESS", f"chunk_index={i}/{len(hdl_files)}  trio_joined_total={tc_trio_total_joined:,}  "
                                     f"valid_total={tc_trio_total_valid:,}  empty_indices={tc_trio_empty_indices}  "
                                     f"best_size={0 if tc_calc_best is None else len(tc_calc_best):,}")
        continue

    trio = h.merge(l, on=["patid","indexdate","obsdate"], how="inner").merge(t, on=["patid","indexdate","obsdate"], how="inner")
    if trio.empty:
        tc_trio_empty_indices += 1
        if i % 50 == 0:
            dbg("TC_CALC_PROGRESS", f"chunk_index={i}/{len(hdl_files)}  trio_joined_total={tc_trio_total_joined:,}  "
                                     f"valid_total={tc_trio_total_valid:,}  empty_indices={tc_trio_empty_indices}  "
                                     f"best_size={0 if tc_calc_best is None else len(tc_calc_best):,}")
        continue

    tc_trio_total_joined += len(trio)

    # Friedewald in mmol/L: TC = HDL + LDL + (TG/2.2)
    trio["tc_calc"] = trio["hdl"] + trio["ldl"] + (trio["triglycerides"] / 2.2)

    trio = trio[(trio["tc_calc"].notna()) & (trio["tc_calc"] >= TC_MIN) & (trio["tc_calc"] <= TC_MAX)]
    if trio.empty:
        if i % 50 == 0:
            dbg("TC_CALC_PROGRESS", f"chunk_index={i}/{len(hdl_files)}  trio_joined_total={tc_trio_total_joined:,}  "
                                     f"valid_total={tc_trio_total_valid:,}  empty_indices={tc_trio_empty_indices}  "
                                     f"best_size={0 if tc_calc_best is None else len(tc_calc_best):,}")
        continue

    tc_trio_total_valid += len(trio)

    trio = trio.rename(columns={"obsdate":"tot_chol_date"})
    trio = trio[["patid","indexdate","tot_chol_date","tc_calc"]]
    trio["days_before"] = (trio["indexdate"] - trio["tot_chol_date"]).dt.days

    tc_calc_best = update_best(tc_calc_best, trio, obs_col="tot_chol_date", value_cols=["tc_calc"])

    if i % 50 == 0:
        dbg("TC_CALC_PROGRESS", f"chunk_index={i}/{len(hdl_files)}  trio_joined_total={tc_trio_total_joined:,}  "
                                 f"valid_total={tc_trio_total_valid:,}  empty_indices={tc_trio_empty_indices}  "
                                 f"best_size={0 if tc_calc_best is None else len(tc_calc_best):,}")

if tc_calc_best is None:
    tc_calc_best = pd.DataFrame(columns=["patid","indexdate","tot_chol_date","days_before","tc_calc"])

tc_calc_count = len(tc_calc_best)
tc_calc_best = tc_calc_best.drop(columns=["days_before"], errors="ignore")
log(f"[Total Cholesterol] calculated TC rows: {tc_calc_count:,}")
dbg("TC_CALC_DONE", f"trio_joined_total={tc_trio_total_joined:,}  trio_valid={tc_trio_total_valid:,}  "
                     f"empty_chunk_indices={tc_trio_empty_indices}")
dbg("TC_CALC_DONE", f"calculated TC best rows={tc_calc_count:,}  "
                     f"coverage: {tc_calc_count:,}/{len(index_map):,} = {100*tc_calc_count/len(index_map):.2f}%")
if tc_calc_count > 0:
    dbg_value_stats(tc_calc_best["tc_calc"], "TC_CALC_BEST", "tc_calc")
    dbg_date_col(tc_calc_best, "tot_chol_date", "TC_CALC_BEST")

# 2b) Recorded TC (fallback)
tc_rec_best = stream_var_best("tot_chol", index_map, "Recorded Total Cholesterol", TC_MIN, TC_MAX, out_col="tot_chol_rec", date_col="tot_chol_rec_date")

# 2c) Combine logic: use calculated if present else recorded
log("[Total Cholesterol] combining calculated-first then recorded fallback...")

tc_calc_best = tc_calc_best.rename(columns={"tc_calc":"tot_chol", "tot_chol_date":"tot_chol_date_calc"})
tc_rec_best  = tc_rec_best.rename(columns={"tot_chol_rec":"tot_chol", "tot_chol_rec_date":"tot_chol_date_rec"})

tc = tc_calc_best.merge(tc_rec_best, on=["patid","indexdate"], how="outer", suffixes=("_calc","_rec"))

tc["tot_chol"] = tc["tot_chol_calc"].combine_first(tc["tot_chol_rec"])
tc["tot_chol_date"] = tc["tot_chol_date_calc"].combine_first(tc["tot_chol_date_rec"])

# Source tracking
n_calc_only = tc["tot_chol_calc"].notna().sum()
n_rec_only = (tc["tot_chol_calc"].isna() & tc["tot_chol_rec"].notna()).sum()
n_both = (tc["tot_chol_calc"].notna() & tc["tot_chol_rec"].notna()).sum()
dbg("TC_COMBINE", f"calculated_used={n_calc_only:,}  recorded_fallback={n_rec_only:,}  "
                   f"had_both(calc_wins)={n_both:,}")

tc = tc[["patid","indexdate","tot_chol_date","tot_chol"]]
log(f"[Total Cholesterol] final TC rows: {len(tc):,}")
dbg("TC_FINAL", f"coverage: {len(tc):,}/{len(index_map):,} = {100*len(tc)/len(index_map):.2f}%")
if len(tc) > 0:
    dbg_value_stats(tc["tot_chol"], "TC_FINAL", "tot_chol")
    dbg_date_col(tc, "tot_chol_date", "TC_FINAL")

tc_elapsed = time.time() - tc_start
log(f"[Total Cholesterol] section elapsed: {tc_elapsed/60:.1f} min")

pre_merge = len(df)
df = df.merge(tc, on=["patid","indexdate"], how="left")
dbg("MERGE_TC", f"rows before={pre_merge:,}  after={len(df):,}")

# ===========================================================================
# 3) HbA1c (%)
# ===========================================================================
log("=" * 60)
log("SECTION 3: HbA1c")
log("=" * 60)
hba1c_start = time.time()

hba1c_files = list_chunks("hba1c")
hba_best = None

hba_chunks = 0
hba_rows_read = 0
hba_rows_merged = 0
hba_rows_date_ok = 0
hba_rows_valid = 0
hba_rows_in_window = 0
hba_rows_in_range = 0
hba_unit_counts = Counter()

log(f"[HbA1c] Streaming + standardising to % ... ({len(hba1c_files)} files)")

for i, fp in enumerate(hba1c_files, 1):
    for chunk in pd.read_csv(fp, sep="\t", dtype=str, compression="gzip", chunksize=READ_CHUNKSIZE):
        hba_chunks += 1
        hba_rows_read += len(chunk)

        chunk["patid"] = chunk["patid"].astype(str)
        chunk = chunk.merge(index_map, on="patid", how="inner")
        if chunk.empty:
            continue
        hba_rows_merged += len(chunk)

        chunk["obsdate"] = parse_date(chunk["obsdate"])
        chunk = enforce_date_bounds(chunk, "obsdate")
        if chunk.empty:
            continue
        hba_rows_date_ok += len(chunk)

        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk.dropna(subset=["value"])
        hba_rows_valid += len(chunk)

        if "unit" not in chunk.columns:
            chunk["unit"] = pd.NA

        # Track units before conversion
        for u in chunk["unit"].value_counts(dropna=False).items():
            hba_unit_counts[str(u[0])] += u[1]

        chunk = chunk[["patid","indexdate","obsdate","value","unit"]]
        chunk = standardize_hba1c_to_percent(chunk)
        chunk = within_lookback_on_or_before(chunk, "obsdate")
        if chunk.empty:
            continue
        hba_rows_in_window += len(chunk)

        chunk = chunk[(chunk["value"] > HBA1C_MIN) & (chunk["value"] <= HBA1C_MAX)]
        if chunk.empty:
            continue
        hba_rows_in_range += len(chunk)

        chunk = chunk.rename(columns={"obsdate":"hba1c_date", "value":"hba1c_perc"})
        hba_best = update_best(hba_best, chunk, obs_col="hba1c_date", value_cols=["hba1c_perc"])

    if i % 50 == 0:
        dbg("HBA1C_PROGRESS", f"files={i}/{len(hba1c_files)}  chunks={hba_chunks}  "
                               f"rows_read={hba_rows_read:,}  merged={hba_rows_merged:,}  "
                               f"in_window={hba_rows_in_window:,}  in_range={hba_rows_in_range:,}  "
                               f"best_size={0 if hba_best is None else len(hba_best):,}")

if hba_best is None:
    hba_best = pd.DataFrame(columns=["patid","indexdate","hba1c_date","days_before","hba1c_perc"])

hba_count = len(hba_best)
hba_best = hba_best.drop(columns=["days_before"], errors="ignore")

hba1c_elapsed = time.time() - hba1c_start
log(f"[HbA1c] best rows: {hba_count:,}  elapsed: {hba1c_elapsed/60:.1f} min")

dbg("HBA1C_DONE", f"chunks={hba_chunks}  rows_read={hba_rows_read:,}  "
                    f"merged={hba_rows_merged:,}  date_ok={hba_rows_date_ok:,}  "
                    f"numeric_valid={hba_rows_valid:,}")
dbg("HBA1C_DONE", f"in_window={hba_rows_in_window:,}  "
                    f"in_range=({HBA1C_MIN},{HBA1C_MAX}]={hba_rows_in_range:,}  "
                    f"best_rows={hba_count:,}")
dbg("HBA1C_DONE", f"coverage: {hba_count:,}/{len(index_map):,} = {100*hba_count/len(index_map):.2f}%")

# Unit distribution (pre-conversion)
if hba_unit_counts:
    top_units = sorted(hba_unit_counts.items(), key=lambda x: -x[1])[:10]
    dbg("HBA1C_UNITS", f"unit distribution (pre-conversion, top 10): {top_units}")

if hba_count > 0:
    dbg_value_stats(hba_best["hba1c_perc"], "HBA1C_BEST", "hba1c_perc")
    dbg_date_col(hba_best, "hba1c_date", "HBA1C_BEST")

pre_merge = len(df)
df = df.merge(hba_best, on=["patid","indexdate"], how="left")
dbg("MERGE_HBA1C", f"rows before={pre_merge:,}  after={len(df):,}")

# ===========================================================================
# Export
# ===========================================================================
log("=" * 60)
log("EXPORT")
log("=" * 60)

dbg_df(df, "PRE_EXPORT")
dbg("PRE_EXPORT", f"columns={list(df.columns)}")

# Final row count check
if len(df) != len(index_map):
    dbg("EXPORT_WARNING", f"FINAL ROW COUNT {len(df):,} != COHORT SIZE {len(index_map):,}")
else:
    dbg("EXPORT_OK", f"final row count matches cohort size: {len(df):,}")

# --- MISSINGNESS REPORT ---
dbg("EXPORT_FINAL", "--- MISSINGNESS REPORT ---")
for col in df.columns:
    n_miss = df[col].isna().sum()
    pct = 100 * n_miss / len(df)
    dbg("EXPORT_FINAL", f"  {col}: missing={n_miss:,} ({pct:.2f}%)")

# --- COVERAGE SUMMARY ---
dbg("EXPORT_FINAL", "--- COVERAGE SUMMARY ---")
total = len(df)
coverage_cols = [
    ("smoking_status", "Smoking"),
    ("bmi", "BMI"),
    ("systolic", "BP (SBP)"),
    ("hdl", "HDL"),
    ("ldl", "LDL"),
    ("triglycerides", "Triglycerides"),
    ("tot_chol", "Total Cholesterol"),
    ("hba1c_perc", "HbA1c"),
]
for col, label in coverage_cols:
    if col in df.columns:
        n = df[col].notna().sum()
        dbg("EXPORT_FINAL", f"  {label}: {n:,}/{total:,} ({100*n/total:.2f}%)")

# --- VALUE DISTRIBUTIONS ---
dbg("EXPORT_FINAL", "--- VALUE DISTRIBUTIONS ---")
#for col in ["bmi", "systolic", "diastolic", "hdl", "ldl", "triglycerides", "tot_chol", "hba1c_perc"]:
for col in ["bmi", "systolic", "hdl", "ldl", "triglycerides", "tot_chol", "hba1c_perc"]:
    if col in df.columns:
        dbg_value_stats(df[col], "EXPORT_FINAL", col)

# --- DATE DIAGNOSTICS ---
dbg("EXPORT_FINAL", "--- DATE DIAGNOSTICS ---")
for col in df.columns:
    if "date" in col.lower() or col.lower() in ("indexdate", "dod_ons"):
        dbg_date_col(df, col, "EXPORT_FINAL")

# --- CATEGORICAL DISTRIBUTIONS ---
for col in ["smoking_status", "gender", "gen_ethnicity", "e2019_imd_10", "death_ons"]:
    if col in df.columns:
        dbg("EXPORT_FINAL", f"{col} value_counts:\n{df[col].value_counts(dropna=False).to_string()}")

# --- DIABETES TYPE PRESERVED ---
if "diabetes_type" in df.columns:
    dbg("EXPORT_FINAL", f"diabetes_type distribution:\n{df['diabetes_type'].value_counts(dropna=False).to_string()}")

log(f"Saving: {OUT_FILE}")
df.to_csv(OUT_FILE, sep="\t", index=False)
dbg("OUTPUT", f"file={OUT_FILE}  size_bytes={os.path.getsize(OUT_FILE):,}")

total_elapsed = time.time() - t0
log(f"TOTAL runtime: {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} hours)")
log("Done.")
