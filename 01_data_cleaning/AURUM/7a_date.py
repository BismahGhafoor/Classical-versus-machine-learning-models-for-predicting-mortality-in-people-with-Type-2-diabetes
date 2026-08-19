import pandas as pd
import zipfile
import os
import sys
import time
import platform
from datetime import datetime
from collections import Counter

# =============================================================================
# Logging helpers
# =============================================================================
import numpy as np

def clean_obsdate(series):
    """
    Clean obsdate strings from CPRD Aurum (DD/MM/YYYY).
    1. Extract year from the raw string to flag implausible dates
    2. Null out dates with year > 2025 or < 1900
    3. Parse with dayfirst=True
    4. Return as datetime
    """
    s = series.astype(str).str.strip()

    # extract last 4 characters as the year
    year = pd.to_numeric(s.str[-4:], errors="coerce")

    # null out implausible years
    bad_mask = (year > 2025) | (year < 1900) | year.isna()
    s = s.where(~bad_mask, other=np.nan)

    # now parse — everything remaining should be DD/MM/YYYY
    parsed = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")

    n_bad_year = bad_mask.sum() - series.isna().sum()  # don't count already-missing
    n_failed_parse = parsed.isna().sum() - bad_mask.sum()
    dbg("CLEAN_DATES", f"total={len(series):,}  already_missing={series.isna().sum():,}  "
                       f"bad_year={int(n_bad_year):,}  failed_parse={int(n_failed_parse):,}  "
                       f"valid_after_clean={parsed.notna().sum():,}")
    return parsed
    
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

# ------------------------
# Config
# ------------------------
t0 = time.perf_counter()
zip_index = int(sys.argv[1])
zip_folder = "/scratch/alice/b/bg205/smoking_data_input/Observation"
code_folder = "/scratch/alice/b/bg205/DataCleaning_Aurum_v2/Codes/clinical_biomarkers_CSV_exports"
output_folder = "/scratch/alice/b/bg205/01_03_AURUM/biomarker_chunks"
os.makedirs(output_folder, exist_ok=True)

log(f"Host: {platform.node()}")
log(f"CWD: {os.getcwd()}")
log(f"Python: {sys.executable} {platform.python_version()}")
dbg("CONFIG", f"zip_index={zip_index}  sys.argv={sys.argv}")
dbg("CONFIG", f"SLURM_ARRAY_TASK_ID={os.environ.get('SLURM_ARRAY_TASK_ID', 'not set')}  "
              f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'not set')}")
dbg("CONFIG", f"zip_folder={zip_folder}")
dbg("CONFIG", f"code_folder={code_folder}")
dbg("CONFIG", f"output_folder={output_folder}")

# ------------------------
# Get list of ZIPs in order
# ------------------------
zip_files = sorted(
    os.path.join(zip_folder, f)
    for f in os.listdir(zip_folder)
    if f.endswith(".zip")
)
if not zip_files:
    raise FileNotFoundError(f"No .zip files found in {zip_folder}")
if not (0 <= zip_index < len(zip_files)):
    raise IndexError(f"zip_index {zip_index} out of range 0..{len(zip_files)-1}")

zip_path = zip_files[zip_index]
log(f"Total ZIPs discovered: {len(zip_files)}")
dbg("ZIPS", f"first={os.path.basename(zip_files[0])}  last={os.path.basename(zip_files[-1])}")
dbg("ZIPS", f"selected zip_index={zip_index}: {os.path.basename(zip_path)}")
dbg("ZIPS", f"zip_size_bytes={os.path.getsize(zip_path):,}")

# ------------------------
# Load medcodeid lists for each biomarker
# ------------------------
def load_medcodes(file):
    path = os.path.join(code_folder, file)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # many CPRD exports have 2 header rows; try 2 then 0
    for skip in (2, 0):
        df = pd.read_csv(path, dtype=str, skiprows=skip)
        cols = df.columns.str.lower().str.strip()
        med = [c for c in cols if "medcode" in c]
        if med:
            col = df.columns[cols.get_loc(med[0])]
            codes_list = df[col].dropna().astype(str).str.strip().tolist()
            dbg("LOAD_MEDCODES", f"{file}: skiprows={skip}  columns={list(df.columns)}  "
                                 f"raw_rows={len(df)}  unique_medcodes={len(set(codes_list))}")
            return codes_list
    raise ValueError(f"No medcode column found in {file}")

log("Loading biomarker medcode CSVs")
codes = {
    "bmi":    load_medcodes("BMI_-_final.csv"),
    "weight": load_medcodes("Weight_final.csv"),
    "height": load_medcodes("Height_-_final.csv"),
    "sbp":    load_medcodes("SBP_final.csv"),
    "dbp":    load_medcodes("DBP_final.csv"),
}

# Convert to sets for fast lookup + report
code_sets = {}
total_unique_all = set()
for var, medcodes in codes.items():
    s = set(medcodes)
    code_sets[var] = s
    total_unique_all |= s
    dbg("BIOMARKER_CODES", f"{var}: {len(s)} unique medcodes  sample={sorted(s)[:5]}")

dbg("BIOMARKER_CODES", f"total unique across all biomarkers={len(total_unique_all)}")

# Check for cross-biomarker overlap
for i, v1 in enumerate(list(codes.keys())):
    for v2 in list(codes.keys())[i+1:]:
        overlap = code_sets[v1] & code_sets[v2]
        if overlap:
            dbg("CODES_WARNING", f"OVERLAP between {v1} and {v2}: {len(overlap)} shared medcodes")
            dbg("CODES_WARNING", f"  examples: {sorted(overlap)[:5]}")
        else:
            dbg("CODES_OK", f"no overlap between {v1} and {v2}")

# Codes to exclude — composite/ambiguous, confirmed with supervisor
drop_codes = {"253866018", "3636694012", "3636695013", "3636696014"}

for var in code_sets:
    removed = code_sets[var] & drop_codes
    if removed:
        code_sets[var] -= drop_codes
        dbg("CODES_DROP", f"{var}: removed {len(removed)} ambiguous codes: {sorted(removed)}")
# ------------------------
# Helpers
# ------------------------
def pick_value_column(df):
    # prefer 'value', fallback to common variants
    for cand in ["value", "value1", "numericvalue", "value_num", "val"]:
        if cand in df.columns:
            return cand
    return None

# ------------------------
# Process zip file
# ------------------------
log(f"Processing file {zip_index + 1}/{len(zip_files)}: {os.path.basename(zip_path)}")

tmp_records = {k: [] for k in codes}
total_rows_scanned = 0
total_files_in_zip = 0
total_skipped = 0
matches_per_var = Counter()
val_col_used = {}

with zipfile.ZipFile(zip_path, "r") as z:
    txt_members = [name for name in z.namelist() if name.lower().endswith(".txt")]
    dbg("ZIP_CONTENTS", f"txt files in zip: {len(txt_members)}  all members: {len(z.namelist())}")

    for name in txt_members:
        total_files_in_zip += 1
        file_start = time.perf_counter()

        with z.open(name) as f:
            try:
                df = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
            except Exception as e:
                log(f"Failed to read {name} in {zip_path}: {e}")
                dbg("ZIP_FILE_ERROR", f"file={name}  error={e}")
                total_skipped += 1
                continue

        dbg("ZIP_FILE", f"{name}: rows={len(df):,}  columns={list(df.columns)}")

        if df.empty or "medcodeid" not in df.columns:
            log(f"No 'medcodeid' in {name}")
            dbg("ZIP_FILE_SKIP", f"{name}: empty={df.empty}  has_medcodeid={'medcodeid' in df.columns}")
            total_skipped += 1
            continue
        if "patid" not in df.columns or "obsdate" not in df.columns:
            log(f"Missing patid/obsdate in {name}; skipping")
            dbg("ZIP_FILE_SKIP", f"{name}: has_patid={'patid' in df.columns}  has_obsdate={'obsdate' in df.columns}")
            total_skipped += 1
            continue

        rows_in_file = len(df)
        total_rows_scanned += rows_in_file

        val_col = pick_value_column(df)
        if val_col is None:
            df["value"] = pd.NA
            val_col = "value"
            dbg("ZIP_FILE_VALCOL", f"{name}: no value column found, created empty 'value'")
        else:
            dbg("ZIP_FILE_VALCOL", f"{name}: using value column='{val_col}'")

        if val_col not in val_col_used:
            val_col_used[val_col] = 0
        val_col_used[val_col] += 1

        df["medcodeid"] = df["medcodeid"].astype(str).str.strip()

        for var, medcodes in codes.items():
            matched = df[df["medcodeid"].isin(code_sets[var])]
            if not matched.empty:
                keep = matched[["patid", "obsdate", "medcodeid", val_col]].rename(columns={val_col: "value"})
                tmp_records[var].append(keep)
                matches_per_var[var] += len(keep)

                dbg("MATCH", f"{name} -> {var}: {len(keep):,} rows  "
                             f"patids={keep['patid'].nunique():,}  "
                             f"unique_medcodes={keep['medcodeid'].nunique()}")

                # Value column diagnostics for this match
                numeric_vals = pd.to_numeric(keep["value"], errors="coerce")
                n_numeric = numeric_vals.notna().sum()
                n_missing = keep["value"].isna().sum()
                if n_numeric > 0:
                    dbg("MATCH_VALUE", f"{name} -> {var}: numeric_values={n_numeric:,}  "
                                       f"missing={n_missing:,}  "
                                       f"min={numeric_vals.min():.2f}  "
                                       f"median={numeric_vals.median():.2f}  "
                                       f"max={numeric_vals.max():.2f}")
                else:
                    dbg("MATCH_VALUE", f"{name} -> {var}: no numeric values  missing={n_missing:,}")

        file_elapsed = time.perf_counter() - file_start
        dbg("ZIP_FILE_DONE", f"{name}: elapsed={file_elapsed:.1f}s")

# ------------------------
# Scanning summary
# ------------------------
scan_elapsed = time.perf_counter() - t0
log(f"Scanning complete in {scan_elapsed:.1f}s")
dbg("SCAN_SUMMARY", f"txt_files_scanned={total_files_in_zip}  skipped={total_skipped}  "
                     f"total_rows_scanned={total_rows_scanned:,}")
dbg("SCAN_SUMMARY", f"value_columns_used: {dict(val_col_used)}")
dbg("SCAN_SUMMARY", f"matches_per_biomarker:")
for var in codes:
    n_chunks = len(tmp_records[var])
    dbg("SCAN_SUMMARY", f"  {var}: {matches_per_var[var]:,} rows across {n_chunks} chunks")

# ------------------------
# Save outputs (TSV .txt.gz)
# ------------------------
log("Saving outputs")
for var, dfs in tmp_records.items():
    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        # --- inside the save loop, after result = pd.concat(dfs, ...) ---
        result["obsdate"] = clean_obsdate(result["obsdate"])
        output_file = os.path.join(output_folder, f"{var}_chunk_{zip_index:04d}.txt.gz")

        dbg_df(result, f"SAVE_{var.upper()}")
        dbg_date_col(result, "obsdate", f"SAVE_{var.upper()}")

        # Value diagnostics
        numeric_vals = pd.to_numeric(result["value"], errors="coerce")
        n_numeric = numeric_vals.notna().sum()
        n_missing_val = result["value"].isna().sum()
        dbg(f"SAVE_{var.upper()}", f"value: numeric={n_numeric:,}  missing={n_missing_val:,}  "
                                    f"non_numeric={len(result) - n_numeric - n_missing_val:,}")
        if n_numeric > 0:
            dbg(f"SAVE_{var.upper()}", f"value stats: min={numeric_vals.min():.2f}  "
                                        f"p25={numeric_vals.quantile(0.25):.2f}  "
                                        f"median={numeric_vals.median():.2f}  "
                                        f"p75={numeric_vals.quantile(0.75):.2f}  "
                                        f"max={numeric_vals.max():.2f}")

            # Flag implausible values per biomarker
            plausible = {
                "bmi":    (10, 80),
                "weight": (10, 400),
                "height": (50, 250),
                "sbp":    (40, 250),
                "dbp":    (20, 200),
            }
            if var in plausible:
                lo, hi = plausible[var]
                out_of_range = ((numeric_vals < lo) | (numeric_vals > hi)).sum()
                dbg(f"SAVE_{var.upper()}", f"plausibility check ({lo}-{hi}): "
                                            f"out_of_range={int(out_of_range):,} "
                                            f"({100*out_of_range/max(n_numeric,1):.2f}%)")

        # Missingness
        for col in result.columns:
            n_miss = result[col].isna().sum()
            if n_miss > 0:
                dbg(f"SAVE_{var.upper()}_MISSING", f"{col}: {n_miss:,} ({100*n_miss/len(result):.2f}%)")

        # Patient-level stats
        pat_counts = result.groupby("patid").size()
        dbg(f"SAVE_{var.upper()}", f"records_per_patient: min={pat_counts.min()}  "
                                    f"p50={int(pat_counts.median())}  "
                                    f"mean={pat_counts.mean():.1f}  "
                                    f"max={pat_counts.max()}")

        result.to_csv(output_file, sep="\t", index=False, compression="gzip")
        log(f"Saved {len(result):,} rows to {output_file} ({var})")
        dbg(f"SAVE_{var.upper()}", f"output_size_bytes={os.path.getsize(output_file):,}")
    else:
        log(f"No {var} records found in {os.path.basename(zip_path)}")
        dbg(f"SAVE_{var.upper()}", f"EMPTY — 0 rows for {var}")

# ------------------------
# Final summary
# ------------------------
total_elapsed = time.perf_counter() - t0
log(f"TOTAL runtime: {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)")
dbg("DONE", f"zip={os.path.basename(zip_path)}  zip_index={zip_index}  "
             f"total_rows_scanned={total_rows_scanned:,}")
dbg("DONE", f"rows saved per biomarker:")
for var in codes:
    total_saved = sum(len(d) for d in tmp_records[var])
    dbg("DONE", f"  {var}: {total_saved:,}")
