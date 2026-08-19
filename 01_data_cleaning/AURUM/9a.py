#!/usr/bin/env python3
# Extract HDL, LDL, Triglycerides, HbA1c (from Excel tabs) and Total Cholesterol (from TXT)
# from CPRD Aurum Observation ZIPs into per-ZIP gzipped TSV chunks.
# Converts HbA1c to % (DCCT) and lipids to mmol/L using numunitid.

import os
import sys
import zipfile
import time
import platform
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd

# =============================================================================
# Logging helpers
# =============================================================================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def dbg(tag: str, msg: str) -> None:
    print(f"DBG| [{tag}] {msg}", flush=True)

def dbg_date_col(df: pd.DataFrame, col: str, tag: str) -> None:
    s = df[col]
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors="coerce", format="%d/%m/%Y")

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

# ==============================
# CONFIG — EDIT THESE PATHS ONLY
# ==============================
zip_folder    = "/scratch/alice/b/bg205/smoking_data_input/Observation"
excel_codes   = "/scratch/alice/b/bg205/DataCleaning_Aurum_v2/modified_LRWE_Lilly_Aurum_medcodeid_clinical biomarkers.xlsx"
tot_chol_txt  = "/scratch/alice/b/bg205/DataCleaning_Aurum_v2/Codelist_Total_Cholesterol.txt"
output_folder = "/scratch/alice/b/bg205/01_03_AURUM/test_chunks"

# =============================================================================
# UNIT CODE MAPPINGS (from CPRD Aurum NumUnit lookup)
# =============================================================================

HBA1C_PCT_CODES = {
    "1",      # %
    "2",      # % HB
    "246",    # per cent
    "249",    # percentage unit
    "355",    # % of Hb
    "849",    # %Hb
    "860",    # % total haemoglobin
    "912",    # DCCT %
    "1282",   # percent
    "1748",   # %HbA1c
    "1905",   # % HbA0
    "2758",   # % (DCCT)
    "3411",   # %(DCCT)
    "6277",   # % HbA1c
    "1043",   # % total Hb
}

HBA1C_MMOL_MOL_CODES = {
    "220",    # mmol/mol
    "879",    # IFCCmmol/mol
    "892",    # mmol/molHb
    "916",    # mmol/mol HbA0
    "1675",   # mmol/mol Hb
    "2233",   # IFFC
    "3841",   # IFCC
    "11001",  # IFCC mmol/mol
    "1031"    # mM/M
}

LIPID_MMOL_L_CODES = {
    "218",    # mmol/L
}

LIPID_MG_DL_CODES = {
    "182",    # MG/DL
}

TRIG_G_L_CODES = {
    "139",    # g/L
}

def ifcc_to_dcct(mmol_mol):
    return mmol_mol * 0.0915 + 2.15

def mgdl_to_mmol_chol(mgdl):
    return mgdl / 38.67

def mgdl_to_mmol_trig(mgdl):
    return mgdl / 88.57

# =============================================================================
# UNIT CONVERSION (no plausibility filtering)
# =============================================================================

def clean_hba1c(df):
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["unit"]  = df["unit"].astype(str).str.strip()
    df.loc[df["unit"].isin(["nan", "None", "", "<NA>"]), "unit"] = np.nan

    n_start = len(df)
    df = df[df["value"].notna() & (df["value"] > 0)]
    dbg("CLEAN_HBA1C", f"dropped {n_start - len(df):,} missing/zero/negative values")

    has_unit = df["unit"].notna()
    is_pct   = has_unit & df["unit"].isin(HBA1C_PCT_CODES)
    is_mmol  = has_unit & df["unit"].isin(HBA1C_MMOL_MOL_CODES)
    is_unknown = has_unit & ~is_pct & ~is_mmol
    no_unit  = ~has_unit

    dbg("CLEAN_HBA1C", f"unit breakdown: pct={is_pct.sum():,}  mmol/mol={is_mmol.sum():,}  "
                        f"unknown_code={is_unknown.sum():,}  no_unit={no_unit.sum():,}")

    if is_unknown.sum() > 0:
        dbg("CLEAN_HBA1C", f"WARNING unrecognised unit codes:")
        for uid, cnt in df.loc[is_unknown, "unit"].value_counts().items():
            dbg("CLEAN_HBA1C", f"  numunitid={uid}: {cnt:,} rows")

    df["value_cleaned"] = np.nan
    df.loc[is_pct, "value_cleaned"] = df.loc[is_pct, "value"]
    df.loc[is_mmol, "value_cleaned"] = ifcc_to_dcct(df.loc[is_mmol, "value"])

    no_unit_pct  = no_unit & (df["value"] <= 20)
    no_unit_mmol = no_unit & (df["value"] > 20)
    df.loc[no_unit_pct,  "value_cleaned"] = df.loc[no_unit_pct, "value"]
    df.loc[no_unit_mmol, "value_cleaned"] = ifcc_to_dcct(df.loc[no_unit_mmol, "value"])

    dbg("CLEAN_HBA1C", f"no-unit inference: {no_unit_pct.sum():,} assumed %  |  "
                        f"{no_unit_mmol.sum():,} assumed mmol/mol")

    n_before = len(df)
    df = df[df["value_cleaned"].notna()]
    dbg("CLEAN_HBA1C", f"dropped {n_before - len(df):,} rows with unrecognised units")

    df["value"] = df["value_cleaned"]
    df["unit"]  = "%"
    df.drop(columns=["value_cleaned"], inplace=True)

    dbg("CLEAN_HBA1C", f"final: {len(df):,} rows  "
                        f"min={df['value'].min():.2f}  median={df['value'].median():.2f}  "
                        f"max={df['value'].max():.2f}")
    return df


def clean_lipid(df, biomarker):
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["unit"]  = df["unit"].astype(str).str.strip()
    df.loc[df["unit"].isin(["nan", "None", "", "<NA>"]), "unit"] = np.nan

    n_start = len(df)
    df = df[df["value"].notna() & (df["value"] > 0)]
    dbg(f"CLEAN_{biomarker.upper()}", f"dropped {n_start - len(df):,} missing/zero/negative values")

    has_unit    = df["unit"].notna()
    is_mmol     = has_unit & df["unit"].isin(LIPID_MMOL_L_CODES)
    is_mgdl     = has_unit & df["unit"].isin(LIPID_MG_DL_CODES)
    is_unknown  = has_unit & ~is_mmol & ~is_mgdl
    no_unit     = ~has_unit

    dbg(f"CLEAN_{biomarker.upper()}", f"unit breakdown: mmol/L={is_mmol.sum():,}  mg/dL={is_mgdl.sum():,}  "
                                       f"unknown_code={is_unknown.sum():,}  no_unit={no_unit.sum():,}")

    if is_unknown.sum() > 0:
        dbg(f"CLEAN_{biomarker.upper()}", f"WARNING unrecognised unit codes:")
        for uid, cnt in df.loc[is_unknown, "unit"].value_counts().head(10).items():
            dbg(f"CLEAN_{biomarker.upper()}", f"  numunitid={uid}: {cnt:,} rows")

    if biomarker == "triglycerides":
        convert_fn = mgdl_to_mmol_trig
    else:
        convert_fn = mgdl_to_mmol_chol

    df["value_cleaned"] = np.nan
    df.loc[is_mmol, "value_cleaned"] = df.loc[is_mmol, "value"]
    df.loc[is_mgdl, "value_cleaned"] = convert_fn(df.loc[is_mgdl, "value"])
    
    # Handle g/L for triglycerides (mmol/L = g/L × 1.129)
    if biomarker == "triglycerides":
        is_g_l = has_unit & df["unit"].isin(TRIG_G_L_CODES)
        if is_g_l.sum() > 0:
            df.loc[is_g_l, "value_cleaned"] = df.loc[is_g_l, "value"] * 1.129
            is_unknown = is_unknown & ~is_g_l
            dbg(f"CLEAN_{biomarker.upper()}", f"g/L conversion: {is_g_l.sum():,} rows")

    mgdl_threshold = {"tot_chol": 15, "hdl": 5, "ldl": 10, "triglycerides": 20}
    threshold = mgdl_threshold[biomarker]

    no_unit_mmol = no_unit & (df["value"] <= threshold)
    no_unit_mgdl = no_unit & (df["value"] > threshold)
    df.loc[no_unit_mmol, "value_cleaned"] = df.loc[no_unit_mmol, "value"]
    df.loc[no_unit_mgdl, "value_cleaned"] = convert_fn(df.loc[no_unit_mgdl, "value"])

    dbg(f"CLEAN_{biomarker.upper()}", f"no-unit inference: {no_unit_mmol.sum():,} assumed mmol/L  |  "
                                       f"{no_unit_mgdl.sum():,} assumed mg/dL")

    n_before = len(df)
    df = df[df["value_cleaned"].notna()]
    dbg(f"CLEAN_{biomarker.upper()}", f"dropped {n_before - len(df):,} rows with unrecognised units")

    df["value"] = df["value_cleaned"]
    df["unit"]  = "mmol/L"
    df.drop(columns=["value_cleaned"], inplace=True)

    dbg(f"CLEAN_{biomarker.upper()}", f"final: {len(df):,} rows  "
                                       f"min={df['value'].min():.2f}  median={df['value'].median():.2f}  "
                                       f"max={df['value'].max():.2f}")
    return df


CLEANERS = {
    "hba1c":        lambda df: clean_hba1c(df),
    "hdl":          lambda df: clean_lipid(df, "hdl"),
    "ldl":          lambda df: clean_lipid(df, "ldl"),
    "triglycerides": lambda df: clean_lipid(df, "triglycerides"),
    "tot_chol":     lambda df: clean_lipid(df, "tot_chol"),
}

EXCEL_SHEETS = {
    "hba1c"        : "HbA1c - final",
    "hdl"          : "HDL_final",
    "ldl"          : "LDL_final",
    "triglycerides": "Trig_final",
}

# ==============================
# HELPERS
# ==============================
def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr); sys.exit(code)

def pick_medcode_column(columns):
    for c in columns:
        if "medcode" in str(c).lower():
            return c
    for c in columns:
        if str(c).lower() == "code":
            return c
    raise ValueError("No medcode/'code' column found.")

def load_medcodes_from_excel(xlsx_path, sheet_name):
    for hdr in (0, 1, 2, 3, 4, 5):
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype=str, header=hdr)
        except Exception as e:
            dbg("LOAD_EXCEL", f"sheet='{sheet_name}' header={hdr} failed: {e}")
            continue
        if df is None or df.empty:
            dbg("LOAD_EXCEL", f"sheet='{sheet_name}' header={hdr}: empty dataframe")
            continue
        df.columns = [str(c).strip() for c in df.columns]
        dbg("LOAD_EXCEL", f"sheet='{sheet_name}' header={hdr}: columns={list(df.columns)}  rows={len(df)}")
        try:
            col = pick_medcode_column(df.columns)
            codes_list = df[col].dropna().astype(str).str.strip().unique().tolist()
            dbg("LOAD_EXCEL", f"sheet='{sheet_name}': medcode_col='{col}'  unique_codes={len(codes_list)}  sample={codes_list[:5]}")
            return codes_list
        except Exception as e:
            dbg("LOAD_EXCEL", f"sheet='{sheet_name}' header={hdr}: pick_medcode_column failed: {e}")
            continue
    dbg("LOAD_EXCEL", f"sheet='{sheet_name}': FALLBACK to first column (no medcode col found)")
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype=str, header=None)
    first_col = df.columns[0]
    codes_list = df[first_col].dropna().astype(str).str.strip().unique().tolist()
    dbg("LOAD_EXCEL", f"sheet='{sheet_name}': fallback unique_codes={len(codes_list)}  sample={codes_list[:5]}")
    return codes_list

def load_totchol_from_txt(txt_path):
    dbg("LOAD_TOTCHOL", f"reading {txt_path}")
    try:
        df = pd.read_csv(txt_path, sep="\t", dtype=str)
        dbg("LOAD_TOTCHOL", f"tab-separated read OK: rows={len(df)}  columns={list(df.columns)}")
    except Exception as e:
        dbg("LOAD_TOTCHOL", f"tab-separated read failed ({e}), trying auto-detect sep")
        df = pd.read_csv(txt_path, sep=None, engine="python", dtype=str)
        dbg("LOAD_TOTCHOL", f"auto-detect read OK: rows={len(df)}  columns={list(df.columns)}")
    if df is None or df.empty:
        dbg("LOAD_TOTCHOL", "WARNING: empty dataframe, returning []")
        return []
    df.columns = [str(c).strip() for c in df.columns]
    col = pick_medcode_column(df.columns)
    codes_list = df[col].dropna().astype(str).str.strip().unique().tolist()
    dbg("LOAD_TOTCHOL", f"medcode_col='{col}'  unique_codes={len(codes_list)}  sample={codes_list[:5]}")
    return codes_list

def find_unit_column(df):
    candidates = (
        "numunitid",
        "valueunitid", "valueunitsid", "unit", "value_unit",
        "unitid", "valueunit", "value_unitsid"
    )
    for u in candidates:
        if u in df.columns:
            return u
    return None

def pick_value_column(df):
    for c in ("value", "value1", "value2", "numvalue"):
        if c in df.columns:
            return c
    return None

def standardise_obsdate(df, date_col="obsdate", tag="DATE_FIX"):
    df = df.copy()

    # Keep raw version for debugging
    raw_col = f"{date_col}_raw"
    df[raw_col] = df[date_col]

    # Extract year from end of raw string (supervisor-style)
    df[f"{date_col}_year"] = df[date_col].astype(str).str[-4:]
    df[f"{date_col}_year"] = pd.to_numeric(df[f"{date_col}_year"], errors="coerce")

    dbg(tag, f"{date_col}: raw non-missing={df[date_col].notna().sum():,}")
    dbg(tag, f"{date_col}_year describe:\n{df[f'{date_col}_year'].describe().apply(lambda x: format(x, 'f'))}")

    # Set impossible future years to missing before parsing
    n_future = (df[f"{date_col}_year"] > 2025).sum()
    if n_future > 0:
        dbg(tag, f"{date_col}: setting {n_future:,} rows with year > 2025 to NaN before parsing")
        df.loc[df[f"{date_col}_year"] > 2025, date_col] = np.nan

    # Parse using exact UK format
    df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors="coerce")

    dbg(tag, f"{date_col}: parsed non-missing={df[date_col].notna().sum():,}  missing={df[date_col].isna().sum():,}")

    # Optional: show a few bad raw values that failed parsing
    bad = df[df[date_col].isna() & df[raw_col].notna()]
    if not bad.empty:
        dbg(tag, f"{date_col}: sample failed raw values={bad[raw_col].astype(str).head(10).tolist()}")

    return df
    
# ==============================
# MAIN
# ==============================
def main():
    t0 = time.perf_counter()

    if len(sys.argv) < 2:
        die("Usage: python 9a.py <zip_index>")

    try:
        zip_index = int(sys.argv[1])
    except ValueError:
        die("zip_index must be an integer (0-based)")

    log(f"Host: {platform.node()}")
    log(f"CWD: {os.getcwd()}")
    log(f"Python: {sys.executable} {platform.python_version()}")
    dbg("CONFIG", f"zip_index={zip_index}  sys.argv={sys.argv}")
    dbg("CONFIG", f"SLURM_ARRAY_TASK_ID={os.environ.get('SLURM_ARRAY_TASK_ID', 'not set')}  "
                  f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'not set')}")
    dbg("CONFIG", f"zip_folder={zip_folder}")
    dbg("CONFIG", f"excel_codes={excel_codes}")
    dbg("CONFIG", f"tot_chol_txt={tot_chol_txt}")
    dbg("CONFIG", f"output_folder={output_folder}")

    if not os.path.isdir(zip_folder):      die(f"zip_folder not found: {zip_folder}")
    if not os.path.isfile(excel_codes):    die(f"Excel codes file not found: {excel_codes}")
    if not os.path.isfile(tot_chol_txt):   die(f"Total Cholesterol TXT file not found: {tot_chol_txt}")
    os.makedirs(output_folder, exist_ok=True)

    zip_files = sorted(os.path.join(zip_folder, f) for f in os.listdir(zip_folder) if f.endswith(".zip"))
    if not zip_files: die(f"No .zip files found in {zip_folder}")
    if not (0 <= zip_index < len(zip_files)): die(f"zip_index out of range (0..{len(zip_files)-1})")

    zip_path = zip_files[zip_index]
    log(f"Total ZIPs discovered: {len(zip_files)}")
    dbg("ZIPS", f"first={os.path.basename(zip_files[0])}  last={os.path.basename(zip_files[-1])}")
    dbg("ZIPS", f"selected zip_index={zip_index}: {os.path.basename(zip_path)}")
    dbg("ZIPS", f"zip_size_bytes={os.path.getsize(zip_path):,}")
    log(f"Processing ZIP {zip_index+1}/{len(zip_files)}: {os.path.basename(zip_path)}")

    log("Loading biomarker medcodes from Excel sheets")
    codes = {}
    for var, sheet in EXCEL_SHEETS.items():
        codes[var] = load_medcodes_from_excel(excel_codes, sheet)
    log("Loading Total Cholesterol medcodes from TXT")
    codes["tot_chol"] = load_totchol_from_txt(tot_chol_txt)

    code_sets = {}
    total_unique_all = set()
    for k, v in codes.items():
        s = set(v)
        code_sets[k] = s
        total_unique_all |= s
        dbg("BIOMARKER_CODES", f"{k:15s}: {len(s):6d} unique medcodes  sample={sorted(s)[:5]}")
    dbg("BIOMARKER_CODES", f"total unique across all biomarkers={len(total_unique_all)}")

    varnames = list(codes.keys())
    for i, v1 in enumerate(varnames):
        for v2 in varnames[i+1:]:
            overlap = code_sets[v1] & code_sets[v2]
            if overlap:
                dbg("CODES_WARNING", f"OVERLAP between {v1} and {v2}: {len(overlap)} shared medcodes")
                dbg("CODES_WARNING", f"  examples: {sorted(overlap)[:5]}")
            else:
                dbg("CODES_OK", f"no overlap between {v1} and {v2}")

    tmp_records = {k: [] for k in codes}
    total_rows_scanned = 0
    total_files_in_zip = 0
    total_skipped = 0
    matches_per_var = Counter()
    val_col_used = {}

    with zipfile.ZipFile(zip_path, 'r') as z:
        txt_members = [name for name in z.namelist() if name.lower().endswith(".txt")]
        dbg("ZIP_CONTENTS", f"txt files in zip: {len(txt_members)}  all members: {len(z.namelist())}")

        for name in txt_members:
            total_files_in_zip += 1
            file_start = time.perf_counter()

            with z.open(name) as f:
                try:
                    df = pd.read_csv(
                        f, sep="\t", dtype=str, low_memory=False,
                        usecols=lambda c: str(c).lower() in {
                            "patid","obsdate","medcodeid",
                            "value","value1","value2","numvalue",
                            "numunitid",
                            "valueunitid","valueunitsid","unit","value_unit","unitid","valueunit","value_unitsid"
                        }
                    )
                except Exception as e:
                    log(f"Skipping {name} (read error): {e}")
                    dbg("ZIP_FILE_ERROR", f"file={name}  error={e}")
                    total_skipped += 1
                    continue

                dbg("ZIP_FILE", f"{name}: rows={len(df):,}  columns={list(df.columns)}")

                need_base = {"patid","obsdate","medcodeid"}
                if not need_base.issubset(df.columns):
                    missing_cols = ", ".join(sorted(need_base - set(df.columns)))
                    log(f"Skipping {name}: missing columns [{missing_cols}]")
                    dbg("ZIP_FILE_SKIP", f"{name}: missing={missing_cols}")
                    total_skipped += 1
                    continue

                total_rows_scanned += len(df)
                df["medcodeid"] = df["medcodeid"].astype(str).str.strip()

                val_col  = pick_value_column(df)
                unit_col = find_unit_column(df)

                if val_col is None:
                    log(f"Skipping {name}: no numeric value column")
                    dbg("ZIP_FILE_SKIP", f"{name}: no value column found among {list(df.columns)}")
                    total_skipped += 1
                    continue

                dbg("ZIP_FILE_VALCOL", f"{name}: value_col='{val_col}'  unit_col='{unit_col}'")
                if val_col not in val_col_used:
                    val_col_used[val_col] = 0
                val_col_used[val_col] += 1

                base_keep = ["patid","obsdate","medcodeid", val_col]
                if unit_col: base_keep.append(unit_col)

                for var, medcodes in codes.items():
                    if not medcodes:
                        continue
                    matched = df[df["medcodeid"].isin(code_sets[var])]
                    if matched.empty:
                        continue
                    out = matched[base_keep].copy()
                    out = out.rename(columns={val_col: "value"})
                    if unit_col:
                        out = out.rename(columns={unit_col: "unit"})
                    else:
                        out["unit"] = pd.NA
                    tmp_records[var].append(out)
                    matches_per_var[var] += len(out)

                    dbg("MATCH", f"{name} -> {var}: {len(out):,} rows  "
                                 f"patids={out['patid'].nunique():,}  "
                                 f"unique_medcodes={out['medcodeid'].nunique()}")

            file_elapsed = time.perf_counter() - file_start
            dbg("ZIP_FILE_DONE", f"{name}: elapsed={file_elapsed:.1f}s")

    scan_elapsed = time.perf_counter() - t0
    log(f"Scanning complete in {scan_elapsed:.1f}s")
    dbg("SCAN_SUMMARY", f"txt_files_scanned={total_files_in_zip}  skipped={total_skipped}  "
                         f"total_rows_scanned={total_rows_scanned:,}")
    dbg("SCAN_SUMMARY", f"value_columns_used: {dict(val_col_used)}")
    dbg("SCAN_SUMMARY", f"matches_per_biomarker (before cleaning):")
    for var in codes:
        n_chunks = len(tmp_records[var])
        dbg("SCAN_SUMMARY", f"  {var}: {matches_per_var[var]:,} rows across {n_chunks} chunks")

    log("Cleaning and converting units")
    wrote_any = False
    for var, dfs in tmp_records.items():
        if not dfs:
            log(f"No matches for {var} in this ZIP")
            dbg(f"SAVE_{var.upper()}", f"EMPTY — 0 rows for {var}")
            continue

        result = pd.concat(dfs, ignore_index=True)
        raw_count = len(result)
        dbg(f"SAVE_{var.upper()}", f"raw rows before cleaning: {raw_count:,}")

        # ---- UNIT CONVERSION (no plausibility filter) ----
        result = CLEANERS[var](result)

        if result.empty:
            log(f"No {var} rows survived cleaning in this ZIP")
            continue
        
        result = standardise_obsdate(result, date_col="obsdate", tag=f"DATE_FIX_{var.upper()}")

        # ---- DIAGNOSTICS ----
        out_file = os.path.join(output_folder, f"{var}_chunk_{zip_index:04d}.txt.gz")

        dbg_df(result, f"SAVE_{var.upper()}")
        dbg_date_col(result, "obsdate", f"SAVE_{var.upper()}")

        dbg(f"SAVE_{var.upper()}", f"value stats: min={result['value'].min():.2f}  "
                                    f"p25={result['value'].quantile(0.25):.2f}  "
                                    f"median={result['value'].median():.2f}  "
                                    f"p75={result['value'].quantile(0.75):.2f}  "
                                    f"max={result['value'].max():.2f}")
        dbg(f"SAVE_{var.upper()}", f"standard unit: {result['unit'].iloc[0]}")
        dbg(f"SAVE_{var.upper()}", f"rows dropped by cleaning: {raw_count - len(result):,} "
                                    f"({100*(raw_count - len(result))/max(raw_count,1):.1f}%)")

        for col in result.columns:
            n_miss = result[col].isna().sum()
            if n_miss > 0:
                dbg(f"SAVE_{var.upper()}_MISSING", f"{col}: {n_miss:,} ({100*n_miss/len(result):.2f}%)")

        pat_counts = result.groupby("patid").size()
        dbg(f"SAVE_{var.upper()}", f"records_per_patient: min={pat_counts.min()}  "
                                    f"p50={int(pat_counts.median())}  "
                                    f"mean={pat_counts.mean():.1f}  "
                                    f"max={pat_counts.max()}")
        result = result.drop(columns=[c for c in ["obsdate_raw", "obsdate_year"] if c in result.columns])

        result.to_csv(out_file, sep="\t", index=False, compression="gzip")
        log(f"Saved {len(result):,} rows to {out_file} ({var})")
        dbg(f"SAVE_{var.upper()}", f"output_size_bytes={os.path.getsize(out_file):,}")
        wrote_any = True

    if not wrote_any:
        log("No biomarker rows matched in this ZIP.")

    total_elapsed = time.perf_counter() - t0
    log(f"TOTAL runtime: {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)")
    dbg("DONE", f"zip={os.path.basename(zip_path)}  zip_index={zip_index}  "
                 f"total_rows_scanned={total_rows_scanned:,}")
    dbg("DONE", f"rows saved per biomarker (after cleaning):")
    for var in codes:
        out_file = os.path.join(output_folder, f"{var}_chunk_{zip_index:04d}.txt.gz")
        if os.path.exists(out_file):
            saved = pd.read_csv(out_file, sep="\t", compression="gzip", nrows=0)
            dbg("DONE", f"  {var}: see chunk file")
        else:
            dbg("DONE", f"  {var}: 0 (no output)")

if __name__ == "__main__":
    main()
