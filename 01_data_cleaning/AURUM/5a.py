# -*- coding: utf-8 -*-
"""
Extract smoking status from CPRD Aurum zipped observation files using medcodeids (with debug output).
One-zip-per-task: pass SLURM_ARRAY_TASK_ID or a numeric CLI arg. Task i reads only zip i.

Also supports: `python test_smoke.py merge`
→ merges Aurum_Clinical_SmokingStatus_task*.txt.gz into Aurum_Clinical_SmokingStatus_ALL.txt.gz
"""

import pandas as pd
import numpy as np
import time
import os
import zipfile
import glob
import warnings
import platform
import sys
import gzip
from datetime import datetime
from collections import Counter

warnings.simplefilter(action='ignore')

# =============================================================================
# Configuration
# =============================================================================
current_directory = '/scratch/alice/b/bg205/01_03_AURUM'
current_directory_hpc = '/scratch/alice/b/bg205/01_03_AURUM'

observation_zip_folder = "/scratch/alice/b/bg205/smoking_data_input/Observation"
smoking_csv_folder = "/scratch/alice/b/bg205/DataCleaning_Aurum_v2/Codes/smoking_CSV_exports"
csv_files = [
    "Current_smoker.csv",
    "Ex-smoker.csv",
    "Never_smoked.csv"
]

output_dir = "/scratch/alice/b/bg205/01_03_AURUM/smoking_chunks"
output_basename = "Aurum_Clinical_SmokingStatus"   # chunks: *_task####.txt.gz
final_columns = ["patid", "obsdate", "medcodeid", "value"]

# =============================================================================
# Logging helpers
# =============================================================================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def dbg(tag: str, msg: str) -> None:
    print(f"DBG| [{tag}] {msg}", flush=True)

def dbg_date_col(df: pd.DataFrame, col: str, tag: str) -> None:
    """Print date column diagnostics matching pipeline style."""
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
    """Print shape + unique patids."""
    n_patids = df[patid_col].nunique() if patid_col in df.columns else "N/A"
    dbg(tag, f"rows={len(df):,}  patids={n_patids:,}")

# =============================================================================
# Helpers
# =============================================================================
def change_directory(current_directory, current_directory_hpc=None):
    print(f"{'-'*60}")
    if platform.system() == 'Windows':
        path = current_directory
    elif platform.system() == 'Linux':
        path = current_directory_hpc
    else:
        raise OSError("Unsupported OS")
    if path and os.path.isdir(path):
        os.chdir(path)
        print(f"Changed directory to: {os.getcwd()}")
    else:
        print(f"WARNING: directory not found or inaccessible: {path}. Staying in {os.getcwd()}")
    print(f"{'-'*60}\n")

def parse_task_id():
    """
    Get task id from CLI or SLURM_ARRAY_TASK_ID.
    Returns: int (0-based), "merge", or None.
    """
    # special CLI arg: 'merge'
    if len(sys.argv) > 1 and sys.argv[1].strip().lower() == "merge":
        return "merge"
    # CLI arg numeric
    if len(sys.argv) > 1 and sys.argv[1].strip() != "":
        try:
            return int(sys.argv[1])
        except ValueError:
            print(f"WARNING: invalid task id '{sys.argv[1]}'; ignoring.")
    # Slurm env var
    env_tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_tid is not None:
        try:
            return int(env_tid)
        except ValueError:
            print(f"WARNING: invalid SLURM_ARRAY_TASK_ID '{env_tid}'; ignoring.")
    return None

def merge_per_task_outputs(out_dir, basename):
    """
    Merge TSV gzip chunks named '<basename>_task*.txt.gz' into
    '<basename>_ALL.txt.gz' in out_dir, streaming to keep memory low.
    """
    pattern = os.path.join(out_dir, f"{basename}_task*.txt.gz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No chunk files found matching {pattern}")

    target = os.path.join(os.path.dirname(out_dir), f"Aurum_smoking_records_all.txt.gz")
    log(f"[merge] Merging {len(files)} files -> {target}")
    dbg("MERGE", f"pattern={pattern}")
    dbg("MERGE", f"files found: {len(files)}")
    dbg("MERGE", f"first={os.path.basename(files[0])}  last={os.path.basename(files[-1])}")

    wrote_header = False
    total_rows = 0
    start = time.perf_counter()
    with gzip.open(target, "wt") as w:
        for fi, f in enumerate(files, start=1):
            file_rows = 0
            for chunk in pd.read_csv(f, sep="\t", dtype=str, chunksize=500_000):
                chunk.to_csv(w, sep="\t", index=False, header=not wrote_header)
                wrote_header = True
                file_rows += len(chunk)
                total_rows += len(chunk)
            if fi % 20 == 0:
                dbg("MERGE_PROGRESS", f"processed {fi}/{len(files)} files  total_rows_so_far={total_rows:,}")

    mins = round((time.perf_counter() - start) / 60, 2)
    log(f"[merge] Done in {mins} min")
    dbg("MERGE_DONE", f"total_rows={total_rows:,}  output={target}")
    dbg("MERGE_DONE", f"output_size_bytes={os.path.getsize(target):,}")

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    t0 = time.perf_counter()
    change_directory(current_directory, current_directory_hpc)

    log(f"Host: {platform.node()}")
    log(f"CWD: {os.getcwd()}")
    log(f"Python: {sys.executable} {platform.python_version()}")

    task_id = parse_task_id()
    dbg("TASK", f"task_id={task_id}  sys.argv={sys.argv}")
    dbg("TASK", f"SLURM_ARRAY_TASK_ID={os.environ.get('SLURM_ARRAY_TASK_ID', 'not set')}  "
                f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'not set')}")

    if task_id == "merge":
        merge_per_task_outputs(output_dir, output_basename)
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Load medcodeids from CSVs
    # -------------------------------------------------------------------------
    log("Loading smoking medcode CSVs")
    medcodeids = []
    per_file_counts = {}

    for f in csv_files:
        full_path = os.path.join(smoking_csv_folder, f)
        if not os.path.exists(full_path):
            dbg("SMOKE_CSV_ERROR", f"FILE NOT FOUND: {full_path}")
            raise FileNotFoundError(f"Smoking CSV not found: {full_path}")

        df = pd.read_csv(full_path, dtype=str, skiprows=2)
        df.columns = [c.lower().strip() for c in df.columns]
        col_candidates = [col for col in df.columns if 'medcode' in col]
        if not col_candidates:
            raise ValueError(f"Could not find medcodeid column in {f}")

        codes = df[col_candidates[0]].dropna().astype(str).str.strip().tolist()
        per_file_counts[f] = len(set(codes))
        dbg("SMOKE_CSV", f"{f}: columns={list(df.columns)}  raw_rows={len(df)}  "
                         f"unique_medcodes={len(set(codes))}")
        medcodeids.extend(codes)

    smoking_medcodeids = sorted(set(medcodeids))
    smoking_medcode_set = set(smoking_medcodeids)
    log(f"Loaded {len(smoking_medcodeids)} unique smoking medcodeids.")
    dbg("SMOKE_CODES", f"per_file: {per_file_counts}")
    dbg("SMOKE_CODES", f"total unique (after dedup across files)={len(smoking_medcodeids)}")
    dbg("SMOKE_CODES", f"sample medcodes (first 10): {smoking_medcodeids[:10]}")

    # Check for overlap between files
    codes_per_file = {}
    for f in csv_files:
        full_path = os.path.join(smoking_csv_folder, f)
        df = pd.read_csv(full_path, dtype=str, skiprows=2)
        df.columns = [c.lower().strip() for c in df.columns]
        col_candidates = [col for col in df.columns if 'medcode' in col]
        codes_per_file[f] = set(df[col_candidates[0]].dropna().astype(str).str.strip().tolist())

    for i, f1 in enumerate(csv_files):
        for f2 in csv_files[i+1:]:
            overlap = codes_per_file[f1] & codes_per_file[f2]
            if overlap:
                dbg("SMOKE_CODES_WARNING", f"OVERLAP between {f1} and {f2}: {len(overlap)} shared medcodes")
                dbg("SMOKE_CODES_WARNING", f"  examples: {sorted(overlap)[:5]}")
            else:
                dbg("SMOKE_CODES", f"no overlap between {f1} and {f2} (good)")

    # -------------------------------------------------------------------------
    # Discover all ZIPs
    # -------------------------------------------------------------------------
    all_zip_files = sorted(glob.glob(os.path.join(observation_zip_folder, "*.zip")))
    assert all_zip_files, f"No zip files found in {observation_zip_folder}"
    log(f"Discovered {len(all_zip_files)} observation zip(s).")
    dbg("OBS_ZIPS", f"first={os.path.basename(all_zip_files[0])}  last={os.path.basename(all_zip_files[-1])}")

    # Decide which ZIP(s) to process for this task
    if task_id is not None:
        if isinstance(task_id, int):
            if task_id < 0 or task_id >= len(all_zip_files):
                raise IndexError(f"Task id {task_id} out of range 0..{len(all_zip_files)-1}")
            zip_files = [all_zip_files[task_id]]
            output_filename = os.path.join(output_dir, f"{output_basename}_task{task_id:04d}.txt.gz")
            log(f"Array task {task_id}: processing {os.path.basename(zip_files[0])}")
            dbg("TASK_ZIP", f"zip={zip_files[0]}  output={output_filename}")
        else:
            raise ValueError("Unexpected task_id state.")
    else:
        zip_files = all_zip_files
        output_filename = os.path.join(output_dir, f"{output_basename}_all.txt.gz")
        log(f"No task id provided; processing ALL {len(zip_files)} zip(s)")

    os.makedirs(output_dir, exist_ok=True)

    tmp_records = []
    start = time.perf_counter()
    total_rows_scanned = 0
    total_matches = 0
    total_files_in_zips = 0
    total_skipped_no_medcode = 0
    total_empty_zips = 0
    matches_per_zip = {}

    log(f"Processing {len(zip_files)} zipped file(s)...")

    for idx, zip_path in enumerate(zip_files, start=1):
        zip_start = time.perf_counter()
        zip_name = os.path.basename(zip_path)
        log(f"[{idx}/{len(zip_files)}] Reading ZIP: {zip_name}")
        dbg("ZIP_START", f"size_bytes={os.path.getsize(zip_path):,}")

        zip_matches = 0

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            txt_members = [m for m in zip_ref.namelist() if m.lower().endswith(".txt")]
            dbg("ZIP_CONTENTS", f"txt files in zip: {len(txt_members)}")

            for file_name in txt_members:
                total_files_in_zips += 1
                with zip_ref.open(file_name) as obs_file:
                    df = pd.read_csv(obs_file, sep="\t", dtype=str, low_memory=False)

                dbg("ZIP_FILE", f"{file_name}: rows={len(df):,}  columns={list(df.columns)}")

                if 'medcodeid' not in df.columns:
                    log(f"  Skipping '{file_name}' (no 'medcodeid' column).")
                    total_skipped_no_medcode += 1
                    continue

                rows_in_file = len(df)
                total_rows_scanned += rows_in_file

                df['medcodeid'] = df['medcodeid'].astype(str).str.strip()
                matches = df['medcodeid'].isin(smoking_medcode_set)
                match_count = int(matches.sum())
                total_matches += match_count
                zip_matches += match_count

                dbg("ZIP_FILE_MATCH", f"{file_name}: scanned={rows_in_file:,}  matches={match_count:,}  "
                                      f"match_rate={100*match_count/max(rows_in_file,1):.4f}%")

                if match_count == 0:
                    total_empty_zips += (1 if len(txt_members) == 1 else 0)
                    continue

                # Ensure required columns present; create empties if missing
                missing_cols = [col for col in final_columns if col not in df.columns]
                if missing_cols:
                    dbg("ZIP_FILE_MISSING_COLS", f"{file_name}: creating empty columns: {missing_cols}")
                for col in final_columns:
                    if col not in df.columns:
                        df[col] = pd.NA

                matched_df = df.loc[matches, final_columns]
                tmp_records.append(matched_df)

                # Debug: unique patids and medcodes in this match
                dbg("ZIP_FILE_DETAIL", f"{file_name}: matched_patids={matched_df['patid'].nunique():,}  "
                                       f"matched_unique_medcodes={matched_df['medcodeid'].nunique()}")

        zip_elapsed = time.perf_counter() - zip_start
        matches_per_zip[zip_name] = zip_matches
        dbg("ZIP_DONE", f"{zip_name}: matches={zip_matches:,}  elapsed={zip_elapsed:.1f}s")

    # -------------------------------------------------------------------------
    # Summary of scanning phase
    # -------------------------------------------------------------------------
    scan_elapsed = time.perf_counter() - start
    log(f"Scanning complete in {scan_elapsed/60:.2f} min")
    dbg("SCAN_SUMMARY", f"zips_processed={len(zip_files):,}  txt_files_scanned={total_files_in_zips:,}")
    dbg("SCAN_SUMMARY", f"total_rows_scanned={total_rows_scanned:,}  total_matches={total_matches:,}")
    dbg("SCAN_SUMMARY", f"overall_match_rate={100*total_matches/max(total_rows_scanned,1):.4f}%")
    dbg("SCAN_SUMMARY", f"skipped_no_medcode_col={total_skipped_no_medcode}")
    dbg("SCAN_SUMMARY", f"tmp_record_chunks={len(tmp_records)}")

    # Top/bottom zips by match count
    if matches_per_zip:
        sorted_zips = sorted(matches_per_zip.items(), key=lambda x: -x[1])
        dbg("SCAN_TOP_ZIPS", f"top 5 by matches: {sorted_zips[:5]}")
        zero_match_zips = [z for z, c in sorted_zips if c == 0]
        dbg("SCAN_TOP_ZIPS", f"zips with 0 matches: {len(zero_match_zips)}")

    if not tmp_records:
        raise ValueError("Still no smoking-related rows found across the processed zips. "
                         "Check medcodeids and column names.")

    # -------------------------------------------------------------------------
    # Combine, clean, and save
    # -------------------------------------------------------------------------
    log("Concatenating matched records")
    final_df = pd.concat(tmp_records, ignore_index=True)
    dbg_df(final_df, "PRE_CLEAN")
    dbg("PRE_CLEAN", f"columns={list(final_df.columns)}")
    dbg("PRE_CLEAN", f"dtypes:\n{final_df.dtypes.to_string()}")

    # Date parsing
    final_df['obsdate'] = pd.to_datetime(final_df['obsdate'], errors='coerce', dayfirst=True)
    na_before_drop = final_df['obsdate'].isna().sum()
    rows_before_drop = len(final_df)
    final_df = final_df.dropna(subset=['obsdate'])
    rows_after_drop = len(final_df)

    dbg("DATE_CLEAN", f"NaT_obsdate={na_before_drop:,}  rows_dropped={rows_before_drop - rows_after_drop:,}  "
                      f"rows_remaining={rows_after_drop:,}")
    dbg_date_col(final_df, "obsdate", "POST_DATE_CLEAN")

    # Keep EXACT final columns + order
    final_df = final_df[final_columns]

    # -------------------------------------------------------------------------
    # Final diagnostics
    # -------------------------------------------------------------------------
    dbg_df(final_df, "FINAL")
    dbg("FINAL", f"columns={list(final_df.columns)}")

    # Missingness
    dbg("FINAL", "--- MISSINGNESS REPORT ---")
    for col in final_df.columns:
        n_miss = final_df[col].isna().sum()
        pct = 100 * n_miss / max(len(final_df), 1)
        dbg("FINAL", f"  {col}: missing={n_miss:,} ({pct:.2f}%)")

    # Medcode distribution (which smoking codes are most common?)
    medcode_counts = final_df['medcodeid'].value_counts()
    dbg("FINAL_MEDCODES", f"unique medcodes in output={medcode_counts.shape[0]}")
    dbg("FINAL_MEDCODES", f"top 10 medcodes:\n{medcode_counts.head(10).to_string()}")

    # Value column distribution (smoking status values)
    if 'value' in final_df.columns:
        dbg("FINAL_VALUE", f"value column value_counts:\n{final_df['value'].value_counts(dropna=False).head(20).to_string()}")

    # Patient-level stats
    pat_counts = final_df.groupby('patid').size()
    dbg("FINAL_PATIENTS", f"unique_patids={final_df['patid'].nunique():,}")
    dbg("FINAL_PATIENTS", f"records_per_patient: min={pat_counts.min()}  "
                          f"p50={int(pat_counts.median())}  "
                          f"mean={pat_counts.mean():.1f}  "
                          f"max={pat_counts.max()}")
    dbg("FINAL_PATIENTS", f"patients_with_single_record={int((pat_counts == 1).sum()):,}")

    mem_usage = np.round(final_df.memory_usage(deep=True).sum() / (1024**2), 1)
    dbg("FINAL", f"memory_usage={mem_usage} MB")
    log(f"Final dataset: {len(final_df):,} rows, ~{mem_usage} MB")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    log(f"Writing output: {output_filename}")
    final_df.to_csv(output_filename, sep="\t", index=False, compression="gzip", date_format='%d/%m/%Y')
    elapsed = round((time.perf_counter() - t0) / 60, 2)

    dbg("OUTPUT", f"file={output_filename}")
    dbg("OUTPUT", f"size_bytes={os.path.getsize(output_filename):,}")
    log(f"Saved '{output_filename}' in {elapsed} minutes total.")
