#!/usr/bin/env python3
"""
================================================================================
Baseline Comorbidity Extraction — CPRD + HES combined (v3)
================================================================================
Extracts baseline comorbidities from:
  - CPRD GOLD Clinical files   (medcode)  → CKD, HTN
  - CPRD Aurum Observation files (medcodeid) → CKD, HTN
  - HES APC episode-level diagnosis files   → CVD, HTN, CKD, Cancer

Final patient-level flags:
  - CKD  = CPRD OR HES;  first_date = min(cprd, hes)
  - HTN  = CPRD OR HES;  first_date = min(cprd, hes)
  - CVD  = HES only (I* excluding I10-I15, single composite)
  - Cancer (any + subtypes) = HES only

Per supervisor guidance (Sharmin, 12/05/2026):
  - CVD components (MI, HF, CHD, Stroke) come from HES only
  - HTN kept SEPARATE from CVD
  - Cancer from HES only
  - Individual CVD components to be discussed with Francesco

Outputs:
  - {db}_codelists_parsed.txt           (CPRD codelists)
  - {db}_long_summary.txt               (CPRD long format)
  - {db}_wide_baseline_comorbidities.txt (CPRD wide: CKD + HTN)
  - {db}_missing_date_qc.txt            (CPRD date QC)
  - hes_codelists_applied.txt           (HES ICD-10 definitions)
  - hes_long_summary.txt                (HES long format)
  - hes_wide_baseline_comorbidities.txt (HES wide: CVD, HTN, CKD, Cancer)
  - hes_qc_summary.txt                 (HES prevalence QC)
  - final_combined_comorbidities.txt    (all patients, all conditions, CPRD+HES merged)
  - final_qc_summary.txt               (final prevalence summary)
  - qc_prevalence_summary.txt          (CPRD-only prevalence)

References for cancer type selection:
  Pearson-Stuttard et al. (2021) Cancer Epidemiol Biomarkers Prev. 30(6):1218-1228.
  Ling et al. (2022) Diabetes Res Clin Pract. 185:109237.
================================================================================
"""
from __future__ import annotations

import gc
import io
import logging
import os
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION BLOCK — edit everything here
# ============================================================

RUN_GOLD  = True
RUN_AURUM = True
RUN_HES   = True

# --- Cohort ---
COHORT_FILE           = "/scratch/alice/b/bg205/16_02_26/Combined_GOLD_Aurum_recoded.txt"
COHORT_SEP            = "\t"
COHORT_COL_PATID      = "patid"
COHORT_COL_INDEX_DATE = "indexdate"
COHORT_COL_DATABASE   = "database"

# --- GOLD: directory containing Clinical zip files ---
GOLD_ZIP_DIR     = "/scratch/alice/b/bg205/DataCleaning_Gold_v2/GOLD"
GOLD_ZIP_PATTERN = "FZ_GOLD_All_Extract_Clinical_*.zip"
GOLD_SEP         = "\t"

GOLD_COL_PATID     = "patid"
GOLD_COL_MEDCODE   = "medcode"
GOLD_COL_EVENTDATE = "eventdate"

# --- Aurum: directory containing Observation zip files ---
AURUM_ZIP_DIR     = "/scratch/alice/b/bg205/DataCleaning_Aurum_v2/Observation"
AURUM_ZIP_PATTERN = "FZ_Aurum_*_Extract_Observation_*.zip"
AURUM_SEP         = "\t"

AURUM_COL_PATID     = "patid"
AURUM_COL_MEDCODEID = "medcodeid"
AURUM_COL_OBSDATE   = "obsdate"

# --- HES diagnosis files (episode-level) ---
HES_GOLD_FILE  = "/scratch/alice/b/bg205/HES_linked/GOLD/hes_diagnosis_epi_23_002869_DM.txt"
HES_AURUM_FILE = "/scratch/alice/b/bg205/HES_linked/Aurum/hes_diagnosis_epi_23_002869_DM.txt"
HES_SEP        = "\t"

HES_COL_PATID    = "patid"
HES_COL_ICD      = "ICD"
HES_COL_EPISTART = "epistart"
HES_COL_D_ORDER  = "d_order"

# --- CPRD Codelists ---
GOLD_CODELIST_FILE  = "/scratch/alice/b/bg205/Codes/LRWE_Lilly_GOLD_medcode_co-morbidities.xlsx"
AURUM_CODELIST_FILE = "/scratch/alice/b/bg205/Codes/LRWE_Lilly_Aurum_medcodeid_co-morbidities.xlsx"

GOLD_SHEETS = {
    "ckd" : "CKD_final",
    "htn" : "HTN_final",
    # CVD components removed — now HES only (Sharmin, 12/05/2026)
}

AURUM_SHEETS = {
    # CVD components removed — now HES only (Sharmin, 12/05/2026)
}

AURUM_TEXT_CODELISTS = {
    "ckd": {
        "path"     : "/scratch/alice/b/bg205/Codes/CKD/Clinical_CPRD/AURUM_disease_codes.txt",
        "sep"      : "\t",
        "code_col" : "snomed",
        "term_col" : "term",
    },
    "htn": {
        "path"     : "/scratch/alice/b/bg205/Codes/HYP/Final - codes and terms.txt",
        "sep"      : ",",
        "code_col" : "medcodeid",
        "term_col" : "term",
    },
}

ALLOWED_CODE_OVERLAPS = {
    "GOLD": {},
    "AURUM": {
        299665018: {"ckd", "htn"},
        299673010: {"ckd", "htn"},
        8044061000006119: {"ckd", "htn"},
    },
}

# --- HES ICD-10 condition definitions ---
HES_CONDITIONS = {
    "cvd": {
        "prefixes": ["I"],
        "exclude_prefixes": ["I10", "I11", "I12", "I13", "I14", "I15"],
        "exact_codes": [],
    },
    "htn": {
        "prefixes": ["I10", "I11", "I12", "I13", "I14", "I15"],
        "exclude_prefixes": [],
        "exact_codes": [],
    },
    "ckd": {
        "prefixes": [],
        "exclude_prefixes": [],
        "exact_codes": ["N18.3", "N18.4", "N18.5", "N18.6", "N19", "Z99.2"],
    },
    "cancer_any": {
        "prefixes": ["C"],
        "exclude_prefixes": [],
        "exact_codes": [],
    },
    "cancer_breast": {
        "prefixes": ["C50"],
        "exclude_prefixes": [],
        "exact_codes": [],
    },
    "cancer_lung": {
        "prefixes": ["C33", "C34"],
        "exclude_prefixes": [],
        "exact_codes": [],
    },
    "cancer_colorectal": {
        "prefixes": ["C18", "C19", "C20"],
        "exclude_prefixes": [],
        "exact_codes": [],
    },
    "cancer_prostate": {
        "prefixes": ["C61"],
        "exclude_prefixes": [],
        "exact_codes": [],
    },
    "cancer_pancreatic": {
        "prefixes": ["C25"],
        "exclude_prefixes": [],
        "exact_codes": [],
    },
}

# --- Output ---
OUTPUT_DIR    = "/scratch/alice/b/bg205/16_02_26/comorbidityV2"
OUTPUT_FORMAT = "txt"

# --- Date plausibility floor ---
# Pre-1900 dates are placeholders/artefacts, not real diagnoses.
# Applied to both CPRD and HES event dates before any filtering.
MIN_EVENT_DATE = pd.Timestamp("1900-01-01")

# --- Processing ---
CHUNKSIZE              = 500_000
CONSOLIDATE_EVERY_N_ZIPS = 100

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_df(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FORMAT == "parquet":
        df.to_parquet(path, index=False)
    elif OUTPUT_FORMAT == "txt":
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)
    log.info(f"  Saved → {path}  ({len(df):,} rows)")


def out_path(filename: str) -> str:
    ext = {"parquet": "parquet", "txt": "txt"}.get(OUTPUT_FORMAT, "csv")
    return os.path.join(OUTPUT_DIR, f"{filename}.{ext}")

# ============================================================
# ZIP FILE HELPERS
# ============================================================

def find_zip_files(directory: str, pattern: str) -> list[Path]:
    zips = sorted(Path(directory).glob(pattern))
    if not zips:
        raise FileNotFoundError(
            f"No zip files matching '{pattern}' in '{directory}'."
        )
    log.info(f"  Found {len(zips):,} zip files matching '{pattern}' in {directory}")
    return zips


def get_txt_filename_in_zip(zf: zipfile.ZipFile, zip_path: Path) -> str:
    txt_files = [n for n in zf.namelist() if n.lower().endswith(".txt")]
    if not txt_files:
        raise ValueError(f"No .txt file in {zip_path.name}. Contents: {zf.namelist()}")
    if len(txt_files) > 1:
        log.warning(f"  Multiple .txt files in {zip_path.name}; using: {txt_files[0]}")
    return txt_files[0]

# ============================================================
# CPRD CODELIST PARSING
# ============================================================

def _find_header_row(xl: pd.ExcelFile, sheet: str,
                     keywords: list[str], max_scan: int = 10) -> int:
    probe = xl.parse(sheet, header=None, nrows=max_scan)
    for i, row in probe.iterrows():
        row_lower = row.astype(str).str.lower().tolist()
        if any(kw in " ".join(row_lower) for kw in keywords):
            return i
    log.warning(f"    Cannot detect header row in sheet '{sheet}'; using row 0.")
    return 0


def parse_gold_sheet(xl: pd.ExcelFile, sheet_name: str,
                     comorbidity_label: str) -> pd.DataFrame:
    header_row = _find_header_row(xl, sheet_name,
                                   keywords=["medcode", "readterm", "read", "code"])
    df = xl.parse(sheet_name, header=header_row, dtype=str)
    df.columns = df.columns.str.strip().str.lower()

    code_col = next((c for c in df.columns if "medcode" in c), None)
    if code_col is None:
        raise ValueError(f"Cannot find medcode column in GOLD sheet '{sheet_name}'. "
                         f"Columns: {df.columns.tolist()}")

    term_col = next((c for c in df.columns
                     if any(t in c for t in ["readterm", "term", "description", "read"])), None)

    out = pd.DataFrame()
    out["code"] = pd.to_numeric(df[code_col], errors="coerce")
    out["term"] = df[term_col].str.strip() if term_col else ""
    out["comorbidity"] = comorbidity_label
    out = out.dropna(subset=["code"])
    out["code"] = out["code"].astype(np.int64)
    out = out.drop_duplicates(subset=["code"])
    log.info(f"    GOLD sheet '{sheet_name}' → '{comorbidity_label}': {len(out):,} unique codes")
    return out[["comorbidity", "code", "term"]]


def parse_aurum_sheet(xl: pd.ExcelFile, sheet_name: str,
                      comorbidity_label: str) -> pd.DataFrame:
    header_row = _find_header_row(xl, sheet_name,
                                   keywords=["medcodeid", "term", "description", "code"])
    df = xl.parse(sheet_name, header=header_row, dtype=str)
    df.columns = df.columns.str.strip().str.lower()

    code_col = next((c for c in df.columns if "medcodeid" in c), None)
    if code_col is None:
        raise ValueError(f"Cannot find medcodeid column in Aurum sheet '{sheet_name}'. "
                         f"Columns: {df.columns.tolist()}")

    term_col = next((c for c in df.columns
                     if any(t in c for t in ["term", "description"]) and "code" not in c), None)

    out = pd.DataFrame()
    out["code"] = pd.to_numeric(df[code_col], errors="coerce")
    out["term"] = df[term_col].str.strip() if term_col else ""
    out["comorbidity"] = comorbidity_label
    out = out.dropna(subset=["code"])
    out["code"] = out["code"].astype(np.int64)
    out = out.drop_duplicates(subset=["code"])
    log.info(f"    Aurum sheet '{sheet_name}' → '{comorbidity_label}': {len(out):,} unique codes")
    return out[["comorbidity", "code", "term"]]


def parse_aurum_text_codelist(path: str, sep: str, code_col: str,
                               term_col: str, comorbidity_label: str) -> pd.DataFrame:
    log.info(f"    Loading text codelist: {path}")

    try:
        df = pd.read_csv(path, sep=sep, dtype=str)
    except pd.errors.ParserError:
        log.warning(
            f"    ParserError reading {path}; retrying by splitting each row only on the first separator."
        )

        rows = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip().lower().split(sep, 1)

            if len(header) < 2:
                raise ValueError(f"Could not parse header in {path}: {header}")

            for line_num, line in enumerate(f, start=2):
                line = line.rstrip("\n")
                if not line.strip():
                    continue

                parts = line.split(sep, 1)

                if len(parts) == 1:
                    parts.append("")

                rows.append(parts)

        df = pd.DataFrame(rows, columns=header[:2])

    df.columns = df.columns.str.strip().str.lower()

    if code_col.lower() not in df.columns:
        raise ValueError(f"Code column '{code_col}' not found in {path}. Columns: {df.columns.tolist()}")
    if term_col.lower() not in df.columns:
        raise ValueError(f"Term column '{term_col}' not found in {path}. Columns: {df.columns.tolist()}")

    out = pd.DataFrame()
    out["code"] = pd.to_numeric(df[code_col.lower()], errors="coerce")
    out["term"] = df[term_col.lower()].str.strip()
    out["comorbidity"] = comorbidity_label
    out = out.dropna(subset=["code"])
    out["code"] = out["code"].astype(np.int64)
    out = out.drop_duplicates(subset=["code"])
    log.info(f"    Text codelist '{comorbidity_label}': {len(out):,} unique codes")
    return out[["comorbidity", "code", "term"]]

# ============================================================
# CODE OVERLAP DETECTION
# ============================================================

def _check_code_overlaps(codelist: pd.DataFrame, db_label: str,
                         allowed_overlaps: dict[int, set[str]] | None = None) -> None:
    if allowed_overlaps is None:
        allowed_overlaps = {}

    dup_mask = codelist.duplicated(subset=["code"], keep=False)
    dup_codes = codelist[dup_mask]
    if dup_codes.empty:
        log.info(f"  [{db_label}] Code overlap check: no overlaps found. ✓")
        return

    overlap_summary = (
        dup_codes.groupby("code")["comorbidity"]
        .apply(lambda x: sorted(set(x))).reset_index()
    )

    unexpected = []
    for _, row in overlap_summary.iterrows():
        code = int(row["code"])
        assigned = set(row["comorbidity"])
        allowed = allowed_overlaps.get(code)
        if allowed is not None and assigned == allowed:
            log.warning(f"    [{db_label}] Allowed overlap: {code} → {', '.join(sorted(assigned))}")
        else:
            unexpected.append((code, assigned))

    if unexpected:
        for code, assigned in unexpected:
            log.error(f"    [{db_label}] Unexpected overlap: {code} → {', '.join(sorted(assigned))}")
        raise ValueError(f"[{db_label}] Unexpected code overlaps. See log.")

    log.info(f"  [{db_label}] Code overlap check: only approved overlaps found. ✓")

# ============================================================
# LOAD CPRD CODELISTS
# ============================================================

def load_gold_codelists() -> pd.DataFrame:
    log.info("Loading GOLD codelists...")
    xl = pd.ExcelFile(GOLD_CODELIST_FILE)
    frames = []
    for label, sheet in GOLD_SHEETS.items():
        if sheet not in xl.sheet_names:
            log.warning(f"  Sheet '{sheet}' not found → skipping '{label}'. Available: {xl.sheet_names}")
            continue
        frames.append(parse_gold_sheet(xl, sheet, label))

    if not frames:
        raise RuntimeError(f"No GOLD codelists loaded. Available sheets: {xl.sheet_names}")

    codelist = pd.concat(frames, ignore_index=True)
    _check_code_overlaps(codelist, "GOLD", ALLOWED_CODE_OVERLAPS.get("GOLD", {}))
    log.info(f"GOLD codelists ready: {sorted(codelist['comorbidity'].unique().tolist())}")
    return codelist


def load_aurum_codelists() -> pd.DataFrame:
    log.info("Loading Aurum codelists...")
    xl = pd.ExcelFile(AURUM_CODELIST_FILE)
    frames = []
    for label, sheet in AURUM_SHEETS.items():
        if sheet not in xl.sheet_names:
            log.warning(f"  Sheet '{sheet}' not found → skipping '{label}'. Available: {xl.sheet_names}")
            continue
        frames.append(parse_aurum_sheet(xl, sheet, label))

    for label, cfg in AURUM_TEXT_CODELISTS.items():
        frames.append(parse_aurum_text_codelist(
            path=cfg["path"], sep=cfg["sep"],
            code_col=cfg["code_col"], term_col=cfg["term_col"],
            comorbidity_label=label,
        ))

    if not frames:
        raise RuntimeError("No Aurum codelists loaded.")

    codelist = pd.concat(frames, ignore_index=True)
    _check_code_overlaps(codelist, "Aurum", ALLOWED_CODE_OVERLAPS.get("AURUM", {}))
    log.info(f"Aurum codelists ready: {sorted(codelist['comorbidity'].unique().tolist())}")
    return codelist

# ============================================================
# LOAD COHORT
# ============================================================

def load_cohort() -> pd.DataFrame:
    log.info(f"Loading cohort from {COHORT_FILE}...")

    try:
        probe = pd.read_csv(COHORT_FILE, sep=COHORT_SEP, nrows=0)
    except Exception as e:
        raise RuntimeError(f"Cannot read cohort file: {e}") from e

    probe_cols = [c.strip().lower() for c in probe.columns]
    for req, name in [(COHORT_COL_PATID, "PATID"), (COHORT_COL_INDEX_DATE, "INDEXDATE"),
                      (COHORT_COL_DATABASE, "DATABASE")]:
        if req.lower() not in probe_cols:
            raise ValueError(f"Column '{req}' not found. Available: {probe.columns.tolist()}")

    cohort = pd.read_csv(
        COHORT_FILE, sep=COHORT_SEP, dtype=str,
        usecols=[COHORT_COL_PATID, COHORT_COL_INDEX_DATE, COHORT_COL_DATABASE],
    )
    cohort = cohort.rename(columns={
        COHORT_COL_PATID: "patid", COHORT_COL_INDEX_DATE: "indexdate",
        COHORT_COL_DATABASE: "database",
    })

    cohort["patid"]    = pd.to_numeric(cohort["patid"], errors="coerce")
    cohort["indexdate"] = pd.to_datetime(cohort["indexdate"], dayfirst=False, errors="coerce")
    cohort["database"] = cohort["database"].str.strip().str.upper()

    n_before = len(cohort)
    cohort = cohort.dropna(subset=["patid", "indexdate", "database"])
    cohort["patid"] = cohort["patid"].astype(np.int64)

    dup_mask = cohort.duplicated(subset=["database", "patid"], keep=False)
    if dup_mask.any():
        log.warning(f"  {dup_mask.sum():,} duplicate (database, patid) rows — keeping first.")
    cohort = cohort.drop_duplicates(subset=["database", "patid"])

    dropped = n_before - len(cohort)
    if dropped:
        log.warning(f"  Dropped {dropped:,} rows with missing patid/indexdate.")

    n_gold  = (cohort["database"] == "GOLD").sum()
    n_aurum = (cohort["database"] == "AURUM").sum()
    log.info(f"  Cohort: {len(cohort):,} patients (GOLD: {n_gold:,}, AURUM: {n_aurum:,})")
    return cohort

# ============================================================
# CPRD CHUNK PROCESSING
# ============================================================

def _process_cprd_chunk(
    chunk: pd.DataFrame, col_patid: str, col_code: str, col_date: str,
    cohort_patids: set, cohort_idx: pd.Series,
    code_map: dict, valid_codes: set, qc_accum: dict,
) -> pd.DataFrame | None:

    chunk.columns = chunk.columns.str.strip().str.lower()
    needed = {col_patid.lower(), col_code.lower(), col_date.lower()}
    available = [c for c in chunk.columns if c in needed]
    chunk = chunk[available].copy()
    chunk = chunk.rename(columns={
        col_patid.lower(): "patid", col_code.lower(): "code", col_date.lower(): "event_date",
    })

    chunk["patid"] = pd.to_numeric(chunk["patid"], errors="coerce")
    chunk["code"]  = pd.to_numeric(chunk["code"], errors="coerce")
    chunk.dropna(subset=["patid", "code"], inplace=True)
    if chunk.empty:
        return None

    chunk["patid"] = chunk["patid"].astype(np.int64)
    chunk["code"]  = chunk["code"].astype(np.int64)

    chunk = chunk[chunk["patid"].isin(cohort_patids)]
    if chunk.empty:
        return None
    chunk = chunk[chunk["code"].isin(valid_codes)]
    if chunk.empty:
        return None

    chunk["comorbidity"] = chunk["code"].map(code_map)
    chunk = chunk.explode("comorbidity", ignore_index=True)
    chunk = chunk[chunk["comorbidity"].notna()].copy()
    if chunk.empty:
        return None
    
    # Parse dates — keep NaT so we can measure missingness BEFORE dropping
    chunk["event_date"] = pd.to_datetime(
        chunk["event_date"], dayfirst=False, errors="coerce"
    )
    # Set pre-1900 placeholder dates to NaT
    chunk.loc[chunk["event_date"] < MIN_EVENT_DATE, "event_date"] = pd.NaT
    chunk["_date_missing"] = chunk["event_date"].isna()

    # QC accumulation — before date drop
    for comorbidity, grp in chunk.groupby("comorbidity", sort=False):
        if comorbidity not in qc_accum:
            qc_accum[comorbidity] = {
                "n_records": 0, "n_records_missing": 0,
                "patids_any": set(), "patids_missing": set(), "patids_valid": set(),
            }
        acc = qc_accum[comorbidity]
        acc["n_records"] += len(grp)
        missing_mask = grp["_date_missing"]
        n_miss = int(missing_mask.sum())
        acc["n_records_missing"] += n_miss
        acc["patids_any"].update(grp["patid"].values.tolist())
        if n_miss > 0:
            acc["patids_missing"].update(grp.loc[missing_mask, "patid"].values.tolist())
        if n_miss < len(grp):
            acc["patids_valid"].update(grp.loc[~missing_mask, "patid"].values.tolist())

    chunk = chunk[~chunk["_date_missing"]].copy()
    if chunk.empty:
        return None

    chunk["indexdate"] = chunk["patid"].map(cohort_idx)
    chunk = chunk[chunk["event_date"] <= chunk["indexdate"]]
    if chunk.empty:
        return None

    return (
        chunk.groupby(["patid", "comorbidity"], sort=False)["event_date"]
        .agg(first_date="min", last_date="max").reset_index()
    )

# ============================================================
# CPRD MISSING DATE QC
# ============================================================

def _build_missing_date_qc(qc_accum: dict, db_label: str,
                            all_comorbidities: list[str]) -> pd.DataFrame:
    rows = []
    for c in sorted(all_comorbidities):
        acc = qc_accum.get(c, {"n_records": 0, "n_records_missing": 0,
                                "patids_any": set(), "patids_missing": set(), "patids_valid": set()})
        n_any = len(acc["patids_any"])
        n_miss = len(acc["patids_missing"])
        n_valid = len(acc["patids_valid"])
        n_only_miss = len(acc["patids_any"] - acc["patids_valid"])
        pct = round(100.0 * n_miss / n_any, 1) if n_any > 0 else 0.0
        rows.append({
            "database": db_label, "comorbidity": c,
            "n_records_matching_code": acc["n_records"],
            "n_records_missing_date": acc["n_records_missing"],
            "n_patients_matching_code": n_any,
            "n_patients_with_any_missing_date_record": n_miss,
            "n_patients_with_any_valid_dated_record": n_valid,
            "n_patients_only_missing_dated_records": n_only_miss,
            "pct_patients_with_any_missing_date_record": pct,
        })

    qc_df = pd.DataFrame(rows)
    log.info(f"\n{'='*80}\nMISSING DATE QC — {db_label}\n{'='*80}")
    log.info(f"  {'Comorbidity':<25} {'Rec':>8} {'Miss':>8} {'Pat':>8} "
             f"{'Miss':>8} {'Valid':>8} {'Only':>8} {'%':>6}")
    log.info(f"  {'-'*80}")
    for _, r in qc_df.iterrows():
        log.info(f"  {r['comorbidity']:<25} {r['n_records_matching_code']:>8,} "
                 f"{r['n_records_missing_date']:>8,} {r['n_patients_matching_code']:>8,} "
                 f"{r['n_patients_with_any_missing_date_record']:>8,} "
                 f"{r['n_patients_with_any_valid_dated_record']:>8,} "
                 f"{r['n_patients_only_missing_dated_records']:>8,} "
                 f"{r['pct_patients_with_any_missing_date_record']:>5.1f}%")
    log.info(f"{'='*80}\n")
    return qc_df

# ============================================================
# CPRD EXTRACTION ACROSS ZIP FILES
# ============================================================

def _consolidate_summaries(summaries: list[pd.DataFrame]) -> list[pd.DataFrame]:
    if len(summaries) <= 1:
        return summaries
    combined = pd.concat(summaries, ignore_index=True)
    consolidated = (
        combined.groupby(["patid", "comorbidity"], sort=False)
        .agg(first_date=("first_date", "min"), last_date=("last_date", "max"))
        .reset_index()
    )
    del combined
    return [consolidated]


def extract_cprd_preindex_records(
    zip_files: list[Path], sep: str, col_patid: str, col_code: str,
    col_date: str, codelist: pd.DataFrame, cohort: pd.DataFrame,
    db_label: str = "DB",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    cohort_patids = set(cohort["patid"].tolist())
    cohort_idx    = cohort.set_index("patid")["indexdate"]
    code_map      = codelist.groupby("code")["comorbidity"].apply(lambda s: sorted(set(s))).to_dict()
    valid_codes   = set(code_map.keys())
    all_comorb    = sorted(codelist["comorbidity"].unique().tolist())

    qc_accum = {c: {"n_records": 0, "n_records_missing": 0,
                     "patids_any": set(), "patids_missing": set(), "patids_valid": set()}
                for c in all_comorb}

    chunk_summaries = []
    total_rows = 0
    kept_rows = 0
    col_validated = False

    log.info(f"[{db_label}] Starting CPRD extraction: {len(zip_files):,} zips, "
             f"{len(valid_codes):,} codes, conditions: {all_comorb}")

    for zip_num, zip_path in enumerate(zip_files, 1):
        if zip_num == 1 or zip_num % 50 == 0:
            log.info(f"  [{db_label}] Zip {zip_num}/{len(zip_files)}: {zip_path.name} | "
                     f"scanned: {total_rows:,}, kept: {kept_rows:,}")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                txt_name = get_txt_filename_in_zip(zf, zip_path)
                with zf.open(txt_name) as raw_file:
                    text_stream = io.TextIOWrapper(raw_file, encoding="utf-8", errors="replace")
                    reader = pd.read_csv(text_stream, sep=sep, dtype=str,
                                          chunksize=CHUNKSIZE, low_memory=False)
                    for chunk in reader:
                        total_rows += len(chunk)

                        if not col_validated:
                            cols_lower = [c.strip().lower() for c in chunk.columns]
                            for cfg_col, cfg_name in [(col_patid, "patid"), (col_code, "code"),
                                                       (col_date, "date")]:
                                if cfg_col.lower() not in cols_lower:
                                    raise ValueError(
                                        f"[{db_label}] Column '{cfg_col}' not found in "
                                        f"{zip_path.name}. Available: {cols_lower}")
                            col_validated = True

                        result = _process_cprd_chunk(
                            chunk, col_patid, col_code, col_date,
                            cohort_patids, cohort_idx, code_map, valid_codes, qc_accum)

                        if result is not None and not result.empty:
                            kept_rows += len(result)
                            chunk_summaries.append(result)

                        del chunk; gc.collect()
        except zipfile.BadZipFile:
            log.error(f"  Skipping corrupt zip: {zip_path.name}")
            continue

        if zip_num % CONSOLIDATE_EVERY_N_ZIPS == 0 and chunk_summaries:
            log.info(f"  [{db_label}] Consolidating at zip {zip_num}...")
            chunk_summaries = _consolidate_summaries(chunk_summaries)
            gc.collect()

    log.info(f"[{db_label}] CPRD extraction complete: scanned {total_rows:,}, kept {kept_rows:,}")

    missing_date_qc = _build_missing_date_qc(qc_accum, db_label, all_comorb)
    del qc_accum; gc.collect()

    if not chunk_summaries:
        log.warning(f"[{db_label}] No CPRD records matched.")
        return pd.DataFrame(columns=["patid", "comorbidity", "first_date", "last_date"]), missing_date_qc

    chunk_summaries = _consolidate_summaries(chunk_summaries)
    final = chunk_summaries[0]
    del chunk_summaries; gc.collect()
    return final, missing_date_qc

# ============================================================
# CPRD PIVOT TO WIDE
# ============================================================

def pivot_cprd_to_wide(summary: pd.DataFrame, cohort: pd.DataFrame,
                        db_label: str) -> pd.DataFrame:
    if summary.empty:
        log.warning(f"[{db_label}] Summary empty; returning cohort shell.")
        return cohort[["patid", "indexdate"]].copy()

    comorbidities = summary["comorbidity"].unique().tolist()

    first_wide = summary.pivot_table(index="patid", columns="comorbidity",
                                      values="first_date", aggfunc="min")
    first_wide.columns = [f"{c}_cprd_first_date" for c in first_wide.columns]

    last_wide = summary.pivot_table(index="patid", columns="comorbidity",
                                     values="last_date", aggfunc="max")
    last_wide.columns = [f"{c}_cprd_last_date" for c in last_wide.columns]

    wide = first_wide.join(last_wide, how="outer").reset_index()
    result = cohort[["patid", "indexdate"]].merge(wide, on="patid", how="left")

    for c in comorbidities:
        first_col = f"{c}_cprd_first_date"
        result[f"{c}_cprd_bin"] = (
            result[first_col].notna().astype(np.int8) if first_col in result.columns
            else np.int8(0)
        )
        result[f"{c}_cprd_duration_days"] = np.where(
            result[f"{c}_cprd_bin"] == 1,
            (result["indexdate"] - result[first_col]).dt.days, np.nan)
        result[f"{c}_cprd_duration_years"] = result[f"{c}_cprd_duration_days"] / 365.25

    ordered = ["patid", "indexdate"]
    for c in sorted(comorbidities):
        for suf in ["_cprd_bin", "_cprd_first_date", "_cprd_last_date",
                     "_cprd_duration_days", "_cprd_duration_years"]:
            col = f"{c}{suf}"
            if col in result.columns:
                ordered.append(col)
    return result[ordered]

# ============================================================
# HES ICD-10 MATCHING
# ============================================================

def classify_icd_code(icd: str) -> list[str]:
    if not icd or pd.isna(icd):
        return []
    icd = str(icd).strip().upper()
    matched = []
    for condition, rules in HES_CONDITIONS.items():
        if any(icd.startswith(ex) for ex in rules["exclude_prefixes"]):
            continue
        prefix_match = any(icd.startswith(p) for p in rules["prefixes"])
        exact_match = icd in rules["exact_codes"]
        icd_nodot = icd.replace(".", "")
        exact_nodot = any(icd_nodot == c.replace(".", "") for c in rules["exact_codes"])
        if prefix_match or exact_match or exact_nodot:
            matched.append(condition)
    return matched


def build_icd_lookup(codes: set[str]) -> dict[str, list[str]]:
    lookup = {}
    for code in codes:
        conditions = classify_icd_code(code)
        lookup[code] = conditions  # empty list if no match — avoids re-checking
    return lookup

# ============================================================
# HES EXTRACTION
# ============================================================

def extract_hes_comorbidities(
    hes_file: str, cohort: pd.DataFrame, db_label: str,
) -> tuple[pd.DataFrame, dict]:

    log.info(f"\n{'='*60}\n  HES EXTRACTION — {db_label}\n  File: {hes_file}\n{'='*60}")

    if not os.path.exists(hes_file):
        log.warning(f"  HES file not found: {hes_file} — skipping {db_label}")
        return pd.DataFrame(columns=["patid", "condition", "first_date", "last_date"]), {}

    cohort_patids = set(cohort["patid"].tolist())
    cohort_idx    = cohort.set_index("patid")["indexdate"]

    qc = {cond: {"n_records": 0, "n_records_preindex": 0, "patids": set()}
          for cond in HES_CONDITIONS}

    chunk_summaries = []
    total_rows = 0
    matched_rows = 0
    icd_lookup: dict[str, list[str]] = {}

    reader = pd.read_csv(hes_file, sep=HES_SEP, dtype=str,
                          chunksize=CHUNKSIZE, low_memory=False)

    for chunk_num, chunk in enumerate(reader, 1):
        chunk.columns = chunk.columns.str.strip().str.lower()
        total_rows += len(chunk)

        if chunk_num == 1:
            log.info(f"  Columns: {chunk.columns.tolist()}")

        col_map = {HES_COL_PATID.lower(): "patid", HES_COL_ICD.lower(): "icd",
                    HES_COL_EPISTART.lower(): "epistart"}
        for old, new in col_map.items():
            if old not in chunk.columns:
                raise ValueError(f"[{db_label}] Column '{old}' not found. Available: {chunk.columns.tolist()}")
        chunk = chunk.rename(columns=col_map)
        chunk = chunk[["patid", "icd", "epistart"]].copy()

        chunk["patid"] = pd.to_numeric(chunk["patid"], errors="coerce")
        chunk = chunk.dropna(subset=["patid"])
        chunk["patid"] = chunk["patid"].astype(np.int64)
        chunk = chunk[chunk["patid"].isin(cohort_patids)]
        if chunk.empty:
            continue

        chunk["icd"] = chunk["icd"].astype(str).str.strip().str.upper()
        chunk = chunk[chunk["icd"].notna() & (chunk["icd"] != "") & (chunk["icd"] != "NAN")]
        if chunk.empty:
            continue

        # Update ICD lookup with new codes
        new_codes = set(chunk["icd"].unique()) - set(icd_lookup.keys())
        if new_codes:
            icd_lookup.update(build_icd_lookup(new_codes))

        chunk["conditions"] = chunk["icd"].map(icd_lookup)
        chunk = chunk[chunk["conditions"].apply(lambda x: len(x) > 0)]
        if chunk.empty:
            continue

        matched_rows += len(chunk)
        chunk = chunk.explode("conditions", ignore_index=True)
        chunk = chunk.rename(columns={"conditions": "condition"})
        
        chunk["epistart"] = pd.to_datetime(chunk["epistart"], dayfirst=True, errors="coerce")
        chunk.loc[chunk["epistart"] < MIN_EVENT_DATE, "epistart"] = pd.NaT
        chunk = chunk.dropna(subset=["epistart"])
        if chunk.empty:
            continue

        chunk["indexdate"] = chunk["patid"].map(cohort_idx)

        # QC — all matched
        for cond, grp in chunk.groupby("condition", sort=False):
            qc[cond]["n_records"] += len(grp)

        preindex = chunk[chunk["epistart"] <= chunk["indexdate"]].copy()

        for cond, grp in preindex.groupby("condition", sort=False):
            qc[cond]["n_records_preindex"] += len(grp)
            qc[cond]["patids"].update(grp["patid"].tolist())

        if not preindex.empty:
            summary = (
                preindex.groupby(["patid", "condition"], sort=False)["epistart"]
                .agg(first_date="min", last_date="max").reset_index()
            )
            chunk_summaries.append(summary)

        if chunk_num % 20 == 0:
            log.info(f"  [{db_label}] Chunk {chunk_num}: scanned {total_rows:,}, "
                     f"matched {matched_rows:,}")

        del chunk, preindex; gc.collect()

    log.info(f"  [{db_label}] HES complete: scanned {total_rows:,}, matched {matched_rows:,}")

    if not chunk_summaries:
        log.warning(f"  [{db_label}] No HES records matched.")
        return pd.DataFrame(columns=["patid", "condition", "first_date", "last_date"]), qc

    combined = pd.concat(chunk_summaries, ignore_index=True)
    final = (
        combined.groupby(["patid", "condition"], sort=False)
        .agg(first_date=("first_date", "min"), last_date=("last_date", "max"))
        .reset_index()
    )
    del combined, chunk_summaries; gc.collect()

    log.info(f"    Summary rows: {len(final):,}")
    return final, qc

# ============================================================
# HES PIVOT TO WIDE
# ============================================================

def pivot_hes_to_wide(summary: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    all_conditions = sorted(HES_CONDITIONS.keys())

    if summary.empty:
        wide = cohort[["patid", "indexdate", "database"]].copy()
        for cond in all_conditions:
            wide[f"{cond}_hes_bin"] = np.int8(0)
            wide[f"{cond}_hes_first_date"] = pd.NaT
            wide[f"{cond}_hes_last_date"] = pd.NaT
            wide[f"{cond}_hes_duration_days"] = np.nan
            wide[f"{cond}_hes_duration_years"] = np.nan
        return wide

    first_wide = summary.pivot_table(index="patid", columns="condition",
                                      values="first_date", aggfunc="min")
    first_wide.columns = [f"{c}_hes_first_date" for c in first_wide.columns]

    last_wide = summary.pivot_table(index="patid", columns="condition",
                                     values="last_date", aggfunc="max")
    last_wide.columns = [f"{c}_hes_last_date" for c in last_wide.columns]

    wide = first_wide.join(last_wide, how="outer").reset_index()
    result = cohort[["patid", "indexdate", "database"]].merge(wide, on="patid", how="left")

    for cond in all_conditions:
        first_col = f"{cond}_hes_first_date"
        if first_col not in result.columns:
            result[first_col] = pd.NaT
        if f"{cond}_hes_last_date" not in result.columns:
            result[f"{cond}_hes_last_date"] = pd.NaT

        result[f"{cond}_hes_bin"] = result[first_col].notna().astype(np.int8)
        result[f"{cond}_hes_duration_days"] = np.where(
            result[f"{cond}_hes_bin"] == 1,
            (result["indexdate"] - result[first_col]).dt.days, np.nan)
        result[f"{cond}_hes_duration_years"] = result[f"{cond}_hes_duration_days"] / 365.25

    ordered = ["patid", "indexdate", "database"]
    for cond in all_conditions:
        ordered += [f"{cond}_hes_bin", f"{cond}_hes_first_date", f"{cond}_hes_last_date",
                    f"{cond}_hes_duration_days", f"{cond}_hes_duration_years"]
    return result[[c for c in ordered if c in result.columns]]

# ============================================================
# COMBINE CPRD + HES INTO FINAL FLAGS
# ============================================================

def combine_cprd_hes(
    cohort:     pd.DataFrame,
    cprd_wide:  pd.DataFrame | None,
    hes_wide:   pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Merge CPRD and HES flags into final patient-level comorbidity variables.

    Logic:
      CKD, HTN: final = max(cprd, hes); first_date = min(cprd, hes)
      CVD, cancer_*: HES only (no CPRD source)
    """
    log.info(f"\n{'='*60}\n  COMBINING CPRD + HES FLAGS\n{'='*60}")

    final = cohort[["patid", "indexdate", "database"]].copy()

    # --- CKD and HTN: CPRD OR HES ---
    for cond in ["ckd", "htn"]:
        cprd_bin_col   = f"{cond}_cprd_bin"
        cprd_first_col = f"{cond}_cprd_first_date"
        hes_bin_col    = f"{cond}_hes_bin"
        hes_first_col  = f"{cond}_hes_first_date"

        # Get CPRD values
        if cprd_wide is not None and cprd_bin_col in cprd_wide.columns:
            final = final.merge(
                cprd_wide[["patid", cprd_bin_col, cprd_first_col]],
                on="patid", how="left"
            )
            final[cprd_bin_col] = final[cprd_bin_col].fillna(0).astype(np.int8)
        else:
            final[cprd_bin_col] = np.int8(0)
            final[cprd_first_col] = pd.NaT

        # Get HES values
        if hes_wide is not None and hes_bin_col in hes_wide.columns:
            final = final.merge(
                hes_wide[["patid", hes_bin_col, hes_first_col]],
                on="patid", how="left"
            )
            final[hes_bin_col] = final[hes_bin_col].fillna(0).astype(np.int8)
        else:
            final[hes_bin_col] = np.int8(0)
            final[hes_first_col] = pd.NaT

        # Combined
        final[f"{cond}_bin"] = final[[cprd_bin_col, hes_bin_col]].max(axis=1).astype(np.int8)
        final[f"{cond}_first_date"] = final[[cprd_first_col, hes_first_col]].min(axis=1)
        final[f"{cond}_duration_days"] = np.where(
            final[f"{cond}_bin"] == 1,
            (final["indexdate"] - final[f"{cond}_first_date"]).dt.days, np.nan)
        final[f"{cond}_duration_years"] = final[f"{cond}_duration_days"] / 365.25

        # Drop intermediate columns
        final = final.drop(columns=[cprd_bin_col, cprd_first_col,
                                     hes_bin_col, hes_first_col])

    # --- CVD and Cancer: HES only ---
    hes_only_conditions = [c for c in HES_CONDITIONS if c not in ["ckd", "htn"]]
    for cond in hes_only_conditions:
        hes_bin_col   = f"{cond}_hes_bin"
        hes_first_col = f"{cond}_hes_first_date"

        if hes_wide is not None and hes_bin_col in hes_wide.columns:
            final = final.merge(
                hes_wide[["patid", hes_bin_col, hes_first_col]],
                on="patid", how="left"
            )
            final[f"{cond}_bin"] = final[hes_bin_col].fillna(0).astype(np.int8)
            final[f"{cond}_first_date"] = final[hes_first_col]
            final = final.drop(columns=[hes_bin_col, hes_first_col])
        else:
            final[f"{cond}_bin"] = np.int8(0)
            final[f"{cond}_first_date"] = pd.NaT

        final[f"{cond}_duration_days"] = np.where(
            final[f"{cond}_bin"] == 1,
            (final["indexdate"] - final[f"{cond}_first_date"]).dt.days, np.nan)
        final[f"{cond}_duration_years"] = final[f"{cond}_duration_days"] / 365.25

    # --- Order columns ---
    all_conditions = ["ckd", "htn"] + sorted(hes_only_conditions)
    ordered = ["patid", "indexdate", "database"]
    for cond in all_conditions:
        ordered += [f"{cond}_bin", f"{cond}_first_date",
                    f"{cond}_duration_days", f"{cond}_duration_years"]
    final = final[[c for c in ordered if c in final.columns]]

    log.info(f"  Final combined output: {len(final):,} rows, {len(final.columns)} columns")
    return final

# ============================================================
# QC SUMMARY
# ============================================================

def print_final_qc(final: pd.DataFrame) -> pd.DataFrame:
    n_total = len(final)
    n_gold  = (final["database"] == "GOLD").sum()
    n_aurum = (final["database"] == "AURUM").sum()

    log.info(f"\n{'='*80}\n  FINAL COMORBIDITY QC SUMMARY\n{'='*80}")
    log.info(f"  Cohort: {n_total:,} (GOLD: {n_gold:,}, AURUM: {n_aurum:,})")

    bin_cols = sorted([c for c in final.columns if c.endswith("_bin")])

    log.info(f"\n  {'Condition':<25} {'N':>8} {'%':>7}   "
             f"{'GOLD n':>8} {'GOLD %':>7}   "
             f"{'Aurum n':>8} {'Aurum %':>7}   "
             f"{'First date range'}")
    log.info(f"  {'-'*110}")

    qc_rows = []
    for col in bin_cols:
        cond = col.replace("_bin", "")
        first_col = f"{cond}_first_date"

        n_pos = int(final[col].sum())
        pct   = 100 * n_pos / n_total if n_total > 0 else 0

        n_g = int(final.loc[final["database"] == "GOLD", col].sum())
        pct_g = 100 * n_g / n_gold if n_gold > 0 else 0
        n_a = int(final.loc[final["database"] == "AURUM", col].sum())
        pct_a = 100 * n_a / n_aurum if n_aurum > 0 else 0

        fmin = final[first_col].min() if first_col in final.columns else ""
        fmax = final[first_col].max() if first_col in final.columns else ""

        log.info(f"  {cond:<25} {n_pos:>8,} {pct:>6.1f}%   "
                 f"{n_g:>8,} {pct_g:>6.1f}%   {n_a:>8,} {pct_a:>6.1f}%   "
                 f"[{fmin} – {fmax}]")

        qc_rows.append({"condition": cond, "n_total": n_pos, "pct_total": round(pct, 2),
                         "n_gold": n_g, "pct_gold": round(pct_g, 2),
                         "n_aurum": n_a, "pct_aurum": round(pct_a, 2),
                         "earliest_first_date": str(fmin), "latest_first_date": str(fmax)})

    # Negative duration check
    log.info(f"\n  Negative-duration check:")
    any_neg = False
    for col in bin_cols:
        cond = col.replace("_bin", "")
        dur_col = f"{cond}_duration_days"
        if dur_col in final.columns:
            neg = (final[dur_col] < 0).sum()
            if neg > 0:
                log.warning(f"    WARNING — {cond}: {neg:,} negative durations!")
                any_neg = True
    if not any_neg:
        log.info(f"    PASS — no negative durations.")

    # Duration summary
    log.info(f"\n  Duration summary (years):")
    for col in bin_cols:
        cond = col.replace("_bin", "")
        dur_col = f"{cond}_duration_years"
        if dur_col in final.columns:
            vals = final.loc[final[col] == 1, dur_col].dropna()
            if len(vals) > 0:
                log.info(f"    {cond:<25} n={len(vals):>7,}  "
                         f"median={vals.median():.1f}  mean={vals.mean():.1f}  "
                         f"min={vals.min():.1f}  max={vals.max():.1f}")

    log.info(f"\n  Final rows: {len(final):,}")
    log.info(f"{'='*80}\n")
    return pd.DataFrame(qc_rows)

# ============================================================
# CPRD-ONLY QC
# ============================================================

def save_cprd_qc_summary(wide_gold: pd.DataFrame | None,
                          wide_aurum: pd.DataFrame | None) -> None:
    rows = []
    for wide, db in [(wide_gold, "GOLD"), (wide_aurum, "Aurum")]:
        if wide is None:
            continue
        n = len(wide)
        for col in [c for c in wide.columns if c.endswith("_cprd_bin")]:
            n_pos = int(wide[col].sum())
            rows.append({"database": db, "comorbidity": col, "n_cohort": n,
                         "n_positive": n_pos,
                         "prevalence_pct": round(100 * n_pos / n, 2) if n else 0})
    if rows:
        save_df(pd.DataFrame(rows), out_path("qc_prevalence_summary"))

# ============================================================
# CPRD PIPELINES
# ============================================================

def run_gold_pipeline(cohort: pd.DataFrame) -> pd.DataFrame:
    log.info("\n" + "=" * 60 + "\nCPRD GOLD PIPELINE\n" + "=" * 60)

    gold_codes = load_gold_codelists()
    save_df(gold_codes, out_path("gold_codelists_parsed"))

    zip_files = find_zip_files(GOLD_ZIP_DIR, GOLD_ZIP_PATTERN)

    summary, qc_missing = extract_cprd_preindex_records(
        zip_files, GOLD_SEP, GOLD_COL_PATID, GOLD_COL_MEDCODE,
        GOLD_COL_EVENTDATE, gold_codes, cohort, "GOLD")

    save_df(summary, out_path("gold_long_summary"))
    save_df(qc_missing, out_path("gold_missing_date_qc"))
    del qc_missing; gc.collect()

    wide = pivot_cprd_to_wide(summary, cohort, "GOLD")
    del summary; gc.collect()

    save_df(wide, out_path("gold_wide_baseline_comorbidities"))
    return wide


def run_aurum_pipeline(cohort: pd.DataFrame) -> pd.DataFrame:
    log.info("\n" + "=" * 60 + "\nCPRD AURUM PIPELINE\n" + "=" * 60)

    aurum_codes = load_aurum_codelists()
    save_df(aurum_codes, out_path("aurum_codelists_parsed"))

    zip_files = find_zip_files(AURUM_ZIP_DIR, AURUM_ZIP_PATTERN)

    summary, qc_missing = extract_cprd_preindex_records(
        zip_files, AURUM_SEP, AURUM_COL_PATID, AURUM_COL_MEDCODEID,
        AURUM_COL_OBSDATE, aurum_codes, cohort, "Aurum")

    save_df(summary, out_path("aurum_long_summary"))
    save_df(qc_missing, out_path("aurum_missing_date_qc"))
    del qc_missing; gc.collect()

    wide = pivot_cprd_to_wide(summary, cohort, "Aurum")
    del summary; gc.collect()

    save_df(wide, out_path("aurum_wide_baseline_comorbidities"))
    return wide

# ============================================================
# HES PIPELINE
# ============================================================

def run_hes_pipeline(cohort: pd.DataFrame) -> pd.DataFrame:
    log.info("\n" + "=" * 60 + "\nHES ICD-10 PIPELINE\n" + "=" * 60)

    # Save codelist reference
    cl_rows = []
    for cond, rules in HES_CONDITIONS.items():
        cl_rows.append({
            "condition": cond,
            "include_prefixes": ", ".join(rules["prefixes"]) if rules["prefixes"] else "",
            "exclude_prefixes": ", ".join(rules["exclude_prefixes"]) if rules["exclude_prefixes"] else "",
            "exact_codes": ", ".join(rules["exact_codes"]) if rules["exact_codes"] else "",
        })
    save_df(pd.DataFrame(cl_rows), out_path("hes_codelists_applied"))

    cohort_gold  = cohort[cohort["database"] == "GOLD"].copy()
    cohort_aurum = cohort[cohort["database"] == "AURUM"].copy()

        # Extract from GOLD-linked HES
    gold_summary, qc_gold = extract_hes_comorbidities(
        HES_GOLD_FILE, cohort_gold, "GOLD")

    # Extract from Aurum-linked HES only if RUN_AURUM = True
    if RUN_AURUM and not cohort_aurum.empty:
        aurum_summary, qc_aurum = extract_hes_comorbidities(
            HES_AURUM_FILE, cohort_aurum, "Aurum")
    else:
        log.info("Skipping Aurum HES extraction because RUN_AURUM = False")
        aurum_summary = pd.DataFrame(columns=["patid", "condition", "first_date", "last_date"])
        qc_aurum = {cond: {"n_records": 0, "n_records_preindex": 0, "patids": set()}
                    for cond in HES_CONDITIONS}

    # Combine
    combined_summary = pd.concat([gold_summary, aurum_summary], ignore_index=True)
    log.info(f"  Combined HES summary: {len(combined_summary):,} rows")

    save_df(combined_summary, out_path("hes_long_summary"))
    del gold_summary, aurum_summary; gc.collect()

    # Pivot to wide
    wide = pivot_hes_to_wide(combined_summary, cohort)
    del combined_summary; gc.collect()

    save_df(wide, out_path("hes_wide_baseline_comorbidities"))

    # HES QC summary
    hes_qc_rows = []
    n_total = len(wide)
    n_gold  = (wide["database"] == "GOLD").sum()
    n_aurum = (wide["database"] == "AURUM").sum()

    log.info(f"\n{'='*80}\n  HES QC SUMMARY\n{'='*80}")
    log.info(f"  Cohort: {n_total:,} (GOLD: {n_gold:,}, AURUM: {n_aurum:,})")

    # Record counts
    log.info(f"\n  HES record counts:")
    log.info(f"  {'Condition':<25} {'GOLD rec':>10} {'GOLD pre':>10} "
             f"{'Aurum rec':>10} {'Aurum pre':>10}")
    log.info(f"  {'-'*70}")
    for cond in sorted(HES_CONDITIONS.keys()):
        g = qc_gold.get(cond, {"n_records": 0, "n_records_preindex": 0})
        a = qc_aurum.get(cond, {"n_records": 0, "n_records_preindex": 0})
        log.info(f"  {cond:<25} {g['n_records']:>10,} {g['n_records_preindex']:>10,} "
                 f"{a['n_records']:>10,} {a['n_records_preindex']:>10,}")

    # Prevalence
    log.info(f"\n  HES prevalence:")
    for cond in sorted(HES_CONDITIONS.keys()):
        bin_col = f"{cond}_hes_bin"
        if bin_col in wide.columns:
            n_pos = int(wide[bin_col].sum())
            pct = 100 * n_pos / n_total if n_total > 0 else 0
            n_g = int(wide.loc[wide["database"] == "GOLD", bin_col].sum())
            n_a = int(wide.loc[wide["database"] == "AURUM", bin_col].sum())
            log.info(f"    {cond:<25} n={n_pos:>8,} ({pct:5.1f}%)  "
                     f"GOLD: {n_g:>7,}  AURUM: {n_a:>7,}")

            hes_qc_rows.append({"condition": cond, "n_total": n_pos,
                                "pct_total": round(pct, 2),
                                "n_gold": n_g, "n_aurum": n_a})
    log.info(f"{'='*80}\n")

    save_df(pd.DataFrame(hes_qc_rows), out_path("hes_qc_summary"))
    return wide

# ============================================================
# MAIN
# ============================================================

def main():
    log.info("=" * 60)
    log.info("Baseline Comorbidity Extraction (CPRD + HES) — START")
    log.info("=" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    cohort = load_cohort()
    cohort_gold  = cohort[cohort["database"] == "GOLD"].copy()
    cohort_aurum = cohort[cohort["database"] == "AURUM"].copy()

    log.info(f"  Total: {len(cohort):,}  GOLD: {len(cohort_gold):,}  AURUM: {len(cohort_aurum):,}")

    # ---- CPRD extraction (CKD + HTN only) ----
    wide_gold  = None
    wide_aurum = None
    cprd_wide  = None

    if RUN_GOLD and not cohort_gold.empty:
        wide_gold = run_gold_pipeline(cohort_gold)

    if RUN_AURUM and not cohort_aurum.empty:
        wide_aurum = run_aurum_pipeline(cohort_aurum)

    save_cprd_qc_summary(wide_gold, wide_aurum)

    # Stack GOLD + Aurum CPRD wide outputs
    cprd_parts = [w for w in [wide_gold, wide_aurum] if w is not None]
    if cprd_parts:
        cprd_wide = pd.concat(cprd_parts, ignore_index=True)
        log.info(f"  Combined CPRD wide: {len(cprd_wide):,} rows")
    del wide_gold, wide_aurum; gc.collect()

    # ---- HES extraction (CVD, HTN, CKD, Cancer) ----
    hes_wide = None
    if RUN_HES:
        hes_wide = run_hes_pipeline(cohort)

    # ---- Combine CPRD + HES ----
    final = combine_cprd_hes(cohort, cprd_wide, hes_wide)
    del cprd_wide, hes_wide; gc.collect()

    # ---- Final QC ----
    qc_df = print_final_qc(final)
    save_df(qc_df, out_path("final_qc_summary"))

    # ---- Row count check ----
    assert len(final) == len(cohort), (
        f"ROW COUNT MISMATCH — expected {len(cohort):,}, got {len(final):,}")

    # ---- Save final output ----
    save_df(final, out_path("final_combined_comorbidities"))

    log.info("\n" + "=" * 60)
    log.info("Baseline Comorbidity Extraction — COMPLETE")
    log.info("=" * 60)
    log.info("  Outputs:")
    for f in ["gold_codelists_parsed", "gold_long_summary", "gold_wide_baseline_comorbidities",
              "gold_missing_date_qc", "aurum_codelists_parsed", "aurum_long_summary",
              "aurum_wide_baseline_comorbidities", "aurum_missing_date_qc",
              "hes_codelists_applied", "hes_long_summary", "hes_wide_baseline_comorbidities",
              "hes_qc_summary", "final_combined_comorbidities", "final_qc_summary",
              "qc_prevalence_summary"]:
        log.info(f"    {out_path(f)}")


if __name__ == "__main__":
    main()
