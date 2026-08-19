import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_curve, roc_auc_score, brier_score_loss, log_loss,
                             precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pickle
import argparse
import os
import gc
import hashlib
import json
from missforest import MissForest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import warnings
from sklearn.exceptions import ConvergenceWarning

SCRIPT_VERSION = "mortality_pipeline_v3.2_manuscript"

# =============================================================================
# CHANGES vs previous version
#   1. Fit MissForest on TRAIN and transform TEST once per unique, outcome-
#      specific predictor set, then reuse it for all models and hybrid subsets.
#      MissForest is not re-run inside GridSearchCV or the hybrid sweep.
#   2. No joblib.Memory disk cache (previously wrote ~950 GB to scratch).
#   3. --outcome flag so outcomes run as independent SLURM array tasks.
#   4. Pruned RF/XGB grids to the region the benchmark actually selected.
#   5. Hybrid feature sweep uses a geometric grid, not 1..N one-by-one.
#   6. --sensitivity complete_case: block-specific complete-case, no imputation.
#   7. HYBRID SELECTION = CAPTURE RULE (see build_hybrid_model). Replaces the
#      paired 1-SE rule, which degenerated to argmax at n~1.6M.
#   8. Output files organised into per-outcome subdirectories.
#   9. Calibration plots overlaid (one panel per block) with QUANTILE binning
#      and data-scaled axes; DCA thresholds scaled to outcome prevalence.
#  10. --cache_imputed writes imputed predictors to validated, configuration-
#      specific serialised caches for cheap and safe re-runs.
#  11. HYBRID is included in formal paired bootstrap comparisons with LR/RF/XGB.
#  12. Run-level outputs are outcome-scoped, preventing SLURM array overwrites.
#  13. Outcome-excluded CVD/cancer predictors are also excluded from imputation.
#
# Fold-safety note: imputation is unsupervised (outcome never in the matrix)
# and the TEST set only ever goes through transform(), so held-out test metrics
# remain leakage-free. Impute-once affects only the internal CV used to pick
# hyperparameters, which cannot inflate the reported test numbers.
# =============================================================================

warnings.resetwarnings()
warnings.simplefilter("default")
warnings.filterwarnings("always", category=ConvergenceWarning)

# =============================================================================
# 0a. PARSE COMMAND LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description='Mortality prediction models for CPRD data')
parser.add_argument('--sex', type=str, default='all', choices=['all', 'male', 'female'])
parser.add_argument('--database', type=str, default='all', choices=['all', 'gold', 'aurum'])
parser.add_argument('--diabetes_type', type=int, default=2, choices=[1, 2])
parser.add_argument('--tuning_metric', type=str, default='brier', choices=['brier', 'auc'],
                     help='GridSearchCV scoring: neg_brier_score (calibration-optimal, PRIMARY/DEFAULT) '
                          'or roc_auc (discrimination-optimal, sensitivity run)')
parser.add_argument(
    '--sensitivity',
    type=str,
    default='none',
    choices=['none', 'complete_fu', 'complete_case'],
    help=(
        'none = main imputed analysis; complete_fu = restrict to patients with complete 10y follow-up; '
        'complete_case = drop patients missing any required predictor within each model block and skip imputation.'
    )
)
parser.add_argument('--n_splits', type=int, default=1,
                     help='Number of repeated practice-level train/test splits for robustness. Default 1.')
parser.add_argument('--study_end_date', type=str, default='2021-03-31',
                     help='Administrative study end date for outcome derivation and follow-up capping. '
                          'Must match the post-processing scripts. Default = 2021-03-31.')
parser.add_argument('--pilot', action='store_true',
                     help='Restrict to Block A, all-cause mortality only, and a lighter bootstrap count. '
                          'For timing/smoke-testing before a full run — NOT for real results.')
parser.add_argument('--outcome', type=str, default='all',
                    choices=['all', 'death_10y', 'death_cvd', 'death_cancer'],
                    help='Run a single outcome (for SLURM parallelism) or all three.')
parser.add_argument('--hybrid_capture', type=float, default=0.90,
                    help='HYBRID SELECTION RULE. Smallest number of predictors capturing at '
                         'least this fraction of the achievable cross-validated improvement '
                         '(n=1 baseline -> best subset). Scale-free, so it behaves sensibly '
                         'for both common and rare outcomes. PRE-SPECIFY and report. '
                         'Default 0.90.')
parser.add_argument('--cache_imputed', action='store_true',
                    help='Write validated imputed-predictor caches and reuse them if present. '
                         'Makes future reruns that only change modelling choices cheaper.')
parser.add_argument(
    "--rf_oob_test",
    action="store_true",
    help=(
        "Run a development-data RF out-of-bag tree-count diagnostic "
        "and exit without running the full pipeline."
    )
)
parser.add_argument(
    "--data_path",
    type=str,
    default=(
        "/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/"
        "Combined_GOLD_Aurum_with_meds_comorbidities_studyend_cod.txt"
    ),
    help="Path to the full dataset or a pre-generated subset TSV file."
)
parser.add_argument(
    "--n_bootstraps",
    type=int,
    default=None,
    help=(
        "Override the number of practice-cluster bootstrap resamples. "
        "Default: 2000 for full runs and 100 in pilot mode."
    )
)
parser.add_argument(
    "--oob_sample_n",
    type=int,
    default=100_000,
    help="Maximum development-sample size used by --rf_oob_test."
)
parser.add_argument(
    "--oob_outcome",
    type=str,
    default="death_10y",
    choices=["death_10y", "death_cvd", "death_cancer"],
    help="Outcome used for the RF OOB tree-stability diagnostic."
)
parser.add_argument(
    "--rf_n_estimators",
    type=int,
    default=None,
    help=(
        "Fix the main random forest to a tree count selected from the OOB "
        "diagnostic. If omitted, a default of 750 is used."
    )
)
args = parser.parse_args()

if args.pilot and args.rf_oob_test:
    parser.error("--pilot and --rf_oob_test cannot be used together.")

if args.n_splits < 1:
    parser.error("--n_splits must be at least 1.")

if args.n_bootstraps is not None and args.n_bootstraps < 1:
    parser.error("--n_bootstraps must be at least 1.")

if args.oob_sample_n < 1:
    parser.error("--oob_sample_n must be at least 1.")

if args.rf_n_estimators is not None and args.rf_n_estimators < 1:
    parser.error("--rf_n_estimators must be at least 1.")

if not (0.0 < args.hybrid_capture <= 1.0):
    parser.error("--hybrid_capture must be in (0, 1].")

DM_TYPE = args.diabetes_type
SEX_FILTER = args.sex
DB_FILTER = args.database
TUNING_METRIC = args.tuning_metric

suffix_parts = []
if DB_FILTER != 'all':
    suffix_parts.append(DB_FILTER)
if SEX_FILTER != 'all':
    suffix_parts.append(SEX_FILTER)
suffix_parts.append(f"tune-{TUNING_METRIC}")
suffix_parts.append(f"cap-{int(round(args.hybrid_capture * 100))}")
if args.pilot:
    suffix_parts.append("PILOT")
if args.n_bootstraps is not None:
    suffix_parts.append(f"boot-{args.n_bootstraps}")
if args.rf_n_estimators is not None:
    suffix_parts.append(f"rf-{args.rf_n_estimators}")
FILE_SUFFIX = '_' + '_'.join(suffix_parts)

print("=" * 60)
print("MORTALITY PREDICTION MODELS")
print(f"  Script version: {SCRIPT_VERSION}")
print(f"  Database: {DB_FILTER.upper()} | Sex: {SEX_FILTER.upper()} | Tuning: {TUNING_METRIC}")
print(f"  Outcome: {args.outcome} | Sensitivity: {args.sensitivity} | n_splits: {args.n_splits}")
print(f"  Hybrid capture rule: {args.hybrid_capture:.0%} of achievable CV gain")
print(f"  Study end date: {args.study_end_date} | Pilot mode: {args.pilot}")
print("=" * 60)

# =============================================================================
# 0b. OUTPUT DIRECTORIES
#
# Layout:
#   Combined_male_type_2/
#     death_10y/                          <- main imputed analysis
#     death_cvd/
#     death_cancer/
#     SENSITIVITY_completeCase_death_10y/
#     SENSITIVITY_completeCase_death_cvd/
#     SENSITIVITY_completeFU_death_10y/
#     _run_level/                         <- split info, model index, aggregates
# =============================================================================

db_name = 'Combined' if DB_FILTER == 'all' else DB_FILTER.upper()
sex_name = 'allsex' if SEX_FILTER == 'all' else SEX_FILTER.lower()
diabetes_type = 'type_1' if DM_TYPE == 1 else 'type_2'
db_stem = DB_FILTER.upper() if DB_FILTER != 'all' else 'Combined'

OUTPUT_DIR = f'{db_name}_{sex_name}_{diabetes_type}'
RUN_LEVEL_DIR = os.path.join(OUTPUT_DIR, '_run_level')
CACHE_DIR = os.path.join(OUTPUT_DIR, '_imputed_cache')
os.makedirs(RUN_LEVEL_DIR, exist_ok=True)
if args.cache_imputed:
    os.makedirs(CACHE_DIR, exist_ok=True)

FILE_STEM = f'{db_stem}_{diabetes_type}'
RUN_SCOPE = 'all_outcomes' if args.outcome == 'all' else args.outcome

SENSITIVITY_TAG = {
    'none': '',
    'complete_fu': 'SENSITIVITY_completeFU_',
    'complete_case': 'SENSITIVITY_completeCase_',
}[args.sensitivity]


def outcome_dir(outcome_col):
    """Subdirectory for one outcome under the current analysis type."""
    path = os.path.join(OUTPUT_DIR, f'{SENSITIVITY_TAG}{outcome_col}')
    os.makedirs(path, exist_ok=True)
    return path


def outcome_prefix(outcome_col, split_idx=0):
    """Full path prefix for per-outcome output files."""
    stem = FILE_STEM if args.n_splits == 1 else f'{FILE_STEM}_split{split_idx}'
    return os.path.join(outcome_dir(outcome_col), stem)


def run_level_prefix(split_idx=0):
    """Outcome-scoped prefix for files shared within one SLURM task."""
    stem = f'{FILE_STEM}_{RUN_SCOPE}'
    if args.n_splits != 1:
        stem = f'{stem}_split{split_idx}'
    if SENSITIVITY_TAG:
        stem = f'{SENSITIVITY_TAG}{stem}'
    return os.path.join(RUN_LEVEL_DIR, stem)


print(f"  Output root: {OUTPUT_DIR}/")

# =============================================================================
# 1. LOAD DATA
# =============================================================================

data_path = args.data_path
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path, sep="\t", low_memory=False)
if df.empty:
    raise ValueError("The input dataset is empty.")

required_input_columns = {"database", "gender", "diabetes_type", "pracid", "indexdate", "yob"}
missing_input_columns = sorted(required_input_columns.difference(df.columns))
if missing_input_columns:
    raise ValueError(f"Input dataset is missing required columns: {missing_input_columns}")
if not ({"dod_ons", "dod"} & set(df.columns)):
    raise ValueError("Input dataset must contain either 'dod_ons' or 'dod'.")

date_cols = ['indexdate', 'dod_ons', 'tod', 'regenddate', 'eventdate', 'smoking_date', 'bmi_date',
             'bp_date', 'tot_chol_date', 'hdl_date', 'ldl_date', 'trigly_date', 'hba1c_date', 'lcd',
             'censor_date', 'comorb_ckd_first_date', 'comorb_htn_first_date', 'comorb_cvd_first_date',
             'comorb_cancer_any_first_date', 'comorb_cancer_breast_first_date',
             'comorb_cancer_colorectal_first_date', 'comorb_cancer_lung_first_date',
             'comorb_cancer_pancreatic_first_date', 'comorb_cancer_prostate_first_date']


def parse_mixed_dates(s):
    s = s.astype("string")
    has_slash = s.str.contains("/", na=False)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if has_slash.any():
        out.loc[has_slash] = pd.to_datetime(s.loc[has_slash], errors="coerce", dayfirst=True)
    if (~has_slash).any():
        out.loc[~has_slash] = pd.to_datetime(s.loc[~has_slash], errors="coerce")
    return out


for col in date_cols:
    if col in df.columns:
        df[col] = parse_mixed_dates(df[col])

df["database"] = df["database"].astype("string").str.strip().str.upper()
if df["database"].isna().any():
    raise ValueError("The database column contains missing values.")
invalid_databases = sorted(set(df["database"].dropna()) - {"GOLD", "AURUM"})
if invalid_databases:
    raise ValueError(f"Unexpected database values: {invalid_databases}")
if df["pracid"].isna().any():
    raise ValueError("The pracid column contains missing values.")
DB_FILTER = args.database.strip().lower()

# =============================================================================
# 2. FILTER BY DATABASE / SEX / DIABETES TYPE
# =============================================================================

if DB_FILTER == 'gold':
    df = df[df['database'] == 'GOLD'].copy()
elif DB_FILTER == 'aurum':
    df = df[df['database'] == 'AURUM'].copy()

if SEX_FILTER != 'all':
    df['gender'] = df['gender'].astype(str).str.strip()
    sex_mapping = {'male': {'M', 'MALE', '1'}, 'female': {'F', 'FEMALE', '2'}}
    df = df[df['gender'].str.upper().isin(sex_mapping[SEX_FILTER])].copy()

df['diabetes_type'] = pd.to_numeric(df['diabetes_type'], errors='coerce')
df = df[df['diabetes_type'] == DM_TYPE].copy()
if df.empty:
    raise ValueError(
        f"No rows remain after filtering database={DB_FILTER!r}, sex={SEX_FILTER!r}, "
        f"diabetes_type={DM_TYPE}."
    )

# =============================================================================
# 3. DERIVE OUTCOMES
#
# NOTE: patients who leave the database before 10y are coded as "alive"
# rather than censored. This is DELIBERATE (ONS-linked dod_ons ascertainment).
# =============================================================================

CAUSE_SPECIFIC_COLS = {'death_cvd': 'cod_cvd', 'death_cancer': 'cod_cancer'}


def derive_mortality_no_censoring(df, years=10, study_end_date="2021-03-31"):
    df = df.copy()
    study_end = pd.to_datetime(study_end_date)
    df = df[df["indexdate"].notna()].copy()
    df["cutoff_date"] = df["indexdate"] + pd.DateOffset(years=years)

    candidates = []
    if "tod" in df.columns:
        candidates.append(df["tod"])
    if "regenddate" in df.columns:
        candidates.append(df["regenddate"])
    elif "regend" in df.columns:
        candidates.append(pd.to_datetime(df["regend"], errors="coerce"))
    if "lcd" in df.columns:
        candidates.append(df["lcd"])
    censor_raw = pd.concat(candidates, axis=1).min(axis=1) if candidates else pd.Series(pd.NaT, index=df.index)
    censor_raw = censor_raw.where(censor_raw >= df["indexdate"], pd.NaT)
    df["censor_date_derived"] = censor_raw.fillna(study_end).clip(upper=study_end)

    dod_col = "dod_ons" if "dod_ons" in df.columns else "dod"
    dod_clean = pd.to_datetime(df[dod_col], errors="coerce")
    dod_clean = dod_clean.where(dod_clean >= df["indexdate"], pd.NaT)

    dod_or_end = dod_clean.fillna(study_end)
    df["follow_up_end"] = pd.concat([dod_or_end, df["censor_date_derived"]], axis=1).min(axis=1).clip(upper=study_end)
    df["follow_up_end"] = df["follow_up_end"].where(df["follow_up_end"] >= df["indexdate"], df["indexdate"])
    df["follow_up_years"] = (df["follow_up_end"] - df["indexdate"]).dt.days / 365.25

    df["died_within_followup"] = dod_clean.notna() & (dod_clean <= df["cutoff_date"])

    # Complete follow-up = the 10y outcome is unambiguous: either the death is
    # observed, or the patient was administratively observable for the full 10y.
    # MUST use censor_date_derived, NOT follow_up_end, because follow_up_end is
    # truncated at the date of death and would exclude every death.
    df["complete_followup"] = (
        (df["censor_date_derived"] >= df["cutoff_date"]) | df["died_within_followup"]
    )
    df["death_10y"] = df["died_within_followup"].astype(int)

    for outcome_col, source_col in CAUSE_SPECIFIC_COLS.items():
        if source_col in df.columns:
            df[outcome_col] = (df["died_within_followup"] &
                                (df[source_col].fillna(0).astype(int) == 1)).astype(int)
        else:
            df[outcome_col] = np.nan

    df["eligible"] = True
    print(f"Complete {years}y follow-up: {df['complete_followup'].mean() * 100:.1f}%")
    print(f"Died within {years}y (all-cause): {df['death_10y'].mean() * 100:.1f}%")
    return df


df = derive_mortality_no_censoring(df, years=10, study_end_date=args.study_end_date)
df_eligible = df[df['eligible']].copy()

# --- 3b. SENSITIVITY ANALYSES ---------------------------------------------

if args.sensitivity == 'complete_fu':
    print(f"\n>>> SENSITIVITY ANALYSIS: complete-followup subset only <<<")
    print(f"Main cohort: {len(df_eligible):,} -> Complete-FU: {(df_eligible['complete_followup']).sum():,}")
    df_eligible = df_eligible[df_eligible['complete_followup']].copy()
elif args.sensitivity == 'complete_case':
    print("\n>>> SENSITIVITY ANALYSIS: block-specific complete-case analysis (no imputation) <<<")

OUTCOMES = {'death_10y': 'All-Cause Mortality'}
if not args.pilot:
    for outcome_col in CAUSE_SPECIFIC_COLS:
        if outcome_col in df_eligible.columns and df_eligible[outcome_col].notna().all():
            OUTCOMES[outcome_col] = outcome_col.replace('death_', '').upper() + ' Mortality'

# --- 3c. Restrict to a single outcome for SLURM parallelism -----------------
if args.outcome != 'all':
    OUTCOMES = {k: v for k, v in OUTCOMES.items() if k == args.outcome}
    if not OUTCOMES:
        raise SystemExit(
            f"Outcome {args.outcome!r} is not available in this cohort "
            f"(pilot mode restricts to death_10y)."
        )
print(f"Outcomes this run: {list(OUTCOMES)}")

for _oc, _lab in OUTCOMES.items():
    _rate = pd.to_numeric(df_eligible[_oc], errors='coerce').mean()
    print(f"  {_lab:<22}: {df_eligible[_oc].sum():,.0f} events ({100 * _rate:.2f}%)")

# =============================================================================
# 4. AGE AT INDEX
# =============================================================================

df_eligible['age_at_index'] = df_eligible['indexdate'].dt.year - df_eligible['yob']

# =============================================================================
# 5. PREDICTOR BLOCKS
# =============================================================================

med_binary_vars = sorted([c for c in df_eligible.columns if c.startswith('med_') and c.endswith('_prescribed')])
comorb_bin_vars = sorted([c for c in df_eligible.columns if c.endswith('_bin')])
comorb_dur_vars = sorted([c for c in df_eligible.columns if c.endswith('_duration_years') and 'comorb' in c])

for col in med_binary_vars + comorb_bin_vars:
    df_eligible[col] = pd.to_numeric(df_eligible[col], errors='coerce').fillna(0).astype(int)
for col in comorb_dur_vars:
    df_eligible[col] = pd.to_numeric(df_eligible[col], errors='coerce').fillna(0)

# Clean IMD quintile and represent it as a categorical variable.
imd_numeric = pd.to_numeric(df_eligible["imd_quintile"], errors="coerce")
imd_numeric = imd_numeric.where(imd_numeric.between(1, 5), np.nan)
df_eligible["imd_quintile"] = imd_numeric.astype("Int64").astype("string")

if SEX_FILTER == "all":
    base_cat = ["gender", "gen_ethnicity", "imd_quintile"]
    base_cat_bio = ["gender", "gen_ethnicity", "smoking_status", "imd_quintile"]
else:
    base_cat = ["gen_ethnicity", "imd_quintile"]
    base_cat_bio = ["gen_ethnicity", "smoking_status", "imd_quintile"]

base_num_demog = ["age_at_index"]
biomarker_num = ['bmi', 'systolic', 'hba1c_perc', 'tot_chol', 'hdl', 'ldl', 'trigly']

BLOCKS = {
    'A': {'label': 'Demographics', 'cat': base_cat.copy(), 'num': base_num_demog.copy()},
    'B': {'label': 'Demographics + Biomarkers', 'cat': base_cat_bio.copy(), 'num': base_num_demog + biomarker_num},
    'C': {'label': 'Demographics + Biomarkers + Medications', 'cat': base_cat_bio.copy(),
          'num': base_num_demog + biomarker_num + med_binary_vars},
    'D': {'label': 'Demographics + Biomarkers + Medications + Comorbidities', 'cat': base_cat_bio.copy(),
          'num': base_num_demog + biomarker_num + med_binary_vars + comorb_bin_vars + comorb_dur_vars},
}

if args.pilot:
    print("\n" + "!" * 60)
    print("PILOT MODE: Block A only, all-cause mortality only, reduced bootstraps")
    print("Results from this run are for TIMING purposes only — do not use them")
    print("!" * 60 + "\n")
    BLOCKS = {'A': BLOCKS['A']}

BLOCK_DISPLAY_NAMES = {'A': 'Model 1', 'B': 'Model 2', 'C': 'Model 3', 'D': 'Model 4'}

# Consistent colours everywhere: ROC, PR, calibration, DCA, elbow.
MODEL_COLORS = {
    'LR': '#1f77b4',       # blue
    'LR_FLEX': '#2ca02c',  # green
    'RF': '#2E8B57',       # sea green
    'XGB': '#ff7f0e',      # orange
    'HYBRID': '#9467bd',   # purple
}
MODEL_MARKERS = {'LR': 'o', 'LR_FLEX': '^', 'RF': 's', 'XGB': 'D', 'HYBRID': 'v'}

# =============================================================================
# 6. SPLIT BY PRACTICE
# =============================================================================


def split_by_practice(df, test_size=0.2, random_state=42, stratify_by_database=False):
    if stratify_by_database and df['database'].nunique() > 1:
        train_dfs, test_dfs, train_prac_all, test_prac_all = [], [], [], []
        for db in df['database'].unique():
            db_df = df[df['database'] == db]
            practices = db_df['pracid'].unique()
            if len(practices) < 5:
                train_dfs.append(db_df)
                train_prac_all.extend(practices)
                continue
            train_prac, test_prac = train_test_split(practices, test_size=test_size, random_state=random_state)
            train_dfs.append(db_df[db_df['pracid'].isin(train_prac)])
            test_dfs.append(db_df[db_df['pracid'].isin(test_prac)])
            train_prac_all.extend(train_prac)
            test_prac_all.extend(test_prac)
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    else:
        practices = df['pracid'].unique()
        train_prac_all, test_prac_all = train_test_split(practices, test_size=test_size, random_state=random_state)
        train_df = df[df['pracid'].isin(train_prac_all)].copy()
        test_df = df[df['pracid'].isin(test_prac_all)].copy()
    return train_df, test_df, {'train_practices': list(train_prac_all), 'test_practices': list(test_prac_all)}


# =============================================================================
# 7. MISSFOREST IMPUTATION (fit on TRAIN, transform TEST, ONCE per block)
# =============================================================================


class FoldSafeMissForest(BaseEstimator, TransformerMixin):
    """MissForest transformer with train-only category mappings and fallbacks."""

    def __init__(self, cat_vars, num_vars, n_estimators=50, max_depth=10, max_iter=5,
                 early_stopping=True, random_state=42, n_jobs=-1):
        self.cat_vars = cat_vars
        self.num_vars = num_vars
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _expected_columns(self):
        columns = list(self.cat_vars) + list(self.num_vars)
        if len(columns) != len(set(columns)):
            raise ValueError("A predictor appears in both cat_vars and num_vars.")
        return columns

    def _clean_input(self, X):
        expected = self._expected_columns()
        if isinstance(X, pd.DataFrame):
            X = X.copy()
        else:
            X = pd.DataFrame(X, columns=expected)
        missing_columns = [col for col in expected if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Input data are missing required columns: {missing_columns}")
        X = X.loc[:, expected].copy()
        for col in self.cat_vars:
            values = X[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
            values = values.astype("string").str.strip()
            values = values.mask(values.isin(["", "nan", "None", "<NA>"]), pd.NA)
            X[col] = values.astype(object)
        for col in self.num_vars:
            X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return X

    def _encode(self, X):
        X_encoded = X.copy()
        for col in self.cat_vars:
            X_encoded[col] = X_encoded[col].map(self.category_mappings_[col])
        return X_encoded.astype(np.float64)

    def fit(self, X, y=None):
        X_clean = self._clean_input(X)
        expected = self._expected_columns()
        self.feature_names_in_ = np.asarray(expected, dtype=object)
        self.n_features_in_ = len(expected)
        self.category_mappings_, self.inverse_category_mappings_, self.fallback_values_ = {}, {}, {}

        for col in self.cat_vars:
            observed = X_clean[col].dropna()
            if observed.empty:
                raise ValueError(f"Categorical predictor '{col}' is completely missing in this training fold.")
            categories = sorted(observed.unique().tolist(), key=str)
            mapping = {category: code for code, category in enumerate(categories)}
            self.category_mappings_[col] = mapping
            self.inverse_category_mappings_[col] = {code: category for category, code in mapping.items()}
            self.fallback_values_[col] = float(mapping[observed.mode().iloc[0]])

        for col in self.num_vars:
            observed = X_clean[col].dropna()
            if observed.empty:
                raise ValueError(f"Numerical predictor '{col}' is completely missing in this training fold.")
            self.fallback_values_[col] = float(observed.median())

        X_encoded = self._encode(X_clean)

        if X_encoded.shape[1] < 2 or not X_encoded.isna().any().any():
            self.imputer_ = None
            return self

        self.imputer_ = MissForest(
            clf=RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                       n_jobs=self.n_jobs, random_state=self.random_state),
            rgr=RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                      n_jobs=self.n_jobs, random_state=self.random_state),
            categorical=list(self.cat_vars), max_iter=self.max_iter,
            early_stopping=self.early_stopping, verbose=0)
        self.imputer_.fit(X_encoded)
        return self

    def transform(self, X):
        check_is_fitted(self, ["feature_names_in_", "category_mappings_",
                               "inverse_category_mappings_", "fallback_values_"])
        X_clean = self._clean_input(X)
        X_encoded = self._encode(X_clean)

        for col in X_encoded.columns:
            if X_encoded[col].isna().all():
                X_encoded[col] = self.fallback_values_[col]

        if not X_encoded.isna().any().any():
            X_imputed = X_encoded.copy()
        elif self.imputer_ is None:
            X_imputed = X_encoded.copy()
            for col in X_imputed.columns:
                X_imputed[col] = X_imputed[col].fillna(self.fallback_values_[col])
        else:
            X_imputed = self.imputer_.transform(X_encoded)

        X_imputed = X_imputed.loc[:, self.feature_names_in_].copy()
        X_imputed.index = X_clean.index

        for col in self.cat_vars:
            inverse_mapping = self.inverse_category_mappings_[col]
            minimum_code, maximum_code = min(inverse_mapping), max(inverse_mapping)
            codes = (pd.to_numeric(X_imputed[col], errors="coerce")
                     .fillna(self.fallback_values_[col]).round()
                     .clip(minimum_code, maximum_code).astype(int))
            X_imputed[col] = codes.map(inverse_mapping)

        for col in self.num_vars:
            X_imputed[col] = pd.to_numeric(X_imputed[col], errors="coerce")

        remaining_missing = int(X_imputed.isna().sum().sum())
        if remaining_missing > 0:
            raise ValueError(f"MissForest left {remaining_missing} missing predictor values.")
        return X_imputed

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_


def impute_block_once(train_df, test_df, cat_vars, num_vars, seed=42):
    """Fit MissForest on TRAIN, transform TEST, once. Reused for all outcomes/subsets."""
    imp = FoldSafeMissForest(cat_vars=cat_vars, num_vars=num_vars, n_estimators=50,
                             max_depth=10, max_iter=5, early_stopping=True,
                             random_state=seed, n_jobs=-1)
    tr = imp.fit_transform(train_df[cat_vars + num_vars])
    te = imp.transform(test_df[cat_vars + num_vars])
    train_out, test_out = train_df.copy(), test_df.copy()
    for c in cat_vars + num_vars:
        train_out[c] = tr[c].values
        test_out[c] = te[c].values
    return train_out, test_out


def _practice_signature(frame):
    """Stable, inexpensive signature of the practices and databases in one split."""
    values = (
        frame[["database", "pracid"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["database", "pracid"])
        .agg("::".join, axis=1)
        .tolist()
    )
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _source_file_metadata(path):
    absolute_path = os.path.abspath(path)
    metadata = {"path": absolute_path}
    try:
        stat = os.stat(absolute_path)
        metadata.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    except OSError:
        metadata.update({"size": None, "mtime_ns": None})
    return metadata


def _imputation_cache_key(train_df, test_df, cat_vars, num_vars, block_name, split_idx, seed):
    payload = {
        "script_version": SCRIPT_VERSION,
        "source": _source_file_metadata(data_path),
        "database": DB_FILTER,
        "sex": SEX_FILTER,
        "diabetes_type": DM_TYPE,
        "sensitivity": args.sensitivity,
        # Outcome scope prevents concurrent SLURM outcome jobs writing one cache.
        "outcome_scope": RUN_SCOPE,
        "study_end_date": args.study_end_date,
        "block_name": block_name,
        "split_idx": split_idx,
        "seed": seed,
        "cat_vars": list(cat_vars),
        "num_vars": list(num_vars),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_practices": _practice_signature(train_df),
        "test_practices": _practice_signature(test_df),
    }
    serialised = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()[:24]


def _restore_cached_predictors(source_df, cached_df, predictor_columns, cache_label):
    expected_columns = list(predictor_columns)
    if list(cached_df.columns) != expected_columns:
        raise ValueError(
            f"Cached {cache_label} columns do not match the requested predictors. "
            f"Expected {expected_columns}, found {list(cached_df.columns)}."
        )
    if len(cached_df) != len(source_df):
        raise ValueError(
            f"Cached {cache_label} row count ({len(cached_df):,}) does not match "
            f"the current split ({len(source_df):,})."
        )
    if not cached_df.index.equals(source_df.index):
        raise ValueError(f"Cached {cache_label} row index does not match the current split.")

    output = source_df.copy()
    for column in expected_columns:
        output[column] = cached_df[column]
    return output


def _atomic_to_pickle(frame, destination):
    """Write a pandas cache atomically without optional parquet dependencies."""
    temporary = f"{destination}.tmp-{os.getpid()}.pkl"
    try:
        frame.to_pickle(temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def impute_block_cached(train_df, test_df, cat_vars, num_vars, block_name, split_idx, seed):
    """Impute once, optionally using a validated configuration-specific cache."""
    if not args.cache_imputed:
        return impute_block_once(train_df, test_df, cat_vars, num_vars, seed=seed)

    predictors = list(cat_vars) + list(num_vars)
    cache_key = _imputation_cache_key(
        train_df, test_df, cat_vars, num_vars, block_name, split_idx, seed
    )
    safe_block_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in block_name)
    stem = os.path.join(CACHE_DIR, f'{safe_block_name}_{cache_key}')
    train_file, test_file = f'{stem}_train.pkl', f'{stem}_test.pkl'

    if os.path.exists(train_file) and os.path.exists(test_file):
        try:
            cached_train = pd.read_pickle(train_file)
            cached_test = pd.read_pickle(test_file)
            restored_train = _restore_cached_predictors(
                train_df, cached_train, predictors, f"training cache for {block_name}"
            )
            restored_test = _restore_cached_predictors(
                test_df, cached_test, predictors, f"test cache for {block_name}"
            )
            print(f"    (loaded validated cached imputation for {block_name})")
            return restored_train, restored_test
        except Exception as exc:
            warnings.warn(
                f"Ignoring invalid imputation cache for {block_name} and recomputing: {exc}"
            )

    tr, te = impute_block_once(train_df, test_df, cat_vars, num_vars, seed=seed)
    try:
        _atomic_to_pickle(tr[predictors], train_file)
        _atomic_to_pickle(te[predictors], test_file)
    except Exception as exc:  # caching must never kill an analysis run
        warnings.warn(f"Could not cache imputed block {block_name}: {exc}")
    return tr, te


# =============================================================================
# 8. PREPROCESSING
# =============================================================================

CONTINUOUS_VARS = ["age_at_index", "bmi", "systolic", "hba1c_perc",
                   "tot_chol", "hdl", "ldl", "trigly"]
SCALE_NUMERIC_VARS = set(CONTINUOUS_VARS + comorb_dur_vars)
TUNING_METRICS = {'brier': 'neg_brier_score', 'auc': 'roc_auc'}


class SelectiveStandardScaler(BaseEstimator, TransformerMixin):
    """Standardise selected numerical columns, pass the rest through, preserve order."""

    def __init__(self, scale_columns=None):
        self.scale_columns = scale_columns

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SelectiveStandardScaler requires a pandas DataFrame.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        requested_columns = list(self.scale_columns) if self.scale_columns is not None else []
        self.scale_columns_ = [col for col in requested_columns if col in X.columns]
        if self.scale_columns_:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X[self.scale_columns_])
        else:
            self.scaler_ = None
        return self

    def transform(self, X):
        check_is_fitted(self, ["feature_names_in_", "scale_columns_"])
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X_out = X.loc[:, self.feature_names_in_].copy()
        if self.scaler_ is not None:
            X_out.loc[:, self.scale_columns_] = self.scaler_.transform(X_out[self.scale_columns_])
        return X_out.to_numpy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_


def create_preprocessor(cat_vars, num_vars, scale_numeric=False, spline_vars=None, n_knots=5):
    spline_vars = spline_vars or []
    linear_num_vars = [v for v in num_vars if v not in spline_vars]
    transformers = []

    if cat_vars:
        transformers.append(("cat", Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_vars))

    if linear_num_vars:
        if scale_numeric:
            columns_to_scale = [v for v in linear_num_vars if v in SCALE_NUMERIC_VARS]
            numerical_transformer = Pipeline(steps=[
                ("selective_scaler", SelectiveStandardScaler(scale_columns=columns_to_scale))])
        else:
            numerical_transformer = "passthrough"
        transformers.append(("num", numerical_transformer, linear_num_vars))

    if spline_vars:
        spline_steps = [("spline", SplineTransformer(n_knots=n_knots, degree=3, include_bias=False))]
        if scale_numeric:
            spline_steps.append(("scaler", StandardScaler()))
        transformers.append(("spline", Pipeline(steps=spline_steps), spline_vars))

    return ColumnTransformer(transformers=transformers, remainder="drop",
                             verbose_feature_names_out=True)


def create_model_pipeline(cat_vars, num_vars, model_type="LR"):
    # NOTE: imputation is NO LONGER in the pipeline — data arrives pre-imputed.

    if model_type == "LR":
        preprocessor = create_preprocessor(cat_vars, num_vars, scale_numeric=True)
        classifier = LogisticRegression(max_iter=2000, random_state=42)
        param_grid = {"classifier__C": [0.001, 0.01, 0.1, 1, 10, 100]}

    elif model_type == "LR_FLEX":
        spline_vars = [v for v in num_vars if v in CONTINUOUS_VARS]
        preprocessor = create_preprocessor(cat_vars, num_vars, scale_numeric=True,
                                           spline_vars=spline_vars, n_knots=5)
        classifier = LogisticRegression(max_iter=2000, random_state=42)
        param_grid = {"classifier__C": [0.001, 0.01, 0.1, 1, 10, 100]}

    elif model_type == "RF":
        preprocessor = create_preprocessor(cat_vars, num_vars, scale_numeric=False)
        classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
        # PRUNED: benchmark always chose min_samples_leaf 10 or 20; trees fixed at 750.
        param_grid = {
            "classifier__max_features": [0.15, 0.25, 0.4, 0.6],
            "classifier__n_estimators": [args.rf_n_estimators if args.rf_n_estimators is not None else 750],
            "classifier__min_samples_leaf": [10, 20],
        }

    elif model_type == "XGB":
        preprocessor = create_preprocessor(cat_vars, num_vars, scale_numeric=False)
        classifier = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1,
                                   min_child_weight=2, subsample=0.84, colsample_bytree=0.75,
                                   reg_lambda=1, reg_alpha=1)
        # PRUNED: full-data runs consistently chose depth 3-6, lr 0.01 or 0.03.
        param_grid = {
            "classifier__max_depth": [3, 4, 6],
            "classifier__learning_rate": [0.01, 0.03],
            "classifier__n_estimators": [500, 1000],
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    return pipeline, param_grid


def run_rf_oob_tree_test(train_df, cat_vars, num_vars, outcome, output_prefix,
                         sample_n=100_000, tree_counts=None, max_features=0.25,
                         min_samples_leaf=5, random_state=42):
    """RF OOB tree-stability diagnostic on development data only."""
    if tree_counts is None:
        tree_counts = [100, 200, 300, 500, 750, 1000]

    required_columns = list(cat_vars) + list(num_vars) + [outcome]
    analysis_df = train_df[required_columns].copy()
    if len(analysis_df) > sample_n:
        analysis_df = analysis_df.sample(n=sample_n, random_state=random_state)

    X_raw = analysis_df[list(cat_vars) + list(num_vars)].copy()
    y = analysis_df[outcome].astype(int).to_numpy()
    print(f"\nRF OOB test sample: {len(analysis_df):,}")
    print(f"Events: {y.sum():,} ({y.mean() * 100:.2f}%)")

    imputer = FoldSafeMissForest(cat_vars=cat_vars, num_vars=num_vars, n_estimators=50,
                                 max_depth=10, max_iter=5, early_stopping=True,
                                 random_state=random_state, n_jobs=-1)
    X_imputed = imputer.fit_transform(X_raw)
    preprocessor = create_preprocessor(cat_vars=cat_vars, num_vars=num_vars, scale_numeric=False)
    X_processed = preprocessor.fit_transform(X_imputed)

    forest = RandomForestClassifier(n_estimators=tree_counts[0], max_features=max_features,
                                    min_samples_leaf=min_samples_leaf, bootstrap=True,
                                    oob_score=True, warm_start=True, random_state=random_state, n_jobs=-1)
    rows = []
    for number_of_trees in tree_counts:
        print(f"Fitting RF with {number_of_trees} trees...")
        forest.set_params(n_estimators=number_of_trees)
        forest.fit(X_processed, y)
        oob_probabilities = forest.oob_decision_function_[:, 1]
        valid = np.isfinite(oob_probabilities)
        y_valid, probabilities_valid = y[valid], oob_probabilities[valid]
        rows.append({
            "n_estimators": number_of_trees,
            "oob_auc": roc_auc_score(y_valid, probabilities_valid),
            "oob_brier": brier_score_loss(y_valid, probabilities_valid),
            "oob_log_loss": log_loss(y_valid, probabilities_valid, labels=[0, 1]),
            "n_oob_predictions": int(valid.sum()),
        })

    results = pd.DataFrame(rows)
    results.to_csv(f"{output_prefix}_rf_oob_curve.csv", index=False)

    for metric, ylabel, fname in [("oob_auc", "OOB AUC", "auc"), ("oob_brier", "OOB Brier score", "brier")]:
        plt.figure(figsize=(8, 5))
        plt.plot(results["n_estimators"], results[metric], marker="o", color=MODEL_COLORS['RF'])
        plt.xlabel("Number of trees")
        plt.ylabel(ylabel)
        plt.title(f"Random Forest {ylabel} Stabilisation")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_rf_oob_{fname}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("\nRF OOB results:")
    print(results.to_string(index=False))
    return results


# =============================================================================
# 9. TRAINING / EVALUATION
# =============================================================================


def train_model(X_train, y_train, groups, cat_vars, num_vars, model_type, tuning_metric):
    n_groups = pd.Series(groups).nunique(dropna=True)
    if n_groups < 5:
        raise ValueError(
            f"At least 5 distinct practices are required for 5-fold GroupKFold; found {n_groups}."
        )
    pipeline, param_grid = create_model_pipeline(cat_vars, num_vars, model_type)
    gkf = GroupKFold(n_splits=5)
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=gkf,
                               scoring=TUNING_METRICS[tuning_metric], n_jobs=1, pre_dispatch=1,
                               refit=True, error_score="raise", return_train_score=False)
    grid_search.fit(X_train, y_train, groups=groups)

    split_score_columns = sorted(
        c for c in grid_search.cv_results_
        if c.startswith("split") and c.endswith("_test_score"))
    best_index = grid_search.best_index_
    best_fold_scores = np.asarray(
        [grid_search.cv_results_[c][best_index] for c in split_score_columns], dtype=float)

    if len(best_fold_scores) > 1:
        best_score_se = float(np.std(best_fold_scores, ddof=1) / np.sqrt(len(best_fold_scores)))
    else:
        best_score_se = 0.0

    return (grid_search.best_estimator_, grid_search.best_params_,
            float(grid_search.best_score_), best_score_se, best_fold_scores)


def calibration_slope_intercept(y_true, y_pred_proba, eps=1e-6):
    """Estimate calibration slope/intercept using an approximately unpenalised LR."""
    p = np.clip(y_pred_proba, eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    cal_model = LogisticRegression(max_iter=2000, C=1e6, solver="lbfgs")
    cal_model.fit(logit_p, y_true)
    return cal_model.coef_[0][0], cal_model.intercept_[0]


def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_val = roc_auc_score(y_test, y_pred_proba)
    average_precision = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    oe_ratio = y_test.mean() / y_pred_proba.mean() if y_pred_proba.mean() > 0 else np.nan
    slope, intercept = calibration_slope_intercept(y_test, y_pred_proba)
    return {'auc': auc_val, 'average_precision': average_precision, 'brier': brier,
            'oe_ratio': oe_ratio, 'cal_slope': slope, 'cal_intercept': intercept,
            'y_pred_proba': y_pred_proba}


def evaluate_model_by_database(model, X_test, y_test, test_df):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    results = {}
    for db in test_df["database"].unique():
        mask = test_df["database"].values == db
        if mask.sum() < 50:
            continue
        y_t, y_p = y_test.values[mask], y_pred_proba[mask]
        if np.unique(y_t).size < 2:
            warnings.warn(f"Cannot calculate AUC for {db}: only one outcome class is present.")
            auc_value = np.nan
        else:
            auc_value = roc_auc_score(y_t, y_p)
        results[db] = {"n": int(mask.sum()), "events": int(y_t.sum()), "auc": auc_value,
                       "average_precision": average_precision_score(y_t, y_p),
                       "brier": brier_score_loss(y_t, y_p),
                       "oe_ratio": y_t.mean() / y_p.mean() if y_p.mean() > 0 else np.nan}
    return results


def get_logistic_convergence_info(fitted_pipeline):
    classifier = fitted_pipeline.named_steps["classifier"]
    iterations_used = int(np.max(classifier.n_iter_))
    maximum_iterations = int(classifier.max_iter)
    return {"iterations_used": iterations_used, "maximum_iterations": maximum_iterations,
            "reached_iteration_limit": iterations_used >= maximum_iterations}


def train_and_evaluate_block(train_df, test_df, cat_vars, num_vars, block_name, outcome, tuning_metric):
    print(f"\n{'=' * 60}\nTRAINING: {block_name} [tuning={tuning_metric}]\n{'=' * 60}")
    all_vars = cat_vars + num_vars
    X_train, X_test = train_df[all_vars].copy(), test_df[all_vars].copy()
    y_train, y_test = train_df[outcome].copy(), test_df[outcome].copy()
    groups = train_df["pracid"]
    multi_db = test_df["database"].nunique() > 1

    results = {}
    for model_type in ["LR", "LR_FLEX", "RF", "XGB"]:
        model, best_params, cv_score, cv_score_se, _ = train_model(
            X_train, y_train, groups, cat_vars, num_vars, model_type, tuning_metric)
        ev = evaluate_model(model, X_test, y_test)
        convergence_info = None
        if model_type in ["LR", "LR_FLEX"]:
            convergence_info = get_logistic_convergence_info(model)
            print(f"  {model_type} convergence: {convergence_info['iterations_used']}/"
                  f"{convergence_info['maximum_iterations']} iterations")
            if convergence_info["reached_iteration_limit"]:
                warnings.warn(f"{model_type} reached max_iter="
                              f"{convergence_info['maximum_iterations']}. Convergence may not have been achieved.",
                              ConvergenceWarning)
        results[model_type] = {"model": model, "best_params": best_params, "cv_score": cv_score,
                               "cv_score_se": cv_score_se, "tuning_metric": tuning_metric,
                               "convergence": convergence_info, **ev}
        if multi_db:
            results[model_type]["by_database"] = evaluate_model_by_database(model, X_test, y_test, test_df)
        clean_params = {k.replace("classifier__", ""): v for k, v in best_params.items()}
        print(f"  {model_type}: CV={cv_score:.4f} ± {cv_score_se:.4f} SE  AUC={ev['auc']:.4f}  "
              f"AP={ev['average_precision']:.4f}  Brier={ev['brier']:.4f}  "
              f"CalSlope={ev['cal_slope']:.3f}  CalInt={ev['cal_intercept']:.3f}  "
              f"params={clean_params}")

    return results, y_test


# =============================================================================
# BLOCK D EXCLUSIONS FOR CAUSE-SPECIFIC OUTCOMES
# =============================================================================

BLOCK_D_EXCLUSIONS = {
    'death_cvd': [c for c in comorb_bin_vars + comorb_dur_vars if 'cvd' in c],
    'death_cancer': [c for c in comorb_bin_vars + comorb_dur_vars if 'cancer' in c],
}


def get_block_vars_for_outcome(block_name, block_def, outcome_col):
    cat_vars, num_vars = block_def['cat'].copy(), block_def['num'].copy()
    if block_name in ['C', 'D'] and outcome_col in BLOCK_D_EXCLUSIONS:
        exclude = BLOCK_D_EXCLUSIONS[outcome_col]
        num_vars = [v for v in num_vars if v not in exclude]
        cat_vars = [v for v in cat_vars if v not in exclude]
    return cat_vars, num_vars


# Zero is an informative value for these fields, so they are excluded from the
# complete-case missingness check.
INFORMATIVE_ZERO_VARS = set(med_binary_vars + comorb_bin_vars + comorb_dur_vars)


def apply_complete_case(train_df, test_df, cat_vars, num_vars, block_name, outcome_label):
    """Drop rows missing required predictors within one outcome-specific block."""
    vars_to_check = [
        variable for variable in (list(cat_vars) + list(num_vars))
        if variable not in INFORMATIVE_ZERO_VARS
    ]

    print(f"\n  Complete-case filtering: {block_name} — {outcome_label}")
    print(f"    Total predictors: {len(cat_vars) + len(num_vars)}")
    print(
        f"    Checked for missingness: {len(vars_to_check)} "
        f"(excluded {len(cat_vars) + len(num_vars) - len(vars_to_check)} informative-zero vars)"
    )

    if vars_to_check:
        train_complete = train_df.dropna(subset=vars_to_check).copy()
        test_complete = test_df.dropna(subset=vars_to_check).copy()
    else:
        train_complete = train_df.copy()
        test_complete = test_df.copy()

    n_train_dropped = len(train_df) - len(train_complete)
    n_test_dropped = len(test_df) - len(test_complete)
    train_drop_pct = 100 * n_train_dropped / len(train_df) if len(train_df) else np.nan
    test_drop_pct = 100 * n_test_dropped / len(test_df) if len(test_df) else np.nan

    print(
        f"    Train: {len(train_df):,} -> {len(train_complete):,} "
        f"(dropped {n_train_dropped:,}, {train_drop_pct:.1f}%)"
    )
    print(
        f"    Test:  {len(test_df):,} -> {len(test_complete):,} "
        f"(dropped {n_test_dropped:,}, {test_drop_pct:.1f}%)"
    )
    return train_complete, test_complete


# =============================================================================
# HYBRID MODEL FUNCTIONS
# =============================================================================


def get_feature_importances_from_pipeline(fitted_pipeline, cat_vars, num_vars):
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    classifier = fitted_pipeline.named_steps['classifier']
    importances = classifier.feature_importances_
    cat_feature_names = []
    if cat_vars and 'cat' in preprocessor.named_transformers_:
        onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = list(onehot.get_feature_names_out(cat_vars))
    all_names = cat_feature_names + list(num_vars)
    imp_df = pd.DataFrame({'transformed_feature': all_names, 'importance': importances})
    original_importance = {}
    for v in num_vars:
        original_importance[v] = imp_df[imp_df['transformed_feature'] == v]['importance'].sum()
    for v in cat_vars:
        mask = imp_df['transformed_feature'].str.startswith(f'{v}_')
        original_importance[v] = imp_df[mask]['importance'].sum()
    return pd.DataFrame([{'feature': k, 'importance': v, 'type': 'categorical' if k in cat_vars else 'numeric'}
                          for k, v in original_importance.items()]).sort_values('importance', ascending=False), imp_df


def get_cv_averaged_importance(train_df, cat_vars, num_vars, groups, xgb_best_params, outcome, n_folds=5):
    """Average XGBoost feature importance across GroupKFold training folds (data pre-imputed)."""
    all_vars = cat_vars + num_vars
    gkf = GroupKFold(n_splits=n_folds)
    importance_accum = None
    for train_idx, _ in gkf.split(train_df[all_vars], groups=groups):
        fold_train = train_df.iloc[train_idx]
        pipeline, _ = create_model_pipeline(cat_vars, num_vars, 'XGB')
        pipeline.set_params(**xgb_best_params)
        pipeline.fit(fold_train[all_vars], fold_train[outcome])
        imp_df, _ = get_feature_importances_from_pipeline(pipeline, cat_vars, num_vars)
        imp_df = imp_df.set_index('feature')['importance']
        importance_accum = imp_df if importance_accum is None else importance_accum.add(imp_df, fill_value=0)
    avg_importance = (importance_accum / n_folds).reset_index()
    avg_importance.columns = ['feature', 'importance']
    avg_importance['type'] = avg_importance['feature'].apply(lambda f: 'categorical' if f in cat_vars else 'numeric')
    return avg_importance.sort_values('importance', ascending=False)


def feature_count_grid(max_features):
    """
    Grid over candidate feature counts. Finer than a pure geometric sequence
    around the elbow region (5-34), where the previous coarse grid produced
    knife-edge selections that jumped from n=8 straight to n=13 or n=21.
    """
    base = [1, 2, 3, 5, 8, 11, 13, 16, 21, 26, 34, 42, 55]
    return sorted({n for n in base if n <= max_features} | {max_features})


def train_hybrid_model(train_df, test_df, cat_vars, num_vars, groups, outcome, tuning_metric):
    all_vars = cat_vars + num_vars
    X_train, X_test = train_df[all_vars].copy(), test_df[all_vars].copy()
    y_train, y_test = train_df[outcome].copy(), test_df[outcome].copy()
    model, best_params, cv_score, cv_score_se, _ = train_model(
        X_train, y_train, groups, cat_vars, num_vars, "LR", tuning_metric)
    ev = evaluate_model(model, X_test, y_test)
    result = {"model": model, "best_params": best_params, "cv_score": cv_score,
              "cv_score_se": cv_score_se, **ev}
    if test_df["database"].nunique() > 1:
        result["by_database"] = evaluate_model_by_database(model, X_test, y_test, test_df)
    return result


def evaluate_hybrid_subset_cv(train_df, cat_vars, num_vars, groups, outcome, tuning_metric):
    """Tune/score one candidate subset on development data only (test never touched)."""
    all_vars = cat_vars + num_vars
    X_train, y_train = train_df[all_vars].copy(), train_df[outcome].copy()
    _, best_params, cv_score, cv_score_se, cv_fold_scores = train_model(
        X_train, y_train, groups, cat_vars, num_vars, "LR", tuning_metric)
    return {"best_params": best_params, "cv_score": cv_score,
            "cv_score_se": cv_score_se, "cv_fold_scores": cv_fold_scores}


def select_top_features(importance_df, cat_vars, num_vars, n_features):
    selected = importance_df.head(n_features)["feature"].tolist()
    return ([f for f in selected if f in cat_vars], [f for f in selected if f in num_vars])


def build_hybrid_model(results, block_cat_vars, block_num_vars, train_df, test_df, groups,
                       outcome, tuning_metric, capture=None):
    """
    Select the number of XGB-ranked predictors for the hybrid LR.

    CAPTURE RULE (pre-specified): keep the smallest subset whose mean CV score
    captures at least `capture` of the total achievable improvement, measured
    from the n=1 baseline to the best-scoring subset.

    Why not a statistical tie-test: at n~1.6M the per-fold CV standard errors
    are ~1e-3 (all-cause) to ~1e-4 (cause-specific), so any "indistinguishable
    from best" criterion detects every difference and degenerates to argmax.
    Why not an absolute Brier tolerance: at ~3.4% prevalence the ENTIRE
    achievable Brier gain is ~3e-4, smaller than any sensible tolerance, so it
    degenerates to n=1. A fraction of the achievable gain is scale-free and
    behaves sensibly for both.
    """
    capture = args.hybrid_capture if capture is None else capture

    xgb_best_params = results["XGB"]["best_params"]
    importance_df = get_cv_averaged_importance(train_df, block_cat_vars, block_num_vars,
                                               groups, xgb_best_params, outcome=outcome)

    max_features = len(block_cat_vars) + len(block_num_vars)
    hybrid_results_by_n = []
    for n_features in feature_count_grid(max_features):
        selected_cat, selected_num = select_top_features(importance_df, block_cat_vars, block_num_vars, n_features)
        if not selected_cat and not selected_num:
            continue
        cv_result = evaluate_hybrid_subset_cv(train_df, selected_cat, selected_num, groups, outcome, tuning_metric)
        hybrid_results_by_n.append({
            "n_features": n_features, "selected_cat": selected_cat, "selected_num": selected_num,
            "cv_score": cv_result["cv_score"], "cv_score_se": cv_result["cv_score_se"],
            "best_params": cv_result["best_params"],
        })

    comp_df = pd.DataFrame(hybrid_results_by_n).sort_values("n_features").reset_index(drop=True)
    if comp_df.empty:
        raise RuntimeError("No valid hybrid feature subsets were evaluated.")

    baseline_score = float(comp_df.loc[comp_df["n_features"].idxmin(), "cv_score"])
    best_idx = comp_df["cv_score"].idxmax()
    best_mean_score = float(comp_df.loc[best_idx, "cv_score"])
    best_marginal_se = float(comp_df.loc[best_idx, "cv_score_se"])
    total_gain = best_mean_score - baseline_score

    if total_gain > 0:
        comp_df["fraction_captured"] = (comp_df["cv_score"] - baseline_score) / total_gain
        capture_threshold = baseline_score + capture * total_gain
        eligible = comp_df[comp_df["cv_score"] >= capture_threshold]
        selected_idx = eligible["n_features"].idxmin()
        selection_label = f"{capture:.0%}-capture rule"
    else:
        # Degenerate curve (no achievable improvement): fall back to the smallest.
        comp_df["fraction_captured"] = 1.0
        capture_threshold = best_mean_score
        selected_idx = comp_df["n_features"].idxmin()
        selection_label = "smallest subset (no achievable gain)"

    comp_df["capture_target"] = capture
    comp_df["capture_threshold"] = capture_threshold
    comp_df["total_achievable_gain"] = total_gain
    # 1-SE band retained for reference/supplementary reporting only.
    comp_df["one_se_threshold"] = best_mean_score - best_marginal_se
    comp_df["within_one_se"] = comp_df["cv_score"] >= comp_df["one_se_threshold"]
    comp_df["selected"] = False
    comp_df.loc[selected_idx, "selected"] = True

    best_n = int(comp_df.loc[selected_idx, "n_features"])
    best_cat = comp_df.loc[selected_idx, "selected_cat"]
    best_num = comp_df.loc[selected_idx, "selected_num"]
    frac = float(comp_df.loc[selected_idx, "fraction_captured"])

    final_hybrid = train_hybrid_model(train_df, test_df, best_cat, best_num, groups, outcome, tuning_metric)

    print(f"  HYBRID ({selection_label}) selected n={best_n} of {max_features} "
          f"(captured {frac:.1%} of achievable gain {total_gain:.6f}); "
          f"Test AUC={final_hybrid['auc']:.4f}, Brier={final_hybrid['brier']:.4f}")

    return final_hybrid, importance_df, comp_df, best_cat, best_num, best_n


def extract_hybrid_coefficients(hybrid_model, cat_vars, num_vars):
    preprocessor = hybrid_model.named_steps['preprocessor']
    classifier = hybrid_model.named_steps['classifier']
    cat_feature_names = []
    if cat_vars and 'cat' in preprocessor.named_transformers_:
        onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = list(onehot.get_feature_names_out(cat_vars))
    feature_names = cat_feature_names + num_vars
    coefficients = classifier.coef_[0]
    return pd.DataFrame({'feature': feature_names, 'coefficient': coefficients,
                          'odds_ratio': np.exp(coefficients)}).sort_values('coefficient', key=abs, ascending=False)


# =============================================================================
# DECISION CURVE ANALYSIS
# =============================================================================


def dca_thresholds_for(prevalence, n_points=100):
    """
    Risk thresholds scaled to the outcome. A fixed 0-0.5 grid wastes most of
    the panel for a 3.4%-prevalence outcome, where no clinician would use a
    40% threshold. Upper limit ~4x prevalence, floored at 0.10, capped at 0.50.
    """
    upper = float(min(0.50, max(0.10, 4.0 * prevalence)))
    return np.linspace(0.001, upper, n_points)


def net_benefit(y_true, y_pred, thresholds):
    """Calculate standard decision-curve net benefit across risk thresholds."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    output = []
    for threshold in thresholds:
        predicted_positive = y_pred >= threshold
        true_positive_rate = np.sum(predicted_positive & (y_true == 1)) / n
        false_positive_rate = np.sum(predicted_positive & (y_true == 0)) / n
        output.append(true_positive_rate - false_positive_rate * (threshold / (1 - threshold)))
    return np.asarray(output)


# =============================================================================
# PRACTICE-CLUSTER BOOTSTRAP
# =============================================================================

N_BOOTSTRAPS = args.n_bootstraps if args.n_bootstraps is not None else (100 if args.pilot else 2000)


def _oe_ratio_metric(y_true, y_pred):
    return y_true.mean() / y_pred.mean() if y_pred.mean() > 0 else np.nan


def _cal_slope_metric(y_true, y_pred):
    return calibration_slope_intercept(y_true, y_pred)[0]


def _cal_intercept_metric(y_true, y_pred):
    return calibration_slope_intercept(y_true, y_pred)[1]


METRIC_FNS = {'auc': roc_auc_score, 'average_precision': average_precision_score,
              'brier': brier_score_loss, 'oe_ratio': _oe_ratio_metric,
              'cal_slope': _cal_slope_metric, 'cal_intercept': _cal_intercept_metric}


def _prepare_cluster_bootstrap(cluster_ids):
    cluster_series = pd.Series(np.asarray(cluster_ids), dtype="object").reset_index(drop=True)
    if cluster_series.isna().any():
        raise ValueError("Practice-cluster identifiers contain missing values.")
    grouped_indices = cluster_series.groupby(cluster_series, sort=False).indices
    unique_clusters = np.asarray(list(grouped_indices.keys()), dtype=object)
    cluster_to_indices = {c: np.asarray(idx, dtype=int) for c, idx in grouped_indices.items()}
    if len(unique_clusters) < 2:
        raise ValueError("At least two practices are required for cluster bootstrapping.")
    return unique_clusters, cluster_to_indices


def _draw_cluster_bootstrap_indices(rng, unique_clusters, cluster_to_indices):
    sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
    return np.concatenate([cluster_to_indices[c] for c in sampled_clusters])


def bootstrap_all_metrics_ci(y_true, y_pred, cluster_ids, n_bootstraps=N_BOOTSTRAPS, ci=0.95, random_state=42):
    y_true, y_pred, cluster_ids = np.asarray(y_true), np.asarray(y_pred), np.asarray(cluster_ids)
    if not (len(y_true) == len(y_pred) == len(cluster_ids)):
        raise ValueError("y_true, y_pred and cluster_ids must have equal lengths.")
    unique_clusters, cluster_to_indices = _prepare_cluster_bootstrap(cluster_ids)
    rng = np.random.default_rng(random_state)
    boot_vals = {m: [] for m in METRIC_FNS}

    for _ in range(n_bootstraps):
        indices = _draw_cluster_bootstrap_indices(rng, unique_clusters, cluster_to_indices)
        y_boot, pred_boot = y_true[indices], y_pred[indices]
        if np.unique(y_boot).size < 2:
            continue

        simple_metrics = {
            "auc": lambda: roc_auc_score(y_boot, pred_boot),
            "average_precision": lambda: average_precision_score(y_boot, pred_boot),
            "brier": lambda: brier_score_loss(y_boot, pred_boot),
            "oe_ratio": lambda: _oe_ratio_metric(y_boot, pred_boot),
        }
        for metric_name, metric_fn in simple_metrics.items():
            try:
                value = metric_fn()
                if np.isfinite(value):
                    boot_vals[metric_name].append(value)
            except Exception:
                continue

        # Fit the one-variable calibration model once and retain both parameters.
        try:
            slope, intercept = calibration_slope_intercept(y_boot, pred_boot)
            if np.isfinite(slope):
                boot_vals["cal_slope"].append(slope)
            if np.isfinite(intercept):
                boot_vals["cal_intercept"].append(intercept)
        except Exception:
            pass

    try:
        point_slope, point_intercept = calibration_slope_intercept(y_true, y_pred)
    except Exception:
        point_slope, point_intercept = np.nan, np.nan
    point_estimates = {
        "auc": float(roc_auc_score(y_true, y_pred)),
        "average_precision": float(average_precision_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "oe_ratio": float(_oe_ratio_metric(y_true, y_pred)),
        "cal_slope": float(point_slope),
        "cal_intercept": float(point_intercept),
    }

    alpha = 1.0 - ci
    output = {}
    for metric_name, values in boot_vals.items():
        values = np.asarray(values, dtype=float)
        point_estimate = point_estimates[metric_name]
        if len(values) == 0:
            output[metric_name] = {
                "estimate": point_estimate,
                "bootstrap_mean": np.nan,
                "lower": np.nan,
                "upper": np.nan,
                "n_successful": 0,
            }
            continue
        output[metric_name] = {
            "estimate": point_estimate,
            "bootstrap_mean": float(np.mean(values)),
            "lower": float(np.percentile(values, 100 * alpha / 2)),
            "upper": float(np.percentile(values, 100 * (1 - alpha / 2))),
            "n_successful": int(len(values)),
        }
    return output


def bootstrap_metric_comparison(y_true, y_pred_1, y_pred_2, cluster_ids, metric_fn,
                                n_bootstraps=N_BOOTSTRAPS, random_state=42):
    y_true, y_pred_1, y_pred_2 = np.asarray(y_true), np.asarray(y_pred_1), np.asarray(y_pred_2)
    cluster_ids = np.asarray(cluster_ids)
    if not (len(y_true) == len(y_pred_1) == len(y_pred_2) == len(cluster_ids)):
        raise ValueError("All prediction, outcome and cluster arrays must have equal lengths.")
    observed_diff = float(metric_fn(y_true, y_pred_2) - metric_fn(y_true, y_pred_1))
    unique_clusters, cluster_to_indices = _prepare_cluster_bootstrap(cluster_ids)
    rng = np.random.default_rng(random_state)
    differences = []
    for _ in range(n_bootstraps):
        indices = _draw_cluster_bootstrap_indices(rng, unique_clusters, cluster_to_indices)
        y_boot = y_true[indices]
        if np.unique(y_boot).size < 2:
            continue
        try:
            difference = metric_fn(y_boot, y_pred_2[indices]) - metric_fn(y_boot, y_pred_1[indices])
            if np.isfinite(difference):
                differences.append(difference)
        except Exception:
            continue
    differences = np.asarray(differences, dtype=float)
    if len(differences) == 0:
        return observed_diff, np.nan, np.nan, np.nan, 0
    lower, upper = float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))
    probability_nonpositive = (np.sum(differences <= 0) + 1) / (len(differences) + 1)
    probability_nonnegative = (np.sum(differences >= 0) + 1) / (len(differences) + 1)
    p_value = float(min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative)))
    return observed_diff, lower, upper, p_value, int(len(differences))


COMPARISON_METRICS = {
    'auc': {'function': roc_auc_score, 'higher_is_better': True},
    'brier': {'function': brier_score_loss, 'higher_is_better': False},
}
# Six primary comparisons among manuscript models, plus two prespecified
# secondary comparisons involving flexible logistic regression.
MODEL_PAIRS_TO_COMPARE = [
    ('LR', 'RF'), ('LR', 'XGB'), ('LR', 'HYBRID'),
    ('RF', 'XGB'), ('RF', 'HYBRID'), ('XGB', 'HYBRID'),
    ('LR', 'LR_FLEX'), ('LR_FLEX', 'XGB'),
]
MODEL_TYPES = ['LR', 'LR_FLEX', 'RF', 'XGB']
MANUSCRIPT_MODELS = ['LR', 'RF', 'XGB', 'HYBRID']


def get_model_predictions(standard_results, hybrid_result, model_type):
    """Return predictions regardless of whether a model is standard or HYBRID."""
    if model_type == 'HYBRID':
        return np.asarray(hybrid_result['y_pred_proba'])
    if model_type not in standard_results:
        raise KeyError(f"Model {model_type!r} is not available.")
    return np.asarray(standard_results[model_type]['y_pred_proba'])


def comparison_favour_label(model_1, model_2, difference, lower, upper, higher_is_better):
    """Human-readable interpretation of model_2 minus model_1."""
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 <= upper:
        return 'No clear difference (95% CI includes 0)'
    model_2_better = difference > 0 if higher_is_better else difference < 0
    return model_2 if model_2_better else model_1

# Calibration plotting: quantile bins put equal NUMBERS of patients in each bin,
# so bins land where the predictions actually are. Uniform-width bins are
# unusable for rare outcomes (nearly all patients fall in the first two bins,
# and the high-risk bins hold a handful of people each, producing wild noise
# that looks like miscalibration even when slope/intercept are ~1.00/0.00).
CAL_N_BINS = 20
CAL_STRATEGY = 'quantile'


def save_block_pickle(prefix, file_suffix, outcome_col, block_name,
                      results, hybrid_result, importance_df, hybrid_vars, best_n):
    """Save one outcome/block's fitted models immediately to limit peak RAM."""
    key_str = f"{outcome_col}__{block_name}"
    block_output = {
        f'results_{key_str}': {mt: {k: v for k, v in result.items() if k != 'model'}
                               for mt, result in results.items()},
        f'models_{key_str}': {mt: result['model'] for mt, result in results.items()},
        f'hybrid_result_{key_str}': {k: v for k, v in hybrid_result.items() if k != 'model'},
        f'hybrid_model_{key_str}': hybrid_result['model'],
        f'hybrid_vars_{key_str}': hybrid_vars,
        f'hybrid_best_n_{key_str}': best_n,
        f'importance_{key_str}': importance_df,
    }
    filename = f'{prefix}_models_{block_name}{file_suffix}.pkl'
    with open(filename, 'wb') as handle:
        pickle.dump(block_output, handle)
    return filename


# =============================================================================
# MAIN LOOP — wrapped for repeated splits
# =============================================================================

stratify = (DB_FILTER == 'all')
all_split_summaries = []

for split_idx in range(args.n_splits):
    split_seed = 42 + split_idx
    RUN_PREFIX = run_level_prefix(split_idx)
    print(f"\n{'#' * 60}\nSPLIT {split_idx + 1}/{args.n_splits} (seed={split_seed})\n{'#' * 60}")
    split_t0 = time.perf_counter()

    train_df, test_df, split_info = split_by_practice(df_eligible, random_state=split_seed, stratify_by_database=stratify)
    pd.to_pickle(split_info, f'{RUN_PREFIX}_practice_split_info{FILE_SUFFIX}.pkl')
    run_configuration = {
        'script_version': SCRIPT_VERSION,
        'data_path': os.path.abspath(data_path),
        'database': DB_FILTER,
        'sex': SEX_FILTER,
        'diabetes_type': DM_TYPE,
        'outcome_scope': args.outcome,
        'outcomes_available_this_run': list(OUTCOMES),
        'sensitivity': args.sensitivity,
        'tuning_metric': TUNING_METRIC,
        'hybrid_capture': args.hybrid_capture,
        'n_splits': args.n_splits,
        'split_index': split_idx,
        'split_seed': split_seed,
        'study_end_date': args.study_end_date,
        'n_bootstraps': N_BOOTSTRAPS,
        'rf_n_estimators': args.rf_n_estimators if args.rf_n_estimators is not None else 750,
        'cache_imputed': args.cache_imputed,
    }
    with open(f'{RUN_PREFIX}_run_configuration{FILE_SUFFIX}.json', 'w', encoding='utf-8') as handle:
        json.dump(run_configuration, handle, indent=2, sort_keys=True)

    if args.rf_oob_test:
        block_definition = BLOCKS["D"]
        oob_cat_vars, oob_num_vars = get_block_vars_for_outcome("D", block_definition, args.oob_outcome)
        if args.oob_outcome not in train_df.columns:
            raise ValueError(f"Requested OOB outcome {args.oob_outcome!r} is not available.")
        oob_train_df = train_df
        if args.sensitivity == 'complete_case':
            oob_train_df, _ = apply_complete_case(
                train_df, test_df, oob_cat_vars, oob_num_vars,
                BLOCK_DISPLAY_NAMES["D"], OUTCOMES.get(args.oob_outcome, args.oob_outcome))
        run_rf_oob_tree_test(train_df=oob_train_df, cat_vars=oob_cat_vars, num_vars=oob_num_vars,
                             outcome=args.oob_outcome,
                             output_prefix=f"{outcome_prefix(args.oob_outcome, split_idx)}_oob",
                             sample_n=args.oob_sample_n, tree_counts=[100, 200, 300, 500, 750, 1000],
                             max_features=0.25, min_samples_leaf=5, random_state=split_seed)
        raise SystemExit("RF OOB diagnostic complete.")

    train_imp, test_imp = {}, {}
    if args.sensitivity == 'complete_case':
        print("\n" + "=" * 60)
        print("COMPLETE-CASE FILTERING (no imputation)")
        print("=" * 60)
    else:
        # Impute each UNIQUE outcome-specific predictor set once. This preserves
        # speed for identical A/B/C definitions while ensuring that CVD/cancer
        # predictors excluded from Block D cannot influence imputation indirectly.
        imp_t0 = time.perf_counter()
        imputed_by_signature = {}
        for outcome_col in OUTCOMES:
            for bname, bdef in BLOCKS.items():
                cat_vars, num_vars = get_block_vars_for_outcome(bname, bdef, outcome_col)
                signature = (bname, tuple(cat_vars), tuple(num_vars))
                if signature not in imputed_by_signature:
                    bt = time.perf_counter()
                    cache_label = f"block{bname}_{outcome_col}"
                    imputed_by_signature[signature] = impute_block_cached(
                        train_df, test_df, cat_vars, num_vars,
                        cache_label, split_idx, split_seed
                    )
                    print(
                        f"  Block {bname} ({outcome_col}) imputed in "
                        f"{time.perf_counter() - bt:.1f}s"
                    )
                train_imp[(outcome_col, bname)], test_imp[(outcome_col, bname)] = (
                    imputed_by_signature[signature]
                )
        print(
            f"  Imputation total: {time.perf_counter() - imp_t0:.1f}s "
            f"across {len(imputed_by_signature)} unique predictor sets"
        )

    ALL_RESULTS, ALL_HYBRID, ALL_Y_TEST = {}, {}, {}
    ALL_TEST_FRAMES, ALL_SAMPLE_SIZES = {}, {}
    ALL_IMPORTANCE, ALL_HYBRID_COMP, ALL_HYBRID_VARS, ALL_HYBRID_BEST_N = {}, {}, {}, {}
    BLOCK_PICKLE_FILES = {}

    for outcome_col, outcome_label in OUTCOMES.items():
        OUT_PREFIX = outcome_prefix(outcome_col, split_idx)

        for block_name, block_def in BLOCKS.items():
            key = (outcome_col, block_name)
            cat_vars, num_vars = get_block_vars_for_outcome(block_name, block_def, outcome_col)

            if args.sensitivity == 'complete_case':
                tr_b, te_b = apply_complete_case(
                    train_df, test_df, cat_vars, num_vars,
                    BLOCK_DISPLAY_NAMES[block_name], outcome_label)
            else:
                tr_b, te_b = train_imp[key], test_imp[key]

            n_train_events = int(tr_b[outcome_col].sum())
            n_test_events = int(te_b[outcome_col].sum())
            n_train_non_events = int(len(tr_b) - n_train_events)
            n_test_non_events = int(len(te_b) - n_test_events)
            if (n_train_events < 50 or n_test_events < 20 or
                    n_train_non_events < 50 or n_test_non_events < 20):
                print(
                    f"  SKIPPING {BLOCK_DISPLAY_NAMES[block_name]} — {outcome_label}: "
                    f"too few events after preprocessing "
                    f"(train events/non-events={n_train_events:,}/{n_train_non_events:,}; "
                    f"test events/non-events={n_test_events:,}/{n_test_non_events:,})"
                )
                continue

            ALL_Y_TEST[key] = te_b[outcome_col].copy()
            ALL_TEST_FRAMES[key] = te_b[['database', 'pracid']].copy()
            ALL_SAMPLE_SIZES[key] = {
                'n_train': int(len(tr_b)),
                'n_test': int(len(te_b)),
                'events_train': n_train_events,
                'events_test': n_test_events,
            }

            results, _ = train_and_evaluate_block(
                tr_b, te_b, cat_vars, num_vars,
                f"{BLOCK_DISPLAY_NAMES[block_name]} — {outcome_label}", outcome_col, TUNING_METRIC)

            groups = tr_b["pracid"]
            hybrid_result, importance, hybrid_comp, best_cat, best_num, best_n = build_hybrid_model(
                results, cat_vars, num_vars, tr_b, te_b, groups, outcome_col, TUNING_METRIC)

            # Save fitted models immediately, then retain only predictions/metrics in RAM.
            model_file = save_block_pickle(
                OUT_PREFIX, FILE_SUFFIX, outcome_col, block_name,
                results, hybrid_result, importance, (best_cat, best_num), best_n)
            BLOCK_PICKLE_FILES[key] = model_file
            print(f"  Saved block models to {model_file}")

            # Coefficients require the fitted hybrid model, so save before cleanup.
            coef_df = extract_hybrid_coefficients(hybrid_result['model'], best_cat, best_num)
            coef_df.to_csv(f'{OUT_PREFIX}_hybrid_{block_name}_coef{FILE_SUFFIX}.csv', index=False)

            ALL_RESULTS[key] = {
                mt: {k: v for k, v in result.items() if k != 'model'}
                for mt, result in results.items()
            }
            ALL_HYBRID[key] = {k: v for k, v in hybrid_result.items() if k != 'model'}
            ALL_IMPORTANCE[key] = importance
            ALL_HYBRID_COMP[key] = hybrid_comp
            ALL_HYBRID_VARS[key] = (best_cat, best_num)
            ALL_HYBRID_BEST_N[key] = best_n

            del results, hybrid_result, coef_df
            gc.collect()

    # =========================================================================
    # PLOTS — manuscript models only: LR, RF, XGB and HYBRID
    # LR_FLEX remains trained and saved for internal/supervisor review.
    # =========================================================================
    analysis_label = {
        'none': 'Imputed',
        'complete_case': 'Complete Case',
        'complete_fu': 'Complete Follow-up',
    }[args.sensitivity]
    plot_label = f"DB={DB_FILTER.upper()}, Sex={SEX_FILTER.upper()}, {analysis_label}"

    for outcome_col, outcome_label in OUTCOMES.items():
        available_blocks = [b for b in BLOCKS if (outcome_col, b) in ALL_RESULTS]
        if not available_blocks:
            continue
        OUT_PREFIX = outcome_prefix(outcome_col, split_idx)
        n_blocks = len(available_blocks)
        outcome_dca_rows = []

        # --- ROC CURVES ---
        fig, axes = plt.subplots(1, n_blocks, figsize=(5 * n_blocks, 5))
        if n_blocks == 1:
            axes = [axes]
        for i, bname in enumerate(available_blocks):
            key = (outcome_col, bname)
            y_test = ALL_Y_TEST[key].values
            ax = axes[i]
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, lw=1)
            for mt in MANUSCRIPT_MODELS:
                model_result = ALL_HYBRID[key] if mt == 'HYBRID' else ALL_RESULTS[key][mt]
                fpr, tpr, _ = roc_curve(y_test, model_result['y_pred_proba'])
                ax.plot(fpr, tpr, color=MODEL_COLORS[mt], lw=2,
                        label=f'{mt} ({model_result["auc"]:.3f})')
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
            ax.set_title(BLOCK_DISPLAY_NAMES[bname])
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.suptitle(f'ROC Curves — {outcome_label} ({plot_label})')
        plt.tight_layout()
        plt.savefig(f'{OUT_PREFIX}_roc{FILE_SUFFIX}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # --- PRECISION-RECALL CURVES ---
        fig, axes = plt.subplots(1, n_blocks, figsize=(5 * n_blocks, 5))
        if n_blocks == 1:
            axes = [axes]
        for i, bname in enumerate(available_blocks):
            key = (outcome_col, bname)
            y_test = ALL_Y_TEST[key].values
            prevalence = float(y_test.mean())
            ax = axes[i]
            ax.axhline(prevalence, color='k', linestyle='--', alpha=0.6, lw=1,
                       label=f'No skill ({prevalence:.3f})')
            for mt in MANUSCRIPT_MODELS:
                model_result = ALL_HYBRID[key] if mt == 'HYBRID' else ALL_RESULTS[key][mt]
                precision, recall, _ = precision_recall_curve(y_test, model_result['y_pred_proba'])
                ax.plot(recall, precision, color=MODEL_COLORS[mt], lw=2,
                        label=f'{mt} (AP {model_result["average_precision"]:.3f})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(BLOCK_DISPLAY_NAMES[bname])
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        plt.suptitle(f'Precision–Recall Curves — {outcome_label} ({plot_label})')
        plt.tight_layout()
        plt.savefig(f'{OUT_PREFIX}_precision_recall{FILE_SUFFIX}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # --- CALIBRATION (overlaid, one panel per block, quantile bins) ---
        fig, axes = plt.subplots(1, n_blocks, figsize=(5 * n_blocks, 5))
        if n_blocks == 1:
            axes = [axes]
        for i, bname in enumerate(available_blocks):
            key = (outcome_col, bname)
            y_test = ALL_Y_TEST[key].values
            ax = axes[i]
            axis_max = 0.0
            for mt in MANUSCRIPT_MODELS:
                model_result = ALL_HYBRID[key] if mt == 'HYBRID' else ALL_RESULTS[key][mt]
                prob_true, prob_pred = calibration_curve(
                    y_test, model_result['y_pred_proba'],
                    n_bins=CAL_N_BINS, strategy=CAL_STRATEGY)
                ax.plot(prob_pred, prob_true, marker=MODEL_MARKERS[mt], markersize=4,
                        lw=1.8, color=MODEL_COLORS[mt],
                        label=f'{mt} (slope {model_result["cal_slope"]:.2f}, '
                              f'int {model_result["cal_intercept"]:.2f})')
                axis_max = max(axis_max, float(prob_pred.max()), float(prob_true.max()))
            # Scale axes to the observed prediction range rather than always [0, 1],
            # so rare-outcome panels are not 95% empty space.
            lim = float(min(1.0, max(0.05, axis_max * 1.10)))
            ax.plot([0, lim], [0, lim], 'k--', alpha=0.7, lw=1, label='Perfect calibration')
            ax.set_xlim([0, lim])
            ax.set_ylim([0, lim])
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('Predicted risk')
            ax.set_ylabel('Observed risk')
            ax.set_title(BLOCK_DISPLAY_NAMES[bname])
            ax.legend(loc='upper left', fontsize=7)
            ax.grid(True, alpha=0.3)
        plt.suptitle(f'Calibration — {outcome_label} ({plot_label}); '
                     f'{CAL_N_BINS} quantile bins')
        plt.tight_layout()
        plt.savefig(f'{OUT_PREFIX}_calibration{FILE_SUFFIX}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # --- DECISION CURVE ANALYSIS (thresholds scaled to prevalence) ---
        fig, axes = plt.subplots(1, n_blocks, figsize=(5 * n_blocks, 5))
        if n_blocks == 1:
            axes = [axes]
        for i, bname in enumerate(available_blocks):
            key = (outcome_col, bname)
            y_test = ALL_Y_TEST[key].values
            prevalence = float(y_test.mean())
            thresholds = dca_thresholds_for(prevalence)
            treat_none = np.zeros_like(thresholds)
            treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
            ax = axes[i]
            ax.plot(thresholds, treat_none, 'k--', lw=1.2, label='Treat none')
            ax.plot(thresholds, treat_all, 'k:', lw=1.2, label='Treat all')
            for mt in MANUSCRIPT_MODELS:
                model_result = ALL_HYBRID[key] if mt == 'HYBRID' else ALL_RESULTS[key][mt]
                net_benefits = net_benefit(y_test, model_result['y_pred_proba'], thresholds)
                ax.plot(thresholds, net_benefits, color=MODEL_COLORS[mt], lw=2, label=mt)
                for threshold, nb_value in zip(thresholds, net_benefits):
                    outcome_dca_rows.append({
                        'Outcome': outcome_label,
                        'Block': BLOCK_DISPLAY_NAMES[bname],
                        'Model': mt,
                        'Threshold': threshold,
                        'Net_benefit': nb_value,
                    })
            ax.set_xlim(thresholds.min(), thresholds.max())
            ax.set_ylim(-0.01, max(0.02, prevalence * 1.15))
            ax.set_xlabel('Risk threshold')
            ax.set_ylabel('Net benefit')
            ax.set_title(f'{BLOCK_DISPLAY_NAMES[bname]} (prev {prevalence:.3f})')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.suptitle(f'Decision Curve Analysis — {outcome_label} ({plot_label})')
        plt.tight_layout()
        plt.savefig(f'{OUT_PREFIX}_dca{FILE_SUFFIX}.png', dpi=150, bbox_inches='tight')
        plt.close()
        pd.DataFrame(outcome_dca_rows).to_csv(
            f'{OUT_PREFIX}_dca_net_benefit_values{FILE_SUFFIX}.csv', index=False)

        # --- ELBOW / FEATURE-SELECTION PLOTS (capture rule) ---
        fig, axes = plt.subplots(1, n_blocks, figsize=(5 * n_blocks, 5))
        if n_blocks == 1:
            axes = [axes]
        for i, bname in enumerate(available_blocks):
            key = (outcome_col, bname)
            hcomp = ALL_HYBRID_COMP[key]
            best_n = ALL_HYBRID_BEST_N[key]
            ax = axes[i]
            ax.errorbar(hcomp["n_features"], hcomp["cv_score"], yerr=hcomp["cv_score_se"],
                        marker="o", markersize=4, capsize=3, lw=1.6,
                        color=MODEL_COLORS['HYBRID'], label=f"CV {TUNING_METRIC} ± SE")
            cap_threshold = float(hcomp["capture_threshold"].iloc[0])
            ax.axhline(y=cap_threshold, linestyle="--", alpha=0.8, color='#d62728',
                       label=f"{args.hybrid_capture:.0%} capture")
            ax.axvline(x=best_n, linestyle=":", alpha=0.8, color='k',
                       label=f"Selected (n={best_n})")
            ax.set_xlabel('Number of predictors')
            ax.set_ylabel(f'CV {TUNING_METRIC} (higher is better)')
            ax.set_title(BLOCK_DISPLAY_NAMES[bname])
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.suptitle(f'Hybrid Feature Selection ({args.hybrid_capture:.0%}-capture rule) — '
                     f'{outcome_label} ({plot_label})')
        plt.tight_layout()
        plt.savefig(f'{OUT_PREFIX}_elbow{FILE_SUFFIX}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # --- FEATURE IMPORTANCE ---
        fig, axes = plt.subplots(1, n_blocks, figsize=(5 * n_blocks, max(6, n_blocks * 2)))
        if n_blocks == 1:
            axes = [axes]
        legend_elements = [
            Patch(facecolor='steelblue', label='Numeric'),
            Patch(facecolor='forestgreen', label='Categorical'),
        ]
        for i, bname in enumerate(available_blocks):
            key = (outcome_col, bname)
            imp = ALL_IMPORTANCE[key].sort_values('importance', ascending=True)
            if len(imp) > 20:
                imp = imp.tail(20)
            colors_bar = ['steelblue' if t == 'numeric' else 'forestgreen' for t in imp['type']]
            ax = axes[i]
            ax.barh(imp['feature'], imp['importance'], color=colors_bar)
            ax.set_xlabel('Importance (CV-averaged)')
            ax.set_title(BLOCK_DISPLAY_NAMES[bname])
            ax.grid(True, alpha=0.3, axis='x')
            ax.legend(handles=legend_elements, loc='lower right', fontsize=7)
        plt.suptitle(f'XGBoost Feature Importance — {outcome_label} ({plot_label})')
        plt.tight_layout()
        plt.savefig(f'{OUT_PREFIX}_importance{FILE_SUFFIX}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Plots saved for {outcome_label} -> {outcome_dir(outcome_col)}/")

    # =========================================================================
    # BOOTSTRAP CIs — written per outcome
    # =========================================================================
    boot_t0 = time.perf_counter()
    print(f"\nPractice-cluster bootstrapping all metrics ({N_BOOTSTRAPS} reps) — slowest step...")

    for outcome_col, outcome_label in OUTCOMES.items():
        available_blocks = sorted([b for b in BLOCKS if (outcome_col, b) in ALL_RESULTS])
        if not available_blocks:
            continue
        OUT_PREFIX = outcome_prefix(outcome_col, split_idx)
        ci_rows, model_comp_rows, block_comp_rows = [], [], []

        for bname in available_blocks:
            key = (outcome_col, bname)
            y_true = ALL_Y_TEST[key].values
            block_test_frame = ALL_TEST_FRAMES[key]
            practice_cluster_ids = (
                block_test_frame["database"].astype(str)
                + "::"
                + block_test_frame["pracid"].astype(str)
            ).to_numpy()

            for mt in MODEL_TYPES + ['HYBRID']:
                y_pred = (ALL_HYBRID[key]['y_pred_proba'] if mt == 'HYBRID'
                          else ALL_RESULTS[key][mt]['y_pred_proba'])
                ci_out = bootstrap_all_metrics_ci(y_true, y_pred, practice_cluster_ids)
                row = {'Outcome': outcome_label, 'Block': BLOCK_DISPLAY_NAMES[bname], 'Model': mt}
                for metric_name, stats in ci_out.items():
                    row[f'{metric_name}_estimate'] = round(stats['estimate'], 6)
                    row[f'{metric_name}_bootstrap_mean'] = round(stats['bootstrap_mean'], 6)
                    row[f'{metric_name}_lower'] = round(stats['lower'], 6)
                    row[f'{metric_name}_upper'] = round(stats['upper'], 6)
                    row[f'{metric_name}_n_successful'] = stats['n_successful']
                ci_rows.append(row)
                print(f"  {mt}-{BLOCK_DISPLAY_NAMES[bname]} ({outcome_label}): "
                      f"AUC={ci_out['auc']['estimate']:.4f} "
                      f"({ci_out['auc']['lower']:.4f}-{ci_out['auc']['upper']:.4f})  "
                      f"AP={ci_out['average_precision']['estimate']:.4f} "
                      f"({ci_out['average_precision']['lower']:.4f}-"
                      f"{ci_out['average_precision']['upper']:.4f})  "
                      f"Brier={ci_out['brier']['estimate']:.4f} "
                      f"({ci_out['brier']['lower']:.4f}-{ci_out['brier']['upper']:.4f})")

            available_models = set(ALL_RESULTS[key]) | {'HYBRID'}
            for m1, m2 in MODEL_PAIRS_TO_COMPARE:
                if m1 not in available_models or m2 not in available_models:
                    continue
                y1 = get_model_predictions(ALL_RESULTS[key], ALL_HYBRID[key], m1)
                y2 = get_model_predictions(ALL_RESULTS[key], ALL_HYBRID[key], m2)
                for metric_name, metric_spec in COMPARISON_METRICS.items():
                    metric_fn = metric_spec['function']
                    higher_is_better = metric_spec['higher_is_better']
                    diff, lo, hi, p, n_successful = bootstrap_metric_comparison(
                        y_true, y1, y2, practice_cluster_ids, metric_fn)
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    model_comp_rows.append({
                        'Outcome': outcome_label,
                        'Block': BLOCK_DISPLAY_NAMES[bname],
                        'Model_1': m1,
                        'Model_2': m2,
                        'Comparison': f'{m1} vs {m2}',
                        'Metric': metric_name,
                        'Estimate_model_1': round(float(metric_fn(y_true, y1)), 6),
                        'Estimate_model_2': round(float(metric_fn(y_true, y2)), 6),
                        'Difference_model_2_minus_model_1': round(diff, 6),
                        'CI_lower': round(lo, 6),
                        'CI_upper': round(hi, 6),
                        'p_value': round(p, 6),
                        'Significant': sig,
                        'Favours': comparison_favour_label(
                            m1, m2, diff, lo, hi, higher_is_better
                        ),
                        'Positive_difference_favours': m2 if higher_is_better else m1,
                        'n_successful_bootstraps': n_successful,
                    })

        # Cross-block comparisons require the same patients in each block.
        # Complete-case cohorts differ by block, so only within-block is valid.
        if args.sensitivity != 'complete_case':
            for i in range(len(available_blocks) - 1):
                b1, b2 = available_blocks[i], available_blocks[i + 1]
                comparison_key = (outcome_col, b1)
                y_true_blocks = ALL_Y_TEST[comparison_key].values
                comparison_test_frame = ALL_TEST_FRAMES[comparison_key]
                comparison_cluster_ids = (
                    comparison_test_frame["database"].astype(str)
                    + "::"
                    + comparison_test_frame["pracid"].astype(str)
                ).to_numpy()
                y1 = ALL_RESULTS[(outcome_col, b1)]['XGB']['y_pred_proba']
                y2 = ALL_RESULTS[(outcome_col, b2)]['XGB']['y_pred_proba']
                for metric_name, metric_spec in COMPARISON_METRICS.items():
                    metric_fn = metric_spec['function']
                    diff, lo, hi, p, n_successful = bootstrap_metric_comparison(
                        y_true_blocks, y1, y2, comparison_cluster_ids, metric_fn)
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    block_1_label = BLOCK_DISPLAY_NAMES[b1]
                    block_2_label = BLOCK_DISPLAY_NAMES[b2]
                    higher_is_better = metric_spec['higher_is_better']
                    block_comp_rows.append({
                        'Outcome': outcome_label,
                        'Model': 'XGB',
                        'Block_1': block_1_label,
                        'Block_2': block_2_label,
                        'Comparison': f'XGB: {block_1_label}→{block_2_label}',
                        'Metric': metric_name,
                        'Estimate_block_1': round(float(metric_fn(y_true_blocks, y1)), 6),
                        'Estimate_block_2': round(float(metric_fn(y_true_blocks, y2)), 6),
                        'Difference_block_2_minus_block_1': round(diff, 6),
                        'CI_lower': round(lo, 6),
                        'CI_upper': round(hi, 6),
                        'p_value': round(p, 6),
                        'Significant': sig,
                        'Favours': comparison_favour_label(
                            block_1_label, block_2_label, diff, lo, hi, higher_is_better
                        ),
                        'Positive_difference_favours': (
                            block_2_label if higher_is_better else block_1_label
                        ),
                        'n_successful_bootstraps': n_successful,
                    })

        pd.DataFrame(ci_rows).to_csv(
            f'{OUT_PREFIX}_all_metric_confidence_intervals{FILE_SUFFIX}.csv', index=False)
        pd.DataFrame(model_comp_rows).to_csv(
            f'{OUT_PREFIX}_model_vs_model{FILE_SUFFIX}.csv', index=False)
        pd.DataFrame(block_comp_rows).to_csv(
            f'{OUT_PREFIX}_auc_brier_block_comparisons{FILE_SUFFIX}.csv', index=False)

    print(f"  Bootstrap total: {time.perf_counter() - boot_t0:.1f}s")

    # =========================================================================
    # Per-block importance / selection CSVs (per outcome)
    # =========================================================================
    for outcome_col in OUTCOMES:
        OUT_PREFIX = outcome_prefix(outcome_col, split_idx)
        for block_name in BLOCKS:
            key = (outcome_col, block_name)
            if key in ALL_IMPORTANCE:
                ALL_IMPORTANCE[key].to_csv(
                    f'{OUT_PREFIX}_importance_{block_name}{FILE_SUFFIX}.csv', index=False)
            if key in ALL_HYBRID_COMP:
                ALL_HYBRID_COMP[key].to_csv(
                    f'{OUT_PREFIX}_hybrid_selection_{block_name}{FILE_SUFFIX}.csv', index=False)

    # --- Summary tables: one per outcome, plus a combined run-level copy ---
    all_rows = []
    for outcome_col, outcome_label in OUTCOMES.items():
        outcome_rows = []
        for block_name in BLOCKS:
            key = (outcome_col, block_name)
            if key not in ALL_RESULTS:
                continue
            sample_sizes = ALL_SAMPLE_SIZES[key]
            base_row = {
                'Split': split_idx, 'Outcome': outcome_label,
                'Block': BLOCK_DISPLAY_NAMES[block_name],
                'N_train': sample_sizes['n_train'], 'N_test': sample_sizes['n_test'],
                'Events_train': sample_sizes['events_train'],
                'Events_test': sample_sizes['events_test'],
            }
            for mt in MODEL_TYPES:
                m = ALL_RESULTS[key][mt]
                outcome_rows.append({**base_row, 'Model': mt, 'N_predictors': np.nan,
                                     'Test_AUC': round(m['auc'], 4),
                                     'Average_precision': round(m['average_precision'], 4),
                                     'Brier': round(m['brier'], 4),
                                     'O/E': round(m['oe_ratio'], 3),
                                     'Cal_slope': round(m['cal_slope'], 3),
                                     'Cal_intercept': round(m['cal_intercept'], 3)})
            h = ALL_HYBRID[key]
            outcome_rows.append({**base_row, 'Model': 'HYBRID',
                                 'N_predictors': ALL_HYBRID_BEST_N[key],
                                 'Test_AUC': round(h['auc'], 4),
                                 'Average_precision': round(h['average_precision'], 4),
                                 'Brier': round(h['brier'], 4),
                                 'O/E': round(h['oe_ratio'], 3),
                                 'Cal_slope': round(h['cal_slope'], 3),
                                 'Cal_intercept': round(h['cal_intercept'], 3)})
        if outcome_rows:
            pd.DataFrame(outcome_rows).to_csv(
                f'{outcome_prefix(outcome_col, split_idx)}_model_summary{FILE_SUFFIX}.csv',
                index=False)
            all_rows.extend(outcome_rows)

    split_summary = pd.DataFrame(all_rows)
    split_summary.to_csv(f'{RUN_PREFIX}_model_summary_ALL{FILE_SUFFIX}.csv', index=False)
    all_split_summaries.append(split_summary)

    # --- Save lightweight split index; fitted models are in per-block files ---
    output = {
        'script_version': SCRIPT_VERSION,
        'run_configuration': run_configuration,
        'sex_filter': SEX_FILTER,
        'db_filter': DB_FILTER,
        'tuning_metric': TUNING_METRIC,
        'sensitivity': args.sensitivity,
        'hybrid_capture': args.hybrid_capture,
        'split_info': split_info,
        'outcomes': OUTCOMES,
        'sample_sizes': ALL_SAMPLE_SIZES,
        'hybrid_best_n': ALL_HYBRID_BEST_N,
        'blocks': {k: {kk: vv for kk, vv in v.items() if kk != 'label'}
                   for k, v in BLOCKS.items()},
        'block_labels': {k: v['label'] for k, v in BLOCKS.items()},
        'block_pickle_files': BLOCK_PICKLE_FILES,
    }
    for outcome_col in OUTCOMES:
        for block_name in BLOCKS:
            key = (outcome_col, block_name)
            if key not in ALL_RESULTS:
                continue
            key_str = f"{outcome_col}__{block_name}"
            output[f'results_{key_str}'] = ALL_RESULTS[key]
            output[f'hybrid_result_{key_str}'] = ALL_HYBRID[key]
            output[f'hybrid_vars_{key_str}'] = ALL_HYBRID_VARS[key]
            output[f'importance_{key_str}'] = ALL_IMPORTANCE[key]

    with open(f'{RUN_PREFIX}_all_models{FILE_SUFFIX}.pkl', 'wb') as handle:
        pickle.dump(output, handle)

    print(f"  SPLIT {split_idx + 1} total: {time.perf_counter() - split_t0:.1f}s")

# =============================================================================
# AGGREGATE ACROSS SPLITS
# =============================================================================

if args.n_splits > 1:
    combined = pd.concat(all_split_summaries, ignore_index=True)
    agg = combined.groupby(['Outcome', 'Block', 'Model']).agg(
        AUC_mean=('Test_AUC', 'mean'), AUC_std=('Test_AUC', 'std'),
        AP_mean=('Average_precision', 'mean'), AP_std=('Average_precision', 'std'),
        Brier_mean=('Brier', 'mean'), Brier_std=('Brier', 'std'),
        CalSlope_mean=('Cal_slope', 'mean')).reset_index()
    agg_file = os.path.join(RUN_LEVEL_DIR,
                            f'{SENSITIVITY_TAG}{FILE_STEM}_{RUN_SCOPE}_aggregated_across_splits'
                            f'{FILE_SUFFIX}.csv')
    agg.to_csv(agg_file, index=False)
    print(f"\nAggregated {args.n_splits} splits -> {agg_file}")

print("\n" + "=" * 60)
print("RUN COMPLETE")
print("=" * 60)
