#!/usr/bin/env python3
"""
Database-stratified post-processing for CPRD GOLD + Aurum mortality models.

Evaluates saved combined GOLD+Aurum model predictions separately within
CPRD GOLD and CPRD Aurum test patients. Does NOT retrain any model.

Works for main/imputed, complete-case and complete-followup pickles produced by
mortality_pipeline_v3.2_manuscript.

DESIGN NOTE
-----------
This script does not reimplement the cohort definition. It re-runs the
pipeline's own deterministic filtering, outcome derivation and practice split
so that the reconstructed test set is byte-for-byte the same rows in the same
ORDER as the one the saved predictions were generated on. Every step is then
asserted against values the pipeline recorded in the pickle:

  * test practice list      vs split_info['test_practices']
  * per-block N and events  vs output['sample_sizes']
  * per-database AUC/Brier  vs results[...][model]['by_database']

If any assertion fires, the reconstruction has drifted and no output is
written. Do not "improve" the mirrored functions below — they must stay
equivalent to the pipeline, not correct in isolation.

Main outputs:
  tables/database_stratified_performance_with_ci.csv
  tables/database_stratified_performance_formatted.csv
  tables/database_stratified_threshold_metrics_with_ci.csv
  tables/database_stratified_counts.csv
  tables/alignment_verification.csv

Suggested use:
  Smoke test: --n_bootstraps 10     (assertions all fire before bootstrapping)
  Final:      --n_bootstraps 2000
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
)
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


# NOTE: 'dod' is deliberately absent. The pipeline does not pre-parse it, so it
# is parsed month-first downstream; parsing it dayfirst here would flip dates.
DATE_COLS = [
    "indexdate", "dod_ons", "tod", "regenddate", "eventdate",
    "smoking_date", "bmi_date", "bp_date", "tot_chol_date", "hdl_date",
    "ldl_date", "trigly_date", "hba1c_date", "lcd", "censor_date",
    "comorb_ckd_first_date", "comorb_htn_first_date", "comorb_cvd_first_date",
    "comorb_cancer_any_first_date", "comorb_cancer_breast_first_date",
    "comorb_cancer_colorectal_first_date", "comorb_cancer_lung_first_date",
    "comorb_cancer_pancreatic_first_date", "comorb_cancer_prostate_first_date",
]

CAUSE_SPECIFIC_COLS = {
    "death_cvd": "cod_cvd",
    "death_cancer": "cod_cancer",
}

SENSITIVITY_TO_ANALYSIS_TYPE = {
    "none": "imputed",
    "complete_case": "complete_case",
    "complete_fu": "complete_fu",
}


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Database-stratified performance for saved CPRD GOLD+Aurum mortality models."
    )

    parser.add_argument(
        "--model_pkl", required=True,
        help=("Path to the run-level '*_all_models*.pkl' file. "
              "Do not pass a per-block '*_models_A/B/C/D*.pkl' file."),
    )

    # All of the following default to the values recorded in the pickle's
    # run_configuration. Override only if you know why you are doing so.
    parser.add_argument("--data_path", default=None,
                        help="Override. Default: run_configuration['data_path'].")
    parser.add_argument("--diabetes_type", type=int, default=None, choices=[1, 2],
                        help="Override. Default: run_configuration['diabetes_type'].")
    parser.add_argument("--study_end_date", default=None,
                        help="Override. Default: run_configuration['study_end_date'].")
    parser.add_argument("--split_idx", type=int, default=None,
                        help="Override. Default: run_configuration['split_index'].")
    parser.add_argument("--years", type=int, default=10)

    parser.add_argument("--n_bootstraps", type=int, default=1000)
    parser.add_argument("--bootstrap_unit", choices=["practice", "patient"], default="practice")
    parser.add_argument("--thresholds", default="0.01,0.05,0.10,0.20")
    parser.add_argument("--models", default="LR,RF,XGB,HYBRID")
    parser.add_argument("--blocks", default="A,B,C,D")
    parser.add_argument("--outcomes", default="all")
    parser.add_argument("--include_calibration_ci", action="store_true",
                        help="Also bootstrap calibration intercept/slope/in-the-large. Slower.")
    parser.add_argument("--verify_tolerance", type=float, default=1e-6,
                        help="Relative tolerance when checking against pipeline-stored metrics.")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output_dir", default=None)

    return parser.parse_args()


# =============================================================================
# MIRRORED PIPELINE FUNCTIONS
# Keep equivalent to mortality_pipeline_v3.2_manuscript. Do not "fix" these.
# =============================================================================

def parse_mixed_dates(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    has_slash = s.str.contains("/", na=False)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    if has_slash.any():
        out.loc[has_slash] = pd.to_datetime(s.loc[has_slash], errors="coerce", dayfirst=True)
    if (~has_slash).any():
        out.loc[~has_slash] = pd.to_datetime(s.loc[~has_slash], errors="coerce")

    return out


def load_and_filter_data(data_path, db_filter, sex_filter, diabetes_type):
    """Mirror of pipeline sections 1-2."""
    print("\n" + "=" * 70)
    print("RECONSTRUCTING FILTERED COHORT")
    print("=" * 70)
    print(f"Data path:      {data_path}")
    print(f"Database:       {str(db_filter).upper()}")
    print(f"Sex:            {str(sex_filter).upper()}")
    print(f"Diabetes type:  Type {diabetes_type}")

    df = pd.read_csv(data_path, sep="\t", low_memory=False)
    if df.empty:
        raise ValueError("The input dataset is empty.")

    for col in DATE_COLS:
        if col in df.columns:
            df[col] = parse_mixed_dates(df[col])

    df["database"] = df["database"].astype("string").str.strip().str.upper()
    if df["database"].isna().any():
        raise ValueError("The database column contains missing values.")
    invalid = sorted(set(df["database"].dropna()) - {"GOLD", "AURUM"})
    if invalid:
        raise ValueError(f"Unexpected database values: {invalid}")
    if df["pracid"].isna().any():
        raise ValueError("The pracid column contains missing values.")

    db_filter = str(db_filter).strip().lower()
    sex_filter = str(sex_filter).strip().lower()

    if db_filter == "gold":
        df = df[df["database"] == "GOLD"].copy()
    elif db_filter == "aurum":
        df = df[df["database"] == "AURUM"].copy()
    elif db_filter != "all":
        raise ValueError(f"Unknown db_filter: {db_filter}")

    if sex_filter != "all":
        df["gender"] = df["gender"].astype(str).str.strip()
        sex_mapping = {"male": {"M", "MALE", "1"}, "female": {"F", "FEMALE", "2"}}
        if sex_filter not in sex_mapping:
            raise ValueError(f"Unknown sex_filter: {sex_filter}")
        df = df[df["gender"].str.upper().isin(sex_mapping[sex_filter])].copy()

    df["diabetes_type"] = pd.to_numeric(df["diabetes_type"], errors="coerce")
    df = df[df["diabetes_type"] == diabetes_type].copy()
    if df.empty:
        raise ValueError("No rows remain after database/sex/diabetes-type filtering.")

    print(f"Filtered cohort before outcome derivation: {len(df):,}")
    print("Database counts:")
    print(df["database"].value_counts(dropna=False).to_string())

    return df


def derive_mortality_outcomes(df, years=10, study_end_date="2021-03-31"):
    """Exact mirror of derive_mortality_no_censoring in the modelling pipeline.

    In particular there is NO clamp of the death date at study_end before
    computing died_within_followup. Adding one would reclassify deaths recorded
    after study_end but within index+10y, and y_true would no longer match the
    labels the models were evaluated against.
    """
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

    censor_raw = (pd.concat(candidates, axis=1).min(axis=1)
                  if candidates else pd.Series(pd.NaT, index=df.index))
    censor_raw = censor_raw.where(censor_raw >= df["indexdate"], pd.NaT)
    df["censor_date_derived"] = censor_raw.fillna(study_end).clip(upper=study_end)

    dod_col = "dod_ons" if "dod_ons" in df.columns else "dod"
    if dod_col not in df.columns:
        raise ValueError("No death date column found. Expected dod_ons or dod.")
    dod_clean = pd.to_datetime(df[dod_col], errors="coerce")
    dod_clean = dod_clean.where(dod_clean >= df["indexdate"], pd.NaT)

    df["died_within_followup"] = dod_clean.notna() & (dod_clean <= df["cutoff_date"])
    df["complete_followup"] = (
        (df["censor_date_derived"] >= df["cutoff_date"]) | df["died_within_followup"]
    )
    df["death_10y"] = df["died_within_followup"].astype(int)

    for outcome_col, source_col in CAUSE_SPECIFIC_COLS.items():
        if source_col in df.columns:
            df[outcome_col] = (
                df["died_within_followup"]
                & (df[source_col].fillna(0).astype(int) == 1)
            ).astype(int)
        else:
            df[outcome_col] = np.nan

    df["eligible"] = True

    print(f"Cohort after indexdate filter: {len(df):,}")
    print(f"Complete {years}y follow-up: {df['complete_followup'].mean() * 100:.1f}%")
    for oc in ["death_10y", "death_cvd", "death_cancer"]:
        if oc in df.columns and df[oc].notna().all():
            print(f"  {oc}: {df[oc].sum():,.0f} ({df[oc].mean() * 100:.2f}%)")

    return df


def prepare_predictors(df):
    """Mirror of pipeline sections 4-5 (column preparation only).

    Block membership itself is read from the pickle, not redefined here.
    """
    df = df.copy()
    df["age_at_index"] = df["indexdate"].dt.year - pd.to_numeric(df["yob"], errors="coerce")

    med_binary_vars = sorted(c for c in df.columns
                             if c.startswith("med_") and c.endswith("_prescribed"))
    comorb_bin_vars = sorted(c for c in df.columns if c.endswith("_bin"))
    comorb_dur_vars = sorted(c for c in df.columns
                             if c.endswith("_duration_years") and "comorb" in c)

    for col in med_binary_vars + comorb_bin_vars:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in comorb_dur_vars:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # CRITICAL: matches the pipeline. Out-of-range IMD becomes NaN, and the
    # column is categorical. Both affect which rows the complete-case filter
    # drops, so a numeric/unchecked version here would break row alignment.
    imd = pd.to_numeric(df["imd_quintile"], errors="coerce")
    imd = imd.where(imd.between(1, 5), np.nan)
    df["imd_quintile"] = imd.astype("Int64").astype("string")

    informative_zero = set(med_binary_vars + comorb_bin_vars + comorb_dur_vars)
    block_d_exclusions = {
        "death_cvd": [c for c in comorb_bin_vars + comorb_dur_vars if "cvd" in c],
        "death_cancer": [c for c in comorb_bin_vars + comorb_dur_vars if "cancer" in c],
    }
    return df, informative_zero, block_d_exclusions


def get_block_vars_for_outcome(block_name, block_def, outcome_col, block_d_exclusions):
    """Mirror of the pipeline function of the same name."""
    cat_vars, num_vars = list(block_def["cat"]), list(block_def["num"])
    if block_name in ["C", "D"] and outcome_col in block_d_exclusions:
        exclude = set(block_d_exclusions[outcome_col])
        num_vars = [v for v in num_vars if v not in exclude]
        cat_vars = [v for v in cat_vars if v not in exclude]
    return cat_vars, num_vars


def split_by_practice(df, test_size=0.2, random_state=42, stratify_by_database=False):
    """Verbatim copy of the pipeline's split. Row order of the returned test
    frame is what the saved predictions are aligned to."""
    if stratify_by_database and df["database"].nunique() > 1:
        train_dfs, test_dfs, train_prac_all, test_prac_all = [], [], [], []
        for db in df["database"].unique():
            db_df = df[df["database"] == db]
            practices = db_df["pracid"].unique()
            if len(practices) < 5:
                train_dfs.append(db_df)
                train_prac_all.extend(practices)
                continue
            train_prac, test_prac = train_test_split(
                practices, test_size=test_size, random_state=random_state)
            train_dfs.append(db_df[db_df["pracid"].isin(train_prac)])
            test_dfs.append(db_df[db_df["pracid"].isin(test_prac)])
            train_prac_all.extend(train_prac)
            test_prac_all.extend(test_prac)
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    else:
        practices = df["pracid"].unique()
        train_prac_all, test_prac_all = train_test_split(
            practices, test_size=test_size, random_state=random_state)
        train_df = df[df["pracid"].isin(train_prac_all)].copy()
        test_df = df[df["pracid"].isin(test_prac_all)].copy()
    return train_df, test_df, {"train_practices": list(train_prac_all),
                               "test_practices": list(test_prac_all)}


# =============================================================================
# TEST SET RECONSTRUCTION
# =============================================================================

def rebuild_test_set(df_eligible, db_filter, split_seed, split_info):
    """Re-run the deterministic split to recover test_df in the ORIGINAL ROW ORDER.

    Do NOT substitute pracid.isin(test_practices): pracids are not unique across
    GOLD and Aurum, and the stratified branch concatenates database blocks in an
    order that source-file order does not reproduce.
    """
    print("\n" + "=" * 70)
    print("REPRODUCING PRACTICE SPLIT")
    print("=" * 70)

    stratify = (str(db_filter).strip().lower() == "all")
    _, test_df, rebuilt = split_by_practice(
        df_eligible, test_size=0.2, random_state=split_seed,
        stratify_by_database=stratify,
    )

    saved = sorted(map(str, split_info["test_practices"]))
    rerun = sorted(map(str, rebuilt["test_practices"]))
    if saved != rerun:
        only_saved = len(set(saved) - set(rerun))
        only_rerun = len(set(rerun) - set(saved))
        raise ValueError(
            "Reconstructed practice split does not match the saved one "
            f"({len(rerun)} rebuilt vs {len(saved)} saved test practices; "
            f"{only_saved} saved-only, {only_rerun} rebuilt-only). "
            "Check data_path, study_end_date, sensitivity, diabetes_type and split_idx."
        )

    test_df = test_df.reset_index(drop=True)
    print(f"Split reproduced. Test practices: {len(rerun):,}")
    print(f"Base test cohort: {len(test_df):,}")
    print("Base test database counts:")
    print(test_df["database"].value_counts(dropna=False).to_string())
    return test_df


def make_count_row(analysis_type, outcome_col, block_name, db, db_df):
    return {
        "Analysis_type": analysis_type,
        "Outcome_col": outcome_col,
        "Block": block_name,
        "Database": db,
        "N_test": len(db_df),
        "Unique_practices": db_df["pracid"].nunique(),
        "death_10y_events": int(db_df["death_10y"].sum()) if "death_10y" in db_df else np.nan,
        "death_cvd_events": int(db_df["death_cvd"].sum()) if "death_cvd" in db_df else np.nan,
        "death_cancer_events": int(db_df["death_cancer"].sum()) if "death_cancer" in db_df else np.nan,
    }


def make_test_sets(base_test, model_output, sensitivity, analysis_type,
                   informative_zero, block_d_exclusions):
    """Build one test frame per (outcome, block), asserting N and events."""
    print("\n" + "=" * 70)
    print("RECONSTRUCTING PER-BLOCK TEST SETS")
    print("=" * 70)

    blocks = model_output["blocks"]
    outcomes = model_output["outcomes"]
    sample_sizes = model_output.get("sample_sizes", {})
    if not sample_sizes:
        raise ValueError(
            "Pickle contains no 'sample_sizes'. Cannot verify the reconstruction; "
            "refusing to produce stratified results."
        )

    test_sets, count_rows = {}, []

    for outcome_col in outcomes:
        for block_name, block_def in blocks.items():
            key = (outcome_col, block_name)
            if key not in sample_sizes:
                continue  # pipeline skipped this block (too few events)

            cat_vars, num_vars = get_block_vars_for_outcome(
                block_name, block_def, outcome_col, block_d_exclusions)
            predictors = cat_vars + num_vars

            missing = [v for v in predictors if v not in base_test.columns]
            if missing:
                raise ValueError(f"Missing predictors for {key}: {missing}")

            if sensitivity == "complete_case":
                vars_to_check = [v for v in predictors if v not in informative_zero]
                tb = (base_test.dropna(subset=vars_to_check).copy()
                      if vars_to_check else base_test.copy())
            else:
                tb = base_test

            expected_n = sample_sizes[key]["n_test"]
            expected_e = sample_sizes[key]["events_test"]
            actual_e = int(tb[outcome_col].sum())

            if len(tb) != expected_n:
                raise ValueError(
                    f"{key}: reconstructed N={len(tb):,} but pipeline recorded "
                    f"N={expected_n:,}. Cohort or complete-case definition has drifted."
                )
            if actual_e != expected_e:
                raise ValueError(
                    f"{key}: reconstructed events={actual_e:,} but pipeline recorded "
                    f"{expected_e:,}. Outcome derivation has drifted."
                )

            tb = tb.reset_index(drop=True)
            test_sets[key] = tb
            print(f"  {outcome_col} | Block {block_name}: N={len(tb):,}, "
                  f"events={actual_e:,}  [matches pipeline]")

            for db in sorted(tb["database"].dropna().unique()):
                count_rows.append(make_count_row(
                    analysis_type, outcome_col, block_name, db, tb[tb["database"] == db]))

    if not test_sets:
        raise ValueError("No (outcome, block) combinations could be reconstructed.")

    return test_sets, pd.DataFrame(count_rows)


# =============================================================================
# PREDICTION EXTRACTION
# =============================================================================

def extract_predictions(model_output, requested_models, requested_blocks,
                        requested_outcomes, valid_keys):
    predictions = []
    outcomes = dict(model_output.get("outcomes", {}))
    blocks = model_output.get("blocks", {})

    if requested_outcomes != ["all"]:
        outcomes = {k: v for k, v in outcomes.items() if k in requested_outcomes}

    for outcome_col, outcome_label in outcomes.items():
        for block_name in blocks:
            if block_name not in requested_blocks:
                continue
            if (outcome_col, block_name) not in valid_keys:
                continue

            key_str = f"{outcome_col}__{block_name}"

            results_key = f"results_{key_str}"
            if results_key in model_output:
                result_dict = model_output[results_key]
                for model_type in ["LR", "LR_FLEX", "RF", "XGB"]:
                    if model_type not in requested_models:
                        continue
                    if model_type in result_dict and "y_pred_proba" in result_dict[model_type]:
                        predictions.append({
                            "outcome_col": outcome_col,
                            "outcome_label": outcome_label,
                            "block": block_name,
                            "model": model_type,
                            "y_pred": np.asarray(
                                result_dict[model_type]["y_pred_proba"], dtype=float),
                        })

            hybrid_key = f"hybrid_result_{key_str}"
            if "HYBRID" in requested_models and hybrid_key in model_output:
                if "y_pred_proba" in model_output[hybrid_key]:
                    predictions.append({
                        "outcome_col": outcome_col,
                        "outcome_label": outcome_label,
                        "block": block_name,
                        "model": "HYBRID",
                        "y_pred": np.asarray(
                            model_output[hybrid_key]["y_pred_proba"], dtype=float),
                    })

    print("\n" + "=" * 70)
    print("SAVED PREDICTIONS FOUND")
    print("=" * 70)
    print(f"Prediction arrays found: {len(predictions)}")

    if not predictions:
        raise ValueError("No saved predictions found in pickle for the requested selection.")

    pred_summary = pd.DataFrame([
        {
            "Outcome_col": p["outcome_col"],
            "Outcome": p["outcome_label"],
            "Block": p["block"],
            "Model": p["model"],
            "N_predictions": len(p["y_pred"]),
        }
        for p in predictions
    ])
    print(pred_summary.groupby(["Outcome_col", "Block"])["Model"].count().to_string())
    return predictions, pred_summary


# =============================================================================
# METRICS
# =============================================================================

def clip_probs(p, eps=1e-6):
    return np.clip(np.asarray(p, dtype=float), eps, 1 - eps)


def logit(p):
    p = clip_probs(p)
    return np.log(p / (1 - p))


def safe_auc(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_pred)


def safe_ap(y_true, y_pred):
    """Match sklearn/pipeline AP behaviour, including one-class strata."""
    y_true = np.asarray(y_true).astype(int)
    if len(y_true) == 0:
        return np.nan
    return average_precision_score(y_true, y_pred)


def oe_ratio(y_true, y_pred):
    mean_pred = np.mean(y_pred)
    return np.mean(y_true) / mean_pred if mean_pred > 0 else np.nan


def calibration_in_large(y_true, y_pred):
    event_rate = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    if event_rate <= 0 or event_rate >= 1 or mean_pred <= 0 or mean_pred >= 1:
        return np.nan
    return logit(event_rate) - logit(mean_pred)


def calibration_intercept_slope(y_true, y_pred):
    """Matches the pipeline's calibration_slope_intercept (C=1e6, lbfgs, 2000 iters)."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    x = logit(y_pred).reshape(-1, 1)
    try:
        model = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(x, y_true)
        return float(model.intercept_[0]), float(model.coef_[0][0])
    except Exception:
        return np.nan, np.nan


def calculate_metrics(y_true, y_pred, include_calibration=True):
    """Discrimination/calibration metrics.

    y_pred is used UNCLIPPED for AUC/AP/Brier/O-E so that these reproduce the
    pipeline's stored by_database values exactly. Clipping happens only inside
    the logit-based calibration terms, as it does in the pipeline.

    During bootstrap runs, calibration fitting can be skipped unless its CIs
    were explicitly requested. This avoids thousands of unnecessary logistic
    regression fits while leaving all point estimates unchanged.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred, dtype=float)

    if include_calibration:
        intercept, slope = calibration_intercept_slope(y_true, y_pred)
        cil = calibration_in_large(y_true, y_pred)
    else:
        intercept, slope, cil = np.nan, np.nan, np.nan

    return {
        "n": int(len(y_true)),
        "events": int(np.sum(y_true)),
        "event_rate": float(np.mean(y_true)),
        "mean_predicted_risk": float(np.mean(y_pred)),
        "auc": safe_auc(y_true, y_pred),
        "average_precision": safe_ap(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_pred),
        "oe_ratio": oe_ratio(y_true, y_pred),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_in_large": cil,
    }


def threshold_metrics(y_true, y_pred, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred, dtype=float)
    y_bin = (y_pred >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel()

    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        "f1": f1_score(y_true, y_bin, zero_division=0),
        "predicted_positive_rate": float(np.mean(y_bin)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# =============================================================================
# ALIGNMENT VERIFICATION
# =============================================================================

def verify_against_pipeline(model_output, outcome_col, block, model, db, core, tol):
    """Compare reconstructed stratified metrics to the pipeline's own by_database
    output. This is the definitive check that rows and predictions are aligned.

    Returns 'OK' or 'not-stored'. Raises on disagreement.
    """
    key_str = f"{outcome_col}__{block}"
    if model == "HYBRID":
        stored = model_output.get(f"hybrid_result_{key_str}", {}).get("by_database", {})
    else:
        stored = model_output.get(f"results_{key_str}", {}).get(model, {}).get("by_database", {})

    ref = stored.get(db)
    if ref is None:
        # Pipeline only writes by_database when >1 database is present, and
        # skips strata with <50 test rows.
        return "not-stored"

    for field in ("n", "events", "auc", "average_precision", "brier", "oe_ratio"):
        if field not in ref:
            continue
        a, b = float(core[field]), float(ref[field])
        if np.isnan(a) or np.isnan(b):
            if np.isnan(a) and np.isnan(b):
                continue
            raise ValueError(
                f"MISALIGNED {outcome_col} | Block {block} | {model} | {db}: "
                f"{field} reconstructed={a!r} vs pipeline={b!r}. "
                "One value is missing and the other is not — no results written."
            )
        if abs(a - b) > tol * max(1.0, abs(b)):
            raise ValueError(
                f"MISALIGNED {outcome_col} | Block {block} | {model} | {db}: "
                f"{field} reconstructed={a!r} vs pipeline={b!r}. "
                "Predictions are not aligned to the reconstructed rows — "
                "no results written."
            )
    return "OK"


# =============================================================================
# BOOTSTRAP
# =============================================================================

def bootstrap_indices(n, cluster_ids, n_bootstraps, unit, random_state):
    rng = np.random.default_rng(random_state)

    if unit == "patient":
        for _ in range(n_bootstraps):
            yield rng.integers(0, n, size=n)

    elif unit == "practice":
        if cluster_ids is None:
            raise ValueError("cluster_ids are required for practice-level bootstrap.")
        cluster_ids = np.asarray(cluster_ids)
        unique_clusters = np.unique(cluster_ids)
        if len(unique_clusters) < 2:
            raise ValueError("At least two practices are required for cluster bootstrapping.")
        cluster_to_indices = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
        for _ in range(n_bootstraps):
            sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            yield np.concatenate([cluster_to_indices[c] for c in sampled])

    else:
        raise ValueError(f"Unknown bootstrap unit: {unit}")


def _percentile_ci(values_dict, metric_names, valid):
    ci = {"valid_bootstrap_samples": int(valid)}
    for m in metric_names:
        arr = np.asarray(values_dict[m], dtype=float)
        if len(arr) >= 10:
            ci[f"{m}_lower"] = float(np.percentile(arr, 2.5))
            ci[f"{m}_upper"] = float(np.percentile(arr, 97.5))
        else:
            ci[f"{m}_lower"] = np.nan
            ci[f"{m}_upper"] = np.nan
        ci[f"{m}_bootstrap_n"] = int(len(arr))
    return ci


def bootstrap_metric_cis(y_true, y_pred, cluster_ids, n_bootstraps, unit,
                         random_state, include_calibration_ci=False):
    metrics_for_ci = ["event_rate", "mean_predicted_risk", "auc",
                      "average_precision", "brier", "oe_ratio"]
    if include_calibration_ci:
        metrics_for_ci += ["calibration_intercept", "calibration_slope", "calibration_in_large"]

    values = {m: [] for m in metrics_for_ci}
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = 0

    for idx in bootstrap_indices(len(y_true), cluster_ids, n_bootstraps, unit, random_state):
        y_b, p_b = y_true[idx], y_pred[idx]
        met = calculate_metrics(
            y_b, p_b, include_calibration=include_calibration_ci
        )
        for m in metrics_for_ci:
            if np.isfinite(met[m]):
                values[m].append(met[m])
        # Count every resample. Metric-specific counts are retained separately,
        # so one-class resamples can contribute to Brier/event-rate/AP/threshold
        # metrics while AUC and calibration are simply omitted for that resample.
        valid += 1

    return _percentile_ci(values, metrics_for_ci, valid)


def bootstrap_threshold_cis(y_true, y_pred, threshold, cluster_ids,
                            n_bootstraps, unit, random_state):
    metric_names = ["sensitivity", "specificity", "ppv", "npv", "f1",
                    "predicted_positive_rate"]
    values = {m: [] for m in metric_names}
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = 0

    for idx in bootstrap_indices(len(y_true), cluster_ids, n_bootstraps, unit, random_state):
        y_b, p_b = y_true[idx], y_pred[idx]
        tm = threshold_metrics(y_b, p_b, threshold)
        for m in metric_names:
            if np.isfinite(tm[m]):
                values[m].append(tm[m])
        valid += 1

    return _percentile_ci(values, metric_names, valid)


# =============================================================================
# FORMATTING
# =============================================================================

def format_ci(point, lower=None, upper=None, digits=3):
    if point is None or not np.isfinite(point):
        return ""
    if (lower is not None and upper is not None
            and np.isfinite(lower) and np.isfinite(upper)):
        return f"{point:.{digits}f} ({lower:.{digits}f}, {upper:.{digits}f})"
    return f"{point:.{digits}f}"


def make_formatted_table(perf_df):
    rows = []
    for _, r in perf_df.iterrows():
        def f(name, digits):
            return format_ci(r[name], r.get(f"{name}_lower", np.nan),
                             r.get(f"{name}_upper", np.nan), digits)
        rows.append({
            "Outcome": r["Outcome"],
            "Block": r["Block"],
            "Model": r["Model"],
            "Database": r["Database"],
            "N": int(r["n"]),
            "Events": int(r["events"]),
            "Event rate": f("event_rate", 4),
            "Mean predicted risk": f("mean_predicted_risk", 4),
            "AUC": f("auc", 3),
            "PR-AUC": f("average_precision", 3),
            "Brier": f("brier", 4),
            "O/E ratio": f("oe_ratio", 3),
            "Calibration intercept": f("calibration_intercept", 3),
            "Calibration slope": f("calibration_slope", 3),
            "Calibration-in-the-large": f("calibration_in_large", 3),
            "Alignment check": r.get("Alignment_check", ""),
        })
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    model_pkl = Path(args.model_pkl)

    requested_models = [x.strip().upper() for x in args.models.split(",") if x.strip()]
    requested_blocks = [x.strip().upper() for x in args.blocks.split(",") if x.strip()]
    requested_outcomes = [x.strip() for x in args.outcomes.split(",") if x.strip()]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    with open(model_pkl, "rb") as f:
        model_output = pickle.load(f)

    cfg = model_output.get("run_configuration", {})
    split_info = model_output.get("split_info", {})
    if "test_practices" not in split_info:
        raise ValueError("split_info does not contain test_practices.")

    sensitivity = model_output.get("sensitivity", cfg.get("sensitivity", "none"))
    if sensitivity not in SENSITIVITY_TO_ANALYSIS_TYPE:
        raise ValueError(f"Unrecognised sensitivity value in pickle: {sensitivity!r}")
    analysis_type = SENSITIVITY_TO_ANALYSIS_TYPE[sensitivity]

    db_filter = str(model_output.get("db_filter", cfg.get("database", "all"))).strip().lower()
    sex_filter = str(model_output.get("sex_filter", cfg.get("sex", "all"))).strip().lower()

    diabetes_type = args.diabetes_type if args.diabetes_type is not None else cfg.get("diabetes_type")
    study_end_date = args.study_end_date or cfg.get("study_end_date")
    data_path = args.data_path or cfg.get("data_path")
    for name, value in [("diabetes_type", diabetes_type),
                        ("study_end_date", study_end_date),
                        ("data_path", data_path)]:
        if value is None:
            raise ValueError(
                f"{name} is not recorded in the pickle's run_configuration. "
                f"Pass --{name} explicitly."
            )
    data_path = Path(data_path)

    split_idx = args.split_idx if args.split_idx is not None else cfg.get("split_index", 0)
    split_seed = cfg.get("split_seed", 42 + split_idx)

    if args.output_dir is None:
        # The modelling pipeline writes separate outcome/configuration/split
        # pickles into the same _run_level directory. Use the pickle stem as a
        # unique subdirectory so one post-processing run cannot overwrite another.
        output_dir = (
            model_pkl.parent
            / "postprocessing_database_stratified"
            / model_pkl.stem
        )
    else:
        output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("DATABASE-STRATIFIED POST-PROCESSING")
    print("=" * 70)
    print(f"Model pickle:     {model_pkl}")
    print(f"Pipeline version: {model_output.get('script_version', 'unknown')}")
    print(f"Data path:        {data_path}")
    print(f"Sensitivity:      {sensitivity}  ->  analysis_type={analysis_type}")
    print(f"Database filter:  {db_filter} | Sex filter: {sex_filter} | DM type: {diabetes_type}")
    print(f"Study end date:   {study_end_date}")
    print(f"Split index/seed: {split_idx} / {split_seed}")
    print(f"Bootstraps:       {args.n_bootstraps} ({args.bootstrap_unit}-level)")
    print(f"Models:           {requested_models}")
    print(f"Blocks:           {requested_blocks}")
    print(f"Outcomes:         {requested_outcomes}")
    print(f"Calibration CIs:  {args.include_calibration_ci}")
    print(f"Output directory: {output_dir}")

    if db_filter != "all":
        print("\nWARNING: this pickle was produced from a single-database run "
              f"({db_filter.upper()}). Stratification is trivial and the pipeline "
              "stored no by_database values to verify against.")

    # --- Rebuild cohort ------------------------------------------------------
    df = load_and_filter_data(data_path, db_filter, sex_filter, diabetes_type)
    df = derive_mortality_outcomes(df, years=args.years, study_end_date=study_end_date)

    df_eligible = df[df["eligible"]].copy()
    if sensitivity == "complete_fu":
        print(f"\nApplying complete-followup restriction: "
              f"{len(df_eligible):,} -> {int(df_eligible['complete_followup'].sum()):,}")
        df_eligible = df_eligible[df_eligible["complete_followup"]].copy()

    df_eligible, informative_zero, block_d_exclusions = prepare_predictors(df_eligible)

    base_test = rebuild_test_set(df_eligible, db_filter, split_seed, split_info)
    del df, df_eligible

    test_sets, counts_df = make_test_sets(
        base_test, model_output, sensitivity, analysis_type,
        informative_zero, block_d_exclusions)
    counts_df.to_csv(tables_dir / "database_stratified_counts.csv", index=False)

    predictions, pred_summary = extract_predictions(
        model_output, requested_models, requested_blocks,
        requested_outcomes, set(test_sets))
    pred_summary.to_csv(tables_dir / "saved_prediction_arrays.csv", index=False)

    # --- Stratified evaluation ----------------------------------------------
    print("\n" + "=" * 70)
    print("CALCULATING DATABASE-STRATIFIED PERFORMANCE")
    print("=" * 70)

    perf_rows, threshold_rows, verify_rows = [], [], []
    n_ok, n_not_stored = 0, 0

    for p in predictions:
        outcome_col, outcome_label = p["outcome_col"], p["outcome_label"]
        block, model, y_pred_full = p["block"], p["model"], p["y_pred"]

        test_block = test_sets[(outcome_col, block)]
        if len(test_block) != len(y_pred_full):
            raise ValueError(
                f"Length mismatch for {outcome_col} | Block {block} | {model}: "
                f"test rows={len(test_block):,}, predictions={len(y_pred_full):,}."
            )

        cluster_ids = (test_block["database"].astype(str) + "::"
                       + test_block["pracid"].astype(str)).to_numpy()

        for db in sorted(test_block["database"].dropna().unique()):
            mask = (test_block["database"].values == db)
            y_true = test_block.loc[mask, outcome_col].values.astype(int)
            y_pred = y_pred_full[mask]
            db_clusters = cluster_ids[mask]
            if len(y_true) == 0:
                continue

            core = calculate_metrics(y_true, y_pred)

            status = verify_against_pipeline(
                model_output, outcome_col, block, model, db, core, args.verify_tolerance)
            n_ok += (status == "OK")
            n_not_stored += (status == "not-stored")
            verify_rows.append({
                "Outcome_col": outcome_col, "Block": block, "Model": model,
                "Database": db, "N": core["n"], "Alignment_check": status,
            })

            ci = bootstrap_metric_cis(
                y_true, y_pred, db_clusters, args.n_bootstraps,
                args.bootstrap_unit, args.random_state,
                include_calibration_ci=args.include_calibration_ci)

            perf_rows.append({
                "Analysis_type": analysis_type,
                "Outcome": outcome_label,
                "Outcome_col": outcome_col,
                "Block": block,
                "Model": model,
                "Database": db,
                "Alignment_check": status,
                **core, **ci,
            })

            print(f"  {outcome_col} | Block {block} | {model} | {db}: "
                  f"N={core['n']:,}, events={core['events']:,}, "
                  f"AUC={core['auc']:.4f}, Brier={core['brier']:.4f}, "
                  f"O/E={core['oe_ratio']:.3f}  [{status}]")

            for threshold in thresholds:
                tm = threshold_metrics(y_true, y_pred, threshold)
                tm_ci = bootstrap_threshold_cis(
                    y_true, y_pred, threshold, db_clusters,
                    args.n_bootstraps, args.bootstrap_unit, args.random_state)
                threshold_rows.append({
                    "Analysis_type": analysis_type,
                    "Outcome": outcome_label,
                    "Outcome_col": outcome_col,
                    "Block": block,
                    "Model": model,
                    "Database": db,
                    "Threshold": threshold,
                    **tm, **tm_ci,
                })

    perf_df = pd.DataFrame(perf_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    verify_df = pd.DataFrame(verify_rows)

    perf_path = tables_dir / "database_stratified_performance_with_ci.csv"
    formatted_path = tables_dir / "database_stratified_performance_formatted.csv"
    threshold_path = tables_dir / "database_stratified_threshold_metrics_with_ci.csv"
    verify_path = tables_dir / "alignment_verification.csv"

    perf_df.to_csv(perf_path, index=False)
    make_formatted_table(perf_df).to_csv(formatted_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    verify_df.to_csv(verify_path, index=False)

    print("\n" + "=" * 70)
    print("ALIGNMENT VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  Verified against pipeline by_database values: {n_ok}")
    print(f"  Not stored by pipeline (unverified):          {n_not_stored}")
    if n_not_stored:
        print("  NOTE: the pipeline omits by_database for strata with <50 test rows "
              "and for single-database runs. Unverified rows are flagged in "
              "alignment_verification.csv.")
    if n_ok == 0:
        print("  WARNING: nothing could be independently verified. Treat these "
              "results as provisional.")

    print("\n" + "=" * 70)
    print("DATABASE-STRATIFIED POST-PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Tables saved to: {tables_dir}")
    print("\nMain files:")
    for path in (perf_path, formatted_path, threshold_path, verify_path,
                 tables_dir / "database_stratified_counts.csv"):
        print(f"  {path}")
    print("\nNote: These are stratified evaluations of the same combined model "
          "predictions. No models were retrained.")


if __name__ == "__main__":
    main()
