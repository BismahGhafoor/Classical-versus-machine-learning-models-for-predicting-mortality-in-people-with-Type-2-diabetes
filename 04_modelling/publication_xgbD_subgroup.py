#!/usr/bin/env python3
"""
Publication-focused subgroup analysis for outcome-scoped CPRD mortality pickles.

Primary analysis:
  - saved combined GOLD+Aurum predictions
  - XGB, Block D
  - IMD quintile and ethnicity subgroups
  - practice-cluster bootstrap confidence intervals
  - no model retraining

The script mirrors mortality_pipeline_v3.2_manuscript when rebuilding the
cohort and practice split. It verifies row/prediction alignment against the
pipeline's saved sample sizes and by-database metrics before writing results.
"""

import argparse
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="IMD/ethnicity subgroup performance for saved outcome-scoped predictions."
    )
    parser.add_argument("--model_pkl", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--diabetes_type", type=int, default=None, choices=[1, 2])
    parser.add_argument("--subgroup_cols", default="imd_quintile,gen_ethnicity")
    parser.add_argument("--model", default="XGB", choices=["LR", "LR_FLEX", "RF", "XGB", "HYBRID"])
    parser.add_argument("--block", default="D", choices=["A", "B", "C", "D"])
    parser.add_argument("--outcomes", default="all")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--study_end_date", default=None)
    parser.add_argument("--n_bootstraps", type=int, default=2000)
    parser.add_argument("--bootstrap_unit", choices=["practice", "patient"], default="practice")
    parser.add_argument("--min_n", type=int, default=500)
    parser.add_argument("--min_events", type=int, default=20)
    parser.add_argument("--include_missing_subgroup", action="store_true")
    parser.add_argument("--missing_label", default="Missing/Unknown")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--verify_tolerance", type=float, default=1e-6)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def parse_mixed_dates(series):
    series = series.astype("string")
    has_slash = series.str.contains("/", na=False)
    output = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if has_slash.any():
        output.loc[has_slash] = pd.to_datetime(
            series.loc[has_slash], errors="coerce", dayfirst=True
        )
    if (~has_slash).any():
        output.loc[~has_slash] = pd.to_datetime(
            series.loc[~has_slash], errors="coerce"
        )
    return output


def load_and_filter_data(data_path, db_filter, sex_filter, diabetes_type):
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

    required = {"database", "pracid", "gender", "diabetes_type", "indexdate", "yob"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    if not ({"dod_ons", "dod"} & set(df.columns)):
        raise ValueError("Input data must contain dod_ons or dod.")

    df["database"] = df["database"].astype("string").str.strip().str.upper()
    if df["database"].isna().any():
        raise ValueError("The database column contains missing values.")
    invalid = sorted(set(df["database"].dropna()) - {"GOLD", "AURUM"})
    if invalid:
        raise ValueError(f"Unexpected database values: {invalid}")
    if df["pracid"].isna().any():
        raise ValueError("The pracid column contains missing values.")

    db_filter = str(db_filter).strip().lower()
    if db_filter == "gold":
        df = df[df["database"] == "GOLD"].copy()
    elif db_filter == "aurum":
        df = df[df["database"] == "AURUM"].copy()
    elif db_filter != "all":
        raise ValueError(f"Unknown db_filter: {db_filter}")

    sex_filter = str(sex_filter).strip().lower()
    if sex_filter != "all":
        sex_mapping = {
            "male": {"M", "MALE", "1"},
            "female": {"F", "FEMALE", "2"},
        }
        if sex_filter not in sex_mapping:
            raise ValueError(f"Unknown sex_filter: {sex_filter}")
        df["gender"] = df["gender"].astype(str).str.strip()
        df = df[
            df["gender"].str.upper().isin(sex_mapping[sex_filter])
        ].copy()

    df["diabetes_type"] = pd.to_numeric(df["diabetes_type"], errors="coerce")
    df = df[df["diabetes_type"] == diabetes_type].copy()
    if df.empty:
        raise ValueError("No rows remain after database/sex/diabetes filtering.")

    print(f"Filtered cohort before outcome derivation: {len(df):,}")
    print(df["database"].value_counts(dropna=False).to_string())
    return df


def derive_mortality_outcomes(df, years=10, study_end_date="2021-03-31"):
    """Exact mortality-label logic used by mortality_pipeline_v3.2_manuscript."""
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

    censor_raw = (
        pd.concat(candidates, axis=1).min(axis=1)
        if candidates
        else pd.Series(pd.NaT, index=df.index)
    )
    censor_raw = censor_raw.where(censor_raw >= df["indexdate"], pd.NaT)
    df["censor_date_derived"] = censor_raw.fillna(study_end).clip(upper=study_end)

    dod_col = "dod_ons" if "dod_ons" in df.columns else "dod"
    dod_clean = pd.to_datetime(df[dod_col], errors="coerce")
    dod_clean = dod_clean.where(dod_clean >= df["indexdate"], pd.NaT)

    df["died_within_followup"] = (
        dod_clean.notna() & (dod_clean <= df["cutoff_date"])
    )
    df["complete_followup"] = (
        (df["censor_date_derived"] >= df["cutoff_date"])
        | df["died_within_followup"]
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
    for outcome_col in ["death_10y", "death_cvd", "death_cancer"]:
        if outcome_col in df.columns and df[outcome_col].notna().all():
            print(
                f"  {outcome_col}: {int(df[outcome_col].sum()):,} "
                f"({df[outcome_col].mean() * 100:.2f}%)"
            )
    return df


def prepare_subgroup_columns(df):
    """Match the pipeline's age/IMD preparation without changing row order."""
    df = df.copy()
    df["age_at_index"] = (
        df["indexdate"].dt.year - pd.to_numeric(df["yob"], errors="coerce")
    )
    imd = pd.to_numeric(df["imd_quintile"], errors="coerce")
    imd = imd.where(imd.between(1, 5), np.nan)
    df["imd_quintile"] = imd.astype("Int64").astype("string")
    return df


def split_by_practice(df, test_size=0.2, random_state=42, stratify_by_database=False):
    """Verbatim practice split used by the modelling pipeline."""
    if stratify_by_database and df["database"].nunique() > 1:
        train_dfs, test_dfs = [], []
        train_prac_all, test_prac_all = [], []
        for db in df["database"].unique():
            db_df = df[df["database"] == db]
            practices = db_df["pracid"].unique()
            if len(practices) < 5:
                train_dfs.append(db_df)
                train_prac_all.extend(practices)
                continue
            train_prac, test_prac = train_test_split(
                practices,
                test_size=test_size,
                random_state=random_state,
            )
            train_dfs.append(db_df[db_df["pracid"].isin(train_prac)])
            test_dfs.append(db_df[db_df["pracid"].isin(test_prac)])
            train_prac_all.extend(train_prac)
            test_prac_all.extend(test_prac)
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = (
            pd.concat(test_dfs, ignore_index=True)
            if test_dfs
            else pd.DataFrame()
        )
    else:
        practices = df["pracid"].unique()
        train_prac_all, test_prac_all = train_test_split(
            practices,
            test_size=test_size,
            random_state=random_state,
        )
        train_df = df[df["pracid"].isin(train_prac_all)].copy()
        test_df = df[df["pracid"].isin(test_prac_all)].copy()

    return train_df, test_df, {
        "train_practices": list(train_prac_all),
        "test_practices": list(test_prac_all),
    }


def rebuild_test_set(df_eligible, db_filter, split_seed, split_info):
    if "test_practices" not in split_info:
        raise ValueError("split_info does not contain test_practices.")

    print("\n" + "=" * 70)
    print("REPRODUCING PRACTICE SPLIT")
    print("=" * 70)

    stratify = str(db_filter).strip().lower() == "all"
    _, test_df, rebuilt = split_by_practice(
        df_eligible,
        test_size=0.2,
        random_state=split_seed,
        stratify_by_database=stratify,
    )

    saved = sorted(map(str, split_info["test_practices"]))
    rerun = sorted(map(str, rebuilt["test_practices"]))
    if saved != rerun:
        only_saved = len(set(saved) - set(rerun))
        only_rerun = len(set(rerun) - set(saved))
        raise ValueError(
            "Reconstructed practice split does not match the saved split "
            f"({len(rerun)} rebuilt vs {len(saved)} saved; "
            f"{only_saved} saved-only and {only_rerun} rebuilt-only)."
        )

    test_df = test_df.reset_index(drop=True)
    print(f"Split reproduced. Test practices: {len(rerun):,}")
    print(f"Base test cohort: {len(test_df):,}")
    print(test_df["database"].value_counts(dropna=False).to_string())
    return test_df


def extract_prediction(model_output, outcome_col, block, model):
    key_str = f"{outcome_col}__{block}"
    if model == "HYBRID":
        result = model_output.get(f"hybrid_result_{key_str}", {})
    else:
        result = model_output.get(f"results_{key_str}", {}).get(model, {})

    if "y_pred_proba" not in result:
        raise KeyError(
            f"Missing saved {model} predictions for {outcome_col}, Block {block}."
        )
    return np.asarray(result["y_pred_proba"], dtype=float)


def safe_auc(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_pred)


def safe_average_precision(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_pred)


def oe_ratio(y_true, y_pred):
    expected = float(np.sum(y_pred))
    return float(np.sum(y_true) / expected) if expected > 0 else np.nan


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "events": int(np.sum(y_true)),
        "event_rate": float(np.mean(y_true)),
        "mean_predicted_risk": float(np.mean(y_pred)),
        "auc": safe_auc(y_true, y_pred),
        "average_precision": safe_average_precision(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_pred),
        "oe_ratio": oe_ratio(y_true, y_pred),
    }


def verify_sample_size(model_output, outcome_col, block, test_df):
    key = (outcome_col, block)
    sample_sizes = model_output.get("sample_sizes", {})
    if key not in sample_sizes:
        raise ValueError(
            f"Pickle contains no sample-size record for {outcome_col}, Block {block}."
        )

    expected_n = int(sample_sizes[key]["n_test"])
    expected_events = int(sample_sizes[key]["events_test"])
    actual_n = len(test_df)
    actual_events = int(test_df[outcome_col].sum())

    if actual_n != expected_n:
        raise ValueError(
            f"Reconstructed test N={actual_n:,}, but pipeline recorded "
            f"N={expected_n:,} for {outcome_col}, Block {block}."
        )
    if actual_events != expected_events:
        raise ValueError(
            f"Reconstructed test events={actual_events:,}, but pipeline recorded "
            f"{expected_events:,} for {outcome_col}, Block {block}."
        )

    print(
        f"Test sample verified: N={actual_n:,}, events={actual_events:,} "
        "[matches pipeline]"
    )


def verify_by_database(
    model_output,
    outcome_col,
    block,
    model,
    test_df,
    y_pred,
    tolerance,
):
    key_str = f"{outcome_col}__{block}"
    if model == "HYBRID":
        stored = model_output.get(
            f"hybrid_result_{key_str}", {}
        ).get("by_database", {})
    else:
        stored = model_output.get(
            f"results_{key_str}", {}
        ).get(model, {}).get("by_database", {})

    rows = []
    for db in sorted(test_df["database"].dropna().unique()):
        mask = test_df["database"].to_numpy() == db
        core = calculate_metrics(
            test_df.loc[mask, outcome_col].to_numpy(dtype=int),
            y_pred[mask],
        )
        ref = stored.get(db)
        status = "not-stored"

        if ref is not None:
            for field in (
                "n",
                "events",
                "auc",
                "average_precision",
                "brier",
                "oe_ratio",
            ):
                if field not in ref:
                    continue
                actual = float(core[field])
                expected = float(ref[field])
                if np.isnan(actual) and np.isnan(expected):
                    continue
                if np.isnan(actual) != np.isnan(expected):
                    raise ValueError(
                        f"MISALIGNED {outcome_col} | Block {block} | {model} | "
                        f"{db}: {field} reconstructed={actual!r}, stored={expected!r}."
                    )
                if abs(actual - expected) > tolerance * max(1.0, abs(expected)):
                    raise ValueError(
                        f"MISALIGNED {outcome_col} | Block {block} | {model} | "
                        f"{db}: {field} reconstructed={actual!r}, stored={expected!r}."
                    )
            status = "OK"

        rows.append(
            {
                "Outcome_col": outcome_col,
                "Block": block,
                "Model": model,
                "Database": db,
                "N": core["n"],
                "Events": core["events"],
                "Alignment_check": status,
            }
        )
        print(
            f"  Alignment | {db}: N={core['n']:,}, events={core['events']:,}, "
            f"AUC={core['auc']:.4f}, Brier={core['brier']:.4f} [{status}]"
        )

    if not rows or not any(row["Alignment_check"] == "OK" for row in rows):
        raise ValueError(
            "No database stratum could be verified against pipeline-saved metrics."
        )
    return pd.DataFrame(rows)


def clean_subgroup_values(df, subgroup_col, include_missing, missing_label):
    if subgroup_col not in df.columns:
        raise ValueError(f"Subgroup column not found: {subgroup_col}")

    series = df[subgroup_col]
    col = subgroup_col.lower()

    if col in {"imd", "imdq", "imd_quintile", "imd_quint"}:
        numeric = pd.to_numeric(series, errors="coerce")
        numeric = numeric.where(numeric.between(1, 5), np.nan)
        labels = numeric.map(
            lambda value: (
                f"IMD quintile {int(value)}"
                if pd.notna(value)
                else np.nan
            )
        ).astype("object")
    else:
        labels = series.astype("string").str.strip()
        labels = labels.replace(
            {
                "": np.nan,
                "nan": np.nan,
                "NaN": np.nan,
                "None": np.nan,
                "NONE": np.nan,
                "<NA>": np.nan,
            }
        ).astype("object")

    labels = pd.Series(labels, index=df.index, dtype="object")
    labels = labels.where(pd.notna(labels), np.nan)
    if include_missing:
        labels = labels.where(labels.notna(), missing_label)
    return labels


def subgroup_sort_key(value):
    text = str(value)
    match = re.search(r"(\d+)", text)
    if text.startswith("IMD quintile") and match:
        return (0, int(match.group(1)))
    if text.lower().startswith("missing"):
        return (9, text)
    return (1, text)


def bootstrap_indices(n, cluster_ids, n_bootstraps, unit, random_state):
    rng = np.random.default_rng(random_state)

    if unit == "patient":
        for _ in range(n_bootstraps):
            yield rng.integers(0, n, size=n)
        return

    cluster_ids = np.asarray(cluster_ids)
    unique_clusters = np.unique(cluster_ids)
    if len(unique_clusters) < 2:
        raise ValueError("At least two practices are required for cluster bootstrap.")
    cluster_to_indices = {
        cluster: np.where(cluster_ids == cluster)[0]
        for cluster in unique_clusters
    }
    for _ in range(n_bootstraps):
        sampled = rng.choice(
            unique_clusters,
            size=len(unique_clusters),
            replace=True,
        )
        yield np.concatenate(
            [cluster_to_indices[cluster] for cluster in sampled]
        )


def bootstrap_metric_cis(
    y_true,
    y_pred,
    cluster_ids,
    n_bootstraps,
    unit,
    random_state,
):
    metric_names = [
        "event_rate",
        "mean_predicted_risk",
        "auc",
        "average_precision",
        "brier",
        "oe_ratio",
    ]
    values = {metric: [] for metric in metric_names}
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred, dtype=float)

    total_samples = 0
    two_class_samples = 0

    for idx in bootstrap_indices(
        len(y_true),
        cluster_ids,
        n_bootstraps,
        unit,
        random_state,
    ):
        y_boot = y_true[idx]
        pred_boot = y_pred[idx]
        total_samples += 1

        # Always-defined metrics are retained even when a resample has one class.
        values["event_rate"].append(float(np.mean(y_boot)))
        values["mean_predicted_risk"].append(float(np.mean(pred_boot)))
        values["brier"].append(float(brier_score_loss(y_boot, pred_boot)))
        oe = oe_ratio(y_boot, pred_boot)
        if np.isfinite(oe):
            values["oe_ratio"].append(oe)

        # Discrimination metrics require both outcome classes.
        if len(np.unique(y_boot)) >= 2:
            two_class_samples += 1
            values["auc"].append(float(roc_auc_score(y_boot, pred_boot)))
            values["average_precision"].append(
                float(average_precision_score(y_boot, pred_boot))
            )

    output = {
        "valid_bootstrap_samples": int(total_samples),
        "two_class_bootstrap_samples": int(two_class_samples),
    }
    for metric, metric_values in values.items():
        array = np.asarray(metric_values, dtype=float)
        output[f"{metric}_bootstrap_n"] = int(len(array))
        if len(array) >= 10:
            output[f"{metric}_lower"] = float(np.percentile(array, 2.5))
            output[f"{metric}_upper"] = float(np.percentile(array, 97.5))
        else:
            output[f"{metric}_lower"] = np.nan
            output[f"{metric}_upper"] = np.nan
    return output


def format_ci(point, lower, upper, digits):
    if point is None or not np.isfinite(point):
        return ""
    if np.isfinite(lower) and np.isfinite(upper):
        return f"{point:.{digits}f} ({lower:.{digits}f}, {upper:.{digits}f})"
    return f"{point:.{digits}f}"


def make_formatted_table(df):
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Sex": row["Sex"],
                "Subgroup variable": row["Subgroup_variable"],
                "Subgroup": row["Subgroup"],
                "Outcome": row["Outcome"],
                "Model": row["Model"],
                "Block": row["Block"],
                "N": int(row["n"]),
                "Events": int(row["events"]),
                "Event rate": format_ci(
                    row["event_rate"],
                    row.get("event_rate_lower", np.nan),
                    row.get("event_rate_upper", np.nan),
                    4,
                ),
                "Mean predicted risk": format_ci(
                    row["mean_predicted_risk"],
                    row.get("mean_predicted_risk_lower", np.nan),
                    row.get("mean_predicted_risk_upper", np.nan),
                    4,
                ),
                "AUC": format_ci(
                    row["auc"],
                    row.get("auc_lower", np.nan),
                    row.get("auc_upper", np.nan),
                    3,
                ),
                "PR-AUC": format_ci(
                    row["average_precision"],
                    row.get("average_precision_lower", np.nan),
                    row.get("average_precision_upper", np.nan),
                    3,
                ),
                "Brier": format_ci(
                    row["brier"],
                    row.get("brier_lower", np.nan),
                    row.get("brier_upper", np.nan),
                    4,
                ),
                "O/E ratio": format_ci(
                    row["oe_ratio"],
                    row.get("oe_ratio_lower", np.nan),
                    row.get("oe_ratio_upper", np.nan),
                    3,
                ),
                "Unique practices": int(row["unique_practices"]),
                "Valid bootstrap samples": int(
                    row.get("valid_bootstrap_samples", 0)
                ),
                "Two-class bootstrap samples": int(
                    row.get("two_class_bootstrap_samples", 0)
                ),
                "Note": row.get("Note", ""),
            }
        )
    return pd.DataFrame(rows)


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def main():
    args = parse_args()
    model_pkl = Path(args.model_pkl)

    with open(model_pkl, "rb") as handle:
        model_output = pickle.load(handle)

    cfg = model_output.get("run_configuration", {})
    split_info = model_output.get("split_info", {})
    if "test_practices" not in split_info:
        raise ValueError("split_info does not contain test_practices.")

    sensitivity = model_output.get(
        "sensitivity",
        cfg.get("sensitivity", "none"),
    )
    if sensitivity != "none":
        raise ValueError(
            "This publication subgroup script is for the primary imputed "
            f"analysis only, but the pickle sensitivity is {sensitivity!r}."
        )

    db_filter = str(
        model_output.get("db_filter", cfg.get("database", "all"))
    ).strip().lower()
    sex_filter = str(
        model_output.get("sex_filter", cfg.get("sex", "all"))
    ).strip().lower()
    sex_label = sex_filter.upper()

    diabetes_type = (
        args.diabetes_type
        if args.diabetes_type is not None
        else cfg.get("diabetes_type")
    )
    study_end_date = args.study_end_date or cfg.get("study_end_date")
    data_path = args.data_path or cfg.get("data_path")
    for name, value in (
        ("diabetes_type", diabetes_type),
        ("study_end_date", study_end_date),
        ("data_path", data_path),
    ):
        if value is None:
            raise ValueError(
                f"{name} is absent from run_configuration; pass --{name}."
            )

    split_index = int(cfg.get("split_index", 0))
    split_seed = int(cfg.get("split_seed", 42 + split_index))

    subgroup_cols = [
        value.strip()
        for value in args.subgroup_cols.split(",")
        if value.strip()
    ]
    requested_outcomes = [
        value.strip()
        for value in args.outcomes.split(",")
        if value.strip()
    ]

    outcomes = dict(model_output.get("outcomes", {}))
    if requested_outcomes != ["all"]:
        outcomes = {
            key: value
            for key, value in outcomes.items()
            if key in requested_outcomes
        }
    if not outcomes:
        raise ValueError("No requested outcomes are present in this pickle.")
    if len(outcomes) != 1:
        raise ValueError(
            "Use one outcome-scoped run-level pickle per invocation."
        )

    outcome_col, outcome_label = next(iter(outcomes.items()))

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            model_pkl.parent
            / f"postprocessing_subgroups_{safe_name(args.model)}_block_{safe_name(args.block)}"
            / safe_name(outcome_col)
            / safe_name(model_pkl.stem)
        )
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("PUBLICATION SUBGROUP ANALYSIS")
    print("=" * 70)
    print(f"Model pickle:       {model_pkl}")
    print(f"Pipeline version:   {model_output.get('script_version', 'unknown')}")
    print(f"Sex/database:       {sex_filter} / {db_filter}")
    print(f"Outcome:            {outcome_col}")
    print(f"Model/block:        {args.model} / Block {args.block}")
    print(f"Subgroup columns:   {subgroup_cols}")
    print(f"Split index/seed:   {split_index} / {split_seed}")
    print(f"Bootstraps:         {args.n_bootstraps}")
    print(f"Bootstrap unit:     {args.bootstrap_unit}")
    print(f"Minimum N/events:   {args.min_n}/{args.min_events}")
    print(f"Output directory:   {output_dir}")

    df = load_and_filter_data(
        data_path,
        db_filter,
        sex_filter,
        diabetes_type,
    )
    df = derive_mortality_outcomes(
        df,
        years=args.years,
        study_end_date=study_end_date,
    )
    df_eligible = prepare_subgroup_columns(
        df[df["eligible"]].copy()
    )
    del df

    test_df = rebuild_test_set(
        df_eligible,
        db_filter,
        split_seed,
        split_info,
    )
    del df_eligible

    for subgroup_col in subgroup_cols:
        if subgroup_col not in test_df.columns:
            raise ValueError(
                f"Subgroup column {subgroup_col!r} is absent from the test set."
            )

    verify_sample_size(
        model_output,
        outcome_col,
        args.block,
        test_df,
    )

    y_pred = extract_prediction(
        model_output,
        outcome_col,
        args.block,
        args.model,
    )
    if len(y_pred) != len(test_df):
        raise ValueError(
            f"Length mismatch for {outcome_col} | {args.model} Block "
            f"{args.block}: test={len(test_df):,}, predictions={len(y_pred):,}."
        )

    alignment_df = verify_by_database(
        model_output,
        outcome_col,
        args.block,
        args.model,
        test_df,
        y_pred,
        args.verify_tolerance,
    )

    rows = []
    count_rows = []

    print("\n" + "=" * 70)
    print("CALCULATING SUBGROUP PERFORMANCE")
    print("=" * 70)

    for subgroup_col in subgroup_cols:
        subgroup_values = clean_subgroup_values(
            test_df,
            subgroup_col,
            include_missing=args.include_missing_subgroup,
            missing_label=args.missing_label,
        )
        valid_mask = (
            np.ones(len(test_df), dtype=bool)
            if args.include_missing_subgroup
            else subgroup_values.notna().to_numpy()
        )
        groups = sorted(
            [
                group
                for group in subgroup_values[valid_mask].dropna().unique()
                if pd.notna(group)
            ],
            key=subgroup_sort_key,
        )

        missing_n = int(subgroup_values.isna().sum())
        count_rows.append(
            {
                "Sex": sex_label,
                "Subgroup_variable": subgroup_col,
                "Subgroup": "__MISSING_NOT_ANALYSED__",
                "N": missing_n,
                "Percent_of_test": (
                    missing_n / len(test_df) * 100
                    if len(test_df)
                    else np.nan
                ),
                "Unique_practices": np.nan,
            }
        )

        for subgroup in groups:
            mask = (
                subgroup_values.eq(subgroup)
                .fillna(False)
                .to_numpy(dtype=bool)
                & valid_mask
            )
            composite_practice = (
                test_df.loc[mask, "database"].astype(str)
                + "::"
                + test_df.loc[mask, "pracid"].astype(str)
            ).to_numpy()

            count_rows.append(
                {
                    "Sex": sex_label,
                    "Subgroup_variable": subgroup_col,
                    "Subgroup": subgroup,
                    "N": int(mask.sum()),
                    "Percent_of_test": float(mask.mean() * 100),
                    "Unique_practices": int(
                        np.unique(composite_practice).size
                    ),
                }
            )

            y_true = test_df.loc[mask, outcome_col].to_numpy(dtype=int)
            subgroup_pred = y_pred[mask]
            n = int(len(y_true))
            events = int(np.sum(y_true))
            note = ""

            if n < args.min_n:
                note = f"Skipped: N < min_n ({n} < {args.min_n})"
            elif events < args.min_events:
                note = (
                    f"Skipped: events < min_events "
                    f"({events} < {args.min_events})"
                )
            elif len(np.unique(y_true)) < 2:
                note = "Skipped: only one outcome class present"
            elif (
                args.bootstrap_unit == "practice"
                and np.unique(composite_practice).size < 2
            ):
                note = "Skipped: fewer than two practices"

            base_row = {
                "Sex": sex_label,
                "Analysis_type": "imputed",
                "Subgroup_variable": subgroup_col,
                "Subgroup": subgroup,
                "Outcome": outcome_label,
                "Outcome_col": outcome_col,
                "Model": args.model,
                "Block": args.block,
                "unique_practices": int(
                    np.unique(composite_practice).size
                ),
                "Note": note,
            }

            if note:
                rows.append(
                    {
                        **base_row,
                        "n": n,
                        "events": events,
                        "event_rate": (
                            float(np.mean(y_true)) if n else np.nan
                        ),
                        "mean_predicted_risk": (
                            float(np.mean(subgroup_pred))
                            if n
                            else np.nan
                        ),
                        "auc": np.nan,
                        "average_precision": np.nan,
                        "brier": np.nan,
                        "oe_ratio": np.nan,
                        "valid_bootstrap_samples": 0,
                        "two_class_bootstrap_samples": 0,
                    }
                )
                print(
                    f"  SKIP | {subgroup_col} | {subgroup} | "
                    f"{outcome_col}: {note}"
                )
                continue

            core = calculate_metrics(y_true, subgroup_pred)
            ci = bootstrap_metric_cis(
                y_true=y_true,
                y_pred=subgroup_pred,
                cluster_ids=composite_practice,
                n_bootstraps=args.n_bootstraps,
                unit=args.bootstrap_unit,
                random_state=args.random_state,
            )
            rows.append({**base_row, **core, **ci})
            print(
                f"  {subgroup_col} | {subgroup} | {outcome_col}: "
                f"N={core['n']:,}, events={core['events']:,}, "
                f"AUC={core['auc']:.4f}, Brier={core['brier']:.4f}, "
                f"O/E={core['oe_ratio']:.3f}"
            )

    performance_df = pd.DataFrame(rows)
    formatted_df = make_formatted_table(performance_df)
    counts_df = pd.DataFrame(count_rows)

    prefix = (
        f"{safe_name(args.model)}_block_{safe_name(args.block)}_"
        f"{safe_name(outcome_col)}"
    )
    performance_path = (
        tables_dir / f"{prefix}_subgroup_performance_with_ci.csv"
    )
    formatted_path = (
        tables_dir / f"{prefix}_subgroup_performance_formatted.csv"
    )
    counts_path = tables_dir / f"{prefix}_subgroup_counts.csv"
    alignment_path = tables_dir / f"{prefix}_alignment_verification.csv"

    performance_df.to_csv(performance_path, index=False)
    formatted_df.to_csv(formatted_path, index=False)
    counts_df.to_csv(counts_path, index=False)
    alignment_df.to_csv(alignment_path, index=False)

    print("\n" + "=" * 70)
    print("SUBGROUP ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Tables saved to: {tables_dir}")
    for path in (
        performance_path,
        formatted_path,
        counts_path,
        alignment_path,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()

