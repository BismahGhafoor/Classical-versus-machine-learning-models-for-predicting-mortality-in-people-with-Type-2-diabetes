import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/scratch/alice/b/bg205/16_02_26/Modelling/V3"
)

DATA_PATH = Path(
    "/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/"
    "Combined_GOLD_Aurum_with_meds_comorbidities_studyend_cod.txt"
)


PICKLES = {

    ("Female", "death_10y"):
        ROOT / "Combined_female_type_2/_run_level/"
        "Combined_type_2_death_10y_all_models_female_tune-brier_cap-90_boot-2000_rf-750.pkl",

    ("Female", "death_cvd"):
        ROOT / "Combined_female_type_2/_run_level/"
        "Combined_type_2_death_cvd_all_models_female_tune-brier_cap-90_boot-2000_rf-750.pkl",

    ("Female", "death_cancer"):
        ROOT / "Combined_female_type_2/_run_level/"
        "Combined_type_2_death_cancer_all_models_female_tune-brier_cap-90_boot-2000_rf-750.pkl",

    ("Male", "death_10y"):
        ROOT / "Combined_male_type_2/_run_level/"
        "Combined_type_2_death_10y_all_models_male_tune-brier_cap-90_boot-2000_rf-750.pkl",

    ("Male", "death_cvd"):
        ROOT / "Combined_male_type_2/_run_level/"
        "Combined_type_2_death_cvd_all_models_male_tune-brier_cap-90_boot-2000_rf-750.pkl",

    ("Male", "death_cancer"):
        ROOT / "Combined_male_type_2/_run_level/"
        "Combined_type_2_death_cancer_all_models_male_tune-brier_cap-90_boot-2000_rf-750.pkl",
}


# ============================================================
# MODEL DISPLAY SETTINGS
# ============================================================

MODELS = [
    "LR",
    "RF",
    "XGB",
    "HYBRID"
]

MODEL_LABELS = {
    "LR": "Logistic regression",
    "RF": "Random forest",
    "XGB": "XGBoost",
    "HYBRID": "Hybrid",
}

MODEL_COLORS = {
    "LR": "#1f77b4",
    "RF": "#2E8B57",
    "XGB": "#ff7f0e",
    "HYBRID": "#9467bd",
}


# ============================================================
# DATE PARSING
# ============================================================

DATE_COLS = [
    "indexdate",
    "dod_ons",
    "tod",
    "regenddate",
    "eventdate",
    "smoking_date",
    "bmi_date",
    "bp_date",
    "tot_chol_date",
    "hdl_date",
    "ldl_date",
    "trigly_date",
    "hba1c_date",
    "lcd",
    "censor_date",
    "comorb_ckd_first_date",
    "comorb_htn_first_date",
    "comorb_cvd_first_date",
    "comorb_cancer_any_first_date",
    "comorb_cancer_breast_first_date",
    "comorb_cancer_colorectal_first_date",
    "comorb_cancer_lung_first_date",
    "comorb_cancer_pancreatic_first_date",
    "comorb_cancer_prostate_first_date",
]


def parse_mixed_dates(s):

    s = s.astype("string")

    has_slash = s.str.contains(
        "/",
        na=False
    )

    out = pd.Series(
        pd.NaT,
        index=s.index,
        dtype="datetime64[ns]"
    )

    if has_slash.any():

        out.loc[has_slash] = pd.to_datetime(
            s.loc[has_slash],
            errors="coerce",
            dayfirst=True
        )

    if (~has_slash).any():

        out.loc[~has_slash] = pd.to_datetime(
            s.loc[~has_slash],
            errors="coerce"
        )

    return out


# ============================================================
# OUTCOME DERIVATION
# ============================================================

def derive_mortality_no_censoring(
    df,
    years=10,
    study_end_date="2021-03-31"
):

    df = df.copy()

    study_end = pd.to_datetime(
        study_end_date
    )

    df = df[
        df["indexdate"].notna()
    ].copy()

    df["cutoff_date"] = (
        df["indexdate"]
        + pd.DateOffset(years=years)
    )

    candidates = []

    if "tod" in df.columns:
        candidates.append(
            df["tod"]
        )

    if "regenddate" in df.columns:
        candidates.append(
            df["regenddate"]
        )

    elif "regend" in df.columns:
        candidates.append(
            pd.to_datetime(
                df["regend"],
                errors="coerce"
            )
        )

    if "lcd" in df.columns:
        candidates.append(
            df["lcd"]
        )

    if candidates:

        censor_raw = pd.concat(
            candidates,
            axis=1
        ).min(axis=1)

    else:

        censor_raw = pd.Series(
            pd.NaT,
            index=df.index
        )

    censor_raw = censor_raw.where(
        censor_raw >= df["indexdate"],
        pd.NaT
    )

    df["censor_date_derived"] = (
        censor_raw
        .fillna(study_end)
        .clip(upper=study_end)
    )

    dod_col = (
        "dod_ons"
        if "dod_ons" in df.columns
        else "dod"
    )

    dod_clean = pd.to_datetime(
        df[dod_col],
        errors="coerce"
    )

    dod_clean = dod_clean.where(
        dod_clean >= df["indexdate"],
        pd.NaT
    )

    df["died_within_followup"] = (
        dod_clean.notna()
        & (dod_clean <= df["cutoff_date"])
    )

    df["death_10y"] = (
        df["died_within_followup"]
        .astype(int)
    )

    df["death_cvd"] = (
        df["died_within_followup"]
        & (
            df["cod_cvd"]
            .fillna(0)
            .astype(int)
            == 1
        )
    ).astype(int)

    df["death_cancer"] = (
        df["died_within_followup"]
        & (
            df["cod_cancer"]
            .fillna(0)
            .astype(int)
            == 1
        )
    ).astype(int)

    df["eligible"] = True

    return df


# ============================================================
# PRACTICE-LEVEL SPLIT
# ============================================================

def split_by_practice(
    df,
    test_size=0.2,
    random_state=42,
    stratify_by_database=False
):

    if (
        stratify_by_database
        and df["database"].nunique() > 1
    ):

        train_dfs = []
        test_dfs = []

        train_prac_all = []
        test_prac_all = []

        for db in df["database"].unique():

            db_df = df[
                df["database"] == db
            ]

            practices = (
                db_df["pracid"]
                .unique()
            )

            if len(practices) < 5:

                train_dfs.append(
                    db_df
                )

                train_prac_all.extend(
                    practices
                )

                continue

            train_prac, test_prac = (
                train_test_split(
                    practices,
                    test_size=test_size,
                    random_state=random_state
                )
            )

            train_dfs.append(
                db_df[
                    db_df["pracid"]
                    .isin(train_prac)
                ]
            )

            test_dfs.append(
                db_df[
                    db_df["pracid"]
                    .isin(test_prac)
                ]
            )

            train_prac_all.extend(
                train_prac
            )

            test_prac_all.extend(
                test_prac
            )

        train_df = pd.concat(
            train_dfs,
            ignore_index=True
        )

        test_df = (
            pd.concat(
                test_dfs,
                ignore_index=True
            )
            if test_dfs
            else pd.DataFrame()
        )

    else:

        practices = (
            df["pracid"]
            .unique()
        )

        train_prac_all, test_prac_all = (
            train_test_split(
                practices,
                test_size=test_size,
                random_state=random_state
            )
        )

        train_df = df[
            df["pracid"]
            .isin(train_prac_all)
        ].copy()

        test_df = df[
            df["pracid"]
            .isin(test_prac_all)
        ].copy()

    return (
        train_df,
        test_df,
        {
            "train_practices":
                list(train_prac_all),

            "test_practices":
                list(test_prac_all),
        }
    )


# ============================================================
# LOAD ORIGINAL DATA
# ============================================================

print("Loading final cleaned dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    low_memory=False
)


for col in DATE_COLS:

    if col in df.columns:

        df[col] = parse_mixed_dates(
            df[col]
        )


df["database"] = (
    df["database"]
    .astype("string")
    .str.strip()
    .str.upper()
)


df["diabetes_type"] = pd.to_numeric(
    df["diabetes_type"],
    errors="coerce"
)


df = df[
    df["diabetes_type"] == 2
].copy()


# ============================================================
# RECONSTRUCT FINAL TEST SETS
# ============================================================

test_sets = {}


for sex in [
    "Female",
    "Male"
]:

    print(
        f"\nReconstructing {sex} test set..."
    )

    representative_pickle = (
        PICKLES[
            (sex, "death_10y")
        ]
    )

    with open(
        representative_pickle,
        "rb"
    ) as f:

        saved = pickle.load(f)


    config = saved[
        "run_configuration"
    ]

    split_info = saved[
        "split_info"
    ]


    sex_df = df.copy()

    sex_df["gender"] = (
        sex_df["gender"]
        .astype(str)
        .str.strip()
    )


    if sex == "Female":

        allowed = {
            "F",
            "FEMALE",
            "2"
        }

    else:

        allowed = {
            "M",
            "MALE",
            "1"
        }


    sex_df = sex_df[
        sex_df["gender"]
        .str.upper()
        .isin(allowed)
    ].copy()


    sex_df = derive_mortality_no_censoring(
        sex_df,
        years=10,
        study_end_date=
            config["study_end_date"]
    )


    sex_df = sex_df[
        sex_df["eligible"]
    ].copy()


    _, test_df, rebuilt_split = (
        split_by_practice(
            sex_df,
            test_size=0.2,
            random_state=
                config["split_seed"],
            stratify_by_database=True
        )
    )


    saved_practices = sorted(
        map(
            str,
            split_info[
                "test_practices"
            ]
        )
    )

    rebuilt_practices = sorted(
        map(
            str,
            rebuilt_split[
                "test_practices"
            ]
        )
    )


    assert (
        saved_practices
        == rebuilt_practices
    ), (
        f"{sex}: reconstructed "
        f"practice split does not "
        f"match modelling run."
    )


    test_sets[sex] = test_df


    print(
        f"{sex} test N = "
        f"{len(test_df):,}"
    )


# ============================================================
# FINAL TEST N CHECK
# ============================================================

assert (
    len(test_sets["Female"])
    == 401565
), (
    "Female test N does not "
    "match final analysis."
)

assert (
    len(test_sets["Male"])
    == 329551
), (
    "Male test N does not "
    "match final analysis."
)

print(
    "\nTest-set sizes match "
    "the final primary analysis."
)


# ============================================================
# EXTRACT MODEL 4 PREDICTIONS
# ============================================================

PANEL_INFO = [

    (
        "Female",
        "death_10y",
        "A. Female: All-cause"
    ),

    (
        "Female",
        "death_cvd",
        "B. Female: CVD"
    ),

    (
        "Female",
        "death_cancer",
        "C. Female: Cancer"
    ),

    (
        "Male",
        "death_10y",
        "D. Male: All-cause"
    ),

    (
        "Male",
        "death_cvd",
        "E. Male: CVD"
    ),

    (
        "Male",
        "death_cancer",
        "F. Male: Cancer"
    ),
]


plot_data = {}


for sex, outcome, title in PANEL_INFO:

    with open(
        PICKLES[(sex, outcome)],
        "rb"
    ) as f:

        saved = pickle.load(f)


    result_key = (
        f"results_{outcome}__D"
    )

    hybrid_key = (
        f"hybrid_result_{outcome}__D"
    )


    standard_results = saved[
        result_key
    ]

    hybrid_result = saved[
        hybrid_key
    ]


    predictions = {

        "LR":
            np.asarray(
                standard_results[
                    "LR"
                ]["y_pred_proba"]
            ),

        "RF":
            np.asarray(
                standard_results[
                    "RF"
                ]["y_pred_proba"]
            ),

        "XGB":
            np.asarray(
                standard_results[
                    "XGB"
                ]["y_pred_proba"]
            ),

        "HYBRID":
            np.asarray(
                hybrid_result[
                    "y_pred_proba"
                ]
            ),
    }


    y_true = (
        test_sets[sex][outcome]
        .astype(int)
        .to_numpy()
    )


    for model in MODELS:

        assert (
            len(predictions[model])
            == len(y_true)
        ), (
            f"{sex} {outcome} "
            f"{model}: prediction "
            f"length mismatch."
        )


    plot_data[
        (sex, outcome)
    ] = {
        "y_true": y_true,
        "predictions": predictions,
    }


    print(
        f"{sex:6s} | "
        f"{outcome:12s} | "
        f"N={len(y_true):,} | "
        f"events={y_true.sum():,}"
    )


# ============================================================
# SUPPLEMENTARY FIGURE S1 — ROC
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(13, 8)
)


for ax, (
    sex,
    outcome,
    title
) in zip(
    axes.flat,
    PANEL_INFO
):

    d = plot_data[
        (sex, outcome)
    ]

    y_true = d[
        "y_true"
    ]


    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.55
    )


    for model in MODELS:

        pred = d[
            "predictions"
        ][model]

        fpr, tpr, _ = roc_curve(
            y_true,
            pred
        )

        auc_value = roc_auc_score(
            y_true,
            pred
        )

        ax.plot(
            fpr,
            tpr,
            color=MODEL_COLORS[
                model
            ],
            linewidth=2,
            label=(
                f"{MODEL_LABELS[model]} "
                f"({auc_value:.3f})"
            )
        )


    ax.set_title(
        title,
        fontsize=11
    )

    ax.set_xlim(
        0,
        1
    )

    # ROC ALWAYS 0–1
    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        loc="lower right",
        fontsize=7,
        frameon=True
    )


axes[0, 0].set_ylabel(
    "True positive rate"
)

axes[1, 0].set_ylabel(
    "True positive rate"
)


for ax in axes[1, :]:

    ax.set_xlabel(
        "False positive rate"
    )


fig.suptitle(
    "Supplementary Figure S1. ROC curves for Model 4",
    fontsize=15,
    y=0.98
)


plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)


plt.savefig(
    "Supplementary_Figure_S1_ROC_Model4.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "Supplementary_Figure_S1_ROC_Model4.pdf",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SUPPLEMENTARY FIGURE S2 — PRECISION–RECALL
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(13, 8)
)


for ax, (
    sex,
    outcome,
    title
) in zip(
    axes.flat,
    PANEL_INFO
):

    d = plot_data[
        (sex, outcome)
    ]

    y_true = d[
        "y_true"
    ]

    prevalence = float(
        y_true.mean()
    )


    ax.axhline(
        prevalence,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label=(
            f"No skill "
            f"({prevalence:.3f})"
        )
    )


    for model in MODELS:

        pred = d[
            "predictions"
        ][model]

        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                pred
            )
        )

        ap = (
            average_precision_score(
                y_true,
                pred
            )
        )

        ax.plot(
            recall,
            precision,
            color=MODEL_COLORS[
                model
            ],
            linewidth=2,
            label=(
                f"{MODEL_LABELS[model]} "
                f"(AP {ap:.3f})"
            )
        )


    ax.set_title(
        title,
        fontsize=11
    )

    ax.set_xlim(
        0,
        1
    )

    # All-cause stays 0–1.
    # Cause-specific panels use 0–0.4
    # to make differences easier to see.
    if outcome == "death_10y":
        ax.set_ylim(
            0,
            1.0
        )
    else:
        ax.set_ylim(
            0,
            0.4
        )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        loc="upper right",
        fontsize=7,
        frameon=True
    )


axes[0, 0].set_ylabel(
    "Precision"
)

axes[1, 0].set_ylabel(
    "Precision"
)


for ax in axes[1, :]:

    ax.set_xlabel(
        "Recall"
    )


fig.suptitle(
    (
        "Supplementary Figure S2. "
        "Precision–recall curves for Model 4"
    ),
    fontsize=15,
    y=0.98
)


plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)


plt.savefig(
    "Supplementary_Figure_S2_PrecisionRecall_Model4.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "Supplementary_Figure_S2_PrecisionRecall_Model4.pdf",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("\nDONE")

print("Saved:")
print("  Supplementary_Figure_S1_ROC_Model4.png")
print("  Supplementary_Figure_S1_ROC_Model4.pdf")
print("  Supplementary_Figure_S2_PrecisionRecall_Model4.png")
print("  Supplementary_Figure_S2_PrecisionRecall_Model4.pdf")
