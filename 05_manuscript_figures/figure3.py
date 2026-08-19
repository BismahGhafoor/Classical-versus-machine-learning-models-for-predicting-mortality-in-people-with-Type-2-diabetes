import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/scratch/alice/b/bg205/16_02_26/Modelling/V3")

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
# FUNCTIONS MIRRORING FINAL MODELLING PIPELINE
# ============================================================

def parse_mixed_dates(s):
    s = s.astype("string")
    has_slash = s.str.contains("/", na=False)

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


def derive_mortality_outcomes(df, study_end_date):
    df = df.copy()

    study_end = pd.to_datetime(study_end_date)

    df = df[df["indexdate"].notna()].copy()

    df["cutoff_date"] = (
        df["indexdate"] + pd.DateOffset(years=10)
    )

    dod_clean = pd.to_datetime(
        df["dod_ons"],
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
        & (df["cod_cvd"].fillna(0).astype(int) == 1)
    ).astype(int)

    df["death_cancer"] = (
        df["died_within_followup"]
        & (df["cod_cancer"].fillna(0).astype(int) == 1)
    ).astype(int)

    return df


def split_by_practice(
    df,
    test_size=0.2,
    random_state=42,
    stratify_by_database=True
):

    if stratify_by_database and df["database"].nunique() > 1:

        train_dfs = []
        test_dfs = []
        train_prac_all = []
        test_prac_all = []

        for db in df["database"].unique():

            db_df = df[df["database"] == db]

            practices = db_df["pracid"].unique()

            train_prac, test_prac = train_test_split(
                practices,
                test_size=test_size,
                random_state=random_state
            )

            train_dfs.append(
                db_df[
                    db_df["pracid"].isin(train_prac)
                ]
            )

            test_dfs.append(
                db_df[
                    db_df["pracid"].isin(test_prac)
                ]
            )

            train_prac_all.extend(train_prac)
            test_prac_all.extend(test_prac)

        train_df = pd.concat(
            train_dfs,
            ignore_index=True
        )

        test_df = pd.concat(
            test_dfs,
            ignore_index=True
        )

    else:

        practices = df["pracid"].unique()

        train_prac_all, test_prac_all = train_test_split(
            practices,
            test_size=test_size,
            random_state=random_state
        )

        train_df = df[
            df["pracid"].isin(train_prac_all)
        ].copy()

        test_df = df[
            df["pracid"].isin(test_prac_all)
        ].copy()

    return train_df, test_df, {
        "train_practices": list(train_prac_all),
        "test_practices": list(test_prac_all),
    }


# ============================================================
# LOAD ORIGINAL DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    low_memory=False
)

for col in [
    "indexdate",
    "dod_ons",
    "tod",
    "regenddate",
    "lcd"
]:
    if col in df.columns:
        df[col] = parse_mixed_dates(df[col])

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

# Type 2 only
df = df[
    df["diabetes_type"] == 2
].copy()


# ============================================================
# REBUILD FEMALE AND MALE TEST SETS
# ============================================================

test_sets = {}

for sex in ["Female", "Male"]:

    print(f"\nReconstructing {sex} test set...")

    representative_pickle = PICKLES[
        (sex, "death_10y")
    ]

    with open(representative_pickle, "rb") as f:
        saved = pickle.load(f)

    config = saved["run_configuration"]
    split_info = saved["split_info"]

    sex_df = df.copy()

    sex_df["gender"] = (
        sex_df["gender"]
        .astype(str)
        .str.strip()
    )

    if sex == "Female":
        allowed = {"F", "FEMALE", "2"}
    else:
        allowed = {"M", "MALE", "1"}

    sex_df = sex_df[
        sex_df["gender"]
        .str.upper()
        .isin(allowed)
    ].copy()

    sex_df = derive_mortality_outcomes(
        sex_df,
        study_end_date=config["study_end_date"]
    )

    _, test_df, rebuilt_split = split_by_practice(
        sex_df,
        test_size=0.2,
        random_state=config["split_seed"],
        stratify_by_database=True
    )

    # Verify same test practices as modelling run
    saved_practices = sorted(
        map(str, split_info["test_practices"])
    )

    rebuilt_practices = sorted(
        map(str, rebuilt_split["test_practices"])
    )

    assert saved_practices == rebuilt_practices, (
        f"{sex}: reconstructed test practices "
        f"do not match modelling run."
    )

    test_sets[sex] = test_df

    print(
        f"{sex} test N = {len(test_df):,}"
    )


# ============================================================
# EXTRACT MODEL 4 PREDICTIONS AND CALIBRATION CURVES
# ============================================================

panel_info = [
    ("Female", "death_10y",    "A. Female: All-cause"),
    ("Female", "death_cvd",    "B. Female: CVD"),
    ("Female", "death_cancer", "C. Female: Cancer"),
    ("Male",   "death_10y",    "D. Male: All-cause"),
    ("Male",   "death_cvd",    "E. Male: CVD"),
    ("Male",   "death_cancer", "F. Male: Cancer"),
]

curves = {}

for sex, outcome, title in panel_info:

    with open(PICKLES[(sex, outcome)], "rb") as f:
        saved = pickle.load(f)

    key = f"results_{outcome}__D"

    results = saved[key]

    lr_pred = np.asarray(
        results["LR"]["y_pred_proba"]
    )

    xgb_pred = np.asarray(
        results["XGB"]["y_pred_proba"]
    )

    y_true = (
        test_sets[sex][outcome]
        .astype(int)
        .to_numpy()
    )

    # Critical safety check
    assert len(y_true) == len(lr_pred), (
        f"{sex} {outcome}: "
        f"LR prediction length mismatch "
        f"{len(lr_pred)} vs {len(y_true)}"
    )

    assert len(y_true) == len(xgb_pred), (
        f"{sex} {outcome}: "
        f"XGB prediction length mismatch "
        f"{len(xgb_pred)} vs {len(y_true)}"
    )

    lr_true, lr_pred_mean = calibration_curve(
        y_true,
        lr_pred,
        n_bins=20,
        strategy="quantile"
    )

    xgb_true, xgb_pred_mean = calibration_curve(
        y_true,
        xgb_pred,
        n_bins=20,
        strategy="quantile"
    )

    curves[(sex, outcome)] = {
        "lr_true": lr_true,
        "lr_pred": lr_pred_mean,
        "xgb_true": xgb_true,
        "xgb_pred": xgb_pred_mean,
    }

    print(
        f"{sex:6s} {outcome:12s}: "
        f"N={len(y_true):,}, "
        f"events={y_true.sum():,}"
    )


# ============================================================
# CREATE 2 × 3 FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

# Keep scales comparable between females/males
# for each mortality outcome.
axis_limits = {
    "death_10y": 0.70,
    "death_cvd": 0.25,
    "death_cancer": 0.20,
}

for ax, (sex, outcome, title) in zip(
    axes.flat,
    panel_info
):

    c = curves[(sex, outcome)]

    lim = axis_limits[outcome]

    # Ideal calibration
    ax.plot(
        [0, lim],
        [0, lim],
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Ideal calibration"
    )

    # Logistic regression
    ax.plot(
        c["lr_pred"],
        c["lr_true"],
        color="#1f77b4",
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="Logistic regression"
    )

    # XGBoost
    ax.plot(
        c["xgb_pred"],
        c["xgb_true"],
        color="#ff7f0e",
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="XGBoost"
    )

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.set_title(
        title,
        fontsize=11
    )

    ax.grid(
        True,
        alpha=0.25
    )


# Labels only where needed
axes[0, 0].set_ylabel("Observed risk")
axes[1, 0].set_ylabel("Observed risk")

for ax in axes[1, :]:
    ax.set_xlabel("Mean predicted risk")


# Overall title
fig.suptitle(
    "Calibration of logistic regression and XGBoost Model 4 models",
    fontsize=15,
    y=0.98
)


# Shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.015)
)


plt.tight_layout(
    rect=[0, 0.06, 1, 0.95]
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    "Figure3_calibration_Model4.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "Figure3_calibration_Model4.pdf",
    bbox_inches="tight"
)

plt.close()

print("\nDONE")
print("Saved:")
print("  Figure3_calibration_Model4.png")
print("  Figure3_calibration_Model4.pdf")
