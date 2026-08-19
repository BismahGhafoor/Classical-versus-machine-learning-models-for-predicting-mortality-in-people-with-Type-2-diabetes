import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/scratch/alice/b/bg205/16_02_26/Modelling/V3")

ANALYSES = [
    ("Female", "death_10y",    "A. Female: All-cause"),
    ("Female", "death_cvd",    "B. Female: CVD"),
    ("Female", "death_cancer", "C. Female: Cancer"),
    ("Male",   "death_10y",    "D. Male: All-cause"),
    ("Male",   "death_cvd",    "E. Male: CVD"),
    ("Male",   "death_cancer", "F. Male: Cancer"),
]


def outcome_folder(sex, outcome):
    sex_folder = (
        "Combined_female_type_2"
        if sex == "Female"
        else "Combined_male_type_2"
    )
    return ROOT / sex_folder / outcome


def get_primary_file(folder, pattern):
    """
    Find one primary-analysis file and explicitly ignore
    complete-case sensitivity outputs.
    """
    matches = [
        p for p in folder.glob(pattern)
        if "SENSITIVITY" not in p.name
    ]

    if len(matches) != 1:
        print(f"\nFolder: {folder}")
        print(f"Pattern: {pattern}")
        print("Matches:")
        for p in matches:
            print(" ", p)

        raise RuntimeError(
            f"Expected exactly one primary file, found {len(matches)}."
        )

    return matches[0]


# ============================================================
# LOAD FINAL DCA VALUES
# ============================================================

plot_data = {}

for sex, outcome, title in ANALYSES:

    folder = outcome_folder(sex, outcome)

    dca_file = get_primary_file(
        folder,
        "*dca_net_benefit_values*.csv"
    )

    summary_file = get_primary_file(
        folder,
        "*model_summary*.csv"
    )

    print(f"\n{sex} — {outcome}")
    print(f"  DCA:     {dca_file.name}")
    print(f"  Summary: {summary_file.name}")

    dca = pd.read_csv(dca_file)
    summary = pd.read_csv(summary_file)

    # --------------------------------------------------------
    # Model 4 only
    # --------------------------------------------------------

    dca["Block"] = dca["Block"].astype(str).str.strip()
    dca["Model"] = dca["Model"].astype(str).str.strip().str.upper()

    dca4 = dca[
        (dca["Block"] == "Model 4")
        & (dca["Model"].isin(["LR", "XGB"]))
    ].copy()

    lr = (
        dca4[dca4["Model"] == "LR"]
        .sort_values("Threshold")
        .reset_index(drop=True)
    )

    xgb = (
        dca4[dca4["Model"] == "XGB"]
        .sort_values("Threshold")
        .reset_index(drop=True)
    )

    if len(lr) == 0 or len(xgb) == 0:
        raise RuntimeError(
            f"{sex} {outcome}: Model 4 LR/XGB rows not found."
        )

    if len(lr) != len(xgb):
        raise RuntimeError(
            f"{sex} {outcome}: LR and XGB threshold counts differ."
        )

    if not np.allclose(
        lr["Threshold"].to_numpy(),
        xgb["Threshold"].to_numpy()
    ):
        raise RuntimeError(
            f"{sex} {outcome}: LR and XGB thresholds differ."
        )

    thresholds = lr["Threshold"].to_numpy()

    # --------------------------------------------------------
    # Obtain exact prevalence from Model 4 summary
    # --------------------------------------------------------

    summary["Block"] = summary["Block"].astype(str).str.strip()
    summary["Model"] = summary["Model"].astype(str).str.strip().str.upper()

    row = summary[
        (summary["Block"] == "Model 4")
        & (summary["Model"] == "XGB")
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"{sex} {outcome}: expected one Model 4 XGB summary row."
        )

    n_test = int(row.iloc[0]["N_test"])
    events_test = int(row.iloc[0]["Events_test"])

    prevalence = events_test / n_test

    # Treat-none and treat-all reference strategies
    treat_none = np.zeros_like(thresholds)

    treat_all = (
        prevalence
        - (1 - prevalence)
        * (thresholds / (1 - thresholds))
    )

    plot_data[(sex, outcome)] = {
        "thresholds": thresholds,
        "lr": lr["Net_benefit"].to_numpy(),
        "xgb": xgb["Net_benefit"].to_numpy(),
        "treat_none": treat_none,
        "treat_all": treat_all,
        "prevalence": prevalence,
        "n_test": n_test,
        "events": events_test,
    }

    print(
        f"  N={n_test:,}, events={events_test:,}, "
        f"prevalence={prevalence:.4f}, "
        f"threshold range={thresholds.min():.3f}–{thresholds.max():.3f}"
    )


# ============================================================
# CREATE 2 × 3 FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

for ax, (sex, outcome, title) in zip(
    axes.flat,
    ANALYSES
):

    d = plot_data[(sex, outcome)]

    thresholds = d["thresholds"]
    prevalence = d["prevalence"]

    # Treat none
    ax.plot(
        thresholds,
        d["treat_none"],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Treat none"
    )

    # Treat all
    ax.plot(
        thresholds,
        d["treat_all"],
        color="grey",
        linestyle=":",
        linewidth=1.4,
        label="Treat all"
    )

    # Logistic regression
    ax.plot(
        thresholds,
        d["lr"],
        color="#1f77b4",
        linewidth=2,
        label="Logistic regression"
    )

    # XGBoost
    ax.plot(
        thresholds,
        d["xgb"],
        color="#ff7f0e",
        linewidth=2,
        label="XGBoost"
    )

    # Match final pipeline scaling
    ax.set_xlim(
        thresholds.min(),
        thresholds.max()
    )

    ax.set_ylim(
        -0.01,
        max(0.02, prevalence * 1.15)
    )

    ax.set_title(
        title,
        fontsize=11
    )

    ax.grid(
        True,
        alpha=0.25
    )


# ============================================================
# AXIS LABELS
# ============================================================

axes[0, 0].set_ylabel("Net benefit")
axes[1, 0].set_ylabel("Net benefit")

for ax in axes[1, :]:
    ax.set_xlabel("Risk threshold")


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    "Decision curve analysis of logistic regression and XGBoost Model 4 models",
    fontsize=15,
    y=0.98
)


# ============================================================
# SHARED LEGEND
# ============================================================

handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
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
    "Figure4_DCA_Model4.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "Figure4_DCA_Model4.pdf",
    bbox_inches="tight"
)

plt.close()


print("\nDONE")
print("Saved:")
print("  Figure4_DCA_Model4.png")
print("  Figure4_DCA_Model4.pdf")
