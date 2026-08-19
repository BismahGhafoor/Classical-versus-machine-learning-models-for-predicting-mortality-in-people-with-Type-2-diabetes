import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/scratch/alice/b/bg205/16_02_26/Modelling/V3")

ANALYSES = [
    ("Female", "death_10y",    "All-cause mortality"),
    ("Female", "death_cvd",    "CVD mortality"),
    ("Female", "death_cancer", "Cancer mortality"),
    ("Male",   "death_10y",    "All-cause mortality"),
    ("Male",   "death_cvd",    "CVD mortality"),
    ("Male",   "death_cancer", "Cancer mortality"),
]


def outcome_folder(sex, outcome):

    sex_folder = (
        "Combined_female_type_2"
        if sex == "Female"
        else "Combined_male_type_2"
    )

    return ROOT / sex_folder / outcome


def get_primary_file(folder, pattern):

    matches = [
        p for p in folder.glob(pattern)
        if "SENSITIVITY" not in p.name
        and "complete" not in p.name.lower()
    ]

    if len(matches) != 1:

        print(f"\nFolder: {folder}")
        print(f"Pattern: {pattern}")
        print("Matches:")

        for p in matches:
            print(" ", p)

        raise RuntimeError(
            f"Expected exactly one primary file; found {len(matches)}."
        )

    return matches[0]


# ============================================================
# ROUNDING
# ============================================================

def round_half_up(value, decimals):

    q = Decimal("1").scaleb(-decimals)

    return format(
        Decimal(str(value)).quantize(
            q,
            rounding=ROUND_HALF_UP
        ),
        f".{decimals}f"
    )


# ============================================================
# EXTRACT MODEL 4 PAIRED COMPARISONS
# ============================================================

rows = []

for sex, outcome, outcome_label in ANALYSES:

    folder = outcome_folder(sex, outcome)

    comparison_file = get_primary_file(
        folder,
        "*model_vs_model*.csv"
    )

    df = pd.read_csv(comparison_file)

    # Clean labels
    for col in ["Model_1", "Model_2", "Metric", "Block"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

    df["Model_1"] = df["Model_1"].str.upper()
    df["Model_2"] = df["Model_2"].str.upper()
    df["Metric"] = df["Metric"].str.lower()

    # Model 4 AUC comparisons only
    df = df[
        (df["Block"] == "Model 4")
        & (df["Metric"] == "auc")
    ].copy()

    # We only want:
    # XGB vs LR
    # XGB vs RF
    for comparator, comparator_label in [
        ("LR", "Logistic regression"),
        ("RF", "Random forest")
    ]:

        pair = df[
            (
                (df["Model_1"] == "XGB")
                & (df["Model_2"] == comparator)
            )
            |
            (
                (df["Model_1"] == comparator)
                & (df["Model_2"] == "XGB")
            )
        ]

        if len(pair) != 1:

            raise RuntimeError(
                f"{sex} {outcome}: expected exactly one "
                f"XGB vs {comparator} Model 4 AUC comparison, "
                f"found {len(pair)}."
            )

        r = pair.iloc[0]

        model_1 = r["Model_1"]
        model_2 = r["Model_2"]

        est_1 = float(r["Estimate_model_1"])
        est_2 = float(r["Estimate_model_2"])

        diff_raw = float(
            r["Difference_model_2_minus_model_1"]
        )

        ci_low_raw = float(r["CI_lower"])
        ci_high_raw = float(r["CI_upper"])

        p_value = float(r["p_value"])

        # ----------------------------------------------------
        # Standardise everything to:
        # XGBoost MINUS comparator
        # ----------------------------------------------------

        if model_2 == "XGB":

            xgb_auc = est_2
            comparator_auc = est_1

            delta = diff_raw
            ci_low = ci_low_raw
            ci_high = ci_high_raw

        elif model_1 == "XGB":

            xgb_auc = est_1
            comparator_auc = est_2

            delta = -diff_raw

            # Reverse and negate CI
            ci_low = -ci_high_raw
            ci_high = -ci_low_raw

        else:

            raise RuntimeError(
                f"{sex} {outcome}: comparison does not contain XGB."
            )

        rows.append({
            "Sex": sex,
            "Outcome": outcome_label,
            "Comparator": comparator_label,
            "XGB_AUC": xgb_auc,
            "Comparator_AUC": comparator_auc,
            "Delta_AUC": delta,
            "CI_low": ci_low,
            "CI_high": ci_high,
            "p_value": p_value,
        })


results = pd.DataFrame(rows)


# ============================================================
# QC PRINT
# ============================================================

print("\nFINAL MODEL 4 COMPARISONS\n")

print(
    results.to_string(index=False)
)


# ============================================================
# FORMAT TABLE CONTENT
# ============================================================

table_rows = []

for sex in ["Female", "Male"]:

    sex_df = results[
        results["Sex"] == sex
    ]

    first_row_for_sex = True

    for outcome in [
        "All-cause mortality",
        "CVD mortality",
        "Cancer mortality"
    ]:

        for comparator in [
            "Logistic regression",
            "Random forest"
        ]:

            r = sex_df[
                (sex_df["Outcome"] == outcome)
                & (sex_df["Comparator"] == comparator)
            ].iloc[0]

            xgb_auc = round_half_up(
                r["XGB_AUC"], 4
            )

            comparator_auc = round_half_up(
                r["Comparator_AUC"], 4
            )

            delta = round_half_up(
                r["Delta_AUC"], 4
            )

            ci_low = round_half_up(
                r["CI_low"], 4
            )

            ci_high = round_half_up(
                r["CI_high"], 4
            )

            # Explicit + sign because table reports XGB - comparator
            if float(r["Delta_AUC"]) >= 0:
                delta_text = (
                    f"+{delta} "
                    f"({ci_low} to {ci_high})"
                )
            else:
                delta_text = (
                    f"{delta} "
                    f"({ci_low} to {ci_high})"
                )

            # Bootstrap p-values
            p = float(r["p_value"])

            if p < 0.001:
                p_text = "<0.001"
            else:
                p_text = round_half_up(
                    p, 3
                )

            table_rows.append([
                sex if first_row_for_sex else "",
                outcome,
                f"XGBoost vs {comparator}",
                xgb_auc,
                comparator_auc,
                delta_text,
                p_text,
            ])

            first_row_for_sex = False


columns = [
    "Sex",
    "Outcome",
    "Comparison",
    "XGBoost AUC",
    "Comparator AUC",
    "ΔAUC (95% CI)",
    "Bootstrap p-value",
]


# ============================================================
# PRINT FORMATTED TABLE FOR QC
# ============================================================

print("\nFORMATTED TABLE 3\n")

print(
    pd.DataFrame(
        table_rows,
        columns=columns
    ).to_string(index=False)
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 5.2),
    facecolor="white"
)

ax.axis("off")


# Optional title above table
ax.text(
    0,
    1.04,
    "Table 3. Paired comparison of XGBoost Model 4 with logistic regression and random forest",
    fontsize=10,
    fontweight="bold",
    ha="left",
    va="bottom",
    transform=ax.transAxes
)


table = ax.table(
    cellText=table_rows,
    colLabels=columns,
    cellLoc="left",
    colLoc="left",
    bbox=[0, 0, 1, 0.96],
    colWidths=[
        0.10,   # Sex
        0.17,   # Outcome
        0.26,   # Comparison
        0.10,   # XGBoost AUC
        0.11,   # Comparator AUC
        0.16,   # ΔAUC (95% CI)
        0.10,   # Bootstrap p-value
    ]
)


# ============================================================
# STYLE — SAME AS TABLES 1 AND 2
# ============================================================

table.auto_set_font_size(False)
table.set_fontsize(8.3)

cells = table.get_celld()

# Header row = 0
# Female rows = 1–6
# Male rows = 7–12

for (row, col), cell in cells.items():

    # Borders
    cell.set_edgecolor("#B7B7B7")
    cell.set_linewidth(0.45)

    # Compact spacing
    cell.PAD = 0.018

    cell.get_text().set_ha("left")
    cell.get_text().set_va("center")

    # Header
    if row == 0:

        cell.set_facecolor("#BFBFBF")
        cell.get_text().set_fontweight("bold")

    # ALL Female rows shaded
    elif 1 <= row <= 6:

        cell.set_facecolor("#E6E6E6")

    # ALL Male rows white
    else:

        cell.set_facecolor("white")

    # Bold sex labels only
    if (
        col == 0
        and row in [1, 7]
    ):
        cell.get_text().set_fontweight("bold")


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    "Table3_Model4_model_comparisons.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    "Table3_Model4_model_comparisons.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.close()


print("\nDONE")
print("Saved:")
print("  Table3_Model4_model_comparisons.png")
print("  Table3_Model4_model_comparisons.pdf")
