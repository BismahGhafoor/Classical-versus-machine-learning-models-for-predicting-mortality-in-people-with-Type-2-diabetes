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
# EXTRACT FINAL RESULTS
# ============================================================

results = []

for sex, outcome, outcome_label in ANALYSES:

    folder = outcome_folder(sex, outcome)

    summary_file = get_primary_file(
        folder,
        "*model_summary*.csv"
    )

    ci_file = get_primary_file(
        folder,
        "*all_metric_confidence_intervals*.csv"
    )

    summary = pd.read_csv(summary_file)
    ci = pd.read_csv(ci_file)

    summary["Model"] = (
        summary["Model"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    summary["Block"] = (
        summary["Block"]
        .astype(str)
        .str.strip()
    )

    ci["Model"] = (
        ci["Model"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    ci["Block"] = (
        ci["Block"]
        .astype(str)
        .str.strip()
    )

    # XGBoost Model 4 only
    srow = summary[
        (summary["Model"] == "XGB")
        & (summary["Block"] == "Model 4")
    ]

    crow = ci[
        (ci["Model"] == "XGB")
        & (ci["Block"] == "Model 4")
    ]

    if len(srow) != 1:
        raise RuntimeError(
            f"{sex} {outcome}: expected one XGB Model 4 summary row."
        )

    if len(crow) != 1:
        raise RuntimeError(
            f"{sex} {outcome}: expected one XGB Model 4 CI row."
        )

    srow = srow.iloc[0]
    crow = crow.iloc[0]

    # Cohort and event counts
    n_total = (
        int(srow["N_train"])
        + int(srow["N_test"])
    )

    events_total = (
        int(srow["Events_train"])
        + int(srow["Events_test"])
    )

    event_rate = (
        events_total / n_total * 100
    )

    # Final model estimates
    auc = float(srow["Test_AUC"])
    auc_low = float(crow["auc_lower"])
    auc_high = float(crow["auc_upper"])
    brier = float(srow["Brier"])
    oe = float(srow["O/E"])

    results.append({
        "Sex": sex,
        "Outcome": outcome_label,
        "N_total": n_total,
        "Events": events_total,
        "Event_rate": event_rate,
        "AUC": auc,
        "AUC_low": auc_low,
        "AUC_high": auc_high,
        "Brier": brier,
        "OE": oe,
    })


results = pd.DataFrame(results)


# ============================================================
# SAFETY CHECK
# ============================================================

for sex in ["Female", "Male"]:

    cohort_sizes = (
        results[
            results["Sex"] == sex
        ]["N_total"]
        .unique()
    )

    if len(cohort_sizes) != 1:
        raise RuntimeError(
            f"{sex}: cohort sizes differ across outcomes: "
            f"{cohort_sizes}"
        )

print("\nAll cohort-size checks passed.")


# ============================================================
# PANEL A DATA
# ============================================================

panel_a_rows = []

for sex in ["Female", "Male"]:

    sub = (
        results[
            results["Sex"] == sex
        ]
        .set_index("Outcome")
    )

    n_total = int(
        sub.iloc[0]["N_total"]
    )

    panel_a_rows.append([
        sex,

        f"{n_total:,}",

        (
            f"{int(sub.loc['All-cause mortality', 'Events']):,} "
            f"({sub.loc['All-cause mortality', 'Event_rate']:.2f}%)"
        ),

        (
            f"{int(sub.loc['CVD mortality', 'Events']):,} "
            f"({sub.loc['CVD mortality', 'Event_rate']:.2f}%)"
        ),

        (
            f"{int(sub.loc['Cancer mortality', 'Events']):,} "
            f"({sub.loc['Cancer mortality', 'Event_rate']:.2f}%)"
        ),
    ])


panel_a_columns = [
    "Sex",
    "Final Type 2 cohort, n",
    "All-cause deaths, n (%)",
    "CVD deaths, n (%)",
    "Cancer deaths, n (%)",
]


# ============================================================
# PANEL B DATA
# ============================================================

panel_b_rows = []

for sex in ["Female", "Male"]:

    sub = results[
        results["Sex"] == sex
    ]

    first = True

    for outcome in [
        "All-cause mortality",
        "CVD mortality",
        "Cancer mortality"
    ]:

        row = sub[
            sub["Outcome"] == outcome
        ].iloc[0]

        auc = round_half_up(
            row["AUC"],
            3
        )

        auc_low = round_half_up(
            row["AUC_low"],
            3
        )

        auc_high = round_half_up(
            row["AUC_high"],
            3
        )

        brier = round_half_up(
            row["Brier"],
            4
        )

        oe = round_half_up(
            row["OE"],
            3
        )

        panel_b_rows.append([
            sex if first else "",
            outcome,
            f"{auc} ({auc_low}–{auc_high})",
            brier,
            oe,
        ])

        first = False


panel_b_columns = [
    "Sex",
    "Outcome",
    "AUC (95% CI)",
    "Brier score",
    "O/E",
]


# ============================================================
# PRINT VALUES FOR QC
# ============================================================

print("\nPANEL A")
print(
    pd.DataFrame(
        panel_a_rows,
        columns=panel_a_columns
    ).to_string(index=False)
)

print("\nPANEL B")
print(
    pd.DataFrame(
        panel_b_rows,
        columns=panel_b_columns
    ).to_string(index=False)
)


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(14, 4.6),
    facecolor="white"
)

ax1 = fig.add_axes(
    [0.02, 0.63, 0.96, 0.25]
)

ax2 = fig.add_axes(
    [0.02, 0.07, 0.96, 0.44]
)

ax1.axis("off")
ax2.axis("off")


# ============================================================
# PANEL TITLES
# ============================================================

ax1.text(
    0,
    1.10,
    "Panel A. Final Type 2 diabetes modelling cohort and outcome event rates",
    fontsize=10,
    fontweight="bold",
    ha="left",
    va="bottom",
    transform=ax1.transAxes
)

ax2.text(
    0,
    1.07,
    "Panel B. XGBoost Model 4 performance in the primary imputed analysis",
    fontsize=10,
    fontweight="bold",
    ha="left",
    va="bottom",
    transform=ax2.transAxes
)


# ============================================================
# TABLE A
# ============================================================

table_a = ax1.table(
    cellText=panel_a_rows,
    colLabels=panel_a_columns,
    cellLoc="left",
    colLoc="left",
    bbox=[0, 0, 1, 0.90],
    colWidths=[
        0.13,
        0.23,
        0.22,
        0.21,
        0.21
    ]
)


# ============================================================
# TABLE B
# ============================================================

table_b = ax2.table(
    cellText=panel_b_rows,
    colLabels=panel_b_columns,
    cellLoc="left",
    colLoc="left",
    bbox=[0, 0, 1, 0.94],
    colWidths=[
        0.13,
        0.33,
        0.25,
        0.17,
        0.12
    ]
)


# ============================================================
# STYLE
# ============================================================

def style_table(
    table,
    shade_rows=None,
    bold_first_col_rows=None
):

    if shade_rows is None:
        shade_rows = []

    if bold_first_col_rows is None:
        bold_first_col_rows = []

    table.auto_set_font_size(False)
    table.set_fontsize(8.3)

    cells = table.get_celld()

    for (row, col), cell in cells.items():

        # Thin borders
        cell.set_edgecolor("#B7B7B7")
        cell.set_linewidth(0.45)

        # Compact row spacing
        cell.PAD = 0.018

        cell.get_text().set_ha("left")
        cell.get_text().set_va("center")

        # Header
        if row == 0:

            cell.set_facecolor("#BFBFBF")
            cell.get_text().set_fontweight("bold")

        # Shaded rows
        elif row in shade_rows:

            cell.set_facecolor("#E6E6E6")

        # White rows
        else:

            cell.set_facecolor("white")

        # Bold sex labels
        if (
            row in bold_first_col_rows
            and col == 0
        ):
            cell.get_text().set_fontweight(
                "bold"
            )


# ============================================================
# APPLY STYLE
# ============================================================

# Panel A
# Female row shaded
# Male row white
style_table(
    table_a,
    shade_rows=[1],
    bold_first_col_rows=[1, 2]
)

# Panel B
# Rows 1,2,3 = Female → all shaded
# Rows 4,5,6 = Male → all white
style_table(
    table_b,
    shade_rows=[1, 2, 3],
    bold_first_col_rows=[1, 4]
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    "Table2_final_results.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    "Table2_final_results.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.close()


print("\nDONE")
print("Saved:")
print("  Table2_final_results.png")
print("  Table2_final_results.pdf")
