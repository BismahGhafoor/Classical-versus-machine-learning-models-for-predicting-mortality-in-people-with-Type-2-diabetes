import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP


# ============================================================
# PATHS — FINAL V3, 2000 BOOTSTRAPS ONLY
# ============================================================

ROOT = Path("/scratch/alice/b/bg205/16_02_26/Modelling/V3")

FILES = {
    ("Female", "All-cause mortality"):
        ROOT / "Combined_female_type_2/postprocessing_subgroups_xgb_blockd/"
        "death_10y/boot_2000_excluding_missing_subgroup/tables/"
        "XGB_block_D_death_10y_subgroup_performance_with_ci.csv",

    ("Female", "CVD mortality"):
        ROOT / "Combined_female_type_2/postprocessing_subgroups_xgb_blockd/"
        "death_cvd/boot_2000_excluding_missing_subgroup/tables/"
        "XGB_block_D_death_cvd_subgroup_performance_with_ci.csv",

    ("Female", "Cancer mortality"):
        ROOT / "Combined_female_type_2/postprocessing_subgroups_xgb_blockd/"
        "death_cancer/boot_2000_excluding_missing_subgroup/tables/"
        "XGB_block_D_death_cancer_subgroup_performance_with_ci.csv",

    ("Male", "All-cause mortality"):
        ROOT / "Combined_male_type_2/postprocessing_subgroups_xgb_blockd/"
        "death_10y/boot_2000_excluding_missing_subgroup/tables/"
        "XGB_block_D_death_10y_subgroup_performance_with_ci.csv",

    ("Male", "CVD mortality"):
        ROOT / "Combined_male_type_2/postprocessing_subgroups_xgb_blockd/"
        "death_cvd/boot_2000_excluding_missing_subgroup/tables/"
        "XGB_block_D_death_cvd_subgroup_performance_with_ci.csv",

    ("Male", "Cancer mortality"):
        ROOT / "Combined_male_type_2/postprocessing_subgroups_xgb_blockd/"
        "death_cancer/boot_2000_excluding_missing_subgroup/tables/"
        "XGB_block_D_death_cancer_subgroup_performance_with_ci.csv",
}


# ============================================================
# ROUNDING
# ============================================================

def round_half_up(value, decimals):
    if pd.isna(value):
        return None

    q = Decimal("1").scaleb(-decimals)

    return format(
        Decimal(str(value)).quantize(
            q,
            rounding=ROUND_HALF_UP
        ),
        f".{decimals}f"
    )


# ============================================================
# CLEAN LABELS
# ============================================================

def clean_subgroup_variable(x):
    x = str(x).strip().lower()

    if x == "imd_quintile":
        return "IMD quintile"

    if x == "gen_ethnicity":
        return "Ethnicity"

    return str(x)


def clean_subgroup(x, subgroup_type):
    x = str(x).strip()

    if subgroup_type == "IMD quintile":
        replacements = {
            "1": "IMD quintile 1",
            "1.0": "IMD quintile 1",
            "2": "IMD quintile 2",
            "2.0": "IMD quintile 2",
            "3": "IMD quintile 3",
            "3.0": "IMD quintile 3",
            "4": "IMD quintile 4",
            "4.0": "IMD quintile 4",
            "5": "IMD quintile 5",
            "5.0": "IMD quintile 5",
        }

        if x in replacements:
            return replacements[x]

        # Already labelled
        if "quintile" in x.lower():
            return x

    return x


# ============================================================
# LOAD SIX FINAL FILES
# ============================================================

frames = []

for (sex, outcome), path in FILES.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Missing expected file:\n{path}"
        )

    print(f"Loading: {sex} | {outcome}")
    print(f"  {path}")

    df = pd.read_csv(path)

    df["Sex_clean"] = sex
    df["Outcome_clean"] = outcome

    df["Subgroup_type"] = (
        df["Subgroup_variable"]
        .apply(clean_subgroup_variable)
    )

    df["Subgroup_clean"] = df.apply(
        lambda r: clean_subgroup(
            r["Subgroup"],
            r["Subgroup_type"]
        ),
        axis=1
    )

    frames.append(df)


data = pd.concat(
    frames,
    ignore_index=True
)


# ============================================================
# QC
# ============================================================

print("\nLoaded rows:", len(data))

print("\nSubgroup types:")
print(
    data["Subgroup_type"]
    .value_counts(dropna=False)
)

print("\nOutcomes:")
print(
    data["Outcome_clean"]
    .value_counts(dropna=False)
)


# ============================================================
# ORDERING
# ============================================================

IMD_ORDER = {
    "IMD quintile 1": 1,
    "IMD quintile 2": 2,
    "IMD quintile 3": 3,
    "IMD quintile 4": 4,
    "IMD quintile 5": 5,
}

ETHNICITY_ORDER = {
    "White": 1,
    "South Asian": 2,
    "Black": 3,
    "Mixed/Other": 4,
    "Unknown": 5,
    "Missing": 6,
}

OUTCOMES = [
    "All-cause mortality",
    "CVD mortality",
    "Cancer mortality",
]


# ============================================================
# FORMAT ONE OUTCOME CELL
# ============================================================

def format_performance(row):

    if row is None:
        return "N/A"

    auc = row.get("auc", np.nan)

    if pd.isna(auc):
        return "N/A"

    auc_low = row.get("auc_lower", np.nan)
    auc_high = row.get("auc_upper", np.nan)
    oe = row.get("oe_ratio", np.nan)

    auc_text = round_half_up(
        auc,
        3
    )

    if pd.notna(auc_low) and pd.notna(auc_high):

        low = round_half_up(
            auc_low,
            3
        )

        high = round_half_up(
            auc_high,
            3
        )

        auc_text = (
            f"{auc_text} "
            f"({low}–{high})"
        )

    if pd.notna(oe):

        oe_text = round_half_up(
            oe,
            3
        )

        return (
            f"AUC {auc_text}; "
            f"O/E {oe_text}"
        )

    return f"AUC {auc_text}"


# ============================================================
# BUILD PANEL
# ============================================================

def build_panel(subgroup_type):

    df = data[
        data["Subgroup_type"] == subgroup_type
    ].copy()

    rows = []

    for sex in ["Female", "Male"]:

        sex_df = df[
            df["Sex_clean"] == sex
        ].copy()

        subgroups = (
            sex_df["Subgroup_clean"]
            .dropna()
            .unique()
            .tolist()
        )

        if subgroup_type == "IMD quintile":

            subgroups = sorted(
                subgroups,
                key=lambda x: IMD_ORDER.get(
                    x,
                    999
                )
            )

        else:

            subgroups = sorted(
                subgroups,
                key=lambda x: ETHNICITY_ORDER.get(
                    x,
                    999
                )
            )

        first_sex_row = True

        for subgroup in subgroups:

            sg = sex_df[
                sex_df["Subgroup_clean"] == subgroup
            ].copy()

            # ----------------------------------------
            # N
            # ----------------------------------------

            n_values = (
                pd.to_numeric(
                    sg["n"],
                    errors="coerce"
                )
                .dropna()
                .unique()
            )

            if len(n_values) == 0:
                n_text = ""

            else:
                n_text = f"{int(max(n_values)):,}"

            # ----------------------------------------
            # Outcome cells
            # ----------------------------------------

            cells = {}

            for outcome in OUTCOMES:

                og = sg[
                    sg["Outcome_clean"] == outcome
                ]

                if og.empty:
                    cells[outcome] = "N/A"

                else:
                    cells[outcome] = (
                        format_performance(
                            og.iloc[0]
                        )
                    )

            rows.append([
                sex if first_sex_row else "",
                subgroup,
                n_text,
                cells["All-cause mortality"],
                cells["CVD mortality"],
                cells["Cancer mortality"],
            ])

            first_sex_row = False

    return rows


# ============================================================
# BUILD TABLE
# ============================================================

panel_a_rows = build_panel(
    "IMD quintile"
)

panel_b_rows = build_panel(
    "Ethnicity"
)

columns = [
    "Sex",
    "Subgroup",
    "N",
    "All-cause mortality",
    "CVD mortality",
    "Cancer mortality",
]


# ============================================================
# PRINT FOR QC
# ============================================================

print("\nPANEL A — IMD")
print(
    pd.DataFrame(
        panel_a_rows,
        columns=columns
    ).to_string(index=False)
)

print("\nPANEL B — ETHNICITY")
print(
    pd.DataFrame(
        panel_b_rows,
        columns=columns
    ).to_string(index=False)
)


# ============================================================
# SAVE CSV VERSIONS
# ============================================================

pd.DataFrame(
    panel_a_rows,
    columns=columns
).to_csv(
    "Table4_PanelA_IMD.csv",
    index=False
)

pd.DataFrame(
    panel_b_rows,
    columns=columns
).to_csv(
    "Table4_PanelB_ethnicity.csv",
    index=False
)


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(16, 8.3),
    facecolor="white"
)

ax1 = fig.add_axes(
    [0.02, 0.57, 0.96, 0.34]
)

ax2 = fig.add_axes(
    [0.02, 0.06, 0.96, 0.42]
)

ax1.axis("off")
ax2.axis("off")


# ============================================================
# TITLES
# ============================================================

fig.suptitle(
    "Table 4. XGBoost Model 4 performance across IMD and ethnicity subgroups",
    fontsize=11,
    fontweight="bold",
    x=0.02,
    ha="left",
    y=0.985
)

ax1.text(
    0,
    1.06,
    "Panel A. IMD quintile",
    fontsize=10,
    fontweight="bold",
    ha="left",
    va="bottom",
    transform=ax1.transAxes
)

ax2.text(
    0,
    1.06,
    "Panel B. Ethnicity",
    fontsize=10,
    fontweight="bold",
    ha="left",
    va="bottom",
    transform=ax2.transAxes
)


# ============================================================
# TABLES
# ============================================================

col_widths = [
    0.08,
    0.14,
    0.08,
    0.235,
    0.235,
    0.235,
]

table_a = ax1.table(
    cellText=panel_a_rows,
    colLabels=columns,
    cellLoc="left",
    colLoc="left",
    bbox=[0, 0, 1, 0.95],
    colWidths=col_widths
)

table_b = ax2.table(
    cellText=panel_b_rows,
    colLabels=columns,
    cellLoc="left",
    colLoc="left",
    bbox=[0, 0, 1, 0.95],
    colWidths=col_widths
)


# ============================================================
# STYLE
# ============================================================

def female_rows(rows):

    out = []
    current_sex = None

    for i, row in enumerate(
        rows,
        start=1
    ):

        if row[0] != "":
            current_sex = row[0]

        if current_sex == "Female":
            out.append(i)

    return out


def style_table(table, rows, font_size=7.8):

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    cells = table.get_celld()

    female_row_set = set(
        female_rows(rows)
    )

    for (row, col), cell in cells.items():

        cell.set_edgecolor(
            "#B7B7B7"
        )

        cell.set_linewidth(
            0.45
        )

        cell.PAD = 0.014

        cell.get_text().set_ha(
            "left"
        )

        cell.get_text().set_va(
            "center"
        )

        # Header
        if row == 0:

            cell.set_facecolor(
                "#BFBFBF"
            )

            cell.get_text().set_fontweight(
                "bold"
            )

        # Female rows
        elif row in female_row_set:

            cell.set_facecolor(
                "#E6E6E6"
            )

        # Male rows
        else:

            cell.set_facecolor(
                "white"
            )

        # Bold only Female / Male labels
        if (
            col == 0
            and row > 0
            and cell.get_text()
            .get_text()
            .strip()
        ):

            cell.get_text().set_fontweight(
                "bold"
            )


style_table(
    table_a,
    panel_a_rows,
    font_size=7.9
)

style_table(
    table_b,
    panel_b_rows,
    font_size=7.7
)


# ============================================================
# FOOTNOTE
# ============================================================

fig.text(
    0.02,
    0.012,
    (
        "Values are AUC (95% practice-bootstrap CI) and observed-to-expected (O/E) ratio. "
        "N/A indicates that performance was not estimated because the subgroup-outcome "
        "analysis did not meet the prespecified minimum event requirement. "
        "IMD = Index of Multiple Deprivation; CVD = cardiovascular disease."
    ),
    fontsize=7.3,
    ha="left",
    va="bottom"
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    "Table4_Model4_IMD_ethnicity.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    "Table4_Model4_IMD_ethnicity.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.close()


print("\nDONE")
print("Saved:")
print("  Table4_Model4_IMD_ethnicity.png")
print("  Table4_Model4_IMD_ethnicity.pdf")
print("  Table4_PanelA_IMD.csv")
print("  Table4_PanelB_ethnicity.csv")
