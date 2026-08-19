import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/"
    "Combined_GOLD_Aurum_with_meds_comorbidities_studyend_cod.txt"
)


# ============================================================
# READ HEADER ONLY
# We only need the final variable names, not the full dataset.
# ============================================================

print("Reading final cleaned dataset header...")

columns = pd.read_csv(
    DATA_PATH,
    sep="\t",
    nrows=0
).columns.tolist()


# ============================================================
# IDENTIFY FINAL PREDICTORS EXACTLY AS MODEL PIPELINE DOES
# ============================================================

med_vars = sorted([
    c for c in columns
    if c.startswith("med_")
    and c.endswith("_prescribed")
])

comorb_binary = sorted([
    c for c in columns
    if c.endswith("_bin")
])

comorb_duration = sorted([
    c for c in columns
    if c.endswith("_duration_years")
    and "comorb" in c
])

print(f"Medication predictors: {len(med_vars)}")
print(f"Comorbidity binary predictors: {len(comorb_binary)}")
print(f"Comorbidity duration predictors: {len(comorb_duration)}")


# ============================================================
# HUMAN-READABLE LABELS
# ============================================================

SPECIAL_LABELS = {
    "age_at_index": "Age at index",
    "gen_ethnicity": "Ethnicity",
    "imd_quintile": "IMD quintile",
    "smoking_status": "Smoking status",
    "bmi": "Body mass index (BMI)",
    "systolic": "Systolic blood pressure",
    "hba1c_perc": "HbA1c",
    "tot_chol": "Total cholesterol",
    "hdl": "HDL cholesterol",
    "ldl": "LDL cholesterol",
    "trigly": "Triglycerides",
}


def humanise_variable(var):

    if var in SPECIAL_LABELS:
        return SPECIAL_LABELS[var]

    label = var

    # Remove technical prefixes/suffixes
    label = re.sub(r"^med_", "", label)
    label = re.sub(r"^comorb_", "", label)

    label = re.sub(
        r"_prescribed$",
        "",
        label
    )

    label = re.sub(
        r"_duration_years$",
        "",
        label
    )

    label = re.sub(
        r"_bin$",
        "",
        label
    )

    label = label.replace("_", " ")

    # Common abbreviations
    replacements = {
        "cvd": "CVD",
        "ckd": "CKD",
        "htn": "Hypertension",
        "sglt2": "SGLT2 inhibitors",
        "glp1": "GLP-1 receptor agonists",
        "bp": "Blood pressure",
    }

    words = []

    for word in label.split():

        if word.lower() in replacements:
            words.append(
                replacements[word.lower()]
            )
        else:
            words.append(word)

    label = " ".join(words)

    # Avoid changing recognised abbreviations
    if not any(
        label.startswith(x)
        for x in [
            "CVD",
            "CKD",
            "SGLT2",
            "GLP-1"
        ]
    ):
        label = (
            label[0].upper()
            + label[1:]
            if label
            else label
        )

    return label


# ============================================================
# CODING / DEFINITIONS
# ============================================================

def definition_for(var, group):

    definitions = {

        "age_at_index":
            "Age in years at cohort entry, derived from index year and year of birth.",

        "gen_ethnicity":
            "Categorical ethnicity: White, Black, South Asian, Mixed/Other or Unknown; "
            "HES ethnicity prioritised with CPRD used as fallback.",

        "imd_quintile":
            "Index of Multiple Deprivation 2019 quintile, coded 1–5 and modelled categorically.",

        "smoking_status":
            "Baseline smoking status derived from the most relevant recorded status before or at index.",

        "bmi":
            "BMI (kg/m²); baseline routinely recorded measurement selected using prespecified "
            "validity and nearest-to-index rules.",

        "systolic":
            "Systolic blood pressure (mmHg); baseline measurement selected using prespecified "
            "validity and nearest-to-index rules.",

        "hba1c_perc":
            "HbA1c (%); baseline measurement selected using prespecified validity and "
            "nearest-to-index rules.",

        "tot_chol":
            "Total cholesterol (mmol/L); baseline measurement selected using prespecified "
            "validity and nearest-to-index rules.",

        "hdl":
            "HDL cholesterol (mmol/L); baseline measurement selected using prespecified "
            "validity and nearest-to-index rules.",

        "ldl":
            "LDL cholesterol (mmol/L); baseline measurement selected using prespecified "
            "validity and nearest-to-index rules.",

        "trigly":
            "Triglycerides (mmol/L); baseline measurement selected using prespecified "
            "validity and nearest-to-index rules.",
    }

    if var in definitions:
        return definitions[var]

    if group == "Medication":
        return (
            "Binary indicator of ≥1 prescription on or before the index date."
        )

    if group == "Comorbidity":
        if var.endswith("_duration_years"):
            return (
                "Duration in years from first recorded diagnosis to index; "
                "0 where no prior diagnosis was recorded."
            )

        return (
            "Binary indicator of a recorded diagnosis before or at the index date."
        )

    return ""


# ============================================================
# MODEL MEMBERSHIP
# ============================================================

def membership(group):

    if group == "Sociodemographic":
        return ["✓", "✓", "✓", "✓"]

    if group == "Behaviour / biomarker":
        return ["", "✓", "✓", "✓"]

    if group == "Medication":
        return ["", "", "✓", "✓"]

    if group == "Comorbidity":
        return ["", "", "", "✓"]

    return ["", "", "", ""]


# ============================================================
# CAUSE-SPECIFIC EXCLUSION NOTE
# ============================================================

def exclusion_note(var):

    # Mirrors the final pipeline:
    # CVD-labelled comorbidity predictors excluded for CVD mortality;
    # cancer-labelled comorbidity predictors excluded for cancer mortality.

    if (
        var.startswith("comorb_")
        and "cvd" in var.lower()
    ):
        return "Excluded from CVD mortality Models 3–4"

    if (
        var.startswith("comorb_")
        and "cancer" in var.lower()
    ):
        return "Excluded from cancer mortality Models 3–4"

    return ""


# ============================================================
# BUILD TABLE ROWS
# ============================================================

rows = []


def add_row(group, var):

    m1, m2, m3, m4 = membership(group)

    rows.append([
        group,
        humanise_variable(var),
        definition_for(var, group),
        m1,
        m2,
        m3,
        m4,
        exclusion_note(var),
    ])


# Model 1 variables
for var in [
    "age_at_index",
    "gen_ethnicity",
    "imd_quintile"
]:
    add_row(
        "Sociodemographic",
        var
    )


# Model 2 additions
for var in [
    "smoking_status",
    "bmi",
    "systolic",
    "hba1c_perc",
    "tot_chol",
    "hdl",
    "ldl",
    "trigly"
]:
    add_row(
        "Behaviour / biomarker",
        var
    )


# Model 3 additions
for var in med_vars:
    add_row(
        "Medication",
        var
    )


# Model 4 additions
for var in comorb_binary:
    add_row(
        "Comorbidity",
        var
    )

for var in comorb_duration:
    add_row(
        "Comorbidity",
        var
    )


columns_out = [
    "Predictor group",
    "Predictor",
    "Definition / coding",
    "Model 1",
    "Model 2",
    "Model 3",
    "Model 4",
    "Outcome-specific exclusion",
]


table_df = pd.DataFrame(
    rows,
    columns=columns_out
)


# ============================================================
# QC
# ============================================================

print("\nFINAL SUPPLEMENTARY TABLE S1\n")

print(
    table_df.to_string(
        index=False
    )
)

print("\nPredictors by group:")
print(
    table_df[
        "Predictor group"
    ].value_counts()
)

print(
    f"\nTotal predictor rows: "
    f"{len(table_df)}"
)


# ============================================================
# SAVE CSV
# ============================================================

table_df.to_csv(
    "Supplementary_Table_S1_predictors.csv",
    index=False
)


# ============================================================
# FIGURE SIZE — AUTOMATICALLY SCALE WITH NUMBER OF ROWS
# ============================================================

n_rows = len(table_df)

fig_height = max(
    7,
    0.34 * n_rows + 2.2
)

fig, ax = plt.subplots(
    figsize=(18, fig_height),
    facecolor="white"
)

ax.axis("off")


# ============================================================
# TITLE
# ============================================================

ax.text(
    0,
    1.025,
    (
        "Supplementary Table S1. Predictor definitions "
        "and composition of Models 1–4"
    ),
    fontsize=11,
    fontweight="bold",
    ha="left",
    va="bottom",
    transform=ax.transAxes
)


# ============================================================
# DRAW TABLE
# ============================================================

table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    cellLoc="left",
    colLoc="left",
    bbox=[0, 0.03, 1, 0.96],
    colWidths=[
        0.11,   # group
        0.14,   # predictor
        0.34,   # definition
        0.055,  # M1
        0.055,  # M2
        0.055,  # M3
        0.055,  # M4
        0.18,   # exclusion
    ]
)


# ============================================================
# STYLE — SAME FAMILY AS MAIN TABLES
# ============================================================

table.auto_set_font_size(False)
table.set_fontsize(7.6)

cells = table.get_celld()


# Find group row ranges so groups alternate white / grey
group_order = [
    "Sociodemographic",
    "Behaviour / biomarker",
    "Medication",
    "Comorbidity",
]

group_colours = {
    "Sociodemographic": "#E6E6E6",
    "Behaviour / biomarker": "white",
    "Medication": "#E6E6E6",
    "Comorbidity": "white",
}


for (row, col), cell in cells.items():

    cell.set_edgecolor(
        "#B7B7B7"
    )

    cell.set_linewidth(
        0.45
    )

    cell.PAD = 0.014

    cell.get_text().set_va(
        "center"
    )

    if col in [3, 4, 5, 6]:
        cell.get_text().set_ha(
            "center"
        )
    else:
        cell.get_text().set_ha(
            "left"
        )

    # Header
    if row == 0:

        cell.set_facecolor(
            "#BFBFBF"
        )

        cell.get_text().set_fontweight(
            "bold"
        )

    else:

        group = table_df.iloc[
            row - 1
        ]["Predictor group"]

        cell.set_facecolor(
            group_colours.get(
                group,
                "white"
            )
        )


# ============================================================
# ONLY DISPLAY GROUP NAME ON FIRST ROW OF EACH GROUP
# ============================================================

previous_group = None

for i in range(
    1,
    len(table_df) + 1
):

    group = table_df.iloc[
        i - 1
    ]["Predictor group"]

    if group == previous_group:

        cells[
            (i, 0)
        ].get_text().set_text("")

    else:

        cells[
            (i, 0)
        ].get_text().set_fontweight(
            "bold"
        )

    previous_group = group


# ============================================================
# FOOTNOTE
# ============================================================

fig.text(
    0.01,
    0.005,
    (
        "✓ indicates inclusion in the predictor model. "
        "Sex was omitted from the sex-specific analyses. "
        "Detailed biomarker validity ranges, lookback windows and "
        "measurement-selection rules are provided in the Supplementary Methods. "
        "CVD = cardiovascular disease; HES = Hospital Episode Statistics; "
        "IMD = Index of Multiple Deprivation."
    ),
    fontsize=7.4,
    ha="left",
    va="bottom"
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    "Supplementary_Table_S1_predictors.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    "Supplementary_Table_S1_predictors.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.close()


print("\nDONE")
print("Saved:")
print("  Supplementary_Table_S1_predictors.png")
print("  Supplementary_Table_S1_predictors.pdf")
print("  Supplementary_Table_S1_predictors.csv")
