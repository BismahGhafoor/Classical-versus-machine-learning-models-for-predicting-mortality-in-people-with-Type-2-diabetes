# Study codelists

This directory contains the study-specific code lists and mappings used by the data-cleaning and feature-extraction scripts. These files define *what counts as* diabetes, smoking status, ethnicity, selected biomarkers, medication classes, and CPRD-recorded CKD/hypertension.

> **Important:** these are definitions, not patient-level study data. Some files originated from shared project resources. Confirm redistribution/provenance permissions before making the repository public.

## Directory contents

### `diabetes/`
- `GOLD_final.txt` — diabetes codes used by `01_data_cleaning/GOLD/1.py`.
- `aurum_final.txt` — diabetes codes used by `01_data_cleaning/AURUM/1.py`.

Both source files contain a `type` field. The corresponding `1.py` scripts coerce this field to numeric, remap `0 -> 2`, retain types `1` and `2`, and produce the filtered diabetes codelist used to identify the cohort and assign diabetes type.

### `gold_codes/`
- `GOLD_Codes_FZ.xlsx` — workbook used by the GOLD pipeline.
  - `Ethn` sheet: CPRD ethnicity fallback in `GOLD/4.py`.
  - `Smok` sheet: smoking definitions used by `GOLD/5.py` and `GOLD/6.py`.

Only these sheets are required by the current curated pipeline even though the workbook contains additional sheets.

### `aurum_ethnicity/`
- `Black.csv`
- `Missing.csv`
- `Other_Mixed.csv`
- `South_Asian.csv`
- `White.csv`

Used by `AURUM/4.py` to map Aurum `medcodeid` values to broad ethnicity groups when HES ethnicity is unavailable. HES ethnicity is given priority; CPRD observation-based ethnicity is used as fallback.

### `aurum_smoking/`
- `Current_smoker.csv`
- `Ex-smoker.csv`
- `Never_smoked.csv`

Used by `AURUM/5a.py` to find smoking-related Observation records and by `AURUM/8_bmi_debugged.py` to assign smoking status (`Yes`, `Ex`, `No`).

### `aurum_biomarkers/`
- `BMI_-_final.csv`
- `Weight_final.csv`
- `Height_-_final.csv`
- `SBP_final.csv`
- `DBP_final.csv`
- `modified_LRWE_Lilly_Aurum_medcodeid_clinical biomarkers.xlsx`
- `Codelist_Total_Cholesterol.txt`

The five individual CSV files are used by `AURUM/7a_date.py` to extract BMI, weight, height, systolic BP and diastolic BP Observation records. The Excel workbook is used by `AURUM/9a.py` for HbA1c, HDL, LDL and triglycerides. Total cholesterol is supplied separately in `Codelist_Total_Cholesterol.txt`.

### `medications/`
- `GOLD_all_medication_lookup_stage3.csv`
- `AURUM_all_medication_lookup_stage3.csv`

Used by `GOLD/11.py` and `AURUM/13.py`. They map GOLD `prodcode` / Aurum `productcodeid` values to harmonised medication classes. The medication scripts then create patient-level binary indicators for prescriptions on or before the patient's index date.

### `comorbidities/`
- `LRWE_Lilly_GOLD_medcode_co-morbidities.xlsx` — the final script uses the `CKD_final` and `HTN_final` sheets for GOLD.
- `AURUM_disease_codes.txt` — Aurum CKD definition used by `02_comorbidity_extraction/comorbidities.py`.
- `Final - codes and terms.txt` — Aurum hypertension definition used by the same script.

CVD and cancer definitions for HES are **not external files**: their ICD-10 rules are coded directly inside `comorbidities.py`. Cause-of-death ICD-10 rules are likewise coded directly in `03_censoring_and_qc/cause_of_death.py`.

## External lookup resources not included here

The pipeline also uses resources that are dependencies but are not study codelists and should not be copied into a public repository without checking their licence/redistribution terms:

- GOLD unit lookup `SUM.txt`, used by `GOLD/8.py`.
- CPRD Aurum lookup archive `202205_Lookups_CPRDAurum.zip` (specifically `gender.txt`), used by `AURUM/4.py`.
- Raw CPRD, HES, ONS and linked patient-level files.

## Reproducibility note

The scripts currently contain ALICE-specific absolute paths to the original locations of these codelists. Placing copies in `00_codelists/` documents the exact definitions used, but the scripts must be reconfigured to point here before the repository is portable outside the original ALICE environment.
