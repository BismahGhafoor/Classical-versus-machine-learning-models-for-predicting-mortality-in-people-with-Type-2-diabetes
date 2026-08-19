# Data cleaning and baseline predictor extraction

This directory contains the separate CPRD GOLD and CPRD Aurum cleaning pipelines. The two databases are processed independently because their raw file structures, terminology identifiers and field names differ. They are only combined after each database has been converted to a broadly harmonised patient-level dataset.

## High-level flow

```text
CPRD GOLD raw extracts                         CPRD Aurum raw extracts
        |                                             |
        v                                             v
  GOLD/1.py ... GOLD/8.py                     AURUM/1.py ... AURUM/10_optimised.py
        |                                             |
        v                                             v
extracted_lab_data_1YEAR.txt                  FINAL_Aurum_with_Tests_1year.txt
        \                                             /
         \                                           /
          -> 03_censoring_and_qc/final_cleaning_v3.py
                         |
                         v
               GOLD_with_censoring.txt
               Aurum_with_censoring.txt
                         |
                         v
                      combine.py
                         |
                         v
                 imd_eth_groups.py
                         |
                         v
             Combined_GOLD_Aurum_recoded.txt
```

The recoded combined cohort is then used to build the medication and comorbidity predictors before the final master dataset is assembled.

## Important: repository folder numbers are not a strict one-pass run order

Medication extraction in this folder and comorbidity extraction in `02_comorbidity_extraction/` require the combined recoded cohort produced by the **early** scripts in `03_censoring_and_qc/`.

The practical order is therefore:

1. Run the GOLD core clinical/biomarker branch through `GOLD/8.py`.
2. Run the Aurum core clinical/biomarker branch through `AURUM/10_optimised.py`.
3. Run `03_censoring_and_qc/final_cleaning_v3.py`, `combine.py`, then `imd_eth_groups.py`.
4. Build GOLD and Aurum medication outputs.
5. Run `02_comorbidity_extraction/`.
6. Return to `03_censoring_and_qc/` for medication/comorbidity merge, study-end restriction, QC and cause-of-death derivation.

See the READMEs inside `GOLD/`, `AURUM/`, `02_comorbidity_extraction/` and `03_censoring_and_qc/` for exact dependencies.

## Main outputs passed to the next stage

- GOLD: `extracted_lab_data_1YEAR.txt`
- Aurum: `FINAL_Aurum_with_Tests_1year.txt`
- GOLD medications: `GOLD_medication_wide.csv`
- Aurum medications: `AURUM_medication_wide.csv`

Generated patient-level files are deliberately excluded from GitHub.
