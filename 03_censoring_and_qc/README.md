# Database harmonisation, final cohort assembly and QC

This folder bridges the separately cleaned GOLD/Aurum datasets and the final modelling dataset. It is best understood as **two phases** separated by medication and comorbidity extraction.

## True pipeline order

```text
PHASE A — create the harmonised base cohort

GOLD extracted_lab_data_1YEAR.txt        Aurum FINAL_Aurum_with_Tests_1year.txt
              \                              /
               -> final_cleaning_v3.py <-
                        |
             GOLD_with_censoring.txt
             Aurum_with_censoring.txt
                        |
                        v
                    combine.py
                        |
                        v
              Combined_GOLD_Aurum.txt
                        |
                        v
                imd_eth_groups.py
                        |
                        v
          Combined_GOLD_Aurum_recoded.txt
                /                 \
               /                   \
      medication extraction    02_comorbidity_extraction
               \                   /
                \                 /

PHASE B — assemble the final analytical cohort

GOLD_medication_wide.csv + AURUM_medication_wide.csv
+ final_combined_comorbidities_fixed.txt
                        |
                        v
             merge_meds_comorbidites.py
                        |
                        v
Combined_GOLD_Aurum_with_meds_comorbidities.txt
                        |
                        v
                studyendfilter.py
                        |
                        v
..._studyend_clean.txt
                        |
                        v
                    QCfix.py
          (modifies/overwrites same file)
                        |
                        v
                cause_of_death.py
                        |
                        v
Combined_GOLD_Aurum_with_meds_comorbidities_studyend_cod.txt
                        |
                        v
                  04_modelling/
```

## `final_cleaning_v3.py`

Takes the final GOLD and Aurum core patient-level datasets and adds raw CPRD registration/practice information needed for censoring/follow-up.

- GOLD: adds patient `regend`/`tod` and practice `lcd` to `extracted_lab_data_1YEAR.txt`.
- Aurum: adds patient `regenddate` and practice `lcd` to `FINAL_Aurum_with_Tests_1year.txt`.

Outputs:

- `GOLD_with_censoring.txt`
- `Aurum_with_censoring.txt`

Despite the filename, this script primarily **adds the source fields needed to derive censoring later**; it does not by itself create the final 10-year modelling outcomes.

## `combine.py`

Reads the two database-specific outputs, adds a `database` identifier (`GOLD`/`AURUM`), harmonises selected column names (for example Aurum `regenddate -> regend` and triglyceride naming) and concatenates the rows into:

- `Combined_GOLD_Aurum.txt`

## `imd_eth_groups.py`

Creates harmonised modelling categories:

- converts IMD deciles (`e2019_imd_10`) to quintiles using `ceil(decile/2)`;
- collapses detailed ethnicity labels into broader groups including South Asian, Black and Mixed/Other.

Output:

- `Combined_GOLD_Aurum_recoded.txt`

This file is a central cross-folder dependency for medication and comorbidity extraction.

## `merge_meds_comorbidites.py`

Runs only after both medication branches and `02_comorbidity_extraction/temp_fix.py` are complete. It:

1. loads the recoded combined master cohort;
2. combines GOLD/Aurum patient-level medication-wide files;
3. adds `med_` prefixes and left-merges by `database + patid`;
4. fills absent binary medication indicators with zero;
5. loads `final_combined_comorbidities_fixed.txt`;
6. prefixes comorbidity fields with `comorb_` and left-merges them;
7. fills missing binary comorbidity flags with zero while preserving missing dates;
8. checks that merges do **not change the master row count**.

Output:

- `Combined_GOLD_Aurum_with_meds_comorbidities.txt`

## `studyendfilter.py`

Applies the administrative study-end restriction:

- `STUDY_END_DATE = 2021-03-31`
- retains patients with `indexdate <= study_end`.

Main output:

- `Combined_GOLD_Aurum_with_meds_comorbidities_studyend_clean.txt`

The script also creates study-end-filtered copies of the medication and comorbidity source files for QC/consistency checks. These copies are not the file passed to modelling.

## `QCfix.py`

Applies three final data corrections to `...studyend_clean.txt`:

1. harmonises gender labels `Male/Female -> M/F`;
2. removes patients whose ONS death date is before index date;
3. removes patients with implausible year of birth `<1900`.

The current code reads `...studyend_clean.txt` and, because of its filename replacement expression, writes back to **the same path**, so it effectively overwrites the clean file. Run it before `cause_of_death.py`.

## `cause_of_death.py`

Reads GOLD and Aurum ONS death files, classifies the **underlying cause of death** using ICD-10 and appends three binary fields:

- `cod_cvd`: `I*` excluding `I10-I15`;
- `cod_cancer`: `C*`;
- `cod_ckd`: `N18.3-N18.6`, `N19`, `Z99.2`.

It writes COD QC files plus the final analytical dataset:

- `Combined_GOLD_Aurum_with_meds_comorbidities_studyend_cod.txt`

This is the `--data_path` consumed by the final modelling pipeline.

## Verified execution history

The final reconstruction was verified from the ALICE shell history: on 19 May 2026 the base cleaning/combination/recode/merge and study-end/QC scripts were rerun in sequence, followed by `cause_of_death.py` on 23 May 2026. The older `..._studyend.txt` present in the working directory predates that final reconstruction and is not part of the final May pipeline.

## Data policy

All `.txt` master datasets and linked patient-level source files are deliberately excluded from GitHub. Only code and approved codelists should be version controlled.
