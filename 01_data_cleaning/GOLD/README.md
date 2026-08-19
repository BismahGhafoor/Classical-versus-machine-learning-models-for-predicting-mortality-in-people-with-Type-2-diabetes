# CPRD GOLD data-preparation pipeline

This folder converts raw CPRD GOLD extracts into a patient-level baseline dataset containing demographics, smoking, BMI, blood pressure and laboratory predictors, plus a separate medication branch.

## Core clinical / biomarker branch

Recommended dependency order:

```text
1.py
  -> filtered_diabetes_codes.txt
        |
        v
2.py
  -> Cleaned_GOLD_Extract_Clinical_*.txt
  -> full Clinical key chunks used for Additional-date recovery
        |
        v
3.py
  -> gold_baseline_grouped_df_NoNA.txt
        |
        v
4.py
  -> Enriched_baseline_with_demographics.txt
        |
        +-------------------+
        |                   |
        v                   v
5.py                 raw Clinical/Additional/HES
  -> Clinical_SmokingStatus_all.txt.gz
        |                   |
        +---------> 6.py <--+
                     -> Cleaned_Patient_Smoking_BMI_BP_Data_3YEAR.txt
                                   |
raw GOLD Test extracts -> 7.py     |
                         -> Test_entities_all.txt.gz
                                   |
                                   v
                                 8.py
                                   -> extracted_lab_data_1YEAR.txt
```

### `1.py` — prepare the GOLD diabetes codelist

**Input:** `00_codelists/diabetes/GOLD_final.txt` (the code currently points to its original shared ALICE location).

The script detects the code column, coerces `type` to numeric, remaps type `0` to type `2`, retains diabetes types 1 and 2, harmonises terminology naming and writes `filtered_diabetes_codes.txt`.

### `2.py` — filter/append raw GOLD extracts

Reads raw GOLD Clinical, Therapy and Test ZIPs. In the final configuration:

- `filter_clinical = True`: Clinical rows are filtered using `filtered_diabetes_codes.txt` for cohort/index-date construction.
- `filter_therapy = False` and `filter_test = False`: the therapy/test codelist variables in this script are not active filters.
- A second slim full-Clinical key output (`patid`, `adid`, `enttype`, `eventdate`) is created so that dates can later be recovered for records stored in the Additional files.

The dependency needed by `3.py` is the set of `Cleaned_GOLD_Extract_Clinical_*.txt` chunks.

### `3.py` — build the GOLD baseline cohort and index date

Combines the diabetes-filtered Clinical chunks, removes unusable diagnosis dates, maps diagnosis codes back to diabetes type and retains types 1/2. It then derives one patient-level baseline row by taking the **earliest qualifying diabetes diagnosis date** as `indexdate`.

Key output: `gold_baseline_grouped_df_NoNA.txt`. Separate Type 1 and Type 2 files are also written for QC/inspection.

### `4.py` — enrich the baseline with demographics

Starts from the grouped baseline and adds patient/practice-linked variables including sex/gender, year of birth and practice information. Ethnicity is obtained with **HES as the primary source** and CPRD Clinical ethnicity as fallback using the `Ethn` sheet of `GOLD_Codes_FZ.xlsx`. IMD and ONS death information are also joined.

Output: `Enriched_baseline_with_demographics.txt`.

### `5.py` — extract GOLD smoking records

Scans raw GOLD Clinical data for smoking medcodes from the `Smok` sheet of `GOLD_Codes_FZ.xlsx`, cleans event dates and retains the smoking records needed by `6.py`.

Output: `Clinical_SmokingStatus_all.txt.gz`.

### `6.py` — derive smoking, BMI and blood pressure

Combines the demographic baseline with information from Clinical, Additional and HES sources. It recovers dates for Additional records through the Clinical key linkage created in `2.py`, then selects baseline risk-factor measurements relative to `indexdate`.

Current configuration uses a **1,095-day (3-year) lookback**. BMI gives priority to recorded BMI and uses calculated BMI from weight/height as fallback. The active final BP section derives systolic BP; older paired SBP/DBP logic remains in the script but is not the final active branch.

Output: `Cleaned_Patient_Smoking_BMI_BP_Data_3YEAR.txt`.

> **Missing repository dependency:** `6.py` imports functions from a local `helper_functions.py`, but that file is not present in the current curated repository. The script will not start in a fresh clone until that module is added or the imports are refactored.

### `7.py` — extract raw laboratory Test records

Streams the raw GOLD Test extracts and retains the fields required for downstream laboratory derivation (`patid`, date, entity type and data fields), while cleaning implausible/unparseable dates.

Output: `Test_entities_all.txt.gz`.

### `8.py` — derive laboratory predictors

Merges the patient-level output from `6.py` with Test records from `7.py`. The active window is **365 days before/on index date**. It derives/cleans lipid measures and HbA1c, converts units using the external `SUM.txt` unit lookup, and constructs total cholesterol using a panel-first calculation with recorded total cholesterol as fallback.

Output: `extracted_lab_data_1YEAR.txt`. This is the final GOLD core input to `03_censoring_and_qc/final_cleaning_v3.py`.

> `8.py` also imports the missing local `helper_functions.py` module noted above.

## GOLD medication branch

```text
2.py clinical cohort
       |
       v
10.py -> Cleaned_GOLD_Therapy_*.txt
       |
       +---- Combined_GOLD_Aurum_recoded.txt (for index dates)
       |                from 03_censoring_and_qc/imd_eth_groups.py
       v
11.py -> GOLD_medication_wide.csv
```

### `10.py` — restrict Therapy records to cohort patients

Reads GOLD Therapy ZIPs and keeps prescriptions for patients appearing in the filtered clinical cohort. It writes `Cleaned_GOLD_Therapy_*.txt` files.

**Path warning:** the checked-in `10.py` looks for `Cleaned_GOLD_Extract_Clinical_*.txt` in `/scratch/alice/b/bg205/DataCleaning_Gold_v2`, whereas the checked-in `2.py` writes its outputs from `/scratch/alice/b/bg205/28_02_GOLD`. Before a fresh rerun, confirm/correct this path relationship rather than assuming the files are co-located.

### `11.py` — map prescriptions to harmonised medication classes

Uses `GOLD_all_medication_lookup_stage3.csv` to map GOLD product codes to medication classes. For each patient/class, `prescribed = 1` when at least one matching prescription occurs on or before index date. It writes long, wide and QC files; the downstream master-data merge uses:

- `GOLD_medication_wide.csv`

**Cross-folder dependency:** `11.py` requires `Combined_GOLD_Aurum_recoded.txt` for cohort/index-date information, so it is run only after the early combination/recode stage in `03_censoring_and_qc/`.

**Path warning:** the medication script currently points to `/scratch/alice/b/bg205/16_02_26/Combined_GOLD_Aurum_recoded.txt`, while `imd_eth_groups.py` writes the curated output under `/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/`. Confirm these refer to the intended same dataset before rerunning.

## External inputs not stored in GitHub

Raw CPRD GOLD extracts, HES/ONS linked files, patient/practice files and the CPRD/unit lookup `SUM.txt` are not included. Absolute ALICE paths therefore need configuration for any new environment.
