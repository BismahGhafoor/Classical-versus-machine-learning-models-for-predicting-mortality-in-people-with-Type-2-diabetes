# CPRD Aurum data-preparation pipeline

This folder converts raw CPRD Aurum data into a harmonised patient-level baseline dataset and a separate patient-level medication dataset. Because the raw Aurum Observation/DrugIssue extracts are large, several scripts are designed for SLURM arrays.

## Core clinical / biomarker branch

```text
1.py
 -> filtered_diabetes_AURUM_codes.txt
        |
        v
2c.sh -> 2b.sbatch -> 2.py
 -> Cleaned_AURUM_Observation_*.txt
        |
        v
3.py
 -> aurum_baseline_grouped_df_NoNA.txt
        |
        v
4b.sbatch -> 4.py
 -> Enriched_baseline_with_demographics.txt
        |
        +-------------------------+
        |                         |
smoking branch               BMI/BP record branch
5a.py (per Observation ZIP)  7b.sbatch -> 7a_date.py (per ZIP)
 -> task*.txt.gz              -> biomarker_chunks/*.txt.gz
        |                         |
        v                         |
6.py -> Aurum_smoking_records_all.txt.gz
        |                         |
        +-----------> 8b.sbatch -> 8_bmi_debugged.py
                         -> Enriched_Aurum_with_Biomarkers_3year.txt
                                        |
raw Observation ZIPs -> 9a.py (per ZIP) |
 -> test_chunks for HbA1c/lipids         |
                                        v
                              10b.sbatch -> 10_optimised.py
                                 -> FINAL_Aurum_with_Tests_1year.txt
```

### `1.py` — prepare the Aurum diabetes codelist

Uses `00_codelists/diabetes/aurum_final.txt`, identifies the medcode/medcodeid column, coerces `type`, remaps `0 -> 2`, retains types 1 and 2, and writes `filtered_diabetes_AURUM_codes.txt`.

### `2.py`, `2b.sbatch`, `2c.sh` — filter raw Observation data

`2.py` is the Python worker. Its final configuration filters Aurum Observation rows by the diabetes medcodeid list and writes `Cleaned_AURUM_Observation_<task>.txt` chunks.

`2b.sbatch` is an array-job **template** containing the placeholder `__ARRAY_RANGE_AND_THROTTLE__`. `2c.sh` is the wrapper that replaces that placeholder in temporary job files and submits batches with `sbatch --wait`. In other words, the normal HPC relationship is:

```text
2c.sh -> temporary copy of 2b.sbatch -> 2.py <SLURM_ARRAY_TASK_ID>
```

The DrugIssue filtering option inside `2.py` is disabled in the final configuration; the final medication branch uses `11_drugissue_patids.py` + `12.py` instead.

### `3.py` — construct baseline and index date

Combines the filtered Observation chunks, maps medcodeids to diabetes type, filters to types 1/2 and derives the **earliest qualifying diabetes observation** per patient as `indexdate`.

Key output: `aurum_baseline_grouped_df_NoNA.txt`.

### `4.py` / `4b.sbatch` — add demographics and linked information

`4b.sbatch` launches `4.py`. The Python script enriches the baseline with patient demographics, decodes gender using the CPRD Aurum gender lookup, gives HES ethnicity priority and fills remaining ethnicity using the five CSV codelists in `00_codelists/aurum_ethnicity/`. It also joins IMD and ONS death information.

Output: `Enriched_baseline_with_demographics.txt`.

### `5a.py` + `6.py` — smoking extraction

`5a.py` scans Observation ZIPs for medcodeids appearing in the Current/Ex/Never smoking codelists. It supports per-ZIP task execution and writes `Aurum_Clinical_SmokingStatus_task####.txt.gz` files.

`6.py` then performs the merge expected by the final downstream script and creates:

- `smoking_chunks/Aurum_smoking_records_all.txt.gz`

**Important:** no SLURM launcher for `5a.py` is included in the curated repository. To reproduce the task-chunk route, `5a.py` must be invoked once per Observation ZIP (for example through an array job), followed by `6.py`. The built-in `5a.py merge` mode writes to a different location than the path expected by `8_bmi_debugged.py`, so `6.py` is the documented final merge step.

### `7a_date.py` / `7b.sbatch` — extract BMI, weight, height and BP records

`7a_date.py` is a per-Observation-ZIP worker. It uses the five codelists in `00_codelists/aurum_biomarkers/` and writes gzipped biomarker chunks. It also cleans implausible dates and removes explicitly listed ambiguous codes.

`7b.sbatch` launches `7a_date.py`, but it is a **template** with `__ARRAY_RANGE_AND_THROTTLE__`. Unlike stages 2 and 12, the corresponding wrapper is not present in this curated repository, so the placeholder must be replaced or an equivalent array submission supplied before use.

### `8_bmi_debugged.py` / `8b.sbatch` — derive smoking, BMI and BP at baseline

`8b.sbatch` launches `8_bmi_debugged.py`. The script starts from `Enriched_baseline_with_demographics.txt`, combines the smoking file and biomarker chunks, and derives patient-level baseline risk factors.

Active logic includes:

- CPRD smoking within the pre-index window, with HES current-smoking codes as fallback when CPRD smoking is absent.
- BMI with recorded BMI preferred and calculated BMI (weight/height) as fallback.
- Current `WINDOW_DAYS = 1095` (3 years) for the general baseline window.
- The final active blood-pressure section is **SBP-only**; older paired SBP/DBP code remains in the script for history but is not the final active branch.

Output: `Enriched_Aurum_with_Biomarkers_3year.txt`.

### `9a.py` — extract HbA1c and lipid laboratory records

Per Observation ZIP, extracts HbA1c, HDL, LDL and triglycerides from the relevant sheets of the Aurum biomarker workbook and total cholesterol from `Codelist_Total_Cholesterol.txt`. It applies unit conversions (including HbA1c conversion and lipid unit handling) and writes gzipped variable-specific chunks under `test_chunks/`.

No `.sbatch` launcher for this worker is included in the curated repository, so a per-ZIP loop/array must be supplied when reproducing it.

### `10_optimised.py` / `10b.sbatch` — merge laboratory predictors into final Aurum core dataset

`10b.sbatch` launches `10_optimised.py`. The script reads `Enriched_Aurum_with_Biomarkers_3year.txt` and the laboratory chunks from `9a.py`. The active laboratory window is `WINDOW_DAYS = 365` (1 year). It selects/derives HDL, LDL, triglycerides, total cholesterol and HbA1c and writes:

- `FINAL_Aurum_with_Tests_1year.txt`

This is the final Aurum core input to `03_censoring_and_qc/final_cleaning_v3.py`.

## Aurum medication branch

```text
filtered Aurum Observation cohort
        |
        v
11_drugissue_patids.py
 -> cohort_patids.txt
        |
        v
12c.sh -> 12b.sbatch -> 12.py
 -> Cleaned_AURUM_DrugIssue_*.txt
        |
        +---- Combined_GOLD_Aurum_recoded.txt (index dates)
        |                from 03_censoring_and_qc/imd_eth_groups.py
        v
13b.sbatch -> 13.py
 -> AURUM_medication_wide.csv
```

### `11_drugissue_patids.py`

Collects unique `patid` values from the filtered Observation chunks and writes `cohort_patids.txt`.

### `12.py`, `12b.sbatch`, `12c.sh`

`12.py` filters raw DrugIssue ZIPs to the cohort patients. `12b.sbatch` is the task template and `12c.sh` replaces its array placeholder and submits the tasks in batches.

Output: `Cleaned_AURUM_DrugIssue_*.txt`.

### `13.py` / `13b.sbatch`

Maps Aurum product codes to harmonised medication classes using `AURUM_all_medication_lookup_stage3.csv`. It uses the combined recoded cohort to obtain index dates and defines a medication class as prescribed when at least one matching DrugIssue occurs on or before index date.

Outputs include long and QC files; the downstream master-data merge uses `AURUM_medication_wide.csv`.

**Path warning:** as checked in, `13.py` points to `/scratch/alice/b/bg205/16_02_26/Combined_GOLD_Aurum_recoded.txt`, while `imd_eth_groups.py` writes the curated recoded file in `/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/`. Confirm/correct this relationship before rerunning.

## External resources not in GitHub

Raw CPRD Aurum Patient, Practice, Observation and DrugIssue extracts, HES/ONS linkage data and the CPRD Aurum lookup archive are intentionally excluded. The checked-in scripts also contain ALICE-specific absolute paths.
