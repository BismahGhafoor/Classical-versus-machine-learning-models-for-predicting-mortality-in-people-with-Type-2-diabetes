# Baseline comorbidity extraction

This stage creates patient-level baseline comorbidity indicators and dates from both CPRD primary-care records and HES hospital diagnoses. It is run **after** the GOLD/Aurum core datasets have been combined and recoded, because `comorbidities.py` needs `Combined_GOLD_Aurum_recoded.txt` to know each patient's database and index date.

## Run order

```text
03_censoring_and_qc/imd_eth_groups.py
        |
        v
Combined_GOLD_Aurum_recoded.txt
        |
        v
comorbidities.sbatch -> comorbidities.py
        |
        v
final_combined_comorbidities.txt
        |
        v
temp_fix.py
        |
        v
final_combined_comorbidities_fixed.txt
        |
        v
03_censoring_and_qc/merge_meds_comorbidites.py
```

## `comorbidities.py`

The script has three extraction sources.

### GOLD CPRD records

Scans raw GOLD Clinical ZIPs and uses `LRWE_Lilly_GOLD_medcode_co-morbidities.xlsx`. The final configuration uses the `CKD_final` and `HTN_final` sheets.

### Aurum CPRD records

Scans raw Aurum Observation ZIPs. The final configuration uses:

- `AURUM_disease_codes.txt` for CKD.
- `Final - codes and terms.txt` for hypertension.

The script still opens a historical Aurum comorbidity Excel workbook, but `AURUM_SHEETS = {}` means no sheets from that workbook are used in the final phenotype definition. This is a vestigial technical dependency that can be refactored later.

### HES diagnoses

HES ICD-10 rules are encoded directly in `comorbidities.py`:

- CVD: `I*`, excluding hypertension prefixes `I10-I15`.
- Hypertension: `I10-I15`.
- CKD: `N18.3`, `N18.4`, `N18.5`, `N18.6`, `N19`, `Z99.2`.
- Any cancer: `C*`.
- Selected cancer subtypes: breast, lung, colorectal, prostate and pancreatic cancer.

Only diagnoses occurring on/before the patient's index date contribute to baseline comorbidity status. Event dates before 1900 are treated as implausible.

### Final combination rules

For CKD and hypertension, CPRD and HES evidence are combined: the binary indicator is positive if either source identifies the condition and the first date is the earliest available source date. CVD and cancer are HES-only in the final specification.

The script writes multiple long/wide/QC files. The main downstream output is:

- `final_combined_comorbidities.txt`

## `comorbidities.sbatch`

SLURM launcher for `comorbidities.py`. It requests 64 GB RAM and activates the ALICE CPRD virtual environment before running the Python script.

## `temp_fix.py`

Applies the final pre-1900 correction to the already-created patient-level comorbidity file without rerunning the full extraction. For every `*_first_date` column it:

1. converts pre-1900 values to missing;
2. recalculates the corresponding binary flag;
3. recalculates duration in days/years from first diagnosis to index date;
4. checks for negative durations.

Output used downstream:

- `final_combined_comorbidities_fixed.txt`

## Important path note

The checked-in `comorbidities.py` points to `/scratch/alice/b/bg205/16_02_26/Combined_GOLD_Aurum_recoded.txt`, whereas the curated `imd_eth_groups.py` writes the recoded dataset under `/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/`. Before rerunning, confirm which path is canonical and update the configuration accordingly.

Patient-level CPRD/HES inputs and generated comorbidity datasets are intentionally not stored in GitHub.
