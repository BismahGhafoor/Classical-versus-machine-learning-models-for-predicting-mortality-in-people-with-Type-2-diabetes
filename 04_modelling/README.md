# Mortality modelling and post-processing

This folder contains the final V3 modelling pipeline plus post-processing of saved predictions. The primary modelling script is `modelling_fixedhybrid.py`; the two other Python scripts **do not retrain models** but reconstruct the saved test set and calculate database/subgroup performance from the stored predictions.

## Intended order

```text
03_censoring_and_qc/
Combined_GOLD_Aurum_with_meds_comorbidities_studyend_cod.txt
        |
        v
modelling_optimised.sbatch
        |
        v
modelling_fixedhybrid.py
        |
        +--> outcome/block metrics, plots and CSVs
        +--> run-level *_all_models*.pkl files
        |
        +-----------------------------+
        |                             |
        v                             v
postprocess_Database_          publication_xgbD_subgroup.sbatch
stratified.sbatch                     |
        |                             v
        v                    publication_xgbD_subgroup.py
postprocess_database_                 |
stratified_performance.py             v
        |                    IMD/ethnicity subgroup tables
        v
GOLD-vs-Aurum performance tables
        |
        v
05_manuscript_figures/
```

## `modelling_fixedhybrid.py`

### Cohort selection and outcomes

The script reads the final COD dataset and can filter by database, sex and diabetes type. It derives 10-year mortality outcomes relative to `indexdate` using registration/practice/death information and the administrative study-end date. Supported outcomes are:

- `death_10y` — all-cause mortality;
- `death_cvd` — CVD mortality using `cod_cvd`;
- `death_cancer` — cancer mortality using `cod_cancer`.

The code's documented convention is that patients leaving follow-up before 10 years are coded as alive in the primary analysis; `complete_fu` is available as a sensitivity restriction.

### Predictor blocks

The pipeline fits nested predictor blocks A-D. The exact lists are defined in the source code and are intentionally incremental:

- **A:** demographic/core baseline predictors;
- **B:** adds behavioural/biomarker predictors;
- **C:** adds comorbidities;
- **D:** adds medications.

Cause-specific outcomes additionally exclude predictors that would directly encode the corresponding cause information where specified by the script.

### Train/test design

Patients are split at the **practice level**, not independently by patient. The held-out test split is 20%. When GOLD and Aurum are combined, the split routine stratifies the practice split by database so both sources are represented. Hyperparameter tuning uses 5-fold `GroupKFold` by practice on the development data.

### Missing data

Primary analyses use `MissForest` fitted on the training predictors and then applied to the held-out test predictors. The outcome is never passed into the imputation matrix. The script also supports `complete_case` sensitivity analysis, which drops rows missing required predictors within each block and skips imputation.

### Models

For each block/outcome the pipeline evaluates:

- `LR` — regularised logistic regression;
- `LR_FLEX` — logistic regression with spline transformations for continuous predictors;
- `RF` — random forest;
- `XGB` — XGBoost;
- `HYBRID` — XGBoost-guided feature selection followed by logistic regression.

The final hybrid uses a **capture rule** (`--hybrid_capture`, default 0.90): it chooses the smallest feature subset that captures at least the specified fraction of the achievable cross-validated improvement from the smallest to best subset.

### Tuning and evaluation

The primary/default tuning metric is Brier score (`--tuning_metric brier`), with AUC available as an alternative. The final test evaluation includes discrimination, calibration and overall prediction error measures, decision-curve analysis and paired model comparisons. Confidence intervals are obtained by **practice-cluster bootstrap**; the full-run default is 2,000 resamples.

### Important outputs

The script creates per-outcome directories plus a `_run_level/` directory. Run-level `*_all_models*.pkl` files store the configuration, practice split, model outputs and saved predictions needed by the post-processing and manuscript-figure scripts.

## `modelling_optimised.sbatch`

SLURM array launcher for `modelling_fixedhybrid.py`.

### Important checked-in inconsistencies — verify before rerunning

The current file should **not be treated as a blindly runnable canonical Type 2 command** without checking two lines:

The final manuscript modelling run used Type 2 diabetes (`--diabetes_type 2`). This was verified from the SLURM execution logs and the resulting `Combined_male_type_2` and `Combined_female_type_2` output directories.

The script defines 18 possible task combinations covering the primary analysis, complete-case sensitivity analysis, and complete-follow-up (`complete_fu`) sensitivity analysis. The final submitted SLURM array uses `#SBATCH --array=0-11`, which executes 12 tasks: six primary analyses (male/female × three mortality outcomes) and six corresponding complete-case sensitivity analyses. The six `complete_fu` tasks remain defined in the script but were not submitted as part of this final array.

## `postprocess_database_stratified_performance.py`

Does **not** refit models. It loads a run-level pickle and deterministically reconstructs the same held-out test set using the saved practice split. It asserts alignment against the stored test practices/sample sizes/metrics before writing results. Saved predictions are then evaluated separately in GOLD and Aurum.

Outputs include database-stratified:

- performance with CIs;
- formatted performance tables;
- threshold metrics;
- counts;
- alignment-verification tables.

The post-processing can bootstrap by practice (recommended) and optionally bootstrap calibration metrics.

## `postprocess_Database_stratified.sbatch`

Maps 12 tasks across female/male x three outcomes x primary/complete-case analyses. It searches for the relevant Type 2 run-level pickle and refuses to proceed if zero or multiple files match. Its default `N_BOOTSTRAPS=10` is a smoke-test default; the script comments indicate using 2,000 for final output.

## `publication_xgbD_subgroup.py`

Publication-focused subgroup analysis using **saved predictions only**. The final launcher requests:

- XGBoost;
- Block D;
- IMD quintile and ethnicity subgroups;
- practice-cluster bootstrap CIs;
- minimum subgroup N/event thresholds.

Metrics include AUC, PR-AUC, Brier score and observed/expected ratio. The script verifies that reconstructed predictions align with the saved test set before calculating subgroup results.

## `publication_xgbD_subgroup.sbatch`

Maps six tasks across female/male x three outcomes for the primary Type 2 analysis. Like the database post-processing launcher, the checked-in default is a small smoke-test bootstrap count (`10`); set `N_BOOTSTRAPS=2000` for the final publication run used by `05_manuscript_figures/table4.py`.

## Environment

The code requires Python packages including pandas, NumPy, scikit-learn, XGBoost, matplotlib and `missforest`. Paths and Conda environments in the checked-in `.sbatch` files are ALICE-specific.
