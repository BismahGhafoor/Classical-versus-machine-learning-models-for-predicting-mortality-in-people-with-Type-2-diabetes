# Manuscript figures and tables

These scripts are **presentation/post-processing scripts**. They do not build the cohort or fit the primary models. They consume the final modelling dataset, saved model pickles and/or CSV outputs produced by `04_modelling/`.

Run them only after the required final modelling and post-processing outputs exist.

## Main manuscript figures

### `figure2.py`

Creates the figure showing change in XGBoost AUC across predictor Models 1-4 for female and male Type 2 diabetes analyses. The values are currently hard-coded in the script rather than read dynamically from result files.

Outputs:

- `Figure2_XGBoost_AUC_across_models.png`
- `Figure2_XGBoost_AUC_across_models.pdf`

### `figure3.py`

Reconstructs the saved held-out test sets using the final COD dataset and the run-level model pickles, verifies the practice split, extracts Model 4 predictions and creates calibration plots.

Outputs:

- `Figure3_calibration_Model4.png`
- `Figure3_calibration_Model4.pdf`

### `figure4.py`

Reads the final primary-analysis decision-curve and model-summary CSVs from the six sex/outcome analysis folders and creates Model 4 decision-curve panels. It explicitly avoids complete-case sensitivity outputs when locating files.

Outputs:

- `Figure4_DCA_Model4.png`
- `Figure4_DCA_Model4.pdf`

## Main manuscript tables

### `table2.py`

Reads the primary model-summary and confidence-interval result files for female/male x all-cause/CVD/cancer analyses and renders the final model-performance table.

Outputs:

- `Table2_final_results.png`
- `Table2_final_results.pdf`

### `table3.py`

Reads paired model-comparison outputs and formats the final comparison of XGBoost Model 4 against logistic regression and random forest.

Outputs:

- `Table3_Model4_model_comparisons.png`
- `Table3_Model4_model_comparisons.pdf`

### `table4.py`

Reads the six final subgroup-performance files created by `publication_xgbD_subgroup.py` for XGBoost Block D, using the **2,000-bootstrap, excluding-missing-subgroup** directories. It builds IMD and ethnicity panels.

Outputs:

- `Table4_Model4_IMD_ethnicity.png`
- `Table4_Model4_IMD_ethnicity.pdf`
- `Table4_PanelA_IMD.csv`
- `Table4_PanelB_ethnicity.csv`

The CSVs are generated outputs and are intentionally excluded from GitHub by `.gitignore`.

## Supplementary outputs

### `supplementary_figures/figures.py`

Reconstructs the final test sets from the COD dataset and run-level pickles and generates Model 4 ROC and precision-recall curves for the six female/male outcome analyses.

Outputs:

- `Supplementary_Figure_S1_ROC_Model4.png/.pdf`
- `Supplementary_Figure_S2_PrecisionRecall_Model4.png/.pdf`

### `supplementary_figures/tables1.py`

Reads only the header of the final COD dataset, identifies the predictor variables using the same naming conventions as the modelling pipeline, and builds the predictor-definition/model-membership supplementary table.

Outputs:

- `Supplementary_Table_S1_predictors.csv`
- `Supplementary_Table_S1_predictors.png`
- `Supplementary_Table_S1_predictors.pdf`

## Suggested execution order

There is no strict dependency among most figure/table scripts once their required modelling outputs exist. A sensible publication-build sequence is:

```text
04_modelling final primary runs
        |
        +--> database/subgroup post-processing as required
        |
        v
figure2.py
figure3.py
figure4.py
table2.py
table3.py
table4.py
supplementary_figures/tables1.py
supplementary_figures/figures.py
```

`table4.py` specifically requires the subgroup post-processing to have been completed first.

## Reproducibility notes

- Most scripts contain absolute ALICE result paths and exact expected pickle/result filenames. Update these paths if outputs are moved.
- Generated PNG/PDF/CSV files are not version controlled; only the scripts are stored in the repository.
- `figure2.py` contains hard-coded AUC values. If final modelling results change, update or refactor this script rather than assuming it automatically refreshes from new results.
