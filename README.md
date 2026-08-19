# CPRD Diabetes Prediction Models

Code used for data preparation, cohort construction, comorbidity and medication extraction, statistical/machine-learning modelling, and manuscript figure generation.

## Repository structure

### 01_data_cleaning
CPRD GOLD and CPRD Aurum data preparation pipelines, including medication extraction.

### 02_comorbidity_extraction
Extraction and quality control of baseline comorbidities from CPRD and linked HES data.

### 03_censoring_and_qc
Database combination, IMD/ethnicity recoding, medication/comorbidity merging, study-end censoring, cohort QC, and cause-of-death derivation.

### 04_modelling
Final V3 statistical and machine-learning modelling pipeline, including database-stratified and subgroup analyses.

### 05_manuscript_figures
Scripts used to generate main and supplementary manuscript figures and tables.

## Data availability

Individual-level CPRD, HES, ONS and linked study data are not included in this repository.

The code was developed for execution within the University of Leicester ALICE high-performance computing environment. Some scripts currently contain environment-specific file paths that should be configured before use in another environment.
