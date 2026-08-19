# CPRD Diabetes Prediction Models

Code and study codelists used for data preparation, cohort construction, comorbidity and medication extraction, statistical/machine-learning modelling, and manuscript figure generation.

## Repository structure

### 00_codelists
Study codelists and phenotype definitions used throughout the analysis pipeline, including diabetes, ethnicity, smoking, biomarkers, medications, and comorbidities for CPRD GOLD and CPRD Aurum.

### 01_data_cleaning
CPRD GOLD and CPRD Aurum data preparation pipelines, including cohort identification, demographic enrichment, smoking and biomarker extraction, and medication processing.

### 02_comorbidity_extraction
Extraction and quality control of baseline comorbidities from CPRD and linked HES data.

### 03_censoring_and_qc
Database combination, IMD/ethnicity recoding, medication/comorbidity merging, study-end filtering, cohort quality control, and cause-of-death derivation.

### 04_modelling
Final statistical and machine-learning modelling pipeline, including model development, database-stratified analyses, subgroup analyses, and performance post-processing.

### 05_manuscript_figures
Scripts used to generate the main and supplementary manuscript figures and tables from modelling outputs.

## Documentation

Each major folder contains its own `README.md` describing the purpose of the scripts, their required inputs, expected outputs, dependencies, and the order in which they should be run.

## Data availability

Individual-level CPRD, HES, ONS, IMD-linked, and other patient-level study data are not included in this repository.

The `00_codelists` directory contains study phenotype and code definitions used by the analysis. Some codelists originate from shared project resources; redistribution permissions should be confirmed before making the repository publicly available.

The code was developed for execution within the University of Leicester ALICE high-performance computing environment. Some scripts contain environment-specific file paths and rely on external CPRD lookup resources that are not distributed with this repository. These paths and dependencies should be configured before running the pipeline in another environment.
