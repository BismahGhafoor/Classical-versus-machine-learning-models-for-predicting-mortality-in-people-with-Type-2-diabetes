#!/bin/bash
#SBATCH --job-name=mortality_by_outcome
#SBATCH --array=0-11
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/alice/b/bg205/16_02_26/Modelling/V3/modellinglogs/hybridfixed/mortality_%A_%a.out
#SBATCH --error=/scratch/alice/b/bg205/16_02_26/Modelling/V3/modellinglogs/hybridfixed/mortality_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bg205@leicester.ac.uk

set -euo pipefail

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

WORK_DIR="/scratch/alice/b/bg205/16_02_26/Modelling/V3"

#SCRIPT="${WORK_DIR}/modelling_final_optimised.py"
SCRIPT="${WORK_DIR}/modelling_fixedhybrid.py"

DATA="/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/Combined_GOLD_Aurum_with_meds_comorbidities_studyend_cod.txt"

PYTHON="/home/b/bg205/miniconda3/envs/aurum_en/bin/python"

cd "${WORK_DIR}"

# ------------------------------------------------------------------
# Validate required files
# ------------------------------------------------------------------

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: Python executable not found or not executable:"
    echo "  ${PYTHON}"
    exit 1
fi

if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: Modelling script not found:"
    echo "  ${SCRIPT}"
    exit 1
fi

if [[ ! -f "${DATA}" ]]; then
    echo "ERROR: Dataset not found:"
    echo "  ${DATA}"
    exit 1
fi

# ------------------------------------------------------------------
# Array-task mapping
#
# Tasks 0-5:  Main analysis
# Tasks 6-11: Complete case sensitivity analysis
# Tasks 12-17: Complete 10-year follow-up sensitivity analysis
# ------------------------------------------------------------------

SEXES=(
    male
    male
    male
    female
    female
    female
    male
    male
    male
    female
    female
    female
    male
    male
    male
    female
    female
    female
)

OUTCOMES=(
    death_10y
    death_cvd
    death_cancer
    death_10y
    death_cvd
    death_cancer
    death_10y
    death_cvd
    death_cancer
    death_10y
    death_cvd
    death_cancer
    death_10y
    death_cvd
    death_cancer
    death_10y
    death_cvd
    death_cancer
)

SENSITIVITIES=(
    none
    none
    none
    none
    none
    none
    complete_case
    complete_case
    complete_case
    complete_case
    complete_case
    complete_case
    complete_fu
    complete_fu
    complete_fu
    complete_fu
    complete_fu
    complete_fu
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"

if (( TASK_ID < 0 || TASK_ID >= ${#SEXES[@]} )); then
    echo "ERROR: Invalid SLURM array task ID: ${TASK_ID}"
    exit 1
fi

SEX="${SEXES[$TASK_ID]}"
OUTCOME="${OUTCOMES[$TASK_ID]}"
SENSITIVITY="${SENSITIVITIES[$TASK_ID]}"

# ------------------------------------------------------------------
# Job information
# ------------------------------------------------------------------

echo "============================================================"
echo "MORTALITY MODELLING JOB"
echo "============================================================"
echo "SLURM job ID:       ${SLURM_JOB_ID}"
echo "Array task ID:      ${TASK_ID}"
echo "Host:               $(hostname)"
echo "Start time:         $(date)"
echo "Sex:                ${SEX}"
echo "Outcome:            ${OUTCOME}"
echo "Sensitivity:        ${SENSITIVITY}"
echo "Diabetes type:      Type 2"
echo "CPUs allocated:     ${SLURM_CPUS_PER_TASK}"
echo "Memory requested:   64G"
echo "Python:             ${PYTHON}"
echo "Script:             ${SCRIPT}"
echo "Data:               ${DATA}"
echo "============================================================"

echo "Python version:     $("${PYTHON}" --version 2>&1)"

# Confirm required Python packages are available
"${PYTHON}" -c \
"import pandas, numpy, sklearn, xgboost, missforest; print('Python environment check: OK')"

# ------------------------------------------------------------------
# Thread settings
# ------------------------------------------------------------------

export PYTHONUNBUFFERED=1

# Allow OpenMP-based models to use the allocated CPUs.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Prevent nested BLAS threading.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ------------------------------------------------------------------
# Run modelling pipeline
# ------------------------------------------------------------------

set +e

/usr/bin/time -v "${PYTHON}" -u "${SCRIPT}" \
    --data_path "${DATA}" \
    --database all \
    --sex "${SEX}" \
    --diabetes_type 2 \
    --tuning_metric brier \
    --sensitivity "${SENSITIVITY}" \
    --n_splits 1 \
    --rf_n_estimators 750 \
    --n_bootstraps 2000 \
    --outcome "${OUTCOME}"

EXIT_STATUS=$?

set -e

# ------------------------------------------------------------------
# Completion information
# ------------------------------------------------------------------

echo "============================================================"
echo "TASK COMPLETED"
echo "============================================================"
echo "Sex:                ${SEX}"
echo "Outcome:            ${OUTCOME}"
echo "Sensitivity:        ${SENSITIVITY}"
echo "Exit status:        ${EXIT_STATUS}"
echo "Finish time:        $(date)"
echo "============================================================"

exit "${EXIT_STATUS}"
