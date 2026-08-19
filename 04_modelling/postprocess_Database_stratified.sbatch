#!/bin/bash
#SBATCH --job-name=dbstrat_t2
#SBATCH --array=0-11%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/alice/b/bg205/16_02_26/Modelling/logsPostProcess/dbstrat_t2_%A_%a.out
#SBATCH --error=/scratch/alice/b/bg205/16_02_26/Modelling/logsPostProcess/dbstrat_t2_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bg205@leicester.ac.uk

set -Eeuo pipefail

trap 'status=$?; echo "ERROR: command failed on line ${LINENO} with exit code ${status}" >&2; exit ${status}' ERR

# =============================================================================
# USER SETTINGS
# =============================================================================
MODELLING_DIR="/scratch/alice/b/bg205/16_02_26/Modelling/V3"
POSTPROCESS_SCRIPT="${MODELLING_DIR}/postprocess_database_stratified_performance.py"
DATA_PATH="/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/Combined_GOLD_Aurum_with_meds_comorbidities_studyend_cod.txt"
LOG_DIR="/scratch/alice/b/bg205/16_02_26/Modelling/logsPostProcess"

CONDA_SH="/home/b/bg205/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="/home/b/bg205/miniconda3/envs/aurum_en"

# Defaults to a smoke test. Override at submission time.
N_BOOTSTRAPS="${N_BOOTSTRAPS:-10}"
INCLUDE_CALIBRATION_CI="${INCLUDE_CALIBRATION_CI:-0}"
TUNING_METRIC="${TUNING_METRIC:-brier}"
HYBRID_CAPTURE_PERCENT="${HYBRID_CAPTURE_PERCENT:-90}"

# Narrow this when more than one modelling pickle matches.
# Example: _boot-2000_rf-750
MODEL_SUFFIX_GLOB="${MODEL_SUFFIX_GLOB:-*}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# =============================================================================
# ARRAY MAPPING
# =============================================================================
SEXES=("female" "male")
OUTCOMES=("death_10y" "death_cvd" "death_cancer")
SENSITIVITIES=("none" "complete_case")

TASK_ID="${SLURM_ARRAY_TASK_ID}"
SENSITIVITY_INDEX=$((TASK_ID / 6))
WITHIN_SENSITIVITY=$((TASK_ID % 6))
SEX_INDEX=$((WITHIN_SENSITIVITY / 3))
OUTCOME_INDEX=$((WITHIN_SENSITIVITY % 3))

SEX="${SEXES[$SEX_INDEX]}"
OUTCOME="${OUTCOMES[$OUTCOME_INDEX]}"
SENSITIVITY="${SENSITIVITIES[$SENSITIVITY_INDEX]}"

if [[ "${SENSITIVITY}" == "none" ]]; then
    ANALYSIS_LABEL="imputed"
    PICKLE_PREFIX=""
else
    ANALYSIS_LABEL="complete_case"
    PICKLE_PREFIX="SENSITIVITY_completeCase_"
fi

MODEL_ROOT="${MODELLING_DIR}/Combined_${SEX}_type_2"
RUN_LEVEL_DIR="${MODEL_ROOT}/_run_level"

MODEL_GLOB="${RUN_LEVEL_DIR}/${PICKLE_PREFIX}Combined_type_2_${OUTCOME}_all_models_${SEX}_tune-${TUNING_METRIC}_cap-${HYBRID_CAPTURE_PERCENT}${MODEL_SUFFIX_GLOB}.pkl"

if [[ "${INCLUDE_CALIBRATION_CI}" == "1" ]]; then
    CALIBRATION_LABEL="with_calibration_ci"
else
    CALIBRATION_LABEL="without_calibration_ci"
fi

OUTPUT_DIR="${MODEL_ROOT}/postprocessing_database_stratified/${ANALYSIS_LABEL}/${OUTCOME}/boot_${N_BOOTSTRAPS}_${CALIBRATION_LABEL}"

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

[[ -f "${POSTPROCESS_SCRIPT}" ]] || {
    echo "ERROR: post-processing script not found: ${POSTPROCESS_SCRIPT}" >&2
    exit 1
}

[[ -f "${DATA_PATH}" ]] || {
    echo "ERROR: source dataset not found: ${DATA_PATH}" >&2
    exit 1
}

[[ -d "${RUN_LEVEL_DIR}" ]] || {
    echo "ERROR: run-level model directory not found: ${RUN_LEVEL_DIR}" >&2
    exit 1
}

[[ -f "${CONDA_SH}" ]] || {
    echo "ERROR: conda initialisation script not found: ${CONDA_SH}" >&2
    exit 1
}

mapfile -t MODEL_PKLS < <(compgen -G "${MODEL_GLOB}" | sort)

if (( ${#MODEL_PKLS[@]} == 0 )); then
    echo "ERROR: no model pickle matched:" >&2
    echo "  ${MODEL_GLOB}" >&2
    echo >&2
    echo "Available outcome-level run pickles:" >&2

    find "${RUN_LEVEL_DIR}" \
        -maxdepth 1 \
        -type f \
        -name "*${OUTCOME}*all_models*.pkl" \
        -print | sort >&2 || true

    exit 1
fi

if (( ${#MODEL_PKLS[@]} > 1 )); then
    echo "ERROR: more than one model pickle matched; refusing to guess:" >&2
    printf '  %s\n' "${MODEL_PKLS[@]}" >&2
    echo >&2
    echo "Resubmit with MODEL_SUFFIX_GLOB narrowed to the required suffix." >&2
    echo "Example: MODEL_SUFFIX_GLOB=_boot-2000_rf-750" >&2
    exit 1
fi

MODEL_PKL="${MODEL_PKLS[0]}"

# =============================================================================
# ENVIRONMENT
# =============================================================================
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${MODELLING_DIR}"

PYTHON_BIN="$(command -v python)"

echo "============================================================"
echo "DATABASE-STRATIFIED POST-PROCESSING"
echo "============================================================"
echo "Started:                 $(date)"
echo "Host:                    $(hostname)"
echo "Working directory:       $(pwd)"
echo "SLURM job ID:            ${SLURM_JOB_ID}"
echo "SLURM array task ID:     ${SLURM_ARRAY_TASK_ID}"
echo "CPUs:                    ${SLURM_CPUS_PER_TASK}"
echo "Memory requested:        64G"
echo "Python:                  ${PYTHON_BIN}"
echo "Python version:          $(python --version 2>&1)"
echo "Sex:                     ${SEX}"
echo "Outcome:                 ${OUTCOME}"
echo "Sensitivity:             ${SENSITIVITY} (${ANALYSIS_LABEL})"
echo "Model pickle:            ${MODEL_PKL}"
echo "Dataset:                 ${DATA_PATH}"
echo "Bootstraps:              ${N_BOOTSTRAPS}"
echo "Calibration CIs:         ${INCLUDE_CALIBRATION_CI}"
echo "Output directory:        ${OUTPUT_DIR}"
echo "============================================================"

ARGS=(
    --model_pkl "${MODEL_PKL}"
    --data_path "${DATA_PATH}"
    --diabetes_type 2
    --n_bootstraps "${N_BOOTSTRAPS}"
    --bootstrap_unit practice
    --outcomes "${OUTCOME}"
    --models "LR,RF,XGB,HYBRID"
    --blocks "A,B,C,D"
    --output_dir "${OUTPUT_DIR}"
)

if [[ "${INCLUDE_CALIBRATION_CI}" == "1" ]]; then
    ARGS+=(--include_calibration_ci)
fi

/usr/bin/time -v python -u \
    "${POSTPROCESS_SCRIPT}" \
    "${ARGS[@]}"

echo "============================================================"
echo "Completed successfully: $(date)"
echo "Tables: ${OUTPUT_DIR}/tables"
echo "============================================================"
