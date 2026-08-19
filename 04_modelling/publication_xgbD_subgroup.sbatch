#!/bin/bash
#SBATCH --job-name=t2_subgroups
#SBATCH --array=0-5%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/alice/b/bg205/16_02_26/Modelling/V3/logsPostProcess/t2_subgroups_%A_%a.out
#SBATCH --error=/scratch/alice/b/bg205/16_02_26/Modelling/V3/logsPostProcess/t2_subgroups_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bg205@leicester.ac.uk

set -Eeuo pipefail

trap 'status=$?; echo "ERROR: command failed on line ${LINENO} with exit code ${status}" >&2; exit ${status}' ERR

# =============================================================================
# USER SETTINGS
# =============================================================================
MODELLING_DIR="/scratch/alice/b/bg205/16_02_26/Modelling/V3"
SUBGROUP_SCRIPT="${MODELLING_DIR}/publication_xgbD_subgroup.py"
DATA_PATH="/scratch/alice/b/bg205/16_02_26/CLEANED_DATA/Combined_GOLD_Aurum_with_meds_comorbidities_studyend_cod.txt"
LOG_DIR="/scratch/alice/b/bg205/16_02_26/Modelling/V3/logsPostProcess"

CONDA_SH="/home/b/bg205/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="/home/b/bg205/miniconda3/envs/aurum_en"

# Defaults to a smoke test. Override at submission.
N_BOOTSTRAPS="${N_BOOTSTRAPS:-10}"
TUNING_METRIC="${TUNING_METRIC:-brier}"
HYBRID_CAPTURE_PERCENT="${HYBRID_CAPTURE_PERCENT:-90}"
MODEL_SUFFIX_GLOB="${MODEL_SUFFIX_GLOB:-*}"

SUBGROUP_COLS="${SUBGROUP_COLS:-imd_quintile,gen_ethnicity}"
MIN_N="${MIN_N:-500}"
MIN_EVENTS="${MIN_EVENTS:-20}"
INCLUDE_MISSING_SUBGROUP="${INCLUDE_MISSING_SUBGROUP:-0}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# =============================================================================
# ARRAY MAPPING: 2 sexes x 3 outcomes = 6 tasks
# =============================================================================
SEXES=("female" "male")
OUTCOMES=("death_10y" "death_cvd" "death_cancer")

TASK_ID="${SLURM_ARRAY_TASK_ID}"
SEX_INDEX=$((TASK_ID / 3))
OUTCOME_INDEX=$((TASK_ID % 3))

SEX="${SEXES[$SEX_INDEX]}"
OUTCOME="${OUTCOMES[$OUTCOME_INDEX]}"

MODEL_ROOT="${MODELLING_DIR}/Combined_${SEX}_type_2"
RUN_LEVEL_DIR="${MODEL_ROOT}/_run_level"

MODEL_GLOB="${RUN_LEVEL_DIR}/Combined_type_2_${OUTCOME}_all_models_${SEX}_tune-${TUNING_METRIC}_cap-${HYBRID_CAPTURE_PERCENT}${MODEL_SUFFIX_GLOB}.pkl"

MISSING_LABEL="excluding_missing_subgroup"
if [[ "${INCLUDE_MISSING_SUBGROUP}" == "1" ]]; then
    MISSING_LABEL="including_missing_subgroup"
fi

OUTPUT_DIR="${MODEL_ROOT}/postprocessing_subgroups_xgb_blockd/${OUTCOME}/boot_${N_BOOTSTRAPS}_${MISSING_LABEL}"

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

[[ -f "${SUBGROUP_SCRIPT}" ]] || {
    echo "ERROR: subgroup script not found: ${SUBGROUP_SCRIPT}" >&2
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
echo "TYPE 2 SUBGROUP POST-PROCESSING"
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
echo "Model/block:             XGB / D"
echo "Model pickle:            ${MODEL_PKL}"
echo "Dataset:                 ${DATA_PATH}"
echo "Subgroups:               ${SUBGROUP_COLS}"
echo "Bootstraps:              ${N_BOOTSTRAPS}"
echo "Minimum N/events:        ${MIN_N}/${MIN_EVENTS}"
echo "Include missing group:   ${INCLUDE_MISSING_SUBGROUP}"
echo "Output directory:        ${OUTPUT_DIR}"
echo "============================================================"

ARGS=(
    --model_pkl "${MODEL_PKL}"
    --data_path "${DATA_PATH}"
    --diabetes_type 2
    --subgroup_cols "${SUBGROUP_COLS}"
    --model XGB
    --block D
    --outcomes "${OUTCOME}"
    --n_bootstraps "${N_BOOTSTRAPS}"
    --bootstrap_unit practice
    --min_n "${MIN_N}"
    --min_events "${MIN_EVENTS}"
    --output_dir "${OUTPUT_DIR}"
)

if [[ "${INCLUDE_MISSING_SUBGROUP}" == "1" ]]; then
    ARGS+=(--include_missing_subgroup)
fi

/usr/bin/time -v python -u \
    "${SUBGROUP_SCRIPT}" \
    "${ARGS[@]}"

echo "============================================================"
echo "Completed successfully: $(date)"
echo "Tables: ${OUTPUT_DIR}/tables"
echo "============================================================"

