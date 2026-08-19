#!/bin/bash
#SBATCH --job-name=medication_job
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=/scratch/alice/b/bg205/01_03_AURUM/medicationlogs/meds_%j.out
#SBATCH --error=/scratch/alice/b/bg205/01_03_AURUM/medicationlogs/meds_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bg205@leicester.ac.uk

set -euo pipefail

# Load whichever your cluster uses:
module purge # or: module load Anaconda3; source activate <env-name>

LOG_DIR="/scratch/alice/b/bg205/01_03_AURUM/logs"
SCRATCH_WORK_DIR="/scratch/alice/b/bg205/01_03_AURUM"
#PYTHON_PROCESSING_SCRIPT="/scratch/alice/b/bg205/01_03_AURUM/8_optimised.py"
PYTHON_PROCESSING_SCRIPT="/scratch/alice/b/bg205/01_03_AURUM/13.py"

mkdir -p "$LOG_DIR"
cd "$SCRATCH_WORK_DIR"

echo "Host: $(hostname)"
echo "CWD:  $(pwd)"
echo "Python: $(which python)"
python --version
echo "Running: python ${PYTHON_PROCESSING_SCRIPT}"

python "${PYTHON_PROCESSING_SCRIPT}"

echo "Done."
