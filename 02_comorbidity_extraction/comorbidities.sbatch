#!/bin/bash
#SBATCH --job-name=cprd_comorbidities
#SBATCH --output=/scratch/alice/b/bg205/16_02_26/logs/cprd_comorbidities_%j.out
#SBATCH --error=/scratch/alice/b/bg205/16_02_26/logs/cprd_comorbidities_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bg205@leicester.ac.uk

# --- Create log directory if it doesn't exist ---
mkdir -p /scratch/alice/b/bg205/16_02_26/logs

# --- Load modules ---
module purge
# module load python/3.11.9-awvqkg4   # currently not loading on your account

# --- Activate virtual environment ---
source /scratch/alice/b/bg205/venvs/cprd/bin/activate

# --- Print some context to the log ---
echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Start time : $(date)"
echo "Python     : $(python --version 2>&1)"
echo "Which py   : $(which python)"
echo "=========================================="

# --- Run the script ---
python -u /scratch/alice/b/bg205/16_02_26/comorbidities.py

EXIT_CODE=$?

echo "=========================================="
echo "End time   : $(date)"
echo "Exit code  : $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
