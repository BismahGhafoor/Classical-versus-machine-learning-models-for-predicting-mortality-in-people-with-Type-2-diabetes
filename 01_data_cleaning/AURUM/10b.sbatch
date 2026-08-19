#!/bin/bash
#SBATCH --job-name=aurum_biomarker_cleaning
#SBATCH --output=/scratch/alice/b/bg205/01_03_AURUM/logs/aurum_cleaningV2_%j.out
#SBATCH --error=/scratch/alice/b/bg205/01_03_AURUM/logs/aurum_cleaningV3_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=medium
#SBATCH --mem=32G          # Request all memory on node
#SBATCH --cpus-per-task=8   
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bg205@leicester.ac.uk

# Load Python module
module load python/3.10.12-tggsi7t
source ~/aurum_env/bin/activate

# Run the script
cd /scratch/alice/b/bg205/01_03_AURUM
python3 -u 10_optimised.py
