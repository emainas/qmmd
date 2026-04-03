#!/usr/bin/env bash
#SBATCH --job-name=precompute-fes2d
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=8G
#SBATCH --output=precompute_fes2d.%j.out
#SBATCH --error=precompute_fes2d.%j.err

set -euo pipefail

# Adjust these two variables before submitting
RUNS_PATH="systems/HID/solv_4.5/dftb/N1T64C1"
CV_DIR="2dwtmeta-20"
WINDOW="0.1"

python scripts/precompute_fes2d.py --runs-path "${RUNS_PATH}" --cv-dir "${CV_DIR}" --window "${WINDOW}"
