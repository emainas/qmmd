#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
#SBATCH -J qmmd-fes-flood
#SBATCH -N 1
#SBATCH --ntasks 48
#SBATCH --mem=8G
#SBATCH -t 2-00:00:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

cd /users/e/m/emainas/qmmd
echo "PWD=$(pwd)"
echo "PYTHON=/nas/longleaf/home/emainas/.conda/envs/qmmd/bin/python"
/nas/longleaf/home/emainas/.conda/envs/qmmd/bin/python -V
ls -l /users/e/m/emainas/qmmd/configs/HIST/plotting/all-meta-hid.yaml
/nas/longleaf/home/emainas/.conda/envs/qmmd/bin/python -u -m qmmd.cli gif-run /users/e/m/emainas/qmmd/configs/HIST/plotting/all-meta-hid.yaml
