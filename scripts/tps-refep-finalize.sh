#!/usr/bin/env bash
#SBATCH -J tps-refep-report
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 2:00:00
#SBATCH --mem=8G
#SBATCH -o systems/TPS/solv_10.0/refep/refep-report.%j.out
#SBATCH -e systems/TPS/solv_10.0/refep/refep-report.%j.err

set -euo pipefail

cd /users/e/m/emainas/qmmd
module purge
module load amber/26
export MPLCONFIGDIR=/tmp/qmmd-tps-refep-report-${SLURM_JOB_ID}

/nas/sycamore/home/emainas/.conda/envs/qmmd/bin/qmmd refep-prod-report configs/TPS/refep/app/prod-implicit.yaml
/nas/sycamore/home/emainas/.conda/envs/qmmd/bin/qmmd refep-prod-report configs/TPS/refep/bpp/prod-implicit.yaml
/nas/sycamore/home/emainas/.conda/envs/qmmd/bin/qmmd refep-prod-report configs/TPS/refep/cpp/prod-implicit.yaml
/nas/sycamore/home/emainas/.conda/envs/qmmd/bin/qmmd refep-prod-report configs/TPS/refep/dpp/prod-implicit.yaml
