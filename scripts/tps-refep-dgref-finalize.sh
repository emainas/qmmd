#!/usr/bin/env bash
#SBATCH -J tps-refep-dgref-report
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 2:00:00
#SBATCH --mem=8G
#SBATCH -o systems/TPS/solv_10.0/refep/dgref-comparison/slurm.%j.out
#SBATCH -e systems/TPS/solv_10.0/refep/dgref-comparison/slurm.%j.err

set -euo pipefail

cd /users/e/m/emainas/qmmd
module purge
module load amber/26
export MPLCONFIGDIR=/tmp/qmmd-tps-finalize-${SLURM_JOB_ID}

/nas/sycamore/home/emainas/.conda/envs/qmmd/bin/qmmd cphmd-dgref-report configs/TPS/cphmd/dgref-mbondi2-app-restart.yaml
/nas/sycamore/home/emainas/.conda/envs/qmmd/bin/qmmd cphmd-dgref-report configs/TPS/cphmd/dgref-mbondi2-bpp-restart.yaml
/nas/sycamore/home/emainas/.conda/envs/qmmd/bin/qmmd cphmd-dgref-report configs/TPS/cphmd/dgref-mbondi2-cpp.yaml
/nas/sycamore/home/emainas/.conda/envs/qmmd/bin/qmmd cphmd-dgref-report configs/TPS/cphmd/dgref-mbondi2-dpp-restart.yaml

/nas/sycamore/home/emainas/.conda/envs/qmmd/bin/qmmd refep-prod-report configs/TPS/refep/app/prod-implicit.yaml
/nas/sycamore/home/emainas/.conda/envs/qmmd/bin/qmmd refep-prod-report configs/TPS/refep/bpp/prod-implicit.yaml
/nas/sycamore/home/emainas/.conda/envs/qmmd/bin/qmmd refep-prod-report configs/TPS/refep/cpp/prod-implicit.yaml
/nas/sycamore/home/emainas/.conda/envs/qmmd/bin/qmmd refep-prod-report configs/TPS/refep/dpp/prod-implicit.yaml

/nas/sycamore/home/emainas/.conda/envs/qmmd/bin/python plotting/compare_tps_refep_dgref.py
