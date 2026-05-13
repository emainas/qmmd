#!/usr/bin/env bash
#SBATCH -J qmmd-fes-gif
#SBATCH -N 1
#SBATCH --ntasks 48
#SBATCH -t 2-00:00:00
#SBATCH --mem=8G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

/nas/longleaf/home/emainas/.conda/envs/qmmd/bin/python \
  /users/e/m/emainas/qmmd/plotting/plot_fes_pairs_gif.py \
  --runs-path /users/e/m/emainas/qmmd/systems/HIST/solv_4.5/dftb/N1T64C1 \
  --cv-dir all-meta-hid \
  --pairs-script /users/e/m/emainas/qmmd/plotting/plot_fes_pairs.py \
  --out /users/e/m/emainas/qmmd/reports/fes_pairs_all-meta-hid.gif \
  --fps 4 \
  --step 1
