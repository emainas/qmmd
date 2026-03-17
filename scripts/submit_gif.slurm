#!/usr/bin/env bash
#SBATCH -J qmmd-gif
#SBATCH -p small
#SBATCH -N 1
#SBATCH --ntasks 16
#SBATCH -t 2-00:00:00
#SBATCH --mem=8G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

/nas/longleaf/home/emainas/.conda/envs/qmmd/bin/python \
  /users/e/m/emainas/qmmd/plotting/plot_coord_dist_hist_gif.py \
  --runs-path /users/e/m/emainas/qmmd/systems/HIST/solv_5.5/dftb/N1T64C1 \
  --cv-dir meta-hid \
  --bins 50 \
  --step 200 \
  --run1-frac 0.15
