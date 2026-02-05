#!/usr/bin/env bash
#SBATCH -J mdequil
#SBATCH -p small
#SBATCH -N 2
#SBATCH -t 10:00:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

bash run.sh
