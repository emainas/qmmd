#!/usr/bin/env bash
#SBATCH --job-name=MEA-N1T64C1
#SBATCH --partition=small
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=50G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

bash run.sh
