#!/bin/bash
#SBATCH --job-name=16-MEA
#SBATCH --partition=small
#SBATCH --time=5-00:00:00
#SBATCH --mem=50G
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --output=mpi-test.%j.out
#SBATCH --error=mpi-test.%j.err

set -euo pipefail

EXE=/nas/longleaf/home/emainas/software/dcdftbmd.2.0/dr_nishimura/20260126_Dcdftbmd/bin/dftb_mpiomp.00.x

module purge
module load openmpi/5.0.6_intel-2024.2.1

export PATH=/nas/longleaf/home/emainas/software/dcdftbmd.2.0/:$PATH

# yoshifumi's suggestion - need this to fix segmentation fault
# the default memore limit is 8192KB=8MB - check with 'ulimit -s'
# this is tiny for the large arrays the fortran code is building
ulimit -s unlimited

time mpirun -np 16 "${EXE}"
