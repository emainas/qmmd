#!/usr/bin/env bash
set -euo pipefail

module purge
module load openmpi/5.0.6_intel-2024.2.1

export PATH="/nas/longleaf/home/emainas/software/dcdftbmd.2.0/:$PATH"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_STACKSIZE=1G

ulimit -s unlimited

mpirun -np 16 "/nas/longleaf/home/emainas/software/dcdftbmd.2.0/dr_nishimura/20260126_Dcdftbmd/bin/dftb_mpiomp.00.x"
