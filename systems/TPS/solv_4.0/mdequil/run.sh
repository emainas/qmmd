#!/usr/bin/env bash
set -euo pipefail

module purge
module load amber/24

export TEMP=300
export PRESS=1.0

echo "==> Running min/heat/equil (serial sander)"

# Minimize
if [[ ! -f min.out ]]; then
    echo "  Minimization..."
    sander -O \
      -i min.in \
      -p "/nas/longleaf/home/emainas/software/dcdftbmd.2.0/molecules/MEA/qmmd/systems/TPS/solv_4.0/prep/solv.parm7" \
      -c "/nas/longleaf/home/emainas/software/dcdftbmd.2.0/molecules/MEA/qmmd/systems/TPS/solv_4.0/prep/solv.rst7" \
      -r min.rst7 \
      -o min.out \
      -inf min.info
else
    echo "  Skipping minimization (min.out exists)"
fi

# Heat (start from minimized)
if [[ ! -f heat.out ]]; then
    echo "  Heating..."
    sander -O \
      -i heat.in \
      -p "/nas/longleaf/home/emainas/software/dcdftbmd.2.0/molecules/MEA/qmmd/systems/TPS/solv_4.0/prep/solv.parm7" \
      -c min.rst7 \
      -r heat.rst7 \
      -o heat.out \
      -inf heat.info \
      -x heat.nc
else
    echo "  Skipping heating (heat.out exists)"
fi

# Equilibration NVT (start from heated)
if [[ ! -f equil-nvt.out ]]; then
    echo "  Equilibration (NVT)..."
    sander -O \
      -i equil-nvt.in \
      -p "/nas/longleaf/home/emainas/software/dcdftbmd.2.0/molecules/MEA/qmmd/systems/TPS/solv_4.0/prep/solv.parm7" \
      -c heat.rst7 \
      -r equil-nvt.rst7 \
      -o equil-nvt.out \
      -inf equil-nvt.info \
      -x equil-nvt.nc
else
    echo "  Skipping equilibration (equil-nvt.out exists)"
fi

# Equilibration NPT (start from NVT)
if [[ ! -f equil-npt.out ]]; then
    echo "  Equilibration (NPT)..."
    sander -O \
      -i equil-npt.in \
      -p "/nas/longleaf/home/emainas/software/dcdftbmd.2.0/molecules/MEA/qmmd/systems/TPS/solv_4.0/prep/solv.parm7" \
      -c equil-nvt.rst7 \
      -r equil-npt.rst7 \
      -o equil-npt.out \
      -inf equil-npt.info \
      -x equil-npt.nc
else
    echo "  Skipping equilibration (equil-npt.out exists)"
fi
