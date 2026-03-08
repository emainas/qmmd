# qmmd

Lightweight workflows for preparing and running QMMD, QMMD-WTMETA and QM/MM MD simulations.
Uses AmberTools / Amber and DCDFTBMD.

---

## Installation

```bash
git clone https://github.com/emainas/qmmd.git
cd qmmd
conda env create -f environment.yml
conda activate qmmd
pip install -e .
qmmd --help
```

---

## prep — System Preparation

Generates a solvated system using `tleap`.

**Inputs**
- `mol2`, `frcmod`
- water model
- buffer size
- optional counterions

**Output**
- solv.par7
- solv.rst7
- spec.yaml

**Run**

```bash
qmmd prep configs/<molecule>/prep/prep.yaml
```

---

## mdequil - MD Equilibration

NVT and NPT equilibration with classical force field using Amber's sander module

**Run**

```bash
qmmd mdequil configs/<molecule>/mdequil/mdequil.yaml
```

---

## salt - Post-equilibration System Adjustment

Deletes tleap's counterion which is given as input and mutates the furthest water molecule into hydroxide anion

**Run**

```bash
qmmd salt configs/<molecule>/salt/salt.yaml
```

---

## dftb-prep - Prepare DCDFTBMD Runs

Reads the xyz file from previous step and writes one or many DFTB runs (no submission)

**Inputs**
- `ready.xyz` (from salt)
- `params/*.skf`
- `dftb.yaml`

**Output**
- `dftb.inp`
- `run.sh`
- `slurm.sh`
- `spec.yaml`

**Notes**
- `replicas` + `append` control how many runs are created and whether to start after existing runs
- `MD=(... SEEDTYPE=3 RANDOMSEED=0 ...)` will replace `0` with a unique seed per run

**Run**

```bash
qmmd dftb-prep configs/<molecule>/dftb/dftb.yaml
```

---

## dftb-submit - Submit DCDFTBMD Runs

Submits Slurm jobs for run directories that match the provided config (by spec.yaml)

**Run**

```bash
qmmd dftb-submit configs/<molecule>/dftb/dftb.yaml
```
