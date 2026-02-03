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
qmmd prep configs/<molecule>/prep.yaml
```

---

## mdequil - NVT and NPT equilibration with classical force field using Amber's sander module

Generates Amber input files and slurm input file to run on HPC cluster. 

**Run**

```bash
qmmd prep configs/<molecule>/mdequil.yaml
```

---

## salt - Post-equilibration system adjustment

Deletes tleap's counterion which is given as input and mutates the furthest water molecule into hydroxide anion

**Run**

```bash
qmmd salt configs/<molecule>/salt.yaml
```
