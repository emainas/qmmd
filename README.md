# qmmd

Lightweight workflows for preparing and running QMMD, QMMD-WTMETA and QM/MM MD simulations.
Uses AmberTools / Amber and DCDFTBMD.

---

## Installation

```bash
git clone <REPO_URL>
cd qmmd
conda env create -f environment.yml
conda activate qmmd
pip install -e .
qmmd --help
