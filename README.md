# qmmd

Lightweight workflows for preparing and running QM/MM and DFTB-MD systems.
Uses AmberTools / Amber and DCDFTBMD.

---

## Installation

Clone the repository and create the Conda environment:

```bash
git clone <REPO_URL>
cd qmmd
conda env create -f environment.yml
conda activate qmmd

pip install -e .

qmmd --help
