## Reproducible environment 

git clone https://github.com/emainas/bmqm.git
cd bmqm

conda create -n bmqm --file conda-linux-64.lock 
conda activate bmqm

# Tree structure
.
├── analysis
├── conda-lock.yml
├── environment.yml
├── input
│   └── monoethanolamine
│       └── 1310-atoms
│           └── dftb.inp
├── logs
├── params
│   ├── *.skf
|
├── README.md
├── results
├── src
│   └── make_jobs.py
└── templates
    ├── plotting
    │   └── prl.mplstyle
    └── slurm
        └── run.sh

