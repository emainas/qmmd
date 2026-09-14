# BV equilibration helicity distribution

Run `python plotting/plot_bv_equil_helicity_pmf.py` from the repository root.
The default selection is all 200 runs under
`systems/BV/solv_4.0/dftb/N1T48C1`, using only `equil/traject`.
An existing output directory is refused; choose `--out-dir` for a new analysis.

Use `--system CPP` or `--system BPP` for the corresponding 50-run sets.
Their topology names NA/NB/NC/ND resolve the ring anchors independently;
their atom numbers are not assumed to match BV. Each five-membered bonded
cycle is checked against the named atoms N, C1, C2, C3, C4 of that ring.
Outputs default to `reports/<SYSTEM>/equil_helicity_pmf`.

The signed helicity is the A–B–C–D geometric-ring-center dihedral, exactly
as in `acid_base_BV.calculate_bv_torsions`, not a sum of bridge torsions.
Each ring contains its five heavy atoms, identified from the Amber bond graph:
A (1,3,4,5,6), B (19,21,22,23,24), C (31,33,34,35,36), D (43,45,46,47,48).
These are one-based indices. Ring atoms are minimum-image unwrapped about an
anchor before taking their arithmetic mean; center-to-center bonds also use
minimum imaging. Fixed orthorhombic boxes are required.

All complete saved frames are used by default. `--discard-ps 10` would discard
the first 10 ps of each run; that is **not** the default. No smoothing is used.
The pooled frame-weighted histogram uses 5-degree bins on [-180,180).
P(theta) integrates to one; F(theta) = -RT ln[P(theta)/max(P)] at 300 K.
Both `--bin-deg` and `--temperature` are configurable. Empty bins remain
undefined in the PMF, with no pseudocounts. There is no extra spherical Jacobian
for this dihedral coordinate. Finite sampling and initial-condition bias can
affect the apparent PMF; this calculation does not establish convergence or
equilibrium populations between conformational basins.

Outputs include the two-panel `helicity_pmf.png`, binned `helicity_pmf.csv`,
all run/time-aligned samples in `helicity_timeseries.csv`, and `provenance.json`
with snapshot hashes, ring indices, per-run frame counts, boxes, and time ranges.
The optimized calculation is checked against the original summary helpers on
the first, middle, and final complete frame of every run.
