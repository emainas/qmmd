# BV equilibration conformational sampling

`plot_bv_equil_basins.py` reads only `run-*/equil/traject` and `equil/dftb.inp`.
It does not read metadynamics trajectories or change simulation data. Defaults
are finalists 39, 57, 99, 142 and 190, written under
`reports/BV/meta-hic-finalist-runs/run-*/`.

```bash
python plotting/plot_bv_equil_basins.py
```

The analysis reuses the existing BV single-bridge definitions, with unique atom
names resolved from the Amber topology and reported as one-based atom IDs:

| Axis | Atom quartet |
| --- | --- |
| single5 | NB–C1B–C5–C4A |
| single10 | NB–C4B–C10–C1C |
| single15 | NC–C4C–C15–C1D |

Every complete saved equilibration frame is included, including initial
relaxation. No burn-in or subsampling is imposed. A size-bounded file snapshot
is hashed; a partial final XYZ frame is ignored. Nonmonotonic timestamps and
degenerate torsions raise errors instead of silently changing alignment.
Orthorhombic minimum-image bond vectors are used. Variable-cell NPT and
nonorthorhombic boxes are rejected by this implementation.

## Operational basin definition

1. Histogram the three angles on a periodic 360-degree cube, using 10-degree bins.
2. Apply wrapped Gaussian smoothing with sigma 15 degrees along each axis.
3. Follow the steepest ascending neighbor among the 26 surrounding grid cells
   until a local density maximum is reached. Each frame inherits its bin's
   catchment. Periodicity applies both at the histogram boundary and during ascent.
4. Process occupied catchments in descending population; merge nearby modes
   into an existing representative when their wrapped three-angle Euclidean
   separation is <=30 degrees. This is a deterministic proximity merge, not
   a saddle-height or kinetic-metastability criterion.
5. Label merged basins with >=5% of all frames. Smaller catchments remain grey,
   label 0. Label B1 is the largest basin **within that run**, not a cross-run
   conformer identifier.

The peak marker is a smoothed-density grid mode, not necessarily an observed
structure. The sidebar gives circular mean angles. The basin CSV also records
circular standard deviations. A single basin may contain all frames, including
its low-density tails; a 100% population does not mean a rigid conformation.

These are descriptive sampling basins, **not demonstrated equilibrium
free-energy minima or metastable states**. No dwell-time test or reweighting is
performed. Frame populations are correlated observations, include equilibration
relaxation, and depend on bin size, bandwidth, merge separation and population
threshold. Explore these settings before assigning physical significance.

All five plots use the same axis limits and view. Angular coordinates are
displayed around a pooled circular reference to reduce artificial seam splits;
ticks and saved raw angles use the canonical [-180,180) convention. Basin finding
does not depend on this display reference.

## Options and outputs

Relevant controls: `--bin-deg`, `--bandwidth-deg`, `--merge-deg`, and
`--min-population` (fraction, not percent). Paths, run IDs and solute atom count
are configurable. Existing conformation outputs are protected unless
`--overwrite` is explicitly provided. Existing acid–base summaries are untouched.

Each run receives:

- `equil_single_bridge_3d.png`: 3D cloud, density-mode stars and basin labels;
- `equil_single_bridge_3d.csv`: frame number, original time in ps, three signed
  torsions, basin ID and smoothed histogram-cell probability;
- `equil_single_bridge_3d_basins.csv`: populations, modes and circular statistics;
- `equil_single_bridge_3d_provenance.json`: source hash/size, atom quartets,
  box, parameters, time range, display reference and limitations.
