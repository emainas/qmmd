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

Optional top-level `dihedral_restraints` applies Amber NMR torsion restraints
throughout minimization, heating, NVT and NPT:

```yaml
dihedral_restraints:
  - atoms: [5, 3, 4, 11]  # one-based topology IDs; verify for your system
    target_deg: 180.0
    force_constant: 50.0  # kcal/mol/rad^2; E = k * displacement^2
```

The writer generates `dihedral.rst`, enables `nmropt=1`, and appends `DISANG`
to each stage, preserving heating weight schedules. Targets are in degrees;
equal r2/r3 give no flat-bottom region. Omit this option for unrestrained MD.
Restraints suppress conformer transitions but also alter intrabasin sampling.
`qmmd mdequil` launches/submits after writing inputs; it is not preparation-only.

---

## us-pull-prep - Prepare sequential Amber pulling windows

Prepares a single Slurm job containing a serial chain of short, harmonically
restrained Amber MD windows. Pull files live under each window's `pull/`
subdirectory. The first window reads the configured equilibrated restart;
every later window reads the preceding window's `pull/pull.rst7`.

```bash
qmmd us-pull-prep configs/<molecule>/us/pull.yaml
qmmd us-pull-submit configs/<molecule>/us/pull.yaml
qmmd us-pull-report configs/<molecule>/us/pull.yaml
```

Window centers are inclusive and specified by `start_deg`, `stop_deg`, and a
positive `spacing_deg`. Atom IDs are one-based. The command writes inputs and
scripts only; it never runs Amber or submits the generated job.
The submit command requires an exact `pull_spec.yaml` match, shows the single
target job, and asks for confirmation before calling `sbatch`.
After the serial pull completes, `us-pull-report` reads that same YAML, verifies
its exact prepared-config snapshot and every window's completion, then uses
Amber `cpptraj` to calculate and stitch the restrained dihedral. It writes the
plot, aligned CSV data, and cpptraj provenance directly under
`systems/<system>/<prefix_buffer>/us-pull/`; it does not write under `reports/`.
The CSV retains both cpptraj's raw periodic angle and the equivalent angle
branch nearest each window target. The report also writes `pull-clock.vmd`, a
static bond-axis clock overlay of every saved pull frame, and one portable
companion file, `pull-clock.mol2`. The MOL2 stores all conformers as separately
bonded residues in one coordinate frame, so VMD shows the complete static
O2-CG-O1-H11 ensemble around one full reference solute. Those are the only two
files needed to view the clock.
Small-partition pull jobs default to two nodes and reject
configurations requesting fewer than two.

### us-equil-prep - Prepare restrained DCDFTBMD equilibration

```bash
qmmd us-equil-prep configs/<molecule>/us/equil.yaml
qmmd us-equil-submit configs/<molecule>/us/equil.yaml
qmmd us-equil-report configs/<molecule>/us/equil.yaml
```

For every completed pull window, this creates `window-XXX/equil/` containing
`dftb.inp`, `metacv.dat`, the converted starting XYZ, execution scripts, an
exact YAML snapshot, and local SKF links. The dihedral atoms and centers come
from the referenced `pull.yaml`. The DFTB metadynamics module remains enabled,
but `METAHEIGHT=0` prevents Gaussian bias deposition. Two `METAWALL` entries
at the same center (`L` and `U`) form the full even-polynomial restraint.
`METAPRINTFES=FALSE` avoids grid columns. Preparation never runs DCDFTBMD or
submits a job.
The submit command reads the same YAML, requires every window's exact
`equil_spec.yaml` snapshot and required input/script files, displays the full
window set, and asks once for confirmation before submitting one Slurm job per
window. If any window fails preflight, no jobs are submitted.
The report command can be run while jobs are active. It reads complete CV
records already flushed to each window's `biaspot`, skips windows without data,
and writes `equil-report.png` plus `equil_dihedral.csv` directly under the
`us-pull/` directory. Each trace uses local window time and the periodic angle
branch nearest its restraint target.

### us-prod-prep - Prepare restrained DCDFTBMD production

```bash
qmmd us-prod-prep configs/<molecule>/us/prod.yaml
qmmd us-prod-submit configs/<molecule>/us/prod.yaml
qmmd us-prod-report configs/<molecule>/us/prod.yaml
```

This requires every referenced equilibration window to have terminated
normally. For each window it creates `window-XXX/prod/`, copies the final binary
equilibration `restart`, and writes the production `dftb.inp`, `metacv.dat`,
scripts, exact `prod_spec.yaml`, and SKF links. Production uses
`RESTART=TRUE`, so coordinates, velocities, and the MD restart state come from
that binary file. Preparation never runs DCDFTBMD or submits a job, and it
refuses to overwrite any existing production directory.
The submit command requires the complete production window set, exact
`prod_spec.yaml` matches, and all required inputs, restart files, and scripts.
It displays every target and asks once before submitting one Slurm job per
window. If any preflight check fails, it submits nothing.
The report command uses cpptraj to calculate the restrained torsion from every
complete coordinate frame in each production `traject`, writing a
`prod/dihedral.dat` file (plus cpptraj input/log provenance) in every usable
window. It converts cumulative trajectory timestamps to local production time; the
`biaspot` CV sampling is not used for production traces or densities. Solute bonds
split by wrapped DFTB coordinates are repaired before cpptraj imaging. The four
aligned panels show, from bottom to top, the Amber pull, DFTB equilibration,
DFTB production, and each production window's normalized dihedral histogram
as 50-bin scatter points with a fitted Gaussian curve. Dashed lines mark every window center, and the legend lists
the target and empirical mean for every production window plus the
pull/equilibration/production spring constants with units. It writes
`prod-report.png`, `prod_dihedral.csv`, and the plotted histogram/fit data in
`prod_density.csv` directly under `us-pull/`. Windows without usable samples
are skipped and reported.

### us-wham - Construct the umbrella PMF

```bash
qmmd us-wham configs/<molecule>/us/wham.yaml
qmmd us-wham-report configs/<molecule>/us/wham.yaml
```

This reads the complete branch-aligned production series written by
`us-prod-report`, writes uniquely named WHAM input series and metadata, runs the
configured Grossfield WHAM executable, and constructs the original-data PMF.
For the DCDFTBMD restraint `V=kappa*(s-s0)^2`, `kappa` is interpreted in
kJ/mol/CV². The metadata coefficient is written as `K=2*kappa/4.184` in
kcal/mol/CV², matching Grossfield WHAM's `V=(1/2)*K*(s-s0)^2` convention.
The YAML controls the coordinate domain, bin count, periodicity, convergence
tolerance, temperature, production discard, PMF zero-reference interval, and
block-bootstrap parameters. Circular moving-block resampling preserves every
window's sample count and assigns a unique file to every window. The original
PMF remains the central estimate; bootstrap mean, standard deviation, and
pointwise confidence intervals are saved separately. Outputs are written under
`systems/<system>/<prefix_buffer>/us-pull/wham/`, including `pmf.png`,
`pmf.csv`, `bootstrap_pmfs.csv`, `thermodynamics.csv`, `thermodynamics.txt`, `metadata.dat`,
`window_summary.csv`, and `overlap.csv`. Thermodynamic outputs report both the
anti-minus-syn PMF-minimum difference and integrated two-basin molar fractions,
as well as forward/reverse barriers and bootstrap confidence/minimum/maximum
statistics.
The PMF figure retains the original-data WHAM points and bootstrap error bars,
then overlays a second-difference-penalized smooth guide. Its dimensionless
plot-only penalty is configurable as `plot.smoothing_penalty`; neither the raw
PMF table nor the reported thermodynamics are calculated from this guide. The
dense coordinates used to draw it are saved in `pmf_smooth.csv`.
After WHAM completes, `us-wham-report` combines the rainbow-colored Amber pull,
production-window probability densities, and the smooth PMF into a shared-axis
three-tier `wham-report.png`. The PMF curve is colored by its nearest umbrella
center using the same window colors. All plotted values are also saved in
`wham-report-data.csv`.

---

## salt - Post-equilibration System Adjustment

Deletes tleap's counterion which is given as input and mutates the furthest water molecule into hydroxide anion

**Run**

```bash
qmmd salt configs/<molecule>/salt/salt.yaml
```

---

## density - Solute/Box Volume

Computes solute volume (mask) and total box volume from salt outputs

**Inputs**
- `ready.rst7`
- `ready.parm7`
- `density.yaml`

**Output**
- `cpptraj.in`
- `cpptraj.out`
- `solute_volume.dat`
- `box_volume.dat`

**Notes**
- Let r2 = L2/L1 and r3 = L3/L1 (keep box ratios fixed)
- Water mass: m = (Nw * 18.01528) / NA  [g]
- Target solvent volume (Ang^3): Vsolv = (m / rho_target) * 1e24
- Target box volume: Vbox = Vsolv + Vsolute
- L1 = (Vbox / (r2 * r3))^(1/3)
- L2 = r2 * L1
- L3 = r3 * L1

**Run**

```bash
qmmd density configs/<molecule>/density/density.yaml
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
- Optional `source_xyz` selects a repository-relative or absolute XYZ instead of
  `salt/ready.xyz`; optional `source_sha256` verifies its contents before preparation.
  Both `Box X: ... Y: ... Z: ...` and extended-XYZ `Lattice="..."` boxes are supported.
  With 20 existing runs, `replicas: 30` and `append: true` creates runs 21–50.
  Repeating preparation appends another 30; this is an additional count, not a target total.
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

For existing runs outside the buffer-based layout, set `system_dir` to a
repository-relative or absolute directory, for example
`systems/PRN-anti/solv_100`. In that case `buffer` can be omitted.
`slurm.job.qos` optionally specifies a Slurm QoS such as `highpri`.
Submission selects runs whose `equil/spec.yaml` exactly matches the supplied
configuration; it submits their existing `slurm.sh` files without rewriting them.

---

## ncoord - Write metacv.dat from dftb.inp

Generates `metacv.dat` for selected runs by reading atom indices from the `dftb.inp` coordinate block

**Inputs**
- `dftb.inp` (from equil)
- `ncoord.yaml`

**Output**
- `metacv.dat`
- `spec.yaml`

**Notes**
- `group` can be `indices`, `range`, or `all_water_H`
- `all_water_H` selects all H atoms after `solute_end` and can include extra `indices` (listed first)
- `system_dir` optionally sets a repository-relative or absolute system directory
  (e.g. `systems/PRN-anti/solv_100`); when set, `buffer` can be omitted.
  Otherwise the existing `systems/<system>/<prefix>_<buffer:.1f>` layout is used.
- `run_ids` supports a list or a range string like `"1-2"`

**Run**

```bash
qmmd ncoord configs/<molecule>/ncoord/ncoord.yaml
```

---

## lcod - Bond-distance-difference CV

```bash
qmmd lcod configs/HPD/lcod/lcod.yaml
```

Reads `lcod.type: BONDDISTANCEDIFFERENCE`, `lcod.gaussian_width` (Å), and
four one-based `lcod.atoms`. Writes `metacv.dat` and a YAML snapshot into each
selected run's `cv_dirname`, without submitting jobs. Existing target directories
are skipped unchanged; all new targets are validated before writing.
Supports either `system_dir` or the usual `system`/`buffer`/`prefix` layout.

The HPD example selects runs 1–20 and creates `meta-lcod` with:

```text
BONDDISTANCEDIFFERENCE 0.1 1 8 7 8 -3 3 0.01
```

The `lcod` mapping accepts `grid_min`, `grid_max`, and `grid_step` (all Å).
Supply all three together; bounds must increase and spacing must be positive.
They are appended after the four atom IDs in that order. The HPD example uses
−3 to +3 Å with spacing 0.01 Å. These are FES output settings, not walls or
sampling limits; coverage should be checked against the sampled CV values.
The grid is required by DCDFTBMD when `METAPRINTFES=TRUE`. Omitting all three
remains supported for workflows without FES output. Existing CV directories
are still skipped, so changing YAML does not update already generated files.

This is r(N1,H8) − r(O7,H8), tracking the fixed original H8, not whichever
proton O7 acquires from water. No coordination-number exponents or normalization
apply. This command prepares only the CV; subsequent `meta-prep` requires a
separate meta YAML with `cv_dirname: meta-lcod`. Do not use `meta-h` for that step.
This does not establish the suitability of the CV for bulk pKa estimation.

### CPP LCOD umbrella seeds and equilibration

```bash
qmmd us-lcod-pull-prep configs/CPP/us/pull.yaml
qmmd us-lcod-pull-report configs/CPP/us/pull.yaml
qmmd us-lcod-equil-prep configs/CPP/us/equil.yaml
qmmd us-lcod-equil-submit configs/CPP/us/equil.yaml
qmmd us-lcod-equil-report configs/CPP/us/equil.yaml
qmmd us-lcod-wham configs/CPP/us/wham.yaml
```

This specialized reactive-coordinate workflow selects, for every requested
umbrella center, the completed source-trajectory frame with the nearest
minimum-image `r(NB19,H20) - r(NC31,H20)`. It saves each complete system as
`window-XXX/pull/start.xyz`, plus `windows.csv` and `pull.png` recording the
target, selected LCOD, error, and source time. Reused nearest frames are
retained explicitly when a target lies outside or between sparsely sampled
source values.

The equilibration step writes zero-height metadynamics with a
`BONDDISTANCEDIFFERENCE` CV and coincident lower/upper quadratic walls. It
validates the complete prepared set before submission and asks once before
submitting one job per window. The CPP configuration spans -2.0 through +2.3
angstrom at 0.1 angstrom spacing (44 windows), using the literal DCDFTBMD wall
coefficient 500 kJ/mol/angstrom^2. Grossfield WHAM receives the converted
half-K coefficient, 239.005736 kcal/mol/angstrom^2.

The report command analyzes every complete frame of each `equil/traject` with
`cpptraj`, using the box vectors in `dftb.inp` for minimum-image geometry. It
writes `lcod.cpptraj.in`, `lcod.cpptraj.log`, and `lcod.dat` inside each window's
`equil/` directory. The aligned LCOD values are collected into `equil_lcod.csv`,
normalized per-window histograms and Gaussian visual guides into
`equil_density.csv`, and a three-tier `equil-report.png` under `us-lcod/`.
The bottom tier shows handpicked source-frame LCOD for each umbrella target;
CPP has no sequential pull-MD trajectory.

`us-lcod-wham` consumes that cpptraj-derived `equil_lcod.csv`, not biaspot,
and runs the same Grossfield WHAM executable and DCD-to-WHAM wall conversion
used by the torsional workflow. It writes `wham-equil/pmf.csv`, the input
series and metadata, and a four-tier `wham-equil/wham-report.png` (PMF,
densities, equilibration traces, handmade seeds). `pmf_smooth.csv` is a
plot-only guide; the PMF panel subtracts the lowest displayed raw or smooth
value so its y-axis starts at zero, with displayed values saved in
`pmf_plot.csv`. `pmf_features.csv` records the marked smooth-curve minima,
their gap, and both directional barrier heights. No bootstrap uncertainty is
claimed. Because these are
unequally long equilibration runs rather than independent production, the PMF
is exploratory and should not be interpreted as a converged free energy.

## 2dncoord - Write 2D metacv.dat from dftb.inp

Generates `metacv.dat` with two CV blocks back-to-back

**Inputs**
- `dftb.inp` (from equil)
- `2dncoord.yaml`

**Output**
- `metacv.dat`
- `spec.yaml`

**Notes**
- `cv1` and `cv2` each define a full CV spec (title, exponents, groups, grid)
- `run_ids` supports a list or a range string like `"1-2"`

**Run**

```bash
qmmd 2dncoord configs/<molecule>/ncoord/2dncoord.yaml
```

---

## meta-prep - Prepare Metadynamics Runs

The optional `slurm.job.qos` field is passed through to the generated
`#SBATCH --qos` line, as in `dftb-prep`.

Writes metadynamics inputs/scripts in a CV directory (created by `ncoord`)

**Inputs**
- `dftb.inp` (from equil)
- `metacv.dat`
- `meta.yaml`

**Output**
- `dftb.inp`
- `run.sh`
- `slurm.sh`
- `meta_spec.yaml`
- `restart` (copied from equil)

**Notes**
- `cv_dirname` must already exist (created by `ncoord`)
- `meta-prep` and `meta-submit` accept a repository-relative or absolute
  `system_dir`; when set, `buffer` can be omitted. Existing buffer-based paths
  remain supported.
- `run_ids` supports a list or a range string like `"1-2"`
- `MD=(... SEEDTYPE=3 RANDOMSEED=0 ...)` will replace `0` with a unique seed per run

**Run**

```bash
qmmd meta-prep configs/<molecule>/meta/meta.yaml
```

---

## meta-submit - Submit Metadynamics Runs

Submits Slurm jobs for meta directories that match the provided config (by meta_spec.yaml)

**Run**

```bash
qmmd meta-submit configs/<molecule>/meta/meta.yaml
```

---

## cv-coord - Compute Coordination from Trajectory

Computes rational coordination from xyz trajectories and optionally validates against biaspot

**Inputs**
- `traject` (equil + prod)
- `biaspot` (prod, optional for validation)
- `coord.yaml`

**Output**
- `manual-cv/coord.dat` (equil + prod)
- `manual-cv/coord_prod.dat` (prod only)
- `manual-cv/dist.dat`
- `manual-cv/dist_prod.dat`

**Notes**
- Uses formula: s(r) = (1-(r/r0)^p) / (1-(r/r0)^q)
- `run_ids` supports a list or range string like `"1-20"`
- Validation compares prod-only values to biaspot Coordinate lines

**Run**

```bash
qmmd cv-coord configs/<molecule>/cv/coord.yaml --validate
```

---
