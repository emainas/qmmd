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
