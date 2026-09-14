# HPD LCOD summary

The merged figure now has **four rows**, adding the O7–C2–N1–H8 proper
torsion (one-based quartet 7–2–1–8). The periodic dihedral uses minimum-image
bond vectors. Each phase reports circular mean and circular SD
`sqrt(-2 ln |mean(exp(i phi))|)`, converted to degrees. Displayed angles are
centered at 0°; wrap-crossing line segments are broken without removing those
samples from statistics. Values and statistics are in `summary_ocnh_timeseries.csv`
and `summary_ocnh_stats.csv`. The third row also annotates only O7's arithmetic
mean ± sample SD (`ddof=1`), separately per phase, with values saved in
`summary_oxygen_charge_stats.csv`. All these SDs describe fluctuations, not
uncertainty of a mean. All complete finite samples are used, without burn-in
removal. The fixed-H torsion is not a bonded torsion after H8 detaches.

The bending legend reports arithmetic mean ± sample standard deviation
(`ddof=1`) of finite signed improper deviations, separately for H8/O7 and
equilibration/metadynamics, using every available frame of each phase.
These are bounded, planar-centered bending variables, not circular means
of raw torsions. SD describes sampled fluctuations, not uncertainty in the
mean; trajectory autocorrelation is not corrected. Statistics and frame
counts are saved in `summary_bending_stats.csv`.

## Current merged timeline

The figure now has **one column with three rows**. Equilibration retains
its raw times; metadynamics is plotted at
`meta_raw_time - first_meta_raw_time + last_equil_raw_time`.
For run 1 this means equilibration 0–40 ps and metadynamics 40–60 ps.
A dashed line marks 40 ps. Distance and bending traces use blue/green for
equilibration and orange/purple for metadynamics. Mulliken atom colors are
lighter in equilibration and darker in metadynamics. No segment is drawn
between phases, and both endpoint samples are retained at the boundary.
`summary_merged_timeseries.csv` contains phase ID, plotted and raw timestamps,
and all displayed values. Existing per-phase CSVs retain raw timestamps.
This supersedes the side-by-side layout described below.

## Current figure: equilibration versus metadynamics

`summary.png` now compares the complete available `equil` trajectory (left)
against `meta-lcod` (right), replacing the former FES/CV/minimum-difference
column. Both columns show distances, signed H8/O7 improper deviations, and
all solute Mulliken charges. Each row shares vertical limits. Each phase
uses its own actual raw timestamps, without assuming an intended duration
from a newer YAML configuration. `--equil-dir` overrides the sibling equil
directory. Geometry and charges use the same definitions in both phases.
The displayed values are saved in `equil_comparison_timeseries.csv` and
`metad_comparison_timeseries.csv`; phase details are recorded in provenance.
Older FES and planarity products are retained, but are not in this figure.

## Usage and retained diagnostics

Run from the repository root:

```bash
python plotting/tautomer_HPD_lcod.py --run 1
```

Writes `reports/HPD/meta-lcod/run-1/summary.png`, numerical CSVs and
`provenance.json`. Re-running refreshes these derived files only.
`--runs-path`, `--cv-dir`, `--topology`, and `--out` override locations.
Only the selected run's base metadynamics directory is analyzed; restart
subdirectories are not stitched automatically.

The left column shows the last complete FES (minimum shifted to zero), the
fixed-H LCOD, and the time-dependent difference between the O-side and N-side
FES grid minima. LCOD = r(N1,H8) − r(O7,H8), in Å. The configurable
`--basin-split 0` separates negative/N-side and positive/O-side grid points;
the split point is excluded. FES energies are converted from Hartree to kcal/mol.
This difference is a bias-derived diagnostic, not an acid pKa, an integrated
basin free energy, or evidence that both tautomers were sampled. Unsampled
grid regions can give misleading minimum differences.

The right column shows N1–H8 (blue) and O7–H8 (red), signed H8/O7 improper bending, and
all 12 atomic Mulliken charges on a common unshifted charge axis. Oxygen ID 7
has topology name O2; the biased hydrogen ID 8 is named HN1.
The ring N1–C2–C3–C4–C5–C6 and closure bonds are verified from MOL2.
The bending panel uses improper quartets **8–2–1–6 (H8)** and
**7–3–2–1 (O7)**. Their local ring planes are C2–N1–C6 and C3–C2–N1.
Magnitude is the absolute deviation of each improper from the nearest planar
value, `abs((phi+90) modulo 180 - 90)`. The sign is the substituent's displacement
along its local plane normal, oriented to the area normal of the ordered
ring 1–2–3–4–5–6. Thus positive values indicate the same oriented ring side
for both substituents, independent of molecular rotation and periodic wrapping.
These are improper deviations in degrees, **not** bond-to-plane angles or
distances from a globally fitted plane. Values are bounded by ±90°; folding
about the nearest planar value does not retain rotations beyond 90°.
Local planes may differ when the ring puckers. H8 remains a fixed labeled atom:
after N–H dissociation its angle is no longer a bonded N–H bending coordinate.
Degenerate geometry returns NaN. No correlation coefficient is inferred from
visual overlap. `summary_out_of_plane.csv` includes raw and signed angles.

For compatibility, the original planarity CSV and column remain available
but are no longer plotted. The PRD `ring_metrics` implementation is reused: six consecutive cyclic
torsions, each transformed as abs((phi + 90) modulo 180 − 90), then averaged
within each frame. Values range from 0° (planar) to 90°; a degenerate torsion
makes the frame metric undefined. This measures ring distortion, not H8's
out-of-plane displacement.

All complete base-trajectory frames are included without smoothing or
burn-in removal. Minimum-image geometry uses the orthorhombic input box.
Mulliken s/p orbital contributions are summed per atom and joined to the
trajectory by timestamp; absent frames remain NaN. FES blocks are matched to
bias samples by Gaussian count. Original raw DFTB times are used everywhere,
without adding equilibration duration again. The calculated LCOD is checked
against bias values at shared timestamps to 0.001 Å. No diffusion detection,
proton ownership switching, or simulation-file changes are performed.

CSV outputs contain all displayed values plus the individual ring torsions
and deviations. `provenance.json` records the atom IDs, quartets, sampling
range, box, missing charge frames, and LCOD consistency error.
