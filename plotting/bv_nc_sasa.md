# BV ring-C nitrogen solvent-accessible surface area

Run `python plotting/plot_bv_nc_sasa.py` from the repository root. Outputs are
added to `reports/BV/meta-hic-finalist-runs`; existing reports are not changed.

The custom target-atom Shrake–Rupley calculation places 1,920 golden-spiral
points on NC's expanded sphere. SASA = 4π(R_N + R_probe)² times the fraction
not inside another solute atom's expanded sphere. Defaults, in angstroms:
probe 1.4; H 1.20, C 1.70, N 1.55, O 1.52. These are the corresponding
[MDTraj radii and algorithm conventions](https://mdtraj.readthedocs.io/en/latest/_modules/mdtraj/geometry/sasa.html).
This script uses its own NumPy implementation, not the MDTraj library.
SASA is reported in Å², for nitrogen atom 31 alone, not the entire N–H group.

All solute heavy atoms (within the first 78 atoms) block accessibility.
Solvent is excluded. Hydrogens capable of blocking NC are assigned to the
nearest heavy atom using periodic minimum-image distances. They block only
if that heavy atom is in the solute and within 1.3 Å. Thus transferred solvent
H can be included and departed original H excluded. NC-H counts use the same
geometric rule, not a Mulliken charge or permanent-topology definition.
Frames near proton transfer are threshold-sensitive.

CLI controls include `--probe-A`, `--sphere-points`, `--bond-cutoff-A`,
`--nitrogen-id`, `--solute-atoms`, and `--radii-json` (element→radius in Å).
The nitrogen ID is checked against the existing NC torsion definition.
Only fixed orthorhombic boxes are supported. Local blocker images are placed
relative to NC using MIC; the present boxes are larger than twice the local
surface/blocker reach. This is geometric exposure, not water occupancy or a
test of connectivity of a pocket to bulk solvent.

Equilibration and metadynamics use exactly the saved 3D-plot timestamps.
Every matched frame's three bridge torsions are checked against saved values;
trajectory hashes are checked before and after reading. Helicity is paired
by these same timestamps. Metadynamics uses saved summary windows only and
is not reweighted. All sampling includes initial relaxation where present.

The existing metadynamics summaries for runs 142/190 can label torsions from
an earlier nearby frame with the requested sample time (the old reader's
half-interval tolerance). To preserve the actual SASA–torsion pairing, the
reader searches within 0.51 of the median saved interval and accepts only
a frame reproducing all three saved torsions. `time_ps` retains the report
label; `geometry_time_ps` records the matching trajectory timestamp.
Provenance records the maximum offset. Existing summaries are not rewritten.

Per-run `{equil,metad}_nc_sasa.csv` contains time, atomic SASA, a heavy-only
control (all H excluded), NC-H count, helicity, and bridge-basin label.
Root-level plots and matching CSVs compare arithmetic mean SASA to the same
saved run-level ΔF = F(s≈0) − F(s≈1), or to circular-mean ring-center helicity.
Both axes summarize the entire saved stage window, not density-peak structures.
Different runs/stages cover different durations. The scatter plots pool
protonation states; `*_nc_sasa_by_protonation.csv` additionally separates
NH0, NH1 and NH2plus. The same run-level ΔF is copied to each stratum for
reference, not estimated independently per state. Standard deviations are
frame spreads, not standard errors. Five selected runs cannot establish a
causal exposure/free-energy relationship.

Validation includes analytic isolated-sphere/full-burial/equal-sphere-cap tests,
periodic wrapping and H-transfer tests. Every 100th analyzed frame is also
calculated with twice the surface-point count; mean/max absolute changes
are recorded in per-run provenance JSON.
