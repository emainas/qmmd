# Equilibration site–water RDFs

Load `ml amber/26`, then run:

```bash
python plotting/tetrapyrrole_equil_rdf.py --system BV
python plotting/tetrapyrrole_equil_rdf.py --system BPP
python plotting/tetrapyrrole_equil_rdf.py --system CPP
```

Defaults use all available complete equilibration frames from N1T48C1 runs
1–200 for BV and 1–50 for BPP/CPP. There is no 10 ps burn-in discard.
`--runs`, `--discard-ps`, `--spacing` (Angstrom), `--rmax` (Angstrom),
`--temperature` (K), and `--workers` are configurable. Output goes to
`reports/<SYSTEM>/equil_site_water_rdf`; existing output directories are refused.
Use `--report-root /tmp/new-validation-path` for independent validation.

One-based reference atoms are resolved by topology names:

| System | Pair 1 | Pair 2 | Intact topology waters |
| --- | --- | --- | --- |
| BV | HB20–water O | HC32–water O | 263 |
| BPP | HC31–water O | NB19–water H | 264 |
| CPP | HB20–water O | NC31–water H | 264 |

BV atoms 163–164 form a two-atom hydroxide residue misleadingly labelled WAT;
it is excluded from the water set. Water membership is fixed by topology, not
reassigned when protons exchange. Water-H pairs include both H atoms (528 targets
for BPP/CPP), with hydrogen number density, not oxygen number density.

For each reference/target pair, cpptraj computes a minimum-image raw pair histogram,
RDF, and cumulative coordination in the fixed orthorhombic box taken from dftb.inp.
The 7 Angstrom default maximum must not exceed half the shortest box side.
Bins default to 0.05 Angstrom. Exact shell volumes are
4*pi/3*(r_outer^3-r_inner^3). The pooled RDF is:

g_i = sum_run(count_i) / [shell_volume_i * sum_run(frames * N_target / V_run)].

Each pair has one reference atom. The masks are disjoint, so no N-1 correction
is used. Number density uses the entire periodic box, not solvent-accessible
volume. There is no empirical tail rescaling. Finite-box and solute-exclusion
effects can leave g(r) different from one over the displayed radius range.

w(r) = -RT ln[g(r)] in kcal/mol at 300 K, unshifted: w=0 when g=1. The RDF
already includes shell-volume normalization; no extra radial Jacobian is applied.
Zero-count PMF bins are NaN, without smoothing or pseudocounts. This is a pooled
structural pair PMF, not a standard-state binding free energy or a convergence claim.

Every run's cpptraj normalization and coordination integral are independently
audited against raw counts. All pair counts from the first selected trajectory
are also checked with a separate NumPy minimum-image implementation. Each input
trajectory is read as a size-bounded snapshot; incomplete final frames are omitted.
Source files are never modified. Timestamp ranges, frame counts, box lengths,
selection IDs, and hashes are saved in provenance.json.

The PNG has two panels with two curves each, using lefteris.mplstyle. Numerical
data are consolidated in rdf_pmf.csv and per_run_rdf.csv. One example cpptraj
input/log is retained; temporary snapshots and per-run cpptraj output files are
automatically cleaned up. No statistical uncertainty is estimated.
