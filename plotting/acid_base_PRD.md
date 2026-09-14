# PRD acid-base summary

`acid_base_PRD.py` is a copied/adapted HIST summary, with a dedicated copied
`prd_competitor_analysis.py` helper. Existing HIST files are unchanged.
PRD here is pyridinium (N1, C2–C6; labeled proton H7), not pyrimidinium.
The full solvated simulation has zero total charge because of the counterion;
the solute is the positive ion.

All six solute heavy atoms are shown in the competitor column: Mulliken
charges, geometric paths to the solvent defect, and rational coordination
summed over all H (r0=1.6 Å, exponents 6/12). C–H coordination includes the
ordinary ring hydrogens. Carbon paths are exploratory geometric contacts,
not conventional hydrogen bonds or evidence of a protonated carbon state.
No fictitious histidine chi1/chi2 conformer column is drawn.

The inherited CLI is retained; `--hist-parm` and histidine conformer options
are not needed for PRD. The current PRD atom ordering is explicitly checked.
The distance/angle path cutoffs remain configurable via the inherited CLI.

For run 10, `reports/PRD_meta-h_cv_20runs.csv` records a +20 ps display offset.
User-selected t_diff=47 ps therefore becomes `--diffusive-start 27` for raw
trajectory/Mulliken/FES clocks. Reports use raw times throughout. The analysis
window ends at raw 28.75 ps (grid 48.75 ps). Experimental pKa=5.23 is supplied
by the user; temperature=300 K. The inherited cation estimator is
[F(s≈0)−F(s≈1)]/[RT ln(10)], not the reversed PRN convention.

The run-specific command and clock mapping are recorded in
`reports/PRD/meta-h_selected/run-10/provenance.json`. Plotted numerical data
are saved beside `summary.png`. Defect input uses the most positive solvent
O in the Mulliken window [-0.625,-0.525] e, with exact trajectory/charge
timestamp matching and interpolated bias coordination. Missing assignments
are not interpreted as absence of proton transfer. F(s) minima estimates
are not integrated thermodynamic state populations.
