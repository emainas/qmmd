# HPD integrated acid-base summary

`acid_base_HPD.py` adapts the PRD/HIST summary layout to HPD: biased N1–H8,
12 solute atoms, and comparison sites N1 and O7 (one-based). It retains the three acid-base
columns and adds N1/O7 Mulliken charges, defect water-wire connectivity, and
ordinary overlaid time-series lines for coordination summed over all hydrogens
in the same PNG, without tier offsets or a coordination colorbar. No unrelated histidine
dihedrals or carbon competitors are included.

`hpd_competitor_analysis.py` supplies timestamp-matched, minimum-image geometry.
Missing solvent-defect assignments stay missing, not disconnected. The inherited
charge window is an operational identifier, not proof of defect absence.
Water-wire criteria and maximum bridge counts remain CLI-configurable. The
all-H coordination uses the inherited reference distance 1.6 Å and exponents 6/12.

Use the command saved in each run's `provenance.json`. User grid times are
20 ps higher than the raw summary clock for these runs. Manual onsets do not
certify diffusion. The 1.75 ps window, cation-oriented FES-minimum pKa convention,
and 300 K conversion are unchanged. Numerical series are saved beside the PNG.

The selected HPD runs are in `reports/HPD/meta-h_selected/`; `summary.png`
is now the canonical integrated figure. Older separate competitor figures are
retained as supplementary outputs, not required to read the summary.

The final column shows the unsmoothed N1–C2–O7–Hnew dihedral in [-180,180]°.
Hydrogens are assigned to their closest heavy atom only within 1.4 Å and with
a 0.15 Å advantage over the next heavy atom. Exactly one H must be assigned to
O7 continuously for at least 0.05 ps. These values are configurable through
`--oh-bond-cutoff`, `--oh-ownership-margin`, and `--oh-persistence-ps`.
Accepted intervals include their first frame retrospectively. Missing/ambiguous
assignments and identity/wrapping changes break the plotted line; missing does
not establish deprotonation. No original water ownership is assumed.
`summary_O7H_dihedral.csv` saves timestamps, candidate and accepted one-based
H IDs, candidate count, O–H distance and the plotted angle at every saved
trajectory frame through the display endpoint.
