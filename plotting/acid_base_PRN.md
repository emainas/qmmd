# PRN acid/base summary

The fourth column now also has a new-proton acquisition distance panel below
the dihedrals. It tracks sustained uniquely owned O4–Hnew and O5–Hnew bonds,
excluding original H11, after the existing purple sustained labeled-H-loss
marker. The same configurable distance, ownership-margin, and persistence
criteria apply. Bond intervals are labelled by hydrogen ID; missing/ambiguous
intervals are gaps, not proof of an unprotonated site. If no loss marker is
detected, the panel says so rather than assuming a departure. Full bond-assignment
data for both sites are retained in `proton_dihedrals.csv`, including before the
marker. Existing FES/event/defect logic is unchanged.

`acid_base_PRN.py` is a PRN-specific copy/adaptation of `acid_base_HIST.py`.
The HIST original is unchanged. Snapshot reading, timestamp alignment, and batch
screening live in `prn_acid_base_analysis.py`.

From the repository root, in the qmmd environment:

```bash
python plotting/acid_base_PRN.py
python plotting/acid_base_PRN.py --runs 3,5,6 --out-dir /tmp/prn-check
python plotting/acid_base_PRN.py --screen-only --out-dir /tmp/prn-screen
python plotting/acid_base_PRN.py --runs-path systems/PRN-syn/solv_100/dftb/N1T48C1 --out-dir reports/PRN-syn/acid_base --dihedral-center 0
```

Defaults select PRN-anti runs 1–20, `meta-h`, acid oxygen 4, hydrogen 11,
and dihedral 5–3–4–11 (one-based). The fourth column overlays unsmoothed,
bond-resolved traces: blue 5–3–4–11 for the original O4–H11 bond and orange
4–3–5–Hnew for a hydrogen acquired by O5. Hnew is identified dynamically and
may be H11 itself or a different hydrogen. A fifth column shows
competitor oxygen 5: its Mulliken charge, its hydrogen-bond wire to the same
solvent defect, and its rational coordination summed over every hydrogen.
The competitor graph includes O5 and solvent oxygens (not O4 as an intermediate),
matching HIST's endpoint-specific definition. It searches up to three bridging
waters by default (`--competitor-max-bridging-waters`). Missing defect assignments
produce NaN wire states, not disconnection. All-H coordination uses
`sum[1/(1+(r/r0)^6)]`, r0=1.6 Å (`--competitor-refdist`), with minimum-image
distances and no permanent hydrogen ownership. `--competitor-oxygen` changes the
one-based endpoint. `competitor.csv` contains every plotted competitor sample.
Dihedral visibility is now independent of the diffusion/deprotonation markers.
Each H is assigned to its closest heavy atom only when distance <=1.4 Å
(`--bond-cutoff`) and the next closest heavy atom is >=0.15 Å farther away
(`--bond-ownership-margin`). An assignment must persist continuously for >=0.05 ps
(`--bond-persistence-ps`); accepted intervals include their first frame after
retrospective confirmation. Short/ambiguous intervals are gaps, not proof of
deprotonation. The old trace can reappear if O4 reacquires H11. The new trace is
omitted when O5 has multiple assigned H atoms. Lines never connect different
hydrogen identities or cross the angular wrapping seam.

`prn_proton_dihedrals.py` contains this geometry-based detection. It does not use
Mulliken charges or fixed water membership. `proton_dihedrals.csv` saves one-based
H IDs, candidate O5 assignments, O–H distances, and signed dihedral values for
every frame in the displayed window. Its NaN angles mark excluded frames;
`new_assigned_H_count` distinguishes multiple candidates. Candidate distances may
be present even when persistence is not met. `aligned.csv` retains the unmasked
old raw dihedral and the bond-masked plotted old dihedral.
`--dihedral-center 0` displays syn on -180–180 degrees (anti defaults to 0–360).
Only the metadynamics segment is analyzed here. Axes retain DFTB's reported
timestamps; `screening.csv` instead reports elapsed time since the first meta frame.

The inherited HIST detection heuristic selects the most positive solvent-oxygen
Mulliken charge within [-0.625, -0.525] e. The solvent defect may change identity;
water/proton ownership is not fixed. Missing assignments stay NaN in plots.
For event detection only, missing defect distances are interpolated between
available endpoints, exactly as in HIST. Long unassigned intervals therefore
limit event certainty. This charge window is inherited, not independently
calibrated for PRN.

The separated state requires acid-O/defect-O distance >=4 Å and labeled O4–H11
coordination <=0.05 for >=0.05 ps. The diffusion marker requires subsequent
coordination >=0.20, still separated, for >=0.05 ps. This is an operational
diffusion marker, not proof of acid reprotonation or unrestricted bulk diffusion.
Coordination is interpolated from bias timestamps, without extrapolation.
Mulliken and trajectory samples are matched by timestamp. FES surfaces are
matched to bias timestamps by Gaussian count. Incomplete trailing frames are ignored.

The original three summary columns retain water wires, defect distances,
coordination, Mulliken charges, FES, and apparent pKa. Statistics use 1.75 ps
after diffusion; a full window is required before plotting. Conversion temperature
is 300 K. PRN uses `[F(min near s=0) - F(min near s=1)] / (0.004576*T)`,
with F in kcal/mol and T in K: the shared `deltaf` convention, without the former
PRN-specific sign reversal. Minima windows and statistical sampling are unchanged.
Older reports retain their recorded convention until explicitly regenerated.
The FES-derived pKa is an apparent estimator, not a validated equilibrium
prediction. H bonds use short heavy–H <=1.3 Å, H–acceptor <=2.5 Å, D–H–A >=135°.
Minimum-image geometry uses the orthorhombic simulation box.

Use `--help` for configurable thresholds, atom IDs, temperature, and window length.
Manual diffusion times can be supplied with repeated `--diffusion-start RUN:PS`
options, where PS is on the raw DFTB-reported clock. `--diffusion-clock-note`
records the origin of a converted visual selection. Manual mode bypasses the
automatic separated-state/recovery screen. If a preceding sustained labeled-H
coordination loss is detected, the purple marker denotes that loss alone,
not confirmed defect separation; the black marker is explicitly labeled MANUAL.
Otherwise the purple marker is omitted, and both the plot and provenance report that no proton-loss marker
was detected. A manual marker does not establish physical bulk diffusion.
Missing defect assignments remain missing even with a manual diffusion override.
Partial-run refreshes preserve screening records for other runs.
Output goes to `reports/PRN-anti/acid_base/`: screening CSV plus one directory
per eligible run containing `summary.png`, aligned numerical CSVs, FES/pKa data,
and `provenance.json`. Refreshing overwrites only these generated analysis outputs.
Inputs are read into temporary fixed-size snapshots; jobs and raw data are untouched.
Restart subdirectories are deliberately rejected until explicit stitching is supplied.
