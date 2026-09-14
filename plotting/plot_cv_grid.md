# CV grid time axis

Use `--ylim MIN MAX` to set shared vertical limits in the CV's own units.
The default remains `0 2`. Limits must be finite and increasing. For HPD LCOD:

```bash
python plotting/plot_cv_grid.py --runs-path systems/HPD/solv_5.5/dftb/N1T48C1/ --cv-dir meta-lcod --ylim -3 3
```

This changes display limits only, not CV values or time alignment. The existing
`s` axis label and `coordination` CSV column name are unchanged for compatibility;
LCOD values in those outputs are in angstrom.

The default `--time-axis restart-aligned` shifts the actual plotted times onto
the parent trajectory's clock. Each run and each CV directory has its own offset.
There is no tick-label-only transformation and no shared formatter state.

The initial `traject` step/time is the anchor. Its time on the parent clock is
`parent_first_time + (initial_step - parent_first_step) * parent_timestep`.
The parent comes from `replica_dirname` in the saved meta spec (default `equil`).
This requires a retained step counter and a constant parent timestep. It handles
PRN's 1 fs equilibration to 0.5 fs metadynamics transition and early restart copies.
It does not use the final equilibration time, the first Gaussian as the restart
instant, or values guessed from the mutable binary restart file.

An explicit `restart_time_ps` in the saved spec, or `restart_time_ps.txt`, gives
the first CV trajectory frame's time on the parent clock. This can anchor a reset
clock. Insufficient metadata raises an error rather than silently inventing a
shift. `--time-axis raw` displays the DFTB-reported timestamps without this mapping.
For a multi-stage parent whose own clock needs correction, supply an explicit
anchor; automatic mapping refers to the immediate parent's reported clock.

All existing CLI options remain available. Each PNG now has a matching CSV
containing run ID, CV directory, raw time, plotted time, coordination, offset, and
clock convention. `--data-out` overrides its location. `--debug` prints both clocks.

```bash
python plotting/plot_cv_grid.py --runs-path systems/PRN-syn/solv_100/dftb/N1T48C1 --cv-dir meta-h --debug
python plotting/plot_cv_grid.py --runs-path systems/PRN-syn/solv_100/dftb/N1T48C1 --cv-dir meta-h --time-axis raw --out /tmp/prn_raw.png
```

Bias samples are paired within timestamp blocks and read from fixed-size snapshots.
Incomplete final samples are ignored; missing interior CV values and nonmonotonic
timestamps are rejected. Absolute-clock continuation files retain only newer
samples. Ambiguous reset-clock continuations require explicit stitching instead
of automatic concatenation at the last deposited hill.

Earlier CV grids are not valid clock references. Existing acid-base plots and
manual analysis windows are not changed when this grid is regenerated; their
documented raw-clock selections remain the same physical trajectory samples.
