# pKa grid time axis

`plot_pka_grid.py` shares `infer_offset_ps` with `plot_cv_grid.py`.
Both default to `--time-axis restart-aligned`: actual plotted x values are raw
DFTB times plus the run-specific parent-clock offset. Tick labels do not apply
any additional transformation. A 40 ps equilibration restart therefore appears
at approximately 40 ps, with the first hill slightly later.

Use `--time-axis raw` for the original DFTB-reported clock. `--debug` prints both
time ranges and the applied offset. Each PNG saves a matching CSV with raw and
plotted timestamps, delta F, pKa, temperature, and clock convention. `--data-out`
sets an alternate CSV location.

```bash
python plotting/plot_pka_grid.py --runs-path systems/HPD/solv_5.5/dftb/N1T48C1 --cv-dir meta-h --temp 300 --debug
```

This change does not alter the FES minima selection, pKa sign, temperature default
(313.15 K), or existing FES/bias loading functions used by other analysis scripts.
Set `--temp` to the intended simulation temperature. Grid layouts support one run,
one row, and multi-row selections. See `plot_cv_grid.md` for clock prerequisites.

This is a time-axis correction, not validation of the pKa estimator or of the
legacy FES/restart pairing. Files containing reset-clock continuations or missing
interior FES blocks require a separate alignment audit before interpretation.
