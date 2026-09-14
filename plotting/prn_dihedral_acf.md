# PRN dihedral autocorrelation panel

Current normalized-correlation view replaces the biexponential with tail-only
`B*exp(-t/tau_slow)` fits on 0.15–1, 0.15–1.5 and 0.15–2 ps. The first 0.1 ps
is marked as a transient and excluded. B is extrapolated to t=0, not the tail
start. The added fourth panel shows ln of positive normalized tail values and
the transformed nonlinear fits; separate log-linear R² values quantify
linearity. Current reproduction: use `--acf-ratio-panel --acf-xmax 2` with the
anti/autocorrelation options. `_acf_slow_tail_fits.csv`, `_curves.csv`, and
`.json` save B, tau, windows, both fit diagnostics and numerical curves.
The fitted tau varies from about 0.89 to 1.30 ps across windows; this and
semilog curvature limit interpretation as a single robust relaxation time.

Underdamped comparison: add `--underdamped --acf-xmax 1 --acf-ymin 0.945`.
This fits the user's harmonic cosine model with fixed histogram sigma, only
gamma and omega_d adjustable. Four inclusive windows (0.2, 0.3, 0.5, 1 ps)
are compared; the 0.3 ps fit is red dashed. Initial frequency comes from
the median spacing of local maxima in the first 0.2 ps (0.04 ps here).
Multiple starts check optimizer sensitivity. `_acf_underdamped_fits.csv`,
`_curves.csv` and `.json` save parameters, full curves and bounds/definitions.
For this anti dataset all four fits approach omega_d=0; oscillation periods
are unresolved and huge numerical values reflect the lower parameter bound,
not a physical measurement. The single-mode fixed-variance model fits poorly.

Latest figure: `python plotting/fit_prn_equil_pmf.py --states anti --autocorrelation --acf-fit-max-ps 5 --acf-ymin 0.945 --acf-fit-plateau`.
This fits both the plateau (bounded between 0 and 1) and positive tau over
0–5 ps inclusive, retaining C(0)=1. The blue dashed histogram prediction
remains independent of the red fitted curve. JSON records the plateau
difference and relative difference. No parameter confidence intervals are
claimed because the lag points are correlated.

Run `python plotting/fit_prn_equil_pmf.py --states anti --autocorrelation`.
This updates `reports/PRN-anti/prn_anti_equil_pmf_parabola.png`, retaining
the existing PMF, parabola, density, Gaussian and table, and adding the ACF.

Each trajectory independently uses the requested uncentered cosine formula:
`C[m] = sum(cos(theta[n] - theta[n+m])) / (N-m)`, with radians inside cosine.
Zero-padded FFT correlations of sine and cosine implement this exact linear
correlation efficiently; trajectories are never concatenated. No baseline
subtraction, smoothing, or additional normalization is applied.

The saved source snapshot must match the PMF's SHA256. Runs and initial-time
cutoff are taken from its provenance, so anti uses 49 runs, excluding run 2.
Every run must have uniform, identical time spacing and finite angles.
The common lag range ends at the shortest trajectory's maximum lag, ensuring
all 49 curves contribute equally at every point. The band is plus/minus one
sample standard deviation across runs (ddof=1), not standard error or a
confidence interval. Long lags have fewer time origins and are less reliable.
A nonzero plateau is expected for angles confined within a conformer basin;
it does not by itself establish persistent dynamical memory.

Outputs beside the PNG:

- `prn_anti_equil_pmf_acf_runs.csv`: each run's full ACF and origin counts.
- `prn_anti_equil_pmf_acf.csv`: common lags, mean, standard deviation, band, run count.
- `prn_anti_equil_pmf_acf.json`: formula, sampling, source hash and limitations.

Validation: direct sums versus FFT on synthetic wrapped angles and selected
real run-1 lags, including the final lag; constant-angle check; C(0)=1;
49 contributors throughout. Numerical test functions passed directly;
pytest was unavailable in the qmmd environment. Generated PNG visually checked.

The semilog fourth panel is now disabled. A red dashed curve in the C(t)
panel fits `C_infinity+(1-C_infinity)*exp(-t/tau)` directly by unweighted
least squares with positive tau. Plateau and amplitude are fixed. All finite
lag values strictly below 35 ps are included, even values below the plateau.
The curve is displayed over the full lag range. `_acf_exponential.csv` and
`_acf_exponential.json` record the curve, fit mask, tau and diagnostics.

The previous (retained, no longer regenerated) fourth-panel data plotted
`ln[(mean C(t)-C_infinity)/(1-C_infinity)]`, where
`C_infinity=exp(-sigma_rad^2)` uses the pooled histogram's angular variance.
It fits an unweighted line constrained through zero, slope `-1/tau`, to
finite log values at lags strictly below 35 ps. Nonpositive excess correlations
are left missing, never clipped or replaced with a pseudocount. The cutoff is
configurable with `--acf-fit-exclude-from-ps`. CSV and JSON files ending in
`_acf_log_decay` save the selected points, fitted curve, tau, centered R-squared,
and log-space RMSE. These statistics are descriptive, since lag points are
correlated and the plateau is estimated; a poor fit is not a relaxation-time
measurement. The actual anti fit is poor and has negative centered R-squared.
