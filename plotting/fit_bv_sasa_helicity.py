"""OLS fits of run-mean NC SASA (y) against circular-mean helicity (x)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, t


def fit_line(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x) == 0:
        raise ValueError('Need at least three finite observations with varying x')
    f = linregress(x, y)
    residual = y - (f.intercept + f.slope*x)
    critical = t.ppf(.975, len(x)-2)
    return dict(n=len(x), slope=float(f.slope), intercept=float(f.intercept),
                pearson_r=float(f.rvalue), r_squared=float(f.rvalue**2),
                adjusted_r_squared=float(1-(1-f.rvalue**2)*(len(x)-1)/(len(x)-2)),
                slope_p_two_sided=float(f.pvalue), slope_stderr=float(f.stderr),
                slope_ci95_low=float(f.slope-critical*f.stderr),
                slope_ci95_high=float(f.slope+critical*f.stderr),
                rmse_A2=float(np.sqrt(np.mean(residual**2))),
                residual_standard_error_A2=float(np.sqrt(np.sum(residual**2)/(len(x)-2))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir', type=Path, default=Path('reports/BV/meta-hic-finalist-runs'))
    p.add_argument('--exclude-run-ids', nargs='*', type=int, default=[])
    p.add_argument('--stages', nargs='+', choices=('equil','metad'), default=['equil','metad'])
    p.add_argument('--window-label', default='', help='Optional description shown in the fit title')
    args = p.parse_args()
    for stage in args.stages:
        source = args.report_dir/f'{stage}_nc_sasa_vs_helicity.csv'
        with source.open() as handle:
            rows = list(csv.DictReader(handle))
        kept = [r for r in rows if int(r['run_id']) not in args.exclude_run_ids]
        x = np.array([float(r['helicity_mean_deg']) for r in kept])
        y = np.array([float(r['nc_sasa_mean_A2']) for r in kept])
        if np.ptp(x) >= 180:
            raise ValueError('Angular range crosses a possible branch cut; linear fit needs an explicit unwrapping choice')
        stats = fit_line(x, y)
        stem = args.report_dir/f'{stage}_nc_sasa_vs_helicity_fit'
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(x, y, s=75, color='#326c9b', zorder=3)
        for i, r in enumerate(kept):
            offset = (6, -17) if stage == 'equil' and int(r['run_id']) in (39,57) else (6,8)
            ax.annotate(f"Run {r['run_id']}", (x[i], y[i]), xytext=offset, textcoords='offset points')
        grid = np.linspace(x.min(), x.max(), 200)
        prediction = stats['intercept'] + stats['slope']*grid
        ci = t.ppf(.975,len(x)-2)*stats['residual_standard_error_A2']*np.sqrt(1/len(x)+(grid-x.mean())**2/np.sum((x-x.mean())**2))
        ax.plot(grid, prediction, color='#b25b24', label='OLS fit')
        ax.fill_between(grid, prediction-ci, prediction+ci, color='#b25b24', alpha=.13, label='95% CI of mean line')
        ax.set_xlabel('Helicity circular mean (degrees)')
        ax.set_ylabel('Mean NC SASA (Å²)')
        ax.set_title(f'BV HIC · {stage} · SASA versus helicity' + (f'\n{args.window_label}' if args.window_label else ''))
        ax.grid(alpha=.2); ax.margins(x=.2, y=.25)
        ax.legend(loc='lower right', fontsize=9)
        caption = (f"SASA = {stats['slope']:.4f} × helicity {stats['intercept']:+.4f}\n"
                   f"n = {stats['n']} runs   |   R² = {stats['r_squared']:.3f}   |   adjusted R² = {stats['adjusted_r_squared']:.3f}\n"
                   f"Pearson r = {stats['pearson_r']:.3f}   |   slope p (two-sided) = {stats['slope_p_two_sided']:.4g}\n"
                   f"Slope 95% CI: [{stats['slope_ci95_low']:.4f}, {stats['slope_ci95_high']:.4f}] Å²/degree   |   RMSE = {stats['rmse_A2']:.3f} Å²")
        fig.text(.12, .10, caption, fontsize=10, linespacing=1.5)
        excluded = ', '.join(map(str,args.exclude_run_ids)) or 'none'
        fig.text(.12, .025, f'Excluded runs: {excluded}. OLS assumes independent, normal, constant-variance residuals.\n'
                 'Small, selected sample; p tests zero slope, not linearity. Angular means treated locally as linear x; x uncertainty ignored.', fontsize=8)
        fig.tight_layout(rect=(0,.26,1,1)); fig.savefig(stem.with_suffix('.png'),dpi=200); plt.close(fig)
        stats.update(included_run_ids=[int(r['run_id']) for r in kept], excluded_run_ids=args.exclude_run_ids,
                     window_label=args.window_label,
                     source=str(source.resolve()), x='helicity_mean_deg', y='nc_sasa_mean_A2',
                     limitations='Unweighted run-level OLS; ignores uncertainty in means and x. Selected sample, pooled protonation, unreweighted metadynamics. Zero-slope test is not a linearity test.')
        stem.with_suffix('.json').write_text(json.dumps(stats,indent=2)+'\n')
        with stem.with_suffix('.csv').open('w',newline='') as handle:
            fields=['run_id','included_in_fit','helicity_mean_deg','nc_sasa_mean_A2','predicted_sasa_A2','residual_A2']
            writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
            for row in rows:
                xx,yy=float(row['helicity_mean_deg']),float(row['nc_sasa_mean_A2'])
                pred=stats['intercept']+stats['slope']*xx
                writer.writerow(dict(run_id=row['run_id'],included_in_fit=int(int(row['run_id']) not in args.exclude_run_ids),
                                     helicity_mean_deg=xx,nc_sasa_mean_A2=yy,predicted_sasa_A2=pred,residual_A2=yy-pred))
        print(stage, json.dumps(stats),flush=True)


if __name__ == '__main__':
    main()
