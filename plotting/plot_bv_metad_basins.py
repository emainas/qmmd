"""Metadynamics conformation basins from saved summary torsions, without reweighting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bv_conformation_basins import circular_mean_std, density_basins, wrap_degrees
from plot_bv_equil_basins import LABELS, draw

STEM = 'metad_single_bridge_3d'


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(rows: list[dict], kind: str, path: Path, exclude_run_ids: tuple[int, ...] = (190,), predeparture: bool = False, summary_marker: bool = False) -> None:
    rows = [r for r in rows if int(r['run_id']) not in exclude_run_ids]
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.25), 6))
    if kind == 'basin':
        xs = np.arange(len(rows))
        labels = ['(' + ', '.join(f"{r[f'{n}_mode_deg']:.0f}" for n in LABELS) + ')' for r in rows]
        ax.set_xticks(xs, labels)
        xlabel = 'Metadynamics density-peak triplet (single5, single10, single15), degrees'
    else:
        field = f'{kind}_mode_deg' if kind in LABELS else 'helicity_mean_deg'
        xs = [r[field] for r in rows]
        xlabel = (f'Metadynamics {kind} density-peak coordinate (degrees)' if kind in LABELS
                  else 'Metadynamics basin A–B–C–D helicity circular mean (degrees)')
    for i, (x, row) in enumerate(zip(xs, rows)):
        ax.scatter(x, row['delta_G_kcal_mol'], s=80, color='#326c9b', zorder=3)
        ax.annotate(f"Run {row['run_id']} · B{row['basin_id']}", (x, row['delta_G_kcal_mol']),
                    xytext=(0, 12), textcoords='offset points', ha='center', fontsize=9)
    ax.set_xlabel(xlabel, labelpad=12)
    ax.set_ylabel(r'$\Delta F = F(s\approx0)-F(s\approx1)$ (kcal mol$^{-1}$)')
    ax.set_title('BV HIC finalists: metadynamics conformation vs ΔF' +
                 ('\nBefore purple dashed summary marker; ' if summary_marker else '\nBefore first sampled NC–H loss; ' if predeparture else '\n') +
                 ('Excluded runs: ' + ', '.join(map(str, exclude_run_ids)) if exclude_run_ids else ''))
    ax.margins(x=.2, y=.3)
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .025, 'Saved summary sampling only; biased and unreweighted. Labels are run-local.\n'
             'ΔG is the saved run-level F(s) minima difference, not a separate estimate per conformational basin.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir', type=Path, default=Path('reports/BV/meta-hic-finalist-runs'))
    p.add_argument('--bin-deg', type=float, default=10.)
    p.add_argument('--bandwidth-deg', type=float, default=15.)
    p.add_argument('--merge-deg', type=float, default=30.)
    p.add_argument('--min-population', type=float, default=.05)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--predeparture', action='store_true', help='Use exactly the saved predeparture SASA report timestamps')
    p.add_argument('--summary-marker', action='store_true', help='Use the same before-summary-marker timestamps as the SASA/helicity analysis')
    p.add_argument('--exclude-run-ids', nargs='*', type=int, default=[190])
    args = p.parse_args()
    if args.predeparture and args.summary_marker:
        p.error('Choose only one sampling window')
    settings = {k: getattr(args, k) for k in ('bin_deg', 'bandwidth_deg', 'merge_deg', 'min_population')}
    dg_path = args.report_dir / 'equil_basin_vs_meta_dg.csv'
    with dg_path.open() as handle:
        dg_rows = list(csv.DictReader(handle))
    if not args.overwrite and (any(args.report_dir.glob('metad_*')) or
                              any(args.report_dir.glob(f'run-*/{STEM}*'))):
        raise FileExistsError('Metadynamics outputs exist; pass --overwrite to replace them')
    results = []
    for dg in dg_rows:
        run = int(dg['run_id'])
        if run in args.exclude_run_ids:
            continue
        folder = args.report_dir / f'run-{run}'
        source = folder / 'summary_bv_torsions.csv'
        data = np.genfromtxt(source, delimiter=',', names=True)
        time = data['time_ps']
        phi = np.column_stack([data[f'{n}_deg'] for n in LABELS])
        if len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError(f'Invalid summary timestamps: {source}')
        valid = np.isfinite(phi).all(axis=1)
        if args.predeparture or args.summary_marker:
            selection_path = folder / ('metad_nc_sasa_before_summary_marker.csv' if args.summary_marker else 'metad_nc_sasa_predeparture.csv')
            selected = np.genfromtxt(selection_path, delimiter=',', names=True)
            valid &= np.isin(np.round(time, 8), np.round(selected['time_ps'], 8))
            if valid.sum() != len(selected):
                raise ValueError(f'Selected-window timestamp mismatch: run-{run}')
        labels, basins, density = density_basins(phi[valid], **settings)
        if not basins:
            raise ValueError(f'No major basin: run-{run}')
        results.append((run, folder, source, data, valid, labels, basins, density, dg))
        print(f'run-{run}: {valid.sum()} frames, {time[valid][0]:.3f}–{time[valid][-1]:.3f} ps, '
              f'{len(basins)} basins', flush=True)
    pooled = np.concatenate([np.column_stack([r[3][f'{n}_deg'][r[4]] for n in LABELS]) for r in results])
    reference, _ = circular_mean_std(pooled)
    display = reference + wrap_degrees(pooled - reference)
    limits = np.column_stack([display.min(axis=0) - 10, display.max(axis=0) + 10])
    comparison = []
    for run, folder, source, data, valid, labels, basins, density, dg in results:
        time = data['time_ps'][valid]
        phi = np.column_stack([data[f'{n}_deg'][valid] for n in LABELS])
        helicity = data['helicity_deg'][valid]
        np.savetxt(folder / f'{STEM}.csv',
                   np.column_stack([np.flatnonzero(valid) + 1, time, phi, labels, density, helicity]),
                   delimiter=',', comments='', header='source_row,time_ps,single5_deg,single10_deg,single15_deg,basin_id,smoothed_cell_probability,helicity_deg')
        basin_rows = []
        for b in basins:
            h = helicity[(labels == b['label']) & np.isfinite(helicity)]
            if not len(h):
                raise ValueError(f'No finite helicity: run-{run} B{b["label"]}')
            hm, hs = circular_mean_std(h)
            row = dict(basin_id=b['label'], frames=b['count'], population=b['population'])
            for kind in ('mode', 'mean', 'circular_std'):
                row.update({f'{n}_{kind}_deg': v for n, v in zip(LABELS, b[kind + '_deg'])})
            row.update(helicity_mean_deg=float(hm), helicity_circular_std_deg=float(hs), helicity_frames=len(h))
            basin_rows.append(row)
            comparison.append(dict(run_id=run, delta_G_kcal_mol=float(dg['delta_G_kcal_mol']),
                                   **row, fes_snapshot_time_ps=float(dg['fes_snapshot_time_ps'])))
        write_csv(folder / f'{STEM}_basins.csv', basin_rows)
        provenance = dict(source=str(source.resolve()), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          settings=settings, input_rows=len(data), valid_torsion_rows=int(valid.sum()),
                          omitted_torsion_rows=int((~valid).sum()), time_range_ps=[float(time[0]), float(time[-1])],
                          basins=basin_rows, display_reference_deg=reference.tolist(),
                          common_display_limits_deg=limits.tolist(),
                          dg_source=str(dg_path.resolve()), dg_source_sha256=hashlib.sha256(dg_path.read_bytes()).hexdigest(),
                          scope=('Start to before first sampled NC-H loss; timestamps matched to metad_nc_sasa_predeparture.csv.' if args.predeparture else 'All finite torsion rows in the saved metadynamics summary. Original report timestamps retained.'),
                          limitations='Biased unreweighted frame fractions; grid-dependent modes, not equilibrium populations. Multiple basins share the same run-level DG. Helicity is a separate conditional circular mean, not part of the 3D peak.')
        (folder / f'{STEM}_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        if args.summary_marker:
            selection_path = folder / 'metad_nc_sasa_before_summary_marker.csv'
            provenance.update(scope='Metadynamics start strictly before purple dashed summary marker; exact same report timestamps as SASA/helicity.',
                              selection_source=str(selection_path.resolve()),
                              selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                              endpoints_source=str((args.report_dir/'sasa_summary_dashed_endpoints.json').resolve()))
            (folder / f'{STEM}_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        draw(folder / f'{STEM}.png', run, time, phi, labels, basins, reference, limits, settings,
             stage='metadynamics — before summary dashed marker' if args.summary_marker else 'metadynamics — before first NC–H loss' if args.predeparture else 'metadynamics', show_modes=True)
    for kind in ('basin', 'single5', 'single10', 'single15', 'helicity'):
        stem = args.report_dir / f'metad_{kind}_vs_meta_dg'
        write_csv(stem.with_suffix('.csv'), comparison)
        write_csv(stem.with_name(stem.name + '_plotted').with_suffix('.csv'),
                  [row for row in comparison if row['run_id'] not in args.exclude_run_ids])
        compare(comparison, kind, stem.with_suffix('.png'), tuple(args.exclude_run_ids), args.predeparture, args.summary_marker)


if __name__ == '__main__':
    main()
