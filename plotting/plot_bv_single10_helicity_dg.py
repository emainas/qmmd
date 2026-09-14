"""Plot saved metadynamics ΔG against equilibration single10 and helicity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acid_base_BV import calculate_bv_torsions
from bv_conformation_basins import circular_mean_std


def draw(rows: list[dict], field: str, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        x, y = row[field], row['delta_G_kcal_mol']
        ax.scatter(x, y, s=85, color='#326c9b', zorder=3)
        ax.annotate(f"Run {row['run_id']}", (x, y), xytext=(8, 8),
                    textcoords='offset points', fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$\Delta G = F(s\approx0)-F(s\approx1)$ (kcal mol$^{-1}$)')
    ax.set_title('BV HIC finalists: equilibration conformation vs metadynamics ΔG')
    ax.margins(x=.25, y=.2)
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .025, 'Equilibration basin circular means; ΔG from saved F(s) snapshots.\n'
             f'Descriptive comparison of {len(rows)} runs, not a fitted relationship.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=Path('reports/BV/meta-hic-finalist-runs'))
    parser.add_argument('--parm', type=Path, default=Path('systems/BV/solv_4.0/salt/ready.parm7'))
    parser.add_argument('--solute-atoms', type=int, default=78)
    args = parser.parse_args()
    with (args.report_dir / 'equil_basin_vs_meta_dg.csv').open() as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for source in source_rows:
        run = int(source['run_id'])
        folder = args.report_dir / f'run-{run}'
        provenance = json.loads((folder / 'equil_single_bridge_3d_provenance.json').read_text())
        trajectory = Path(provenance['source'])
        if trajectory.parent.name != 'equil':
            raise ValueError('Only equilibration trajectories are allowed')
        with trajectory.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != provenance['sha256']:
            raise ValueError(f'Equilibration snapshot changed: {trajectory}')
        saved = np.genfromtxt(folder / 'equil_single_bridge_3d.csv', delimiter=',', names=True)
        box = np.diag(np.asarray(provenance['box_vectors_A']))
        series, helicity = calculate_bv_torsions(trajectory, saved['time_ps'], args.parm,
                                                args.solute_atoms, box)
        if not np.isfinite(helicity).all():
            raise ValueError(f'Unmatched or degenerate helicity frames: run-{run}')
        for name in ('single5', 'single10', 'single15'):
            error = (series[name] - saved[f'{name}_deg'] + 180) % 360 - 180
            if not np.allclose(error, 0, atol=1e-7):
                raise ValueError(f'Torsion alignment failed: run-{run} {name}')
        mask = saved['basin_id'] == int(source['equil_basin_id'])
        mean, std = circular_mean_std(helicity[mask])
        rows.append(dict(run_id=run, delta_G_kcal_mol=float(source['delta_G_kcal_mol']),
                         single10_mean_deg=float(source['single10_mean_deg']),
                         helicity_mean_deg=float(mean), helicity_circular_std_deg=float(std),
                         equil_basin_id=int(source['equil_basin_id']), frames=int(mask.sum())))
        np.savetxt(folder / 'equil_helicity.csv',
                   np.column_stack([saved['time_ps'], saved['basin_id'], helicity]),
                   delimiter=',', header='time_ps,basin_id,helicity_deg', comments='')
        print(f'run-{run}: helicity {mean:.3f} ± {std:.3f} degrees; alignment verified', flush=True)
    for name, field, xlabel in (
        ('single10', 'single10_mean_deg', 'Equilibration basin single10 circular mean (degrees)'),
        ('helicity', 'helicity_mean_deg', 'Equilibration basin A–B–C–D ring-center helicity circular mean (degrees)'),
    ):
        stem = args.report_dir / f'equil_{name}_vs_meta_dg'
        with stem.with_suffix('.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        draw(rows, field, xlabel, stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
