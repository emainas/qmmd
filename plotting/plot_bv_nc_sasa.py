"""NC SASA versus saved DF and helicity for finalist equil/metad windows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acid_base_BV import iter_xyz_frames, _signed_dihedral_deg
from bv_conformation_basins import circular_mean_std
from bv_nc_sasa import RADII, nc_sasa, sphere_points
from plot_bv_equil_basins import LABELS


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def draw(rows: list[dict], stage: str, field: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in rows:
        x, y = row['nc_sasa_mean_A2'], row[field]
        ax.scatter(x, y, s=80, color='#326c9b', zorder=3)
        offset = (6, -16) if stage == 'equil' and field == 'helicity_mean_deg' and row['run_id'] in (39, 57) else (6, 8)
        ax.annotate(f"Run {row['run_id']}", (x, y), xytext=offset, textcoords='offset points')
    ax.set_xlabel('Mean NC solvent-accessible surface area (Å²)')
    ax.set_ylabel('Helicity circular mean (degrees)' if field == 'helicity_mean_deg'
                  else r'$\Delta F=F(s\approx0)-F(s\approx1)$ (kcal mol$^{-1}$)')
    ax.set_title(f'BV HIC finalists · {stage} · NC solvent exposure')
    ax.margins(x=.25, y=.25); ax.grid(alpha=.2)
    fig.text(.5, .025, '1.4 Å probe; dynamic solute-bound H; all sampled protonation states pooled.\n'
             'Run-level averages, not fitted relationships. Metadynamics sampling is unreweighted.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, 1)); fig.savefig(path, dpi=200); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir', type=Path, default=Path('reports/BV/meta-hic-finalist-runs'))
    p.add_argument('--runs-path', type=Path, default=Path('systems/BV/solv_4.0/dftb/N1T48C1'))
    p.add_argument('--probe-A', type=float, default=1.4)
    p.add_argument('--sphere-points', type=int, default=1920)
    p.add_argument('--bond-cutoff-A', type=float, default=1.3)
    p.add_argument('--nitrogen-id', type=int, default=31)
    p.add_argument('--solute-atoms', type=int, default=78)
    p.add_argument('--radii-json', type=Path, help='Optional element radii in angstroms')
    p.add_argument('--resume', action='store_true', help='Reuse this script\'s completed, hash-validated per-frame outputs')
    args = p.parse_args()
    if args.probe_A <= 0 or args.bond_cutoff_A <= 0:
        p.error('Probe and bond cutoff must be positive')
    radii = RADII if args.radii_json is None else json.loads(args.radii_json.read_text())
    points = sphere_points(args.sphere_points)
    finer = sphere_points(args.sphere_points*2)
    with (args.report_dir/'equil_basin_vs_meta_dg.csv').open() as handle:
        sources = list(csv.DictReader(handle))
    for stage in ('equil', 'metad'):
        summary, strata = [], []
        for source in sources:
            run = int(source['run_id']); folder = args.report_dir/f'run-{run}'
            saved = np.genfromtxt(folder/f'{stage}_single_bridge_3d.csv', delimiter=',', names=True)
            if stage == 'equil':
                hs = np.genfromtxt(folder/'equil_helicity.csv', delimiter=',', names=True)
                if not np.allclose(hs['time_ps'], saved['time_ps'], atol=1e-9, rtol=0):
                    raise ValueError('Helicity times do not match')
                helicity = hs['helicity_deg']
            else:
                helicity = saved['helicity_deg']
            eq = json.loads((folder/'equil_single_bridge_3d_provenance.json').read_text())
            box = np.diag(np.array(eq['box_vectors_A']))
            traj = args.runs_path/f'run-{run}'/('equil' if stage == 'equil' else 'meta-hic')/'traject'
            inp = traj.parent/'dftb.inp'
            text = inp.read_text()
            tv = np.array([[float(v) for v in line.split()[1:4]] for line in text.splitlines() if line.startswith('TV')])
            if not np.allclose(tv, np.diag(box)) or 'NPT=TRUE' in text.upper().replace(' ', ''):
                raise ValueError('Need matching fixed orthorhombic box')
            before = digest(traj)
            with traj.open() as handle:
                count = int(handle.readline()); handle.readline()
                elements = np.array([handle.readline().split()[0] for _ in range(count)])
            target = args.nitrogen_id - 1
            if target + 1 != eq['quartets_one_based']['single15'][0] or elements[target] != 'N':
                raise ValueError('Target is not ring-C nitrogen')
            rows, errors, i = [], [], 0
            cached = folder/f'{stage}_nc_sasa.csv'
            cached_provenance = folder/f'{stage}_nc_sasa_provenance.json'
            reuse = args.resume and cached.exists() and cached_provenance.exists()
            if reuse:
                old = json.loads(cached_provenance.read_text())
                for key, value in dict(sha256=before, nitrogen_id=args.nitrogen_id, radii_A=radii,
                                      probe_A=args.probe_A, sphere_points=args.sphere_points,
                                      bond_cutoff_A=args.bond_cutoff_A).items():
                    if old[key] != value: raise ValueError(f'Cached settings/source changed: {key}')
                with cached.open() as handle:
                    rows = [{k: float(v) for k,v in row.items()} for row in csv.DictReader(handle)]
                if len(rows) != len(saved) or not np.allclose([r['time_ps'] for r in rows], saved['time_ps'], atol=1e-9, rtol=0):
                    raise ValueError('Cached times changed')
                for row in rows: row.setdefault('geometry_time_ps', row['time_ps'])
                i = len(rows)
                errors = [old['mean_abs_refinement_difference_A2']]
            tolerance = .51 * float(np.median(np.diff(saved['time_ps'])))
            for time, xyz in (() if reuse else iter_xyz_frames(traj)):
                if time is None:
                    raise ValueError('Missing timestamp')
                wanted = saved['time_ps'][i]
                if time < wanted - tolerance:
                    continue
                if time > wanted + tolerance:
                    raise ValueError(f'No matching geometry near timestamp {wanted}: {traj}')
                matches = True
                for name in LABELS:
                    value = _signed_dihedral_deg(xyz[np.array(eq['quartets_one_based'][name])-1], box)
                    if abs((value - saved[f'{name}_deg'][i] + 180) % 360 - 180) > 1e-6:
                        matches = False
                if not matches:
                    continue
                full, heavy, nh = nc_sasa(xyz, elements, box, target, args.solute_atoms,
                                          args.probe_A, points, args.bond_cutoff_A, radii)
                if i % 100 == 0:
                    refined = nc_sasa(xyz, elements, box, target, args.solute_atoms,
                                      args.probe_A, finer, args.bond_cutoff_A, radii)[0]
                    errors.append(abs(full-refined))
                rows.append(dict(time_ps=float(wanted), geometry_time_ps=time, nc_sasa_A2=full, nc_heavy_only_sasa_A2=heavy,
                                 nc_bonded_H_count=nh, helicity_deg=helicity[i], basin_id=int(saved['basin_id'][i])))
                i += 1
                if i == len(saved):
                    break
            if i != len(saved) or digest(traj) != before:
                raise ValueError('Incomplete match or changed trajectory')
            write_csv(folder/f'{stage}_nc_sasa.csv', rows)
            for state in ('all', 'NH0', 'NH1', 'NH2plus'):
                subset = [r for r in rows if state == 'all' or
                          (state == 'NH0' and r['nc_bonded_H_count'] == 0) or
                          (state == 'NH1' and r['nc_bonded_H_count'] == 1) or
                          (state == 'NH2plus' and r['nc_bonded_H_count'] >= 2)]
                if not subset:
                    continue
                h = np.array([r['helicity_deg'] for r in subset]); h = h[np.isfinite(h)]
                hm, _ = circular_mean_std(h)
                s = np.array([r['nc_sasa_A2'] for r in subset])
                record = dict(run_id=run, state=state, frames=len(subset), nc_sasa_mean_A2=float(s.mean()),
                              nc_sasa_std_A2=float(s.std()), helicity_mean_deg=float(hm),
                              delta_F_kcal_mol=float(source['delta_G_kcal_mol']),
                              nc_heavy_only_sasa_mean_A2=float(np.mean([r['nc_heavy_only_sasa_A2'] for r in subset])))
                strata.append(record)
                if state == 'all': summary.append(record)
            provenance = dict(trajectory=str(traj.resolve()), sha256=before, nitrogen_id=args.nitrogen_id,
                              radii_A=radii, probe_A=args.probe_A, sphere_points=args.sphere_points,
                              bond_cutoff_A=args.bond_cutoff_A, settings=vars(args).copy(),
                              convergence_check_frames=len(errors), mean_abs_refinement_difference_A2=float(np.mean(errors)),
                              max_abs_refinement_difference_A2=float(max(errors)),
                              definition='NC expanded sphere area times unoccluded fraction. Solute heavy atoms and dynamically owned H are blockers; solvent excluded. Nearest-heavy ownership within bond cutoff; MIC geometry.',
                              interpretation='Direct N atomic SASA, not N-H group area or solvent occupancy. All-state means pool protonation; separate strata and heavy-only control provided. No metadynamics reweighting.')
            if reuse:
                provenance = old
            provenance['max_abs_geometry_time_offset_ps'] = float(max(abs(r['time_ps']-r['geometry_time_ps']) for r in rows))
            provenance['alignment'] = 'Matched all three saved bridge torsions within 0.51 of the median saved interval; time_ps is the summary label and geometry_time_ps the actual matching trajectory timestamp.'
            cached_provenance.write_text(json.dumps(provenance, indent=2, default=str)+'\n')
            print(f'{stage} run-{run}: {i} frames; SASA={summary[-1]["nc_sasa_mean_A2"]:.3f} Å²; refinement MAE={np.mean(errors):.3f}', flush=True)
        write_csv(args.report_dir/f'{stage}_nc_sasa_by_protonation.csv', strata)
        for suffix, field in (('df', 'delta_F_kcal_mol'), ('helicity', 'helicity_mean_deg')):
            stem = args.report_dir/f'{stage}_nc_sasa_vs_{suffix}'
            write_csv(stem.with_suffix('.csv'), summary)
            draw(summary, stage, field, stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
