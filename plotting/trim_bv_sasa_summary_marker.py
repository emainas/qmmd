"""Use explicitly documented summary dashed markers as SASA/helicity endpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from bv_conformation_basins import circular_mean_std
from plot_bv_nc_sasa import draw, write_csv


def marker_mask(times: np.ndarray, endpoint: float) -> np.ndarray:
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Invalid report clock')
    if not times[0] < endpoint <= times[-1]:
        raise ValueError('Marker outside saved window')
    return times < endpoint - 1e-9


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir',type=Path,default=Path('reports/BV/meta-hic-finalist-runs'))
    p.add_argument('--endpoints',type=Path)
    args=p.parse_args(); root=args.report_dir
    ep=args.endpoints or root/'sasa_summary_dashed_endpoints.json'
    config=json.loads(ep.read_text())
    archive=root.parent/'meta-hic_sasa_before_summary_marker'
    for path in root.glob('metad_nc_sasa_vs_*'):
        if path.is_file():
            archive.mkdir(parents=True,exist_ok=True)
            dest=archive/path.name
            if not dest.exists(): shutil.copy2(path,dest)
    with (root/'equil_basin_vs_meta_dg.csv').open() as handle:
        dg={int(row['run_id']):float(row['delta_G_kcal_mol']) for row in csv.DictReader(handle)}
    rows=[]; provenance=[]
    for key,endpoint in config['endpoints_ps'].items():
        run=int(key)
        if run in config['excluded_runs']: continue
        path=root/f'run-{run}'/'metad_nc_sasa.csv'
        data=np.genfromtxt(path,delimiter=',',names=True)
        mask=marker_mask(data['time_ps'],float(endpoint)); subset=data[mask]
        if not len(subset) or not np.isfinite(subset['helicity_deg']).all():
            raise ValueError('Missing matched helicity')
        mean,_=circular_mean_std(subset['helicity_deg'])
        row=dict(run_id=run,state='before_summary_purple_dashed_marker',frames=len(subset),
                 nc_sasa_mean_A2=float(subset['nc_sasa_A2'].mean()),nc_sasa_std_A2=float(subset['nc_sasa_A2'].std()),
                 helicity_mean_deg=float(mean),delta_F_kcal_mol=dg[run],
                 nc_heavy_only_sasa_mean_A2=float(subset['nc_heavy_only_sasa_A2'].mean()),
                 start_report_time_ps=float(subset['time_ps'][0]),last_report_time_ps=float(subset['time_ps'][-1]),
                 endpoint_report_time_ps=endpoint,endpoint_elapsed_ps=endpoint-float(subset['time_ps'][0]))
        rows.append(row)
        np.savetxt(path.with_name('metad_nc_sasa_before_summary_marker.csv'),
                   np.column_stack([subset[n] for n in data.dtype.names]),delimiter=',',
                   header=','.join(data.dtype.names),comments='')
        provenance.append(dict(**row,source=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               marker_evidence=config['evidence'][key],
                               selection='report time_ps strictly less than dashed marker; retain paired SASA/helicity geometry; no bond-distance filtering.'))
        print(f'run-{run}: start to {endpoint} ps ({row["endpoint_elapsed_ps"]:.4f} ps elapsed), {len(subset)} frames')
    write_csv(root/'metad_nc_sasa_summary_marker_windows.csv',rows)
    (root/'metad_nc_sasa_summary_marker_windows.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for suffix,field in (('df','delta_F_kcal_mol'),('helicity','helicity_mean_deg')):
        stem=root/f'metad_nc_sasa_vs_{suffix}'
        write_csv(stem.with_suffix('.csv'),rows)
        write_csv(stem.with_name(stem.name+'_plotted.csv'),rows)
        draw(rows,'metad — before summary dashed marker',field,stem.with_suffix('.png'))


if __name__=='__main__': main()
