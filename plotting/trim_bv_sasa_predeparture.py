"""Re-average saved metadynamics SASA strictly before first sampled NC-H loss."""
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


def predeparture_mask(data: np.ndarray) -> tuple[np.ndarray, int]:
    """First zero-H observation under the saved geometric bond criterion."""
    t = data['geometry_time_ps']
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Invalid geometry clock')
    count = data['nc_bonded_H_count']
    if not np.isfinite(count).all():
        raise ValueError('Missing H assignment')
    lost = np.flatnonzero(count == 0)
    if not len(lost) or lost[0] == 0:
        raise ValueError('Need a bound initial segment and an observed departure')
    index = int(lost[0])
    return np.arange(len(data)) < index, index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir',type=Path,default=Path('reports/BV/meta-hic-finalist-runs'))
    args=p.parse_args(); root=args.report_dir
    # Keep the original whole-window comparisons and fits recoverable.
    for pattern in ('metad_nc_sasa_vs_df.*','metad_nc_sasa_vs_helicity.*','metad_nc_sasa_vs_helicity_fit.*'):
        for path in root.glob(pattern):
            backup=path.with_name(path.stem+'_fullwindow'+path.suffix)
            if not backup.exists(): shutil.copy2(path,backup)
    with (root/'equil_basin_vs_meta_dg.csv').open() as handle:
        runs=list(csv.DictReader(handle))
    summaries=[]; events=[]
    for source in runs:
        run=int(source['run_id']); folder=root/f'run-{run}'
        path=folder/'metad_nc_sasa.csv'
        data=np.genfromtxt(path,delimiter=',',names=True)
        mask,index=predeparture_mask(data); selected=data[mask]
        hm,_=circular_mean_std(selected['helicity_deg'])
        record=dict(run_id=run,state='before_first_NC_H_loss',frames=len(selected),
                    nc_sasa_mean_A2=float(selected['nc_sasa_A2'].mean()),
                    nc_sasa_std_A2=float(selected['nc_sasa_A2'].std()),
                    helicity_mean_deg=float(hm),delta_F_kcal_mol=float(source['delta_G_kcal_mol']),
                    nc_heavy_only_sasa_mean_A2=float(selected['nc_heavy_only_sasa_A2'].mean()),
                    start_geometry_time_ps=float(selected['geometry_time_ps'][0]),
                    last_included_geometry_time_ps=float(selected['geometry_time_ps'][-1]),
                    departure_geometry_time_ps=float(data['geometry_time_ps'][index]),
                    departure_report_time_ps=float(data['time_ps'][index]),
                    departure_elapsed_ps=float(data['geometry_time_ps'][index]-data['geometry_time_ps'][0]))
        summaries.append(record)
        np.savetxt(folder/'metad_nc_sasa_predeparture.csv',np.column_stack([selected[n] for n in data.dtype.names]),
                   delimiter=',',header=','.join(data.dtype.names),comments='')
        prov=json.loads((folder/'metad_nc_sasa_provenance.json').read_text())
        events.append(dict(**record,source=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           bond_cutoff_A=prov['bond_cutoff_A'],definition='First sampled NC bonded-H count of zero; nearest heavy owner within saved cutoff. Strict prefix excludes loss frame and all later frames, including returns. This conservative first geometric loss can be transient, not necessarily sustained chemical deprotonation.'))
        print(f"run-{run}: cutoff {record['departure_geometry_time_ps']:.4f} ps ({record['departure_elapsed_ps']:.4f} ps elapsed), {len(selected)} retained frames",flush=True)
    (root/'metad_nc_sasa_predeparture_windows.json').write_text(json.dumps(events,indent=2)+'\n')
    write_csv(root/'metad_nc_sasa_predeparture_windows.csv',summaries)
    for suffix,field in (('df','delta_F_kcal_mol'),('helicity','helicity_mean_deg')):
        stem=root/f'metad_nc_sasa_vs_{suffix}'
        write_csv(stem.with_suffix('.csv'),summaries)
        draw(summaries,'metad — before first NC–H loss',field,stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
