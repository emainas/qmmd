"""Apply four-run/predeparture presentation consistently; preserve old reports."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from plot_bv_basin_dg import draw as triplet
from plot_bv_single10_helicity_dg import draw as scalar
from plot_bv_nc_sasa import draw as sasa, write_csv

ROOT = Path('reports/BV/meta-hic-finalist-runs')
ARCHIVE = ROOT.parent/'meta-hic_before_four_run_predeparture'


def selected(path: Path) -> list[dict]:
    with path.open() as handle:
        raw = list(csv.DictReader(handle))
    rows=[]
    for row in raw:
        if int(row['run_id']) == 190: continue
        parsed={}
        for key,value in row.items():
            try: parsed[key]=float(value)
            except ValueError: parsed[key]=value
        parsed['run_id']=int(parsed['run_id']); rows.append(parsed)
    return rows


def main() -> None:
    # Archive only generated report artifacts in this explicitly scoped directory.
    for pattern in ('*.png','*.csv','*.json','run-*/metad_single_bridge_3d*','run-*/summary.png','run-190/*.png'):
        for source in ROOT.glob(pattern):
            if not source.is_file(): continue
            target=ARCHIVE/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists(): shutil.copy2(source,target)
    for source in list(ROOT.glob('*_fullwindow.png'))+list(ROOT.glob('run-*/summary.png'))+list(ROOT.glob('run-190/*.png')):
        target=ARCHIVE/source.relative_to(ROOT)
        if source.exists():
            # Archive was verified/copied above; remove only identical archived files.
            if source.read_bytes()!=target.read_bytes(): raise ValueError(f'Archive differs: {source}')
            source.unlink()
    subprocess.run([sys.executable,'plotting/trim_bv_sasa_summary_marker.py'],check=True)
    subprocess.run([sys.executable,'plotting/plot_bv_metad_basins.py','--summary-marker','--exclude-run-ids','190','--overwrite'],check=True)
    path=ROOT/'equil_basin_vs_meta_dg.csv'; rows=selected(path)
    triplet(rows,path.with_suffix('.png'))
    write_csv(path.with_name(path.stem+'_plotted.csv'),rows)
    for name,field,xlabel in (
        ('single10','single10_mean_deg','Equilibration basin single10 circular mean (degrees)'),
        ('helicity','helicity_mean_deg','Equilibration basin A–B–C–D helicity circular mean (degrees)')):
        path=ROOT/f'equil_{name}_vs_meta_dg.csv'; rows=selected(path)
        scalar(rows,field,xlabel,path.with_suffix('.png'))
        write_csv(path.with_name(path.stem+'_plotted.csv'),rows)
    for stage in ('equil','metad'):
        for suffix,field in (('df','delta_F_kcal_mol'),('helicity','helicity_mean_deg')):
            path=ROOT/f'{stage}_nc_sasa_vs_{suffix}.csv'; rows=selected(path)
            write_csv(path.with_name(path.stem+'_plotted.csv'),rows)
            label = ('metad — before summary dashed marker' if rows[0].get('state') == 'before_summary_purple_dashed_marker'
                     else 'metad — before first NC–H loss')
            sasa(rows,stage if stage=='equil' else label,field,path.with_suffix('.png'))
    for stage in ('equil','metad'):
        subprocess.run([sys.executable,'plotting/fit_bv_sasa_helicity.py','--stages',stage,
                        '--exclude-run-ids','190','--window-label',
                        ('Before summary purple dashed marker' if (ROOT/'sasa_summary_dashed_endpoints.json').exists()
                         else 'Before first sampled NC–H loss') if stage=='metad' else 'Full equilibration'],check=True)
    subprocess.run([sys.executable,'plotting/fit_bv_dihedral_plane.py',
                    '--report-dir',str(ROOT),'--exclude-run-ids','190'],check=True)
    manifest=dict(included_run_ids=[39,57,99,142],excluded_run_ids=[190],
                  metad_window='Metadynamics start strictly before purple dashed summary marker; same report timestamps for conformation, SASA and helicity.',
                  archive=str(ARCHIVE.resolve()), pngs=[str(p.relative_to(ROOT)) for p in sorted(ROOT.rglob('*.png'))],
                  unchanged='Raw simulations, original per-frame data, and F(s) minima differences preserved. Equilibration averaging unchanged.')
    (ROOT/'active_png_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__': main()
