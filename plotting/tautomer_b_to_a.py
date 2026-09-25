"""Replot APP's saved summary as FES/CV/estimator plus NB/NA charge/LCOD/coordination.

Regenerates numerical summary products from source; calculates fixed-H LCOD
directly from the source trajectory with periodic minimum-image distances.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prn_anti_dih import TIME, box_from_input, minimum_image
from acid_base_BV import BV_DIHEDRAL_ATOM_NAMES, _signed_dihedral_deg, _circular_running_average_deg


def refresh_full_run(p: Path, provenance: dict, nb: int, na: int) -> float:
    """Regenerate displayed observables through the requested post-marker window."""
    from prn_acid_base_analysis import xyz_frames, bias_samples
    from cpp_competitor_analysis import rational_coordination, calculate_histidine_charges
    from acid_base_CPP import load_pka_series, save_pka_csv, load_fes_snapshot, save_fes_snapshot_csv
    source = Path(provenance['source'])
    box = box_from_input(source/'dftb.inp')
    with (source/'traject').open() as handle:
        count = int(handle.readline()); handle.readline()
        symbols = [handle.readline().split()[0] for _ in range(count)]
    hs = np.array([i for i,s in enumerate(symbols) if s=='H'])
    bt, bs, _ = bias_samples(source/'biaspot')
    rows, coord = [], []
    for t, xyz in xyz_frames(source/'traject'):
        if t > provenance['raw_stop_ps'] + 1e-8:
            break
        rows.append([t, np.interp(t,bt,bs,left=np.nan,right=np.nan)])
        coord.append([t, *[rational_coordination(np.linalg.norm(minimum_image(xyz[hs]-xyz[i-1],box),axis=1)) for i in (nb,na)]])
    times = np.array(rows)[:,0]
    if not len(times) or np.any(np.diff(times)<=0): raise ValueError('Invalid trajectory timestamps')
    stop = float(provenance['raw_stop_ps'])
    if stop-times[-1] > np.median(np.diff(times))+1e-8:
        raise ValueError('Trajectory does not cover the requested window')
    np.savetxt(p/'summary_tautomer_timeseries.csv',rows,delimiter=',',header='time_ps,coordination_s',comments='')
    np.savetxt(p/'summary_app_competitors.csv',coord,delimiter=',',header='time_ps,NB_all_H_coordination,NA_all_H_coordination',comments='')
    water_ids = [i+1 for i,s in enumerate(symbols) if i>=77 and s=='O']
    endpoints = {'NB':nb,'NA':na,**{f'O{i}':i for i in water_ids}}
    qt, q = calculate_histidine_charges(source/'mulliken',endpoints,77,stop)
    lookup = {round(float(t),8):i for i,t in enumerate(qt)}
    charge_rows = [[t,*[q[label][lookup[round(float(t),8)]] if round(float(t),8) in lookup else np.nan for label in ('NB','NA')]] for t in times]
    np.savetxt(p/'summary_app_heavy_charges.csv',charge_rows,delimiter=',',header='time_ps,NB_charge,NA_charge',comments='')
    water_rows = [[t,*[q[f'O{atom}'][lookup[round(float(t),8)]]
                       if round(float(t),8) in lookup else np.nan for atom in water_ids]] for t in times]
    np.savetxt(p/'summary_water_oxygen_charges.csv',water_rows,delimiter=',',
               header=','.join(['time_ps']+[f'O{i}_charge_e' for i in water_ids]),comments='')
    pt, pk = load_pka_series(source/'fes.dat',temperature=provenance['temperature_K'])
    if pt[-1] < stop:
        raise ValueError('FES does not cover the requested window')
    # Retain an interpolated endpoint for exact-window statistical sampling.
    keep = pt < stop
    save_pka_csv(p/'summary_pka.csv',np.r_[pt[keep],stop],np.r_[pk[keep],np.interp(stop,pt,pk)])
    ft, fes = load_fes_snapshot(source/'fes.dat',stop)
    save_fes_snapshot_csv(p/'summary_fes_snapshot.csv',ft,fes)
    return stop


def bridge_series(source: Path, topology: Path, times: np.ndarray) -> tuple[np.ndarray, dict]:
    sections = topology.read_text().split('@<TRIPOS>')
    atoms = [line.split() for line in next(s for s in sections if s.startswith('ATOM')).splitlines()[1:] if line.strip()]
    names = [a[1] for a in atoms]
    def resolve(name: str) -> int:
        if names.count(name) != 1:
            raise ValueError(f'Non-unique topology name {name}')
        return names.index(name)
    quartets = {key: [resolve(n) for n in group] for key, group in BV_DIHEDRAL_ATOM_NAMES.items()}
    rings = {r: [resolve(n+r) for n in ('N', 'C1', 'C2', 'C3', 'C4')] for r in 'ABCD'}
    bonds = {frozenset(int(v)-1 for v in line.split()[1:3]) for line in next(s for s in sections if s.startswith('BOND')).splitlines()[1:] if line.strip()}
    for group in list(quartets.values()) + [g+[g[0]] for g in rings.values()]:
        if any(frozenset((a,b)) not in bonds for a,b in zip(group,group[1:])):
            raise ValueError(f'Topology does not connect requested quartet/ring {group}')
    box = box_from_input(source/'dftb.inp')
    target = {round(float(t),8): i for i,t in enumerate(times)}
    values = np.full((len(times),7), np.nan)
    with (source/'traject').open() as handle:
        while line := handle.readline():
            n = int(line); match = TIME.search(handle.readline())
            if match is None: raise ValueError('Missing timestamp')
            t = float(match[1])/1000
            if t > times[-1]+1e-8: break
            fields = [handle.readline().split() for _ in range(n)]
            if any(len(f)<4 for f in fields): break
            idx = target.get(round(t,8))
            if idx is None: continue
            xyz = np.array([list(map(float,f[1:4])) for f in fields])
            centers = []
            for group in rings.values():
                points = xyz[group]
                centers.append((points[0]+minimum_image(points-points[0],box)).mean(axis=0))
            values[idx] = [_signed_dihedral_deg(xyz[g],box) for g in quartets.values()] + [_signed_dihedral_deg(np.array(centers),box)]
    if not np.all(np.isfinite(values)): raise ValueError('Unmatched/invalid conformation samples')
    return values, {'quartets': {k:[i+1 for i in g] for k,g in quartets.items()}, 'ring_atom_ids':{k:[i+1 for i in g] for k,g in rings.items()}}


def load_csv(path: Path) -> np.ndarray:
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))


def lcod_series(source: Path, ids: tuple[int, int, int], stop: float) -> np.ndarray:
    """Return time, NB-H, NA-H, and their difference (ps and angstrom)."""
    box = box_from_input(source/'dftb.inp')
    indices = np.array(ids)-1
    rows = []
    with (source/'traject').open() as handle:
        while line := handle.readline():
            n = int(line)
            match = TIME.search(handle.readline())
            if match is None:
                raise ValueError('Missing trajectory timestamp')
            t = float(match[1])/1000
            if t > stop+1e-8:
                break
            fields = [handle.readline().split() for _ in range(n)]
            if any(len(f) < 4 for f in fields):
                break
            if [fields[i][0] for i in indices] != ['N', 'N', 'H']:
                raise ValueError('Expected NB, NA, H atom elements')
            xyz = np.array([list(map(float, fields[i][1:4])) for i in indices])
            distances = np.linalg.norm(minimum_image(xyz[:2]-xyz[2], box), axis=1)
            rows.append([t, *distances, distances[0]-distances[1]])
    data = np.array(rows)
    if not len(data) or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError('Missing or nonmonotonic LCOD samples')
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('reports/taut_app_meta_hib/run-5'))
    parser.add_argument('--nb', type=int, default=18)
    parser.add_argument('--na', type=int, default=1)
    parser.add_argument('--hydrogen', type=int, default=19, help='Fixed, one-based hydrogen ID; no ownership switching')
    parser.add_argument('--topology', type=Path, default=Path('systems/APP/init/app.mol2'))
    parser.add_argument('--smoothing-window-ps', type=float, default=.10, help='Circular averaging window for conformation plots (ps)')
    parser.add_argument('--pka-samples',type=int,default=10,help='Equally spaced interpolated samples across the post-marker window')
    parser.add_argument('--window-ps',type=float,default=1.75,help='Post-marker analysis window, ps')
    parser.add_argument('--lcod-bin-width',type=float,default=.10,help='LCOD probability-density histogram bin width (angstrom)')
    parser.add_argument('--source',type=Path,default=Path('systems/APP/solv_4.0/dftb/N1T64C1/run-5/meta-hib'))
    parser.add_argument('--grid-onset-ps',type=float,default=48.,help='Marker on plot_cv_grid clock, ps')
    parser.add_argument('--temperature',type=float,default=300.,help='Estimator conversion temperature, K')
    args = parser.parse_args()
    if not np.isfinite(args.window_ps) or args.window_ps <= 0:
        parser.error('Window must be positive and finite')
    if min(args.nb, args.na, args.hydrogen) < 1 or len({args.nb, args.na, args.hydrogen}) != 3:
        parser.error('Three distinct positive atom IDs required')
    p = args.report
    if not np.isfinite(args.smoothing_window_ps) or args.smoothing_window_ps <= 0:
        parser.error('Smoothing window must be positive and finite')
    if args.pka_samples<2: parser.error('At least two pKa samples required')
    if not np.isfinite(args.lcod_bin_width) or args.lcod_bin_width <= 0:
        parser.error('LCOD bin width must be positive and finite')
    from plot_cv_grid import parse_biaspot, infer_offset_ps
    source = args.source.resolve()
    if (source/'metad-restart').exists():
        raise ValueError('Restart directory needs explicit stitching')
    sections = args.topology.read_text().split('@<TRIPOS>')
    atoms = [line.split() for line in next(s for s in sections if s.startswith('ATOM')).splitlines()[1:] if line.strip()]
    bonds = {frozenset(map(int,line.split()[1:3])) for line in next(s for s in sections if s.startswith('BOND')).splitlines()[1:] if line.strip()}
    for atom_id, name in ((args.nb,'NB'),(args.na,'NA'),(args.hydrogen,'HB')):
        if atoms[atom_id-1][1] != name:
            raise ValueError(f'Topology atom {atom_id} is not {name}')
    if len(atoms)!=77 or frozenset((args.nb,args.hydrogen)) not in bonds:
        raise ValueError('Expected 77 solute atoms and an NB-HB bond')
    from acid_base_CPP import read_xyz_symbols
    symbols = read_xyz_symbols(source/'traject')
    expected = [a[5].split('.')[0][0].upper() for a in atoms]
    if symbols[:77] != expected:
        raise ValueError('Solute trajectory elements differ from topology')
    bt,_ = parse_biaspot(source/'biaspot')
    offset = infer_offset_ps(source.parent,source.name,float(bt[0]))
    provenance = dict(source=str(source),grid_onset_ps=args.grid_onset_ps,
                      raw_onset_ps=args.grid_onset_ps-offset,grid_minus_raw_ps=offset,
                      competitors={'NB':args.nb,'NA':args.na},solute_atoms=77,
                      donor_hydrogen_id=args.hydrogen,temperature_K=args.temperature,
                      topology=str(args.topology.resolve()))
    provenance.update(raw_stop_ps=provenance['raw_onset_ps']+args.window_ps,
                      grid_stop_ps=args.grid_onset_ps+args.window_ps,
                      window_ps=args.window_ps)
    p.mkdir(parents=True,exist_ok=True)
    (p/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if (args.nb, args.na) != (provenance['competitors']['NB'], provenance['competitors']['NA']):
        parser.error('NB/NA IDs must agree with the saved charge/coordination products')
    onset = provenance['raw_onset_ps']
    stop = refresh_full_run(p,provenance,args.nb,args.na)
    summary = load_csv(p/'summary_tautomer_timeseries.csv')
    charges = load_csv(p/'summary_app_heavy_charges.csv')
    coord = load_csv(p/'summary_app_competitors.csv')
    estimator = load_csv(p/'summary_pka.csv')
    from acid_base_CPP import sample_pka_window, save_pka_window_csv
    sample_end = min(stop,float(estimator['time_ps'][-1]))
    st, sv = sample_pka_window(estimator['time_ps'],estimator['pka'],onset,sample_end,args.pka_samples)
    save_pka_window_csv(p/'summary_pka_window_samples.csv',st,sv)
    samples = np.column_stack([st,sv])
    fes = np.loadtxt(p/'summary_fes_snapshot.csv', delimiter=',', comments='#')
    lcod = lcod_series(Path(provenance['source']), (args.nb, args.na, args.hydrogen), stop)
    # Exact timestamp join, never assume matching row counts/sampling cadence.
    lookup = {round(row[0], 8): row[1:] for row in lcod}
    joined = np.array([lookup.get(round(float(t), 8), [np.nan]*3) for t in summary['time_ps']])
    if not np.all(np.isfinite(joined)):
        raise ValueError('Summary timestamps without an exact trajectory match')
    np.savetxt(p/'summary_lcod.csv', np.column_stack([summary['time_ps'], joined]), delimiter=',',
               header=f'time_ps,r_NB{args.nb}_H{args.hydrogen}_A,r_NA{args.na}_H{args.hydrogen}_A,LCOD_A', comments='')
    torsions, definitions = bridge_series(Path(provenance['source']), args.topology, summary['time_ps'])
    labels = [*BV_DIHEDRAL_ATOM_NAMES, 'helicity']
    smooth = np.column_stack([_circular_running_average_deg(summary['time_ps'],torsions[:,i],args.smoothing_window_ps) for i in range(7)])
    np.savetxt(p/'summary_conformation.csv', np.column_stack([summary['time_ps'],torsions,smooth]), delimiter=',', header=','.join(['time_ps']+[f'{s}_deg' for s in labels]+[f'{s}_smoothed_deg' for s in labels]), comments='')
    fig = plt.figure(figsize=(17,13),dpi=220,layout='constrained')
    grid = fig.add_gridspec(4,3,height_ratios=[1.1,1,1,.85],hspace=.08,wspace=.16)
    axes = np.array([[fig.add_subplot(grid[r,c]) for c in range(3)] for r in range(3)])
    ax_water = fig.add_subplot(grid[3,:])
    water = load_csv(p/'summary_water_oxygen_charges.csv')
    water_labels = list(water.dtype.names[1:])
    water_q = np.column_stack([water[label] for label in water_labels])
    for col in range(water_q.shape[1]):
        ax_water.plot(water['time_ps'],water_q[:,col],color='0.65',lw=.45,alpha=.55,rasterized=True)
    peaks = np.max(np.where(np.isfinite(water_q),water_q,-np.inf),axis=0)
    highlighted = np.argsort(peaks)[-3:][::-1]
    for col,color in zip(highlighted,('#D55E00','#0072B2','#009E73')):
        ax_water.plot(water['time_ps'],water_q[:,col],color=color,lw=1.,
                      label=water_labels[col].removesuffix('_charge_e'),rasterized=True)
    ax_water.set(xlabel='Raw DFTB time (ps)',ylabel='Solvent O Mulliken charge (e)',
                 title='All solvent oxygens; colored traces = three highest peak charges (one-based atom IDs)')
    ax_water.legend(loc='upper right',ncol=3,fontsize=10)
    ax_fes, ax_q, ax_single = axes[0]
    ax_cv, ax_lcod, ax_double = axes[1]
    ax_est, ax_coord, ax_hel = axes[2]
    for panel, start in [(ax_single,0),(ax_double,3)]:
        for i, color in enumerate(('#0072B2','#D55E00','#009E73')):
            y = smooth[:,start+i].copy()
            y[1:][np.abs(np.diff(y))>180] = np.nan
            panel.plot(summary['time_ps'],y,color=color,lw=1.6,label=labels[start+i])
        panel.legend(loc='upper left',frameon=False,fontsize=9)
    y = smooth[:,6].copy(); y[1:][np.abs(np.diff(y))>180] = np.nan
    ax_hel.plot(summary['time_ps'],y,lw=1.8,color='#6A3D9A')
    for panel,label,cols in [(ax_single,'single-bond dihedral (deg)',slice(0,3)),(ax_double,'double-bond dihedral (deg)',slice(3,6)),(ax_hel,'ring-center helicity (deg)',slice(6,7))]:
        limit = 90 if np.nanmax(np.abs(smooth[:,cols]))<=90 else 180
        panel.set(ylabel=label,ylim=(-limit,limit),yticks=np.linspace(-limit,limit,5))
    ax_hel.set_xlabel('Raw DFTB time (ps)')
    ax_fes.plot(fes[:, 0], fes[:, 1], color='#5B3A8A', lw=2)
    snapshot = (p/'summary_fes_snapshot.csv').read_text().splitlines()[0].removeprefix('# snapshot_time_ps=')
    ax_fes.set(title=f'FES at raw {snapshot} ps', xlabel='Coordination s', ylabel='F(s) (kcal/mol)', xlim=(0, 1))
    ax_cv.plot(summary['time_ps'], np.where(summary['time_ps']<=onset,summary['coordination_s'],np.nan), color='#356A8A', lw=2.2)
    ax_cv.plot(summary['time_ps'], np.where(summary['time_ps']>=onset,summary['coordination_s'],np.nan), color='#9B4A4A', lw=2.2)
    ax_cv.set(ylabel='Biased NB–HB coordination s(t)', ylim=(0, 1))
    ax_est.plot(estimator['time_ps'], estimator['pka'], color='black', lw=1.2)
    ax_est.scatter(estimator['time_ps'], estimator['pka'],color='#FFA500',edgecolor=(0,0,0,.35),s=18)
    ax_est.scatter(samples[:, 0], samples[:, 1], marker='*', color='#D62728', zorder=4)
    ax_est.text(.03, .08, f'{len(samples)} samples: {samples[:,1].mean():.3f} ± {samples[:,1].std():.3f}', transform=ax_est.transAxes)
    ax_est.set(ylabel='Inherited apparent pKa estimator', xlabel='Raw DFTB time (ps)')
    for i, (label, atom) in enumerate([('NB', args.nb), ('NA', args.na)]):
        ax_q.plot(charges['time_ps'], charges[f'{label}_charge']+i*1.1, color=f'C{i}', lw=1.1)
        # Temporarily disabled: all-H coordination remains in the saved CSV.
        # ax_coord.plot(coord['time_ps'], coord[f'{label}_all_H_coordination'], color=f'C{i}', label=f'{label}{atom}', lw=1.1)
    ax_q.set(yticks=[-.35, .75], yticklabels=[f'NB{args.nb}', f'NA{args.na}'], ylim=(-1.05, 1.45), ylabel='Mulliken charge tiers (NA offset +1.1 e)')
    ax_lcod.plot(lcod[:, 0], lcod[:, 3], color='#6A3D9A', lw=1.2)
    ax_lcod.axhline(0, color='.5', lw=.8, ls=':')
    ax_lcod.set(ylabel=f'r(NB{args.nb},H{args.hydrogen}) − r(NA{args.na},H{args.hydrogen}) (Å)', title='Fixed-H LCOD: negative NB-side / positive NA-side')
    # Equal-frame histogram: the trajectory has uniform sampling intervals.
    if not np.allclose(np.diff(lcod[:,0]),np.diff(lcod[:,0])[0],rtol=0,atol=1e-7):
        raise ValueError('Nonuniform trajectory sampling requires explicit time weighting')
    values = lcod[:,3]
    width = args.lcod_bin_width
    edges = np.arange(np.floor(values.min()/width),np.ceil(values.max()/width)+1)*width
    if len(edges)<2: edges=np.array([values[0]-width/2,values[0]+width/2])
    counts, edges = np.histogram(values,bins=edges)
    density = counts/(len(values)*np.diff(edges))
    neglog = np.full(density.shape,np.inf)
    np.log(density,out=neglog,where=density>0)
    neglog[density>0] *= -1
    centers = (edges[:-1]+edges[1:])/2
    np.savetxt(p/'summary_lcod_density.csv',np.column_stack([edges[:-1],edges[1:],centers,counts,density,neglog]),delimiter=',',header='left_A,right_A,center_A,count,density_per_A,minus_ln_density_1A',comments='')
    ax_coord.plot(centers,np.where(np.isfinite(neglog),neglog,np.nan),color='#6A3D9A',lw=1.8,marker='.',ms=4)
    ax_coord.set(xlabel='LCOD (Å)',ylabel=r'$-\ln[P(\mathrm{LCOD})\,\times\,1\,\mathrm{Å}]$',title=f'Sampled density, raw {lcod[0,0]:g}–{lcod[-1,0]:g} ps (biased)')
    ax_coord.grid(alpha=.2)
    for ax in (ax_q, ax_cv, ax_lcod, ax_est,ax_single,ax_double,ax_hel,ax_water):
        ax.set_xlim(summary['time_ps'][0], stop)
        ax.axvline(onset, color='black', ls='--', lw=1)
        ax.axvspan(onset, stop, color='#F8DADA', alpha=.4, zorder=0)
        ax.grid(alpha=.2)
    for ax in (ax_q,ax_cv,ax_lcod,ax_single,ax_double):
        ax.tick_params(axis='x',which='both',labelbottom=False)
    fig.suptitle(f'APP {p.name}: tautomerization summary\nNB{args.nb} / NA{args.na}; fixed H{args.hydrogen}; marker raw {onset:g} ps = grid {provenance["grid_onset_ps"]:g} ps')
    fig.supxlabel('Inherited FES estimator retained as requested; not a validated acid pKa or tautomer equilibrium constant.', fontsize=9)
    fig.savefig(p/'summary.png', dpi=220)
    plt.close(fig)
    (p/'tautomer_summary_provenance.json').write_text(json.dumps({
        'script': str(Path(__file__)), 'source': provenance['source'], 'NB_id': args.nb, 'NA_id': args.na,
        'fixed_H_id': args.hydrogen, 'raw_stop_ps': stop, 'raw_marker_ps': onset,
        'definition': 'LCOD = minimum-image r(NB,H) - minimum-image r(NA,H), angstrom; fixed atom IDs.',
        'lcod_samples': len(lcod), 'aligned_csv': 'summary_lcod.csv',
        'conformation': definitions, 'circular_smoothing_window_ps':args.smoothing_window_ps,
        'water_oxygen_charges': {'csv':'summary_water_oxygen_charges.csv',
            'oxygen_count':len(water_labels),'highlighted_atom_labels':[water_labels[i] for i in highlighted],
            'alignment':'Exact trajectory/Mulliken timestamp matching; missing samples remain NaN',
            'selection':'All solvent O atoms (IDs >77); highlight top three peak charges, not a defect assignment'},
        'pka_sample_count':len(samples),'pka_sample_start_ps':onset,'pka_sample_end_ps':sample_end,
        'mean_apparent_pka':float(sv.mean()),'std_apparent_pka':float(sv.std()),'std_ddof':0,
        'lcod_density':{'bin_width_A':width,'time_range_ps':[float(lcod[0,0]),float(lcod[-1,0])],'samples':len(values),'definition':'-ln(density * 1 angstrom); unshifted; zero-count bins +inf and omitted from curve; no pseudocounts or metadynamics reweighting','csv':'summary_lcod_density.csv'},
        'plot_data': ['summary_tautomer_timeseries.csv', 'summary_fes_snapshot.csv', 'summary_pka.csv', 'summary_pka_window_samples.csv', 'summary_app_heavy_charges.csv', 'summary_app_competitors.csv'],
        'note': 'Columns 2 and 3 and competitor wire panel removed. No defect analysis performed.'
    }, indent=2)+'\n')
    print(f'Saved {p / "summary.png"} and aligned LCOD CSV ({len(lcod)} trajectory samples)')


if __name__ == '__main__':
    main()
