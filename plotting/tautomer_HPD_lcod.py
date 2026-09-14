"""Dedicated six-panel HPD LCOD summary; no proton-defect analysis.

Example: python plotting/tautomer_HPD_lcod.py --run 1
See tautomer_HPD_lcod.md for definitions and limitations.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prd_ring_planarity import verify_ring, ring_metrics
from prn_anti_dih import box_from_input, minimum_image, dihedral
from prn_acid_base_analysis import xyz_frames, mulliken_frames, snapshot_bytes, TIME_RE


def signed_impropers(xyz: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw improper angles and signed nearest-planar deviations for H8/O7.

    Quartets X-next-center-previous: 8-2-1-6 and 7-3-2-1.
    Sign comes from the substituent's side of its local plane, oriented
    along the area normal of the ordered ring 1-2-3-4-5-6.
    """
    raw = np.full((len(xyz),2),np.nan)
    bending = np.full_like(raw,np.nan)
    for frame, coords in enumerate(xyz):
        ring = np.zeros((6,3))
        ring[1:] = np.cumsum(minimum_image(np.diff(coords[:6],axis=0),box),axis=0)
        reference = np.cross(ring,np.roll(ring,-1,axis=0)).sum(axis=0)
        for col,(sub,nxt,center,prev) in enumerate(((7,1,0,5),(6,2,1,0))):
            normal = np.cross(minimum_image(coords[nxt]-coords[center],box),
                              minimum_image(coords[prev]-coords[center],box))
            alignment = np.dot(normal,reference)
            if np.linalg.norm(normal) < 1e-12 or abs(alignment) < 1e-12:
                continue
            normal *= np.sign(alignment)
            phi = dihedral(coords[[sub,nxt,center,prev]],box)
            raw[frame,col] = phi
            magnitude = abs((phi+90.) % 180.-90.)
            side = np.dot(minimum_image(coords[sub]-coords[center],box),normal)
            bending[frame,col] = np.sign(side)*magnitude
    return raw,bending


def lcod_metadynamics(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Read dimensional LCOD bias records and join FES by Gaussian number."""
    hills = {}
    number = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?'
    for chunk in snapshot_bytes(source/'biaspot').decode().split('GAUSSIAN BIAS POTENTIAL:')[1:]:
        time = TIME_RE.search(chunk)
        coord = re.search(r'Coordinate\s*=\s*('+number+r')\s+Angstrom',chunk)
        width = re.search(r'Gaussian width\s*=\s*('+number+r')\s+Angstrom\s*\n',chunk)
        if time and coord and width:
            hills[int(chunk.split()[0])] = (float(time[1])/1000,float(coord[1].replace('D','E').replace('d','e')))
    chunks = re.split(r'### FREE ENERGY SURFACE CONSISTING OF\s+(\d+) GAUSSIANS',
                      snapshot_bytes(source/'fes.dat').decode())
    ft, blocks, grid = [], [], None
    for i in range(1,len(chunks),2):
        rows = [list(map(float,line.split())) for line in chunks[i+1].splitlines(keepends=True)
                if len(line.split()) == 2 and line.endswith('\n')]
        if not rows:
            continue
        block = np.asarray(rows)
        if grid is None:
            grid = block[:,0].copy()
        if len(block) != len(grid) or not np.allclose(block[:,0],grid):
            if i+2 < len(chunks):
                raise ValueError('Incomplete interior FES block')
            continue
        hill = int(chunks[i])
        if hill in hills:
            ft.append(hills[hill][0]); block[:,1] *= 627.509474; blocks.append(block)
    bias = np.array(list(hills.values()))
    if not len(bias) or np.any(np.diff(bias[:,0]) <= 0):
        raise ValueError('Missing or nonmonotonic LCOD bias timestamps')
    return bias[:,0],bias[:,1],np.array(ft),blocks


def side_minima(block: np.ndarray, split: float) -> tuple[float, float, float]:
    """O-side minus N-side grid minimum (not an integrated basin free energy)."""
    x, f = block.T
    left, right = x < split, x > split
    if not np.any(left) or not np.any(right):
        raise ValueError('FES must contain grid points on both sides of the split')
    n, o = float(np.min(f[left])), float(np.min(f[right]))
    return n, o, o-n


def save_csv(path: Path, values: np.ndarray, names: list[str]) -> None:
    np.savetxt(path, values, delimiter=',', header=','.join(names), comments='')


def analyze(source: Path, topology: Path, split: float, geometry_only: bool = False) -> dict:
    ring, names = verify_ring(topology)
    if len(names) != 12 or names[7] != 'O2' or names[8] != 'HN1':
        raise ValueError('Expected HPD topology: 12 atoms, oxygen ID 7, hydrogen ID 8')
    if not geometry_only:
        cv = (source/'metacv.dat').read_text().split()
        if cv[0] != 'BONDDISTANCEDIFFERENCE' or list(map(int, cv[2:6])) != [1,8,7,8]:
            raise ValueError('Expected LCOD r(N1,H8) - r(O7,H8)')
    with (source/'traject').open() as f:
        natoms = int(f.readline()); f.readline()
        symbols = [f.readline().split()[0] for _ in range(natoms)]
    expected = ['N', *['C']*5, 'O', *['H']*5]
    if symbols[:12] != expected:
        raise ValueError('HPD trajectory atom order does not match topology')
    box = box_from_input(source/'dftb.inp')
    times, solute = [], []
    for t, xyz in xyz_frames(source/'traject'):
        times.append(t); solute.append(xyz[:12])
    t, xyz = np.asarray(times), np.asarray(solute)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError('Empty/nonmonotonic trajectory; stitch restarts explicitly')
    distances = np.linalg.norm(minimum_image(xyz[:, [0,6]]-xyz[:,7,None], box), axis=2)
    phi, dev, planarity = ring_metrics(xyz[:, np.array(ring)-1], box)
    improper, bending = signed_impropers(xyz,box)
    ocnh = np.array([dihedral(c[[6,1,0,7]],box) for c in xyz])
    qt, charges = mulliken_frames(source/'mulliken', symbols)
    if not len(qt) or np.any(np.diff(qt) <= 0):
        raise ValueError('Empty/nonmonotonic Mulliken series')
    lookup = {round(float(v),8): i for i,v in enumerate(qt)}
    q = np.full((len(t),12), np.nan)
    for i, v in enumerate(t):
        j = lookup.get(round(float(v),8))
        if j is not None:
            q[i] = charges[j,:12]
    if geometry_only:
        return dict(t=t,distances=distances,improper=improper,bending=bending,
                    charges=q,names=names,box=box,ocnh=ocnh)
    bt, bs, ft, blocks = lcod_metadynamics(source)
    keep = (ft >= t[0]-1e-8) & (ft <= t[-1]+1e-8)
    blocks = [b for b,k in zip(blocks,keep) if k]; ft = ft[keep]
    if not blocks or np.any(np.diff(ft) <= 0):
        raise ValueError('No complete time-aligned FES surfaces or nonmonotonic times')
    minima = np.array([side_minima(b,split) for b in blocks])
    # Direct check at exactly matching trajectory/bias timestamps, not interpolation.
    idx = {round(float(v),8): i for i,v in enumerate(t)}
    errors = [distances[idx[round(float(v),8)],0]-distances[idx[round(float(v),8)],1]-s
              for v,s in zip(bt,bs) if round(float(v),8) in idx]
    if not errors or np.max(np.abs(errors)) > 1e-3:
        raise ValueError('Trajectory LCOD disagrees with recorded bias coordinate')
    return dict(t=t, distances=distances, phi=phi, dev=dev, planarity=planarity,
                improper=improper,bending=bending,ocnh=ocnh,
                charges=q, ft=ft, blocks=blocks, minima=minima, bt=bt, bs=bs,
                names=names, box=box, max_cv_error_A=float(np.max(np.abs(errors))))


def render(data: dict, out: Path, run: int, split: float) -> None:
    t = data['t']; d = data['distances']; ft = data['ft']
    fes = data['blocks'][-1].copy(); fes[:,1] -= fes[:,1].min()
    fig, axes = plt.subplots(3,2,figsize=(13,10),dpi=220,layout='constrained',
                             gridspec_kw={'height_ratios':[1.1,1,1], 'hspace':.08,'wspace':.16})
    af, ad = axes[0]; ac, ap = axes[1]; ag, aq = axes[2]
    af.plot(*fes.T,color='#5B3A8A',lw=2)
    af.axvline(split,color='.5',ls=':',lw=1)
    af.set(xlabel='LCOD (Å)',ylabel='F(LCOD) (kcal/mol)',title=f'FES at raw {ft[-1]:g} ps')
    ac.plot(t,d[:,0]-d[:,1],color='#356A8A',lw=1.2)
    ac.axhline(0,color='.5',ls=':',lw=1)
    ac.set(ylabel='LCOD (Å)',title='r(N1,H8) − r(O7,H8)')
    ag.plot(ft,data['minima'][:,2],color='black',lw=1.2)
    ag.scatter(ft,data['minima'][:,2],color='#FFA500',edgecolor=(0,0,0,.35),s=12)
    ag.set(ylabel='Fmin(O-side) − Fmin(N-side) (kcal/mol)',title='Side-minimum difference (not pKa)')
    ad.plot(t,d[:,0],color='blue',lw=1,label='N1–H8')
    ad.plot(t,d[:,1],color='red',lw=1,label='O7–H8')
    ad.set(ylabel='Distance (Å)',title='Fixed biased proton H8'); ad.legend(frameon=False)
    ap.plot(t,data['bending'][:,0],color='blue',lw=.8,label='H8: 8–2–1–6')
    ap.plot(t,data['bending'][:,1],color='red',lw=.8,label='O7: 7–3–2–1')
    ap.axhline(0,color='.4',ls='--',lw=.8)
    limit = max(5.,float(np.nanmax(np.abs(data['bending'])))*1.08)
    ap.set(ylabel='Signed improper deviation (°)',ylim=(-limit,limit),
           title='H8 / O7 out-of-plane bending; common ring-side sign')
    ap.legend(frameon=False,fontsize=8,loc='upper right')
    for i in range(12):
        aq.plot(t,data['charges'][:,i],color=plt.get_cmap('tab20')(i),lw=.7,
                label=f'{data["names"][i+1]} (ID {i+1})')
    aq.set(ylabel='Atomic Mulliken charge (e)',title='All 12 solute atoms; no offsets')
    aq.legend(frameon=False,fontsize=7,ncol=4,loc='upper center')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in (ad,ac,ap,ag,aq):
        ax.set_xlim(t[0],t[-1])
        ax.set_xlabel('Raw DFTB time (ps)')
    fig.suptitle(f'HPD run-{run}: LCOD metadynamics / tautomerization summary')
    fig.supxlabel('Negative LCOD: N-side; positive LCOD: O-side. Side minima alone do not establish sampling or convergence.',fontsize=9)
    fig.savefig(out/'summary.png',dpi=220); plt.close(fig)
    save_csv(out/'summary_fes_snapshot.csv',fes,['LCOD_A','shifted_F_kcal_mol'])


def joined_meta_times(equil_times: np.ndarray, meta_times: np.ndarray) -> np.ndarray:
    """Place first metadynamics frame at the last equilibration timestamp."""
    return meta_times-meta_times[0]+equil_times[-1]


def circular_stats_deg(values: np.ndarray) -> tuple[float, float]:
    """Circular mean and SD sqrt(-2 ln R), degrees, finite frames only."""
    valid = np.asarray(values)[np.isfinite(values)]
    if not len(valid):
        return float('nan'),float('nan')
    z = np.mean(np.exp(1j*np.deg2rad(valid)))
    r = min(1.,float(abs(z)))
    if r < 1e-15:
        return float('nan'),float('inf')
    return float(np.rad2deg(np.angle(z))),float(np.rad2deg(np.sqrt(-2*np.log(r))))


def render_comparison(equil: dict, meta: dict, out: Path, run: int) -> None:
    """Four stacked observables on a continuous, phase-labelled time axis."""
    fig, axes = plt.subplots(4,1,figsize=(13,13),dpi=220,layout='constrained',sharex=True)
    ad,ap,aq,at = axes
    boundary = float(equil['t'][-1])
    meta_t = joined_meta_times(equil['t'],meta['t'])
    combined = []
    bending_stats = []
    oxygen_stats, torsion_stats, torsion_rows = [], [], []
    for col,(label,data) in enumerate((('Equilibration',equil),('LCOD metadynamics',meta))):
        raw_t = data['t']
        t = raw_t if col == 0 else meta_t
        colors = ('#0072B2','#009E73') if col == 0 else ('#D55E00','#AA3377')
        phase = 'equil' if col == 0 else 'meta'
        oxygen = data['charges'][:,6]
        valid = oxygen[np.isfinite(oxygen)]
        qm = float(np.mean(valid)) if len(valid) else float('nan')
        qs = float(np.std(valid,ddof=1)) if len(valid)>1 else float('nan')
        oxygen_stats.append([col,len(valid),qm,qs])
        aq.text(.02 if col == 0 else .55,.73,f'O7 {phase}: {qm:.3f} ± {qs:.3f} e',
                transform=aq.transAxes,fontsize=9,color=colors[0],
                bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
        torsion = data['ocnh']
        tm,ts = circular_stats_deg(torsion)
        torsion_stats.append([col,np.isfinite(torsion).sum(),tm,ts])
        # Center plotting around planar cis (0 degrees), break wrap seams.
        shown = (torsion+180)%360-180
        shown[1:][np.abs(np.diff(shown))>180] = np.nan
        at.plot(t,shown,color=colors[0],lw=.9,
                label=f'{phase}: {tm:.2f} ± {ts:.2f}° (circular)')
        torsion_rows.append(np.column_stack([np.full(len(t),col),raw_t,t,torsion,shown]))
        for i,name in enumerate(('N1–H8','O7–H8')):
            ad.plot(t,data['distances'][:,i],color=colors[i],lw=1,label=f'{name}: {phase}')
        for i,name in enumerate(('H8','O7')):
            values = data['bending'][:,i]
            finite = values[np.isfinite(values)]
            mean = float(np.mean(finite)) if len(finite) else float('nan')
            std = float(np.std(finite,ddof=1)) if len(finite)>1 else float('nan')
            bending_stats.append([col,8 if i == 0 else 7,len(finite),mean,std])
            ap.plot(t,values,color=colors[i],lw=.8,
                    label=f'{name} {phase}: {mean:.2f} ± {std:.2f}°')
        for i in range(12):
            base = np.array(plt.get_cmap('tab20')(i)[:3])
            color = .5*base+.5 if col == 0 else .85*base
            aq.plot(t,data['charges'][:,i],color=color,lw=.7,
                    label=f'{data["names"][i+1]} (ID {i+1})' if col == 1 else None)
        values = np.column_stack([raw_t,data['distances'],data['improper'],data['bending'],data['charges']])
        combined.append(np.column_stack([np.full(len(t),col),t,values]))
        save_csv(out/f'{"equil" if col == 0 else "metad"}_comparison_timeseries.csv',values,
                 ['raw_time_ps','r_N1_H8_A','r_O7_H8_A','H8_raw_improper_deg','O7_raw_improper_deg',
                  'H8_signed_improper_deviation_deg','O7_signed_improper_deviation_deg']+
                 [f'atom_{i}_charge_e' for i in range(1,13)])
    save_csv(out/'summary_merged_timeseries.csv',np.vstack(combined),
             ['phase_0_equil_1_meta','plotted_time_ps','raw_time_ps','r_N1_H8_A','r_O7_H8_A',
              'H8_raw_improper_deg','O7_raw_improper_deg','H8_signed_improper_deviation_deg',
              'O7_signed_improper_deviation_deg']+[f'atom_{i}_charge_e' for i in range(1,13)])
    save_csv(out/'summary_bending_stats.csv',np.asarray(bending_stats),
             ['phase_0_equil_1_meta','atom_id','finite_frames','mean_deg','sample_std_deg_ddof1'])
    save_csv(out/'summary_oxygen_charge_stats.csv',np.asarray(oxygen_stats),
             ['phase_0_equil_1_meta','finite_frames','mean_e','sample_std_e_ddof1'])
    save_csv(out/'summary_ocnh_stats.csv',np.asarray(torsion_stats),
             ['phase_0_equil_1_meta','finite_frames','circular_mean_deg','circular_std_deg'])
    save_csv(out/'summary_ocnh_timeseries.csv',np.vstack(torsion_rows),
             ['phase_0_equil_1_meta','raw_time_ps','plotted_time_ps','O7_C2_N1_H8_dihedral_deg','displayed_deg'])
    ad.set_ylabel('Distance (Å)')
    ap.set_ylabel('Signed improper deviation (°)')
    aq.set_ylabel('Atomic Mulliken charge (e)')
    at.set_ylabel('O7–C2–N1–H8 dihedral (°)')
    at.axhline(0,color='.4',ls='--',lw=.8)
    at.legend(frameon=False,fontsize=9,ncol=2,loc='upper left')
    at.set_title('O–C–N–H torsion: circular mean ± circular SD',fontsize=10)
    ap.axhline(0,color='.4',ls='--',lw=.8)
    ad.legend(frameon=False,fontsize=9,ncol=4,loc='upper left')
    ap.legend(frameon=False,fontsize=9,ncol=2,loc='upper left')
    ap.set_title('Signed bending: mean ± sample SD (degrees), each phase separately',fontsize=10)
    aq.legend(frameon=False,fontsize=8,ncol=6,loc='upper left')
    aq.set_title('Mulliken charges: lighter atom colors = equil; darker = meta',fontsize=10)
    for ax in axes:
        ax.axvline(boundary,color='black',ls='--',lw=1.2)
        ax.axvspan(equil['t'][0],boundary,color='#0072B2',alpha=.035,zorder=0)
        ax.axvspan(boundary,meta_t[-1],color='#D55E00',alpha=.035,zorder=0)
        ax.grid(alpha=.2); ax.set_xlim(equil['t'][0],meta_t[-1])
    at.set_xlabel('Combined time (ps)')
    limit = max(5.,max(float(np.nanmax(np.abs(d['bending']))) for d in (equil,meta))*1.08)
    ap.set_ylim(-limit,limit)
    fig.suptitle(f'HPD run-{run}: equilibration {equil["t"][0]:g}–{boundary:g} ps → LCOD metadynamics {boundary:g}–{meta_t[-1]:g} ps')
    fig.supxlabel('Metadynamics shifted to the equilibration endpoint; phases plotted separately without a connecting segment. No smoothing.',fontsize=9)
    fig.savefig(out/'summary.png',dpi=220); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=int,default=1)
    p.add_argument('--runs-path',type=Path,default=Path('systems/HPD/solv_5.5/dftb/N1T48C1'))
    p.add_argument('--cv-dir',default='meta-lcod')
    p.add_argument('--equil-dir',default='equil',help='Equilibration directory beside the metadynamics directory')
    p.add_argument('--topology',type=Path,default=Path('systems/HPD/init/hpd.mol2'))
    p.add_argument('--out',type=Path,help='Default: reports/HPD/meta-lcod/run-ID')
    p.add_argument('--basin-split',type=float,default=0.,help='LCOD grid split in Å; exclude split itself')
    args = p.parse_args()
    if args.run < 1 or not np.isfinite(args.basin_split):
        p.error('Positive run ID and finite basin split required')
    source = args.runs_path/f'run-{args.run}'/args.cv_dir
    out = args.out or Path('reports/HPD/meta-lcod')/f'run-{args.run}'
    data = analyze(source,args.topology,args.basin_split)
    equil_source = source.parent/args.equil_dir
    equil = analyze(equil_source,args.topology,args.basin_split,geometry_only=True)
    out.mkdir(parents=True,exist_ok=True)
    t, d = data['t'], data['distances']
    save_csv(out/'summary_timeseries.csv',np.column_stack([t,d,d[:,0]-d[:,1],data['planarity'],data['charges'],data['bending']]),
             ['raw_time_ps','r_N1_H8_A','r_O7_H8_A','LCOD_A','mean_planarity_deviation_deg']+
             [f'atom_{i}_charge_e' for i in range(1,13)]+['H8_signed_improper_deviation_deg','O7_signed_improper_deviation_deg'])
    save_csv(out/'summary_out_of_plane.csv',np.column_stack([t,data['improper'],data['bending']]),
             ['raw_time_ps','H8_improper_8_2_1_6_deg','O7_improper_7_3_2_1_deg',
              'H8_signed_improper_deviation_deg','O7_signed_improper_deviation_deg'])
    save_csv(out/'summary_planarity.csv',np.column_stack([t,data['phi'],data['dev'],data['planarity']]),
             ['raw_time_ps']+[f'phi_{i}_deg' for i in range(1,7)]+[f'dev_{i}_deg' for i in range(1,7)]+['mean_deviation_deg'])
    save_csv(out/'summary_delta_f.csv',np.column_stack([data['ft'],data['minima']]),
             ['raw_time_ps','N_side_min_kcal_mol','O_side_min_kcal_mol','O_minus_N_kcal_mol'])
    save_csv(out/'summary_bias.csv',np.column_stack([data['bt'],data['bs']]),['raw_time_ps','LCOD_A'])
    render_comparison(equil,data,out,args.run)
    provenance = dict(source=str(source.resolve()),topology=str(args.topology.resolve()),
        atom_names=data['names'],ring_ids=list(range(1,7)),lcod_ids=[1,8,7,8],
        ring_quartets=[[(i+j)%6+1 for j in range(4)] for i in range(6)],
        out_of_plane=dict(improper_quartets={'H8':[8,2,1,6],'O7':[7,3,2,1]},
            definition='Nearest-planar improper magnitude abs((phi+90)%180-90), signed by substituent displacement '
                       'along local plane normal oriented to ordered ring area normal 1-2-3-4-5-6. '
                       'Local planes C2-N1-C6 and C3-C2-N1; not a bond-to-plane angle. Fixed atom IDs.',
            csv='summary_out_of_plane.csv'),
        box_A=data['box'].tolist(),raw_time_range_ps=[float(t[0]),float(t[-1])],
        trajectory_frames=len(t),missing_charge_frames=int(np.sum(~np.isfinite(data['charges']).all(axis=1))),
        fes_snapshot_raw_ps=float(data['ft'][-1]),basin_split_A=args.basin_split,
        comparison=dict(equil_source=str(equil_source.resolve()),
            equil_raw_time_range_ps=[float(equil['t'][0]),float(equil['t'][-1])],
            equil_frames=len(equil['t']),
            equil_missing_charge_frames=int(np.sum(~np.isfinite(equil['charges']).all(axis=1))),
            equil_box_A=equil['box'].tolist(),layout='four rows, one column; phases merged',
            ocnh_quartet=[7,2,1,8],ocnh_statistics='Circular mean and sqrt(-2 ln R) circular SD, degrees',
            metad_plot_offset_ps=float(equil['t'][-1]-t[0]),
            metad_plotted_time_range_ps=joined_meta_times(equil['t'],t)[[0,-1]].tolist(),
            csvs=['equil_comparison_timeseries.csv','metad_comparison_timeseries.csv','summary_merged_timeseries.csv']),
        max_bias_geometry_difference_A=data['max_cv_error_A'],
        note='Raw CSV timestamps retained; merged figure shifts meta start to equil end. No defect analysis; no smoothing. Charges matched by timestamp. '
             'FES matched to bias timestamps by Gaussian count. Delta F is a grid-minimum diagnostic, not pKa or a converged basin population ratio.')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2)); print(out/'summary.png')


if __name__ == '__main__':
    main()
