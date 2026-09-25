"""Joint PRN anti/syn comparison preserving individual report windows/signs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acid_base_PRN import iter_xyz_frames, read_box_lengths_from_dftb_inp, _minimum_image
from plot_pka_grid import PKA_FACTOR
from acid_base_BV import load_fes_snapshot, sample_pka_window

ROOT = Path(__file__).resolve().parents[1]
SELECTION = {'anti': [15, 93], 'syn': [6, 44]}
COLORS = ['blue', 'red', 'green', 'black']
WINDOW_OVERRIDES_PS = {('syn', 6): 1.25}


def read(path: Path) -> np.ndarray:
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))


def save(path: Path, data: np.ndarray, header: str) -> None:
    np.savetxt(path, data, delimiter=',', header=header, comments='')


def load_record(rotamer: str, run: int, out: Path, refresh: bool) -> dict:
    base = ROOT/f'reports/PRN-{rotamer}/solv_5.5/acid_base'
    folder = base/f'run-{run}'
    meta = json.loads((folder/'provenance.json').read_text())
    with (base/'index.csv').open() as handle:
        entry = next(r for r in csv.DictReader(handle) if int(r['run']) == run)
    onset, stop = float(entry['tdiff_raw_ps']), float(entry['tstop_raw_ps'])
    np.testing.assert_allclose(onset, meta['diffusion_reported_ps'])
    np.testing.assert_allclose(stop-onset, 1.75)
    override = WINDOW_OVERRIDES_PS.get((rotamer, run))
    if override is not None:
        stop = onset + override
    series = read(folder/'summary_data.csv')
    series = series[series['time_ps'] <= stop+1e-8]
    origin = float(meta['reported_time_origin_ps'])
    offset = 40.-origin
    source = Path(meta['source'])
    assert meta['dihedral'] == [5, 3, 4, 11] and meta['solute_atoms'] == 11
    assert meta['acid_oxygen'] == 4 and meta['competitor_oxygen'] == 5
    label = f'{rotamer}-{run}'
    distance_file = out/f'{label}_distance.csv'
    if refresh or not distance_file.exists():
        mapping = {round(float(r['time_ps']), 8): float(r['defect_oxygen_id']) for r in series}
        box = read_box_lengths_from_dftb_inp(source/'dftb.inp')
        rows = []
        previous = np.nan
        for t, xyz in iter_xyz_frames(source/'traject'):
            if t > stop+1e-8:
                break
            key = round(t, 8)
            if key not in mapping:
                continue
            oid = mapping[key]
            distance = atom = np.nan
            if np.isfinite(oid):
                assert oid > 11
                ds = np.linalg.norm(_minimum_image(xyz[:11]-xyz[int(oid)-1], box), axis=1)
                atom = int(np.argmin(ds))+1
                distance = float(ds[atom-1])
                previous = distance
            rows.append([t, t+offset, oid, atom, distance, previous])
        save(distance_file, np.asarray(rows), 'raw_time_ps,aligned_time_ps,defect_oxygen_id,nearest_solute_atom_id,distance_A,display_distance_A')
    distances = read(distance_file)
    distances = distances[distances['raw_time_ps'] <= stop+1e-8]
    np.testing.assert_allclose(distances['raw_time_ps'], series['time_ps'], atol=1e-8)
    np.testing.assert_allclose(distances['defect_oxygen_id'], series['defect_oxygen_id'], equal_nan=True)
    np.testing.assert_allclose(distances['aligned_time_ps'], series['time_ps']+offset)
    pka = read(folder/'pka.csv')
    samples = np.loadtxt(folder/'pka_window_samples.csv', delimiter=',')
    if override is not None:
        sample_times, sample_values = sample_pka_window(pka['time_ps'], pka['pka'], onset, stop, 10)
        samples = np.column_stack([sample_times, sample_values])
    pka = pka[pka['time_ps'] <= stop+1e-8]
    assert samples.shape == (10, 2)
    np.testing.assert_allclose(samples[[0,-1],0], [onset,stop])
    factor = PKA_FACTOR*float(meta['temperature'])
    fes = np.loadtxt(folder/'fes_snapshot.csv', delimiter=',')
    first = (folder/'fes_snapshot.csv').read_text().splitlines()[0]
    snapshot = float(first.split('=')[1])
    if override is not None:
        snapshot, fes = load_fes_snapshot(source/'fes.dat', stop)
    assert snapshot <= stop+1e-8 and stop-snapshot < .081
    dih = read(folder/'proton_dihedrals.csv')
    dih = dih[dih['time_ps'] <= stop+1e-8]
    valid = (series['wire_evaluated']==1) & np.isfinite(series['defect_oxygen_id'])
    hbonds = np.where(valid, series['hbond_bridging_water_count']+1, np.nan)
    save(out/f'{label}_fes.csv', fes, 'coordination_s,F_kcal_mol')
    save(out/f'{label}_delta_f.csv', np.column_stack([pka['time_ps'],pka['time_ps']+offset,pka['pka']*factor]), 'raw_time_ps,aligned_time_ps,delta_F_kcal_mol')
    save(out/f'{label}_samples.csv', np.column_stack([samples[:,0],samples[:,0]+offset,samples[:,1]*factor,samples[:,1]]), 'raw_time_ps,aligned_time_ps,delta_F_kcal_mol,pka')
    save(out/f'{label}_dihedrals.csv', np.column_stack([dih['time_ps'],dih['time_ps']+offset,dih['old_dihedral_deg'],dih['new_dihedral_deg'],dih['new_hydrogen_id']]), 'raw_time_ps,aligned_time_ps,original_O4_H11_dihedral_deg,new_O5_H_dihedral_deg,new_hydrogen_id')
    save(out/f'{label}_hbonds.csv', np.column_stack([series['time_ps'],series['time_ps']+offset,hbonds]), 'raw_time_ps,aligned_time_ps,number_of_hydrogen_bonds')
    print(f'{label}: grid tdiff={entry["tdiff_grid_ps"]}; aligned={onset+offset:g}; stop={stop+offset:g}', flush=True)
    return dict(rotamer=rotamer,run=run,label=label,source=str(source),report=str(folder),grid_tdiff=float(entry['tdiff_grid_ps']),
                onset=onset,stop=stop,window_ps=stop-onset,offset=offset,snapshot=snapshot,factor=factor,
                reaction=float(meta['deprotonation_reported_ps']),time=pka['time_ps']+offset,energy=pka['pka']*factor,
                samples=samples[:,1]*factor,fes=fes,dihedrals=dih,distance=distances,
                wire_time=series['time_ps']+offset,hbonds=hbonds,pka_convention=meta['pka_convention'])


def torsion_line(ax, times, angles, color, style):
    """Common periodic branch [-90,270), leaving bond-assignment gaps intact."""
    angles = (np.asarray(angles)+90)%360-90
    angles = angles.copy()
    angles[1:][np.abs(np.diff(angles)) > 180] = np.nan
    ax.plot(times, angles, color=color, ls=style, lw=3.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'reports/PRN/joint_summary_anti_and_syn')
    parser.add_argument('--refresh-distances', action='store_true')
    parser.add_argument('--dihedral-bin-deg', type=float, default=5., help='Histogram bin width in degrees; must divide 360')
    args = parser.parse_args()
    if not np.isfinite(args.dihedral_bin_deg) or args.dihedral_bin_deg <= 0 or not np.isclose(360/args.dihedral_bin_deg, round(360/args.dihedral_bin_deg)):
        parser.error('--dihedral-bin-deg must be positive and divide 360')
    edges = np.linspace(-90,270,round(360/args.dihedral_bin_deg)+1)
    args.out.mkdir(parents=True,exist_ok=True)
    data = [load_record(rotamer, run, args.out, args.refresh_distances) for rotamer,runs in SELECTION.items() for run in runs]
    plt.style.use(ROOT/'plotting/lefteris.mplstyle')
    plt.rcParams.update({'font.size':20,'axes.labelsize':25,'xtick.labelsize':22,'ytick.labelsize':22,'legend.fontsize':17})
    fig = plt.figure(figsize=(29,25))
    grid = fig.add_gridspec(2,3,left=.06,right=.98,bottom=.065,top=.95,wspace=.25,hspace=.22)
    axes = {}
    for i in range(5):
        inner = grid[i//3,i%3].subgridspec(2,1,hspace=0)
        a = fig.add_subplot(inner[0]); b = fig.add_subplot(inner[1],sharex=a if i!=2 else None,sharey=a)
        axes[i] = [a,b]
        a.tick_params(labelbottom=False)
    wire_grid = grid[1,2].subgridspec(len(data),1,hspace=.04)
    wires=[];stats=[];group_stats=[]
    histogram_max = 0.
    maxt = max(d['stop']+d['offset'] for d in data)
    for tier,(rotamer,runs) in enumerate(SELECTION.items()):
        group=[d for d in data if d['rotamer']==rotamer]
        af,at,ab,adih,adist=[axes[i][tier] for i in range(5)]
        for j,d in enumerate(group):
            c=COLORS[j];label=f'Run {d["run"]}'
            af.plot(*d['fes'].T,color=c,lw=4.5,label=f'{label}: t={d["snapshot"]+d["offset"]:.2f} ps')
            at.plot(d['time'],d['energy'],color=c,lw=4,label=f'{label}: tdiff={d["onset"]+d["offset"]:.2f} ps')
            at.axvline(d['onset']+d['offset'],color=c,ls='--',lw=2)
            mu,sd=d['samples'].mean(),d['samples'].std()
            ab.bar(j,mu,yerr=sd,color=c,alpha=.85,capsize=5,label=f'{label}: pKa={mu/d["factor"]:.2f}')
            ab.annotate(f'{mu:.2f} ± {sd:.2f}',(j,mu+sd),xytext=(0,6),textcoords='offset points',ha='center',fontsize=17)
            dih=d['dihedrals']
            hist_columns = [edges[:-1],edges[1:]]
            for field,style in [('old_dihedral_deg','-'),('new_dihedral_deg','--')]:
                values = dih[field]
                values = (values[np.isfinite(values)]+90)%360-90
                counts,_ = np.histogram(values,bins=edges)
                density = counts/(len(values)*np.diff(edges)) if len(values) else np.full(len(edges)-1,np.nan)
                hist_columns.extend([counts,density])
                if len(values):
                    histogram_max = max(histogram_max,float(density.max()))
                    np.testing.assert_allclose(np.sum(density*np.diff(edges)),1.)
                    adih.stairs(density,edges,color=c,lw=3.5,ls=style,label=label if style=='-' else f'{label}: new O5–H')
            save(args.out/f'{d["label"]}_dihedral_histogram.csv',np.column_stack(hist_columns),
                 'bin_left_deg,bin_right_deg,old_count,old_density_per_degree,new_count,new_density_per_degree')
            adist.scatter(d['distance']['aligned_time_ps'],d['distance']['display_distance_A'],color=c,s=32,linewidths=0,label=label)
            ax=fig.add_subplot(wire_grid[len(wires)],sharex=wires[0] if wires else None,sharey=wires[0] if wires else None)
            wires.append(ax)
            ax.scatter(d['wire_time'],d['hbonds'],color=c,s=30,linewidths=0,label=f'{rotamer} {d["run"]}')
            ax.set(xlim=(40,maxt),ylim=(-.5,5.5),yticks=[0,2,5])
            ax.tick_params(labelbottom=len(wires)==len(data),labelsize=18)
            ax.legend(loc='upper right',fontsize=16)
            stats.append([tier,d['run'],d['grid_tdiff'],d['onset'],d['offset'],d['onset']+d['offset'],d['stop']+d['offset'],d['snapshot']+d['offset'],mu,sd,mu/d['factor']])
        af.set_xlim(0,1)
        af.text(.03,.9,rotamer.upper(),transform=af.transAxes,fontsize=23,fontweight='bold')
        for ax in [af,at,ab,adih,adist]:ax.legend(loc='best')
        mean_df = float(np.mean([d['samples'].mean() for d in group]))
        mean_pka = float(np.mean([d['samples'].mean()/d['factor'] for d in group]))
        group_stats.append([tier,len(group),mean_df,mean_pka])
        ab.text(.97,.48,
                f'Arithmetic means\n⟨ΔF⟩ = {mean_df:.2f} kcal/mol\n⟨pKa⟩ = {mean_pka:.2f}',
                transform=ab.transAxes,ha='right',va='center',fontsize=19,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
        for ax in [at,adist]:ax.set_xlim(40,maxt)
        adih.set(xlim=(-90,270),xticks=[-90,0,90,180,270],ylim=(0,None))
        adist.set_ylim(bottom=0)
        ab.set_xticks(range(len(runs)),[str(r) for r in runs])
        ab.tick_params(labelbottom=True)
        ab.set_xlim(-.6,3.6)
    axes[2][0].set_ylim(bottom=0,top=max(r[8]+r[9] for r in stats)*1.45)
    axes[3][0].set_ylim(0,histogram_max*1.15 if histogram_max else 1.)
    labels=['F(s) (kcal/mol)','ΔF (kcal/mol)','Mean ΔF (kcal/mol)','Probability density (degree⁻¹)','Defect–nearest solute atom (Å)']
    for i,label in enumerate(labels):
        upper,lower=axes[i]
        a,b=upper.get_position(),lower.get_position()
        fig.text(a.x0-.04,(a.y1+b.y0)/2,label,rotation=90,ha='center',va='center',fontsize=26)
        lower.set_xlabel('Coordination s' if i==0 else 'Run ID' if i==2 else 'O–C–O–H dihedral (degrees)' if i==3 else 't (ps; metadynamics starts at 40)')
    a,b=wires[0].get_position(),wires[-1].get_position()
    fig.text(a.x0-.04,(a.y1+b.y0)/2,'Number of hydrogen bonds',rotation=90,ha='center',va='center',fontsize=26)
    wires[-1].set_xlabel('t (ps; metadynamics starts at 40)')
    for ax in fig.axes:ax.grid(axis='y',alpha=.15)
    fig.suptitle('PRN · anti (upper) / syn (lower) · solv 5.5 · 300 K',fontsize=30)
    fig.savefig(args.out.with_suffix('.png'),dpi=180)
    plt.close(fig)
    save(args.out/'statistics.csv',np.array(stats),'tier_0_anti_1_syn,run,tdiff_grid_ps,tdiff_raw_ps,display_offset_ps,tdiff_aligned_ps,tstop_aligned_ps,fes_snapshot_aligned_ps,mean_delta_F_kcal_mol,std_delta_F_kcal_mol,mean_pka')
    save(args.out/'group_means.csv',np.array(group_stats),'tier_0_anti_1_syn,number_of_runs,arithmetic_mean_delta_F_kcal_mol,arithmetic_mean_pka')
    metadata=[{k:v for k,v in d.items() if not isinstance(v,np.ndarray)} for d in data]
    (args.out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(args.out.with_suffix('.png'))


if __name__=='__main__':main()
