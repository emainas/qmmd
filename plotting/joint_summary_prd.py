"""PRD multi-run energy, ring-planarity and solvent-defect comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acid_base_PRD import augment_csv, iter_xyz_frames, read_box_lengths_from_dftb_inp, _minimum_image
from prd_ring_planarity import verify_ring, ring_metrics
from plot_pka_grid import PKA_FACTOR


def table(path: Path) -> np.ndarray:
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))


def save(path: Path, values: np.ndarray, header: str) -> None:
    np.savetxt(path, values, delimiter=',', header=header, comments='')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=int, nargs='+', default=[20, 66])
    p.add_argument('--refresh', action='store_true', help='Regenerate this derived comparison in its existing directory')
    p.add_argument('--reports', type=Path, default=Path('reports/PRD/meta-h_selected'))
    p.add_argument('--out', type=Path, default=Path('reports/PRD/joint_summary_20_and_66'))
    p.add_argument('--topology', type=Path, default=Path('systems/PRD/init/prd.mol2'))
    p.add_argument('--smooth-ps', type=float, default=.2, help='Planarity display running-mean width, ps')
    p.add_argument('--reaction-threshold', type=float, default=.05, help='First sampled s<=threshold defines reaction marker')
    p.add_argument('--max-bridging-waters', type=int, default=8)
    p.add_argument('--covalent-cutoff', type=float, default=1.3, help='Donor-H distance, angstrom')
    p.add_argument('--hydrogen-acceptor-cutoff', type=float, default=2.5, help='H-acceptor distance, angstrom')
    p.add_argument('--angle-cutoff', type=float, default=135., help='Minimum D-H-A angle, degrees')
    args = p.parse_args()
    if args.smooth_ps <= 0 or args.max_bridging_waters < 0:
        p.error('Invalid smoothing or path cap')
    ring, _ = verify_ring(args.topology)
    if args.out.exists() and args.refresh and not (args.out/'provenance.json').exists():
        p.error('Refresh requires an existing comparison provenance file')
    args.out.mkdir(parents=True, exist_ok=args.refresh)
    plt.style.use(Path(__file__).with_name('lefteris.mplstyle'))
    fig = plt.figure(figsize=(15, 11), layout='constrained')
    grid = fig.add_gridspec(2, 2, hspace=.12, wspace=.10)
    af, at = [fig.add_subplot(grid[0, c]) for c in range(2)]
    ad = fig.add_subplot(grid[1, 0])
    wg = grid[1, 1].subgridspec(len(args.runs), 1, hspace=.05)
    wa = [fig.add_subplot(wg[i]) for i in range(len(args.runs))]
    fes_curves, stats, records = [], [], []
    colors = ['blue', 'red']
    global_stop = 40.
    for j, run in enumerate(args.runs):
        color = colors[j % len(colors)]
        report = args.reports/f'run-{run}'
        meta = json.loads((report/'provenance.json').read_text())
        if meta['temperature_K'] != 300:
            raise ValueError('Comparison requires 300 K reports')
        source = Path(meta['source'])
        onset = meta['raw_tdiff_ps']
        stop = onset+meta['probability_window_ps']
        assigned = table(report/'wire_input.csv')
        assigned = assigned[assigned['time_ps'] <= stop+1e-8]
        mapping = {round(float(r['time_ps']), 8):float(r['defect_oxygen_id']) for r in assigned}
        box = read_box_lengths_from_dftb_inp(source/'dftb.inp')
        ts, coords, distances = [], [], []
        previous = np.nan
        offset = None
        for time, xyz in iter_xyz_frames(source/'traject'):
            if offset is None:
                offset = 40.-time
            if time > stop+1e-8:
                break
            ts.append(time)
            coords.append(xyz[:6])
            key = round(time, 8)
            if key not in mapping:
                continue
            defect = mapping[key]
            distance = atom = np.nan
            if np.isfinite(defect):
                ds = np.linalg.norm(_minimum_image(xyz[:12]-xyz[int(defect)-1], box), axis=1)
                atom = int(np.argmin(ds))+1
                distance = float(ds[atom-1])
                previous = distance
            distances.append([time,time+offset,defect,atom,distance,previous,int(not np.isfinite(distance) and np.isfinite(previous))])
        ts = np.asarray(ts)
        phi, dev, mean = ring_metrics(np.asarray(coords), box)
        # Planarity is a nonperiodic deviation, so use an ordinary local mean.
        smooth = np.array([np.mean(mean[(ts>=t-args.smooth_ps/2)&(ts<=t+args.smooth_ps/2)]) for t in ts])
        candidates = assigned['time_ps'][(assigned['coordination_s']<=args.reaction_threshold)&(assigned['time_ps']<=onset)]
        reaction = float(candidates[0]) if len(candidates) else np.nan
        pre, post = mean[ts<reaction], mean[ts>=reaction]
        save(args.out/f'run-{run}_planarity.csv', np.column_stack([ts,ts+offset,phi,dev,mean,smooth]),
             ','.join(['raw_time_ps','aligned_time_ps']+[f'phi_{i+1}_deg' for i in range(6)]+[f'deviation_{i+1}_deg' for i in range(6)]+['mean_planarity_deg','smoothed_mean_deg']))
        distances = np.array(distances)
        save(args.out/f'run-{run}_distance.csv', distances,
             'raw_time_ps,aligned_time_ps,defect_oxygen_id,nearest_solute_atom_id,distance_A,display_distance_A,carried_forward')
        ad.plot(distances[:,1],distances[:,5],color=color,lw=2.5,label=f'Run {run}')
        fes = np.loadtxt(report/'summary_fes_snapshot.csv',delimiter=',',comments='#')
        fes[:,1] -= fes[:,1].min()
        if fes_curves:
            np.testing.assert_allclose(fes[:,0],fes_curves[0][:,0])
        fes_curves.append(fes)
        af.plot(fes[:,0],fes[:,1],color=color,alpha=.4,lw=2.5,label=f'Run {run}')
        save(args.out/f'run-{run}_fes.csv',fes,'coordination_s,F_kcal_mol')
        series = table(report/'summary_delta_f.csv')
        keep = series['time_ps']<=stop+1e-8
        energy = series['delta_F_kcal_mol'][keep]
        et = series['time_ps'][keep]+offset
        at.plot(et,energy,color=color,lw=2.5)
        save(args.out/f'run-{run}_delta_f.csv',np.column_stack([et-offset,et,energy]),'raw_time_ps,aligned_time_ps,delta_F_kcal_mol')
        samples = np.loadtxt(report/'summary_delta_f_window_samples.csv',delimiter=',')
        np.testing.assert_allclose(samples[:,0],np.linspace(onset,stop,10))
        mu, sd = samples[:,1].mean(), samples[:,1].std()
        save(args.out/f'run-{run}_samples.csv',np.column_stack([samples[:,0],samples[:,0]+offset,samples[:,1]]),'raw_time_ps,aligned_time_ps,delta_F_kcal_mol')
        at.axhline(mu,color=color,ls='--',lw=2,
                   label=f'ΔF = {mu:.1f} kcal/mol, pKa = {mu / (PKA_FACTOR * 300):.1f}')
        # Recompute paths with the same eight-water cap used by the BV comparison.
        wire_input = args.out/f'run-{run}_wire_input.csv'
        save(wire_input,np.column_stack([assigned[n] for n in assigned.dtype.names]),','.join(assigned.dtype.names))
        wires = args.out/f'run-{run}_wires.csv'
        augment_csv(wire_input,wires,source/'traject',1,12,box=box,
                    max_bridging_waters=args.max_bridging_waters,covalent_cutoff=args.covalent_cutoff,
                    hydrogen_acceptor_cutoff=args.hydrogen_acceptor_cutoff,angle_cutoff=args.angle_cutoff)
        w = table(wires)
        valid = (w['wire_evaluated']==1)&np.isfinite(w['defect_oxygen_id'])
        hb = np.where(valid,w['hbond_bridging_water_count']+1,np.nan)
        wa[j].scatter(w['time_ps']+offset,hb,color=color,s=23,linewidths=0,label=f'Run {run}')
        save(args.out/f'run-{run}_hbond_plot.csv',np.column_stack([w['time_ps'],w['time_ps']+offset,hb]),'raw_time_ps,aligned_time_ps,number_of_hydrogen_bonds')
        wa[j].set(ylim=(-.5,args.max_bridging_waters+1.5),yticks=[0,3,6,9])
        wa[j].legend(loc='upper left')
        wa[j].tick_params(labelbottom=j==len(args.runs)-1)
        stats.append([run,onset+offset,stop+offset,reaction+offset,mu,sd,np.mean(pre),np.mean(post)])
        records.append(dict(run=run,source=str(source),report=str(report),display_offset_ps=offset,
                            grid_tdiff_ps=onset+offset,window_ps=meta['probability_window_ps']))
        global_stop=max(global_stop,stop+offset)
        print(f'Completed run {run}',flush=True)
    fes_mean=np.mean([f[:,1] for f in fes_curves],axis=0)
    save(args.out/'fes_mean.csv',np.column_stack([fes_curves[0][:,0],fes_mean]),'coordination_s,mean_F_kcal_mol')
    ref=5*PKA_FACTOR*300
    at.axhline(ref,color='green',ls='--',lw=2,
               label=f'ΔF_exp = {ref:.1f} kcal/mol, pKa = 5.0')
    af.set(xlabel='Coordination s',ylabel='F(s) (kcal/mol)',xlim=(0,1))
    at.set(ylabel='ΔF (kcal/mol)')
    ad.set(ylabel='Defect–nearest solute atom (Å)',ylim=(0,None))
    wa[len(wa)//2].set_ylabel('Number of hydrogen bonds')
    for ax in [at,ad,*wa]:
        ax.set_xlim(40,global_stop)
    for ax in [at,ad,wa[-1]]:
        ax.set_xlabel('t (ps; metadynamics starts at 40)')
    for ax in [af,at,ad]:
        ax.legend(fontsize=12)
    at.legend(fontsize=12, loc='upper right')
    for ax in fig.axes:
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('PRD · solv 5.5 · meta-h · runs 20 and 66 · 300 K',fontsize=22)
    fig.savefig(args.out.with_suffix('.png'),dpi=200)
    plt.close(fig)
    save(args.out/'statistics.csv',np.array(stats),'run,tdiff_aligned_ps,tstop_aligned_ps,treact_aligned_ps,mean_delta_F_kcal_mol,std_delta_F_kcal_mol,mean_planarity_pre_reaction_deg,mean_planarity_post_reaction_deg')
    (args.out/'provenance.json').write_text(json.dumps(dict(runs=records,ring_ids=ring,quartets=[[ring[(i+j)%6] for j in range(4)] for i in range(6)],settings=vars(args),horizontal_lines='Per-run window means, labels rounded to one decimal; not a convergence test',layout='2x2: FES, delta F, distance, H-bond strips',colors=colors),default=str,indent=2)+'\n')


if __name__ == '__main__':
    main()
