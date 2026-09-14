"""Full-record HPD acid-base reports with the O7 tautomerization competitor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acid_base_HIST import iter_xyz_frames, read_xyz_symbols, read_box_lengths_from_dftb_inp, _minimum_image
from hist_competitor_analysis import rational_coordination
from prn_acid_base_analysis import mulliken_frames, aligned_fes


def site_geometry(xyz: np.ndarray, site: int, hydrogens: np.ndarray,
                  box: np.ndarray, refdist: float) -> tuple[float, float, int]:
    distances = np.linalg.norm(_minimum_image(xyz[hydrogens]-xyz[site], box), axis=1)
    nearest = int(np.argmin(distances))
    return rational_coordination(distances, refdist), float(distances[nearest]), int(hydrogens[nearest]+1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir', type=Path, default=Path('reports/HPD/meta-h_selected'))
    p.add_argument('--runs', type=int, nargs='+', default=[1,2,4,7,11,13])
    p.add_argument('--oxygen-id', type=int, default=7, help='One-based competing solute oxygen')
    p.add_argument('--nitrogen-id', type=int, default=1)
    p.add_argument('--hydrogen-id', type=int, default=8, help='Original labeled proton')
    p.add_argument('--refdist', type=float, default=1.6, help='All-H rational coordination reference distance (Å); exponents 6/12')
    p.add_argument('--bond-guide', type=float, default=1.3, help='Distance guide in Å, not a diffusion criterion')
    args = p.parse_args()
    if args.refdist <= 0 or args.bond_guide <= 0:
        p.error('Distances must be positive')
    for run in args.runs:
        folder = args.report_dir/f'run-{run}'
        original = json.loads((folder/'provenance.json').read_text())
        source = Path(original['source'])
        saved = np.genfromtxt(folder/'wire_input.csv', delimiter=',', names=True)
        symbols = read_xyz_symbols(source/'traject')
        n, o, h = args.nitrogen_id-1, args.oxygen_id-1, args.hydrogen_id-1
        assert symbols[n]=='N' and symbols[o]=='O' and symbols[h]=='H'
        box = read_box_lengths_from_dftb_inp(source/'dftb.inp')
        hydrogens = np.array([i for i,s in enumerate(symbols) if s=='H'])
        mt, charges = mulliken_frames(source/'mulliken', symbols)
        if np.any(np.diff(mt)<=0):
            raise ValueError('Nonmonotonic Mulliken times')
        charge_lookup = {round(t,8):i for i,t in enumerate(mt)}
        targets = {round(t,8):i for i,t in enumerate(saved['time_ps'])}
        data = np.full((len(saved), 11), np.nan)
        seen = set()
        for time, xyz in iter_xyz_frames(source/'traject'):
            key = round(time,8)
            if key not in targets:
                continue
            if key in seen:
                raise ValueError('Duplicate trajectory timestamp')
            seen.add(key)
            ni = site_geometry(xyz,n,hydrogens,box,args.refdist)
            oi = site_geometry(xyz,o,hydrogens,box,args.refdist)
            labeled = np.linalg.norm(_minimum_image(xyz[[n,o]]-xyz[h],box),axis=1)
            qi = charge_lookup.get(key)
            qn,qo = charges[qi,[n,o]] if qi is not None else [np.nan,np.nan]
            data[targets[key]] = [time,*ni,*oi,*labeled,qn,qo]
        if len(seen)!=len(saved):
            raise ValueError('Some saved report times lack exact trajectory frames')
        header='time_ps,N_all_H_coord,N_nearest_H_A,N_nearest_H_id,O_all_H_coord,O_nearest_H_A,O_nearest_H_id,N_labeled_H_A,O_labeled_H_A,N_charge_e,O_charge_e'
        np.savetxt(folder/'summary_competitor.csv',data,delimiter=',',header=header,comments='')
        times,blocks = aligned_fes(source/'fes.dat')
        fes = blocks[-1].copy(); fes[:,1]-=fes[:,1].min()
        np.savetxt(folder/'summary_competitor_fes.csv',fes,delimiter=',',header='coordination_s,F_kcal_mol',comments='')
        t=data[:,0]
        fig,ax=plt.subplots(3,2,figsize=(15,10),constrained_layout=True)
        ax[0,0].plot(t,saved['coordination_s'],color='steelblue')
        ax[0,0].set(title=f'Biased N{n+1}–H{h+1} coordinate',ylabel='Coordination s')
        ax[1,0].scatter(t,saved['N1_Odefect_distance_A'],s=7,color='seagreen')
        ax[1,0].axhline(4,color='gray',ls='--')
        ax[1,0].set(title='Solvent defect (Mulliken-window assignments only)',ylabel='N–defect distance (Å)')
        ax[2,0].plot(fes[:,0],fes[:,1],color='rebeccapurple')
        ax[2,0].set(title=f'Final saved FES at {times[-1]:.3f} ps — not reaction-window pKa',xlabel='Coordination s',ylabel='F(s) (kcal/mol)')
        ax[0,1].plot(t,data[:,1],label=f'N{n+1}–all H',color='steelblue')
        ax[0,1].plot(t,data[:,4],label=f'O{o+1}–all H',color='darkorange')
        ax[0,1].set(title=f'O{o+1} competing proton-acceptor site',ylabel='All-H coordination sum')
        ax[1,1].plot(t,data[:,2],label='N–nearest H',color='steelblue')
        ax[1,1].plot(t,data[:,5],label='O–nearest H',color='darkorange')
        ax[1,1].plot(t,data[:,8],label=f'O–original H{h+1}',color='purple',lw=.8,alpha=.65)
        ax[1,1].axhline(args.bond_guide,color='gray',ls='--',label=f'{args.bond_guide:g} Å guide')
        ax[1,1].set(ylabel='Distance (Å)')
        ax[2,1].plot(t,data[:,9],label=f'N{n+1}',color='steelblue')
        ax[2,1].plot(t,data[:,10],label=f'O{o+1}',color='darkorange')
        ax[2,1].set(ylabel='Mulliken charge (e)')
        for axes in (ax[0,0],ax[1,0],*ax[:,1]):
            axes.set(xlim=(t[0],t[-1]),xlabel='Raw DFTB time (ps)')
            axes.grid(alpha=.2)
        for axes in ax[:,1]:
            axes.legend(fontsize=8)
        fig.suptitle(f'HPD run {run}: solvent proton transfer and competing N–H → O–H tautomerization\n'
                     'Full saved metadynamics window; all-H coordination includes solvent H; no forced diffusion onset',fontsize=13)
        fig.savefig(folder/'summary_with_competitor.png',dpi=160)
        plt.close(fig)
        provenance=dict(settings=vars(args)|{'report_dir':str(args.report_dir)},source=str(source),
                        exact_geometry_frames=len(seen),matched_charge_frames=int(np.isfinite(data[:,9]).sum()),
                        coordination='Sum over all H, r0 in Å, exponents 6/12; not the labeled biased CV',
                        caveat='Bond distance guide alone is not a tautomer or diffusion classification. No reassignment of solvent defect to solute O.',
                        final_fes_time_ps=float(times[-1]))
        (folder/'summary_competitor_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        print(f'run-{run}: competitor report saved; {len(seen)} exact-matched frames',flush=True)


if __name__=='__main__':
    main()
