"""HIST HIE/HID tiered comparison using the original individual-summary windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acid_base_HIST import (augment_csv, iter_xyz_frames, read_xyz_symbols,
    read_box_lengths_from_dftb_inp, _minimum_image, _signed_dihedral_deg,
    _circular_running_average_deg, load_pka_series, load_fes_snapshot, sample_pka_window)
from plot_pka_grid import PKA_FACTOR, deltaf
from plot_cv_grid import infer_offset_ps
from hist_competitor_analysis import classify_histidine_conformers

SELECTION = [('meta-hie',20,'three-CV_oxygen-wire',14),
             ('meta-hid',1,'four-row-reaction-summary',10),
             ('meta-hid',49,'four-row-reaction-summary',10)]


def save(path: Path, rows: np.ndarray, header: str) -> None:
    np.savetxt(path, rows, delimiter=',', header=header, comments='')


def read(path: Path) -> np.ndarray:
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))


def cached_records(out: Path, settings: dict) -> list[dict]:
    """Redraw existing numerical results without rereading trajectories."""
    metadata=json.loads((out/'provenance.json').read_text())
    for key,value in metadata['settings'].items():
        if key not in {'reuse_wires','plot_only','out','hid49_fes_time','hie20_fes_time','hid1_fes_time'} and str(settings[key])!=str(value):
            raise ValueError(f'Cached setting differs: {key}; rerun without --plot-only')
    records=metadata['runs']
    for d in records:
        label=d['label']
        energy=read(out/f'{label}_delta_f.csv')
        angles=read(out/f'{label}_dihedrals.csv')
        hb=read(out/f'{label}_hbond_plot.csv')
        d.update(time=energy['aligned_time_ps'],energy=energy['delta_F_kcal_mol'],
                 samples=read(out/f'{label}_samples.csv')['delta_F_kcal_mol'],
                 fes=np.loadtxt(out/f'{label}_fes.csv',delimiter=',',skiprows=1),
                 structure_time=angles['aligned_time_ps'],
                 angles=np.column_stack([angles['chi1_deg'],angles['chi2_deg']]),
                 distance=np.loadtxt(out/f'{label}_distance.csv',delimiter=',',skiprows=1),
                 wire_time=hb['aligned_time_ps'],hbonds=hb['number_of_hydrogen_bonds'])
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('reports/HIST/joint_summary_hie_and_hid'))
    p.add_argument('--plot-only',action='store_true',help='Redraw saved report data; requires unchanged analysis settings')
    p.add_argument('--reuse-wires',action='store_true',help='Reuse this report\'s wire CSVs; only when path settings and raw data are unchanged')
    p.add_argument('--runs-path',type=Path,default=Path('systems/HIST/solv_5.5/dftb/N1T64C1'))
    p.add_argument('--topology',type=Path,default=Path('systems/HIST/init/his.mol2'))
    p.add_argument('--smooth-ps',type=float,default=.2)
    p.add_argument('--hie20-grid-tdiff',type=float,default=31.75,help='plot_pka_grid clock, ps')
    p.add_argument('--hid49-grid-tdiff',type=float,default=50.75,help='plot_pka_grid clock, ps')
    p.add_argument('--hid49-fes-time',type=float,default=52.,help='Run 49 FES/pKa snapshot on the aligned comparison clock, ps')
    p.add_argument('--hie20-fes-time',type=float,default=54.,help='Run 20 FES/pKa snapshot on the aligned comparison clock, ps')
    p.add_argument('--hid1-fes-time',type=float,default=48.72,help='Run 1 FES/pKa snapshot on the aligned comparison clock, ps')
    p.add_argument('--chi-bin-deg',type=float,default=10.,help='Angular occupancy bin width, degrees; must divide 360')
    p.add_argument('--reaction-threshold',type=float,default=.05)
    p.add_argument('--max-bridging-waters',type=int,default=8)
    p.add_argument('--covalent-cutoff',type=float,default=1.3,help='Donor-H cutoff, angstrom')
    p.add_argument('--hydrogen-acceptor-cutoff',type=float,default=2.5,help='H-acceptor cutoff, angstrom')
    p.add_argument('--angle-cutoff',type=float,default=135.,help='D-H-A minimum angle, degrees')
    args=p.parse_args()
    if args.smooth_ps<=0 or args.max_bridging_waters<0:
        p.error('Invalid smoothing/path limit')
    if args.chi_bin_deg<=0 or not np.isclose(360/args.chi_bin_deg,round(360/args.chi_bin_deg)):
        p.error('chi-bin-deg must be positive and divide 360')
    sections=args.topology.read_text().split('@<TRIPOS>')
    atoms=[r.split() for r in next(s for s in sections if s.startswith('ATOM')).splitlines()[1:] if r.strip()]
    bonds={frozenset(map(int,r.split()[1:3])) for r in next(s for s in sections if s.startswith('BOND')).splitlines()[1:] if r.strip()}
    names={a[1]:int(a[0]) for a in atoms}
    quartets={k:[names[n] for n in ns] for k,ns in {'chi1':['N','CA','CB','CG'],'chi2':['CA','CB','CG','ND1']}.items()}
    for q in quartets.values():
        assert all(frozenset((a,b)) in bonds for a,b in zip(q,q[1:]))
    assert len(atoms)==21 and names['ND1']==10 and names['NE2']==14
    args.out.mkdir(parents=True,exist_ok=True)
    records=cached_records(args.out,vars(args)) if args.plot_only else []
    for cv,run,tag,nid in ([] if args.plot_only else SELECTION):
        label=f'{cv.removeprefix("meta-").upper()}-{run}'
        stem=Path(f'reports/HIST_solv_5.5_{cv}_run-{run}_{tag}')
        source=(args.runs_path/f'run-{run}'/cv).resolve()
        old_samples=np.loadtxt(str(stem)+'_pka_window_samples.csv',delimiter=',')
        onset,stop=old_samples[[0,-1],0]
        assert len(old_samples)==10 and np.isclose(stop-onset,1.75)
        pt,pk=load_pka_series(source/'fes.dat',temperature=300.)
        grid_offset=infer_offset_ps(source.parent,cv,float(pt[0]))
        if run==20: onset=args.hie20_grid_tdiff-grid_offset
        if run==49: onset=args.hid49_grid_tdiff-grid_offset
        stop=onset+1.75
        original=read(Path(str(stem)+'.csv'))
        if original['time_ps'][-1]<stop-.021:
            raise ValueError('Original defect assignments do not cover requested window')
        original=original[original['time_ps']<=stop+1e-8]
        mapping={round(float(r['time_ps']),8):float(r['defect_oxygen_id']) for r in original}
        symbols=read_xyz_symbols(source/'traject')
        assert symbols[:21]==[a[5][0].upper() for a in atoms]
        box=read_box_lengths_from_dftb_inp(source/'dftb.inp')
        ts=[]; angles=[]; distances=[]; previous=np.nan; offset=None
        for time,xyz in iter_xyz_frames(source/'traject'):
            if offset is None: offset=40-time
            if time>stop+1e-8: break
            ts.append(time)
            angles.append([_signed_dihedral_deg(xyz[np.array(q)-1],box) for q in quartets.values()])
            key=round(time,8)
            if key not in mapping: continue
            defect=mapping[key]; distance=atom=np.nan
            if np.isfinite(defect):
                ds=np.linalg.norm(_minimum_image(xyz[:21]-xyz[int(defect)-1],box),axis=1)
                atom=int(np.argmin(ds))+1;distance=float(ds[atom-1]);previous=distance
            distances.append([time,time+offset,defect,atom,distance,previous,int(not np.isfinite(distance) and np.isfinite(previous))])
        ts=np.array(ts);angles=np.array(angles); distances=np.array(distances)
        if stop-ts[-1]>.021 or not np.all(np.isfinite(angles)): raise ValueError('Missing structural coverage')
        smooth=np.column_stack([_circular_running_average_deg(ts,angles[:,i],args.smooth_ps) for i in range(2)])
        # Audit recomputed torsions against existing HID structural CSVs.
        existing=Path(str(stem)+'_hist_dihedrals.csv')
        if existing.exists():
            old=read(existing); lookup={round(float(t),8):i for i,t in enumerate(ts)}
            audit=[]
            for r in old:
                i=lookup.get(round(float(r['time_ps']),8))
                if i is not None:
                    error=(angles[i]-np.array([r['chi1_deg'],r['chi2_deg']])+180)%360-180
                    audit.append([r['time_ps'],*error])
            save(args.out/f'{label}_legacy_torsion_audit.csv',np.array(audit),
                 'raw_time_ps,exact_minus_legacy_chi1_wrapped_deg,exact_minus_legacy_chi2_wrapped_deg')
        save(args.out/f'{label}_dihedrals.csv',np.column_stack([ts,ts+offset,angles,smooth]),'raw_time_ps,aligned_time_ps,chi1_deg,chi2_deg,chi1_smoothed_deg,chi2_smoothed_deg')
        save(args.out/f'{label}_distance.csv',distances,'raw_time_ps,aligned_time_ps,defect_oxygen_id,nearest_solute_atom_id,distance_A,display_distance_A,carried_forward')
        pt,pk=load_pka_series(source/'fes.dat',temperature=300.)
        st,sv=sample_pka_window(pt,pk,onset,stop,sample_count=10)
        # Old reports used 313.15 K; energy must be invariant to conversion temperature.
        if run==1:
            np.testing.assert_allclose(sv*300,old_samples[:,1]*313.15,rtol=1e-8,atol=1e-7)
        factor=PKA_FACTOR*300;energy=pk*factor;sample_energy=sv*factor
        ft,fes=load_fes_snapshot(source/'fes.dat',stop)
        prior_fes=np.loadtxt(str(stem)+'_fes_snapshot.csv',delimiter=',')
        if run==1: np.testing.assert_allclose(fes,prior_fes,atol=1e-7)
        save(args.out/f'{label}_fes.csv',fes,'coordination_s,F_kcal_mol')
        keep=pt<=stop+1e-8
        save(args.out/f'{label}_delta_f.csv',np.column_stack([pt[keep],pt[keep]+offset,energy[keep]]),'raw_time_ps,aligned_time_ps,delta_F_kcal_mol')
        save(args.out/f'{label}_samples.csv',np.column_stack([st,st+offset,sample_energy,sv]),'raw_time_ps,aligned_time_ps,delta_F_kcal_mol,pka_300K')
        wire_input=args.out/f'{label}_wire_input.csv'
        dist_col=f'N{nid}_Odefect_distance_A'
        save(wire_input,np.column_stack([original[n] for n in ['time_ps','coordination_s','defect_oxygen_id',dist_col]]),f'time_ps,coordination_s,defect_oxygen_id,{dist_col}')
        wire_out=args.out/f'{label}_wires.csv'
        if not (args.reuse_wires and wire_out.exists()):
            augment_csv(wire_input,wire_out,source/'traject',nid,21,box=box,max_bridging_waters=args.max_bridging_waters,
                        covalent_cutoff=args.covalent_cutoff,hydrogen_acceptor_cutoff=args.hydrogen_acceptor_cutoff,angle_cutoff=args.angle_cutoff)
        wire=read(wire_out)
        np.testing.assert_allclose(wire['time_ps'],original['time_ps'])
        np.testing.assert_allclose(wire['defect_oxygen_id'],original['defect_oxygen_id'],equal_nan=True)
        valid=(wire['wire_evaluated']==1)&np.isfinite(wire['defect_oxygen_id'])
        hb=np.where(valid,wire['hbond_bridging_water_count']+1,np.nan)
        save(args.out/f'{label}_hbond_plot.csv',np.column_stack([wire['time_ps'],wire['time_ps']+offset,hb]),'raw_time_ps,aligned_time_ps,number_of_hydrogen_bonds')
        candidate=original['time_ps'][(original['coordination_s']<=args.reaction_threshold)&(original['time_ps']<=onset)]
        react=float(candidate[0]) if len(candidate) else np.nan
        records.append(dict(cv=cv,run=run,label=label,source=str(source),original_report=str(stem),offset=offset,
            onset=onset,stop=stop,grid_offset=grid_offset,reaction=react,snapshot=ft,nitrogen_id=nid,
            time=pt[keep]+offset,energy=energy[keep],samples=sample_energy,fes=fes,
            structure_time=ts+offset,angles=angles,smooth=smooth,distance=distances,
            wire_time=wire['time_ps']+offset,hbonds=hb))
        print(f'{label}: raw tdiff {onset:g}; display {onset+offset:g}; stop {stop+offset:g}',flush=True)
    for d in records:
        if d['run'] in (20,49,1):
            requested={20:args.hie20_fes_time,49:args.hid49_fes_time,1:args.hid1_fes_time}[d['run']]-d['offset']
            d['snapshot'],d['fes']=load_fes_snapshot(Path(d['source'])/'fes.dat',requested)
            if not np.isclose(d['snapshot'],requested,atol=1e-8):
                raise ValueError(f'No exact FES snapshot at requested time for {d["label"]}')
            save(args.out/f'{d["label"]}_fes.csv',d['fes'],'coordination_s,F_kcal_mol')
    plt.style.use(Path(__file__).with_name('lefteris.mplstyle'))
    plt.rcParams.update({'font.size':18,'axes.labelsize':22,'axes.titlesize':22,'legend.fontsize':16})
    fig=plt.figure(figsize=(28,22))
    outer=fig.add_gridspec(2,3,left=.06,right=.98,bottom=.07,top=.94,wspace=.28,hspace=.28)
    axes={}; wires={}
    for panel in range(5):
        if panel==3: continue
        r,c=divmod(panel,3)
        tiers=outer[r,c].subgridspec(2,1,hspace=0)
        upper=fig.add_subplot(tiers[0])
        lower=fig.add_subplot(tiers[1],sharex=upper if panel!=2 else None,sharey=upper)
        axes[panel]=[upper,lower]
        upper.tick_params(labelbottom=False)
    # Keep distinct occupancy maps for the two HID runs; never pool conformers.
    chi_grid=outer[1,0].subgridspec(2,2,wspace=.3,hspace=.3)
    chi_axes=[fig.add_subplot(chi_grid[0,0]),fig.add_subplot(chi_grid[1,0]),fig.add_subplot(chi_grid[1,1])]
    axes[3]=chi_axes[:2]
    chi_edges=np.linspace(-180,180,round(360/args.chi_bin_deg)+1)
    occupancies=[]
    for d in records:
        counts,_,_=np.histogram2d(*(((d['angles']+180)%360-180).T),bins=(chi_edges,chi_edges))
        occupancies.append(counts/counts.sum())
        xx,yy=np.meshgrid((chi_edges[:-1]+chi_edges[1:])/2,(chi_edges[:-1]+chi_edges[1:])/2,indexing='ij')
        save(args.out/f'{d["label"]}_chi_occupancy.csv',np.column_stack([xx.ravel(),yy.ravel(),counts.ravel(),occupancies[-1].ravel()]),'chi1_bin_center_deg,chi2_bin_center_deg,count,probability_per_bin')
    for ax,d,prob in zip(chi_axes,records,occupancies):
        heat=ax.pcolormesh(chi_edges,chi_edges,prob.T,cmap='YlGnBu',vmin=0,vmax=max(q.max() for q in occupancies),shading='flat',rasterized=True)
        ax.set(xlim=(-180,180),ylim=(-180,180),xticks=[-180,0,180],yticks=[-180,0,180])
        ax.set_aspect('equal',adjustable='box')
        ax.text(.03,.94,d['label'],transform=ax.transAxes,va='top',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    chi_axes[2].tick_params(labelleft=False)
    cb=fig.colorbar(heat,ax=chi_axes,location='bottom',fraction=.06,pad=.13,aspect=35)
    cb.set_label('Sampled probability per bin',fontsize=18)
    key_ax=fig.add_subplot(chi_grid[0,1])
    key_ax.set_axis_off()
    key_lines=['Dominant conformer']
    fractions=['run,group,conformer,count,fraction,dominant']
    for d in records:
        labels=classify_histidine_conformers(d['angles'][:,0],d['angles'][:,1])
        codes,counts=np.unique(labels[labels!=''],return_counts=True)
        dominant=str(codes[np.argmax(counts)])
        key_lines.append(f'Run {d["run"]} — {dominant}')
        for code,count in zip(codes,counts):
            fractions.append(f'{d["run"]},{d["cv"]},{code},{count},{count/counts.sum():.10f},{int(code==dominant)}')
    (args.out/'conformer_populations.csv').write_text('\n'.join(fractions)+'\n')
    key_ax.text(.5,.5,'\n\n'.join(key_lines),transform=key_ax.transAxes,
                ha='center',va='center',fontsize=20,
                bbox=dict(boxstyle='round,pad=.8',facecolor='white',edgecolor='0.4'))
    tiers=outer[1,2].subgridspec(3,1,hspace=.12)
    wires[0]=[fig.add_subplot(tiers[0])]
    wires[1]=[fig.add_subplot(tiers[i+1],sharex=wires[0][0],sharey=wires[0][0]) for i in range(2)]
    maxstop=max(d['stop']+d['offset'] for d in records)
    colors=['blue','red']
    stats=[]
    for tier,cv in enumerate(['meta-hie','meta-hid']):
        group=[d for d in records if d['cv']==cv]
        af,at,ab,ac,ad=[axes[i][tier] for i in range(5)]
        for j,d in enumerate(group):
            color='green' if cv=='meta-hid' and d['run']==1 else colors[j]
            name=f'Run {d["run"]}'
            fes_label=f'{name}: t={d["snapshot"]+d["offset"]:g} ps'
            af.plot(d['fes'][:,0],d['fes'][:,1],color=color,lw=2.5,alpha=.65,label=fes_label)
            at.plot(d['time'],d['energy'],color=color,lw=2.4,label=f'{name}: tdiff={d["onset"]+d["offset"]:.3f} ps')
            at.axvline(d['onset']+d['offset'],color=color,ls='--',lw=1.4)
            mu,sd=d['samples'].mean(),d['samples'].std()
            endpoint_df=deltaf(d['fes'],0.,1.,.1,0.,1.25)
            endpoint_pka=endpoint_df/(PKA_FACTOR*300)
            pka_label=f'{name}: pKa({d["snapshot"]+d["offset"]:g} ps)={endpoint_pka:.2f}'
            ab.bar(j,mu,yerr=sd,color=color,alpha=.75,capsize=6,width=.5,label=pka_label)
            ab.text(j,mu+sd+.18,f'{mu:.2f} ± {sd:.2f}',ha='center',fontsize=18)
            ad.scatter(d['distance'][:,1],d['distance'][:,4],color=color,s=18,linewidths=0,label=name)
            ax=wires[tier][j]
            ax.scatter(d['wire_time'],d['hbonds'],color=color,s=19,linewidths=0,label=name)
            ax.legend(loc='upper left',fontsize=16)
            ax.set(ylim=(-.5,args.max_bridging_waters+1.5),yticks=[0,3,6,9],xlim=(40,maxstop))
            ax.tick_params(labelbottom=tier==1 and j==len(group)-1)
            stats.append([tier,d['run'],d['onset'],d['stop'],d['offset'],d['reaction'],d['snapshot'],mu,sd,mu/(PKA_FACTOR*300),endpoint_df,endpoint_pka])
        ab.set_xticks([0,1],['20',''] if tier==0 else ['1','49'])
        ab.set_xlim(-.6,1.6)
        # Separate categorical labels despite shared axis limits.
        if tier==0: ab.tick_params(labelbottom=True)
        for ax in [af,at,ab,ad]: ax.legend(fontsize=16,loc='best')
        for ax in [at,ad]: ax.set_xlim(40,maxstop)
        af.set_xlim(0,1);ad.set_ylim(bottom=0)
        ab.set_ylim(0,max(r[7]+r[8] for r in stats)*1.45)
        af.text(.03,.92,cv.removeprefix('meta-').upper(),transform=af.transAxes,fontsize=22,fontweight='bold')
    labels=['F(s) (kcal/mol)','ΔF (kcal/mol)','Mean ΔF (kcal/mol)','χ2 (degrees)','Defect–nearest solute atom (Å)']
    for i,label in enumerate(labels):
        top,bottom=axes[i]
        pos1,pos2=top.get_position(),bottom.get_position()
        fig.text(pos1.x0-.045,(pos1.y1+pos2.y0)/2,label,rotation=90,ha='center',va='center',fontsize=23)
        bottom.set_xlabel('Coordination s' if i==0 else 'Run ID' if i==2 else 'χ1 (degrees)' if i==3 else 't (ps; metadynamics starts at 40)')
    chi_axes[2].set_xlabel('χ1 (degrees)')
    chi_axes[0].set_xlabel('χ1 (degrees)')
    pos1=wires[0][0].get_position();pos2=wires[1][-1].get_position()
    fig.text(pos1.x0-.045,(pos1.y1+pos2.y0)/2,'Number of hydrogen bonds',rotation=90,ha='center',va='center',fontsize=23)
    wires[1][-1].set_xlabel('t (ps; metadynamics starts at 40)')
    for ax in fig.axes: ax.grid(axis='y',alpha=.15);ax.tick_params(labelsize=18)
    for ax in chi_axes: ax.grid(False)
    fig.suptitle('HIST · solv 5.5 · HIE / HID · 300 K',fontsize=28)
    fig.canvas.draw()
    # All angular maps must have identical square axes; all wire panels equal size.
    for group in [chi_axes,wires[0]+wires[1]]:
        sizes=np.array([[ax.bbox.width,ax.bbox.height] for ax in group])
        np.testing.assert_allclose(sizes,np.tile(sizes[0],(len(group),1)),atol=1e-6)
    assert np.isclose(chi_axes[0].bbox.width,chi_axes[0].bbox.height)
    fig.savefig(args.out.with_suffix('.png'),dpi=180)
    plt.close(fig)
    save(args.out/'statistics.csv',np.array(stats),'tier_0_HIE_1_HID,run,tdiff_raw_ps,tstop_raw_ps,display_offset_ps,treact_raw_ps,fes_snapshot_raw_ps,mean_delta_F_kcal_mol,std_delta_F_kcal_mol,pka_300K,endpoint_delta_F_kcal_mol,endpoint_pka_300K')
    provenance=dict(settings=vars(args),quartets=quartets,old_pka_temperature_K=313.15,temperature_K=300,
        runs=[{k:v for k,v in d.items() if not isinstance(v,np.ndarray)} for d in records])
    (args.out/'provenance.json').write_text(json.dumps(provenance,default=str,indent=2)+'\n')


if __name__=='__main__': main()
