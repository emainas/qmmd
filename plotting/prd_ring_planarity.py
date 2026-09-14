"""PRD ring planarity: mean absolute deviation of six torsions from 0/180 degrees."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from prn_anti_dih import box_from_input, dihedral, run_ids
from prn_acid_base_analysis import snapshot_bytes


def planar_deviation(phi: np.ndarray) -> np.ndarray:
    return np.abs((np.asarray(phi)+90.) % 180. - 90.)


def verify_ring(path: Path) -> tuple[list[int], dict[int,str]]:
    atoms={}; bonds=set(); section=''
    for line in path.read_text().splitlines():
        if line.startswith('@<TRIPOS>'): section=line; continue
        p=line.split()
        if section=='@<TRIPOS>ATOM' and p: atoms[int(p[0])]=p[1]
        if section=='@<TRIPOS>BOND' and p: bonds.add(frozenset(map(int,p[1:3])))
    ring=[1,2,3,4,5,6]
    if [atoms.get(i) for i in ring]!=['N1','C2','C3','C4','C5','C6']:
        raise ValueError('Unexpected PRD topology atom names')
    expected={frozenset((ring[i],ring[(i+1)%6])) for i in range(6)}
    actual={b for b in bonds if b<=set(ring)}
    if actual!=expected: raise ValueError('Topology does not contain the expected six-member ring')
    return ring,atoms


def read_ring_frames(path: Path) -> tuple[np.ndarray,np.ndarray,dict]:
    blob=snapshot_bytes(path); lines=blob.splitlines(keepends=True)
    times=[];coords=[];pos=0; count=None
    while pos<len(lines):
        n=int(lines[pos]); block=n+2
        if pos+block>len(lines) or not lines[pos+block-1].endswith(b'\n'):break
        if count is None:count=n
        if n!=count or n<6:raise ValueError('Inconsistent atom count')
        match=re.search(rb'AT T=\s*([\d.]+)\s*FSEC',lines[pos+1])
        if not match:raise ValueError('Missing frame timestamp')
        ringrows=[l.split() for l in lines[pos+2:pos+8]]
        if [r[0] for r in ringrows]!=[b'N',b'C',b'C',b'C',b'C',b'C']:
            raise ValueError('Trajectory ring atom order mismatch')
        times.append(float(match[1])/1000)
        coords.append([[float(v) for v in r[1:4]] for r in ringrows])
        pos+=block
    t=np.array(times)
    if not len(t) or np.any(np.diff(t)<=0):raise ValueError('Empty or nonmonotonic trajectory')
    return t,np.array(coords),dict(snapshot_bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),
                                   complete_frames=len(t),ignored_trailing_lines=len(lines)-pos)


def ring_metrics(coords: np.ndarray,box: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    quartets=[[(i+j)%6 for j in range(4)] for i in range(6)]
    phi=np.array([[dihedral(c[q],box) for q in quartets] for c in coords])
    dev=planar_deviation(phi)
    # A degenerate torsion makes the entire frame metric undefined, not a five-angle mean.
    return phi,dev,np.mean(dev,axis=1)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=run_ids,default=run_ids('1-50'))
    p.add_argument('--runs-path',type=Path,default=Path('systems/PRD/solv_5.5/dftb/N1T48C1'))
    p.add_argument('--topology',type=Path,default=Path('systems/PRD/init/prd.mol2'))
    p.add_argument('--out',type=Path,default=Path('reports/PRD/equil_ring_planarity'))
    p.add_argument('--bin-width',type=float,default=.5,help='Distribution bin width in degrees; must divide 90')
    args=p.parse_args()
    if args.bin_width<=0 or not np.isclose(90/args.bin_width,round(90/args.bin_width)):
        p.error('Bin width must be positive and divide 90 degrees')
    ring,names=verify_ring(args.topology)
    print('Verified topology ring:',[names[i] for i in ring],flush=True)
    args.out.mkdir(parents=True,exist_ok=False)
    labels=['-'.join(str((i+j)%6+1) for j in range(4)) for i in range(6)]
    header=['run_id','time_ps']+[f'phi_{q}_deg' for q in labels]+[f'deviation_{q}_deg' for q in labels]+['mean_planarity_deviation_deg']
    rows=[];series={};provenance=[]
    with (args.out/'combined.csv').open('w') as combined:
        combined.write(','.join(header)+'\n')
        for run in args.runs:
            source=args.runs_path/f'run-{run}/equil/traject'
            box=box_from_input(source.parent/'dftb.inp')
            t,c,info=read_ring_frames(source)
            phi,dev,mean=ring_metrics(c,box)
            values=np.column_stack([np.full(len(t),run),t,phi,dev,mean])
            np.savetxt(args.out/f'run-{run}.csv',values,delimiter=',',header=','.join(header),comments='')
            np.savetxt(combined,values,delimiter=',')
            series[run]=(t,mean)
            rows.append(dict(run_id=run,frames=len(t),invalid_frames=int(np.sum(~np.isfinite(mean))),
                             first_time_ps=t[0],last_time_ps=t[-1],mean_deg=np.nanmean(mean),std_deg=np.nanstd(mean)))
            provenance.append(dict(run_id=run,source=str(source.resolve()),box_A=box.tolist(),**info))
            print(f'run-{run}: {len(t)} frames; mean {np.nanmean(mean):.3f} degrees',flush=True)
    with (args.out/'runs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axs=plt.subplots(int(np.ceil(len(series)/5)),5,figsize=(20,2.3*np.ceil(len(series)/5)),squeeze=False,sharex=True,sharey=True)
    maximum=max(np.nanmax(v[1]) for v in series.values())
    for ax,(run,(t,y)) in zip(axs.flat,series.items()):
        ax.plot(t,y,lw=.4,color='steelblue');ax.set_title(f'Run {run} | mean {np.nanmean(y):.2f}°',fontsize=10)
        ax.set_ylim(0,maximum*1.05);ax.grid(alpha=.2)
    for ax in list(axs.flat)[len(series):]:ax.set_visible(False)
    fig.supxlabel('Equilibration time (ps; trajectory timestamps)')
    fig.supylabel('Mean ring-dihedral deviation from planarity (°)')
    fig.suptitle('PRD equilibration: six-dihedral mean planarity deviation',y=.998)
    fig.tight_layout(rect=(.02,.02,1,.98));fig.savefig(args.out/'timeseries.png',dpi=160);plt.close(fig)
    edges=np.linspace(0,90,round(90/args.bin_width)+1)
    densities=[]; all_values=np.concatenate([v[1] for v in series.values()])
    fig,ax=plt.subplots(figsize=(9,5))
    for run,(t,y) in series.items():
        valid=y[np.isfinite(y)];counts,_=np.histogram(valid,bins=edges)
        density=counts/(len(valid)*np.diff(edges));densities.append(density)
        ax.stairs(density,edges,color='gray',alpha=.18,lw=.7)
    good=all_values[np.isfinite(all_values)];counts,_=np.histogram(good,bins=edges)
    pooled=counts/(len(good)*np.diff(edges))
    ax.stairs(pooled,edges,color='steelblue',lw=2,label='Pooled frames')
    ax.plot([],[],color='gray',alpha=.6,label='Individual runs')
    ax.set(xlabel='Mean ring-dihedral deviation from planarity (°)',ylabel='Probability density (degree⁻¹)',
           title=f'PRD equilibration | {len(series)} runs | {len(good):,} frames',xlim=(0,min(90,maximum+args.bin_width)))
    ax.legend();fig.tight_layout();fig.savefig(args.out/'distribution.png',dpi=200);plt.close(fig)
    np.savetxt(args.out/'distribution.csv',np.column_stack([edges[:-1],edges[1:],counts,pooled,*densities]),delimiter=',',
               header=','.join(['bin_left_deg','bin_right_deg','pooled_count','pooled_density']+[f'run_{r}_density' for r in series]),comments='')
    (args.out/'provenance.json').write_text(json.dumps(dict(topology=str(args.topology.resolve()),ring_ids=ring,
        ring_names=[names[i] for i in ring],quartets=labels,bin_width_deg=args.bin_width,runs=provenance),indent=2)+'\n')
    (args.out/'README.md').write_text('# PRD equilibration ring planarity\n\n'
        'Ring N1-C2-C3-C4-C5-C6 verified against the MOL2 bonds including closure. All six cyclic '
        'four-atom dihedrals are saved in CSV. Per-angle deviation = abs((phi+90) modulo 180 -90); '
        'per-frame metric is the arithmetic mean of all six deviations, in degrees (0 planar, 90 maximum). '
        'This implements the explicit user-supplied definition; the precise Curtis/Martínez paper was not identified.\n\n'
        'All complete available equilibration frames included, no smoothing, no burn-in or metadynamics. '
        'Each bond vector uses minimum-image geometry with the orthorhombic dftb.inp box. '
        'Degenerate torsions yield NaN for the entire frame. Plot axes zoom to the observed range; metric bounds remain 0–90°. '
        'Combined distribution is frame-weighted (longer available runs contribute more), not a density of run means. '
        'Standard deviations describe sampled fluctuations, not uncertainty of the mean. '
        'Initial-size snapshots ignore incomplete trailing frames; raw files are unchanged.\n')


if __name__=='__main__':main()
