"""Snapshot-safe cpptraj H11–water-O RDFs with independently audited normalization."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from prn_anti_dih import box_from_input


def normalize_counts(counts: np.ndarray, frames: int, waters: int, volume: float, dr: float) -> tuple[np.ndarray,np.ndarray]:
    edges=np.arange(len(counts)+1)*dr
    shells=4*np.pi/3*np.diff(edges**3)
    return counts/(frames*(waters/volume)*shells),np.cumsum(counts)/frames


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rotamers',nargs='+',default=['syn','anti'],choices=['syn','anti'])
    p.add_argument('--runs',nargs='+',type=int,default=list(range(1,51)))
    p.add_argument('--cpptraj',default='cpptraj')
    p.add_argument('--spacing',type=float,default=.05,help='Radial bin width (Å)')
    p.add_argument('--rmax',type=float,default=7,help='Maximum radius (Å), <= half smallest box length')
    p.add_argument('--report-root',type=Path,default=Path('reports'))
    args=p.parse_args()
    if args.spacing<=0 or args.rmax<=0 or not np.isclose(round(args.rmax/args.spacing)*args.spacing,args.rmax):
        p.error('Positive spacing and rmax must define an integral number of bins')
    for state in args.rotamers:
        out=args.report_root/f'PRN-{state}'/'equil_H11_waterO_rdf'
        out.mkdir(parents=True,exist_ok=False)
        pdb=Path(f'systems/PRN-{state}/solv_100/solvated.pdb').resolve()
        totals=np.zeros(round(args.rmax/args.spacing)); nframes=0; run_rows=[]; common_box=None
        for run in args.runs:
            if state=='anti' and run==2:
                continue
            source=Path(f'systems/PRN-{state}/solv_100/dftb/N1T48C1/run-{run}/equil/traject').resolve()
            box=box_from_input(source.parent/'dftb.inp')
            if args.rmax>box.min()/2:
                raise ValueError('rmax exceeds half the shortest box length')
            if common_box is None: common_box=box
            if not np.allclose(box,common_box,atol=1e-8,rtol=0):
                raise ValueError('Pooling requires identical boxes')
            volume=float(np.prod(box))
            with source.open('rb') as handle:
                blob=handle.read(os.fstat(handle.fileno()).st_size)
            natoms=int(blob.split(b'\n',1)[0]); block_lines=natoms+2
            complete_lines=blob.count(b'\n'); frames=complete_lines//block_lines
            end=len(blob)
            if not blob.endswith(b'\n'): end=blob.rfind(b'\n')+1
            for _ in range(complete_lines%block_lines):
                end=blob.rfind(b'\n',0,end-1)+1
            blob=blob[:end]
            if frames<1: raise ValueError('No complete frames')
            first=blob.split(b'\n',block_lines)[:block_lines]
            symbols=[line.split()[0].decode() for line in first[2:]]
            if natoms!=311 or symbols[10]!='H': raise ValueError('Unexpected PRN solute/trajectory atom order')
            oxy=[i+1 for i,s in enumerate(symbols) if i>=11 and s=='O']
            if len(oxy)!=100: raise ValueError('Expected 100 water oxygens')
            mask='@'+','.join(map(str,oxy))
            folder=out/f'run-{run}';folder.mkdir()
            with tempfile.TemporaryDirectory(prefix=f'prn-rdf-{state}-{run}-',dir='/tmp') as temp:
                snapshot=Path(temp)/'snapshot.xyz';snapshot.write_bytes(blob)
                script=f'''parm {pdb}
trajin {snapshot} 1 {frames} as xyz
box x {box[0]:.10f} y {box[1]:.10f} z {box[2]:.10f} alpha 90 beta 90 gamma 90
radial out {folder.resolve()}/rdf.dat {args.spacing} {args.rmax} {mask} @11 volume intrdf {folder.resolve()}/integral.dat rawrdf {folder.resolve()}/raw.dat
run
quit
'''
                (folder/'cpptraj.in').write_text(script)
                with (folder/'cpptraj.log').open('w') as log:
                    subprocess.run([args.cpptraj,'-i',str(folder/'cpptraj.in')],stdout=log,stderr=subprocess.STDOUT,check=True)
            raw=np.loadtxt(folder/'raw.dat',comments='#'); rdf=np.loadtxt(folder/'rdf.dat',comments='#')
            integral=np.loadtxt(folder/'integral.dat',comments='#')
            counts=raw[:,1]
            if len(counts)!=len(totals): raise ValueError('Unexpected radial bin count')
            computed,coord=normalize_counts(counts,frames,100,volume,args.spacing)
            if not np.allclose(computed,rdf[:,1],atol=6e-4,rtol=1e-4):
                raise ValueError('Cpptraj volume normalization does not match raw-count audit')
            if not np.allclose(coord,integral[:,1],atol=6e-4,rtol=1e-4):
                raise ValueError('Cpptraj coordination integral mismatch')
            # Independent minimum-image pair histogram on the first run (all frames).
            if not run_rows:
                audited=np.zeros_like(counts)
                lines=blob.splitlines()
                for frame in range(frames):
                    base=frame*block_lines+2
                    h=np.array(list(map(float,lines[base+10].split()[1:4])))
                    o=np.array([list(map(float,lines[base+i-1].split()[1:4])) for i in oxy])
                    delta=o-h;delta-=box*np.round(delta/box)
                    audited+=np.histogram(np.linalg.norm(delta,axis=1),bins=np.arange(len(counts)+1)*args.spacing)[0]
                if not np.array_equal(audited,counts):
                    raise ValueError('Independent MIC histogram differs from cpptraj')
            totals+=counts;nframes+=frames
            record=dict(run_id=run,frames=frames,volume_A3=volume,water_oxygen_density_A3=100/volume,
                        snapshot_bytes=len(blob),snapshot_sha256=hashlib.sha256(blob).hexdigest(),source=str(source),
                        first_frame_comment=first[1].decode().strip(),
                        normalization_max_abs_error=float(np.max(np.abs(computed-rdf[:,1]))))
            (folder/'provenance.json').write_text(json.dumps(record,indent=2)+'\n');run_rows.append(record)
            print(f'{state} run {run}: {frames} frames; normalization verified',flush=True)
        g,coord=normalize_counts(totals,nframes,100,float(np.prod(common_box)),args.spacing)
        radii=(np.arange(len(g))+.5)*args.spacing
        np.savetxt(out/'rdf.csv',np.column_stack([radii,totals,g,coord]),delimiter=',',
                   header='r_mid_A,total_pair_count,g_r,cumulative_water_O_per_H',comments='')
        with (out/'runs.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(run_rows[0]));writer.writeheader();writer.writerows(run_rows)
        fig,ax=plt.subplots(figsize=(8,5.5))
        ax.plot(radii,g,color='darkorange' if state=='syn' else 'steelblue',lw=1.8)
        ax.axhline(1,color='gray',ls='--',lw=1)
        ax.set(xlabel='H11–water O distance (Å)',ylabel='g(r)',xlim=(0,args.rmax),ylim=(0,None),
               title=f'PRN-{state}: equilibration H11–water O RDF\n{len(run_rows)} runs · {nframes:,} frames · 100 water oxygens')
        ax.grid(alpha=.2)
        fig.text(.5,.02,f'Periodic imaging; ρO = 100/V = {100/np.prod(common_box):.6f} Å⁻³; Δr = {args.spacing:g} Å.\n'
                 'Pooled complete-frame snapshot; fixed atom H11. Anti run 2 excluded. No forced tail rescaling.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.08,1,1));fig.savefig(out/'rdf.png',dpi=200);plt.close(fig)
        (out/'README.md').write_text(f'# PRN-{state} H11–water O RDF\n\n'
            f'{len(run_rows)} runs; {nframes} complete frames. Atom IDs one-based: proton 11, solvent oxygens {oxy}.\n'
            f'Box {common_box.tolist()} Å, volume {np.prod(common_box):.10f} Å³; oxygen density 100/V. '
            f'Radius capped at {args.rmax} Å (< half shortest box); bin width {args.spacing} Å.\n\n'
            'Cpptraj radial uses volume normalization and periodic imaging. Pooled g_i = total count_i / '
            '[total frames × 1 reference H × (100/V) × (4π/3)(r_outer³−r_inner³)]. '
            'No N−1 correction: the H and O masks are disjoint. Volume is the entire periodic box, '
            'not solvent-accessible volume. Finite-box/solute-exclusion effects can prevent an exact g=1 plateau; '
            'no empirical tail scaling applied. Longer available trajectories contribute proportionally more frames.\n\n'
            'Normalization and cumulative coordination audited for every run; independent NumPy minimum-image '
            'raw histogram checked for all frames of the first run. Per-run outputs/provenance retained; '
            'temporary complete-XYZ snapshots removed after cpptraj. Source files unmodified. '
            'All saved equilibration frames used, no burn-in discard; no metadynamics. Fixed proton identity '
            'does not track proton exchange. Statistical uncertainties are not estimated.\n\n'
            'Cpptraj documentation: https://amberhub.chpc.utah.edu/radial-rdf/\n')


if __name__=='__main__': main()
