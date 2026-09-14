"""Build BV-style static conformer bouquets viewed along PRN's C3–O4 axis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_bv_finalist_overlay import unwrap
from prn_anti_dih import box_from_input, dihedral


def bond_frame(xyz: np.ndarray) -> np.ndarray:
    """O4 at origin; C3→O4 along +z; carbonyl O5 toward +x. No torsion fit."""
    z = xyz[3]-xyz[2]
    z /= np.linalg.norm(z)
    x = xyz[4]-xyz[2]
    x -= np.dot(x,z)*z
    x /= np.linalg.norm(x)
    y = np.cross(z,x)
    return (xyz-xyz[3]) @ np.column_stack([x,y,z])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames-per-run', type=int, default=10)
    p.add_argument('--rotamers', nargs='+', choices=['anti','syn'], default=['anti','syn'])
    p.add_argument('--anti-exclude', type=int, nargs='*', default=[2])
    p.add_argument('--transition-run', type=int, help='Use every saved frame from 0–0.5 ps of one run; time-colored')
    args = p.parse_args()
    if args.frames_per_run<1:
        p.error('frames-per-run must be positive')
    topology = Path('systems/PRN-anti/init/prn.mol2').read_text().splitlines()
    start=topology.index('@<TRIPOS>ATOM')+1
    atoms=[line.split() for line in topology[start:start+11]]
    start=topology.index('@<TRIPOS>BOND')+1
    bonds=[line.split()[1:4] for line in topology[start:start+10]]
    graph=[(int(a)-1,int(b)-1) for a,b,_ in bonds]
    for rotamer in args.rotamers:
        root=Path(f'reports/PRN-{rotamer}')
        out=root/(f'run{args.transition_run}_transition_clock_vmd' if args.transition_run else 'equil_dihedral_clock_vmd')
        out.mkdir(exist_ok=True)
        if (out/'conformers.mol2').exists():
            raise FileExistsError(out/'conformers.mol2')
        table=np.genfromtxt(root/f'prn_{rotamer}_dih.csv',delimiter=',',names=True)
        conformers=[]; records=[]
        excluded=[] if args.transition_run else (args.anti_exclude if rotamer=='anti' else [])
        for run in ([args.transition_run] if args.transition_run else range(1,51)):
            if run in excluded:
                continue
            series=table[table['run_id']==run]
            if args.transition_run:
                series=series[series['time_ps']<=0.5+1e-9]
            if len(series)<args.frames_per_run:
                raise ValueError(f'Insufficient saved frames: {rotamer} {run}')
            choices=np.arange(len(series)) if args.transition_run else np.linspace(0,len(series)-1,args.frames_per_run,dtype=int)
            source=Path(f'systems/PRN-{rotamer}/solv_100/dftb/N1T48C1/run-{run}/equil/traject')
            box=box_from_input(source.parent/'dftb.inp')
            with source.open('rb') as handle:
                blob=handle.read(os.fstat(handle.fileno()).st_size)
            for selected in choices:
                t=float(series['time_ps'][selected])
                marker=f' {1000*t:.2f} FSEC'.encode()
                pos=blob.find(marker)
                if pos<0 or blob.find(marker,pos+len(marker))>=0:
                    raise ValueError('Missing/ambiguous trajectory timestamp')
                end=blob.find(b'\n',pos)
                lines=blob[end+1:].split(b'\n',11)[:11]
                if len(lines)!=11:
                    raise ValueError('Incomplete solute frame')
                fields=[line.split() for line in lines]
                if [f[0].decode() for f in fields]!=[a[1][0] for a in atoms]:
                    raise ValueError('Solute atom order mismatch')
                xyz=np.array([[float(v) for v in f[1:4]] for f in fields])
                phi=dihedral(xyz[[4,2,3,10]],box)
                expected=float(series['signed_dihedral_deg'][selected])
                if abs((phi-expected+180)%360-180)>1e-6:
                    raise ValueError('Frame does not match saved dihedral')
                transformed=bond_frame(unwrap(xyz,graph,np.diag(box) if box.ndim==2 else box))
                # Rigid transform must preserve the torsion.
                after=dihedral(transformed[[4,2,3,10]],np.ones(3)*100)
                if abs((after-phi+180)%360-180)>1e-6:
                    raise ValueError('Alignment changed torsion')
                conformers.append(transformed)
                records.append(dict(conformer_id=len(conformers),run_id=run,time_ps=t,
                                    dihedral_deg=phi,source=str(source),
                                    source_frame_sha256=hashlib.sha256(b'\n'.join(lines)).hexdigest()))
            print(f'{rotamer} run {run}: {len(choices)} frames',flush=True)
        n=len(conformers)
        content=['@<TRIPOS>MOLECULE',f'PRN_{rotamer}_clock',f'{11*n} {10*n} {n} 0 0','SMALL','USER_CHARGES','','@<TRIPOS>ATOM']
        for k,xyz in enumerate(conformers):
            for i,(atom,coord) in enumerate(zip(atoms,xyz)):
                content.append(f'{11*k+i+1} {atom[1]} '+' '.join(f'{v:.8f}' for v in coord)+f' {atom[5]} {k+1} PRN {atom[8]}')
        content.append('@<TRIPOS>BOND')
        for k in range(n):
            for i,(a,b,order) in enumerate(bonds):
                content.append(f'{10*k+i+1} {11*k+int(a)} {11*k+int(b)} {order}')
        content.append('@<TRIPOS>SUBSTRUCTURE')
        content.extend(f'{k+1} PRN {11*k+1} RESIDUE 0 **** 0 ROOT' for k in range(n))
        (out/'conformers.mol2').write_text('\n'.join(content)+'\n')
        with (out/'frames.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        (out/'clock.tcl').write_text('''# Static overlay: all sampled conformers are present simultaneously.
set clock_dir [file dirname [file normalize [info script]]]
set m [mol new [file join $clock_dir conformers.mol2] type mol2 waitfor all]
mol rename $m PRN_CLOCK
mol delrep 0 $m
# One full reference solute, faint and thin.
mol representation Licorice 0.045 20 20
mol color ColorID 2
mol selection "index 0 to 10"
mol material Opaque
mol addrep $m
# All quartet sticks; bond-axis alignment preserves each sampled torsion.
mol representation Licorice 0.025 16 16
mol color ColorID 0
mol selection "name CG O1 O2 H11"
mol material Opaque
mol addrep $m
# Emphasize the bouquet of proton positions.
mol representation VDW 0.045 16
mol color ColorID 1
mol selection "name H11"
mol addrep $m
color Display Background white
display projection Orthographic
display depthcue off
axes location Off
graphics $m color gray
for {set deg 0} {$deg < 360} {incr deg 30} {
    set a [expr {$deg*acos(-1)/180.0}]
    graphics $m line [list [expr {1.12*cos($a)}] [expr {1.12*sin($a)}] 0] [list [expr {1.25*cos($a)}] [expr {1.25*sin($a)}] 0] width 2
}
graphics $m color black
graphics $m text {1.3 0 0} "syn 0" size 0.7
graphics $m text {-2.0 0 0} "anti 180" size 0.7
display resetview
molinfo $m set rotate_matrix {{1 0 0 0} {0 1 0 0} {0 0 1 0} {0 0 0 1}}
molinfo $m set center_matrix {{1 0 0 0} {0 1 0 0} {0 0 1 0} {0 0 0 1}}
puts "Looking from O4 toward C3. Carbonyl O5 is right. Blue: dihedral sticks; red: H11 tips. Rotate freely for a side view."
''')
        if args.transition_run:
            script=(out/'clock.tcl').read_text()
            values=' '.join(str(r['time_ps']) for r in records for _ in range(11))
            setup=f'set sel [atomselect $m all]\n$sel set user {{{values}}}\n$sel delete\ncolor scale method BWR\n'
            script=script.replace('mol delrep 0 $m','mol delrep 0 $m\n'+setup)
            script=script.replace('mol color ColorID 0','mol color User').replace('mol color ColorID 1','mol color User')
            script+='\nmol scaleminmax $m 1 0 0.5\nmol scaleminmax $m 2 0 0.5\n'
            script+=f'graphics $m text {{-1.8 1.5 0}} "Run {args.transition_run}: blue 0 ps -> red 0.5 ps" size 0.7\n'
            script=script.replace('Blue: dihedral sticks; red: H11 tips.','Blue to red: increasing time; sphere tips: H11.')
            (out/'clock.tcl').write_text(script)
        fig,ax=plt.subplots(figsize=(7,7))
        for xyz,record in zip(conformers,records):
            color=plt.get_cmap('coolwarm')(record['time_ps']/0.5) if args.transition_run else 'royalblue'
            for a,b in [(4,2),(2,3),(3,10)]:
                ax.plot(xyz[[a,b],0],xyz[[a,b],1],color=color,alpha=.7 if args.transition_run else .12,lw=.7)
        tips=np.array([xyz[10] for xyz in conformers])
        ax.scatter(tips[:,0],tips[:,1],s=3,color='crimson',alpha=.35)
        if args.transition_run:
            points=ax.scatter(tips[:,0],tips[:,1],c=[r['time_ps'] for r in records],cmap='coolwarm',vmin=0,vmax=.5,s=14)
            fig.colorbar(points,ax=ax,label='Equilibration time (ps)',shrink=.6)
        for deg in range(0,360,30):
            a=np.deg2rad(deg);ax.plot(np.cos(a)*np.array([1.12,1.25]),np.sin(a)*np.array([1.12,1.25]),color='gray')
        ax.text(1.3,0,'syn 0°');ax.text(-1.9,0,'anti 180°')
        ax.set(aspect='equal',xlim=(-2.1,2.1),ylim=(-1.65,1.65),title=f'PRN-{rotamer}: {n} equilibration conformers\nView O4 → C3; excluded runs: {excluded}',xlabel='Aligned x (Å)',ylabel='Aligned y (Å)')
        if args.transition_run:
            ax.set_title(f'PRN-{rotamer} run {args.transition_run}: anti → syn\n{n} actual saved frames, 0–0.5 ps; view O4 → C3')
        fig.savefig(out/'preview.png',dpi=180);plt.close(fig)
        (out/'README.md').write_text(f'# PRN-{rotamer} bond-axis bouquet\n\n'
            f'Launch `vmd -e clock.tcl`. {n} actual equilibration frames, {args.frames_per_run} equally spaced per run; excluded: {excluded}. '
            'Sampling uses the saved 50-panel CSV snapshot, including available partial runs. It is equal-per-run sampling, not an equilibrium population estimate.\n\n'
            'C3→O4 is +z, O4 at origin, O5 projected toward +x. Default camera looks from O4 toward C3. '
            'No torsion or bond-length changes; periodic solute unwrapped using MOL2 bonds. One gray full solute plus all blue dihedral quartets and red H11 tips. '
            'No solvent. Static combined MOL2, not a trajectory movie. All original 11 atoms per conformer are retained but other conformers’ alkyl groups are hidden.\n\n'
            'frames.csv identifies every source frame and records its dihedral and coordinate-block hash. '
            'preview.png is a numerical bond-axis preview, not a VMD render. VMD is unavailable on this host.\n')
        if args.transition_run:
            path=out/'README.md'
            text=path.read_text().replace(f'{args.frames_per_run} equally spaced per run', 'every saved frame in 0–0.5 ps')
            text=text.replace('all blue dihedral quartets and red H11 tips','time-colored dihedral quartets and H11 tips (blue → red)')
            text+=f'\nOnly run {args.transition_run}; no interpolated or fabricated intermediate conformers.\n'
            path.write_text(text)


if __name__=='__main__':
    main()
