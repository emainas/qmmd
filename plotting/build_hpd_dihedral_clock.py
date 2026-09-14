"""Static HPD O7-C2-N1-H8 bond-axis clock, equil versus LCOD metadynamics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_bv_finalist_overlay import unwrap
from prn_anti_dih import box_from_input, dihedral
from prn_acid_base_analysis import xyz_frames
from prd_ring_planarity import verify_ring


def bond_frame(xyz: np.ndarray) -> np.ndarray:
    """N1 at origin, C2->N1 +z, projected C2->O7 +x; rigid alignment only."""
    z = xyz[0]-xyz[1]
    z = z/np.linalg.norm(z)
    x = xyz[6]-xyz[1]
    x = x-np.dot(x,z)*z
    if np.linalg.norm(x)<1e-10:
        raise ValueError('Degenerate clock reference')
    x /= np.linalg.norm(x)
    y = np.cross(z,x)
    return (xyz-xyz[0]) @ np.column_stack([x,y,z])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=int,default=1)
    p.add_argument('--frames-per-phase',type=int,default=200)
    p.add_argument('--runs-path',type=Path,default=Path('systems/HPD/solv_5.5/dftb/N1T48C1'))
    p.add_argument('--topology',type=Path,default=Path('systems/HPD/init/hpd.mol2'))
    p.add_argument('--out',type=Path)
    args = p.parse_args()
    if args.run<1 or args.frames_per_phase<2:
        p.error('Positive run and at least two frames per phase required')
    verify_ring(args.topology)
    parts = args.topology.read_text().split('@<TRIPOS>')
    atoms = [s.split() for s in next(v for v in parts if v.startswith('ATOM')).splitlines()[1:] if s.strip()]
    bonds = [s.split()[1:4] for s in next(v for v in parts if v.startswith('BOND')).splitlines()[1:] if s.strip()]
    if len(atoms)!=12 or atoms[6][1]!='O2' or atoms[7][1]!='HN1':
        raise ValueError('Unexpected HPD atom identities')
    graph = [(int(a)-1,int(b)-1) for a,b,_ in bonds]
    out = args.out or Path(f'reports/HPD/meta-lcod/run-{args.run}/dihedral_clock_vmd')
    if out.exists():
        raise FileExistsError(f'Choose a new --out; preserving {out}')
    conformers, records = [], []
    for phase in ('equil','meta-lcod'):
        source = args.runs_path/f'run-{args.run}'/phase
        box = box_from_input(source/'dftb.inp')
        frames = [(t,c[:12]) for t,c in xyz_frames(source/'traject')]
        times = np.array([t for t,c in frames])
        if len(times)<args.frames_per_phase or np.any(np.diff(times)<=0):
            raise ValueError('Insufficient or nonmonotonic trajectory frames')
        if phase=='equil':
            boundary = times[-1]
        for idx in np.linspace(0,len(frames)-1,args.frames_per_phase,dtype=int):
            t,xyz = frames[idx]
            phi = dihedral(xyz[[6,1,0,7]],box)
            aligned = bond_frame(unwrap(xyz,graph,box))
            after = dihedral(aligned[[6,1,0,7]],np.full(3,100.))
            if abs((after-phi+180)%360-180)>1e-7:
                raise ValueError('Alignment altered dihedral')
            conformers.append(aligned)
            records.append(dict(conformer_id=len(conformers),phase=phase,frame_index_zero_based=int(idx),
                raw_time_ps=t,plotted_time_ps=t if phase=='equil' else t-times[0]+boundary,
                dihedral_deg=phi,source=str((source/'traject').resolve())))
    n = len(conformers)
    content=['@<TRIPOS>MOLECULE','HPD_clock',f'{12*n} {len(bonds)*n} {n} 0 0','SMALL','USER_CHARGES','','@<TRIPOS>ATOM']
    for k,xyz in enumerate(conformers):
        for i,(atom,c) in enumerate(zip(atoms,xyz)):
            content.append(f'{12*k+i+1} {atom[1]} '+ ' '.join(f'{v:.8f}' for v in c)+f' {atom[5]} {k+1} HPD {atom[8]}')
    content.append('@<TRIPOS>BOND')
    for k in range(n):
        for i,(a,b,order) in enumerate(bonds):
            content.append(f'{len(bonds)*k+i+1} {12*k+int(a)} {12*k+int(b)} {order}')
    content.append('@<TRIPOS>SUBSTRUCTURE')
    content.extend(f'{k+1} HPD {12*k+1} RESIDUE 0 **** 0 ROOT' for k in range(n))
    out.mkdir(parents=True)
    (out/'conformers.mol2').write_text('\n'.join(content)+'\n')
    with (out/'frames.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    cutoff = 12*args.frames_per_phase
    script = '''# All sampled conformers displayed simultaneously; no solvent.
set clock_dir [file dirname [file normalize [info script]]]
set m [mol new [file join $clock_dir conformers.mol2] type mol2 waitfor all]
mol rename $m HPD_RUN_CLOCK
mol delrep 0 $m
mol representation Licorice 0.045 20 20
mol color ColorID 2
mol selection "index 0 to 11"
mol material Opaque
mol addrep $m
'''
    for color,selection in ((0,f'index 0 to {cutoff-1}'),(3,f'index {cutoff} to {12*n-1}')):
        script+=f'''mol representation Licorice 0.025 16 16
mol color ColorID {color}
mol selection "({selection}) and name O2 C2 N1 HN1"
mol addrep $m
mol representation VDW 0.045 16
mol selection "({selection}) and name HN1"
mol addrep $m
'''
    script+='''color Display Background white
display projection Orthographic
display depthcue off
axes location Off
graphics $m color gray
for {set deg -180} {$deg < 180} {incr deg 30} {
    set a [expr {$deg*acos(-1)/180.0}]
    graphics $m line [list [expr {1.2*cos($a)}] [expr {1.2*sin($a)}] 0] [list [expr {1.3*cos($a)}] [expr {1.3*sin($a)}] 0] width 2
}
graphics $m color black
graphics $m text {1.35 0 0} "0 deg" size 0.7
graphics $m text {-1.85 0 0} "180" size 0.7
graphics $m text {0 1.4 0} "+90" size 0.7
graphics $m text {0 -1.5 0} "-90" size 0.7
graphics $m color blue
graphics $m text {-1.8 1.8 0} "Equilibration" size 0.7
graphics $m color orange
graphics $m text {0.1 1.8 0} "Metadynamics" size 0.7
display resetview
molinfo $m set rotate_matrix {{1 0 0 0} {0 1 0 0} {0 0 1 0} {0 0 0 1}}
molinfo $m set center_matrix {{1 0 0 0} {0 1 0 0} {0 0 1 0} {0 0 0 1}}
# Toggle phase bouquets independently; keep the single gray reference.
proc clock_equil {} {global m; mol showrep $m 1 1; mol showrep $m 2 1; mol showrep $m 3 0; mol showrep $m 4 0}
proc clock_meta {} {global m; mol showrep $m 1 0; mol showrep $m 2 0; mol showrep $m 3 1; mol showrep $m 4 1}
proc clock_both {} {global m; foreach rep {1 2 3 4} {mol showrep $m $rep 1}}
puts "View N1 toward C2, O7 to the right. Blue equil; orange meta. Commands: clock_equil, clock_meta, clock_both."
'''
    (out/'clock.tcl').write_text(script)
    fig,ax=plt.subplots(figsize=(7,7))
    for k,xyz in enumerate(conformers):
        color='tab:blue' if k<args.frames_per_phase else 'tab:orange'
        for a,b in ((6,1),(1,0),(0,7)):
            ax.plot(xyz[[a,b],0],xyz[[a,b],1],color=color,alpha=.2,lw=.7)
        ax.scatter(xyz[7,0],xyz[7,1],s=3,color=color,alpha=.3)
    for color,label in (('tab:blue','Equilibration'),('tab:orange','Metadynamics')):
        ax.plot([],[],color=color,label=label)
    for deg in range(-180,180,30):
        a=np.deg2rad(deg)
        ax.plot(np.cos(a)*np.array([1.2,1.3]),np.sin(a)*np.array([1.2,1.3]),color='gray',lw=.8)
    ax.text(1.35,0,'0°');ax.text(-1.7,0,'180°')
    ax.set(aspect='equal',xlim=(-1.9,1.9),ylim=(-1.7,1.7),xlabel='Aligned x (Å)',ylabel='Aligned y (Å)',
           title=f'HPD run-{args.run}: O7–C2–N1–H8\nView N1 → C2; {args.frames_per_phase} conformers per phase')
    ax.legend();fig.tight_layout();fig.savefig(out/'preview.png',dpi=180);plt.close(fig)
    (out/'README.md').write_text(f'''# HPD run-{args.run} dihedral clock

Open with `vmd -e clock.tcl` from this directory, or source clock.tcl in VMD.
Keep conformers.mol2 beside clock.tcl. The static MOL2 contains {n} conformers,
{args.frames_per_phase} equally spaced saved frames per phase, including endpoints.
This is an equal-per-phase illustration, not a population-weighted distribution.
Blue = equilibration; orange = meta-lcod. Commands in VMD's Tk Console:
`clock_equil`, `clock_meta`, `clock_both` toggle the two bouquets.

Looking from N1 toward C2, O7 points right. N1 at origin, C2→N1 along +z;
C2→O7 projected along +x. Each conformation is rigidly aligned, not torsion-fitted.
Dihedral 7–2–1–8 verified unchanged for every selected frame after periodic
MOL2-bond unwrapping. Original bond lengths and internal geometry are preserved.
The four quartet atoms are in licorice, H8 tips are spheres, and one full gray
solute is the reference. Other atoms are retained in MOL2 but hidden. No solvent.
Topology charges in MOL2 are original static charges, not Mulliken charges.
The fixed N1–H8 bond is appropriate for this run's N-bound frames, not a reactive
bond-ownership visualization for arbitrary runs.

frames.csv records every selected source frame, raw and combined time, and angle.
preview.png is a numerical projection, not a VMD rendering. VMD was unavailable
on the generation host; geometry was checked numerically and Tcl syntax checked.
Original simulations and PRN clock files are untouched.
''')
    print(out)


if __name__=='__main__':
    main()
