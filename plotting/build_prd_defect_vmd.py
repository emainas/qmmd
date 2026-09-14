"""PRD defect movie with explicit forward-filled identities and bounded water wires."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from build_hpd_defect_vmd import image_frame, VMD_SCRIPT
from prn_acid_base_analysis import aligned_data
from prn_anti_dih import minimum_image


def wire_edges(xyz: np.ndarray, symbols: list[str], box: np.ndarray, defect: int,
               endpoints: list[int], dh: float, ha: float, angle: float, bridges: int) -> set[tuple[int,int,int]]:
    """Union of all simple undirected H-bond paths, <= bridges solvent intermediates.

    Edges retain donor/H/acceptor orientation even though connectivity is undirected.
    Solute nodes are terminal only. No arbitrary path-count cap.
    """
    nodes = endpoints+[i for i,s in enumerate(symbols,1) if i>12 and s=='O']
    hs = np.array([i for i,s in enumerate(symbols,1) if s=='H'])
    vec = minimum_image(xyz[np.array(nodes)-1,None,:]-xyz[hs-1][None,:,:],box)
    dist = np.linalg.norm(vec,axis=2)
    graph = {a:[] for a in nodes}
    for donor,h in zip(*np.where(dist<=dh)):
        candidates = np.flatnonzero((dist[:,h]<=ha)&(dist[:,h]>=dist[donor,h]))
        for acceptor in candidates:
            if acceptor==donor or dist[donor,h]*dist[acceptor,h]<1e-12:continue
            cosine=np.dot(vec[donor,h],vec[acceptor,h])/(dist[donor,h]*dist[acceptor,h])
            if cosine>np.cos(np.deg2rad(angle)):continue
            a,b=nodes[donor],nodes[acceptor]
            edge=(a,int(hs[h]),b)
            graph[a].append((b,edge));graph[b].append((a,edge))
    result=set()
    def visit(node: int, seen: set[int], edges: list[tuple[int,int,int]]) -> None:
        if node in endpoints:
            result.update(edges);return
        if len(edges)>=bridges+1:return
        for nxt,edge in graph[node]:
            if nxt not in seen:visit(nxt,seen|{nxt},edges+[edge])
    visit(defect,{defect},[])
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('systems/PRD/solv_5.5/dftb/N1T48C1/run-47/meta-h'))
    p.add_argument('--out',type=Path,default=Path('reports/PRD/meta-h_selected/run-47/vmd_solute_defect'))
    p.add_argument('--charge-min',type=float,default=-.625)
    p.add_argument('--charge-max',type=float,default=-.525)
    p.add_argument('--dh-cutoff',type=float,default=1.3,help='Donor-H cutoff, Å')
    p.add_argument('--ha-cutoff',type=float,default=2.5,help='H-acceptor cutoff, Å')
    p.add_argument('--angle-cutoff',type=float,default=135.,help='Minimum D-H-A angle, degrees')
    p.add_argument('--max-bridging-waters',type=int,default=4)
    p.add_argument('--solute-sites',type=int,nargs='+',default=[1],help='One-based endpoints; N1 default. Carbon contacts are exploratory only.')
    p.add_argument('--bond-cutoff',type=float,default=1.4,help='Covalent display threshold, Å')
    p.add_argument('--ownership-margin',type=float,default=.15,help='Covalent display nearest-heavy advantage, Å')
    a=p.parse_args()
    if a.out.exists():p.error('Output exists; choose new --out')
    if a.out.resolve().is_relative_to(Path('systems').resolve()):p.error('Output cannot be in systems')
    if not set(a.solute_sites)<=set(range(1,7)):p.error('Solute endpoints must be ring heavy atoms 1–6')
    if min(a.dh_cutoff,a.ha_cutoff,a.bond_cutoff)<=0 or a.max_bridging_waters<0 or not 0<a.angle_cutoff<=180 or a.charge_min>a.charge_max:
        p.error('Invalid geometry or charge cutoffs')
    a.solute_atoms,a.acid_oxygen,a.competitor_oxygen,a.dihedral=12,1,2,[6,1,2,3]
    data=aligned_data(a.source,a)
    if data['symbols'][:12]!=['N','C','C','C','C','C','H','H','H','H','H','H']:
        raise ValueError('Expected PRD ring N1 C2–C6 and original proton H7')
    a.out.mkdir(parents=True)
    last=None;counts={'direct':0,'carried':0,'unassigned':0,'wire_frames':0}
    with (a.out/'trajectory.xyz').open('w') as traj,(a.out/'frames.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['frame','raw_time_ps','meta_elapsed_ps','display_defect_id','current_charge_e','defect_hydrogen_ids',
                         'geometric_heavy_H_bonds','charge_timestamp_matched','N1_display_defect_distance_A',
                         'assignment_status','raw_assigned_defect_id','wire_D_H_A_edges'])
        for i,(t,xyz) in enumerate(zip(data['times'],data['coords'])):
            raw=data['defect'][i]
            if np.isfinite(raw):last=int(raw);status='direct'
            else:status='carried' if last is not None else 'unassigned'
            counts[status]+=1
            imaged,bonds=image_frame(xyz,data['symbols'],data['box'],bond_cutoff=a.bond_cutoff,ownership_margin=a.ownership_margin)
            edges=set();charge='';distance='';hydrogens=[]
            if last is not None:
                oi=int(np.flatnonzero(data['oxygen_ids']==last)[0])
                charge=data['charges'][oi,i]
                hydrogens=[h for d,h in bonds if d==last]
                distance=np.linalg.norm(minimum_image(xyz[last-1]-xyz[0],data['box']))
                edges=wire_edges(xyz,data['symbols'],data['box'],last,a.solute_sites,a.dh_cutoff,a.ha_cutoff,a.angle_cutoff,a.max_bridging_waters)
            counts['wire_frames']+=bool(edges)
            traj.write(f'{len(xyz)}\nframe={i} raw_time_ps={t:.8f}\n')
            for s,c in zip(data['symbols'],imaged):traj.write(f'{s} {c[0]:.9f} {c[1]:.9f} {c[2]:.9f}\n')
            writer.writerow([i,t,t-data['origin'],last or '',charge,';'.join(map(str,hydrogens)),
                ';'.join(f'{d}-{h}' for d,h in bonds),int(data['matched'][i]),distance,status,
                int(raw) if np.isfinite(raw) else '', ';'.join(f'{d}-{h}-{acc}' for d,h,acc in sorted(edges))])
            if i%500==0:print(f'{i}/{len(data["times"])} frames',flush=True)
    script=VMD_SCRIPT.replace('HPD','PRD').replace('H8','H7').replace('hpd_defect_draw','prd_defect_draw')
    script=script.replace('not index 7','not index 6').replace('"index 7"','"index 6"')
    script=script.replace(' {2 7}','').replace('$d <= 7','$d <= 6')
    script=script.replace('set status "defect O$defect_id; q=[lindex $row 4] e"',
                          'set status "defect O$defect_id; [lindex $row 9]; current q=[lindex $row 4] e"')
    draw=r'''
    # Draw all H...acceptor edges belonging to qualifying defect-to-solute paths.
    graphics $defect_molid color green
    foreach edge [split [lindex $row 11] ";"] {
        if {$edge eq ""} {continue}
        lassign [split $edge "-"] donor h acceptor
        set start [lindex $xyz [expr {$h-1}]]
        set delta [vecsub [lindex $xyz [expr {$acceptor-1}]] $start]
        set wrapped {}
        foreach v $delta L $::prd_box {lappend wrapped [expr {$v-$L*round($v/$L)}]}
        set length [veclength $wrapped]
        set n [expr {max(1,int(ceil($length/.22)))}]
        for {set k 0} {$k<$n} {incr k} {
            set b [vecadd $start [vecscale [expr {double($k)/$n}] $wrapped]]
            set e [vecadd $start [vecscale [expr {($k+.55)/$n}] $wrapped]]
            graphics $defect_molid cylinder $b $e radius .065 resolution 10
        }
    }
'''
    script=script.replace('    graphics $defect_molid color black',draw+'\n    graphics $defect_molid color black')
    script='set prd_box {'+' '.join(map(str,data['box']))+'}\n'+script
    script=script.replace('No wire filtering or wire graphics.','All frames; green dashed water-wire edges.')
    script=script.replace('No connected-frame filtering','No connected-frame filtering')
    (a.out/'view_defect.vmd').write_text(script)
    info={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    info.update(counts,frames=len(data['times']),raw_range_ps=[float(data['times'][0]),float(data['times'][-1])])
    (a.out/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
    (a.out/'README.txt').write_text('Open: vmd -e view_defect.vmd\nKeep frames.csv and trajectory.xyz beside the script.\n'
        'Full available metadynamics; no equil prefix. Solute prominent; original H7 yellow; displayed defect O and current bound H cyan.\n'
        'Missing identities are carried forward until a new direct assignment; leading unassigned frames remain unassigned.\n'
        'CARRIED IS A VISUAL CONTINUITY ASSUMPTION, not evidence the excess proton remained on that oxygen. Current charges and bonds are not carried forward.\n'
        'Green dashed cylinders: union of all simple hydrogen-bond paths from displayed defect to configured solute sites, up to max_bridging_waters.\n'
        'Default solute site N1: the conventional hydrogen-bond-capable solute atom. Carbon endpoints can be requested but are exploratory contacts.\n'
        'Criteria: D-H <= dh-cutoff (1.3 A); H-A <= ha-cutoff (2.5 A); D-H-A >= angle-cutoff (135 deg). No separate D-A cutoff.\n'
        'Paths are undirected connectivity; individual D-H-A edges preserve direction. Edges recomputed every frame even for carried identities.\n'
        'Geometry uses periodic minimum images. Hydrogen-bond dashed segments use nearest-image H-A vectors; wrapped box crossings can remain visible.\n'
        'All solvent retained in the HPD-style small transparent spheres and thin gray dynamic bonds.\n'
        'VMD unavailable on generation host; Tcl syntax and numerical output checked. All raw simulations and summary files unchanged.\n')
    print(info)


if __name__=='__main__':main()
