"""Portable CPP run-5 tautomer movie; geometry only, no defect analysis."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from prn_anti_dih import TIME, box_from_input, minimum_image


VMD = r'''
set here [file dirname [file normalize [info script]]]
set fh [open [file join $here frames.csv] r]
gets $fh
set rows {}
while {[gets $fh line] >= 0} {lappend rows [split $line ,]}
close $fh
source [file join $here topology.tcl]
mol new [file join $here trajectory.xyz] type xyz waitfor all
set mid [molinfo top]
if {[molinfo $mid get numframes] != [llength $rows]} {error "Frame count mismatch"}
mol delrep 0 $mid
color Display Background white
color change rgb 23 0.55 0.90 0.55
display projection Orthographic
axes location Off
display depthcue on
mol representation VDW 0.22 16
mol selection "index 0 to 76"
mol color Name
mol material Opaque
mol addrep $mid
# Faint solvent context, as in the other full-solute movies.
mol representation VDW 0.12 12
mol selection "index 77 to end"
mol color Name
mol material Transparent
mol addrep $mid
mol showrep $mid 1 on
proc cpp_draw {name element op} {
    global mid rows heavy_bonds
    set f $::vmd_frame($mid)
    set row [lindex $rows $f]
    set sel [atomselect $mid all frame $f]
    set xyz [$sel get {x y z}]
    $sel delete
    graphics $mid delete all
    graphics $mid material Opaque
    graphics $mid color gray
    foreach pair $heavy_bonds {
        lassign $pair a b
        graphics $mid cylinder [lindex $xyz [expr {$a-1}]] [lindex $xyz [expr {$b-1}]] radius .12 resolution 14
    }
    foreach pair [split [lindex $row 6] {;}] {
        if {$pair eq ""} {continue}
        lassign [split $pair -] d h
        graphics $mid color gray
        set radius .10
        set sphere_radius .22
        if {$d > 77} {set radius .025; set sphere_radius .10}
        if {$d == 19 || $d == 31} {graphics $mid color yellow}
        graphics $mid cylinder [lindex $xyz [expr {$d-1}]] [lindex $xyz [expr {$h-1}]] radius $radius resolution 12
        # HPD water style: transparent VDW atoms plus thin gray cylinders only.
        # No additional opaque solvent-H spheres.
        if {$d <= 77} {graphics $mid sphere [lindex $xyz [expr {$h-1}]] radius $sphere_radius resolution 16}
    }
    foreach id {19 31} label {NB NC} colour {blue orange} {
        set p [lindex $xyz [expr {$id-1}]]
        graphics $mid color $colour
        graphics $mid sphere $p radius .38 resolution 20
        graphics $mid text [vecadd $p {0 .6 0}] "$label ($id)" size 1.1 thickness 2
    }
    # Fixed-H LCOD vectors, present at every frame: NB19 -> H20 and NC31 -> H20.
    graphics $mid color 23
    foreach id {19 31} {
        set p [lindex $xyz [expr {$id-1}]]
        set q [lindex $xyz 19]
        set v [vecsub $q $p]
        set tipbase [vecadd $p [vecscale .80 $v]]
        graphics $mid cylinder $p $tipbase radius .075 resolution 12
        graphics $mid cone $tipbase $q radius .16 resolution 16
    }
    graphics $mid color yellow
    graphics $mid sphere [lindex $xyz 19] radius .24 resolution 18
    graphics $mid color black
    graphics $mid text {-8 -8 0} "[lindex $row 5] | elapsed [lindex $row 2] ps | raw [lindex $row 1] ps" size .9 thickness 1
    graphics $mid text {-8 -8.7 0} "NB-H20 [lindex $row 8] A | NC-H20 [lindex $row 9] A | LCOD [lindex $row 10] A" size .85 thickness 1
}
trace add variable ::vmd_frame($mid) write cpp_draw
animate goto 0
cpp_draw {} {} {}
display resetview
animate speed .08
proc tautomer_nb {} {global representative_nb; animate goto $representative_nb}
proc tautomer_nc {} {global representative_nc; animate goto $representative_nc}
puts "NB blue; NC orange; fixed H20 yellow; both LCOD vectors light green in all frames."
puts "Commands: tautomer_nb; tautomer_nc; animate forward; animate pause"
puts "Waters visible as small spheres and thin geometric bonds. No H-bond/defect graphics."
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('systems/CPP/solv_4.0/dftb/N1T64C1/run-5/meta-hib'))
    parser.add_argument('--topology', type=Path, default=Path('systems/CPP/init/cpp.mol2'))
    parser.add_argument('--out', type=Path, default=Path('reports/taut_cpp_meta_hib/run-5/vmd_tautomers'))
    parser.add_argument('--refresh', action='store_true', help='Replace derived movie files in an existing movie directory')
    parser.add_argument('--bond-cutoff', type=float, default=1.3, help='Nearest-heavy H ownership cutoff (angstrom)')
    parser.add_argument('--ownership-margin', type=float, default=.15, help='Nearest vs second-nearest distance advantage (angstrom)')
    parser.add_argument('--da-cutoff', type=float, default=3.0, help='H-bond donor-acceptor cutoff (angstrom)')
    parser.add_argument('--angle-cutoff', type=float, default=135., help='Minimum D-H-A angle (degrees)')
    args = parser.parse_args()
    if args.out.exists() and not args.refresh:
        parser.error('Output already exists; choose a new --out')
    if args.out.exists() and not (args.out/'provenance.json').is_file():
        parser.error('Refresh requires an existing generated movie directory')
    if args.out.resolve().is_relative_to(Path('systems').resolve()):
        parser.error('Do not write under systems')
    if not all(np.isfinite(v) for v in (args.bond_cutoff, args.ownership_margin, args.da_cutoff, args.angle_cutoff)) or min(args.bond_cutoff, args.da_cutoff) <= 0 or args.ownership_margin < 0 or not 0 <= args.angle_cutoff <= 180:
        parser.error('Invalid geometry thresholds')
    sections = args.topology.read_text().split('@<TRIPOS>')
    atoms = next(s for s in sections if s.startswith('ATOM')).splitlines()[1:]
    atoms = [s.split() for s in atoms if s.strip()]
    assert len(atoms) == 77 and atoms[18][1] == 'NB' and atoms[30][1] == 'NC'
    topology_symbols = [a[5][0].upper() for a in atoms]
    pairs = [tuple(int(v)-1 for v in s.split()[1:3]) for s in next(s for s in sections if s.startswith('BOND')).splitlines()[1:] if s.strip()]
    heavy_bonds = [(a, b) for a, b in pairs if topology_symbols[a] != 'H' and topology_symbols[b] != 'H']
    box = box_from_input(args.source / 'dftb.inp')
    path = args.source / 'traject'
    size = path.stat().st_size
    with path.open('rb') as handle:
        snapshot = handle.read(size).decode()
    stream = io.StringIO(snapshot)
    rows, contacts, frames = [], [], []
    reference = None
    while line := stream.readline():
        count = int(line)
        match = TIME.search(stream.readline())
        fields = [stream.readline().split() for _ in range(count)]
        if any(len(f) < 4 for f in fields):
            break
        if match is None:
            raise ValueError('Missing timestamp')
        time = float(match[1])/1000
        if rows and time <= rows[-1][1]:
            raise ValueError('Nonmonotonic timestamps')
        symbols = [f[0] for f in fields]
        assert symbols[:77] == topology_symbols
        xyz = np.array([list(map(float, f[1:4])) for f in fields])
        heavy = np.array([i for i, s in enumerate(symbols) if s != 'H'])
        hs = np.array([i for i, s in enumerate(symbols) if s == 'H'])
        vec = minimum_image(xyz[hs, None]-xyz[None, heavy], box)
        dist = np.linalg.norm(vec, axis=2)
        order = np.argsort(dist, axis=1)
        nearest = heavy[order[:, 0]]
        d0 = dist[np.arange(len(hs)), order[:, 0]]
        d1 = dist[np.arange(len(hs)), order[:, 1]]
        bonds = [(int(d), int(h)) for d, h, good in zip(nearest, hs, (d0 <= args.bond_cutoff) & (d1-d0 >= args.ownership_margin)) if good]
        # Unwrap the connected solute heavy skeleton, then image mobile H by geometry.
        image = minimum_image(xyz-xyz[18], box)
        seen = {18}
        while len(seen) < sum(s != 'H' for s in topology_symbols):
            old = len(seen)
            for a, b in heavy_bonds:
                if b in seen and a not in seen:
                    a, b = b, a
                if a in seen and b not in seen:
                    image[b] = image[a]+minimum_image(xyz[b]-xyz[a], box)
                    seen.add(b)
            if len(seen) == old:
                raise ValueError('Disconnected solute topology')
        image[hs] = image[nearest]+vec[np.arange(len(hs)), order[:, 0]]
        fit_ids = np.array(sorted(seen))
        center = image[fit_ids].mean(axis=0)
        image -= center
        if reference is None:
            reference = image[fit_ids].copy()
        u, _, vt = np.linalg.svd(image[fit_ids].T @ reference)
        rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        image = image @ rotation
        nb = [h+1 for d, h in bonds if d == 18]
        nc = [h+1 for d, h in bonds if d == 30]
        state = 'NB-H tautomer' if nb and not nc else 'NC-H tautomer' if nc and not nb else 'both sites protonated' if nb and nc else 'shared/unassigned H'
        frame_contacts = []
        for donor, h in bonds:
            if donor not in (18, 30):
                continue
            acceptor = 30 if donor == 18 else 18
            dh = minimum_image(xyz[donor]-xyz[h], box)
            ha = minimum_image(xyz[acceptor]-xyz[h], box)
            da = np.linalg.norm(minimum_image(xyz[acceptor]-xyz[donor], box))
            angle = np.degrees(np.arccos(np.clip(np.dot(dh, ha)/(np.linalg.norm(dh)*np.linalg.norm(ha)), -1, 1)))
            if da <= args.da_cutoff and angle >= args.angle_cutoff:
                np.testing.assert_allclose(np.linalg.norm(image[acceptor]-image[h]), np.linalg.norm(ha), atol=1e-7)
                frame_contacts.append(f'{h+1}-{acceptor+1}')
                contacts.append([len(rows), time, donor+1, h+1, acceptor+1, np.linalg.norm(dh), np.linalg.norm(ha), da, angle])
        origin = rows[0][1] if rows else time
        distances = np.linalg.norm(minimum_image(xyz[[18,30]]-xyz[19],box),axis=1)
        np.testing.assert_allclose(np.linalg.norm(image[[18,30]]-image[19],axis=1),distances,atol=1e-7)
        rows.append([len(rows), time, time-origin, ';'.join(map(str, nb)), ';'.join(map(str, nc)), state, ';'.join(f'{d+1}-{h+1}' for d, h in bonds), ';'.join(frame_contacts), *[f'{v:.6f}' for v in distances],f'{distances[0]-distances[1]:.6f}'])
        frames.append(image)
    if not rows:
        raise ValueError('No complete frames')
    representatives = {}
    for site in ('NB', 'NC'):
        candidates = [r[0] for r in rows if r[5] == f'{site}-H tautomer']
        if not candidates:
            raise ValueError(f'No uniquely assigned {site} tautomer')
        # Middle of longest uninterrupted residence, not a flickering assignment.
        groups = np.split(np.array(candidates), np.where(np.diff(candidates) != 1)[0]+1)
        longest = max(groups, key=len)
        representatives[site] = int(longest[len(longest)//2])
    args.out.mkdir(parents=True, exist_ok=args.refresh)
    with (args.out/'trajectory.xyz').open('w') as handle:
        for row, xyz in zip(rows, frames):
            handle.write(f'{len(symbols)}\nraw_time_ps={row[1]:.6f}\n')
            for s, p in zip(symbols, xyz):
                handle.write(f'{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n')
    for name, header, data in (
        ('frames.csv', ['frame', 'raw_time_ps', 'elapsed_ps', 'NB_hydrogen_ids', 'NC_hydrogen_ids', 'state', 'geometric_H_bonds', 'hbond_H_acceptor_pairs', 'r_NB19_H20_A', 'r_NC31_H20_A', 'LCOD_A'], rows),
        ('hbonds.csv', ['frame', 'raw_time_ps', 'donor_id', 'hydrogen_id', 'acceptor_id', 'DH_A', 'HA_A', 'DA_A', 'DHA_deg'], contacts)):
        with (args.out/name).open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(data)
    (args.out/'view_tautomers.vmd').write_text(VMD)
    (args.out/'topology.tcl').write_text('set heavy_bonds {'+' '.join('{%d %d}' % (a+1, b+1) for a, b in heavy_bonds)+'}\n'+f'set representative_nb {representatives["NB"]}\nset representative_nc {representatives["NC"]}\n')
    provenance = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    provenance.update(frames=len(rows), raw_time_range_ps=[rows[0][1], rows[-1][1]], snapshot_bytes=size, representatives=representatives, hbond_contacts=len(contacts))
    (args.out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (args.out/'README.txt').write_text(
        'Open with: vmd -e view_tautomers.vmd\nKeep all files together. Full available metadynamics; no equil prefix.\n'
        'NB19 blue; NC31 orange; their currently bound H yellow; solute thick bonds.\n'
        'Both fixed LCOD vectors NB19 -> H20 and NC31 -> H20 are light-green arrows at EVERY frame.\n'
        'Waters are visible as small element-colored spheres and thin dynamic geometric bonds.\n'
        f'Criteria: nearest-heavy donor-H <= {args.bond_cutoff} A; ownership margin >= {args.ownership_margin} A; D-A <= {args.da_cutoff} A; D-H-A >= {args.angle_cutoff} degrees.\n'
        'No H-bond graphics, defect/Mulliken analysis or water-wire tracking. H-bond CSV is legacy supplementary data only.\n'
        'Hydrogen identity may change. Ambiguous ownership is not forced into either tautomer.\n'
        'Tautomer labels describe N-H occupancy, not a formal electronic/bond-order assignment.\n'
        'All geometry uses periodic minimum images; solute unwrapped and heavy-atom aligned.\n'
        'Commands: tautomer_nb / tautomer_nc jump to representative frames; animate forward plays.\n'
        'Optional solvent: mol showrep top 1 on. Raw time +20 ps matches the previous CV grid.\n'
        'frames.csv and hbonds.csv contain one-based atom IDs and actual trajectory timestamps.\n'
        'VMD graphical rendering is not available on the preparation host.\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
