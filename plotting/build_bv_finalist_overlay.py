"""Extract and align actual pre-marker BV basin representatives for VMD."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from acid_base_BV import iter_xyz_frames
from bv_conformation_basins import dihedrals, wrap_degrees


def unwrap(xyz: np.ndarray, bonds: list[tuple[int, int]], box: np.ndarray) -> np.ndarray:
    result = xyz.copy()
    seen = {0}
    pending = [0]
    adjacency = [[] for _ in xyz]
    for a, b in bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)
    while pending:
        a = pending.pop()
        for b in adjacency[a]:
            if b not in seen:
                delta = xyz[b] - xyz[a]
                result[b] = result[a] + delta - np.round(delta / box) * box
                seen.add(b)
                pending.append(b)
    if len(seen) != len(xyz):
        raise ValueError('Disconnected solute topology')
    for a, b in bonds:
        delta = xyz[b] - xyz[a]
        delta -= np.round(delta / box) * box
        if not np.allclose(result[b] - result[a], delta, atol=1e-6):
            raise ValueError('Inconsistent periodic ring closure')
    return result


def align(xyz: np.ndarray, reference: np.ndarray, heavy: np.ndarray) -> tuple[np.ndarray, float]:
    center = xyz[heavy].mean(axis=0)
    target_center = reference[heavy].mean(axis=0)
    u, _, vt = np.linalg.svd((xyz[heavy] - center).T @ (reference[heavy] - target_center))
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotated = (xyz - center) @ (u @ correction @ vt) + target_center
    rmsd = np.sqrt(np.mean(np.sum((rotated[heavy] - reference[heavy]) ** 2, axis=1)))
    return rotated, float(rmsd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=Path('reports/BV/meta-hic-finalist-runs'))
    parser.add_argument('--topology', type=Path, default=Path('systems/BV/init/bv.mol2'))
    args = parser.parse_args()
    out = args.report_dir / 'metad_bv_overlay'
    out.mkdir(exist_ok=True)
    template = args.topology.read_text().splitlines()
    atom_start = template.index('@<TRIPOS>ATOM') + 1
    bond_start = template.index('@<TRIPOS>BOND') + 1
    n_atoms, n_bonds = map(int, template[2].split()[:2])
    atoms = [line.split() for line in template[atom_start:atom_start+n_atoms]]
    bonds = [tuple(int(v)-1 for v in line.split()[1:3]) for line in template[bond_start:bond_start+n_bonds]]
    heavy = np.array([not atom[1].startswith('H') for atom in atoms])
    rows = []
    reference = None
    for run in (39, 57, 99, 142):
        folder = args.report_dir / f'run-{run}'
        provenance = json.loads((folder/'equil_single_bridge_3d_provenance.json').read_text())
        metad = json.loads((folder/'metad_nc_sasa_provenance.json').read_text())
        box = np.diag(provenance['box_vectors_A'])
        quartets = np.array([provenance['quartets_one_based'][name] for name in ('single5','single10','single15')])-1
        saved = np.genfromtxt(folder/'metad_single_bridge_3d.csv', delimiter=',', names=True)
        with (folder/'metad_single_bridge_3d_basins.csv').open() as handle:
            basin = max(csv.DictReader(handle), key=lambda row: float(row['population']))
        peak = np.array([float(basin[f'{name}_mode_deg']) for name in ('single5','single10','single15')])
        angles = np.column_stack([saved[f'{name}_deg'] for name in ('single5','single10','single15')])
        distances = np.linalg.norm(wrap_degrees(angles-peak), axis=1)
        distances[saved['basin_id'] != float(basin['basin_id'])] = np.inf
        chosen = int(np.argmin(distances))
        report_time = saved['time_ps'][chosen]
        mapping = np.genfromtxt(folder/'metad_nc_sasa_before_summary_marker.csv', delimiter=',', names=True)
        matches = np.flatnonzero(np.isclose(mapping['time_ps'], report_time, atol=1e-8, rtol=0))
        if len(matches) != 1:
            raise ValueError('Ambiguous timestamp mapping')
        geometry_time = float(mapping['geometry_time_ps'][matches[0]])
        trajectory = Path(metad['trajectory'])
        with trajectory.open() as handle:
            count = int(handle.readline()); handle.readline()
            elements = [handle.readline().split()[0] for _ in range(count)]
        if elements[:n_atoms] != [a[1][0] for a in atoms]:
            raise ValueError('Topology element order mismatch')
        for time, coordinates in iter_xyz_frames(trajectory):
            if time is not None and abs(time-geometry_time) < 1e-8:
                xyz = coordinates[:n_atoms]
                break
        else:
            raise ValueError(f'Missing frame for run {run}')
        measured = dihedrals(xyz[quartets][None, ...], box)[0]
        if np.max(np.abs(wrap_degrees(measured-angles[chosen]))) > 1e-5:
            raise ValueError('Selected geometry does not reproduce saved torsions')
        xyz = unwrap(xyz, bonds, box)
        if reference is None:
            reference = xyz - xyz[heavy].mean(axis=0)
        xyz, rmsd = align(xyz, reference, heavy)
        result = template.copy()
        result[1] = f'BV run {run}; metad {geometry_time:.5f} ps'
        for i, atom in enumerate(atoms):
            fields = atom.copy()
            fields[2:5] = [f'{v:.8f}' for v in xyz[i]]
            result[atom_start+i] = ' '.join(fields)
        target = out/f'run-{run}.mol2'
        target.write_text('\n'.join(result)+'\n')
        rows.append(dict(run_id=run, report_time_ps=float(report_time), geometry_time_ps=geometry_time,
                         single5_deg=float(measured[0]), single10_deg=float(measured[1]), single15_deg=float(measured[2]),
                         peak_distance_deg=float(distances[chosen]), heavy_rmsd_to_run39_A=rmsd,
                         trajectory=str(trajectory), output_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    with (out/'representatives.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    script = ['# Standalone BV-only overlay; no trajectory or solvent loaded.',
              'set overlay_dir [file dirname [file normalize [info script]]]',
              'display projection Orthographic', 'color Display Background white', 'axes location Off',
              'display depthcue off', 'display shadows on']
    for run, color in zip((39,57,99,142), (0,1,7,4)):
        script += [f'set m [mol new [file join $overlay_dir run-{run}.mol2] type mol2 waitfor all]',
                   f'mol rename $m "BV run {run}"', 'mol delrep 0 $m',
                   'mol representation Licorice 0.13 16 16', f'mol color ColorID {color}',
                   'mol selection all', 'mol material Opaque', 'mol addrep $m']
    script += ['display resetview', 'puts "BV overlay: 39 blue; 57 red; 99 green; 142 yellow. Toggle molecules in VMD Main."']
    (out/'overlay.tcl').write_text('\n'.join(script)+'\n')
    (out/'README.md').write_text('# BV metadynamics representative overlay\n\n'
        'Launch: `vmd -e overlay.tcl` (or `source /absolute/path/to/overlay.tcl` in VMD).\n\n'
        '39 blue, 57 red, 99 green, 142 yellow; run 190 excluded. Each molecule contains only the original 78 BV atoms.\n'
        'One actual frame per run, closest in periodic three-dihedral distance to the dominant saved density peak, '
        'restricted to the current pre-purple-marker window. These are not averaged structures or a movie.\n'
        'Coordinates are unwrapped through the original MOL2 bond graph, then least-squares aligned on all BV heavy atoms to run 39. '
        'Original topology bonds are retained; no solvent or new solvent-owned protons are included.\n'
        'The CSV records actual/report times, measured dihedrals, peak distances and fitted heavy-atom RMSDs. '
        'Individual frames do not represent the full conformational distributions.\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
