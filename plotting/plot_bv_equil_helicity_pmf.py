"""Pool equilibration ring-center helicity using the BV summary definition."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acid_base_BV import _signed_dihedral_deg
from bv_conformation_basins import dihedrals, wrap_degrees
from bv_ring_defect_distances import identify_bv_rings, ring_center
from cpp_competitor_analysis import read_amber_atom_names


def read_helicity(path: Path, rings: dict, box: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a bounded snapshot, parsing only the ring coordinates."""
    with path.open('rb') as handle:
        raw = handle.read(os.fstat(handle.fileno()).st_size)
    info = dict(source=str(path.resolve()), snapshot_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(), incomplete_tail_ignored=False)
    stream = io.BytesIO(raw)
    ring_ids = np.asarray([rings[k] for k in 'ABCD'])
    wanted = sorted(set(ring_ids.ravel()))
    lookup = {atom: i for i, atom in enumerate(wanted)}
    indices = np.array([[lookup[a] for a in ring] for ring in ring_ids])
    times, positions = [], []
    expected = None
    while line := stream.readline():
        if not line.strip():
            continue
        count = int(line)
        if expected is None:
            expected = count
        if count != expected or count < max(wanted):
            raise ValueError(f'Invalid atom count in {path}')
        comment = stream.readline()
        rows = list(itertools.islice(stream, count))
        if len(rows) != count or not rows[-1].endswith(b'\n'):
            info['incomplete_tail_ignored'] = True
            break
        match = re.search(rb'AT T=\s*([\d.]+)\s*FSEC', comment)
        if not match:
            raise ValueError(f'Missing timestamp: {path}')
        times.append(float(match[1]) / 1000)
        positions.append([[float(v) for v in rows[a-1].split()[1:4]] for a in wanted])
    t = np.asarray(times)
    xyz = np.asarray(positions)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError(f'Need increasing trajectory timestamps: {path}')
    points = xyz[:, indices, :]
    anchors = points[:, :, :1, :]
    offsets = points - anchors
    offsets -= box * np.round(offsets / box)
    centers = anchors[:, :, 0, :] + offsets.mean(axis=2)
    helicity = wrap_degrees(dihedrals(centers, box))
    if not np.isfinite(helicity).all():
        raise ValueError(f'Degenerate helicity: {path}')
    # Check the optimized geometry against the actual summary helpers.
    for i in (0, len(t)//2, len(t)-1):
        full = np.zeros((max(wanted), 3))
        full[np.asarray(wanted)-1] = xyz[i]
        reference = _signed_dihedral_deg(np.array([ring_center(full, rings[k], box) for k in 'ABCD']), box)
        if abs(float(wrap_degrees(helicity[i]-reference))) > 1e-8:
            raise ValueError('Summary helicity definition mismatch')
    return t, helicity, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=['BV', 'CPP', 'BPP'], default='BV')
    parser.add_argument('--runs-path', type=Path)
    parser.add_argument('--runs', type=int, nargs='+')
    parser.add_argument('--parm', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--bin-deg', type=float, default=5., help='Bin width in degrees; divides 360')
    parser.add_argument('--discard-ps', type=float, default=0., help='Discard times before this value in ps')
    parser.add_argument('--temperature', type=float, default=300., help='Temperature in K')
    args = parser.parse_args()
    args.runs_path = args.runs_path or Path(f'systems/{args.system}/solv_4.0/dftb/N1T48C1')
    args.parm = args.parm or Path(f'systems/{args.system}/solv_4.0/salt/ready.parm7')
    args.out_dir = args.out_dir or Path(f'reports/{args.system}/equil_helicity_pmf')
    args.runs = args.runs or list(range(1, 201 if args.system == 'BV' else 51))
    if args.bin_deg <= 0 or not np.isclose(360/args.bin_deg, round(360/args.bin_deg)) or args.discard_ps < 0 or args.temperature <= 0:
        parser.error('Invalid histogram, time, or temperature setting')
    solute_atoms = 78 if args.system == 'BV' else 77
    names = read_amber_atom_names(args.parm, solute_atoms)
    nitrogen_ids = []
    for label in 'ABCD':
        if names.count('N'+label) != 1:
            raise ValueError(f'Nonunique ring nitrogen N{label}')
        nitrogen_ids.append(names.index('N'+label)+1)
    rings = identify_bv_rings(args.parm, solute_atoms, nitrogen_ids)
    for label, ids in rings.items():
        if {names[i-1] for i in ids} != {prefix+label for prefix in ('N', 'C1', 'C2', 'C3', 'C4')}:
            raise ValueError(f'Ring {label} does not match summary atom names')
    print('Verified one-based rings:', rings, flush=True)
    for run in args.runs:
        for name in ('traject', 'dftb.inp'):
            if not (args.runs_path/f'run-{run}'/'equil'/name).is_file():
                raise FileNotFoundError(f'Missing equil {name} for run {run}')
    print(f'Verified {len(args.runs)} equilibration inputs and trajectories', flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    provenance, values = [], []
    with (args.out_dir/'helicity_timeseries.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['run_id', 'time_ps', 'helicity_deg'])
        for run in args.runs:
            folder = args.runs_path/f'run-{run}'/'equil'
            inp = (folder/'dftb.inp').read_text()
            vectors = np.array([[float(v) for v in line.split()[1:4]] for line in inp.splitlines() if line.startswith('TV')])
            if vectors.shape != (3, 3) or not np.allclose(vectors, np.diag(np.diag(vectors))) or np.any(np.diag(vectors) <= 0) or 'NPT=TRUE' in inp.upper().replace(' ', ''):
                raise ValueError('Require fixed orthorhombic box')
            t, h, info = read_helicity(folder/'traject', rings, np.diag(vectors))
            keep = t >= args.discard_ps
            t, h = t[keep], h[keep]
            if not len(t):
                raise ValueError(f'No retained frames for run {run}')
            writer.writerows(zip(itertools.repeat(run), t, h))
            values.append(h)
            info.update(run_id=run, frames=len(t), first_ps=float(t[0]), last_ps=float(t[-1]), box_A=np.diag(vectors).tolist())
            provenance.append(info)
            print(f'run-{run}: {len(t)} frames, {t[0]:g}–{t[-1]:g} ps', flush=True)
    edges = np.linspace(-180, 180, round(360/args.bin_deg)+1)
    counts, _ = np.histogram(np.concatenate(values), bins=edges)
    density = counts / (counts.sum()*args.bin_deg)
    pmf = np.full(density.shape, np.nan)
    positive = density > 0
    pmf[positive] = -0.00198720425864083*args.temperature*np.log(density[positive]/density.max())
    assert np.isclose(np.sum(density)*args.bin_deg, 1)
    x = (edges[:-1]+edges[1:])/2
    np.savetxt(args.out_dir/'helicity_pmf.csv', np.column_stack([x, counts, density, pmf]), delimiter=',', comments='', header='helicity_deg,frame_count,probability_density_per_degree,relative_PMF_kcal_mol')
    plt.style.use(Path(__file__).with_name('lefteris.mplstyle'))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    axes[0].bar(x, density, width=args.bin_deg, color='steelblue', alpha=.4)
    axes[0].set(title='Helicity probability density', ylabel=r'$P(\theta)$ (degree$^{-1}$)')
    axes[1].plot(x, pmf, color='steelblue', marker='.', markersize=4)
    axes[1].set(title='Helicity potential of mean force', ylabel='Relative PMF (kcal/mol)')
    axes[1].text(.97, .95, r'$F(\theta)=-RT\ln[P(\theta)/P_{\max}]$', ha='right', va='top', transform=axes[1].transAxes, fontsize=17)
    for ax in axes:
        ax.set(xlim=(-180,180), ylim=(0,None), xlabel='Ring-center helicity (degrees)')
    fig.suptitle(f'{args.system}: equilibration A–B–C–D ring-center helicity · {len(args.runs)} runs', fontsize=19)
    fig.tight_layout(rect=(0,0,1,.94), w_pad=3)
    fig.savefig(args.out_dir/'helicity_pmf.png', dpi=300)
    plt.close(fig)
    metadata = dict(rings_one_based=rings, definition='Signed dihedral of geometric centers of the five heavy atoms of rings A,B,C,D, with minimum-image ring unwrapping and center bonds; same as acid_base_BV.py, unsmoothed.',
                    bin_deg=args.bin_deg, temperature_K=args.temperature, discard_before_ps=args.discard_ps,
                    total_frames=int(counts.sum()), runs=provenance,
                    caveat='Pooled frame-weighted histogram, not a mean of per-run PMFs. No pseudocounts or smoothing; empty bins NaN. Descriptive sampled PMF; convergence and equilibrium basin weights are not established. No metadynamics frames.')
    (args.out_dir/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
