"""Audited cpptraj RDFs and w=-RT ln(g) for BV, BPP, and CPP equilibration."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from bv_ring_defect_distances import read_amber_sections
from cpp_competitor_analysis import read_amber_atom_names
from prn_equil_rdf import normalize_counts

PAIRS = {'BV': [('HB', 'O'), ('HC', 'O')],
         'BPP': [('HC', 'O'), ('NB', 'H')],
         'CPP': [('HB', 'O'), ('NC', 'H')]}
TIME = re.compile(rb'AT T=\s*([\d.]+)\s*FSEC')


def topology_selections(parm: Path, system: str) -> dict:
    sections = read_amber_sections(parm)
    masses = np.array(sections['MASS'], dtype=float)
    names = read_amber_atom_names(parm, len(masses))
    starts = list(map(int, sections['RESIDUE_POINTER'])) + [len(masses)+1]
    waters, excluded = [], []
    for i, label in enumerate(sections['RESIDUE_LABEL'][1:], 1):
        ids = list(range(starts[i], starts[i+1]))
        if label != 'WAT':
            excluded.extend(ids)
            continue
        oxy = [a for a in ids if 15 < masses[a-1] < 17]
        hyd = [a for a in ids if .5 < masses[a-1] < 2]
        if len(ids) == 3 and len(oxy) == 1 and len(hyd) == 2:
            waters.append((oxy[0], *hyd))
        else:
            excluded.extend(ids)
    if not waters:
        raise ValueError('No intact topology waters')
    solvent = {'O': [w[0] for w in waters], 'H': [a for w in waters for a in w[1:]]}
    pairs = []
    for site, element in PAIRS[system]:
        matches = [i+1 for i, name in enumerate(names[:starts[1]-1]) if name == site]
        if len(matches) != 1:
            raise ValueError(f'Nonunique site {site}: {matches}')
        ref = matches[0]
        lo, hi = (.5, 2) if site.startswith('H') else (13, 15)
        if not lo < masses[ref-1] < hi:
            raise ValueError('Reference site element mismatch')
        pairs.append(dict(label=f'{site}_water{element}', site=site, reference_id=ref,
                          target_ids=solvent[element], target_element=element))
    return dict(natoms=len(masses), pairs=pairs, nwater=len(waters), excluded_solvent_ids=excluded,
                topology=str(parm.resolve()), topology_sha256=hashlib.sha256(parm.read_bytes()).hexdigest())


def snapshot(path: Path, expected: int, discard: float) -> tuple[bytes, np.ndarray, dict]:
    with path.open('rb') as handle:
        raw = handle.read(os.fstat(handle.fileno()).st_size)
    stream = io.BytesIO(raw)
    spans, times, previous = [], [], -np.inf
    incomplete = False
    while stream.tell() < len(raw):
        begin = stream.tell()
        line = stream.readline()
        if not line.strip():
            continue
        if int(line) != expected:
            raise ValueError(f'Atom count mismatch: {path}')
        match = TIME.search(stream.readline())
        rows = list(itertools.islice(stream, expected))
        if len(rows) != expected or not rows[-1].endswith(b'\n'):
            incomplete = True
            break
        if match is None:
            raise ValueError('Missing timestamp')
        time = float(match[1])/1000
        if time <= previous:
            raise ValueError('Non-increasing times; no silent restart stitching')
        previous = time
        if time >= discard:
            times.append(time)
            spans.append((begin, stream.tell()))
    if not spans:
        raise ValueError('No complete retained frames')
    retained = raw[spans[0][0]:spans[-1][1]]
    info = dict(source=str(path.resolve()), source_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                snapshot_bytes=len(raw), retained_sha256=hashlib.sha256(retained).hexdigest(),
                incomplete_tail_ignored=incomplete, first_ps=times[0], last_ps=times[-1], frames=len(times))
    return retained, np.array(times), info


def analyze_run(task: tuple) -> tuple:
    system, run, settings, args = task
    folder = Path(f'systems/{system}/solv_4.0/dftb/N1T48C1/run-{run}/equil')
    text = (folder/'dftb.inp').read_text()
    vectors = np.array([[float(v) for v in line.split()[1:4]] for line in text.splitlines() if line.startswith('TV')])
    if vectors.shape != (3,3) or not np.allclose(vectors, np.diag(np.diag(vectors))) or np.any(np.diag(vectors) <= 0) or 'NPT=TRUE' in text.upper().replace(' ', ''):
        raise ValueError('Require fixed positive orthorhombic box')
    box = np.diag(vectors)
    if args.rmax > box.min()/2:
        raise ValueError('rmax exceeds half shortest box side')
    volume = float(np.prod(box))
    blob, times, info = snapshot(folder/'traject', settings['natoms'], args.discard_ps)
    first = blob.split(b'\n', settings['natoms']+2)[:settings['natoms']+2]
    symbols = [line.split()[0].decode() for line in first[2:]]
    for pair in settings['pairs']:
        if symbols[pair['reference_id']-1] != pair['site'][0] or any(symbols[a-1] != pair['target_element'] for a in pair['target_ids']):
            raise ValueError('Topology/trajectory element order mismatch')
    results = []
    with tempfile.TemporaryDirectory(prefix=f'{system}-rdf-{run}-', dir='/tmp') as tmp:
        temp = Path(tmp)
        (temp/'snapshot.xyz').write_bytes(blob)
        lines = [f'parm {settings["topology"]}', f'trajin snapshot.xyz 1 {len(times)} as xyz',
                 f'box x {box[0]:.12f} y {box[1]:.12f} z {box[2]:.12f} alpha 90 beta 90 gamma 90']
        for j, pair in enumerate(settings['pairs']):
            mask = ':WAT@O' if pair['target_element'] == 'O' else ':WAT@H1,H2'
            if settings['excluded_solvent_ids']:
                mask = '('+mask+')&!@'+','.join(map(str, settings['excluded_solvent_ids']))
            lines.append(f'radial RDF{j} out rdf{j}.dat {args.spacing} {args.rmax} {mask} @{pair["reference_id"]} volume rawrdf raw{j}.dat intrdf integral{j}.dat')
        lines += ['run', 'quit']
        (temp/'cpptraj.in').write_text('\n'.join(lines)+'\n')
        result = subprocess.run([args.cpptraj, '-i', 'cpptraj.in'], cwd=temp,
                                env={**os.environ, 'OMP_NUM_THREADS': '1'}, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'cpptraj failed ({result.returncode}): {result.stdout}')
        if run == args.runs[0]:
            out = args.report_root/system/'equil_site_water_rdf'
            (out/'cpptraj_example.in').write_text('\n'.join(lines)+'\n')
            (out/'cpptraj_example.log').write_text(result.stdout)
        for j, pair in enumerate(settings['pairs']):
            counts = np.loadtxt(temp/f'raw{j}.dat')[:,1]
            g = np.loadtxt(temp/f'rdf{j}.dat')[:,1]
            integral = np.loadtxt(temp/f'integral{j}.dat')[:,1]
            audited, coord = normalize_counts(counts, len(times), len(pair['target_ids']), volume, args.spacing)
            if not np.allclose(g, audited, atol=6e-4, rtol=1e-4) or not np.allclose(integral, coord, atol=6e-4, rtol=1e-4):
                raise ValueError('Cpptraj normalization audit failed')
            results.append(counts)
        # Audit the entire first selected trajectory with independent NumPy MIC counts.
        if run == args.runs[0]:
            independent = [np.zeros_like(v) for v in results]
            stream = io.BytesIO(blob)
            edges = np.arange(len(results[0])+1)*args.spacing
            for _ in times:
                stream.readline(); stream.readline()
                rows = list(itertools.islice(stream, settings['natoms']))
                xyz = np.array([[float(v) for v in row.split()[1:4]] for row in rows])
                for j, pair in enumerate(settings['pairs']):
                    delta = xyz[np.array(pair['target_ids'])-1]-xyz[pair['reference_id']-1]
                    delta -= box*np.round(delta/box)
                    independent[j] += np.histogram(np.linalg.norm(delta, axis=1), bins=edges)[0]
            if any(not np.array_equal(a,b) for a,b in zip(independent, results)):
                raise ValueError('Independent MIC pair histogram disagrees with cpptraj')
    info.update(run_id=run, box_A=box.tolist(), volume_A3=volume, normalization_audit='passed',
                independent_MIC_audit=run == args.runs[0])
    print(f'{system} run-{run}: {len(times)} frames; RDF normalization passed', flush=True)
    return results, info


def plot_results(system: str, records: list, settings: dict, args: argparse.Namespace) -> None:
    out = args.report_root/system/'equil_site_water_rdf'
    radii = (np.arange(round(args.rmax/args.spacing))+.5)*args.spacing
    edges = np.arange(len(radii)+1)*args.spacing
    shells = 4*np.pi/3*np.diff(edges**3)
    frames = sum(info['frames'] for _, info in records)
    plt.style.use(Path(__file__).with_name('lefteris.mplstyle'))
    fig, axes = plt.subplots(1,2,figsize=(16,6.5))
    with (out/'rdf_pmf.csv').open('w') as handle, (out/'per_run_rdf.csv').open('w') as per:
        writer, rw = csv.writer(handle), csv.writer(per)
        writer.writerow(['pair','r_A','pair_count','g_r','w_kcal_mol','coordination'])
        rw.writerow(['run_id','pair','r_A','pair_count','g_r'])
        for j, pair in enumerate(settings['pairs']):
            counts = sum(result[j] for result, _ in records)
            norm = sum(info['frames']*len(pair['target_ids'])/info['volume_A3'] for _, info in records)
            g = counts/(norm*shells)
            w = np.full(g.shape, np.nan)
            w[g>0] = -0.00198720425864083*args.temperature*np.log(g[g>0])
            coord = np.cumsum(counts)/frames
            writer.writerows(zip(itertools.repeat(pair['label']), radii, counts, g, w, coord))
            for result, info in records:
                per_g, _ = normalize_counts(result[j], info['frames'], len(pair['target_ids']), info['volume_A3'], args.spacing)
                rw.writerows(zip(itertools.repeat(info['run_id']), itertools.repeat(pair['label']), radii, result[j], per_g))
            label = f'{pair["site"]}{pair["reference_id"]}–water {pair["target_element"]}'
            color = ['steelblue','darkorange'][j]
            axes[0].plot(radii,g,color=color,label=label)
            axes[1].plot(radii,w,color=color,label=label)
    axes[0].axhline(1,color='red',ls='--',lw=2)
    # Leave headroom so the g=1 reference cannot strike through the legend.
    axes[0].set_ylim(0, 1.25*max(1., max(float(np.max(line.get_ydata())) for line in axes[0].lines)))
    axes[1].axhline(0,color='red',ls='--',lw=2)
    axes[0].set(title='Radial distribution functions',ylabel=r'$g(r)$',ylim=(0,None))
    axes[1].set(title='Potential of mean force',ylabel=r'$w(r)$ (kcal/mol)')
    axes[1].text(.97,.70,r'$w(r)=-RT\ln[g(r)]$'+'\n'+rf'$T={args.temperature:g}\,\mathrm{{K}}$',
                 transform=axes[1].transAxes,ha='right',va='top',fontsize=17,
                 bbox=dict(boxstyle='round,pad=0.5',facecolor='white',edgecolor='.7'))
    for ax in axes:
        ax.set(xlim=(0,args.rmax),xlabel=r'Site–water atom distance ($\mathrm{\AA}$)')
        ax.legend(loc='upper right')
    fig.suptitle(f'{system}: equilibration site–water structure · {len(records)} runs',fontsize=19)
    fig.tight_layout(rect=(0,0,1,.94),w_pad=3)
    fig.savefig(out/'rdf_pmf.png',dpi=300); plt.close(fig)
    metadata = dict(system=system, selection=settings, total_frames=frames, runs=[info for _, info in records],
                    dr_A=args.spacing,rmax_A=args.rmax,temperature_K=args.temperature,discard_before_ps=args.discard_ps,
                    normalization='sum counts / [shell volume * sum_run(frames * selected target atoms / box volume)]; one reference atom per pair',
                    w_definition='-RT ln g(r), unshifted, kcal/mol; w=0 at g=1. No extra Jacobian.',
                    caveats='Fixed topology intact-water atom IDs, excluding initial non-water residues including two-atom hydroxide labelled WAT. No dynamic proton tracking. All complete retained equil frames, no metadynamics. Exact spherical-shell volumes and periodic imaging; radius <= half shortest box. No tail rescaling, smoothing, pseudocounts or uncertainty estimate. Unsampled w bins NaN.')
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',choices=list(PAIRS),required=True)
    p.add_argument('--runs',nargs='+',type=int)
    p.add_argument('--spacing',type=float,default=.05,help='Radial bin width in Angstrom')
    p.add_argument('--rmax',type=float,default=7,help='Maximum radius in Angstrom')
    p.add_argument('--temperature',type=float,default=300,help='Temperature in K')
    p.add_argument('--discard-ps',type=float,default=0,help='Discard frames earlier than this time (ps)')
    p.add_argument('--workers',type=int,default=2,help='Concurrent single-threaded analysis tasks')
    p.add_argument('--cpptraj',default='cpptraj')
    p.add_argument('--report-root',type=Path,default=Path('reports'))
    p.add_argument('--replot',action='store_true',help='Redraw existing audited CSV data without rereading trajectories')
    args=p.parse_args()
    if args.replot:
        out=args.report_root/args.system/'equil_site_water_rdf'
        metadata=json.loads((out/'provenance.json').read_text())
        settings=metadata['selection']
        args.spacing=metadata['dr_A']; args.rmax=metadata['rmax_A']
        args.temperature=metadata['temperature_K']; args.discard_ps=metadata['discard_before_ps']
        table=np.genfromtxt(out/'per_run_rdf.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
        records=[]
        for info in metadata['runs']:
            arrays=[table['pair_count'][(table['run_id']==info['run_id']) & (table['pair']==pair['label'])] for pair in settings['pairs']]
            records.append((arrays,info))
        plot_results(args.system,records,settings,args)
        return
    if args.spacing<=0 or args.rmax<=0 or not np.isclose(args.rmax/args.spacing,round(args.rmax/args.spacing)) or args.temperature<=0 or args.discard_ps<0 or args.workers<1:
        p.error('Invalid numerical options')
    args.runs = args.runs or list(range(1,201 if args.system=='BV' else 51))
    parm=Path(f'systems/{args.system}/solv_4.0/salt/ready.parm7')
    settings=topology_selections(parm,args.system)
    print(args.system,[(pair['site'],pair['reference_id'],len(pair['target_ids'])) for pair in settings['pairs']], 'waters',settings['nwater'],flush=True)
    out=args.report_root/args.system/'equil_site_water_rdf'
    out.mkdir(parents=True,exist_ok=False)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records=list(pool.map(analyze_run,[(args.system,run,settings,args) for run in args.runs]))
    plot_results(args.system,records,settings,args)


if __name__=='__main__':
    main()
