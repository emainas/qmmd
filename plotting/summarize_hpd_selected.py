"""Generate HPD acid-base summaries using timestamp-matched solvent defects."""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from acid_base_HIST import read_xyz_symbols, read_box_lengths_from_dftb_inp, iter_xyz_frames, _minimum_image, detect_diffusive_start
from prn_acid_base_analysis import mulliken_frames, bias_samples


def diagnostic(out: Path, source: Path, run: int, reason: str) -> None:
    """Show full records without inventing a diffusion-conditioned pKa window."""
    import matplotlib.pyplot as plt
    from prn_acid_base_analysis import aligned_fes
    data = np.genfromtxt(out/'wire_input.csv', delimiter=',', names=True)
    times, blocks = aligned_fes(source/'fes.dat')
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(data['time_ps'], data['coordination_s'])
    axes[0].set(xlabel='Raw DFTB time (ps)', ylabel='N1–H8 coordination')
    axes[1].scatter(data['time_ps'], data['N1_Odefect_distance_A'], s=5)
    axes[1].axhline(4, color='gray', linestyle='--')
    axes[1].set(xlabel='Raw DFTB time (ps)', ylabel='N1–defect distance (Å)')
    for ax in axes[:2]:
        ax.set_xlim(data['time_ps'][0], data['time_ps'][-1])
    last = blocks[-1].copy(); last[:,1] -= last[:,1].min()
    axes[2].plot(last[:,0],last[:,1])
    axes[2].set(xlabel='Coordination s',ylabel='F(s) (kcal/mol)',title=f'Final saved FES: {times[-1]:.3f} ps')
    np.savetxt(out/'diagnostic_fes.csv',last,delimiter=',',header='coordination_s,F_kcal_mol',comments='')
    fig.suptitle(f'HPD run {run}: diagnostic only — no assigned diffusion window\n{reason}',fontsize=10)
    fig.savefig(out/'diagnostic.png',dpi=160);plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=int, default=[1,2,4,7,11,13])
    p.add_argument('--out-dir', type=Path, default=Path('reports/HPD/meta-h_selected'))
    p.add_argument('--charge-min', type=float, default=-0.625, help='Solvent defect O Mulliken lower bound (e)')
    p.add_argument('--charge-max', type=float, default=-0.525, help='Solvent defect O Mulliken upper bound (e)')
    p.add_argument('--stride', type=int, default=4, help='Trajectory sampling stride; default 0.02 ps')
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for run in args.runs:
        source = Path(f'systems/HPD/solv_5.5/dftb/N1T48C1/run-{run}/meta-h').resolve()
        out = args.out_dir/f'run-{run}'
        out.mkdir(exist_ok=False)
        symbols = read_xyz_symbols(source/'traject')
        box = read_box_lengths_from_dftb_inp(source/'dftb.inp')
        mt, q = mulliken_frames(source/'mulliken', symbols)
        bt, bs, _ = bias_samples(source/'biaspot')
        if np.any(np.diff(mt)<=0) or np.any(np.diff(bt)<=0):
            raise ValueError('Nonmonotonic source timestamps')
        lookup = {round(t,8):i for i,t in enumerate(mt)}
        oxygen = np.array([i for i,s in enumerate(symbols) if i>=12 and s=='O'])
        rows = []
        for frame,(time,xyz) in enumerate(iter_xyz_frames(source/'traject')):
            if frame % args.stride:
                continue
            if time is None:
                raise ValueError('Missing trajectory timestamp')
            mi = lookup.get(round(time,8))
            defect = distance = float('nan')
            if mi is not None:
                charges = q[mi,oxygen]
                valid = (charges>=args.charge_min)&(charges<=args.charge_max)
                if valid.any():
                    atom = oxygen[np.argmax(np.where(valid,charges,-np.inf))]
                    defect = int(atom+1)
                    distance = float(np.linalg.norm(_minimum_image(xyz[atom]-xyz[0],box)))
            coord = float(np.interp(time,bt,bs,left=np.nan,right=np.nan))
            rows.append([time,coord,defect,distance])
        values = np.asarray(rows)
        np.savetxt(out/'wire_input.csv',values,delimiter=',',comments='',
                   header='time_ps,coordination_s,defect_oxygen_id,N1_Odefect_distance_A')
        record = dict(run_id=run,source=str(source),temperature_K=300,nitrogen_id=1,labeled_hydrogen_id=8,
                      solute_atoms=12,charge_window_e=[args.charge_min,args.charge_max],stride=args.stride,
                      clock='Raw DFTB ps, not elapsed: trajectory, Mulliken, bias and FES share this clock',
                      time_start_ps=float(values[0,0]),time_end_ps=float(values[-1,0]))
        try:
            onset = detect_diffusive_start(values[:,0],values[:,1],values[:,3])
            record['t_diffuse_ps'] = onset
            command = [sys.executable,'plotting/acid_base_HIST.py','--input-csv',str(out/'wire_input.csv'),
                       '--traj',str(source/'traject'),'--nitrogen-id','1','--solute-atoms','12',
                       '--dftb-inp',str(source/'dftb.inp'),'--mulliken',str(source/'mulliken'),
                       '--fes',str(source/'fes.dat'),'--temp','300','--split-regime-cutoff','4',
                       '--diffusive-start',str(onset),'--out',str(out/'summary.csv')]
            record['command'] = command
            with (out/'analysis.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            record['status'] = 'summary generated; automatic diffusion onset'
        except (ValueError, subprocess.CalledProcessError) as exc:
            record['status'] = str(exc)
            diagnostic(out, source, run, record['status'])
        (out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
        index.append(dict(run_id=run,status=record['status'],t_diffuse_ps=record.get('t_diffuse_ps',''),
                          summary=f'run-{run}/summary.png' if (out/'summary.png').exists() else f'run-{run}/diagnostic.png'))
        print(json.dumps(record),flush=True)
    with (args.out_dir/'index.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(index[0]));writer.writeheader();writer.writerows(index)
    (args.out_dir/'README.md').write_text('# HPD meta-h selected summaries\n\n'
        'Runs 1, 2, 4, 7, 11, 13. See index.csv for status and links.\n'
        'N1–H8; 12 solute atoms; 300 K. Raw DFTB timestamps (not elapsed). '
        'Mulliken defects are solvent oxygens in the recorded charge window, choosing the most positive candidate. '
        'Trajectory/Mulliken matched by timestamp; coordination interpolated between bias samples only.\n'
        'Automatic diffusion: sustained s≤0.05 and N–defect≥4 Å, then sustained labeled coordination recovery s≥0.20; persistence 0.05 ps. '
        'Existing detector interpolates missing defect distances for state detection. No manual onset overrides.\n'
        'Hydrogen bonds: donor–H≤1.3 Å, H–acceptor≤2.5 Å, angle≥135°, up to four bridging waters. '
        'Post-diffusion analysis window 1.75 ps. Existing HIST-derived generic acid–base panels; no histidine-specific topology supplied. '
        'Reported pKa uses the existing F(s) minima estimator, not integrated state populations; automatic events require visual review. '
        'Original simulation files are read-only.\n')


if __name__ == '__main__':
    main()
