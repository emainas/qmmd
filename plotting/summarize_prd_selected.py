"""Add PRD summaries for RUN:GRID_TDIFF pairs without replacing existing runs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from acid_base_PRD import read_xyz_symbols, read_box_lengths_from_dftb_inp, iter_xyz_frames, _minimum_image
from prn_acid_base_analysis import mulliken_frames, bias_samples, aligned_fes
from plot_cv_grid import infer_offset_ps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pairs', nargs='+', help='Run ID:grid-clock tdiff in ps')
    parser.add_argument('--runs-path', type=Path, default=Path('systems/PRD/solv_5.5/dftb/N1T48C1'))
    parser.add_argument('--out-dir', type=Path, default=Path('reports/PRD/meta-h_selected'))
    parser.add_argument('--window-ps', type=float, default=1.75)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--charge-min', type=float, default=-0.625, help='Defect O Mulliken lower bound, e')
    parser.add_argument('--charge-max', type=float, default=-0.525, help='Defect O Mulliken upper bound, e')
    args = parser.parse_args()
    pairs = [(int(p.split(':')[0]), float(p.split(':')[1])) for p in args.pairs]
    if args.stride < 1 or args.samples < 2 or args.window_ps <= 0 or args.charge_min > args.charge_max:
        parser.error('Invalid sampling/window/charge settings')
    for run, grid in pairs:
        if (args.out_dir/f'run-{run}').exists():
            parser.error(f'Output run-{run} already exists; refusing to overwrite')
        if not np.isfinite(grid):
            parser.error('Nonfinite marker')
    for run, grid in pairs:
        source = (args.runs_path/f'run-{run}'/'meta-h').resolve()
        out = args.out_dir/f'run-{run}'
        symbols = read_xyz_symbols(source/'traject')
        if symbols[:12] != ['N', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']:
            raise ValueError('Unexpected PRD atom ordering')
        box = read_box_lengths_from_dftb_inp(source/'dftb.inp')
        bt, bs, _ = bias_samples(source/'biaspot')
        offset = infer_offset_ps(source.parent, source.name, float(bt[0]))
        onset = grid-offset
        stop = onset+args.window_ps
        ft, _ = aligned_fes(source/'fes.dat')
        if ft[-1] < stop:
            raise ValueError(f'Run {run}: insufficient FES coverage for requested stop {stop}')
        mt, q = mulliken_frames(source/'mulliken', symbols)
        if np.any(np.diff(mt)<=0) or np.any(np.diff(bt)<=0):
            raise ValueError('Nonmonotonic input timestamps; explicit restart stitching needed')
        lookup = {round(float(t),8):i for i,t in enumerate(mt)}
        oxygen = np.array([i for i,s in enumerate(symbols) if i>=12 and s=='O'])
        rows = []
        for frame, (time, xyz) in enumerate(iter_xyz_frames(source/'traject')):
            if time is None:
                raise ValueError('Missing trajectory timestamp')
            if time > stop+1e-8:
                break
            if frame % args.stride:
                continue
            mi = lookup.get(round(time,8))
            defect = distance = float('nan')
            if mi is not None:
                charges = q[mi,oxygen]
                valid = (charges>=args.charge_min)&(charges<=args.charge_max)
                if valid.any():
                    atom = oxygen[np.argmax(np.where(valid,charges,-np.inf))]
                    defect = int(atom+1)
                    distance = float(np.linalg.norm(_minimum_image(xyz[atom]-xyz[0],box)))
            rows.append([time, float(np.interp(time,bt,bs,left=np.nan,right=np.nan)), defect, distance])
        values = np.asarray(rows)
        if len(values)<2 or np.any(np.diff(values[:,0])<=0) or stop-values[-1,0]>np.median(np.diff(values[:,0]))+1e-8:
            raise ValueError('Insufficient/nonmonotonic trajectory coverage')
        out.mkdir(parents=True, exist_ok=False)
        np.savetxt(out/'wire_input.csv',values,delimiter=',',comments='',header='time_ps,coordination_s,defect_oxygen_id,N1_Odefect_distance_A')
        command = [sys.executable,'plotting/acid_base_PRD.py','--input-csv',str(out/'wire_input.csv'),
                   '--traj',str(source/'traject'),'--nitrogen-id','1','--solute-atoms','12',
                   '--dftb-inp',str(source/'dftb.inp'),'--mulliken',str(source/'mulliken'),
                   '--fes',str(source/'fes.dat'),'--temp','300','--split-regime-cutoff','4',
                   '--diffusive-start',str(onset),'--exp-pka','5.23',
                   '--probability-window-ps',str(args.window_ps),'--pka-window-samples',str(args.samples),
                   '--out',str(out/'summary.csv')]
        record = dict(command=command,source=str(source),grid_tdiff_ps=grid,raw_tdiff_ps=onset,
                      offset_ps=offset,raw_stop_ps=stop,grid_stop_ps=grid+args.window_ps,
                      clock='Summary uses raw DFTB ps; add offset_ps for CV grid clock',
                      defect_charge_window_e=[args.charge_min,args.charge_max],competitor_atom_ids=list(range(1,7)),
                      experimental_pka=5.23,temperature_K=300,probability_window_ps=args.window_ps,
                      pka_window_samples=args.samples,stride=args.stride,
                      sampling='Trajectory stride; exact Mulliken timestamp join; bias interpolation; missing defects remain NaN')
        (out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
        with (out/'analysis.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'run-{run}: grid {grid:g} to {grid+args.window_ps:g} ps, raw {onset:g} to {stop:g} ps; {out / "summary.png"}',flush=True)


if __name__ == '__main__':
    main()
