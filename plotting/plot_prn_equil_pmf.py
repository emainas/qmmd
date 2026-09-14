"""Histogram-derived, separately normalized PRN equilibration dihedral PMFs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R_KCAL = 0.00198720425864083


def histogram_pmf(angles: np.ndarray, center: float, bin_deg: float,
                  temperature: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bins = int(round(360/bin_deg))
    if bins<1 or not np.isclose(bins*bin_deg,360) or temperature<=0:
        raise ValueError('Positive temperature and bin width dividing 360 required')
    wrapped=(np.asarray(angles)-center+180)%360+center-180
    counts,edges=np.histogram(wrapped[np.isfinite(wrapped)],bins=np.linspace(center-180,center+180,bins+1))
    if counts.sum()==0:
        raise ValueError('No finite angles')
    density=counts/(counts.sum()*bin_deg)
    free=np.full(len(counts),np.nan)
    occupied=counts>0
    free[occupied]=-R_KCAL*temperature*np.log(density[occupied]/density.max())
    return (edges[:-1]+edges[1:])/2,counts,density,free


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--temperature',type=float,default=300,help='Temperature (K)')
    p.add_argument('--bin-deg',type=float,default=5,help='Periodic histogram width (degrees)')
    p.add_argument('--discard-ps',type=float,default=0,help='Discard frames earlier than this equilibration time (ps)')
    p.add_argument('--report-root',type=Path,default=Path('reports'))
    p.add_argument('--states',nargs='+',choices=['syn','anti'],default=['syn','anti'])
    args=p.parse_args()
    if args.bin_deg<=0 or args.discard_ps<0:
        p.error('Positive bin width and nonnegative discard time required')
    for state,center in [('syn',0),('anti',180)]:
        if state not in args.states:
            continue
        root=args.report_root/f'PRN-{state}'
        source=root/f'prn_{state}_dih.csv'
        data=np.genfromtxt(source,delimiter=',',names=True)
        excluded=[2] if state=='anti' else []
        keep=(data['run_id']>=1)&(data['run_id']<=50)&(data['time_ps']>=args.discard_ps)
        keep &= ~np.isin(data['run_id'],excluded)
        selected=data[keep]
        expected=set(range(1,51))-set(excluded)
        if set(selected['run_id'])!=expected:
            raise ValueError('Missing requested runs in source snapshot')
        x,count,density,free=histogram_pmf(selected['signed_dihedral_deg'],center,args.bin_deg,args.temperature)
        stem=root/f'prn_{state}_equil_pmf'
        np.savetxt(stem.with_suffix('.csv'),np.column_stack([x,count,density,free]),delimiter=',',
                   header='dihedral_center_deg,frame_count,probability_density_per_degree,relative_PMF_kcal_mol',comments='')
        run_rows=[]
        for run in sorted(expected):
            rows=selected[selected['run_id']==run]
            run_rows.append(dict(run_id=run,frames=len(rows),first_time_ps=float(rows['time_ps'].min()),last_time_ps=float(rows['time_ps'].max())))
        with stem.with_name(stem.name+'_runs.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(run_rows[0]));writer.writeheader();writer.writerows(run_rows)
        color='steelblue' if state=='anti' else 'darkorange'
        fig,ax=plt.subplots(figsize=(8,5.5))
        ax.plot(x,free,'o-',color=color,lw=1.6,ms=3)
        ax.axvline(center,color='gray',ls=':',lw=1)
        ax.set(xlabel='O2–C3–O4–H11 dihedral (degrees)',ylabel='Relative PMF (kcal/mol)',
               xlim=(center-180,center+180),ylim=(-.15,max(1,float(np.nanmax(free))+.4)))
        ax.grid(alpha=.2)
        ax.set_title(f'PRN-{state}: equilibration dihedral PMF\n{len(expected)} runs · {len(selected):,} frames · {args.temperature:g} K · {args.bin_deg:g}° bins')
        fig.text(.5,.025,'Separate basin normalization: min F = 0; no syn–anti ΔF inferred.\n'
                 f'Empty bins omitted (unsampled), no pseudocounts or smoothing. Excluded runs: {excluded or "none"}.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.09,1,1));fig.savefig(stem.with_suffix('.png'),dpi=200);plt.close(fig)
        meta=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  temperature_K=args.temperature,bin_width_deg=args.bin_deg,discard_before_ps=args.discard_ps,
                  included_run_ids=sorted(expected),excluded_run_ids=excluded,frames=len(selected),
                  definition='-RT ln(count / maximum count); equal-width periodic histogram, minimum zero',
                  weighting='Pooled frames, so longer trajectories contribute proportionally more; no metadynamics frames',
                  limitations='Saved equilibration snapshot, not refreshed trajectories. No burn-in removal by default. '
                  'Conditional within-conformer profile, not converged global PMF; no relative basin weights or barrier inference. '
                  'Frames correlated; rare-bin tails uncertain; no statistical uncertainty estimated. Excluding flipped anti run 2 conditions the ensemble on run outcome.',
                  empty_bins='NaN free energy denotes unsampled bins; no finite pseudocount estimate')
        stem.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(f'{state}: {len(expected)} runs, {len(selected)} frames; saved {stem}.png',flush=True)


if __name__=='__main__':
    main()
