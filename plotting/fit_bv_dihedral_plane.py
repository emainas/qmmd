"""Fit saved metadynamics DF to single15 and single5 density-peak coordinates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f as f_distribution


def fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[dict, np.ndarray]:
    x,y,z = (np.asarray(v,dtype=float) for v in (x,y,z))
    if x.ndim != 1 or x.shape != y.shape or x.shape != z.shape or len(x)<=3:
        raise ValueError('Need matching vectors with more than three observations')
    if not np.isfinite(np.column_stack([x,y,z])).all():
        raise ValueError('Nonfinite observations')
    design=np.column_stack([x,y,np.ones(len(x))])
    coef,_,rank,_=np.linalg.lstsq(design,z,rcond=None)
    if rank != 3: raise ValueError('Predictor coordinates do not identify a unique plane')
    predicted=design@coef
    sse=float(np.sum((z-predicted)**2));sst=float(np.sum((z-z.mean())**2))
    if sst==0: raise ValueError('Constant DF: R² undefined')
    df=len(x)-3; r2=1-sse/sst
    fstat=((sst-sse)/2)/(sse/df) if sse>0 else float('inf')
    stats=dict(single15_coefficient=float(coef[0]),single5_coefficient=float(coef[1]),intercept=float(coef[2]),
               n=len(x),rank=int(rank),residual_df=df,r_squared=r2,
               adjusted_r_squared=1-(1-r2)*(len(x)-1)/df,rmse_kcal_mol=float(np.sqrt(sse/len(x))),
               sse=sse,overall_f=fstat,overall_f_p=float(f_distribution.sf(fstat,2,df)),
               unique_predictor_pairs=int(len(np.unique(design[:,:2],axis=0))))
    return stats,predicted


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir',type=Path,default=Path('reports/BV/meta-hic-finalist-runs'))
    p.add_argument('--exclude-run-ids',nargs='*',type=int,default=[190])
    args=p.parse_args();root=args.report_dir
    source=root/'metad_basin_vs_meta_dg.csv'
    with source.open() as handle: rows=[r for r in csv.DictReader(handle) if int(r['run_id']) not in args.exclude_run_ids]
    ids=[int(r['run_id']) for r in rows]
    if len(set(ids))!=len(ids): raise ValueError('Require one selected basin per run')
    x=np.array([float(r['single15_mode_deg']) for r in rows])
    y=np.array([float(r['single5_mode_deg']) for r in rows])
    z=np.array([float(r['delta_G_kcal_mol']) for r in rows])
    if np.ptp(x)>=180 or np.ptp(y)>=180: raise ValueError('Angular branch choice needed before linear fit')
    stats,pred=fit_plane(x,y,z)
    gridx,gridy=np.meshgrid(np.linspace(x.min(),x.max(),35),np.linspace(y.min(),y.max(),35))
    gridz=stats['single15_coefficient']*gridx+stats['single5_coefficient']*gridy+stats['intercept']
    fig=plt.figure(figsize=(14,8))
    ax=fig.add_axes([.015,.29,.53,.60],projection='3d')
    ax.plot_surface(gridx,gridy,gridz,color='#deab78',alpha=.35,edgecolor='none')
    ax.scatter(x,y,z,c=z,cmap='viridis',s=95,edgecolor='black',depthshade=False)
    for i,run in enumerate(ids):
        ax.plot([x[i],x[i]],[y[i],y[i]],[pred[i],z[i]],color='#555555',linestyle='--',linewidth=1.7)
        ax.scatter([x[i]],[y[i]],[pred[i]],marker='x',color='#333333',s=35)
        ax.text(x[i]-2,y[i],z[i]+.12,f'Run {run} ',fontsize=10,ha='right')
    ax.set_xlabel('single15 peak (degrees)',labelpad=10)
    ax.set_ylabel('single5 peak (degrees)',labelpad=10)
    ax.set_zlabel('ΔF (kcal/mol)',labelpad=7)
    ax.view_init(elev=23,azim=-55)
    ax.set_title('Observed points, fitted plane and vertical residuals',fontsize=11,pad=5)
    ax2=fig.add_axes([.64,.34,.31,.49])
    lo=min(z.min(),pred.min())-.4; hi=max(z.max(),pred.max())+.4
    ax2.plot([lo,hi],[lo,hi],color='#555555',linestyle='--',label='Perfect agreement')
    ax2.scatter(z,pred,c=z,cmap='viridis',s=85,edgecolor='black')
    for i,run in enumerate(ids):
        ax2.annotate(f'Run {run}',(z[i],pred[i]),xytext=(5,9 if run!=142 else -15),textcoords='offset points',fontsize=9)
    ax2.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Observed ΔF (kcal/mol)',ylabel='Fitted ΔF (kcal/mol)')
    ax2.set_aspect('equal');ax2.grid(alpha=.2);ax2.legend(fontsize=8,loc='lower right')
    ax2.set_title('In-sample fitted versus observed',fontsize=11)
    fig.suptitle('BV HIC · metadynamics ΔF versus single15 and single5',fontsize=17,y=.97)
    fig.text(.5,.917,'Before purple dashed summary marker · run 190 excluded · density-peak coordinates',ha='center',fontsize=11)
    text=(f"ΔF = {stats['single15_coefficient']:.5f} × single15 + {stats['single5_coefficient']:.5f} × single5 + {stats['intercept']:.5f}\n"
          f"R² = {stats['r_squared']:.3f}   |   adjusted R² = {stats['adjusted_r_squared']:.3f}   |   RMSE = {stats['rmse_kcal_mol']:.3f} kcal/mol\n"
          f"n = {stats['n']} runs; 3 coefficients; residual df = {stats['residual_df']}   |   overall F-test p = {stats['overall_f_p']:.3g}\n"
          'Residuals (observed − fitted, kcal/mol): '+ ';  '.join(f'{run}: {v:+.3f}' for run,v in zip(ids,z-pred)))
    fig.text(.08,.12,text,fontsize=10,linespacing=1.65)
    fig.text(.08,.035,'Only three unique predictor pairs: runs 57 and 99 share the same grid-peak coordinates.\n'
             'Descriptive OLS; no validation of planarity or predictive power. Assumes independent equal-variance errors; ignores angle uncertainty.',fontsize=9)
    stem=root/'metad_single15_single5_df_plane'
    fig.savefig(stem.with_suffix('.png'),dpi=200);plt.close(fig)
    with stem.with_suffix('.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['run_id','single15_mode_deg','single5_mode_deg','observed_DF_kcal_mol','fitted_DF_kcal_mol','residual_kcal_mol'])
        writer.writerows(zip(ids,x,y,z,pred,z-pred))
    np.savetxt(stem.with_name(stem.name+'_surface.csv'),np.column_stack([gridx.ravel(),gridy.ravel(),gridz.ravel()]),
               delimiter=',',header='single15_deg,single5_deg,fitted_DF_kcal_mol',comments='')
    stats.update(included_run_ids=ids,excluded_run_ids=args.exclude_run_ids,source=str(source.resolve()),
                 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 window='Metadynamics start strictly before purple dashed summary marker',
                 model='DF = a*single15_mode_deg + b*single5_mode_deg + c; ordinary unweighted least squares',
                 limitation='Four selected runs, three unique predictor pairs, one residual df. In-sample fit is not evidence of planarity; no independent lack-of-fit test or reliable generalization assessment.')
    stem.with_suffix('.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2))


if __name__=='__main__': main()
