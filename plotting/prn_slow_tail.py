"""Slow-tail single-exponential fits, excluding the initial librational transient."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares


def fit_tail(t: np.ndarray, y: np.ndarray, start: float, end: float) -> dict:
    mask=(t>=start-1e-8)&(t<=end+1e-8)&np.isfinite(y)
    x,v=t[mask],y[mask]
    if len(x)<4: raise ValueError('Insufficient tail points')
    positive=v>0
    if positive.sum()<3: raise ValueError('Insufficient positive log-tail points')
    log_slope,log_intercept=np.polyfit(x[positive],np.log(v[positive]),1)
    tau_init=-1/log_slope if log_slope<0 else end-start
    b_init=float(np.exp(np.clip(log_intercept,-20,20)))
    fit=least_squares(lambda p:np.exp(p[0]-x/np.exp(p[1]))-v,
        np.log([b_init,tau_init]),bounds=([-30,-20],[30,20]),ftol=1e-12,xtol=1e-12,gtol=1e-12)
    if not fit.success: raise ValueError('Tail fit did not converge')
    b,tau=map(float,np.exp(fit.x))
    residual=v-b*np.exp(-x/tau)
    r2=1-float(np.sum(residual**2)/np.sum((v-v.mean())**2))
    logy=np.log(v[positive]); logpred=log_intercept+log_slope*x[positive]
    return dict(start_ps=start,end_ps=end,B=b,tau_slow_ps=tau,points=len(x),
        r_squared=r2,rmse=float(np.sqrt(np.mean(residual**2))),
        log_points=int(positive.sum()),log_linear_slope_per_ps=float(log_slope),
        log_linear_intercept=float(log_intercept),
        log_linear_r_squared=1-float(np.sum((logy-logpred)**2)/np.sum((logy-logy.mean())**2)),
        log_linear_rmse=float(np.sqrt(np.mean((logy-logpred)**2))))


def add_tail_plots(lag: np.ndarray, values: np.ndarray, main_ax, inset, log_ax, stem: Path) -> None:
    windows=[(.15,1.),(.15,1.5),(.15,2.)]
    fits=[fit_tail(lag,values,start,end) for start,end in windows]
    colors=('#D55E00','#009E73','#CC79A7','#6A3D9A')
    logvalues=np.full(values.shape,np.nan)
    np.log(values,out=logvalues,where=values>0)
    tail=(lag>=.15-1e-8)&(lag<=2+1e-8)
    log_ax.plot(lag[tail],logvalues[tail],color='steelblue',lw=1.4,label='ln C̃(t), positive tail')
    columns=[lag,values,logvalues]
    headers=['lag_ps','normalized_C','ln_normalized_C']
    for f,color in zip(fits,colors):
        fit=f['B']*np.exp(-lag/f['tau_slow_ps'])
        selected=(lag>=f['start_ps']-1e-8)&(lag<=f['end_ps']+1e-8)
        label=f'{f["start_ps"]:g}–{f["end_ps"]:g} ps: τ={f["tau_slow_ps"]:.3f} ps'
        main_ax.plot(lag[selected],fit[selected],ls='--',lw=1.5,color=color,label=label)
        inset.plot(lag[selected],fit[selected],ls='--',lw=1,color=color)
        log_ax.plot(lag[selected],np.log(f['B'])-lag[selected]/f['tau_slow_ps'],ls='--',lw=1.4,color=color,label=f'Fit {f["start_ps"]:g}–{f["end_ps"]:g} ps')
        columns.extend([fit,selected.astype(int)])
        headers.extend([f'fit_to_{f["end_ps"]:g}ps',f'in_window_{f["end_ps"]:g}ps'])
    main_ax.axvspan(0,.1,color='.85',alpha=.5)
    main_ax.axvline(.15,color='.4',lw=.8,ls=':')
    main_ax.legend(fontsize=7,loc='lower right')
    log_ax.set(xlabel='Lag t (ps)',ylabel='ln C̃(t)',xlim=(.15,2),title='Positive slow tail: semilog diagnostic')
    log_ax.grid(alpha=.2);log_ax.legend(fontsize=7,loc='upper right')
    text='Log-linear R²: '+', '.join(f'{f["end_ps"]:g} ps: {f["log_linear_r_squared"]:.3f}' for f in fits)
    log_ax.text(.02,.04,text,transform=log_ax.transAxes,fontsize=8)
    with stem.with_name(stem.name+'_acf_slow_tail_fits.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(fits[0]));writer.writeheader();writer.writerows(fits)
    np.savetxt(stem.with_name(stem.name+'_acf_slow_tail_curves.csv'),np.column_stack(columns),delimiter=',',header=','.join(headers),comments='')
    stem.with_name(stem.name+'_acf_slow_tail.json').write_text(json.dumps(dict(
        model='B*exp(-t/tau_slow); B is extrapolated amplitude at t=0, not at the window start',
        method='Unweighted nonlinear least squares on normalized correlation; B,tau positive. Separate unweighted linear regression of log positive tail for linearity diagnostics.',
        plateau='Fixed histogram exp(-sigma_rad^2)',fits=fits,
        limitations='Correlated lag points; R2/RMSE are descriptive. Window sensitivity and semilog curvature must be checked. No independent-sample confidence intervals.'),indent=2)+'\n')
