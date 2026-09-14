"""Within-trajectory cosine autocorrelation and equal-run ensemble statistics."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def fit_biexponential(lag: np.ndarray, values: np.ndarray, end_ps: float = 1.) -> tuple[np.ndarray,dict]:
    """Fit normalized correlation; enforce 0<=A<=1 and tslow>tfast>0."""
    from scipy.optimize import least_squares
    selected=np.isfinite(values)&(lag>=0)&(lag<=end_ps+1e-8)
    t,y=lag[selected],values[selected]
    if len(t)<4: raise ValueError('Insufficient biexponential fit points')
    dt=float(np.min(t[t>0]))
    lower,upper=dt*1e-4,end_ps*1e4
    def predict(x: np.ndarray,p: np.ndarray) -> np.ndarray:
        a,fast,gap=p[0],np.exp(p[1]),np.exp(p[2])
        return a*np.exp(-x/fast)+(1-a)*np.exp(-x/(fast+gap))
    solutions=[]
    for a in (.3,.7,.9):
        for fast in (.02,.08):
            for slow in (.3,2.,10.):
                p=[a,np.log(fast),np.log(slow-fast)]
                fit=least_squares(lambda p:predict(t,p)-y,p,
                    bounds=([0,np.log(lower),np.log(lower)],[1,np.log(upper),np.log(upper)]),
                    max_nfev=5000,ftol=1e-12,xtol=1e-12,gtol=1e-12)
                if fit.success: solutions.append(fit)
    if not solutions: raise ValueError('Biexponential fit failed')
    best=min(solutions,key=lambda f:np.sum(f.fun**2))
    a=float(best.x[0]); fast=float(np.exp(best.x[1])); slow=fast+float(np.exp(best.x[2]))
    sse=float(np.sum(best.fun**2)); sst=float(np.sum((y-y.mean())**2))
    info=dict(A=a,tfast_ps=fast,tslow_ps=slow,fit_end_ps=end_ps,fit_points=len(t),
              r_squared=1-sse/sst,rmse=float(np.sqrt(sse/len(t))),
              definition='A*exp(-t/tfast)+(1-A)*exp(-t/tslow); 0<=A<=1; 0<tfast<tslow; unweighted least squares on normalized mean correlation',
              jacobian_condition=float(np.linalg.cond(best.jac)),
              limitations='Correlated lag points; descriptive fit. Smooth biexponential cannot reproduce early oscillations. Beyond the fit window the line is an extrapolation.')
    return predict(lag,best.x),info


def fit_exponential_acf(lag: np.ndarray, correlation: np.ndarray, plateau: float,
                        exclude_from_ps: float = 35., fit_plateau: bool = False) -> tuple[np.ndarray,np.ndarray,dict]:
    """Fit the original correlation, including points below the fixed plateau."""
    from scipy.optimize import minimize_scalar, least_squares
    selected = np.isfinite(correlation)&(lag>=0)&(lag<exclude_from_ps)&~np.isclose(lag,exclude_from_ps,rtol=0,atol=1e-8)
    t,y = lag[selected],correlation[selected]
    if len(t)<3 or not np.any(t>0) or not 0<=plateau<1:
        raise ValueError('Insufficient fit samples or invalid plateau')
    lower = float(np.min(t[t>0]))*1e-6
    upper = float(np.max(t))*1e6
    def objective(log_tau: float) -> float:
        prediction = plateau+(1-plateau)*np.exp(-t/np.exp(log_tau))
        return float(np.sum((y-prediction)**2))
    result = minimize_scalar(objective,bounds=(np.log(lower),np.log(upper)),method='bounded',options={'xatol':1e-12})
    if not result.success: raise ValueError('Exponential fit failed')
    tau = float(np.exp(result.x))
    histogram_plateau = plateau
    if fit_plateau:
        def residuals(parameters: np.ndarray) -> np.ndarray:
            cinf,log_tau = parameters
            return cinf+(1-cinf)*np.exp(-t/np.exp(log_tau))-y
        solutions = [least_squares(residuals,[plateau,np.log(initial)],
                     bounds=([0,np.log(lower)],[1-1e-12,np.log(upper)]),
                     ftol=1e-12,xtol=1e-12,gtol=1e-12)
                     for initial in (tau,float(np.min(t[t>0])),float(np.max(t))/2)]
        solution = min(solutions,key=lambda s: float(np.dot(s.fun,s.fun)))
        if not solution.success: raise ValueError('Free-plateau exponential fit failed')
        plateau,tau = float(solution.x[0]),float(np.exp(solution.x[1]))
    fitted = plateau+(1-plateau)*np.exp(-lag/tau)
    residual = y-fitted[selected]
    sse = float(np.dot(residual,residual))
    sst = float(np.sum((y-y.mean())**2))
    info = dict(tau_ps=tau,plateau=plateau,fit_plateau=fit_plateau,histogram_plateau=histogram_plateau,
                plateau_difference=plateau-histogram_plateau,
                plateau_relative_difference_percent=100*(plateau-histogram_plateau)/histogram_plateau if histogram_plateau else None,
                r_squared=1-sse/sst if sst>0 else None,
                rmse=float(np.sqrt(sse/len(t))),fit_points=len(t),exclude_lags_from_ps=exclude_from_ps,
                definition='Cinf+(1-Cinf)*exp(-t/tau); C(0)=1; unweighted least squares in C(t), not log space; Cinf '+('fitted in [0,1)' if fit_plateau else 'fixed'),
                limitations='Correlated lags; descriptive fit, no independent-sample uncertainty. All finite values below cutoff included, even C<Cinf.',
                tau_search_bounds_ps=[lower,upper])
    return fitted,selected,info


def fit_log_decay(lag: np.ndarray, correlation: np.ndarray, plateau: float,
                  exclude_from_ps: float = 35.) -> tuple[np.ndarray,np.ndarray,dict]:
    """Unweighted zero-intercept least squares of log normalized excess ACF."""
    ratio = (correlation-plateau)/(1-plateau)
    log_decay = np.full(ratio.shape,np.nan)
    np.log(ratio,out=log_decay,where=ratio>0)
    before_cutoff = (lag<exclude_from_ps)&~np.isclose(lag,exclude_from_ps,rtol=0,atol=1e-8)
    selected = np.isfinite(log_decay)&(lag>=0)&before_cutoff
    t,y = lag[selected],log_decay[selected]
    if len(t)<2 or np.dot(t,t)<=0:
        raise ValueError('Insufficient positive excess-correlation points for fit')
    slope = float(np.dot(t,y)/np.dot(t,t))
    residual = y-slope*t
    sse = float(np.dot(residual,residual))
    centered_sst = float(np.sum((y-y.mean())**2))
    info = dict(slope_per_ps=slope,tau_ps=-1/slope if slope<0 else None,
                r_squared_centered=1-sse/centered_sst if centered_sst>0 else None,
                rmse_log_units=float(np.sqrt(sse/len(t))),fit_points=len(t),
                excluded_nonpositive_before_cutoff=int(np.sum(before_cutoff&(ratio<=0))),
                exclude_lags_from_ps=exclude_from_ps,plateau=plateau,
                definition='log((mean_C-C_infinity)/(1-C_infinity)) = slope*t; intercept fixed zero; unweighted OLS; tau=-1/slope',
                limitations='Correlated lag points, fixed estimated plateau, and omission of nonpositive excess correlations make this descriptive, not a validated relaxation time. Centered R-squared may be negative for a zero-intercept fit.')
    return log_decay,selected,info


def cosine_acf(angles_deg: np.ndarray) -> np.ndarray:
    """All lags: sum cos(theta[n]-theta[n+m])/(N-m), without centering.

    Zero-padded FFTs of sine and cosine give linear, not circular, correlation.
    """
    angles = np.asarray(angles_deg,dtype=float)
    if angles.ndim != 1 or len(angles)<2 or not np.isfinite(angles).all():
        raise ValueError('At least two finite angles required')
    theta = np.deg2rad(angles)
    n = len(theta)
    size = 1 << (2*n-1).bit_length()
    spectrum = sum(np.abs(np.fft.rfft(v,n=size))**2 for v in (np.cos(theta),np.sin(theta)))
    sums = np.fft.irfft(spectrum,n=size)[:n]
    return sums/np.arange(n,0,-1)


def integrate_run_correlations(lag: np.ndarray, curves: np.ndarray, plateau: float) -> tuple[np.ndarray,np.ndarray]:
    """Normalize each run to a shared plateau and integrate along its lag axis."""
    if curves.ndim!=2 or curves.shape[1]!=len(lag) or np.any(np.diff(lag)<=0) or plateau>=1:
        raise ValueError('Invalid lag grid, curve matrix, or plateau')
    normalized=(curves-plateau)/(1-plateau)
    integrals=np.column_stack([np.zeros(len(curves)),
        np.cumsum(.5*(normalized[:,:-1]+normalized[:,1:])*np.diff(lag),axis=1)])
    return normalized,integrals


def ensemble_acf(metadata: dict, stem: Path) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict]:
    source = Path(metadata['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=metadata['source_sha256']:
        raise ValueError('Dihedral snapshot changed since the PMF was calculated; regenerate PMF first')
    data = np.genfromtxt(source,delimiter=',',names=True)
    curves, records, steps = [], [], []
    ids = metadata['included_run_ids']
    for run in ids:
        subset = data[(data['run_id']==run)&(data['time_ps']>=metadata['discard_before_ps'])]
        times = subset['time_ps']
        if len(times)<2 or np.any(np.diff(times)<=0):
            raise ValueError(f'Run {run}: missing/nonmonotonic timestamps')
        dt = float(np.median(np.diff(times)))
        if not np.allclose(np.diff(times),dt,rtol=0,atol=1e-7):
            raise ValueError(f'Run {run}: nonuniform sampling/gaps')
        steps.append(dt)
        acf = cosine_acf(subset['signed_dihedral_deg'])
        curves.append(acf)
        records.append(np.column_stack([np.full(len(acf),run),np.arange(len(acf)),np.arange(len(acf))*dt,np.arange(len(acf),0,-1),acf]))
    if len(ids)<2 or not np.allclose(steps,steps[0],rtol=0,atol=1e-7):
        raise ValueError('Need at least two runs with identical sampling interval')
    common = min(map(len,curves))
    values = np.stack([a[:common] for a in curves])
    mean, std = values.mean(axis=0), values.std(axis=0,ddof=1)
    lag = np.arange(common)*steps[0]
    np.savetxt(stem.with_name(stem.name+'_acf_runs.csv'),np.vstack(records),delimiter=',',header='run_id,lag_index,lag_ps,time_origin_count,C',comments='')
    np.savetxt(stem.with_name(stem.name+'_acf.csv'),np.column_stack([lag,mean,std,mean-std,mean+std,np.full(common,len(ids))]),delimiter=',',header='lag_ps,mean_C,std_C,mean_minus_std,mean_plus_std,n_runs',comments='')
    info = dict(definition='C(m*dt)=mean_n cos(theta[n]-theta[n+m]); radians inside cosine; all N-m within-run origins',
                method='Zero-padded linear FFT autocorrelation of sine plus cosine; no mean subtraction or plateau removal',
                included_run_ids=ids,source=str(source),source_sha256=metadata['source_sha256'],
                dt_ps=steps[0],common_max_lag_ps=float(lag[-1]),std_ddof=1,
                averaging='Equal-run mean and sample standard deviation; common lag range only so every point contains all runs',
                limitations='Long lags have fewer time origins. Band is between-run spread, not a confidence interval. Confined angular fluctuations need not decorrelate to zero.')
    return lag,mean,std,info
