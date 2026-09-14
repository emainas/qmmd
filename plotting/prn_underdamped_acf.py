"""Fixed-histogram-variance underdamped fits to mean cosine ACF."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks


def model(t: np.ndarray, gamma: float, omega_d: float, sigma: float) -> np.ndarray:
    # sinc form is stable as omega_d approaches zero.
    relaxation=np.exp(-gamma*t/2)*(np.cos(omega_d*t)+(gamma*t/2)*np.sinc(omega_d*t/np.pi))
    return np.exp(-sigma*sigma*(1-relaxation))


def fit_windows(lag: np.ndarray, mean: np.ndarray, sigma: float, stem: Path) -> tuple[list[dict],np.ndarray]:
    dt=float(np.median(np.diff(lag)))
    early=lag<=.2+1e-8
    peaks,_=find_peaks(mean[early])
    peak_times=lag[early][peaks]
    if len(peak_times)<2: raise ValueError('Fewer than two early peaks for frequency initialization')
    period=float(np.median(np.diff(peak_times)))
    omega_init=2*np.pi/period
    records,curves=[],[]
    for end in (.2,.3,.5,1.):
        keep=(lag>=0)&(lag<=end+1e-8)&np.isfinite(mean)
        t,y=lag[keep],mean[keep]
        bounds=([1e-6,1e-6],[100/dt,np.pi/dt])
        solutions=[]
        for damping in (2/end,20.,80.,200.):
            for frequency in (omega_init*.7,omega_init,omega_init*1.3):
                initial=np.clip([damping,frequency],np.array(bounds[0])*2,np.array(bounds[1])*.99)
                result=least_squares(lambda p:model(t,p[0],p[1],sigma)-y,initial,bounds=bounds,
                                     x_scale='jac',max_nfev=5000,ftol=1e-12,xtol=1e-12,gtol=1e-12)
                if result.success: solutions.append(result)
        if not solutions: raise ValueError('No converged underdamped fit')
        best=min(solutions,key=lambda s:float(np.dot(s.fun,s.fun)))
        gamma,omega=map(float,best.x)
        prediction=model(lag,gamma,omega,sigma)
        sse=float(np.sum(best.fun**2)); sst=float(np.sum((y-y.mean())**2))
        records.append(dict(fit_end_ps=end,points=len(t),sigma_rad=sigma,gamma_per_ps=gamma,
                            omega_d_rad_per_ps=omega,tau_damp_ps=2/gamma,T_osc_ps=2*np.pi/omega,
                            omega_0_rad_per_ps=float(np.sqrt(omega**2+gamma**2/4)),
                            r_squared=1-sse/sst,rmse=float(np.sqrt(sse/len(t))),
                            omega_initial_rad_per_ps=omega_init,peak_spacing_ps=period,
                            near_parameter_bound=bool(np.any(best.x<np.array(bounds[0])*10) or np.any(best.x>np.array(bounds[1])*.999)),
                            jacobian_condition=float(np.linalg.cond(best.jac))))
        curves.append(prediction)
    import csv
    with stem.with_name(stem.name+'_acf_underdamped_fits.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    np.savetxt(stem.with_name(stem.name+'_acf_underdamped_curves.csv'),np.column_stack([lag,mean,*curves]),delimiter=',',header='lag_ps,mean_C,fit_0_0p2ps,fit_0_0p3ps,fit_0_0p5ps,fit_0_1ps',comments='')
    stem.with_name(stem.name+'_acf_underdamped.json').write_text(json.dumps(dict(
        model='exp(-sigma^2*(1-exp(-gamma*t/2)*(cos(omega_d*t)+gamma/(2*omega_d)*sin(omega_d*t))))',
        sigma_source='Pooled anti histogram variance, radians; fixed, not fitted',peak_times_ps=peak_times.tolist(),
        bounds=dict(gamma_per_ps=[1e-6,100/dt],omega_d_rad_per_ps=[1e-6,np.pi/dt]),
        method='Unweighted least squares on ensemble mean C(t); 12 starts per window seeded by observed peak spacing. No offset or amplitude fitted.',
        limitations='Lag points correlated; window sensitivity is a robustness diagnostic, not a confidence interval. Frequency bounded below Nyquist. A bound-hitting or poor fit does not establish underdamped harmonic behavior.',fits=records),indent=2)+'\n')
    return records,np.array(curves)
