"""Fit local quadratic PRN PMFs and locate specified energy-level crossings."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

R_KCAL = 0.00198720425864083


def parabola_gaussian(x: np.ndarray, a: float, mu: float, temperature: float) -> tuple[np.ndarray,float]:
    """Normalized local Gaussian implied by F=a*(phi-mu)^2+constant."""
    if a<=0 or temperature<=0:
        raise ValueError('Positive curvature and temperature required')
    sigma=float(np.sqrt(R_KCAL*temperature/(2*a)))
    density=np.exp(-0.5*((np.asarray(x)-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    return density,sigma


def crossings(x: np.ndarray, f: np.ndarray, level: float) -> tuple[float,float]:
    """First crossings outward from the sampled minimum; never bridge NaN gaps."""
    minimum=int(np.nanargmin(f))
    result=[]
    for direction in (-1,1):
        value=float('nan')
        for i in range(minimum,0 if direction==-1 else len(x)-1,direction):
            j=i+direction
            if not np.isfinite(f[[i,j]]).all():
                break
            if f[i]<=level<=f[j] and f[j]>f[i]:
                value=float(x[i]+(level-f[i])*(x[j]-x[i])/(f[j]-f[i]))
                break
        result.append(value)
    return tuple(result)


def main() -> None:
    plt.style.use(Path(__file__).with_name('lefteris.mplstyle'))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-max',type=float,default=4,help='Maximum sampled PMF energy included in fit (kcal/mol)')
    parser.add_argument('--report-root',type=Path,default=Path('reports'))
    parser.add_argument('--states',nargs='+',choices=['syn','anti'],default=['syn','anti'])
    parser.add_argument('--autocorrelation',action='store_true',help='Add within-run cosine ACF mean and one sample-standard-deviation band')
    parser.add_argument('--acf-fit-exclude-from-ps',type=float,default=35.,help='Exclude this lag and all later lags from log-decay fit (ps)')
    parser.add_argument('--acf-fit-max-ps',type=float,help='Inclusive maximum lag for direct exponential fit (ps)')
    parser.add_argument('--acf-ymin',type=float,help='Lower y limit of the autocorrelation panel')
    parser.add_argument('--acf-xmax',type=float,help='Maximum displayed autocorrelation lag (ps); does not change the fit')
    parser.add_argument('--hide-acf-fit',action='store_true',help='Hide the red fitted curve while retaining the fit results')
    parser.add_argument('--underdamped',action='store_true',help='Fit the fixed-sigma underdamped model in four early-time windows')
    parser.add_argument('--acf-ratio-panel',action='store_true',help='Add (C(t)-histogram Cinf)/(1-histogram Cinf) panel, without fitting')
    parser.add_argument('--acf-fit-plateau',action='store_true',help='Fit Cinf as well as tau, maintaining C(0)=1')
    parser.add_argument('--integral-max-ps',type=float,default=17.5,help='Maximum cumulative-integral cutoff displayed and saved (ps)')
    parser.add_argument('--show-loglinear',action='store_true',help='Restore the optional log-linear tail diagnostic panel')
    args=parser.parse_args()
    if not np.isfinite(args.integral_max_ps) or args.integral_max_ps<=0:
        parser.error('Integral maximum must be positive and finite')
    if not np.isfinite(args.acf_fit_exclude_from_ps) or args.acf_fit_exclude_from_ps<=0:
        parser.error('ACF fit cutoff must be positive and finite')
    if args.acf_fit_max_ps is not None and (not np.isfinite(args.acf_fit_max_ps) or args.acf_fit_max_ps<=0):
        parser.error('ACF maximum fit lag must be positive and finite')
    for state,center in [('syn',0),('anti',180)]:
        if state not in args.states:
            continue
        stem=args.report_root/f'PRN-{state}'/f'prn_{state}_equil_pmf'
        data=np.genfromtxt(stem.with_suffix('.csv'),delimiter=',',names=True)
        x=data['dihedral_center_deg']; f=data['relative_PMF_kcal_mol']
        keep=np.isfinite(f)&(f<=args.fit_max)
        if keep.sum()<4:
            raise ValueError('Insufficient bins for quadratic fit')
        a,b,c=np.polyfit(x[keep]-center,f[keep],2)
        if a<=0:
            raise ValueError('Fit is not a confining parabola')
        vertex=center-b/(2*a); minimum=c-b*b/(4*a)
        metadata=json.loads(stem.with_suffix('.json').read_text())
        temperature=float(metadata['temperature_K'])
        density_grid=np.linspace(center-180,center+180,3601)
        gaussian,sigma=parabola_gaussian(density_grid,a,vertex,temperature)
        gaussian_bins,_=parabola_gaussian(x,a,vertex,temperature)
        np.savetxt(stem.with_name(stem.name+'_gaussian_curve.csv'),np.column_stack([density_grid,gaussian]),delimiter=',',header='dihedral_deg,gaussian_density_per_degree',comments='')
        np.savetxt(stem.with_name(stem.name+'_density.csv'),np.column_stack([x,data['probability_density_per_degree'],gaussian_bins,data['frame_count']]),delimiter=',',header='dihedral_deg,sampled_density_per_degree,gaussian_density_per_degree,frame_count',comments='')
        predicted=np.polyval([a,b,c],x[keep]-center)
        residual=f[keep]-predicted
        r2=1-np.sum(residual**2)/np.sum((f[keep]-f[keep].mean())**2)
        grid=np.linspace(center-100,center+100,1001)
        fitted=np.polyval([a,b,c],grid-center)
        np.savetxt(stem.with_name(stem.name+'_parabola_curve.csv'),np.column_stack([grid,fitted]),delimiter=',',header='dihedral_deg,fitted_PMF_kcal_mol',comments='')
        np.savetxt(stem.with_name(stem.name+'_parabola_fit_bins.csv'),np.column_stack([x[keep],f[keep],predicted,residual]),delimiter=',',header='dihedral_deg,sampled_PMF_kcal_mol,fitted_PMF_kcal_mol,residual_kcal_mol',comments='')
        records=[]
        sampled_density=data['probability_density_per_degree']
        reference_density=float(np.interp(center,x,sampled_density))
        for energy in (1,2,3,4):
            left,right=crossings(x,f,energy)
            width=np.sqrt((energy-minimum)/a) if energy>=minimum else float('nan')
            left_density=float(np.interp(left,x,sampled_density)) if np.isfinite(left) else float('nan')
            right_density=float(np.interp(right,x,sampled_density)) if np.isfinite(right) else float('nan')
            records.append(dict(energy_kcal_mol=energy,sampled_left_deg=left,sampled_right_deg=right,
                                parabola_left_deg=vertex-width,parabola_right_deg=vertex+width,
                                reference_angle_deg=center,reference_density_per_degree=reference_density,
                                left_density_per_degree=left_density,right_density_per_degree=right_density,
                                left_pair_reference_percent=100*reference_density/(reference_density+left_density),
                                left_pair_angle_percent=100*left_density/(reference_density+left_density),
                                right_pair_reference_percent=100*reference_density/(reference_density+right_density),
                                right_pair_angle_percent=100*right_density/(reference_density+right_density)))
        with stem.with_name(stem.name+'_energy_angles.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        compact=args.autocorrelation and args.acf_ratio_panel and not args.show_loglinear
        fig=plt.figure(figsize=(16,12) if compact else (18,8),dpi=220)
        if state in ('anti','syn'):
            layout=fig.add_gridspec(2,2,hspace=.4,wspace=.25,left=.08,right=.98,bottom=.10,top=.92) if compact else fig.add_gridspec(1,((5 if args.show_loglinear else 4) if args.acf_ratio_panel else 3) if args.autocorrelation else 2)
            density_ax=fig.add_subplot(layout[0,0]); ax=fig.add_subplot(layout[0,1])
            if args.autocorrelation:
                from prn_dihedral_acf import ensemble_acf
                lag,mean_acf,std_acf,acf_info=ensemble_acf(metadata,stem)
                # Moments of the pooled sampled density, locally unwrapped about
                # the conformer center; not the fitted Gaussian's width.
                probability=data['probability_density_per_degree']*float(metadata['bin_width_deg'])
                probability=probability/probability.sum()
                theta=np.deg2rad((x-center+180)%360-180)
                theta_mean=float(np.sum(probability*theta))
                theta_variance=float(np.sum(probability*(theta-theta_mean)**2))
                theta_std=float(np.sqrt(theta_variance))
                gaussian_plateau=float(np.exp(-theta_variance))
                if args.acf_ratio_panel:
                    denominator=1-gaussian_plateau
                    ratio=np.full(mean_acf.shape,np.nan)
                    defined=np.isfinite(mean_acf)&(denominator!=0)
                    np.divide(mean_acf-gaussian_plateau,denominator,out=ratio,where=defined)
                    from prn_dihedral_acf import integrate_run_correlations
                    run_table=np.genfromtxt(stem.with_name(stem.name+'_acf_runs.csv'),delimiter=',',names=True)
                    run_ids=acf_info['included_run_ids']
                    curves=np.stack([run_table['C'][run_table['run_id']==r][:len(lag)] for r in run_ids])
                    run_ratios,run_integrals=integrate_run_correlations(lag,curves,gaussian_plateau)
                    ratio_std=run_ratios.std(axis=0,ddof=1)
                    integral_std=run_integrals.std(axis=0,ddof=1)
                    if not np.allclose(run_ratios.mean(axis=0),ratio):
                        raise ValueError('Per-run normalized mean mismatch')
                    manual_override=np.zeros(lag.shape,dtype=bool)
                    ratio_ax=fig.add_subplot(layout[1,0] if compact else layout[0,2])
                    ratio_ax.plot(lag,ratio,color='steelblue',lw=1.5)
                    ratio_ax.fill_between(lag,ratio-ratio_std,ratio+ratio_std,color='steelblue',alpha=.2)
                    ratio_ax.axhline(0,color='.5',ls=':',lw=1)
                    ratio_ax.set(xlabel='Lag t (ps)',ylabel='(C(t) − C∞) / (1 − C∞)',
                                 xlim=(0,args.acf_xmax if args.acf_xmax is not None else lag[-1]),
                                 title='Transformed C(t): mean ± run SD')
                    ratio_ax.grid(alpha=.2)
                    # No inset or fitted curves in the numerical correlation panel.
                    if args.show_loglinear:
                        from prn_slow_tail import add_tail_plots
                        log_ax=fig.add_subplot(layout[0,3])
                        hidden_fit_fig,hidden_axes=plt.subplots(1,2)
                        add_tail_plots(lag,ratio,*hidden_axes,log_ax,stem)
                        plt.close(hidden_fit_fig)
                    visible=lag<=(args.acf_xmax if args.acf_xmax is not None else lag[-1])+1e-8
                    low=float(np.min((ratio-ratio_std)[visible]));high=float(np.max((ratio+ratio_std)[visible]))
                    pad=.06*max(high-low,.1)
                    ratio_ax.set_ylim(low-pad,high+pad)
                    ratio_ax.text(.5,-.19,f'Histogram C∞ = {gaussian_plateau:.6f}; normalized value at t = 0 is 1.',transform=ratio_ax.transAxes,ha='center',fontsize=9)
                    ratio_ax.text(.97,.94,
                        r'$C_\infty=\exp(-\sigma_\theta^2)$',
                        transform=ratio_ax.transAxes,ha='right',va='top',fontsize=20,
                        bbox=dict(boxstyle='round,pad=0.5',facecolor='white',edgecolor='.7',alpha=.95))
                    np.savetxt(stem.with_name(stem.name+'_acf_ratio.csv'),np.column_stack([lag,mean_acf,np.full(len(lag),gaussian_plateau),ratio,defined.astype(int),manual_override.astype(int)]),delimiter=',',header='lag_ps,mean_C,histogram_Cinf,ratio,defined,manual_override',comments='')
                    # Composite trapezoidal integral; retain signed tail values.
                    integral=np.r_[0.,np.cumsum(.5*(ratio[:-1]+ratio[1:])*np.diff(lag))]
                    if not np.allclose(integral,run_integrals.mean(axis=0)):
                        raise ValueError('Mean of individual integrals mismatch')
                    integral=run_integrals.mean(axis=0)
                    integral_ax=fig.add_subplot(layout[1,1] if compact else layout[0,4 if args.show_loglinear else 3])
                    integral_max=min(args.integral_max_ps,float(lag[-1]))
                    early_integral=lag<=integral_max+1e-8
                    integral_ax.plot(lag[early_integral],integral[early_integral],color='steelblue',lw=1.5)
                    integral_ax.axhline(0,color='.5',ls=':',lw=1)
                    integral_ax.set(xlabel='Cutoff time tc (ps)',ylabel='Integral from 0 to tc (ps)',
                                    title='Mean cumulative correlation integral',xlim=(0,integral_max),ylim=(0,2))
                    integral_ax.grid(alpha=.2)
                    histogram_cutoffs=10.+np.arange(501)*.01
                    if lag[-1]<15-1e-8:
                        raise ValueError('At least 15 ps of common lags needed for cutoff histogram')
                    histogram_values=np.interp(histogram_cutoffs,lag,integral)
                    counts,edges=np.histogram(histogram_values,bins=12)
                    cutoff_average=float(np.mean(histogram_values))
                    integral_ax.axhline(cutoff_average,color='red',ls='--',lw=1.5)
                    np.savetxt(stem.with_name(stem.name+'_acf_integral_histogram_samples.csv'),
                        np.column_stack([histogram_cutoffs,histogram_values]),delimiter=',',
                        header='cutoff_ps,mean_cumulative_integral_ps',comments='')
                    np.savetxt(stem.with_name(stem.name+'_acf_integral_histogram.csv'),
                        np.column_stack([edges[:-1],edges[1:],counts]),delimiter=',',
                        header='left_ps,right_ps,count',comments='')
                    integral_ax.text(.5,-.19,'Cutoff dependence shown; convergence is not assumed.',transform=integral_ax.transAxes,ha='center',fontsize=9)
                    cutoffs=np.unique(np.r_[np.array([1.,1.5,2.,3.,5.,7.,10.]),integral_max])
                    cutoffs=cutoffs[cutoffs<=integral_max]
                    cutoff_values=np.interp(cutoffs,lag,integral)
                    cutoff_std=np.interp(cutoffs,lag,integral_std)
                    integral_ax.scatter(cutoffs,cutoff_values,color='#D55E00',s=16,zorder=4)
                    integral_ax.text(.03,.95,rf'$\tau = {cutoff_average:.4f}\ \mathrm{{ps}}$',
                        transform=integral_ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
                    integral_ax.text(.97,.91,
                        r'$\tau_{\rm corr}(t_c)=\int_0^{t_c}\frac{\overline{C}(t)-C_\infty}{1-C_\infty}\,dt$',
                        transform=integral_ax.transAxes,ha='right',va='top',fontsize=20,
                        bbox=dict(boxstyle='round,pad=0.5',facecolor='white',edgecolor='.7',alpha=.95))
                    np.savetxt(stem.with_name(stem.name+'_acf_integral_cutoffs.csv'),np.column_stack([cutoffs,cutoff_values,cutoff_std]),delimiter=',',header='cutoff_ps,tau_corr_ps,run_std_ps',comments='')
                    np.savetxt(stem.with_name(stem.name+'_acf_integral.csv'),np.column_stack([lag[early_integral],ratio[early_integral],integral[early_integral],integral_std[early_integral]]),delimiter=',',header='cutoff_ps,normalized_C,cumulative_integral_ps,run_std_ps',comments='')
                    np.savetxt(stem.with_name(stem.name+'_acf_ratio_stats.csv'),np.column_stack([lag,ratio,ratio_std]),delimiter=',',header='lag_ps,mean_normalized_C,run_std_normalized_C',comments='')
                    np.savetxt(stem.with_name(stem.name+'_acf_integral_runs.csv'),np.vstack([
                        np.column_stack([np.full(early_integral.sum(),r),lag[early_integral],run_ratios[i,early_integral],run_integrals[i,early_integral]])
                        for i,r in enumerate(run_ids)]),delimiter=',',header='run_id,cutoff_ps,normalized_C,cumulative_integral_ps',comments='')
                    stem.with_name(stem.name+'_acf_integral.json').write_text(json.dumps(dict(
                        integrand='(mean_C(t)-histogram_Cinf)/(1-histogram_Cinf)',Cinf=gaussian_plateau,
                        method='Composite trapezoidal integration from zero; no smoothing or clipping',
                        units='ps',last_cutoff_ps=integral_max,last_integral_ps=float(np.interp(integral_max,lag,integral)),
                        mean_integral_over_cutoffs_10_to_15_ps=cutoff_average,cutoff_sample_step_ps=.01,cutoff_sample_count=len(histogram_values),
                        last_run_std_ps=float(np.interp(integral_max,lag,integral_std)),n_runs=len(run_ids),std_ddof=1,
                        uncertainty='Between-run sample SD of individually integrated normalized curves, not SEM; common histogram Cinf held fixed, its uncertainty excluded.',
                        caveat='Cutoff-dependent cumulative integral; no converged relaxation time inferred.'),indent=2)+'\n')
                # Keep legacy C(t) rendering available without displaying it when
                # the normalized-correlation/integral layout is requested.
                if args.acf_ratio_panel:
                    hidden_acf_fig,acf_ax=plt.subplots()
                else:
                    acf_ax=fig.add_subplot(layout[0,2])
                acf_ax.fill_between(lag,mean_acf-std_acf,mean_acf+std_acf,color='steelblue',alpha=.25,label='±1 run-to-run standard deviation')
                acf_ax.plot(lag,mean_acf,color='steelblue',lw=1.3,label='Equal-run mean')
                acf_ax.axhline(gaussian_plateau,color='blue',ls='--',lw=2.8,
                              label=f'exp(−σ²) = {gaussian_plateau:.5f}; σ = {np.rad2deg(theta_std):.2f}°')
                acf_ax.set(xlabel='Lag t (ps)',ylabel='C(t)',xlim=(0,lag[-1]),title=f'PRN-{state}: cosine dihedral autocorrelation\n{len(metadata["included_run_ids"])} runs; all within-run time origins')
                acf_ax.grid(alpha=.2)
                if args.acf_xmax is not None:
                    acf_ax.set_xlim(0,args.acf_xmax)
                if args.acf_ymin is not None:
                    acf_ax.set_ylim(bottom=args.acf_ymin)
                acf_ax.legend(fontsize=8,loc='best')
                acf_ax.text(.5,-.19,'No plateau subtraction; long lags have fewer time origins.',transform=acf_ax.transAxes,ha='center',fontsize=9)
                acf_info.update(density_std_rad=theta_std,density_std_deg=float(np.rad2deg(theta_std)),
                                gaussian_plateau=gaussian_plateau,
                                plateau_definition='exp(-sigma_rad^2); sigma from probability-weighted bin-center variance of pooled P(dihedral), locally unwrapped about conformer center; Gaussian prediction, not fitted Gaussian sigma',
                                plateau_caveat='For a general stationary angular distribution the independent-angle limit is |mean exp(i theta)|^2; exp(-variance) assumes Gaussian angles. Pooled density and equal-run ACF have different weighting if trajectory lengths differ.')
                stem.with_name(stem.name+'_acf.json').write_text(json.dumps(acf_info,indent=2)+'\n')
                np.savetxt(stem.with_name(stem.name+'_acf_plateau.csv'),
                           [[theta_mean,theta_std,np.rad2deg(theta_std),gaussian_plateau]],delimiter=',',
                           header='local_mean_rad,density_std_rad,density_std_deg,exp_minus_variance',comments='')
                # Semilog panel disabled; fit the untransformed mean C(t) instead.
                from prn_dihedral_acf import fit_exponential_acf
                fit_cutoff=args.acf_fit_exclude_from_ps if args.acf_fit_max_ps is None else args.acf_fit_max_ps+1e-7
                exponential,fit_mask,decay_info=fit_exponential_acf(lag,mean_acf,gaussian_plateau,fit_cutoff,fit_plateau=args.acf_fit_plateau)
                fit_range=f'0 ≤ t < {args.acf_fit_exclude_from_ps:g} ps' if args.acf_fit_max_ps is None else f'0 ≤ t ≤ {args.acf_fit_max_ps:g} ps'
                decay_info.update(fit_range=fit_range,acf_ymin=args.acf_ymin)
                acf_ax.plot(lag,exponential,color='red',ls='--',lw=2.2,visible=not (args.hide_acf_fit or args.underdamped),
                            label=f'Fit: τ = {decay_info["tau_ps"]:.4f} ps; C∞ = {decay_info["plateau"]:.5f}')
                handles,labels=acf_ax.get_legend_handles_labels()
                visible=[(h,l) for h,l in zip(handles,labels) if h.get_visible()]
                acf_ax.legend([h for h,l in visible],[l for h,l in visible],fontsize=8,loc='upper right')
                if not args.underdamped and not args.hide_acf_fit:
                    acf_ax.text(.03,.38,f'Fit {fit_range}\nR² = {decay_info["r_squared"]:.4f}; RMSE = {decay_info["rmse"]:.5f}\nC∞ fit − histogram = {decay_info["plateau_difference"]:+.6f}',transform=acf_ax.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
                elif args.underdamped:
                    from prn_underdamped_acf import fit_windows
                    fits,curves=fit_windows(lag,mean_acf,theta_std,stem)
                    chosen=fits[1]
                    acf_ax.plot(lag,curves[1],'r--',lw=2,label='Underdamped: fit 0–0.3 ps')
                    for i in (0,2,3):
                        acf_ax.plot(lag,curves[i],ls=':',lw=1,alpha=.75,label=f'Fit 0–{fits[i]["fit_end_ps"]:g} ps')
                    handles,labels=acf_ax.get_legend_handles_labels()
                    visible=[(h,l) for h,l in zip(handles,labels) if h.get_visible()]
                    acf_ax.legend([h for h,l in visible],[l for h,l in visible],fontsize=7,loc='upper right')
                    period_label='T_osc unresolved: ω_d → 0 (bound)' if chosen['near_parameter_bound'] else f'T_osc = {chosen["T_osc_ps"]:.4f} ps'
                    acf_ax.text(.03,.38,f'0–0.3 ps: τ_damp = {chosen["tau_damp_ps"]:.4f} ps\n{period_label}\nω₀ = {chosen["omega_0_rad_per_ps"]:.2f} rad/ps\nR² = {chosen["r_squared"]:.3f}; RMSE = {chosen["rmse"]:.4f}',transform=acf_ax.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
                np.savetxt(stem.with_name(stem.name+'_acf_exponential.csv'),np.column_stack([lag,mean_acf,exponential,fit_mask.astype(int)]),delimiter=',',header='lag_ps,mean_C,fitted_C,in_fit',comments='')
                stem.with_name(stem.name+'_acf_exponential.json').write_text(json.dumps(decay_info,indent=2)+'\n')
                if args.acf_ratio_panel:
                    plt.close(hidden_acf_fig)
        else:
            ax,density_ax=fig.subplots(1,2)
        ax.plot(x,f,'o-',color='steelblue' if state=='anti' else 'darkorange',ms=3,label=f'Sampled PMF ({metadata["bin_width_deg"]:g}° bins)')
        ax.set(xlim=(center-100,center+100),ylim=(-.2,6.5),xlabel='Dihedral (degrees)',ylabel='Relative PMF (kcal/mol)')
        ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=9)
        ax.set_title('Dihedral potential of mean force')
        histogram_weights=data['frame_count']/np.sum(data['frame_count'])
        histogram_angles=(x-center+180)%360-180
        histogram_mean=float(np.sum(histogram_weights*histogram_angles))
        histogram_sigma=float(np.sqrt(np.sum(histogram_weights*(histogram_angles-histogram_mean)**2)))
        density_ax.bar(x,data['probability_density_per_degree'],width=float(metadata['bin_width_deg']),color='steelblue' if state=='anti' else 'darkorange',alpha=.35,label='Normalized sampled histogram\n'+rf'$\sigma_\theta = {histogram_sigma:.2f}^\circ$')
        density_ax.set(xlim=(center-100,center+100),ylim=(0,1.15*data['probability_density_per_degree'].max()),xlabel='Dihedral (degrees)',ylabel='P(φ) (degree⁻¹)')
        density_ax.set_title('Dihedral probability density')
        density_ax.legend(loc='upper right')
        density_ax.grid(alpha=.2)
        if state in ('anti','syn'):
            for row in records:
                e=row['energy_kcal_mol']
                ax.axhline(e,color='gray',ls=':',lw=.7)
                ax.scatter([row['sampled_left_deg'],row['sampled_right_deg']],[e,e],color='steelblue',s=32,zorder=5)
            cell=[[str(r['energy_kcal_mol']),f"{r['sampled_left_deg']:.1f}°",
                   f"{r['left_pair_reference_percent']:.2f}% : {r['left_pair_angle_percent']:.2f}%",
                   f"{r['sampled_right_deg']:.1f}°",
                   f"{r['right_pair_reference_percent']:.2f}% : {r['right_pair_angle_percent']:.2f}%"] for r in records]
        for panel in fig.axes:
            panel.grid(False)
            panel.tick_params(labelsize=plt.rcParams['xtick.labelsize'])
            panel.title.set_fontsize(plt.rcParams['axes.titlesize'])
            panel.xaxis.label.set_size(plt.rcParams['axes.labelsize']);panel.yaxis.label.set_size(plt.rcParams['axes.labelsize'])
            legend=panel.get_legend()
            if legend is not None:legend.set_frame_on(False)
        fig.suptitle(f'PRN-{state}: intrabasin dihedral sampling and correlation',fontsize=15)
        if not compact:fig.tight_layout(rect=(0,.08,1,.96))
        fig.savefig(stem.with_name(stem.name+'_parabola.png'),dpi=300);plt.close(fig)
        info=dict(a_kcal_mol_per_degree_squared=float(a),b=float(b),c=float(c),center_deg=center,
                  vertex_deg=float(vertex),fitted_minimum_kcal_mol=float(minimum),r_squared=float(r2),
                  fit_max_kcal_mol=args.fit_max,fit_bins=int(keep.sum()),rmse_kcal_mol=float(np.sqrt(np.mean(residual**2))),
                  temperature_K=temperature,gaussian_mu_deg=float(vertex),gaussian_sigma_deg=sigma,
                  gaussian_definition='Normalized local normal density; sigma=sqrt(R*T/(2*a)); no independent density fit. Additive F offset cancels.',
                  pairwise_percent_definition='Sampled density interpolated at center and each PMF crossing, normalized within each pair; not basin populations. F levels reference the sampled minimum; center density may differ slightly from that peak.',
                  definition='Unweighted OLS of F=a*(phi-center)^2+b*(phi-center)+c; fitted minimum not shifted',
                  note='Descriptive local fit to correlated histogram estimates, not a statistical test of harmonicity')
        stem.with_name(stem.name+'_parabola.json').write_text(json.dumps(info,indent=2)+'\n')
        print(state,json.dumps(info),flush=True)
        if state=='anti': print(records,flush=True)


if __name__=='__main__':
    main()
