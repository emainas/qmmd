"""Plot an existing PRN RDF and unshifted w(r) = -RT ln[g(r)]."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R_KCAL = 0.00198720425864083


def free_energies(r: np.ndarray, g: np.ndarray, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    """Return pair PMF and radial free energy, each shifted to minimum zero.

    r is in Angstrom; the logarithm uses r/(1 Angstrom). Empty bins stay NaN.
    """
    valid = np.isfinite(g) & (g > 0) & np.isfinite(r) & (r > 0)
    if not np.any(valid) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive temperature and sampled positive radii required')
    pair = np.full(g.shape, np.nan)
    radial = np.full(g.shape, np.nan)
    pair[valid] = -R_KCAL * temperature * np.log(g[valid])
    radial[valid] = pair[valid] - 2 * R_KCAL * temperature * np.log(r[valid])
    return pair - np.nanmin(pair), radial - np.nanmin(radial)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', choices=['anti', 'syn'], default='anti')
    parser.add_argument('--temperature', type=float, default=300., help='Temperature (K)')
    parser.add_argument('--report-root', type=Path, default=Path('reports'))
    args = parser.parse_args()
    folder = args.report_root / f'PRN-{args.state}' / 'equil_H11_waterO_rdf'
    source = folder / 'rdf.csv'
    data = np.genfromtxt(source, delimiter=',', names=True)
    r, g = data['r_mid_A'], data['g_r']
    pair, radial = free_energies(r, g, args.temperature)
    negative_log_g = np.full(g.shape, np.nan)
    positive = np.isfinite(g) & (g > 0)
    negative_log_g[positive] = -np.log(g[positive])
    w = R_KCAL * args.temperature * negative_log_g
    stem = folder / 'rdf_pmf'
    np.savetxt(stem.with_suffix('.csv'), np.column_stack([r, g, pair, radial, negative_log_g, w]),
               delimiter=',', comments='',
               header='r_A,g_r,pair_PMF_kcal_mol,radial_free_energy_with_Jacobian_kcal_mol,negative_log_g_dimensionless,w_kcal_mol')
    plt.style.use(Path(__file__).with_name('lefteris.mplstyle'))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    color = 'steelblue' if args.state == 'anti' else 'darkorange'
    axes[0].plot(r, g, color=color)
    axes[0].axhline(1, color='red', linestyle='--', linewidth=2)
    axes[0].set(title='Radial distribution function', ylabel=r'$g(r)$', ylim=(0, None))
    axes[1].plot(r, w, color=color)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set(title='Potential of mean force', ylabel=r'$w(r)$ (kcal/mol)')
    axes[1].text(.97, .94,
                 r'$w(r)=-RT\ln[g(r)]$' + '\n' + rf'$T={args.temperature:g}\,\mathrm{{K}}$',
                 transform=axes[1].transAxes, ha='right', va='top', fontsize=17,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='.7'))
    for ax in axes:
        ax.set(xlabel=r'H11–water O distance ($\mathrm{\AA}$)', xlim=(0, r[-1] + (r[1]-r[0])/2))
    fig.suptitle(f'PRN-{args.state}: equilibration solute proton–water oxygen', fontsize=19)
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=3)
    fig.savefig(stem.with_suffix('.png'), dpi=300)
    plt.close(fig)
    metadata = dict(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    temperature_K=args.temperature,
                    displayed_definition='w(r)=-RT ln[g(r)]; kcal/mol, no shift or extra Jacobian; w=0 at g=1',
                    pair_PMF_definition='-RT ln g(r) + constant; saved separately, not plotted',
                    caveats='Existing pooled RDF: all equilibration frames, no 10 ps discard; anti run 2 excluded. '
                    'Bin-center r^2 Jacobian. Zero-count bins omitted, no pseudocounts or smoothing. '
                    'Jacobian-inclusive radial free energy is retained only as a separate legacy CSV column, not plotted. '
                    'No standard-state binding free energy or statistical uncertainty inferred.',
                    reference='https://manual.gromacs.org/current/reference-manual/analysis/radial-distribution-function.html')
    stem.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
