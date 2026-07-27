#!/usr/bin/env python3
"""Condensed 1x2 plot: FES curves and Mulliken time series."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter1d

from plot_coord_prod_grid import (
    EH_TO_KCALMOL,
    apply_tmax,
    infer_system,
    find_minimum_near_target,
    parse_biaspot_times_ps,
    parse_extra_ids,
    parse_fes_times,
    parse_mulliken,
    parse_mulliken_limited,
    read_fes_blocks,
)

PKA_HALF_WINDOW = 0.1
PKA_FES_XMIN = 0.0
PKA_FES_XMAX = 1.25


def gaussian_smooth(y: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    if y.size == 0 or sigma <= 0:
        return y
    return gaussian_filter1d(y, sigma=sigma, mode="nearest")

def main() -> None:
    p = argparse.ArgumentParser(description="Condensed 1x2 plot: FES curves + Mulliken time series (prod).")
    p.add_argument("--runs-path", type=Path, default=None)
    p.add_argument("--run-dir", type=str, default=None, help="Run subdirectory (e.g., meta-hie)")
    p.add_argument("--run-id", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=None, help="Direct path to folder with fes.dat/biaspot/mulliken")
    p.add_argument("--solute-atoms", type=int, default=None)
    p.add_argument("--extra-ids", type=str, default=None)
    p.add_argument("--fes-name", type=str, default="fes.dat")
    p.add_argument("--biaspot-name", type=str, default="biaspot")
    p.add_argument("--fes-times", type=str, default=None, help="Comma-separated FES times in ps")
    p.add_argument("--T-max", type=float, default=None, help="Cap time series at this time (ps)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    style = Path(__file__).resolve().parent / "lefteris.mplstyle"
    if style.exists():
        plt.style.use(style)

    if args.data_dir is not None:
        run_dir = args.data_dir
    else:
        if args.runs_path is None or args.run_dir is None or args.run_id is None:
            raise SystemExit("Provide --data-dir or all of --runs-path/--run-dir/--run-id")
        run_dir = args.runs_path / f"run-{args.run_id}" / args.run_dir

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=220)
    panel_labels = list("ab")
    for ax, label in zip(axes.flat, panel_labels):
        ax.text(
            0.02,
            0.98,
            f"({label})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    # (a) FES curves
    ax_b = axes[0]
    fes_times = parse_fes_times(args.fes_times)
    fes_path = run_dir / args.fes_name
    biaspot_path = run_dir / args.biaspot_name
    if not fes_path.exists() or not biaspot_path.exists():
        ax_b.text(0.5, 0.5, "fes.dat or biaspot not found", ha="center", va="center")
    elif not fes_times:
        ax_b.text(0.5, 0.5, "missing --fes-times", ha="center", va="center")
    else:
        times = parse_biaspot_times_ps(biaspot_path, args.T_max)
        blocks = read_fes_blocks(fes_path, max_blocks=len(times) if times.size > 0 else None)
        if times.size == 0 or not blocks:
            ax_b.text(0.5, 0.5, "fes blocks not found", ha="center", va="center")
        else:
            nblocks = min(len(times), len(blocks))
            times = times[:nblocks]
            blocks = blocks[:nblocks]
            if args.T_max is not None:
                times, idx_keep = apply_tmax(times, args.T_max)
                blocks = [blocks[i] for i in idx_keep] if idx_keep.size > 0 else []
            curves = []
            for t_ps in fes_times:
                idx = int(np.argmin(np.abs(times - t_ps)))
                data = blocks[idx].copy()
                data[:, 1] *= EH_TO_KCALMOL
                curves.append((times[idx], data))
            if curves:
                global_min = min(float(np.min(d[:, 1])) for _, d in curves)
                for t_ps, data in curves:
                    ax_b.plot(data[:, 0], data[:, 1] - global_min, lw=1.2, label=f"{t_ps:.2f} ps")
                ax_b.set_xlabel("s (coord. number)")
                ax_b.set_ylabel("F(s) [kcal/mol]")
                ax_b.set_xlim(0.0, 1.0)
                ax_b.legend(frameon=False, fontsize=7, loc="best")
                # Label minima on earliest curve
                base_t, base_data = curves[0]
                x = base_data[:, 0]
                y = base_data[:, 1] - global_min
                for target, label in [
                    (0.0, "protonated"),
                    (0.5, "contact ion pair"),
                    (1.0, "deprotonated"),
                ]:
                    try:
                        xm, ym = find_minimum_near_target(x, y, target, PKA_HALF_WINDOW, PKA_FES_XMIN, PKA_FES_XMAX)
                    except Exception:
                        continue
                    ax_b.plot([xm], [ym], marker="o", color="black", markersize=3)
                    ax_b.text(xm, ym + 0.3, label, fontsize=8, ha="center")

    # (b) Mulliken time series for waters + selected extra IDs
    ax_c = axes[1]
    mull_data = None
    if args.solute_atoms is not None:
        mulliken_path = run_dir / "mulliken"
        if mulliken_path.exists():
            extra_ids = parse_extra_ids(args.extra_ids)
            if args.T_max is not None:
                times, charges, target_ids, elements = parse_mulliken_limited(
                    mulliken_path, args.solute_atoms, args.T_max, natoms=None, extra_ids=extra_ids
                )
            else:
                times, charges, target_ids, elements = parse_mulliken(
                    mulliken_path, args.solute_atoms, natoms=None, extra_ids=extra_ids
                )
            mull_data = (times, charges, target_ids, elements, extra_ids)

    if mull_data is None:
        ax_c.text(0.5, 0.5, "mulliken not found", ha="center", va="center")
    else:
        times, charges, target_ids, elements, extra_ids = mull_data
        if args.T_max is not None:
            times, idx_keep = apply_tmax(times, args.T_max)
            charges = charges[:, idx_keep] if idx_keep.size > 0 else charges[:, :0]
        colors = plt.cm.coolwarm(np.linspace(0, 1, max(1, len(target_ids))))
        for i, atom_id in enumerate(target_ids):
            elem = elements.get(atom_id, "X")
            lw = 2.2 if atom_id in extra_ids else 1.4
            y = gaussian_smooth(charges[i])
            ax_c.plot(times, y, lw=lw, color=colors[i])
            if elem == "O" and atom_id not in extra_ids and y.size >= 3:
                if (float(np.nanmax(y)) - float(np.nanmin(y))) >= 0.15:
                    median = float(np.median(y))
                    std = float(np.std(y))
                    thresh = median + 0.5 * std
                    for j in range(1, y.size - 1):
                        if y[j] > y[j - 1] and y[j] > y[j + 1] and y[j] > thresh:
                            ax_c.text(
                                times[j],
                                y[j],
                                f"O{atom_id}",
                                color=colors[i],
                                fontsize=7,
                                ha="center",
                                va="bottom",
                            )
        ax_c.set_xlabel("t (ps)")
        ax_c.set_ylabel("q(t) [e$^-$]")
        ax_c.set_ylim(-0.9, -0.5)

    if args.data_dir is not None:
        system = args.data_dir.name
    else:
        system = infer_system(args.runs_path)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.run_dir}_report_prod_v3_run{args.run_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
