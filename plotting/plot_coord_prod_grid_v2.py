#!/usr/bin/env python3
"""Condensed 2x2 plot: s(t), FES curves, Mulliken time series, N–O(defect+) distance."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from plot_coord_prod_grid import (
    EH_TO_KCALMOL,
    apply_tmax,
    compute_time_window,
    infer_system,
    parse_biaspot_coord_series,
    parse_biaspot_times_ps,
    parse_extra_ids,
    parse_fes_times,
    parse_mulliken,
    parse_mulliken_limited,
    read_fes_blocks,
    read_box_lengths_from_dftb_inp,
    iter_xyz_coords,
)


def select_defect_ids(charges: np.ndarray, target_ids: List[int], elements: dict[int, str]) -> np.ndarray:
    oxygen_idx = [i for i, atom_id in enumerate(target_ids) if elements.get(atom_id) == "O"]
    if not oxygen_idx:
        return np.array([], dtype=float)
    oxy_ids = [target_ids[i] for i in oxygen_idx]
    oxy_chg = charges[oxygen_idx, :]
    sel: List[float] = []
    for j in range(oxy_chg.shape[1]):
        vals_j = oxy_chg[:, j]
        mask = np.isfinite(vals_j)
        if not np.any(mask):
            sel.append(np.nan)
        else:
            idx = int(np.argmax(vals_j[mask]))
            cand_indices = np.where(mask)[0]
            sel.append(oxy_ids[cand_indices[idx]])
    return np.array(sel, dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(description="Condensed 2x2 coordination/FES/Mulliken/defect plot (prod).")
    p.add_argument("--runs-path", required=True, type=Path)
    p.add_argument("--run-dir", required=True, help="Run subdirectory (e.g., meta-hie)")
    p.add_argument("--run-id", required=True, type=int)
    p.add_argument("--solute-atoms", type=int, default=None)
    p.add_argument("--extra-ids", type=str, default=None)
    p.add_argument("--n-ids", type=str, default=None, help="Comma-separated N atom IDs for N-O distance")
    p.add_argument("--traj-name", type=str, default="traject")
    p.add_argument("--fes-name", type=str, default="fes.dat")
    p.add_argument("--biaspot-name", type=str, default="biaspot")
    p.add_argument("--fes-times", type=str, default=None, help="Comma-separated FES times in ps")
    p.add_argument("--t-fill", type=float, default=None, help="Start time (ps) for histograms")
    p.add_argument("--T-max", type=float, default=None, help="Cap time series at this time (ps)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    style = Path(__file__).resolve().parent / "lefteris.mplstyle"
    if style.exists():
        plt.style.use(style)

    run_dir = args.runs_path / f"run-{args.run_id}" / args.run_dir
    manual_cv = run_dir / "manual-cv"

    # s(t) from biaspot if available, otherwise coord.dat
    coord_path = manual_cv / "coord.dat"
    s_vals = np.array([], dtype=float)
    if coord_path.exists():
        s_vals = np.loadtxt(coord_path, usecols=[-1]) if coord_path.stat().st_size > 0 else np.array([], dtype=float)

    t = np.arange(s_vals.size, dtype=float) * 0.01
    biaspot_path = run_dir / args.biaspot_name
    if biaspot_path.exists():
        t_bias, vals_bias = parse_biaspot_coord_series(biaspot_path, args.T_max)
        if t_bias.size > 0 and vals_bias.size > 0:
            if args.T_max is not None:
                t_bias, idx_keep = apply_tmax(t_bias, args.T_max)
                vals_bias = vals_bias[idx_keep] if idx_keep.size > 0 else vals_bias[:0]
            t = t_bias
            s_vals = vals_bias

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=220)
    panel_labels = list("abcd")
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

    # (a) s(t)
    ax_a = axes[0, 0]
    if s_vals.size > 0:
        t_plot = t - t[0]
        ax_a.plot(t_plot, s_vals, color="black", lw=1.4)
        ax_a.set_xlim(0.0, float(t_plot[-1]))
        ax_a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x + t[0]:.1f}"))
    ax_a.set_xlabel("t (ps)")
    ax_a.set_ylabel("s(t)")
    ax_a.set_ylim(0.0, 1.0)

    # (b) FES curves
    ax_b = axes[0, 1]
    fes_times = parse_fes_times(args.fes_times)
    fes_path = run_dir / args.fes_name
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
                ax_b.set_xlabel("s")
                ax_b.set_ylabel("F(s) [kcal/mol]")
                ax_b.set_xlim(0.0, 1.0)
                ax_b.legend(frameon=False, fontsize=7, loc="best")

    # (c) Mulliken time series for waters + selected extra IDs
    ax_c = axes[1, 0]
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
            oxygen_idx = [i for i, atom_id in enumerate(target_ids) if elements.get(atom_id) == "O"]
            sel_ids = None
            least_ids = None
            if oxygen_idx:
                oxy_ids = [target_ids[i] for i in oxygen_idx]
                oxy_chg = charges[oxygen_idx, :]
                sel = []
                sel_least = []
                for j in range(oxy_chg.shape[1]):
                    vals_j = oxy_chg[:, j]
                    mask = (vals_j >= -1.4) & (vals_j <= -1.1)
                    if not np.any(mask):
                        sel.append(np.nan)
                    else:
                        idx = int(np.argmin(vals_j[mask]))
                        cand_indices = np.where(mask)[0]
                        sel.append(oxy_ids[cand_indices[idx]])
                    mask_least = (vals_j >= -0.625) & (vals_j <= -0.525)
                    if not np.any(mask_least):
                        sel_least.append(np.nan)
                    else:
                        idx_least = int(np.argmax(vals_j[mask_least]))
                        cand_least = np.where(mask_least)[0]
                        sel_least.append(oxy_ids[cand_least[idx_least]])
                sel_ids = np.array(sel, dtype=float)
                least_ids = np.array(sel_least, dtype=float)
            mull_data = (times, charges, target_ids, elements, extra_ids, sel_ids, least_ids)

    if mull_data is None:
        ax_c.text(0.5, 0.5, "mulliken not found", ha="center", va="center")
    else:
        times, charges, target_ids, elements, extra_ids, _sel_ids, _least_ids = mull_data
        if args.T_max is not None:
            times, idx_keep = apply_tmax(times, args.T_max)
            charges = charges[:, idx_keep] if idx_keep.size > 0 else charges[:, :0]
        colors = plt.cm.coolwarm(np.linspace(0, 1, max(1, len(target_ids))))
        for i, atom_id in enumerate(target_ids):
            elem = elements.get(atom_id, "X")
            lw = 1.6 if atom_id in extra_ids else 0.9
            ax_c.plot(times, charges[i], lw=lw, color=colors[i], label=f"{elem}{atom_id}" if atom_id in extra_ids else None)
        ax_c.set_xlabel("t (ps)")
        ax_c.set_ylabel("q(t) [e$^-$]")
        if extra_ids:
            ax_c.legend(frameon=False, fontsize=7, loc="best")

    # (d) N–O(defect+) distance
    ax_d = axes[1, 1]
    if args.n_ids and mull_data is not None:
        n_ids = [int(x) for x in args.n_ids.split(",") if x.strip()]
        times, charges, target_ids, elements, _extra_ids, _sel_ids, least_ids = mull_data
        if args.T_max is not None:
            times, idx_keep = apply_tmax(times, args.T_max)
            charges = charges[:, idx_keep] if idx_keep.size > 0 else charges[:, :0]
            if least_ids is not None:
                least_ids = least_ids[idx_keep] if idx_keep.size > 0 else least_ids[:0]
        traj_path = run_dir / args.traj_name
        if traj_path.exists() and least_ids is not None and least_ids.size > 0:
            dftb_inp = run_dir / "dftb.inp"
            box = read_box_lengths_from_dftb_inp(dftb_inp) if dftb_inp.exists() else None
            n_series = {nid: [] for nid in n_ids}
            t_series: List[float] = []
            max_frames = len(times)
            for idx, coords in enumerate(iter_xyz_coords(traj_path)):
                if idx >= max_frames:
                    break
                o_id = least_ids[idx]
                if not np.isfinite(o_id):
                    for nid in n_ids:
                        n_series[nid].append(np.nan)
                    t_series.append(times[idx])
                    continue
                o_idx = int(o_id) - 1
                for nid in n_ids:
                    n_idx = int(nid) - 1
                    d = coords[n_idx] - coords[o_idx]
                    if box is not None:
                        d = d - box * np.round(d / box)
                    n_series[nid].append(float(np.linalg.norm(d)))
                t_series.append(times[idx])
            for nid in n_ids:
                ax_d.plot(t_series, n_series[nid], lw=2, color="red", alpha=0.7, label=f"N{nid}")
            ax_d.set_xlabel("t (ps)")
            ax_d.set_ylabel("N-O(defect) distance (Å)")
            ax_d.legend(frameon=False, fontsize=7, loc="best")
        else:
            ax_d.text(0.5, 0.5, "traject not found", ha="center", va="center")
    else:
        ax_d.text(0.5, 0.5, "missing N IDs or mulliken data", ha="center", va="center")

    # synchronize time axis if possible
    if s_vals.size > 0 and mull_data is not None:
        time_window = compute_time_window(t, mull_data[0])
        if time_window is not None:
            x_start, x_end = time_window
            if args.T_max is not None:
                x_end = min(x_end, args.T_max)
            ax_a.set_xlim(x_start - t[0], x_end - t[0])
            ax_c.set_xlim(x_start, x_end)
            ax_d.set_xlim(x_start, x_end)

    system = infer_system(args.runs_path)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.run_dir}_report_prod_v2_run{args.run_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
