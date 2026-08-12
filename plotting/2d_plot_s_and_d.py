#!/usr/bin/env python3
"""Plot coordination s(t) against N--O(defect+) distance.

The script reuses the data readers from ``plot_coord_prod_grid.py`` and the
selection logic from ``plot_coord_prod_grid_v2.py``.  Coordination is
interpolated onto the Mulliken/trajectory time grid.  Missing defect assignments
remain NaN, so Matplotlib does not connect points across gaps.  A line is also
broken whenever the selected defect oxygen changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm

from plot_coord_prod_grid import (
    apply_tmax,
    infer_system,
    iter_xyz_coords,
    parse_biaspot_coord_series,
    parse_extra_ids,
    parse_mulliken,
    parse_mulliken_limited,
    read_box_lengths_from_dftb_inp,
)


def load_coordination(
    run_dir: Path,
    biaspot_name: str,
    t_max: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return coordination times and values, preferring biaspot over coord.dat."""
    coord_path = run_dir / "manual-cv" / "coord.dat"
    s_vals = np.array([], dtype=float)
    if coord_path.exists() and coord_path.stat().st_size > 0:
        s_vals = np.atleast_1d(np.loadtxt(coord_path, usecols=[-1])).astype(float)

    t_coord = np.arange(s_vals.size, dtype=float) * 0.01

    biaspot_path = run_dir / biaspot_name
    if biaspot_path.exists():
        t_bias, s_bias = parse_biaspot_coord_series(biaspot_path, t_max)
        if t_bias.size and s_bias.size:
            if t_max is not None:
                t_bias, keep = apply_tmax(t_bias, t_max)
                s_bias = s_bias[keep] if keep.size else s_bias[:0]
            t_coord, s_vals = t_bias, s_bias

    if not t_coord.size or not s_vals.size:
        raise RuntimeError("No coordination time series found in biaspot or manual-cv/coord.dat")

    n = min(t_coord.size, s_vals.size)
    return np.asarray(t_coord[:n], float), np.asarray(s_vals[:n], float)


def select_defect_oxygen_ids(
    charges: np.ndarray,
    target_ids: List[int],
    elements: Dict[int, str],
) -> np.ndarray:
    """Replicate panel (d)'s defect+ oxygen selection.

    At each frame, select the oxygen with the largest charge in the interval
    [-0.625, -0.525].  Frames without a qualifying oxygen are left as NaN.
    """
    oxygen_idx = [i for i, atom_id in enumerate(target_ids) if elements.get(atom_id) == "O"]
    if not oxygen_idx:
        return np.array([], dtype=float)

    oxy_ids = np.asarray([target_ids[i] for i in oxygen_idx], dtype=int)
    oxy_charges = charges[oxygen_idx, :]
    selected = np.full(oxy_charges.shape[1], np.nan, dtype=float)

    for frame in range(oxy_charges.shape[1]):
        vals = oxy_charges[:, frame]
        mask = np.isfinite(vals) & (vals >= -0.625) & (vals <= -0.525)
        if np.any(mask):
            candidates = np.flatnonzero(mask)
            best = candidates[int(np.argmax(vals[mask]))]
            selected[frame] = float(oxy_ids[best])

    return selected


def load_defect_series(
    run_dir: Path,
    solute_atoms: int,
    extra_ids_text: str | None,
    t_max: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Mulliken times and selected defect oxygen atom IDs."""
    mulliken_path = run_dir / "mulliken"
    if not mulliken_path.exists():
        raise FileNotFoundError(f"Mulliken file not found: {mulliken_path}")

    extra_ids = parse_extra_ids(extra_ids_text)
    if t_max is None:
        times, charges, target_ids, elements = parse_mulliken(
            mulliken_path,
            solute_atoms,
            natoms=None,
            extra_ids=extra_ids,
        )
    else:
        times, charges, target_ids, elements = parse_mulliken_limited(
            mulliken_path,
            solute_atoms,
            t_max,
            natoms=None,
            extra_ids=extra_ids,
        )

    if t_max is not None:
        times, keep = apply_tmax(times, t_max)
        charges = charges[:, keep] if keep.size else charges[:, :0]

    defect_ids = select_defect_oxygen_ids(charges, target_ids, elements)
    n = min(times.size, defect_ids.size)
    return np.asarray(times[:n], float), np.asarray(defect_ids[:n], float)


def compute_distances(
    traj_path: Path,
    times: np.ndarray,
    defect_ids: np.ndarray,
    n_ids: List[int],
    box: np.ndarray | None,
) -> Dict[int, np.ndarray]:
    """Compute minimum-image N--O(defect+) distances on the trajectory grid."""
    distances = {nid: np.full(times.size, np.nan, dtype=float) for nid in n_ids}

    for frame, coords in enumerate(iter_xyz_coords(traj_path)):
        if frame >= times.size:
            break

        o_id = defect_ids[frame]
        if not np.isfinite(o_id):
            continue

        o_idx = int(o_id) - 1
        if o_idx < 0 or o_idx >= len(coords):
            continue

        for nid in n_ids:
            n_idx = nid - 1
            if n_idx < 0 or n_idx >= len(coords):
                continue
            delta = coords[n_idx] - coords[o_idx]
            if box is not None:
                delta = delta - box * np.round(delta / box)
            distances[nid][frame] = float(np.linalg.norm(delta))

    return distances


def coordination_on_distance_grid(
    t_coord: np.ndarray,
    s_coord: np.ndarray,
    t_dist: np.ndarray,
) -> np.ndarray:
    """Interpolate s(t) only inside the actual coordination time interval."""
    result = np.full(t_dist.shape, np.nan, dtype=float)
    finite = np.isfinite(t_coord) & np.isfinite(s_coord)
    if np.count_nonzero(finite) < 2:
        return result

    tc = t_coord[finite]
    sc = s_coord[finite]
    order = np.argsort(tc)
    tc = tc[order]
    sc = sc[order]

    inside = np.isfinite(t_dist) & (t_dist >= tc[0]) & (t_dist <= tc[-1])
    result[inside] = np.interp(t_dist[inside], tc, sc)
    return result

def save_csv(
    path: Path,
    times: np.ndarray,
    coordination: np.ndarray,
    defect_ids: np.ndarray,
    distances: Dict[int, np.ndarray],
) -> None:
    columns = [times, coordination, defect_ids] + [distances[nid] for nid in distances]
    header = ["time_ps", "coordination_s", "defect_oxygen_id"] + [
        f"N{nid}_Odefect_distance_A" for nid in distances
    ]
    data = np.column_stack(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, delimiter=",", header=",".join(header), comments="")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot coordination s(t) versus N--O(defect+) distance without bridging gaps."
    )
    parser.add_argument("--runs-path", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, help="Run subdirectory, e.g. meta-hie")
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--solute-atoms", required=True, type=int)
    parser.add_argument("--n-ids", required=True, help="Comma-separated nitrogen atom IDs")
    parser.add_argument("--extra-ids", default=None)
    parser.add_argument("--traj-name", default="traject")
    parser.add_argument("--biaspot-name", default="biaspot")
    parser.add_argument("--T-max", type=float, default=None)
    parser.add_argument("--split-regime-cutoff", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None) 
    args = parser.parse_args()

    style = Path(__file__).resolve().parent / "lefteris.mplstyle"
    if style.exists():
        plt.style.use(style)

    run_dir = args.runs_path / f"run-{args.run_id}" / args.run_dir
    traj_path = run_dir / args.traj_name
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    n_ids = [int(value) for value in args.n_ids.split(",") if value.strip()]
    if not n_ids:
        raise ValueError("--n-ids must contain at least one atom ID")

    t_coord, s_coord = load_coordination(run_dir, args.biaspot_name, args.T_max)
    t_dist, defect_ids = load_defect_series(
        run_dir,
        args.solute_atoms,
        args.extra_ids,
        args.T_max,
    )

    dftb_inp = run_dir / "dftb.inp"
    box = read_box_lengths_from_dftb_inp(dftb_inp) if dftb_inp.exists() else None
    distances = compute_distances(traj_path, t_dist, defect_ids, n_ids, box)
    s_on_dist = coordination_on_distance_grid(t_coord, s_coord, t_dist)

    nrows = len(n_ids)
    fig, axes = plt.subplots(nrows, 1, figsize=(6.3, 4.5 * nrows), dpi=220, squeeze=False)

    for ax, nid in zip(axes[:, 0], n_ids):
        distance = distances[nid].copy()
        
        if args.split_regime_cutoff is not None:
            crossings = np.flatnonzero(np.isfinite(distance) & (distance >= args.split_regime_cutoff))
            split_idx = (int(crossings[0]) if crossings.size > 0 else distance.size)
            distance[(np.arange(distance.size) < split_idx) & np.isnan(distance)] = 0.0
        
        valid = np.isfinite(s_on_dist) & np.isfinite(distance) 

        points = ax.scatter(
                s_on_dist[valid],
                distance[valid],
                c=t_dist[valid],
                s=12,
                alpha=0.8,
                rasterized=True,
            )
        cbar = fig.colorbar(points, ax=ax)
        cbar.set_label("t (ps)")

        ax.set_xlabel("coordination, s(t)")
        ax.set_ylabel(rf"N{nid}--O(defect$^+$) distance ($\AA$)")
        #ax.set_ylim(-0.2,10.2)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(f"N{nid}: coordination versus defect distance")

    fig.tight_layout()

    system = infer_system(args.runs_path)
    out = args.out or Path("reports") / (
        f"{system}_{args.run_dir}_coord_vs_distance_run{args.run_id}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    
    # Make 2D Histogram here with s_on_dist[valid] and distance[valid] values. 
    
    fig, ax = plt.subplots()
    h = ax.hist2d(s_on_dist[valid], distance[valid], bins=100, density=True, norm=LogNorm(), cmap="RdBu") #cmap="magma_r")

    fig.colorbar(h[3], ax=ax, label=r"$P(s, d)$")
    ax.set_xlabel("coordination, s")
    ax.set_ylabel(r"distance ($\AA$), d")

    fig.tight_layout()
    out2 = args.out or Path("reports") / (
        f"{system}_{args.run_dir}_2DHistogram{args.run_id}.png"
    )
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)
   
    # Make a 2D Free Energy plot
    kBT = 0.596 # kcal/mol at 300K 
    tiny = 1e-12
    
    P, xedges, yedges = np.histogram2d(
        s_on_dist[valid],
        distance[valid],
        bins=100,
        density=True,
    )

    F = - kBT * np.log(P + tiny)

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc)

    fig, ax = plt.subplots()
    levels=50
    cf = ax.contourf(X, Y, F.T, levels=levels, cmap="magma")
    cs = ax.contour(X, Y, F.T, levels=levels, colors="k", linewidths=0.6)
    fig.colorbar(cf, ax=ax, label=r"$F(s, d)$ (kcal/mol)")
    ax.set_xlabel("coordination, s")
    ax.set_ylabel(r"distance ($\AA$), d")

    fig.tight_layout()
    
    out3 = args.out or Path("reports") / (
        f"{system}_{args.run_dir}_2DFE_run{args.run_id}.png"
    )

    fig.savefig(out3, bbox_inches="tight")
    plt.close(fig)

    csv_out = args.csv_out or out.with_suffix(".csv")
    save_csv(csv_out, t_dist, s_on_dist, defect_ids, distances)
    
    distance = distances[nid].copy()
    valid = np.isfinite(s_on_dist) & np.isfinite(distance)
    print(valid.sum())   # should be around 1010
    print(f"Wrote {out}")
    print(f"Wrote {out2}")
    print(f"Wrote {out3}")
    print(f"Wrote {csv_out}")
    for nid in n_ids:
        valid_count = int(np.count_nonzero(np.isfinite(s_on_dist) & np.isfinite(distances[nid])))
        print(f"N{nid}: {valid_count}/{t_dist.size} paired samples")


if __name__ == "__main__":
    main()
