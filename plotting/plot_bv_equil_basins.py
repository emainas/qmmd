#!/usr/bin/env python3
"""Plot BV single5/single10/single15 sampling from equilibration only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from acid_base_BV import BV_DIHEDRAL_ATOM_NAMES, _bv_atom_name_ids
from bv_conformation_basins import circular_mean_std, density_basins, read_torsions, wrap_degrees

LABELS = ("single5", "single10", "single15")
STEM = "equil_single_bridge_3d"


def draw(path: Path, run: int, t: np.ndarray, phi: np.ndarray, labels: np.ndarray,
         basins: list[dict], reference: np.ndarray, limits: np.ndarray, settings: dict,
         stage: str = "equilibration", show_modes: bool = False) -> None:
    display = reference + wrap_degrees(phi - reference)
    fig = plt.figure(figsize=(12.5, 8.5), facecolor="white")
    ax = fig.add_axes([.01, .12, .73, .79], projection="3d")
    colors = plt.get_cmap("tab10")
    for label in np.unique(labels):
        subset = display[labels == label]
        color = "#b7bdc5" if label == 0 else colors((label - 1) % 10)
        ax.scatter(*subset.T, s=5, alpha=.35, color=color, edgecolors="none", depthshade=False)
    for basin in basins:
        peak = reference + wrap_degrees(np.asarray(basin["mode_deg"]) - reference)
        ax.scatter(*peak, s=160, marker="*", c=[colors((basin['label'] - 1) % 10)], edgecolor="black", linewidth=.8, depthshade=False)
        label_position = peak + np.array([0, 0, .045 * np.ptp(limits[2])])
        ax.text(*label_position, f"  B{basin['label']} ({100*basin['population']:.1f}%)", fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=2))
    for axis, label, bounds in zip((ax.xaxis, ax.yaxis, ax.zaxis), LABELS, limits):
        axis.set_label_text(f"{label} (degrees)")
        axis.labelpad = 12
        axis.set_major_formatter(FuncFormatter(lambda x, _: f"{float(wrap_degrees(x)):.0f}"))
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=24, azim=-55)
    fig.suptitle(f"BV · run {run} · {stage} conformational sampling", fontsize=17, fontweight="bold", y=.97)
    scope = "no metadynamics frames" if stage == "equilibration" else "saved summary frames only; original report clock"
    fig.text(.5, .918, f"{t[0]:.3f}–{t[-1]:.3f} ps  |  {len(t):,} frames  |  {scope}", ha="center", fontsize=11, color="#45515e")
    y = .82
    fig.text(.76, y, "Observed density basins", fontsize=12, fontweight="bold"); y -= .045
    for basin in basins[:5]:
        mean = ", ".join(f"{v:.1f}°" for v in basin["mode_deg" if show_modes else "mean_deg"])
        fig.text(.76, y, f"B{basin['label']}  ·  {100*basin['population']:.1f}%", fontsize=11,
                 fontweight="bold", color=colors((basin['label'] - 1) % 10))
        descriptor = "Density peak" if show_modes else "Circular mean"
        fig.text(.76, y - .03, f"{descriptor}:\n({mean})", fontsize=9, linespacing=1.5, va="top")
        y -= .105
    if not basins:
        fig.text(.76, y, "No basin meets\nthe population threshold.", fontsize=10)
    minor = 100 * np.mean(labels == 0)
    fig.text(.76, .19, f"Grey: minor catchments ({minor:.1f}%)\nStars: density modes\nB labels are local to each run", fontsize=9, color="#45515e", linespacing=1.6)
    fig.text(.06, .045, f"Periodic 3D density: {settings['bin_deg']:g}° bins, {settings['bandwidth_deg']:g}° smoothing; "
             f"peaks within {settings['merge_deg']:g}° merged; basin population ≥{100*settings['min_population']:g}%.\n" +
             ("Full equilibration retained. " if stage == "equilibration" else "Biased, unreweighted summary sampling. ") +
             "Populations are descriptive frame fractions, not converged free-energy estimates.", fontsize=9, color="#45515e")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-path", type=Path, default=Path("systems/BV/solv_4.0/dftb/N1T48C1"))
    p.add_argument("--run-ids", type=int, nargs="+", default=[39, 57, 99, 142, 190])
    p.add_argument("--out-dir", type=Path, default=Path("reports/BV/meta-hic-finalist-runs"))
    p.add_argument("--parm", type=Path, default=Path("systems/BV/solv_4.0/salt/ready.parm7"))
    p.add_argument("--solute-atoms", type=int, default=78)
    p.add_argument("--bin-deg", type=float, default=10., help="Periodic histogram bin width (degrees; divides 360)")
    p.add_argument("--bandwidth-deg", type=float, default=15., help="Gaussian smoothing sigma in degrees")
    p.add_argument("--merge-deg", type=float, default=30., help="Periodic Euclidean separation for merging nearby modes (degrees)")
    p.add_argument("--min-population", type=float, default=.05, help="Minimum basin fraction of all sampled frames")
    p.add_argument("--overwrite", action="store_true", help="Replace only this script's outputs")
    args = p.parse_args()
    settings = {k: getattr(args, k) for k in ("bin_deg", "bandwidth_deg", "merge_deg", "min_population")}
    atom_names = _bv_atom_name_ids(args.parm, args.solute_atoms)
    quartets = np.asarray([[atom_names[n] for n in BV_DIHEDRAL_ATOM_NAMES[label]] for label in LABELS])
    for run in args.run_ids:
        folder = args.out_dir / f"run-{run}"
        if not args.overwrite and any(folder.glob(STEM + "*")):
            raise FileExistsError(f"Existing conformation outputs in {folder}; use --overwrite explicitly")
    results = []
    for run in args.run_ids:
        equil = args.runs_path / f"run-{run}" / "equil"
        text = (equil / "dftb.inp").read_text()
        tv = np.asarray([[float(v) for v in line.split()[1:4]] for line in text.splitlines() if line.startswith("TV")])
        if tv.shape != (3, 3) or not np.allclose(tv, np.diag(np.diag(tv))) or np.any(np.diag(tv) <= 0):
            raise ValueError("This reader requires a positive orthorhombic box")
        if "NPT=TRUE" in text.upper().replace(" ", ""):
            raise ValueError("Variable-cell NPT requires per-frame box vectors")
        t, phi, provenance = read_torsions(equil / "traject", quartets, np.diag(tv))
        labels, basins, density = density_basins(phi, **settings)
        results.append((run, t, phi, labels, basins, density, provenance, tv))
        print(f"run-{run}: {len(t)} equil frames, {t[0]:.3f}–{t[-1]:.3f} ps; "
              f"{len(basins)} major basins", flush=True)
    pooled = np.concatenate([r[2] for r in results])
    reference, _ = circular_mean_std(pooled)
    display = reference + wrap_degrees(pooled - reference)
    limits = np.column_stack([display.min(axis=0) - 10, display.max(axis=0) + 10])
    for run, t, phi, labels, basins, density, provenance, tv in results:
        folder = args.out_dir / f"run-{run}"; folder.mkdir(parents=True, exist_ok=True)
        names = ["frame_number", "time_ps", *[f"{label}_deg" for label in LABELS], "basin_id", "smoothed_cell_probability"]
        np.savetxt(folder / f"{STEM}.csv", np.column_stack([np.arange(1, len(t) + 1), t, phi, labels, density]),
                   delimiter=",", header=",".join(names), comments="")
        fields = ["basin_id", "frames", "population"] + [f"{label}_{kind}_deg" for kind in ("mode", "mean", "circular_std") for label in LABELS]
        with (folder / f"{STEM}_basins.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for b in basins:
                row = dict(basin_id=b["label"], frames=b["count"], population=b["population"])
                for kind in ("mode", "mean", "circular_std"):
                    row.update({f"{label}_{kind}_deg": value for label, value in zip(LABELS, b[kind + "_deg"])})
                writer.writerow(row)
        provenance.update(run=run,stage="equil",quartets_one_based=dict(zip(LABELS, quartets.tolist())),
            atom_names={label: BV_DIHEDRAL_ATOM_NAMES[label] for label in LABELS}, box_vectors_A=tv.tolist(),
            settings=settings, basins=basins, frames=len(t), time_range_ps=[float(t[0]),float(t[-1])],
            display_reference_deg=reference.tolist(), common_display_limits_deg=limits.tolist(),
            minor_population=float(np.mean(labels==0)), basin_definition="Wrapped Gaussian histogram; 26-neighbor ascent; nearby modes merged; low-population catchments remain unassigned.",
            limitations="Full equilibration includes relaxation; populations are frame fractions, not equilibrium free energies. Labels are run-local. Results depend on binning/smoothing/merge thresholds; no dwell-time criterion.")
        (folder / f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
        draw(folder / f"{STEM}.png", run, t, phi, labels, basins, reference, limits, settings)


if __name__ == "__main__":
    main()
