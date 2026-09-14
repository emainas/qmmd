"""Compare saved BV FES-minima differences with equilibration basin means."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_pka_grid import deltaf, find_minimum_near_target


def collect_rows(directory: Path, half_window: float = 0.1) -> list[dict]:
    """Read only saved finalist reports; preserve their individual FES snapshots."""
    with (directory / "index.csv").open() as handle:
        runs = [int(row["run"]) for row in csv.DictReader(handle)]
    rows = []
    for run in runs:
        folder = directory / f"run-{run}"
        fes_path = folder / "summary_fes_snapshot.csv"
        with fes_path.open() as handle:
            time = float(handle.readline().strip().split("=", 1)[1])
        block = np.loadtxt(fes_path, delimiter=",")
        if not np.isfinite(block[:, :2]).all():
            raise ValueError(f"Nonfinite FES data: {fes_path}")
        low = find_minimum_near_target(block[:, 0], block[:, 1], 0., half_window, 0., 1.25)
        high = find_minimum_near_target(block[:, 0], block[:, 1], 1., half_window, 0., 1.25)
        dg = deltaf(block, 0., 1., half_window, 0., 1.25)
        with (folder / "equil_single_bridge_3d_basins.csv").open() as handle:
            basins = list(csv.DictReader(handle))
        if len(basins) != 1:
            raise ValueError(f"run-{run}: expected one major basin, found {len(basins)}")
        basin = basins[0]
        rows.append({
            "run_id": run, "delta_G_kcal_mol": dg,
            **{f"{name}_mean_deg": float(basin[f"{name}_mean_deg"])
               for name in ("single5", "single10", "single15")},
            "equil_basin_id": int(basin["basin_id"]),
            "equil_basin_population": float(basin["population"]),
            "fes_snapshot_time_ps": time,
            "s_deprotonated_min": low[0], "s_protonated_min": high[0],
            "F_deprotonated_min_kcal_mol": low[1],
            "F_protonated_min_kcal_mol": high[1],
        })
    return rows


def draw(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rows))
    dg = np.array([r["delta_G_kcal_mol"] for r in rows])
    ax.scatter(x, dg, s=110, c="#326c9b", edgecolors="white", linewidths=1, zorder=3)
    for i, row in enumerate(rows):
        ax.annotate(f"Run {row['run_id']}\n{dg[i]:.2f}", (i, dg[i]),
                    xytext=(0, 12), textcoords="offset points", ha="center", fontsize=11)
    labels = ["(" + ", ".join(f"{r[f'{name}_mean_deg']:.1f}"
              for name in ("single5", "single10", "single15")) + ")" for r in rows]
    ax.set_xticks(x, labels)
    ax.set_xlabel("Equilibration basin circular means (single5, single10, single15), degrees", labelpad=12)
    ax.set_ylabel(r"$\Delta G = F(s\approx0)-F(s\approx1)$ (kcal mol$^{-1}$)")
    ax.set_title("BV HIC finalists: equilibration conformation vs metadynamics ΔG", pad=18)
    ax.grid(axis="y", alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-.5, len(rows) - .5)
    ax.margins(y=.3)
    fig.text(.5, .025, "Triplets are categorical, not a continuous reaction coordinate.\n"
             "ΔG uses each saved F(s) snapshot; no pKa conversion or time-window averaging.",
             ha="center", fontsize=9, color="#505966")
    fig.tight_layout(rect=(0, .095, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/BV/meta-hic-finalist-runs"))
    parser.add_argument("--half-window", type=float, default=.1,
                        help="Minimum search half-width in dimensionless coordination s")
    args = parser.parse_args()
    if not 0 < args.half_window < .5:
        parser.error("--half-window must be between 0 and 0.5")
    rows = collect_rows(args.report_dir, args.half_window)
    if not rows:
        raise ValueError("No runs in index.csv")
    stem = args.report_dir / "equil_basin_vs_meta_dg"
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw(rows, stem.with_suffix(".png"))
    for row in rows:
        print(f"run-{row['run_id']}: ΔG={row['delta_G_kcal_mol']:.6f} kcal/mol")


if __name__ == "__main__":
    main()
