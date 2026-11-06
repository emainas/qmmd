#!/usr/bin/env python3
"""Reproduce an extended torsional PMF and its integrated basin free energy."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path(__file__).resolve().parent
R_KCAL = 0.00198720425864083


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def integrated_values(
    coordinate: np.ndarray,
    pmf: np.ndarray,
    temperature_k: float,
    dividing_surface_deg: float,
) -> dict[str, float]:
    weights = np.exp(-(pmf - float(np.min(pmf))) / (R_KCAL * temperature_k))
    syn_weight = float(np.sum(weights[coordinate < dividing_surface_deg]))
    anti_weight = float(np.sum(weights[coordinate > dividing_surface_deg]))
    ratio = anti_weight / syn_weight
    return {
        "population_delta_g_anti_minus_syn": -R_KCAL * temperature_k * math.log(ratio),
        "syn_molar_fraction": 1.0 / (1.0 + ratio),
        "anti_molar_fraction": ratio / (1.0 + ratio),
        "anti_to_syn_ratio": ratio,
    }


def colored_curve(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
) -> None:
    cmap = plt.get_cmap("turbo_r")
    norm = Normalize(vmin=float(centers.min()), vmax=float(centers.max()))
    midpoint = 0.5 * (x[:-1] + x[1:])
    nearest = np.argmin(np.abs(midpoint[:, None] - centers[None, :]), axis=1)
    segments = np.stack(
        (np.column_stack((x[:-1], y[:-1])), np.column_stack((x[1:], y[1:]))),
        axis=1,
    )
    axis.add_collection(
        LineCollection(
            segments,
            colors=[cmap(norm(centers[index])) for index in nearest],
            linewidths=4.2,
        )
    )
    for center in centers:
        axis.axvline(
            center,
            color=cmap(norm(center)),
            linestyle="--",
            linewidth=0.65,
            alpha=0.20,
            zorder=0,
        )


def reproduce_pmf(metadata: dict[str, object], output: Path) -> None:
    summary = rows(ROOT / "data/pmf_bootstrap_summary.csv")
    coordinate = np.asarray([float(row["dihedral_deg"]) for row in summary])
    pmf = np.asarray([float(row["pmf_kcal_mol"]) for row in summary])
    lower = np.asarray([float(row["ci_lower_kcal_mol"]) for row in summary])
    upper = np.asarray([float(row["ci_upper_kcal_mol"]) for row in summary])
    smooth = rows(ROOT / "data/pmf_smooth.csv")
    smooth_x = np.asarray([float(row["dihedral_deg"]) for row in smooth])
    smooth_y = np.asarray([float(row["smoothed_pmf_kcal_mol"]) for row in smooth])
    centers = np.asarray(metadata["window_centers_deg"], dtype=float)
    values = integrated_values(
        coordinate,
        pmf,
        float(metadata["temperature_K"]),
        float(metadata["dividing_surface_deg"]),
    )
    integrated = {row["feature"]: row for row in rows(ROOT / "data/integrated_thermodynamics.csv")}
    for metric, value in values.items():
        if not math.isclose(value, float(integrated[metric]["original"]), abs_tol=1.0e-9):
            raise ValueError(f"Recomputed {metric} does not match frozen summary")

    plt.style.use(ROOT / "lefteris.mplstyle")
    fig, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    colored_curve(axis, smooth_x, smooth_y, centers)
    axis.errorbar(
        coordinate,
        pmf,
        yerr=np.vstack((pmf - lower, upper - pmf)),
        fmt="o",
        markersize=2.8,
        color="0.20",
        ecolor="0.35",
        elinewidth=0.75,
        capsize=1.2,
        alpha=0.75,
        label=f"WHAM; {100 * float(metadata['bootstrap']['confidence_level']):g}% block-bootstrap CI",
        zorder=3,
    )
    dg = integrated["population_delta_g_anti_minus_syn"]
    anti = integrated["anti_molar_fraction"]
    note = "\n" + str(metadata["sampling_note"]) if metadata.get("sampling_note") else ""
    axis.text(
        0.025,
        0.975,
        f"Integrated at φ = {float(metadata['dividing_surface_deg']):g}°\n"
        f"ΔG(anti−syn) = {float(dg['original']):.3f} ± {float(dg['bootstrap_std']):.3f} kcal mol⁻¹\n"
        f"95% CI: {float(dg['ci_lower']):.3f} to {float(dg['ci_upper']):.3f} kcal mol⁻¹\n"
        f"x(anti) = {100 * float(anti['original']):.5f}%{note}",
        transform=axis.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
    )
    axis.set(xlim=metadata["xlim_deg"], xlabel="ψ_cd dihedral (degrees)", ylabel="PMF (kcal mol⁻¹)")
    axis.set_title(str(metadata["title"]))
    axis.grid(False)
    axis.legend(loc="best", frameon=True, fontsize=8)
    scalar = ScalarMappable(
        norm=Normalize(vmin=float(centers.min()), vmax=float(centers.max())),
        cmap="turbo_r",
    )
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=axis, pad=0.015, fraction=0.04)
    colorbar.set_label("Umbrella center (degrees)")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def reproduce_overlap(metadata: dict[str, object], output: Path) -> None:
    overlap = ROOT / "data/window-overlap"
    entries: list[tuple[Path, float]] = []
    for line in (overlap / "metadata.dat").read_text().splitlines():
        fields = line.split()
        if fields:
            entries.append((overlap / fields[0], float(fields[1])))
    centers = np.asarray([center for _, center in entries])
    bounds = np.asarray(metadata["overlap_xlim_deg"], dtype=float)
    width = float(metadata["overlap_bin_width_deg"])
    edges = np.linspace(bounds[0], bounds[1], int(round(np.ptp(bounds) / width)) + 1)
    x = 0.5 * (edges[:-1] + edges[1:])
    cmap = plt.get_cmap("turbo_r")
    norm = Normalize(vmin=float(centers.min()), vmax=float(centers.max()))
    exported: list[tuple[int, float, float, float]] = []
    plt.style.use(ROOT / "lefteris.mplstyle")
    fig, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    for index, ((path, center), expected_index) in enumerate(zip(entries, metadata["window_indices"], strict=True)):
        values = np.loadtxt(path, comments="#", ndmin=2)
        density, _ = np.histogram(values[:, 1], bins=edges, density=True)
        axis.plot(x, density, color=cmap(norm(center)), linewidth=1.7, alpha=0.92)
        exported.extend((int(expected_index), center, coordinate, value) for coordinate, value in zip(x, density, strict=True))
    with (ROOT / "data/window_overlap_distributions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window_index", "center_deg", "dihedral_deg", "density_per_degree"])
        writer.writerows(exported)
    axis.set(xlim=bounds, xlabel="ψ_cd dihedral (degrees)", ylabel="Normalized density (degree⁻¹)")
    axis.set_title(f"{metadata['system']} extended umbrella-window overlap")
    axis.grid(False)
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=axis, pad=0.015, fraction=0.04)
    colorbar.set_label("Umbrella center (degrees)")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "pmf.png")
    parser.add_argument("--overlap-out", type=Path, default=ROOT / "window_overlap.png")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "metadata.json").read_text())
    reproduce_pmf(metadata, args.out)
    reproduce_overlap(metadata, args.overlap_out)
    print(args.out)
    print(args.overlap_out)


if __name__ == "__main__":
    main()
