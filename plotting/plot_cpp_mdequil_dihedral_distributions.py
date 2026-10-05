#!/usr/bin/env python3
"""Plot stacked normalized dihedral histograms from CPP Amber mdequil."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class DihedralSeries:
    label: str
    path: Path
    frames: np.ndarray
    angles_deg: np.ndarray


ORDER = ("double5", "double10", "double15", "single5", "single10", "single15")
COLORS = ("#D62728", "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E", "#17BECF")
ATOM_DEFINITIONS = {
    "double5": "NA–C4A–C5–C1B",
    "single5": "NB–C1B–C5–C4A",
    "single10": "NB–C4B–C10–C1C",
    "double10": "C4B–C10–C1C–NC",
    "single15": "NC–C4C–C15–C1D",
    "double15": "C4C–C15–C1D–ND",
}
SYMBOLS = {
    "double5": r"$\phi_{ab}$",
    "single5": r"$\psi_{ab}$",
    "single10": r"$\psi_{bc}$",
    "double10": r"$\phi_{bc}$",
    "single15": r"$\psi_{cd}$",
    "double15": r"$\phi_{cd}$",
}
BOND_STATES = {
    "double5": "Z",
    "double10": "Z",
    "double15": "Z",
    "single5": "s",
    "single10": "s",
    "single15": "s",
}


def read_series(directory: Path, label: str) -> DihedralSeries:
    path = directory / f"dih_{label}.dat"
    table = np.loadtxt(path, comments="#", dtype=float)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f"Expected frame and angle columns in {path}")
    frames, angles = table[:, 0], table[:, 1]
    finite = np.isfinite(frames) & np.isfinite(angles)
    frames, angles = frames[finite], angles[finite]
    if not angles.size:
        raise ValueError(f"No finite dihedral values in {path}")
    return DihedralSeries(label, path, frames, angles)


def read_pooled_series(
    runs_path: Path, label: str, run_ids: list[int]
) -> tuple[DihedralSeries, list[tuple]]:
    """Pool available signed-angle samples and retain per-run provenance."""
    frame_parts: list[np.ndarray] = []
    angle_parts: list[np.ndarray] = []
    provenance: list[tuple] = []
    for run_id in run_ids:
        path = runs_path / f"run-{run_id}" / "equil/analysis" / f"dih_{label}.dat"
        item = read_series(path.parent, label)
        signed = (item.angles_deg + 180.0) % 360.0 - 180.0
        frame_parts.append(item.frames)
        angle_parts.append(signed)
        provenance.append((label, run_id, path, len(signed)))
    return (
        DihedralSeries(
            label, runs_path, np.concatenate(frame_parts), np.concatenate(angle_parts)
        ),
        provenance,
    )


def common_edges(
    series: list[DihedralSeries], bin_width_deg: float, padding_deg: float
) -> np.ndarray:
    minimum = min(float(np.min(item.angles_deg)) for item in series) - padding_deg
    maximum = max(float(np.max(item.angles_deg)) for item in series) + padding_deg
    lower = bin_width_deg * np.floor(minimum / bin_width_deg)
    upper = bin_width_deg * np.ceil(maximum / bin_width_deg)
    return np.arange(lower, upper + 0.5 * bin_width_deg, bin_width_deg)


def circular_statistics(angles_deg: np.ndarray) -> tuple[float, float, float]:
    """Return circular mean, circular SD, and mean resultant length."""
    radians = np.deg2rad(angles_deg)
    resultant = np.mean(np.exp(1j * radians))
    length = float(np.abs(resultant))
    mean = float(np.rad2deg(np.angle(resultant)))
    safe_length = min(1.0, max(length, np.finfo(float).tiny))
    std = float(np.rad2deg(np.sqrt(-2.0 * np.log(safe_length))))
    return mean, std, length


def write_data(
    path: Path,
    series: list[DihedralSeries],
    edges: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dihedral", "source", "sample_count", "bin_left_deg", "bin_right_deg",
            "bin_center_deg", "count", "probability_density_per_deg",
        ])
        width = np.diff(edges)
        for item in series:
            counts, _ = np.histogram(item.angles_deg, bins=edges)
            density = counts / (counts.sum() * width)
            centers = 0.5 * (edges[:-1] + edges[1:])
            writer.writerows(
                (
                    item.label, item.path, len(item.angles_deg), left, right,
                    center, int(count), probability,
                )
                for left, right, center, count, probability in zip(
                    edges[:-1], edges[1:], centers, counts, density
                )
            )


def draw(
    series: list[DihedralSeries],
    edges: np.ndarray,
    output: Path,
    title: str,
) -> None:
    plt.style.use(ROOT / "plotting/lefteris.mplstyle")
    fig, histogram_axes = plt.subplots(
        len(series), 1, figsize=(8.2, 10.5), sharex=True,
        gridspec_kw={"hspace": 0.11},
    )
    width = np.diff(edges)
    for index, (axis, item, color) in enumerate(zip(histogram_axes, series, COLORS)):
        counts, _ = np.histogram(item.angles_deg, bins=edges)
        density = counts / (counts.sum() * width)
        mean, std, _resultant = circular_statistics(item.angles_deg)
        axis.stairs(density, edges, color=color, linewidth=2.4, fill=True, alpha=.22)
        axis.stairs(density, edges, color=color, linewidth=2.4)
        axis.text(
            .985, .88,
            item.label + "  " + BOND_STATES[item.label] + "  "
            + SYMBOLS[item.label] + "\n"
            + rf"$\mu={mean:.1f}^\circ$, $\sigma={std:.1f}^\circ$",
            transform=axis.transAxes,
            ha="right", va="top", color=color, fontsize=15,
        )
        axis.text(
            .015, .88, ATOM_DEFINITIONS[item.label], transform=axis.transAxes,
            ha="left", va="top", color=color, fontsize=14,
        )
        axis.set_ylim(bottom=0)
        axis.grid(False)
        axis.tick_params(axis="x", labelbottom=axis is histogram_axes[-1])

    histogram_axes[-1].set_xlabel("Dihedral angle (deg)")
    histogram_axes[-1].set_xlim(edges[0], edges[-1])
    fig.supylabel(r"$P(\phi)$ (deg$^{-1}$)", x=.02)
    sample_counts = [len(item.angles_deg) for item in series]
    if len(set(sample_counts)) == 1:
        count_text = (
            f"N = {sample_counts[0]:,} frames per dihedral; "
            f"{sum(sample_counts):,} dihedral values total"
        )
    else:
        count_text = f"{sum(sample_counts):,} dihedral values total"
    fig.suptitle(title + "\n" + count_text, y=.995)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input-dir", type=Path,
        default=None,
        help="single directory containing dih_*.dat files",
    )
    source.add_argument(
        "--runs-path", type=Path,
        default=None,
        help="directory containing run-*/equil/analysis/dih_*.dat files to pool",
    )
    parser.add_argument(
        "--output", type=Path,
        default=None,
    )
    parser.add_argument("--run-ids", default="1-10")
    parser.add_argument("--title", default=None)
    parser.add_argument("--bin-width-deg", type=float, default=1.0)
    parser.add_argument("--padding-deg", type=float, default=10.0)
    args = parser.parse_args()
    if args.bin_width_deg <= 0.0:
        parser.error("--bin-width-deg must be positive")
    if args.padding_deg < 0.0:
        parser.error("--padding-deg cannot be negative")
    return args


def parse_run_ids(text: str) -> list[int]:
    if "-" in text:
        first, last = (int(value) for value in text.split("-", 1))
        if first > last:
            raise ValueError("run range must be ascending")
        return list(range(first, last + 1))
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def main() -> None:
    args = parse_args()
    provenance: list[tuple] = []
    if args.runs_path is not None:
        run_ids = parse_run_ids(args.run_ids)
        runs_path = args.runs_path.resolve()
        pooled = [read_pooled_series(runs_path, label, run_ids) for label in ORDER]
        series = [item for item, _rows in pooled]
        provenance = [row for _item, rows in pooled for row in rows]
        output = (
            args.output.resolve() if args.output is not None
            else ROOT / "reports/CPP/dftb_equil_dihedral_distributions.png"
        )
        title = args.title or (
            f"CPP · DFTB equilibration · pooled runs {run_ids[0]}–{run_ids[-1]}"
            " · normalized dihedral distributions"
        )
    else:
        input_dir = (
            args.input_dir.resolve() if args.input_dir is not None
            else ROOT / "systems/CPP/solv_4.0/mdequil"
        )
        series = [read_series(input_dir, label) for label in ORDER]
        output = (
            args.output.resolve() if args.output is not None
            else ROOT / "reports/CPP/mdequil_dihedral_distributions.png"
        )
        title = args.title or (
            "CPP · Amber NPT equilibration · normalized dihedral distributions"
        )
    edges = common_edges(series, args.bin_width_deg, args.padding_deg)
    draw(series, edges, output, title)
    write_data(output.with_name(f"{output.stem}_data.csv"), series, edges)
    statistics_path = output.with_name(f"{output.stem}_statistics.csv")
    with statistics_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dihedral", "sample_count", "circular_mean_deg",
            "circular_std_deg", "mean_resultant_length",
        ])
        for item in series:
            mean, std, resultant = circular_statistics(item.angles_deg)
            writer.writerow([item.label, len(item.angles_deg), mean, std, resultant])
    if provenance:
        provenance_path = output.with_name(f"{output.stem}_provenance.csv")
        with provenance_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dihedral", "run", "source", "sample_count"])
            writer.writerows(provenance)
        first_label = series[0].label
        per_run_counts = [int(row[3]) for row in provenance if row[0] == first_label]
        summary_path = output.with_name(f"{output.stem}_sampling_summary.csv")
        with summary_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "run_count", "dihedral_count", "frames_per_dihedral",
                "total_dihedral_values", "minimum_frames_per_run",
                "maximum_frames_per_run",
            ])
            writer.writerow([
                len(per_run_counts), len(series), len(series[0].angles_deg),
                sum(len(item.angles_deg) for item in series),
                min(per_run_counts), max(per_run_counts),
            ])
    print(output)


if __name__ == "__main__":
    main()
