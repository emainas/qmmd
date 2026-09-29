#!/usr/bin/env python3
"""Compare protonated-BV helicity distributions from classical MD force fields."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
R_KCAL_MOL_K = 0.00198720425864083


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    directory: str
    label: str
    color: str


SERIES = (
    SeriesSpec("ff-opt", "FF-opt", "#D62728"),
    SeriesSpec("gaff2", "GAFF2", "#1F77B4"),
    SeriesSpec("sage", "Sage", "#2CA02C"),
)


@dataclass(frozen=True, slots=True)
class HelicityDistribution:
    spec: SeriesSpec
    source: Path
    temperature_k: float
    times_ps: np.ndarray
    helicity_deg: np.ndarray
    edges_deg: np.ndarray
    counts: np.ndarray
    density_per_deg: np.ndarray
    negative_log_relative_probability: np.ndarray
    pmf_kcal_mol: np.ndarray

    @property
    def centers_deg(self) -> np.ndarray:
        return 0.5 * (self.edges_deg[:-1] + self.edges_deg[1:])


def read_helicity(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column time/helicity table and validate its finite samples."""
    table = np.loadtxt(path, comments="#", dtype=float)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}")
    times = table[:, 0]
    helicity = table[:, 1]
    finite = np.isfinite(times) & np.isfinite(helicity)
    times, helicity = times[finite], helicity[finite]
    if times.size == 0:
        raise ValueError(f"No finite helicity samples in {path}")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Times are not strictly increasing in {path}")
    if np.any((helicity < -180.0) | (helicity > 180.0)):
        raise ValueError(f"Helicity lies outside [-180, 180] degrees in {path}")
    return times, helicity


def calculate_distribution(
    input_dir: Path, spec: SeriesSpec, edges_deg: np.ndarray, temperature_k: float
) -> HelicityDistribution:
    source = input_dir / spec.directory / "helicity.dat"
    times, helicity = read_helicity(source)
    counts, returned_edges = np.histogram(helicity, bins=edges_deg)
    widths = np.diff(returned_edges)
    density = counts.astype(float) / (float(np.sum(counts)) * widths)
    probability = counts.astype(float) / float(np.sum(counts))
    relative = np.full(probability.shape, np.nan, dtype=float)
    occupied = probability > 0.0
    relative[occupied] = -np.log(probability[occupied] / np.max(probability))
    return HelicityDistribution(
        spec=spec,
        source=source,
        temperature_k=temperature_k,
        times_ps=times,
        helicity_deg=helicity,
        edges_deg=returned_edges,
        counts=counts,
        density_per_deg=density,
        negative_log_relative_probability=relative,
        pmf_kcal_mol=R_KCAL_MOL_K * temperature_k * relative,
    )


def write_plot_data(path: Path, distributions: list[HelicityDistribution]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "force_field", "source", "bin_left_deg", "bin_right_deg", "bin_center_deg",
            "count", "probability_density_per_deg", "negative_ln_p_over_pmax",
            "pmf_kcal_mol",
        ])
        for distribution in distributions:
            writer.writerows(
                (
                    distribution.spec.label,
                    distribution.source,
                    left,
                    right,
                    center,
                    int(count),
                    density,
                    profile,
                    pmf,
                )
                for left, right, center, count, density, profile, pmf in zip(
                    distribution.edges_deg[:-1],
                    distribution.edges_deg[1:],
                    distribution.centers_deg,
                    distribution.counts,
                    distribution.density_per_deg,
                    distribution.negative_log_relative_probability,
                    distribution.pmf_kcal_mol,
                )
            )


def write_summary(path: Path, distributions: list[HelicityDistribution]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "force_field", "source", "sample_count", "first_time_ps", "last_time_ps",
            "sample_spacing_ps", "sampled_duration_ps", "minimum_helicity_deg",
            "maximum_helicity_deg", "temperature_k", "bin_count", "bin_width_deg",
        ])
        for distribution in distributions:
            times = distribution.times_ps
            writer.writerow([
                distribution.spec.label,
                distribution.source,
                len(times),
                times[0],
                times[-1],
                float(np.median(np.diff(times))),
                times[-1] - times[0],
                float(np.min(distribution.helicity_deg)),
                float(np.max(distribution.helicity_deg)),
                distribution.temperature_k,
                len(distribution.counts),
                distribution.edges_deg[1] - distribution.edges_deg[0],
            ])


def draw(distributions: list[HelicityDistribution], output: Path) -> None:
    plt.style.use(ROOT / "plotting/lefteris.mplstyle")
    fig, (ax_histogram, ax_profile) = plt.subplots(
        1, 2, figsize=(12, 5.3), sharex=True, layout="constrained"
    )

    for distribution in distributions:
        spec = distribution.spec
        ax_histogram.stairs(
            distribution.density_per_deg,
            distribution.edges_deg,
            color=spec.color,
            linewidth=2.3,
            label=spec.label,
        )
        ax_profile.plot(
            distribution.centers_deg,
            distribution.pmf_kcal_mol,
            color=spec.color,
            linewidth=2.3,
            label=spec.label,
        )

    ax_histogram.set(
        title="Normalized helicity histogram",
        xlabel="Helicity (deg)",
        ylabel=r"$P(h)$ (deg$^{-1}$)",
        xlim=(-75, 75),
        ylim=(0, None),
    )
    ax_profile.set(
        title="Helicity PMF",
        xlabel="Helicity (deg)",
        ylabel=r"$-RT\ln[P(h)/P_{\max}]$ (kcal mol$^{-1}$)",
        xlim=(-75, 75),
        ylim=(0, 4),
    )
    for axis in (ax_histogram, ax_profile):
        axis.legend(loc="best")
        axis.grid(False)

    fig.suptitle(r"Protonated BV · 1 $\mu$s classical MD in TIP3P water", fontsize=20)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "systems/BV/CLASSICAL-MD-DATA-OLD",
        help="directory containing ff-opt, gaff2, and sage helicity.dat files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/BV/classical_md_helicity.png",
        help="output PNG; CSV files are written beside it using the same stem",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=720,
        help="number of equal-width bins across [-180, 180] degrees (default: 720)",
    )
    parser.add_argument(
        "--temperature-k",
        type=float,
        default=300.0,
        help="temperature used to convert -ln(P/Pmax) to kcal/mol (default: 300)",
    )
    args = parser.parse_args()
    if args.bins < 4:
        parser.error("--bins must be at least 4")
    if args.temperature_k <= 0.0:
        parser.error("--temperature-k must be positive")
    return args


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output = args.output.resolve()
    edges = np.linspace(-180.0, 180.0, args.bins + 1)
    distributions = [
        calculate_distribution(input_dir, spec, edges, args.temperature_k)
        for spec in SERIES
    ]
    draw(distributions, output)
    write_plot_data(output.with_name(f"{output.stem}_data.csv"), distributions)
    write_summary(output.with_name(f"{output.stem}_summary.csv"), distributions)
    print(output)


if __name__ == "__main__":
    main()
