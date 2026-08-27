#!/usr/bin/env python3
"""Calculate sampled radial ion-pair PMFs from aligned defect distances.

This analysis is deliberately independent of hydrogen-bond water-wire state.
It consumes the ``*_Odefect_distance_A`` column already written to a reaction
summary CSV, constructs the sampled radial probability, removes the spherical
shell Jacobian, and reports ``W(r) = -k_B T ln g(r)`` up to an additive
constant.

The resulting profile describes the ensemble visited during metadynamics.  It
is not an unbiased equilibrium PMF because no time-dependent bias reweighting
is applied.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


KB_KCAL_MOL_K = 0.00198720425864083
INPUT_RE = re.compile(r"^(?P<label>[^=]+)=(?P<path>.+)$")
WINDOW_RE = re.compile(
    r"^(?P<label>[^:]+):(?P<start>[+-]?(?:\d+(?:\.\d*)?|\.\d+)):"
    r"(?P<end>[+-]?(?:\d+(?:\.\d*)?|\.\d+))$"
)


@dataclass(frozen=True)
class InputSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class RadialProfile:
    label: str
    source: Path
    distance_column: str
    time_min_ps: float
    time_max_ps: float
    sample_count: int
    distances_A: np.ndarray
    times_ps: np.ndarray
    bin_centers_A: np.ndarray
    bin_counts: np.ndarray
    radial_probability: np.ndarray
    shell_volume_A3: np.ndarray
    g_relative: np.ndarray
    binned_pmf_kcal_mol: np.ndarray
    grid_A: np.ndarray
    kde_probability_density_Ainv: np.ndarray
    kde_g_relative: np.ndarray
    kde_pmf_kcal_mol: np.ndarray
    kde_supported: np.ndarray
    minima_indices: tuple[int, ...]
    maxima_indices: tuple[int, ...]
    basin_probabilities: tuple[float, ...]
    transition_count: int
    kde_bandwidth_A: float
    wall_grid_A: np.ndarray
    wall_pmf_kcal_mol: np.ndarray


def parse_input(value: str) -> InputSpec:
    """Parse ``LABEL=PATH`` CLI syntax."""
    match = INPUT_RE.match(value)
    if match is None or not match.group("label").strip():
        raise argparse.ArgumentTypeError("input must have the form LABEL=PATH")
    return InputSpec(match.group("label").strip(), Path(match.group("path")))


def parse_window(value: str) -> tuple[str, float, float]:
    """Parse ``LABEL:START_PS:END_PS`` CLI syntax."""
    match = WINDOW_RE.match(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "window must have the form LABEL:START_PS:END_PS"
        )
    label = match.group("label").strip()
    start = float(match.group("start"))
    end = float(match.group("end"))
    if not start < end:
        raise argparse.ArgumentTypeError("window start must be smaller than end")
    return label, start, end


def distance_axis_label(distance_column: str) -> str:
    """Build a user-facing one-based solute--defect distance label."""
    if distance_column == "closest_solute_heavy_distance_A":
        return "closest solute heavy atom–O(defect⁺) distance, r (Å)"
    atom_label = distance_column.split("_Odefect_distance_A", maxsplit=1)[0]
    return f"{atom_label}–O(defect⁺) distance, r (Å)"


def load_distance_csv(
    path: Path,
    distance_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load time and an unmodified N--defect distance series from CSV."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {path}")
        fields = reader.fieldnames
        if "time_ps" not in fields:
            raise ValueError(f"Missing time_ps column in {path}")
        if distance_column is None:
            candidates = [name for name in fields if name.endswith("_Odefect_distance_A")]
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected exactly one *_Odefect_distance_A column in {path}; "
                    f"found {candidates}. Use --distance-column explicitly."
                )
            selected = candidates[0]
        else:
            selected = distance_column
            if selected not in fields:
                raise ValueError(f"Missing {selected} column in {path}")

        times: list[float] = []
        distances: list[float] = []
        for row in reader:
            try:
                time = float(row["time_ps"])
            except (TypeError, ValueError):
                time = float("nan")
            try:
                distance = float(row[selected])
            except (TypeError, ValueError):
                distance = float("nan")
            times.append(time)
            distances.append(distance)
    return np.asarray(times), np.asarray(distances), selected


def select_samples(
    times_ps: np.ndarray,
    distances_A: np.ndarray,
    time_min_ps: float | None,
    time_max_ps: float | None,
    radial_min_A: float,
    radial_max_A: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Select finite, observed distances without interpolation or imputation."""
    finite = np.isfinite(times_ps) & np.isfinite(distances_A)
    if time_min_ps is not None:
        finite &= times_ps >= time_min_ps
    if time_max_ps is not None:
        finite &= times_ps <= time_max_ps
    finite &= (distances_A >= radial_min_A) & (distances_A <= radial_max_A)
    if not np.any(finite):
        raise ValueError("No finite defect distances remain in the requested window")
    selected_times = times_ps[finite]
    selected_distances = distances_A[finite]
    actual_min = float(time_min_ps if time_min_ps is not None else np.min(selected_times))
    actual_max = float(time_max_ps if time_max_ps is not None else np.max(selected_times))
    return selected_times, selected_distances, actual_min, actual_max


def _relative_g_and_pmf(
    probability: np.ndarray,
    shell_measure: np.ndarray,
    temperature_K: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized relative g and a minimum-shifted PMF."""
    g = np.full(probability.shape, np.nan, dtype=float)
    valid = np.isfinite(probability) & (probability > 0.0) & (shell_measure > 0.0)
    g[valid] = probability[valid] / shell_measure[valid]
    if not np.any(valid):
        return g, np.full(g.shape, np.nan)
    # Only the additive PMF constant depends on this normalization.
    g[valid] /= float(np.nanmax(g[valid]))
    pmf = np.full(g.shape, np.nan, dtype=float)
    pmf[valid] = -KB_KCAL_MOL_K * temperature_K * np.log(g[valid])
    pmf[valid] -= float(np.nanmin(pmf[valid]))
    return g, pmf


def gaussian_kde_1d(
    samples: np.ndarray,
    grid: np.ndarray,
    bandwidth_factor: float,
    fallback_bandwidth_A: float,
) -> tuple[np.ndarray, float]:
    """Evaluate a Gaussian KDE using a sample-standard-deviation factor."""
    if bandwidth_factor <= 0.0:
        raise ValueError("KDE bandwidth factor must be positive")
    scale = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
    bandwidth_A = bandwidth_factor * scale
    if not np.isfinite(bandwidth_A) or bandwidth_A <= 0.0:
        bandwidth_A = fallback_bandwidth_A
    scaled = (grid[:, None] - samples[None, :]) / bandwidth_A
    density = np.mean(np.exp(-0.5 * scaled * scaled), axis=1)
    density /= bandwidth_A * math.sqrt(2.0 * math.pi)
    return density, bandwidth_A


def _supported_grid(
    grid: np.ndarray,
    samples: np.ndarray,
    bin_edges: np.ndarray,
    bin_counts: np.ndarray,
    minimum_bin_count: int,
    bandwidth_A: float,
) -> np.ndarray:
    """Mark only grid points with both global and local sample support."""
    occupied = np.flatnonzero(bin_counts >= minimum_bin_count)
    if not occupied.size:
        return np.zeros(grid.shape, dtype=bool)
    inside_occupied_range = (grid >= bin_edges[int(occupied[0])]) & (
        grid <= bin_edges[int(occupied[-1]) + 1]
    )
    sorted_samples = np.sort(samples)
    bin_width_A = float(np.median(np.diff(bin_edges)))
    support_radius_A = max(bin_width_A, 2.0 * bandwidth_A)
    left = np.searchsorted(sorted_samples, grid - support_radius_A, side="left")
    right = np.searchsorted(sorted_samples, grid + support_radius_A, side="right")
    locally_supported = (right - left) >= minimum_bin_count
    return inside_occupied_range & locally_supported


def find_extrema(
    grid: np.ndarray,
    pmf: np.ndarray,
    supported: np.ndarray,
    minimum_separation_A: float = 0.35,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Find separated local minima and maxima in the supported smooth PMF."""
    valid_indices = np.flatnonzero(supported & np.isfinite(pmf))
    if valid_indices.size < 3:
        return (), ()
    start, stop = int(valid_indices[0]), int(valid_indices[-1])
    candidates_min: list[int] = []
    candidates_max: list[int] = []
    for index in range(start + 1, stop):
        left, center, right = pmf[index - 1 : index + 2]
        if not np.all(np.isfinite((left, center, right))):
            continue
        if center <= left and center < right:
            candidates_min.append(index)
        if center >= left and center > right:
            candidates_max.append(index)

    def separated(candidates: list[int], choose_minimum: bool) -> tuple[int, ...]:
        if not candidates:
            return ()
        groups: list[list[int]] = [[candidates[0]]]
        for index in candidates[1:]:
            if grid[index] - grid[groups[-1][-1]] < minimum_separation_A:
                groups[-1].append(index)
            else:
                groups.append([index])
        chooser = min if choose_minimum else max
        return tuple(
            chooser(group, key=lambda index: float(pmf[index])) for group in groups
        )

    minima = separated(candidates_min, True)
    maxima = separated(candidates_max, False)
    if minima:
        maxima = tuple(
            index
            for index in maxima
            if any(left < index < right for left, right in zip(minima, minima[1:]))
        )
    else:
        maxima = ()
    return minima, maxima


def basin_statistics(
    distances_A: np.ndarray,
    grid: np.ndarray,
    minima: Sequence[int],
    maxima: Sequence[int],
) -> tuple[tuple[float, ...], int]:
    """Return radial basin populations and hysteretic core-to-core transitions."""
    if not minima:
        return (), 0
    barriers = np.asarray([grid[index] for index in maxima], dtype=float)
    basin_ids = np.searchsorted(barriers, distances_A, side="right")
    probabilities = tuple(
        float(np.count_nonzero(basin_ids == index) / distances_A.size)
        for index in range(len(barriers) + 1)
    )
    if len(minima) < 2 or len(maxima) != len(minima) - 1:
        return probabilities, 0

    core_positions = np.asarray([grid[index] for index in minima], dtype=float)
    barrier_positions = np.asarray([grid[index] for index in maxima], dtype=float)
    core_lower = np.full(core_positions.shape, -np.inf)
    core_upper = np.full(core_positions.shape, np.inf)
    core_upper[:-1] = 0.5 * (core_positions[:-1] + barrier_positions)
    core_lower[1:] = 0.5 * (barrier_positions + core_positions[1:])
    state: int | None = None
    transitions = 0
    for distance in distances_A:
        matches = np.flatnonzero((distance >= core_lower) & (distance <= core_upper))
        if matches.size != 1:
            continue
        new_state = int(matches[0])
        if state is not None and new_state != state:
            transitions += 1
        state = new_state
    return probabilities, transitions


def calculate_profile(
    label: str,
    source: Path,
    times_ps: np.ndarray,
    distances_A: np.ndarray,
    distance_column: str,
    time_min_ps: float | None = None,
    time_max_ps: float | None = None,
    radial_min_A: float = 2.0,
    radial_max_A: float = 9.5,
    bin_width_A: float = 0.25,
    temperature_K: float = 300.0,
    kde_bandwidth_factor: float = 0.13,
    kde_grid_points: int = 1000,
    minimum_bin_count: int = 2,
    wall_sigma_A: float = 2.42,
    wall_anchor_A: float = 2.48,
    wall_scale_kcal_mol: float = 1.55,
) -> RadialProfile:
    """Calculate binned and smoothed radial PMFs for one run."""
    if not radial_min_A < radial_max_A:
        raise ValueError("Radial minimum must be smaller than radial maximum")
    if bin_width_A <= 0.0 or temperature_K <= 0.0:
        raise ValueError("Bin width and temperature must be positive")
    if kde_grid_points < 3 or minimum_bin_count < 1:
        raise ValueError("KDE grid points and minimum bin count are too small")
    if wall_sigma_A <= 0.0 or wall_scale_kcal_mol <= 0.0:
        raise ValueError("Wall sigma and scale must be positive")
    if not radial_min_A < wall_anchor_A < radial_max_A:
        raise ValueError("Wall anchor must lie inside the radial range")
    selected_times, selected_distances, actual_tmin, actual_tmax = select_samples(
        times_ps,
        distances_A,
        time_min_ps,
        time_max_ps,
        radial_min_A,
        radial_max_A,
    )
    bin_edges = np.arange(radial_min_A, radial_max_A + bin_width_A, bin_width_A)
    if bin_edges[-1] < radial_max_A:
        bin_edges = np.append(bin_edges, radial_max_A)
    else:
        bin_edges[-1] = radial_max_A
    counts, _ = np.histogram(selected_distances, bins=bin_edges)
    probability = counts.astype(float) / selected_distances.size
    shell_volume = (4.0 / 3.0) * math.pi * (
        bin_edges[1:] ** 3 - bin_edges[:-1] ** 3
    )
    g_relative, binned_pmf = _relative_g_and_pmf(
        probability, shell_volume, temperature_K
    )
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    grid = np.linspace(radial_min_A, radial_max_A, kde_grid_points)
    kde_density, bandwidth_A = gaussian_kde_1d(
        selected_distances, grid, kde_bandwidth_factor, bin_width_A
    )
    kde_g, kde_pmf = _relative_g_and_pmf(kde_density, grid * grid, temperature_K)
    wall_grid = np.array([], dtype=float)
    wall_pmf = np.array([], dtype=float)
    if float(np.min(selected_distances)) <= wall_anchor_A + bin_width_A:
        anchor_pmf = float(np.interp(wall_anchor_A, grid, kde_pmf))
        wall_grid = np.linspace(radial_min_A, wall_anchor_A, 160)
        wall_pmf = anchor_pmf + wall_scale_kcal_mol * (
            (wall_sigma_A / wall_grid) ** 12
            - (wall_sigma_A / wall_anchor_A) ** 12
        )
    supported = _supported_grid(
        grid,
        selected_distances,
        bin_edges,
        counts,
        minimum_bin_count,
        bandwidth_A,
    )
    # Follow the ion-pair note's continuous KDE construction between the
    # first and last observed distances. Internal gaps are deliberately
    # interpolated so that the sampled ion-pair basins remain visible.
    interpolation_range = (grid >= np.min(selected_distances)) & (
        grid <= np.max(selected_distances)
    )
    kde_g[~interpolation_range] = np.nan
    kde_pmf[~interpolation_range] = np.nan
    minima, maxima = find_extrema(grid, kde_pmf, interpolation_range)
    basin_probabilities, transition_count = basin_statistics(
        selected_distances, grid, minima, maxima
    )
    return RadialProfile(
        label=label,
        source=source,
        distance_column=distance_column,
        time_min_ps=actual_tmin,
        time_max_ps=actual_tmax,
        sample_count=int(selected_distances.size),
        distances_A=selected_distances,
        times_ps=selected_times,
        bin_centers_A=centers,
        bin_counts=counts,
        radial_probability=probability,
        shell_volume_A3=shell_volume,
        g_relative=g_relative,
        binned_pmf_kcal_mol=binned_pmf,
        grid_A=grid,
        kde_probability_density_Ainv=kde_density,
        kde_g_relative=kde_g,
        kde_pmf_kcal_mol=kde_pmf,
        kde_supported=supported,
        minima_indices=minima,
        maxima_indices=maxima,
        basin_probabilities=basin_probabilities,
        transition_count=transition_count,
        kde_bandwidth_A=bandwidth_A,
        wall_grid_A=wall_grid,
        wall_pmf_kcal_mol=wall_pmf,
    )


def _safe_label(label: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return value or "run"


def save_profile_data(profile: RadialProfile, out_dir: Path) -> tuple[Path, Path]:
    """Save the selected samples and every numerical series used in the plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_label(profile.label)
    samples_path = out_dir / f"{stem}_ion_pair_samples.csv"
    np.savetxt(
        samples_path,
        np.column_stack((profile.times_ps, profile.distances_A)),
        delimiter=",",
        header="time_ps,distance_A",
        comments="",
    )
    profile_path = out_dir / f"{stem}_ion_pair_pmf.csv"
    rows = max(
        profile.bin_centers_A.size,
        profile.grid_A.size,
        profile.wall_grid_A.size,
    )
    columns = np.full((rows, 14), np.nan)
    nbin = profile.bin_centers_A.size
    columns[:nbin, :6] = np.column_stack(
        (
            profile.bin_centers_A,
            profile.bin_counts,
            profile.radial_probability,
            profile.shell_volume_A3,
            profile.g_relative,
            profile.binned_pmf_kcal_mol,
        )
    )
    ngrid = profile.grid_A.size
    columns[:ngrid, 6:12] = np.column_stack(
        (
            profile.grid_A,
            profile.kde_probability_density_Ainv,
            profile.kde_g_relative,
            profile.kde_pmf_kcal_mol,
            profile.kde_supported.astype(int),
            np.full(ngrid, profile.kde_bandwidth_A),
        )
    )
    nwall = profile.wall_grid_A.size
    if nwall:
        columns[:nwall, 12:] = np.column_stack(
            (profile.wall_grid_A, profile.wall_pmf_kcal_mol)
        )
    np.savetxt(
        profile_path,
        columns,
        delimiter=",",
        header=(
            "bin_center_A,bin_count,radial_probability,shell_volume_A3,"
            "binned_g_relative,binned_pmf_kcal_mol,grid_A,kde_density_Ainv,"
            "kde_g_relative,kde_pmf_kcal_mol,kde_supported,kde_bandwidth_A"
            ",wall_grid_A,wall_pmf_kcal_mol"
        ),
        comments="",
    )
    return samples_path, profile_path


def save_summary(profiles: Sequence[RadialProfile], path: Path) -> None:
    """Save compact basin and sampling diagnostics for all runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "source_csv",
        "distance_column",
        "time_min_ps",
        "time_max_ps",
        "sample_count",
        "observed_min_A",
        "observed_max_A",
        "kde_bandwidth_A",
        "minima_A",
        "minima_W_kcal_mol",
        "maxima_A",
        "maxima_W_kcal_mol",
        "basin_probabilities",
        "core_transition_count",
        "modeled_wall_added",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            minima = profile.minima_indices
            maxima = profile.maxima_indices
            writer.writerow(
                {
                    "label": profile.label,
                    "source_csv": profile.source,
                    "distance_column": profile.distance_column,
                    "time_min_ps": f"{profile.time_min_ps:.8f}",
                    "time_max_ps": f"{profile.time_max_ps:.8f}",
                    "sample_count": profile.sample_count,
                    "observed_min_A": f"{np.min(profile.distances_A):.8f}",
                    "observed_max_A": f"{np.max(profile.distances_A):.8f}",
                    "kde_bandwidth_A": f"{profile.kde_bandwidth_A:.8f}",
                    "minima_A": ";".join(f"{profile.grid_A[i]:.8f}" for i in minima),
                    "minima_W_kcal_mol": ";".join(
                        f"{profile.kde_pmf_kcal_mol[i]:.8f}" for i in minima
                    ),
                    "maxima_A": ";".join(f"{profile.grid_A[i]:.8f}" for i in maxima),
                    "maxima_W_kcal_mol": ";".join(
                        f"{profile.kde_pmf_kcal_mol[i]:.8f}" for i in maxima
                    ),
                    "basin_probabilities": ";".join(
                        f"{value:.8f}" for value in profile.basin_probabilities
                    ),
                    "core_transition_count": profile.transition_count,
                    "modeled_wall_added": int(bool(profile.wall_grid_A.size)),
                }
            )


def plot_profiles(
    profiles: Sequence[RadialProfile],
    path: Path,
    layer_boundaries_A: Sequence[float] = (3.5, 5.5, 7.5),
    dpi: int = 220,
) -> None:
    """Plot shared-axis sampled radial PMFs as small multiples."""
    import matplotlib.pyplot as plt

    count = len(profiles)
    columns = 2 if count > 1 else 1
    rows = int(math.ceil(count / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.8 * columns, 3.7 * rows),
        dpi=dpi,
        sharex=True,
        sharey=False,
        squeeze=False,
        layout="constrained",
    )
    for axis, profile in zip(axes.flat, profiles):
        finite_bins = np.isfinite(profile.binned_pmf_kcal_mol)
        axis.scatter(
            profile.bin_centers_A[finite_bins],
            profile.binned_pmf_kcal_mol[finite_bins],
            color="#375A7F",
            s=20,
            alpha=0.75,
            label="0.25 Å bins",
            zorder=3,
        )
        axis.plot(
            profile.grid_A,
            profile.kde_pmf_kcal_mol,
            color="#A23B2A",
            linewidth=2.0,
            label="KDE",
            zorder=2,
        )
        if profile.wall_grid_A.size:
            axis.plot(
                profile.wall_grid_A,
                profile.wall_pmf_kcal_mol,
                color="0.35",
                linewidth=1.8,
                linestyle="--",
                label="modeled excluded-volume wall",
                zorder=1,
            )
        for boundary in layer_boundaries_A:
            axis.axvline(boundary, color="0.75", linestyle="--", linewidth=0.8)
        for index in profile.minima_indices:
            axis.scatter(
                profile.grid_A[index],
                profile.kde_pmf_kcal_mol[index],
                marker="v",
                color="#228833",
                edgecolor="black",
                linewidth=0.4,
                s=45,
                zorder=4,
            )
        for index in profile.maxima_indices:
            axis.scatter(
                profile.grid_A[index],
                profile.kde_pmf_kcal_mol[index],
                marker="^",
                color="#CC3311",
                edgecolor="black",
                linewidth=0.4,
                s=45,
                zorder=4,
            )
        axis.set_title(
            f"{profile.label}: N={profile.sample_count}, "
            f"{profile.time_min_ps:g}-{profile.time_max_ps:g} ps"
        )
        axis.grid(axis="y", alpha=0.22)
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    for column_index, axis in enumerate(axes[-1, :]):
        if axis.get_visible():
            profile_index = (rows - 1) * columns + column_index
            axis.set_xlabel(distance_axis_label(profiles[profile_index].distance_column))
    for axis in axes[:, 0]:
        if axis.get_visible():
            axis.set_ylabel("sampled radial PMF, W(r) (kcal/mol)")
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Ion-pair separation sampled during coordination-biased metadynamics",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate shell-corrected sampled radial PMFs from existing "
            "N--O(defect) distance CSVs."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_input,
        metavar="LABEL=CSV",
        help="Labeled input; repeat for multiple runs",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        type=parse_window,
        metavar="LABEL:TMIN:TMAX",
        help="Optional inclusive time window in ps; repeat per label",
    )
    parser.add_argument("--distance-column", default=None)
    parser.add_argument("--radial-min-A", type=float, default=2.0)
    parser.add_argument("--radial-max-A", type=float, default=9.5)
    parser.add_argument("--bin-width-A", type=float, default=0.25)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument(
        "--kde-bandwidth-factor",
        type=float,
        default=0.13,
        help="Gaussian bandwidth as a factor of the sample standard deviation",
    )
    parser.add_argument("--kde-grid-points", type=int, default=1000)
    parser.add_argument("--minimum-bin-count", type=int, default=2)
    parser.add_argument("--wall-sigma-A", type=float, default=2.42)
    parser.add_argument("--wall-anchor-A", type=float, default=2.48)
    parser.add_argument("--wall-scale-kcal-mol", type=float, default=1.55)
    parser.add_argument(
        "--layer-boundaries-A", type=float, nargs="*", default=(3.5, 5.5, 7.5)
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--figure-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    inputs: list[InputSpec] = args.input
    labels = [item.label for item in inputs]
    if len(labels) != len(set(labels)):
        parser.error("--input labels must be unique")
    windows = {label: (start, end) for label, start, end in args.window}
    unknown_windows = sorted(set(windows) - set(labels))
    if unknown_windows:
        parser.error(f"--window labels have no matching input: {unknown_windows}")

    profiles: list[RadialProfile] = []
    for item in inputs:
        times, distances, column = load_distance_csv(item.path, args.distance_column)
        bounds = windows.get(item.label, (None, None))
        profile = calculate_profile(
            item.label,
            item.path,
            times,
            distances,
            column,
            time_min_ps=bounds[0],
            time_max_ps=bounds[1],
            radial_min_A=args.radial_min_A,
            radial_max_A=args.radial_max_A,
            bin_width_A=args.bin_width_A,
            temperature_K=args.temperature_K,
            kde_bandwidth_factor=args.kde_bandwidth_factor,
            kde_grid_points=args.kde_grid_points,
            minimum_bin_count=args.minimum_bin_count,
            wall_sigma_A=args.wall_sigma_A,
            wall_anchor_A=args.wall_anchor_A,
            wall_scale_kcal_mol=args.wall_scale_kcal_mol,
        )
        profiles.append(profile)
        samples_path, data_path = save_profile_data(profile, args.out_dir)
        print(f"Wrote {samples_path}")
        print(f"Wrote {data_path}")

    figure_out = args.figure_out or args.out_dir / "ion_pair_sampled_pmf.png"
    summary_out = args.summary_out or args.out_dir / "ion_pair_sampled_pmf_summary.csv"
    plot_profiles(profiles, figure_out, args.layer_boundaries_A, args.dpi)
    save_summary(profiles, summary_out)
    print(f"Wrote {figure_out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
