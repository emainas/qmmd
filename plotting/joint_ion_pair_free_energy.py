#!/usr/bin/env python3
"""Construct TP-reweighted coordination--ion-pair free energies for HIST runs.

Every aligned frame is reweighted with eqs 13 and 15 of Tiwary and
Parrinello, J. Phys. Chem. B 119, 736--742 (2015).  An optional radial d**2
correction converts a distance-coordinate free energy into a pair PMF.
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

from ion_pair_pmf import (
    KB_KCAL_MOL_K,
    InputSpec,
    RadialProfile,
    calculate_profile,
    distance_axis_label,
    load_distance_csv,
    parse_input,
    parse_window,
)


EH_TO_KCAL_MOL = 627.509474
LABELED_PATH_RE = re.compile(r"^(?P<label>[^=]+)=(?P<path>.+)$")
TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


@dataclass(frozen=True)
class JointProfile:
    label: str
    source_csv: Path
    fes_path: Path
    sample_count: int
    time_min_ps: float
    time_max_ps: float
    times_ps: np.ndarray
    bias_kcal_mol: np.ndarray
    offset_kcal_mol: np.ndarray
    log_weights: np.ndarray
    normalized_weights: np.ndarray
    effective_sample_size: float
    s_samples: np.ndarray
    distance_samples_A: np.ndarray
    s_grid: np.ndarray
    distance_grid_A: np.ndarray
    fes_s_kcal_mol: np.ndarray
    equilibrium_s_probability: np.ndarray
    sampled_joint_density: np.ndarray
    conditional_distance_density: np.ndarray
    equilibrium_joint_probability: np.ndarray
    joint_free_energy_kcal_mol: np.ndarray
    distance_free_energy_kcal_mol: np.ndarray
    supported: np.ndarray
    radial_profile: RadialProfile
    bandwidth_s: float
    bandwidth_distance_A: float
    apply_radial_jacobian: bool


@dataclass(frozen=True)
class MetadynamicsHills:
    times_ps: np.ndarray
    centers: np.ndarray
    heights_kcal_mol: np.ndarray
    widths: np.ndarray


def load_metadynamics_hills(path: Path) -> MetadynamicsHills:
    """Parse one-dimensional standard-metadynamics hills from biaspot."""
    times: list[float] = []
    centers: list[float] = []
    heights: list[float] = []
    widths: list[float] = []
    current_time: float | None = None
    current_height: float | None = None
    current_center: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        time_match = TIME_RE.search(line)
        if time_match:
            current_time = float(time_match.group(1)) / 1000.0
        elif "Gaussian height" in line:
            current_height = (
                float(line.split("=")[-1].split()[0]) * EH_TO_KCAL_MOL
            )
        elif "Coordinate" in line:
            current_center = float(line.split("=")[-1])
        elif "Gaussian width" in line:
            if current_time is None or current_height is None or current_center is None:
                raise ValueError(f"Incomplete hill preceding line {line!r} in {path}")
            times.append(current_time)
            heights.append(current_height)
            centers.append(current_center)
            widths.append(float(line.split("=")[-1]))
            current_time = current_height = current_center = None
    if not times:
        raise ValueError(f"No metadynamics hills found in {path}")
    order = np.argsort(times)
    return MetadynamicsHills(
        times_ps=np.asarray(times)[order],
        centers=np.asarray(centers)[order],
        heights_kcal_mol=np.asarray(heights)[order],
        widths=np.asarray(widths)[order],
    )


def tiwary_parrinello_weights(
    times_ps: np.ndarray,
    s_samples: np.ndarray,
    hills: MetadynamicsHills,
    temperature_K: float,
    s_min: float,
    s_max: float,
    integration_points: int = 1001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return standard-metaD weights from TP eqs 13 and 15.

    For one biased CV, eq 13 is

        exp(beta*c(t)) = integral ds exp(beta*V(s,t))*g(s-s(t))
                         / (sqrt(2*pi)*sigma),

    where ``g`` is the (unnormalized) Gaussian deposition kernel.  Equation
    15 then gives the frame weight exp(beta*(V(s(t),t)-c(t))).  The current
    implementation supports the constant Gaussian width used by these runs.
    """
    if len(times_ps) != len(s_samples):
        raise ValueError("Times and coordination samples must be aligned")
    beta = 1.0 / (KB_KCAL_MOL_K * temperature_K)
    integration_grid = np.linspace(s_min, s_max, integration_points)
    if integration_points < 2:
        raise ValueError("TP integration requires at least two grid points")
    if not s_max > s_min:
        raise ValueError("s_max must exceed s_min")
    if np.any(hills.widths <= 0.0):
        raise ValueError("Metadynamics Gaussian widths must be positive")
    sigma = float(hills.widths[0])
    if not np.allclose(hills.widths, sigma, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("TP equation 13 currently requires a constant hill width")
    bias_grid = np.zeros_like(integration_grid)
    bias_values = np.zeros_like(s_samples)
    offsets = np.zeros_like(s_samples)
    chronological = np.argsort(times_ps)
    hill_index = 0
    for sample_index in chronological:
        time_ps = times_ps[sample_index]
        while hill_index < hills.times_ps.size and hills.times_ps[hill_index] <= time_ps:
            scaled = (integration_grid - hills.centers[hill_index]) / hills.widths[hill_index]
            bias_grid += hills.heights_kcal_mol[hill_index] * np.exp(-0.5 * scaled * scaled)
            hill_index += 1
        active = slice(0, hill_index)
        if hill_index:
            scaled_sample = (
                s_samples[sample_index] - hills.centers[active]
            ) / hills.widths[active]
            bias_values[sample_index] = float(
                np.sum(hills.heights_kcal_mol[active] * np.exp(-0.5 * scaled_sample**2))
            )
        # TP eq 13: Gaussian-localized integral, not a uniform CV-space mean.
        gaussian = np.exp(
            -0.5 * ((integration_grid - s_samples[sample_index]) / sigma) ** 2
        )
        exponent = beta * bias_grid
        maximum = float(np.max(exponent))
        scaled_integral = float(
            TRAPEZOID(np.exp(exponent - maximum) * gaussian, integration_grid)
        )
        if scaled_integral <= 0.0:
            raise ValueError("Non-positive TP equation-13 integral")
        log_exp_beta_c = (
            maximum
            + math.log(scaled_integral)
            - math.log(math.sqrt(2.0 * math.pi) * sigma)
        )
        offsets[sample_index] = log_exp_beta_c / beta
    log_weights = beta * (bias_values - offsets)
    shifted = log_weights - float(np.max(log_weights))
    normalized = np.exp(shifted)
    normalized /= float(np.sum(normalized))
    effective_sample_size = 1.0 / float(np.sum(normalized**2))
    return bias_values, offsets, log_weights, normalized, effective_sample_size


def locate_basin_minima(
    profile: JointProfile,
    energy_ceiling_kcal_mol: float,
    max_basins: int = 8,
    minimum_s_separation: float = 0.10,
    minimum_distance_separation_A: float = 0.50,
) -> list[tuple[float, float, float]]:
    """Locate separated local minima on the supported joint-FES grid."""
    energy = profile.joint_free_energy_kcal_mol
    finite = np.isfinite(energy) & (energy <= energy_ceiling_kcal_mol)
    candidates: list[tuple[float, int, int]] = []
    for i, j in np.argwhere(finite):
        i0, i1 = max(0, i - 1), min(energy.shape[0], i + 2)
        j0, j1 = max(0, j - 1), min(energy.shape[1], j + 2)
        neighborhood = energy[i0:i1, j0:j1]
        finite_neighbors = neighborhood[np.isfinite(neighborhood)]
        if finite_neighbors.size and energy[i, j] <= float(np.min(finite_neighbors)):
            candidates.append((float(energy[i, j]), int(i), int(j)))

    selected: list[tuple[float, float, float]] = []
    for value, i, j in sorted(candidates):
        s = float(profile.s_grid[i])
        distance = float(profile.distance_grid_A[j])
        if any(
            abs(s - old_s) < minimum_s_separation
            and abs(distance - old_distance) < minimum_distance_separation_A
            for old_s, old_distance, _ in selected
        ):
            continue
        selected.append((s, distance, value))
        if len(selected) >= max_basins:
            break
    return selected


def order_basins_by_first_visit(
    profile: JointProfile,
    basins: Sequence[tuple[float, float, float]],
    s_scale: float = 0.10,
    distance_scale_A: float = 0.50,
) -> list[tuple[float, float, float]]:
    """Order minima by the first trajectory frame assigned nearest to each."""
    if not basins:
        return []
    basin_s = np.asarray([basin[0] for basin in basins])
    basin_d = np.asarray([basin[1] for basin in basins])
    order = np.argsort(profile.times_ps)
    first_visit: dict[int, float] = {}
    for sample_index in order:
        scaled_distance = (
            (profile.s_samples[sample_index] - basin_s) / s_scale
        ) ** 2 + (
            (profile.distance_samples_A[sample_index] - basin_d)
            / distance_scale_A
        ) ** 2
        basin_index = int(np.argmin(scaled_distance))
        first_visit.setdefault(basin_index, float(profile.times_ps[sample_index]))
    return [
        basins[index]
        for index, _ in sorted(first_visit.items(), key=lambda item: item[1])
    ]


def draw_chronological_basin_arrows(axis: object, basins: Sequence[tuple[float, float, float]]) -> None:
    """Connect first-visited basin minima with directional arrows."""
    for start, end in zip(basins[:-1], basins[1:]):
        axis.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops={
                "arrowstyle": "-|>",
                "edgecolor": "0.35",
                "facecolor": "none",
                "fill": False,
                "linewidth": 1.6,
                "mutation_scale": 24,
                "shrinkA": 9,
                "shrinkB": 9,
                "connectionstyle": "arc3,rad=0.08",
            },
            zorder=9,
        )
    for sequence_number, basin in enumerate(basins, start=1):
        axis.annotate(
            str(sequence_number),
            xy=(basin[0], basin[1]),
            xytext=(7, 7),
            textcoords="offset points",
            color="black",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "circle,pad=0.16",
                "facecolor": "white",
                "edgecolor": "0.35",
                "linewidth": 0.8,
                "alpha": 0.9,
            },
            zorder=11,
        )


def chronological_legend_handles() -> list[object]:
    """Return compact artists explaining minima and chronological arrows."""
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            color="black",
            markersize=12,
            label="basin minimum (number = first-visit order)",
        ),
        Line2D(
            [],
            [],
            marker=">",
            markerfacecolor="none",
            markeredgecolor="0.35",
            color="0.35",
            linewidth=1.6,
            markersize=9,
            label="increasing-time pathway",
        ),
    ]


def parse_labeled_path(value: str) -> tuple[str, Path]:
    match = LABELED_PATH_RE.match(value)
    if match is None or not match.group("label").strip():
        raise argparse.ArgumentTypeError("value must have the form LABEL=PATH")
    return match.group("label").strip(), Path(match.group("path"))


def load_coordination_distance_pairs(
    path: Path,
    distance_column: str | None,
    time_min_ps: float | None,
    time_max_ps: float | None,
    radial_min_A: float,
    radial_max_A: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float, float]:
    """Load finite, same-frame (time, s, distance) tuples."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {path}")
        fields = reader.fieldnames
        if "time_ps" not in fields or "coordination_s" not in fields:
            raise ValueError(f"{path} must contain time_ps and coordination_s")
        if distance_column is None:
            candidates = [name for name in fields if name.endswith("_Odefect_distance_A")]
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected one *_Odefect_distance_A column in {path}; found {candidates}"
                )
            selected_column = candidates[0]
        else:
            selected_column = distance_column
            if selected_column not in fields:
                raise ValueError(f"Missing {selected_column} in {path}")
        values: list[tuple[float, float, float]] = []
        for row in reader:
            try:
                values.append(
                    (
                        float(row["time_ps"]),
                        float(row["coordination_s"]),
                        float(row[selected_column]),
                    )
                )
            except (TypeError, ValueError):
                values.append((float("nan"), float("nan"), float("nan")))
    data = np.asarray(values, dtype=float)
    finite = np.all(np.isfinite(data), axis=1)
    if time_min_ps is not None:
        finite &= data[:, 0] >= time_min_ps
    if time_max_ps is not None:
        finite &= data[:, 0] <= time_max_ps
    finite &= (data[:, 2] >= radial_min_A) & (data[:, 2] <= radial_max_A)
    selected = data[finite]
    if not selected.size:
        raise ValueError(f"No aligned coordination--distance pairs remain for {path}")
    actual_min = float(time_min_ps if time_min_ps is not None else selected[:, 0].min())
    actual_max = float(time_max_ps if time_max_ps is not None else selected[:, 0].max())
    return (
        selected[:, 0],
        selected[:, 1],
        selected[:, 2],
        selected_column,
        actual_min,
        actual_max,
    )


def load_fes_at_time(
    path: Path,
    ceiling_time_ps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the latest FES block at or before the analysis cutoff."""
    blocks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("###"):
                if current:
                    blocks.append(current)
                    current = []
                continue
            if stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                current.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if current:
        blocks.append(current)
    if not blocks:
        raise ValueError(f"No FES blocks found in {path}")
    biaspot_path = path.parent / "biaspot"
    times: list[float] = []
    if biaspot_path.exists():
        with biaspot_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                match = TIME_RE.search(line)
                if match:
                    times.append(float(match.group(1)) / 1000.0)
    block_count = min(len(blocks), len(times)) if times else len(blocks)
    if block_count == 0:
        raise ValueError(f"No aligned FES blocks found in {path}")
    if times:
        usable_times = np.asarray(times[:block_count], dtype=float)
        candidates = np.flatnonzero(usable_times <= ceiling_time_ps)
        if not candidates.size:
            raise ValueError(
                f"No FES block in {path} exists at or before {ceiling_time_ps:g} ps"
            )
        block_index = int(candidates[-1])
    else:
        block_index = block_count - 1
    selected = np.asarray(blocks[block_index], dtype=float)
    order = np.argsort(selected[:, 0])
    s = selected[order, 0]
    free_energy = selected[order, 1] * EH_TO_KCAL_MOL
    free_energy -= float(np.nanmin(free_energy))
    return s, free_energy


def gaussian_kernel_matrix(
    grid: np.ndarray,
    samples: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    if bandwidth <= 0.0:
        raise ValueError("KDE bandwidth must be positive")
    scaled = (grid[:, None] - samples[None, :]) / bandwidth
    matrix = np.exp(-0.5 * scaled * scaled)
    matrix /= bandwidth * math.sqrt(2.0 * math.pi)
    return matrix


def calculate_joint_profile(
    label: str,
    source_csv: Path,
    fes_path: Path,
    times_ps: np.ndarray,
    s_samples: np.ndarray,
    distance_samples_A: np.ndarray,
    distance_column: str,
    time_min_ps: float,
    time_max_ps: float,
    temperature_K: float = 300.0,
    s_min: float = 0.0,
    s_max: float = 1.0,
    radial_min_A: float = 2.0,
    radial_max_A: float = 9.5,
    s_grid_points: int = 161,
    distance_grid_points: int = 181,
    bandwidth_s: float = 0.04,
    bandwidth_distance_A: float = 0.18,
    support_fraction: float = 1.0e-4,
    radial_bin_width_A: float = 0.25,
    radial_kde_bandwidth_factor: float = 0.13,
    apply_radial_jacobian: bool = False,
) -> JointProfile:
    """Build a directly TP-reweighted joint probability and free energy."""
    if temperature_K <= 0.0:
        raise ValueError("Temperature must be positive")
    if not 0.0 < support_fraction < 1.0:
        raise ValueError("Support fraction must lie between zero and one")
    s_grid = np.linspace(s_min, s_max, s_grid_points)
    distance_grid = np.linspace(radial_min_A, radial_max_A, distance_grid_points)
    beta = 1.0 / (KB_KCAL_MOL_K * temperature_K)
    hills = load_metadynamics_hills(fes_path.with_name("biaspot"))
    bias_values, offsets, log_weights, normalized_weights, effective_sample_size = (
        tiwary_parrinello_weights(
            times_ps,
            s_samples,
            hills,
            temperature_K,
            s_min,
            s_max,
        )
    )

    kernel_s = gaussian_kernel_matrix(s_grid, s_samples, bandwidth_s)
    kernel_d = gaussian_kernel_matrix(
        distance_grid, distance_samples_A, bandwidth_distance_A
    )
    sampled_joint = (kernel_s @ kernel_d.T) / s_samples.size
    weighted_joint = (kernel_s * normalized_weights[None, :]) @ kernel_d.T
    joint_probability = weighted_joint
    joint_normalization = float(
        TRAPEZOID(
            TRAPEZOID(joint_probability, distance_grid, axis=1), s_grid
        )
    )
    if joint_normalization > 0.0:
        joint_probability /= joint_normalization
    equilibrium_s = TRAPEZOID(joint_probability, distance_grid, axis=1)
    fes_s = np.full_like(equilibrium_s, np.nan)
    positive_s = equilibrium_s > 0.0
    fes_s[positive_s] = -np.log(equilibrium_s[positive_s]) / beta
    fes_s[positive_s] -= float(np.nanmin(fes_s[positive_s]))
    sampled_s = TRAPEZOID(sampled_joint, distance_grid, axis=1)
    conditional = np.zeros(sampled_joint.shape, dtype=float)
    valid_s = sampled_s > 0.0
    conditional[valid_s] = sampled_joint[valid_s] / sampled_s[valid_s, None]

    radial_corrected = joint_probability.copy()
    if apply_radial_jacobian:
        radial_corrected /= distance_grid[None, :] ** 2
    supported = sampled_joint >= support_fraction * float(np.max(sampled_joint))
    free_energy = np.full(radial_corrected.shape, np.nan, dtype=float)
    positive = supported & (radial_corrected > 0.0)
    free_energy[positive] = -KB_KCAL_MOL_K * temperature_K * np.log(
        radial_corrected[positive]
    )
    if np.any(positive):
        free_energy[positive] -= float(np.nanmin(free_energy[positive]))
    distance_probability = TRAPEZOID(joint_probability, s_grid, axis=0)
    if apply_radial_jacobian:
        distance_probability = distance_probability / distance_grid**2
    distance_free_energy = np.full_like(distance_probability, np.nan)
    positive_distance = distance_probability > 0.0
    distance_free_energy[positive_distance] = (
        -np.log(distance_probability[positive_distance]) / beta
    )
    distance_free_energy[positive_distance] -= float(
        np.nanmin(distance_free_energy[positive_distance])
    )
    observed_distance_range = (distance_grid >= np.min(distance_samples_A)) & (
        distance_grid <= np.max(distance_samples_A)
    )
    distance_free_energy[~observed_distance_range] = np.nan

    radial_profile = calculate_profile(
        label,
        source_csv,
        times_ps,
        distance_samples_A,
        distance_column,
        time_min_ps=time_min_ps,
        time_max_ps=time_max_ps,
        radial_min_A=radial_min_A,
        radial_max_A=radial_max_A,
        bin_width_A=radial_bin_width_A,
        temperature_K=temperature_K,
        kde_bandwidth_factor=radial_kde_bandwidth_factor,
    )
    return JointProfile(
        label=label,
        source_csv=source_csv,
        fes_path=fes_path,
        sample_count=int(s_samples.size),
        time_min_ps=time_min_ps,
        time_max_ps=time_max_ps,
        times_ps=times_ps,
        bias_kcal_mol=bias_values,
        offset_kcal_mol=offsets,
        log_weights=log_weights,
        normalized_weights=normalized_weights,
        effective_sample_size=effective_sample_size,
        s_samples=s_samples,
        distance_samples_A=distance_samples_A,
        s_grid=s_grid,
        distance_grid_A=distance_grid,
        fes_s_kcal_mol=fes_s,
        equilibrium_s_probability=equilibrium_s,
        sampled_joint_density=sampled_joint,
        conditional_distance_density=conditional,
        equilibrium_joint_probability=joint_probability,
        joint_free_energy_kcal_mol=free_energy,
        distance_free_energy_kcal_mol=distance_free_energy,
        supported=supported,
        radial_profile=radial_profile,
        bandwidth_s=bandwidth_s,
        bandwidth_distance_A=bandwidth_distance_A,
        apply_radial_jacobian=apply_radial_jacobian,
    )


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "run"


def save_joint_data(profile: JointProfile, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{safe_label(profile.label)}_joint_free_energy.csv"
    s_mesh, d_mesh = np.meshgrid(
        profile.s_grid, profile.distance_grid_A, indexing="ij"
    )
    np.savetxt(
        path,
        np.column_stack(
            (
                s_mesh.ravel(),
                d_mesh.ravel(),
                profile.sampled_joint_density.ravel(),
                profile.conditional_distance_density.ravel(),
                profile.equilibrium_joint_probability.ravel(),
                profile.joint_free_energy_kcal_mol.ravel(),
                profile.supported.astype(int).ravel(),
            )
        ),
        delimiter=",",
        header=(
            "coordination_s,distance_A,sampled_joint_density,"
            "conditional_distance_density,equilibrium_joint_probability,"
            "joint_free_energy_kcal_mol,supported"
        ),
        comments="",
    )
    return path


def save_reweighting_data(profile: JointProfile, out_dir: Path) -> Path:
    """Save aligned TP bias corrections and normalized frame weights."""
    path = out_dir / f"{safe_label(profile.label)}_tp_frame_weights.csv"
    np.savetxt(
        path,
        np.column_stack(
            (
                profile.times_ps,
                profile.s_samples,
                profile.distance_samples_A,
                profile.bias_kcal_mol,
                profile.offset_kcal_mol,
                profile.log_weights,
                profile.normalized_weights,
            )
        ),
        delimiter=",",
        header=(
            "time_ps,coordination_s,distance_A,bias_kcal_mol,"
            "tp_offset_kcal_mol,log_weight,normalized_weight"
        ),
        comments="",
    )
    return path


def save_reweighted_marginals(profile: JointProfile, out_dir: Path) -> tuple[Path, Path]:
    """Save the two marginals obtained by integrating the weighted joint KDE."""
    prefix = safe_label(profile.label)
    s_path = out_dir / f"{prefix}_tp_coordination_fes.csv"
    d_path = out_dir / f"{prefix}_tp_distance_pmf.csv"
    np.savetxt(
        s_path,
        np.column_stack(
            (profile.s_grid, profile.equilibrium_s_probability, profile.fes_s_kcal_mol)
        ),
        delimiter=",",
        header="coordination_s,probability_density,free_energy_kcal_mol",
        comments="",
    )
    np.savetxt(
        d_path,
        np.column_stack(
            (profile.distance_grid_A, profile.distance_free_energy_kcal_mol)
        ),
        delimiter=",",
        header="distance_A,free_energy_kcal_mol",
        comments="",
    )
    return s_path, d_path


def plot_reweighted_distance_profiles(
    profiles: Sequence[JointProfile], path: Path, dpi: int = 220
) -> None:
    """Plot the TP-reweighted distance marginals as a shared eight-run grid."""
    import matplotlib.pyplot as plt

    columns = 2 if len(profiles) > 1 else 1
    rows = int(math.ceil(len(profiles) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.8 * columns, 3.7 * rows),
        dpi=dpi,
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    for axis, profile in zip(axes.flat, profiles):
        axis.plot(
            profile.distance_grid_A,
            profile.distance_free_energy_kcal_mol,
            color="#A23B2A",
            linewidth=2.3,
        )
        axis.set_title(
            f"{profile.label}: N={profile.sample_count}, "
            f"N_eff={profile.effective_sample_size:.1f}"
        )
        axis.set_ylabel("TP-reweighted F(d) (kcal/mol)")
        axis.grid(axis="y", alpha=0.22)
    for axis in axes.flat[len(profiles) :]:
        axis.set_visible(False)
    for column_index, axis in enumerate(axes[-1, :]):
        if axis.get_visible():
            profile_index = (rows - 1) * columns + column_index
            axis.set_xlabel(
                distance_axis_label(
                    profiles[profile_index].radial_profile.distance_column
                )
            )
    fig.suptitle("Tiwary–Parrinello-reweighted ion-pair distance free energies")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def radial_frame_free_energies(profile: JointProfile, temperature_K: float) -> np.ndarray:
    """Convert each frame's TP contribution into a relative score."""
    contribution = profile.normalized_weights.copy()
    if profile.apply_radial_jacobian:
        contribution /= profile.distance_samples_A**2
    free_energy = -KB_KCAL_MOL_K * temperature_K * np.log(contribution)
    return free_energy - float(np.min(free_energy))


def plot_raw_reweighted_points(
    profiles: Sequence[JointProfile],
    path: Path,
    temperature_K: float,
    energy_max_kcal_mol: float = 20.0,
    dpi: int = 220,
) -> None:
    """Plot only observed pairs, colored by their radial TP contribution."""
    import matplotlib.pyplot as plt

    columns = 2 if len(profiles) > 1 else 1
    rows = int(math.ceil(len(profiles) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.0 * columns, 4.2 * rows),
        dpi=dpi,
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    points = None
    for axis, profile in zip(axes.flat, profiles):
        values = radial_frame_free_energies(profile, temperature_K)
        points = axis.scatter(
            profile.s_samples,
            profile.distance_samples_A,
            c=np.clip(values, 0.0, energy_max_kcal_mol),
            vmin=0.0,
            vmax=energy_max_kcal_mol,
            cmap="coolwarm",
            s=14,
            alpha=0.72,
            edgecolors="none",
            rasterized=True,
        )
        axis.set_title(
            f"{profile.label}: N={profile.sample_count}, "
            f"N_eff={profile.effective_sample_size:.1f}"
        )
        axis.set_ylim(bottom=0.0)
    for axis in axes.flat[len(profiles) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("coordination, s")
    for row_index, axis in enumerate(axes[:, 0]):
        if axis.get_visible():
            profile_index = row_index * columns
            axis.set_ylabel(
                distance_axis_label(
                    profiles[profile_index].radial_profile.distance_column
                ).replace(", r", "")
            )
    if points is not None:
        colorbar = fig.colorbar(points, ax=[a for a in axes.flat if a.get_visible()])
        qualifier = "radial " if profiles[0].apply_radial_jacobian else ""
        colorbar.set_label(
            f"relative per-frame TP {qualifier}contribution (kcal/mol)"
        )
    fig.suptitle("Observed TP-reweighted coordination–ion-pair samples (no KDE)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def weighted_radial_histogram(
    profile: JointProfile,
    temperature_K: float,
    bin_width_A: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return TP probabilities and PMFs without interpolation."""
    lower = math.floor(np.min(profile.distance_samples_A) / bin_width_A) * bin_width_A
    upper = math.ceil(np.max(profile.distance_samples_A) / bin_width_A) * bin_width_A
    edges = np.arange(lower, upper + 0.5 * bin_width_A, bin_width_A)
    weighted, _ = np.histogram(
        profile.distance_samples_A,
        bins=edges,
        weights=profile.normalized_weights,
    )
    probability = weighted.copy()
    if profile.apply_radial_jacobian:
        shell_measure = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
        probability /= shell_measure
    pmf = np.full_like(probability, np.nan)
    positive = probability > 0.0
    pmf[positive] = -KB_KCAL_MOL_K * temperature_K * np.log(probability[positive])
    pmf[positive] -= float(np.nanmin(pmf[positive]))
    return 0.5 * (edges[:-1] + edges[1:]), weighted, pmf


def plot_binned_reweighted_distance_profiles(
    profiles: Sequence[JointProfile],
    path: Path,
    temperature_K: float,
    out_dir: Path,
    dpi: int = 220,
) -> None:
    """Plot and save disconnected TP radial-histogram points."""
    import matplotlib.pyplot as plt

    columns = 2 if len(profiles) > 1 else 1
    rows = int(math.ceil(len(profiles) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.8 * columns, 3.7 * rows),
        dpi=dpi,
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    for axis, profile in zip(axes.flat, profiles):
        centers, probability, pmf = weighted_radial_histogram(profile, temperature_K)
        finite = np.isfinite(pmf)
        axis.scatter(centers[finite], pmf[finite], s=32, color="#A23B2A")
        axis.set_title(
            f"{profile.label}: N={profile.sample_count}, "
            f"N_eff={profile.effective_sample_size:.1f}"
        )
        axis.set_ylabel(
            "TP pair PMF (kcal/mol)"
            if profile.apply_radial_jacobian
            else "TP F(r) (kcal/mol)"
        )
        axis.grid(axis="y", alpha=0.22)
        np.savetxt(
            out_dir / f"{safe_label(profile.label)}_tp_radial_histogram.csv",
            np.column_stack((centers, probability, pmf)),
            delimiter=",",
            header="distance_bin_center_A,normalized_bin_weight,free_energy_kcal_mol",
            comments="",
        )
    for axis in axes.flat[len(profiles) :]:
        axis.set_visible(False)
    for column_index, axis in enumerate(axes[-1, :]):
        if axis.get_visible():
            index = (rows - 1) * columns + column_index
            axis.set_xlabel(distance_axis_label(profiles[index].radial_profile.distance_column))
    qualifier = "radial " if profiles[0].apply_radial_jacobian else ""
    fig.suptitle(
        f"TP-reweighted {qualifier}histograms (no KDE or interpolation)"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_joint_grid(
    profiles: Sequence[JointProfile],
    path: Path,
    energy_max_kcal_mol: float = 8.0,
    colormap: str = "viridis",
    color_power: float = 0.5,
    wireframe: bool = False,
    dpi: int = 220,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    columns = 2 if len(profiles) > 1 else 1
    rows = int(math.ceil(len(profiles) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.0 * columns, 4.2 * rows),
        dpi=dpi,
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    levels = np.linspace(0.0, energy_max_kcal_mol, 61)
    line_step = max(1.0, energy_max_kcal_mol / 12.0)
    line_levels = np.arange(line_step, energy_max_kcal_mol, line_step)
    color_norm = PowerNorm(
        gamma=color_power, vmin=0.0, vmax=energy_max_kcal_mol
    )
    contour = None
    for axis, profile in zip(axes.flat, profiles):
        plotted = np.ma.masked_invalid(
            np.clip(profile.joint_free_energy_kcal_mol, 0.0, energy_max_kcal_mol)
        )
        if wireframe:
            contour = axis.contour(
                profile.s_grid,
                profile.distance_grid_A,
                plotted.T,
                levels=levels,
                cmap=colormap,
                norm=color_norm,
                linewidths=1.50,
            )
        else:
            contour = axis.contourf(
                profile.s_grid,
                profile.distance_grid_A,
                plotted.T,
                levels=levels,
                cmap=colormap,
                norm=color_norm,
                extend="max",
            )
            axis.contour(
                profile.s_grid,
                profile.distance_grid_A,
                plotted.T,
                levels=line_levels,
                colors="white",
                linewidths=0.35,
                alpha=0.38,
            )
        axis.scatter(
            profile.s_samples,
            profile.distance_samples_A,
            s=3,
            color="black",
            alpha=0.12,
            linewidths=0.0,
            rasterized=True,
        )
        basins = locate_basin_minima(profile, energy_max_kcal_mol)
        if basins:
            chronological_basins = order_basins_by_first_visit(profile, basins)
            draw_chronological_basin_arrows(axis, chronological_basins)
            axis.scatter(
                [basin[0] for basin in chronological_basins],
                [basin[1] for basin in chronological_basins],
                marker="*",
                s=150,
                facecolor="black",
                edgecolor="black",
                linewidths=1.8,
                zorder=10,
            )
        axis.set_title(
            f"{profile.label}: N={profile.sample_count}, "
            f"N_eff={profile.effective_sample_size:.1f}, "
            f"{profile.time_min_ps:g}-{profile.time_max_ps:g} ps"
        )
    for axis in axes.flat[len(profiles) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("coordination, s")
    for row_index, axis in enumerate(axes[:, 0]):
        if axis.get_visible():
            profile_index = row_index * columns
            label = distance_axis_label(
                profiles[profile_index].radial_profile.distance_column
            ).replace(", r", "")
            axis.set_ylabel(label)
    for axis in axes.flat:
        if axis.get_visible():
            axis.set_ylim(bottom=0.0)
    if contour is not None:
        colorbar = fig.colorbar(contour, ax=[ax for ax in axes.flat if ax.get_visible()])
        colorbar.set_label("joint free energy (kcal/mol)")
    fig.legend(
        handles=chronological_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=2,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.75",
        fontsize=9,
    )
    fig.suptitle("Joint coordination–ion-pair free energies")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_joint_with_marginals(
    profile: JointProfile,
    path: Path,
    energy_max_kcal_mol: float = 8.0,
    colormap: str = "viridis",
    color_power: float = 0.5,
    wireframe: bool = False,
    dpi: int = 220,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    fig = plt.figure(figsize=(9.0, 7.6), dpi=dpi, layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(4.5, 1.5), height_ratios=(1.4, 4.5))
    axis_s = fig.add_subplot(grid[0, 0])
    axis_joint = fig.add_subplot(grid[1, 0], sharex=axis_s)
    axis_d = fig.add_subplot(grid[1, 1], sharey=axis_joint)
    levels = np.linspace(0.0, energy_max_kcal_mol, 61)
    line_step = max(1.0, energy_max_kcal_mol / 12.0)
    line_levels = np.arange(line_step, energy_max_kcal_mol, line_step)
    color_norm = PowerNorm(
        gamma=color_power, vmin=0.0, vmax=energy_max_kcal_mol
    )
    plotted = np.ma.masked_invalid(
        np.clip(profile.joint_free_energy_kcal_mol, 0.0, energy_max_kcal_mol)
    )
    if wireframe:
        contour = axis_joint.contour(
            profile.s_grid,
            profile.distance_grid_A,
            plotted.T,
            levels=levels,
            cmap=colormap,
            norm=color_norm,
            linewidths=1.60,
        )
    else:
        contour = axis_joint.contourf(
            profile.s_grid,
            profile.distance_grid_A,
            plotted.T,
            levels=levels,
            cmap=colormap,
            norm=color_norm,
            extend="max",
        )
        axis_joint.contour(
            profile.s_grid,
            profile.distance_grid_A,
            plotted.T,
            levels=line_levels,
            colors="white",
            linewidths=0.4,
            alpha=0.4,
        )
    axis_joint.scatter(
        profile.s_samples,
        profile.distance_samples_A,
        s=4,
        color="black",
        alpha=0.13,
        linewidths=0.0,
        rasterized=True,
    )
    basins = locate_basin_minima(profile, energy_max_kcal_mol)
    if basins:
        chronological_basins = order_basins_by_first_visit(profile, basins)
        draw_chronological_basin_arrows(axis_joint, chronological_basins)
        axis_joint.scatter(
            [basin[0] for basin in chronological_basins],
            [basin[1] for basin in chronological_basins],
            marker="*",
            s=190,
            facecolor="black",
            edgecolor="black",
            linewidths=2.0,
            zorder=10,
            label="local basin minimum",
        )
    axis_s.plot(profile.s_grid, profile.fes_s_kcal_mol, color="#355C8A", linewidth=2)
    axis_s.set_ylabel("F(s) (kcal/mol)")
    axis_s.tick_params(axis="x", labelbottom=False)
    axis_d.plot(
        profile.distance_free_energy_kcal_mol,
        profile.distance_grid_A,
        color="#A23B2A",
        linewidth=2,
    )
    axis_d.set_xlabel("W(d)\n(kcal/mol)")
    axis_d.tick_params(axis="y", labelleft=False)
    axis_joint.set_xlabel("coordination, s")
    axis_joint.set_ylabel(
        distance_axis_label(profile.radial_profile.distance_column).replace(", r", "")
    )
    axis_joint.set_ylim(bottom=0.0)
    colorbar = fig.colorbar(contour, ax=(axis_joint, axis_s, axis_d), shrink=0.85)
    colorbar.set_label("joint free energy (kcal/mol)")
    fig.legend(
        handles=chronological_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.75",
        fontsize=8,
    )
    fig.suptitle(
        f"{profile.label}: TP-reweighted joint free energy with 1D marginals "
        f"(N={profile.sample_count}, N_eff={profile.effective_sample_size:.1f})"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a directly Tiwary--Parrinello-reweighted joint F(s,d)."
    )
    parser.add_argument("--input", action="append", required=True, type=parse_input)
    parser.add_argument("--fes", action="append", required=True, type=parse_labeled_path)
    parser.add_argument("--window", action="append", default=[], type=parse_window)
    parser.add_argument("--distance-column", default=None)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--s-min", type=float, default=0.0)
    parser.add_argument("--s-max", type=float, default=1.0)
    parser.add_argument("--radial-min-A", type=float, default=2.0)
    parser.add_argument("--radial-max-A", type=float, default=9.5)
    parser.add_argument("--s-grid-points", type=int, default=161)
    parser.add_argument("--distance-grid-points", type=int, default=181)
    parser.add_argument("--bandwidth-s", type=float, default=0.04)
    parser.add_argument("--bandwidth-distance-A", type=float, default=0.18)
    parser.add_argument(
        "--radial-kde-bandwidth-factor",
        type=float,
        default=0.13,
        help="Bandwidth factor for the accompanying 1D radial PMF",
    )
    parser.add_argument("--support-fraction", type=float, default=1.0e-4)
    parser.add_argument(
        "--radial-jacobian",
        action="store_true",
        help="Divide probability by d^2 (appropriate only for a fixed atom pair)",
    )
    parser.add_argument("--energy-max-kcal-mol", type=float, default=8.0)
    parser.add_argument("--colormap", default="viridis")
    parser.add_argument(
        "--color-power",
        type=float,
        default=0.5,
        help="Power-law color normalization; values below one emphasize low F",
    )
    parser.add_argument(
        "--wireframe",
        action="store_true",
        help="Draw colored contour lines without a filled surface",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--grid-out", type=Path, default=None)
    parser.add_argument("--distance-grid-out", type=Path, default=None)
    parser.add_argument("--raw-points-grid-out", type=Path, default=None)
    parser.add_argument("--binned-distance-grid-out", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    inputs: list[InputSpec] = args.input
    labels = [item.label for item in inputs]
    if len(labels) != len(set(labels)):
        parser.error("--input labels must be unique")
    fes_paths = dict(args.fes)
    windows = {label: (start, end) for label, start, end in args.window}
    if set(labels) - set(fes_paths):
        parser.error(f"Missing --fes for labels: {sorted(set(labels) - set(fes_paths))}")

    profiles: list[JointProfile] = []
    for item in inputs:
        bounds = windows.get(item.label, (None, None))
        times, s_values, distances, column, actual_min, actual_max = (
            load_coordination_distance_pairs(
                item.path,
                args.distance_column,
                bounds[0],
                bounds[1],
                args.radial_min_A,
                args.radial_max_A,
            )
        )
        profile = calculate_joint_profile(
            item.label,
            item.path,
            fes_paths[item.label],
            times,
            s_values,
            distances,
            column,
            actual_min,
            actual_max,
            temperature_K=args.temperature_K,
            s_min=args.s_min,
            s_max=args.s_max,
            radial_min_A=args.radial_min_A,
            radial_max_A=args.radial_max_A,
            s_grid_points=args.s_grid_points,
            distance_grid_points=args.distance_grid_points,
            bandwidth_s=args.bandwidth_s,
            bandwidth_distance_A=args.bandwidth_distance_A,
            support_fraction=args.support_fraction,
            radial_kde_bandwidth_factor=args.radial_kde_bandwidth_factor,
            apply_radial_jacobian=args.radial_jacobian,
        )
        profiles.append(profile)
        data_path = save_joint_data(profile, args.out_dir)
        weights_path = save_reweighting_data(profile, args.out_dir)
        marginal_paths = save_reweighted_marginals(profile, args.out_dir)
        individual_path = args.out_dir / f"{safe_label(item.label)}_joint_with_1d_pmfs.png"
        plot_joint_with_marginals(
            profile,
            individual_path,
            args.energy_max_kcal_mol,
            args.colormap,
            args.color_power,
            args.wireframe,
            args.dpi,
        )
        print(f"Wrote {data_path}")
        print(
            f"Wrote {weights_path} "
            f"(Tiwary--Parrinello N_eff={profile.effective_sample_size:.2f})"
        )
        print(f"Wrote {marginal_paths[0]}")
        print(f"Wrote {marginal_paths[1]}")
        print(f"Wrote {individual_path}")

    grid_path = args.grid_out or args.out_dir / "joint_free_energy_grid.png"
    plot_joint_grid(
        profiles,
        grid_path,
        args.energy_max_kcal_mol,
        args.colormap,
        args.color_power,
        args.wireframe,
        args.dpi,
    )
    print(f"Wrote {grid_path}")
    if args.distance_grid_out is not None:
        plot_reweighted_distance_profiles(profiles, args.distance_grid_out, args.dpi)
        print(f"Wrote {args.distance_grid_out}")
    if args.raw_points_grid_out is not None:
        plot_raw_reweighted_points(
            profiles,
            args.raw_points_grid_out,
            args.temperature_K,
            min(args.energy_max_kcal_mol, 20.0),
            args.dpi,
        )
        print(f"Wrote {args.raw_points_grid_out}")
    if args.binned_distance_grid_out is not None:
        plot_binned_reweighted_distance_profiles(
            profiles,
            args.binned_distance_grid_out,
            args.temperature_K,
            args.out_dir,
            args.dpi,
        )
        print(f"Wrote {args.binned_distance_grid_out}")


if __name__ == "__main__":
    main()
