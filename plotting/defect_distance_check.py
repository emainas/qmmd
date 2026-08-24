#!/usr/bin/env python3
"""Compare N--defect and nearest-solute--defect distances over time."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from oxygen_wire import (
    _minimum_image,
    detect_diffusive_start,
    iter_xyz_frames,
    read_box_lengths_from_dftb_inp,
    read_xyz_symbols,
)


DEFAULT_LAYER_BOUNDARIES = (2.0, 3.5, 5.5, 7.5, 9.5)
LAYER_COLORS = ("#DCEEFF", "#E2F3E5", "#FFF0D6", "#E6E6E6")
LAYER_POINT_COLORS = ("#2F6FA5", "#4F8A62", "#B57621", "#000000")


def load_distance_csv(
    path: Path,
) -> tuple[
    list[dict[str, str]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
]:
    """Load aligned time, coordination, defect ID, and N--defect distance."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    required = {"time_ps", "coordination_s", "defect_oxygen_id"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
    distance_columns = [
        name for name in rows[0] if name.endswith("_Odefect_distance_A")
    ]
    if len(distance_columns) != 1:
        raise ValueError(
            f"Expected one *_Odefect_distance_A column in {path}; "
            f"found {len(distance_columns)}"
        )
    distance_column = distance_columns[0]
    times = np.asarray([float(row["time_ps"]) for row in rows], dtype=float)
    coordination = np.asarray(
        [float(row["coordination_s"]) for row in rows], dtype=float
    )
    defect_ids = np.asarray(
        [float(row["defect_oxygen_id"]) for row in rows], dtype=float
    )
    reference_distance = np.asarray(
        [float(row[distance_column]) for row in rows], dtype=float
    )
    return rows, times, coordination, defect_ids, reference_distance, distance_column


def nearest_solute_for_frame(
    coords: np.ndarray,
    defect_oxygen_id: int,
    solute_atoms: int,
    box: np.ndarray | None,
) -> tuple[int, float]:
    """Return the one-based closest solute atom ID and its distance."""
    if not 1 <= defect_oxygen_id <= len(coords):
        raise ValueError(f"Defect oxygen ID {defect_oxygen_id} is outside the trajectory")
    if not 1 <= solute_atoms < len(coords):
        raise ValueError(f"Invalid solute atom count {solute_atoms} for {len(coords)} atoms")
    if defect_oxygen_id <= solute_atoms:
        raise ValueError(
            f"Defect oxygen ID {defect_oxygen_id} lies inside atoms 1..{solute_atoms}"
        )

    delta = coords[:solute_atoms] - coords[defect_oxygen_id - 1]
    delta = _minimum_image(delta, box)
    distances = np.linalg.norm(delta, axis=1)
    closest_index = int(np.argmin(distances))
    return closest_index + 1, float(distances[closest_index])


def align_nearest_solute_distances(
    traj_path: Path,
    times: np.ndarray,
    defect_ids: np.ndarray,
    solute_atoms: int,
    box: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Match trajectory frames by timestamp and compute nearest-solute distances."""
    closest_ids = np.full(times.shape, np.nan, dtype=float)
    closest_distances = np.full(times.shape, np.nan, dtype=float)
    valid_targets = np.flatnonzero(np.isfinite(times) & np.isfinite(defect_ids))
    if not valid_targets.size:
        return closest_ids, closest_distances

    finite_times = np.unique(times[np.isfinite(times)])
    target_dt = float(np.median(np.diff(finite_times))) if finite_times.size > 1 else 0.0
    tolerance = max(1.0e-8, 0.51 * target_dt)
    pending = 0
    previous: tuple[float, np.ndarray] | None = None

    for frame_time, coords in iter_xyz_frames(traj_path):
        if frame_time is None:
            raise ValueError(
                "Timestamp alignment requires trajectory timestamps in XYZ comments"
            )
        while pending < valid_targets.size:
            row_index = int(valid_targets[pending])
            target_time = float(times[row_index])
            if frame_time < target_time:
                break
            candidates = [(frame_time, coords)]
            if previous is not None:
                candidates.append(previous)
            matched_time, matched_coords = min(
                candidates, key=lambda item: abs(item[0] - target_time)
            )
            if abs(matched_time - target_time) <= tolerance:
                atom_id, distance = nearest_solute_for_frame(
                    matched_coords,
                    int(defect_ids[row_index]),
                    solute_atoms,
                    box,
                )
                closest_ids[row_index] = atom_id
                closest_distances[row_index] = distance
            pending += 1
        previous = (frame_time, coords)
        if pending >= valid_targets.size:
            break

    if previous is not None:
        while pending < valid_targets.size:
            row_index = int(valid_targets[pending])
            target_time = float(times[row_index])
            if abs(previous[0] - target_time) <= tolerance:
                atom_id, distance = nearest_solute_for_frame(
                    previous[1],
                    int(defect_ids[row_index]),
                    solute_atoms,
                    box,
                )
                closest_ids[row_index] = atom_id
                closest_distances[row_index] = distance
            pending += 1
    return closest_ids, closest_distances


def save_aligned_data(
    path: Path,
    times: np.ndarray,
    defect_ids: np.ndarray,
    reference_distance: np.ndarray,
    closest_ids: np.ndarray,
    closest_distances: np.ndarray,
    symbols: Sequence[str],
    reference_column: str,
) -> None:
    """Save the exact aligned values used in the two-row plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_ps",
                "defect_oxygen_id",
                reference_column,
                "closest_solute_atom_id",
                "closest_solute_element",
                "closest_solute_distance_A",
            ]
        )
        for index, time in enumerate(times):
            closest_id = closest_ids[index]
            element = ""
            if np.isfinite(closest_id):
                element = symbols[int(closest_id) - 1]
            writer.writerow(
                [
                    f"{time:.10g}",
                    "" if not np.isfinite(defect_ids[index]) else int(defect_ids[index]),
                    ""
                    if not np.isfinite(reference_distance[index])
                    else f"{reference_distance[index]:.10g}",
                    "" if not np.isfinite(closest_id) else int(closest_id),
                    element,
                    ""
                    if not np.isfinite(closest_distances[index])
                    else f"{closest_distances[index]:.10g}",
                ]
            )


def add_layer_plot(
    fig,
    axis,
    times: np.ndarray,
    distances: np.ndarray,
    boundaries: np.ndarray,
):
    """Draw the same solvation-layer encoding used by oxygen_wire.py."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap

    for lower, upper, color in zip(boundaries[:-1], boundaries[1:], LAYER_COLORS):
        axis.axhspan(lower, upper, color=color, alpha=0.75, linewidth=0, zorder=0)
    finite = np.isfinite(times) & np.isfinite(distances)
    assigned = np.zeros(distances.shape, dtype=bool)
    for layer_index, (lower, upper, color) in enumerate(
        zip(boundaries[:-1], boundaries[1:], LAYER_POINT_COLORS)
    ):
        upper_test = distances <= upper if layer_index == len(LAYER_COLORS) - 1 else distances < upper
        selected = finite & (distances >= lower) & upper_test
        assigned |= selected
        axis.scatter(
            times[selected],
            distances[selected],
            color=color,
            edgecolors=color,
            linewidths=0.6,
            s=30,
            alpha=0.95,
            rasterized=True,
            zorder=2,
        )
    outside = finite & ~assigned
    axis.scatter(
        times[outside],
        distances[outside],
        color="#3F3F3F",
        edgecolors="#3F3F3F",
        linewidths=0.6,
        s=30,
        alpha=0.95,
        rasterized=True,
        zorder=2,
    )
    cmap = ListedColormap(LAYER_COLORS)
    norm = BoundaryNorm(boundaries, cmap.N)
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axis,
        boundaries=boundaries,
        ticks=0.5 * (boundaries[:-1] + boundaries[1:]),
        spacing="proportional",
        pad=0.02,
    )
    colorbar.ax.set_yticklabels([str(index) for index in range(1, len(boundaries))])
    colorbar.set_label("solvation layer")
    axis.set_ylim(float(boundaries[0]), float(boundaries[-1]))
    axis.grid(axis="y", alpha=0.25)
    return colorbar


def plot_distances(
    out: Path,
    times: np.ndarray,
    reference_distance: np.ndarray,
    closest_ids: np.ndarray,
    closest_distance: np.ndarray,
    symbols: Sequence[str],
    reference_label: str,
    boundaries: np.ndarray,
    diffusive_start: float,
    window_end: float,
) -> None:
    """Create the two-row defect-distance diagnostic."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.5, 8.0),
        dpi=220,
        sharex=True,
        layout="constrained",
    )
    add_layer_plot(fig, axes[0], times, reference_distance, boundaries)
    axes[0].set_ylabel(reference_label)

    finite = np.isfinite(times) & np.isfinite(closest_distance) & np.isfinite(closest_ids)
    unique_closest_ids = sorted({int(value) for value in closest_ids[finite]})
    atom_colors = plt.get_cmap("tab10")(
        np.linspace(0.0, 1.0, max(1, len(unique_closest_ids)))
    )
    for color, atom_id in zip(atom_colors, unique_closest_ids):
        selected = finite & (closest_ids == atom_id)
        element = symbols[atom_id - 1]
        axes[1].scatter(
            times[selected],
            closest_distance[selected],
            color=color,
            edgecolors="black",
            linewidths=0.4,
            s=34,
            alpha=0.95,
            rasterized=True,
            label=f"{element}{atom_id}",
        )
    if unique_closest_ids:
        axes[1].legend(
            title="closest solute atom",
            loc="upper left",
            frameon=False,
            fontsize=11,
            title_fontsize=11,
            ncols=min(3, len(unique_closest_ids)),
        )
    axes[1].set_ylabel(r"O(defect$^+$)--nearest solute distance ($\AA$)")
    axes[1].set_xlabel("t (ps)")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(axis="y", alpha=0.25)

    finite_times = times[np.isfinite(times)]
    if finite_times.size:
        axes[-1].set_xlim(float(np.min(finite_times)), window_end)
    for axis in axes:
        axis.axvline(diffusive_start, color="black", linewidth=3.0, zorder=5)
        axis.axvline(window_end, color="#D62728", linewidth=3.0, zorder=5)
        axis.tick_params(axis="both", labelsize=12)
        axis.xaxis.label.set_fontsize(13)
        axis.yaxis.label.set_fontsize(13)
    for colorbar_axis in fig.axes[2:]:
        colorbar_axis.tick_params(labelsize=12)
        colorbar_axis.yaxis.label.set_fontsize(13)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot N--defect distance and the defect distance to its nearest solute atom."
        )
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--traj", required=True, type=Path)
    parser.add_argument("--solute-atoms", required=True, type=int)
    parser.add_argument("--dftb-inp", type=Path, default=None)
    parser.add_argument("--diffusive-start", default="auto")
    parser.add_argument("--probability-window-ps", type=float, default=1.75)
    parser.add_argument("--split-regime-cutoff", type=float, default=4.0)
    parser.add_argument("--deprotonated-s-max", type=float, default=0.05)
    parser.add_argument("--returned-s-min", type=float, default=0.20)
    parser.add_argument("--persistence-ps", type=float, default=0.05)
    parser.add_argument(
        "--solvation-layer-boundaries",
        type=float,
        nargs=5,
        default=DEFAULT_LAYER_BOUNDARIES,
        metavar=("D0", "D1", "D2", "D3", "D4"),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-out", required=True, type=Path)
    args = parser.parse_args()

    boundaries = np.asarray(args.solvation_layer_boundaries, dtype=float)
    if np.any(np.diff(boundaries) <= 0.0):
        parser.error("--solvation-layer-boundaries must be strictly increasing")

    _, times, coordination, defect_ids, reference_distance, reference_column = (
        load_distance_csv(args.input_csv)
    )
    box = (
        read_box_lengths_from_dftb_inp(args.dftb_inp)
        if args.dftb_inp is not None
        else None
    )
    symbols = read_xyz_symbols(args.traj)
    closest_ids, closest_distances = align_nearest_solute_distances(
        args.traj,
        times,
        defect_ids,
        args.solute_atoms,
        box,
    )

    if str(args.diffusive_start).strip().lower() == "auto":
        diffusive_start = detect_diffusive_start(
            times,
            coordination,
            reference_distance,
            distance_min=args.split_regime_cutoff,
            deprotonated_s_max=args.deprotonated_s_max,
            returned_s_min=args.returned_s_min,
            persistence_ps=args.persistence_ps,
        )
        print(f"Auto-detected t_diffuse = {diffusive_start:.6f} ps")
    else:
        diffusive_start = float(args.diffusive_start)
        print(f"Using t_diffuse = {diffusive_start:.6f} ps")
    window_end = diffusive_start + args.probability_window_ps

    save_aligned_data(
        args.data_out,
        times,
        defect_ids,
        reference_distance,
        closest_ids,
        closest_distances,
        symbols,
        reference_column,
    )
    nitrogen_label = reference_column.removesuffix("_Odefect_distance_A")
    plot_distances(
        args.out,
        times,
        reference_distance,
        closest_ids,
        closest_distances,
        symbols,
        rf"{nitrogen_label}--O(defect$^+$) distance ($\AA$)",
        boundaries,
        diffusive_start,
        window_end,
    )
    print(f"Wrote {args.out}")
    print(f"Wrote {args.data_out}")


if __name__ == "__main__":
    main()
