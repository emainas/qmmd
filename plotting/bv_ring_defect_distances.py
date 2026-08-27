#!/usr/bin/env python3
"""Plot a BV proton defect's distance from the centers of pyrrole rings A--D."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from oxygen_wire import (
    _minimum_image,
    iter_xyz_frames,
    read_box_lengths_from_dftb_inp,
)


RING_LABELS = ("A", "B", "C", "D")
DEFAULT_RING_NITROGEN_IDS = (1, 19, 31, 43)


def read_amber_sections(path: Path) -> dict[str, list[str]]:
    """Read whitespace-compatible Amber topology sections."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("%FLAG "):
            current = line.split(None, 1)[1].strip()
            sections[current] = []
        elif line.startswith("%FORMAT"):
            continue
        elif current is not None:
            sections[current].extend(line.split())
    return sections


def read_solute_bond_graph(
    parm7: Path, solute_atoms: int
) -> tuple[dict[int, set[int]], np.ndarray]:
    """Return a one-based solute bond graph and atom masses."""
    sections = read_amber_sections(parm7)
    required = {"MASS", "BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"}
    missing = required - sections.keys()
    if missing:
        raise ValueError(f"Missing Amber topology sections: {sorted(missing)}")
    masses = np.asarray([float(value) for value in sections["MASS"]], dtype=float)
    if solute_atoms < 1 or solute_atoms > masses.size:
        raise ValueError(
            f"Invalid solute atom count {solute_atoms} for {masses.size} topology atoms"
        )
    adjacency = {atom_id: set() for atom_id in range(1, solute_atoms + 1)}
    for section_name in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        values = [int(value) for value in sections[section_name]]
        if len(values) % 3:
            raise ValueError(f"Malformed {section_name} section in {parm7}")
        for index in range(0, len(values), 3):
            left = values[index] // 3 + 1
            right = values[index + 1] // 3 + 1
            if left <= solute_atoms and right <= solute_atoms:
                adjacency[left].add(right)
                adjacency[right].add(left)
    return adjacency, masses[:solute_atoms]


def five_membered_heavy_cycles(
    anchor_id: int,
    adjacency: dict[int, set[int]],
    masses: np.ndarray,
) -> list[tuple[int, ...]]:
    """Return unique five-membered heavy-atom cycles containing an anchor."""
    cycles: set[tuple[int, ...]] = set()

    def visit(path: list[int]) -> None:
        current = path[-1]
        if len(path) == 5:
            if anchor_id in adjacency[current]:
                cycles.add(tuple(sorted(path)))
            return
        for neighbor in adjacency[current]:
            if neighbor in path or masses[neighbor - 1] < 2.5:
                continue
            visit([*path, neighbor])

    visit([anchor_id])
    return sorted(cycles)


def identify_bv_rings(
    parm7: Path,
    solute_atoms: int,
    nitrogen_ids: Sequence[int] = DEFAULT_RING_NITROGEN_IDS,
) -> dict[str, tuple[int, ...]]:
    """Identify BV rings A--D as five-membered cycles around known nitrogens."""
    if len(nitrogen_ids) != len(RING_LABELS):
        raise ValueError("Exactly four ring nitrogen IDs are required")
    adjacency, masses = read_solute_bond_graph(parm7, solute_atoms)
    rings: dict[str, tuple[int, ...]] = {}
    for label, nitrogen_id in zip(RING_LABELS, nitrogen_ids):
        if nitrogen_id not in adjacency:
            raise ValueError(f"Ring {label} nitrogen ID {nitrogen_id} is outside solute")
        cycles = five_membered_heavy_cycles(nitrogen_id, adjacency, masses)
        if len(cycles) != 1:
            raise ValueError(
                f"Expected one five-membered heavy ring containing N{nitrogen_id}; "
                f"found {cycles}"
            )
        rings[label] = cycles[0]
    return rings


def identify_terminal_ring_oxygens(
    parm7: Path,
    solute_atoms: int,
    rings: dict[str, tuple[int, ...]],
) -> dict[str, int]:
    """Identify the single oxygen directly bonded to terminal rings A and D."""
    adjacency, masses = read_solute_bond_graph(parm7, solute_atoms)
    oxygen_ids: dict[str, int] = {}
    for label in ("A", "D"):
        ring_ids = set(rings[label])
        candidates = sorted(
            {
                neighbor
                for atom_id in ring_ids
                for neighbor in adjacency[atom_id]
                if neighbor not in ring_ids and 15.5 <= masses[neighbor - 1] <= 16.5
            }
        )
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one oxygen directly attached to ring {label}; "
                f"found {candidates}"
            )
        oxygen_ids[label] = candidates[0]
    return oxygen_ids


def load_defect_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load aligned times and one-based defect oxygen IDs."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "time_ps" not in rows[0] or "defect_oxygen_id" not in rows[0]:
        raise ValueError(f"{path} must contain time_ps and defect_oxygen_id columns")
    times = np.asarray([float(row["time_ps"]) for row in rows], dtype=float)
    defect_ids = np.asarray(
        [float(row["defect_oxygen_id"] or "nan") for row in rows], dtype=float
    )
    return times, defect_ids


def ring_center(
    coords: np.ndarray, ring_atom_ids: Sequence[int], box: np.ndarray | None
) -> np.ndarray:
    """Return a PBC-unwrapped geometric center for one ring."""
    ids = np.asarray(ring_atom_ids, dtype=int)
    anchor = coords[ids[0] - 1]
    offsets = _minimum_image(coords[ids - 1] - anchor, box)
    return anchor + np.mean(offsets, axis=0)


def calculate_ring_distances(
    traj: Path,
    times: np.ndarray,
    defect_ids: np.ndarray,
    rings: dict[str, tuple[int, ...]],
    ring_oxygen_ids: dict[str, int],
    box: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Timestamp-align trajectory frames and calculate four ring distances."""
    distances = {label: np.full(times.shape, np.nan) for label in RING_LABELS}
    oxygen_distances = {
        label: np.full(times.shape, np.nan) for label in ring_oxygen_ids
    }
    finite_times = times[np.isfinite(times)]
    time_steps = np.diff(finite_times)
    positive_steps = time_steps[time_steps > 0.0]
    tolerance = 0.51 * float(np.median(positive_steps)) if positive_steps.size else 1e-6
    target_index = 0
    for frame_time, coords in iter_xyz_frames(traj):
        if frame_time is None:
            raise ValueError(f"Trajectory frame in {traj} has no timestamp")
        while target_index < len(times) and times[target_index] < frame_time - tolerance:
            target_index += 1
        if target_index >= len(times):
            break
        if abs(times[target_index] - frame_time) > tolerance:
            continue
        raw_defect_id = defect_ids[target_index]
        if np.isfinite(raw_defect_id):
            defect_id = int(raw_defect_id)
            if not 1 <= defect_id <= len(coords):
                raise ValueError(
                    f"Defect oxygen ID {defect_id} exceeds {len(coords)} trajectory atoms"
                )
            defect_coord = coords[defect_id - 1]
            for label, atom_ids in rings.items():
                center = ring_center(coords, atom_ids, box)
                displacement = _minimum_image(defect_coord - center, box)
                distances[label][target_index] = float(np.linalg.norm(displacement))
            for label, oxygen_id in ring_oxygen_ids.items():
                displacement = _minimum_image(
                    defect_coord - coords[oxygen_id - 1], box
                )
                oxygen_distances[label][target_index] = float(
                    np.linalg.norm(displacement)
                )
        target_index += 1
    return distances, oxygen_distances


def save_aligned_data(
    path: Path,
    times: np.ndarray,
    defect_ids: np.ndarray,
    distances: dict[str, np.ndarray],
    oxygen_distances: dict[str, np.ndarray],
    ring_oxygen_ids: dict[str, int],
) -> None:
    """Save the numerical data represented in the figure."""
    fieldnames = ["time_ps", "defect_oxygen_id"] + [
        f"ring_{label}_center_defect_distance_A" for label in RING_LABELS
    ] + [
        "ring_A_oxygen_id",
        "ring_A_oxygen_defect_distance_A",
        "ring_D_oxygen_id",
        "ring_D_oxygen_defect_distance_A",
        "closest_ring",
        "closest_ring_center_defect_distance_A",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, time_ps in enumerate(times):
            values = np.asarray([distances[label][index] for label in RING_LABELS])
            closest_ring = ""
            closest_distance = float("nan")
            if np.any(np.isfinite(values)):
                closest_index = int(np.nanargmin(values))
                closest_ring = RING_LABELS[closest_index]
                closest_distance = float(values[closest_index])
            writer.writerow(
                {
                    "time_ps": f"{time_ps:.10g}",
                    "defect_oxygen_id": (
                        "" if not np.isfinite(defect_ids[index]) else int(defect_ids[index])
                    ),
                    **{
                        f"ring_{label}_center_defect_distance_A": (
                            "" if not np.isfinite(distances[label][index])
                            else f"{distances[label][index]:.10g}"
                        )
                        for label in RING_LABELS
                    },
                    "ring_A_oxygen_id": ring_oxygen_ids["A"],
                    "ring_A_oxygen_defect_distance_A": (
                        "" if not np.isfinite(oxygen_distances["A"][index])
                        else f"{oxygen_distances['A'][index]:.10g}"
                    ),
                    "ring_D_oxygen_id": ring_oxygen_ids["D"],
                    "ring_D_oxygen_defect_distance_A": (
                        "" if not np.isfinite(oxygen_distances["D"][index])
                        else f"{oxygen_distances['D'][index]:.10g}"
                    ),
                    "closest_ring": closest_ring,
                    "closest_ring_center_defect_distance_A": (
                        "" if not np.isfinite(closest_distance)
                        else f"{closest_distance:.10g}"
                    ),
                }
            )


def plot_ring_distances(
    path: Path,
    times: np.ndarray,
    distances: dict[str, np.ndarray],
    oxygen_distances: dict[str, np.ndarray],
    ring_oxygen_ids: dict[str, int],
    diffusive_start: float | None,
    window_ps: float,
) -> None:
    """Write one PNG containing the four ring-center distance scatter series."""
    import matplotlib.pyplot as plt

    colors = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73", "D": "#CC79A7"}
    fig, axis = plt.subplots(figsize=(7.2, 5.2), dpi=220, layout="constrained")
    for label in RING_LABELS:
        finite = np.isfinite(times) & np.isfinite(distances[label])
        axis.scatter(
            times[finite],
            distances[label][finite],
            s=24,
            color=colors[label],
            edgecolors="none",
            alpha=0.9,
            rasterized=True,
            label=f"ring {label}",
        )
    oxygen_colors = {"A": "#003B73", "D": "#9A2458"}
    for label in ("A", "D"):
        finite = np.isfinite(times) & np.isfinite(oxygen_distances[label])
        axis.scatter(
            times[finite],
            oxygen_distances[label][finite],
            s=24,
            color=oxygen_colors[label],
            marker="^",
            edgecolors="none",
            alpha=0.9,
            rasterized=True,
            label=f"ring {label} oxygen (O{ring_oxygen_ids[label]})",
        )
    if diffusive_start is not None:
        axis.axvline(diffusive_start, color="black", linewidth=3.0, zorder=5)
        window_end = diffusive_start + window_ps
        axis.axvline(window_end, color="#D62728", linewidth=3.0, zorder=5)
        finite_times = times[np.isfinite(times)]
        if finite_times.size:
            axis.set_xlim(float(np.min(finite_times)), window_end)
    axis.set_xlabel("t (ps)")
    axis.set_ylabel("ring center--O(defect+) distance (Å)")
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left", frameon=False, ncols=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot BV ring A--D center distances to a water-borne proton defect."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--traj", required=True, type=Path)
    parser.add_argument("--parm7", required=True, type=Path)
    parser.add_argument("--dftb-inp", type=Path, default=None)
    parser.add_argument("--solute-atoms", type=int, default=78)
    parser.add_argument(
        "--ring-nitrogen-ids",
        type=int,
        nargs=4,
        default=DEFAULT_RING_NITROGEN_IDS,
        metavar=("NA", "NB", "NC", "ND"),
    )
    parser.add_argument("--diffusive-start", type=float, default=None)
    parser.add_argument("--probability-window-ps", type=float, default=1.75)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-out", required=True, type=Path)
    args = parser.parse_args()

    rings = identify_bv_rings(
        args.parm7, args.solute_atoms, args.ring_nitrogen_ids
    )
    for label, atom_ids in rings.items():
        print(f"Ring {label}: {','.join(map(str, atom_ids))}")
    ring_oxygen_ids = identify_terminal_ring_oxygens(
        args.parm7, args.solute_atoms, rings
    )
    print(
        "Terminal ring oxygens: "
        + ", ".join(
            f"{label}=O{oxygen_id}"
            for label, oxygen_id in ring_oxygen_ids.items()
        )
    )
    times, defect_ids = load_defect_series(args.input_csv)
    box = (
        read_box_lengths_from_dftb_inp(args.dftb_inp)
        if args.dftb_inp is not None
        else None
    )
    distances, oxygen_distances = calculate_ring_distances(
        args.traj, times, defect_ids, rings, ring_oxygen_ids, box
    )
    save_aligned_data(
        args.data_out,
        times,
        defect_ids,
        distances,
        oxygen_distances,
        ring_oxygen_ids,
    )
    plot_ring_distances(
        args.out,
        times,
        distances,
        oxygen_distances,
        ring_oxygen_ids,
        args.diffusive_start,
        args.probability_window_ps,
    )
    print(f"Wrote {args.out}")
    print(f"Wrote {args.data_out}")


if __name__ == "__main__":
    main()
