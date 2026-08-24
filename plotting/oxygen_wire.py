#!/usr/bin/env python3
"""Find oxygen-distance paths from a nitrogen to a per-frame defect oxygen.

This is deliberately a geometric, topology-free first-pass water-wire test.
The nitrogen and all solvent oxygens are graph nodes; two nodes are connected
when their minimum-image distance is no greater than ``cutoff``.  The shortest
path is reported for every trajectory frame represented in an existing
coordination-versus-distance CSV.

The functions in this file are importable by ``2d_plot_s_and_d.py``.  Running
the file directly writes an augmented, frame-aligned CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

@dataclass(frozen=True)
class WireResult:
    """Water-wire result for one frame; atom IDs are one-based."""

    evaluated: bool
    connected: bool
    bridging_water_count: int
    oxygen_path: tuple[int, ...]


@dataclass(frozen=True)
class HBondEdge:
    """Hydrogen-supported edge; atom IDs are one-based."""

    donor_id: int
    hydrogen_id: int
    acceptor_id: int
    heavy_distance: float
    hydrogen_acceptor_distance: float
    angle_degrees: float


@dataclass(frozen=True)
class FrameResult:
    oxygen: WireResult
    oxygen_edge_distances: tuple[float, ...]
    hbond: WireResult
    hbond_edges: tuple[HBondEdge, ...]
    hbond_mean_consecutive_angle_deg: float
    hbond_loopiness_lambda: float


TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")


def iter_xyz_frames(path: Path) -> Iterator[tuple[float | None, np.ndarray]]:
    """Yield optional times in ps and coordinates from an XYZ trajectory."""
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                natoms = int(line)
            except ValueError as exc:
                raise ValueError(f"Invalid XYZ atom-count line: {line!r}") from exc
            comment = handle.readline()
            if not comment:
                raise ValueError(f"Missing XYZ comment line in {path}")
            match = TIME_RE.search(comment)
            time_ps = float(match.group(1)) / 1000.0 if match else None
            coords = np.empty((natoms, 3), dtype=float)
            for atom_index in range(natoms):
                atom_line = handle.readline()
                if not atom_line:
                    raise ValueError(f"Unexpected EOF in {path}")
                parts = atom_line.split()
                if len(parts) < 4:
                    raise ValueError(f"Invalid XYZ atom line: {atom_line.strip()}")
                coords[atom_index] = [float(x) for x in parts[1:4]]
            yield time_ps, coords


def iter_xyz_coords(path: Path) -> Iterator[np.ndarray]:
    for _time_ps, coords in iter_xyz_frames(path):
        yield coords


def read_box_lengths_from_dftb_inp(path: Path) -> np.ndarray:
    """Read diagonal lengths for an orthorhombic DFTB periodic box."""
    vectors: list[list[float]] = []
    for line in path.read_text().splitlines():
        if not line.startswith("TV"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            vectors.append([float(x) for x in parts[1:4]])
    if len(vectors) != 3:
        raise ValueError(f"Expected 3 TV lines in {path}, got {len(vectors)}")
    return np.asarray([vectors[0][0], vectors[1][1], vectors[2][2]], dtype=float)


def read_xyz_symbols(path: Path) -> list[str]:
    """Read atom symbols from the first XYZ frame."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                natoms = int(line)
            except ValueError as exc:
                raise ValueError(f"Invalid XYZ atom-count line: {line!r}") from exc
            if not handle.readline():
                raise ValueError(f"Missing XYZ comment line in {path}")
            symbols: list[str] = []
            for _ in range(natoms):
                atom_line = handle.readline()
                if not atom_line:
                    raise ValueError(f"Unexpected EOF in first frame of {path}")
                parts = atom_line.split()
                if len(parts) < 4:
                    raise ValueError(f"Invalid XYZ atom line: {atom_line.strip()}")
                symbols.append(parts[0])
            return symbols
    raise ValueError(f"No XYZ frames found in {path}")


def _minimum_image(delta: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    if box is None:
        return delta
    return delta - box * np.round(delta / box)


def shortest_oxygen_path(
    coords: np.ndarray,
    nitrogen_id: int,
    defect_oxygen_id: int,
    solvent_oxygen_ids: Sequence[int],
    cutoff: float = 3.5,
    max_bridging_waters: int = 4,
    box: np.ndarray | None = None,
) -> WireResult:
    """Return the shortest N--O...O(defect) path in one frame.

    A direct N--O(defect) contact has zero bridging waters.  The returned
    ``oxygen_path`` excludes the nitrogen but includes the defect oxygen.
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if max_bridging_waters < 0:
        raise ValueError("max_bridging_waters must be non-negative")

    natoms = len(coords)
    if not (1 <= nitrogen_id <= natoms and 1 <= defect_oxygen_id <= natoms):
        return WireResult(True, False, -1, ())

    oxygen_ids = sorted(
        {
            int(atom_id)
            for atom_id in solvent_oxygen_ids
            if 1 <= int(atom_id) <= natoms
        }
        | {int(defect_oxygen_id)}
    )
    node_ids = [int(nitrogen_id)] + [x for x in oxygen_ids if x != nitrogen_id]
    id_to_node = {atom_id: i for i, atom_id in enumerate(node_ids)}
    target = id_to_node.get(int(defect_oxygen_id))
    if target is None:
        return WireResult(True, False, -1, ())

    xyz = coords[np.asarray(node_ids, dtype=int) - 1]
    delta = xyz[:, None, :] - xyz[None, :, :]
    delta = _minimum_image(delta, box)
    distance = np.linalg.norm(delta, axis=2)
    adjacent = (distance <= cutoff) & (distance > 0.0)

    # N may connect to oxygens, while all other graph edges are O--O.
    max_edges = max_bridging_waters + 1
    queue: deque[int] = deque([0])
    parent = {0: -1}
    depth = {0: 0}
    while queue:
        current = queue.popleft()
        if current == target:
            break
        if depth[current] >= max_edges:
            continue
        for neighbor in np.flatnonzero(adjacent[current]):
            neighbor = int(neighbor)
            if neighbor in parent:
                continue
            parent[neighbor] = current
            depth[neighbor] = depth[current] + 1
            queue.append(neighbor)

    if target not in parent:
        return WireResult(True, False, -1, ())

    path_nodes: list[int] = []
    current = target
    while current != -1:
        path_nodes.append(current)
        current = parent[current]
    path_nodes.reverse()
    full_path = tuple(node_ids[i] for i in path_nodes)
    oxygen_path = full_path[1:]
    return WireResult(True, True, max(0, len(oxygen_path) - 1), oxygen_path)


def _path_edge_distances(
    coords: np.ndarray,
    nitrogen_id: int,
    oxygen_path: Sequence[int],
    box: np.ndarray | None,
) -> tuple[float, ...]:
    ids = [nitrogen_id, *oxygen_path]
    values: list[float] = []
    for left, right in zip(ids, ids[1:]):
        delta = _minimum_image(coords[left - 1] - coords[right - 1], box)
        values.append(float(np.linalg.norm(delta)))
    return tuple(values)


def _hbond_shape_metrics(
    coords: np.ndarray,
    nitrogen_id: int,
    oxygen_path: Sequence[int],
    box: np.ndarray | None,
) -> tuple[float, float]:
    """Return mean consecutive-path angle and dimensionless loopiness.

    Path vectors are consistently directed from the selected nitrogen toward
    the Mulliken-assigned defect oxygen and use minimum-image displacements.
    The loopiness is ``lambda = 1 - end_to_end / contour_length``.
    """
    atom_ids = [nitrogen_id, *oxygen_path]
    vectors = np.asarray(
        [
            _minimum_image(coords[right - 1] - coords[left - 1], box)
            for left, right in zip(atom_ids, atom_ids[1:])
        ],
        dtype=float,
    )
    if not len(vectors):
        return float("nan"), float("nan")
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 0.0):
        return float("nan"), float("nan")

    contour_length = float(np.sum(lengths))
    end_to_end = float(np.linalg.norm(np.sum(vectors, axis=0)))
    loopiness = float(np.clip(1.0 - end_to_end / contour_length, 0.0, 1.0))

    if len(vectors) < 2:
        mean_angle = float("nan")
    else:
        tangents = vectors / lengths[:, None]
        cosines = np.sum(tangents[:-1] * tangents[1:], axis=1)
        angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
        mean_angle = float(np.mean(angles))
    return mean_angle, loopiness


def shortest_hbond_path(
    coords: np.ndarray,
    nitrogen_id: int,
    defect_oxygen_id: int,
    solvent_oxygen_ids: Sequence[int],
    hydrogen_ids: Sequence[int],
    covalent_cutoff: float = 1.3,
    hydrogen_acceptor_cutoff: float = 2.5,
    angle_cutoff: float = 135.0,
    max_bridging_waters: int = 4,
    box: np.ndarray | None = None,
) -> tuple[WireResult, tuple[HBondEdge, ...]]:
    """Find the shortest path containing only hydrogen-supported edges."""
    oxygen_ids = sorted(set(map(int, solvent_oxygen_ids)) | {int(defect_oxygen_id)})
    node_ids = [int(nitrogen_id)] + [x for x in oxygen_ids if x != nitrogen_id]
    id_to_node = {atom_id: i for i, atom_id in enumerate(node_ids)}
    target = id_to_node.get(int(defect_oxygen_id))
    if target is None:
        return WireResult(True, False, -1, ()), ()

    node_xyz = coords[np.asarray(node_ids) - 1]
    hydrogen_ids_array = np.asarray(hydrogen_ids, dtype=int)
    hydrogen_xyz = coords[hydrogen_ids_array - 1]
    node_h_vectors = _minimum_image(
        node_xyz[:, None, :] - hydrogen_xyz[None, :, :], box
    )
    node_h_distances = np.linalg.norm(node_h_vectors, axis=2)
    heavy_vectors = _minimum_image(
        node_xyz[:, None, :] - node_xyz[None, :, :], box
    )
    heavy_distances = np.linalg.norm(heavy_vectors, axis=2)
    adjacency: list[list[int]] = [[] for _ in node_ids]
    edge_details: dict[tuple[int, int], HBondEdge] = {}

    for left in range(len(node_ids)):
        for right in range(left + 1, len(node_ids)):
            heavy_distance = float(heavy_distances[left, right])
            # Triangle-inequality optimization only: no pair farther apart than
            # the sum of the two H-distance cutoffs can possibly pass below.
            if heavy_distance > covalent_cutoff + hydrogen_acceptor_cutoff:
                continue
            left_h = node_h_vectors[left]
            right_h = node_h_vectors[right]
            left_dist = node_h_distances[left]
            right_dist = node_h_distances[right]
            candidates = np.flatnonzero(
                (np.minimum(left_dist, right_dist) <= covalent_cutoff)
                & (np.maximum(left_dist, right_dist) <= hydrogen_acceptor_cutoff)
            )
            best: HBondEdge | None = None
            for h_index in candidates:
                if left_dist[h_index] <= right_dist[h_index]:
                    donor_node, acceptor_node = left, right
                    donor_vector, acceptor_vector = left_h[h_index], right_h[h_index]
                    acceptor_distance = right_dist[h_index]
                else:
                    donor_node, acceptor_node = right, left
                    donor_vector, acceptor_vector = right_h[h_index], left_h[h_index]
                    acceptor_distance = left_dist[h_index]
                denominator = np.linalg.norm(donor_vector) * np.linalg.norm(acceptor_vector)
                if denominator == 0.0:
                    continue
                cosine = float(np.clip(np.dot(donor_vector, acceptor_vector) / denominator, -1.0, 1.0))
                angle = float(np.degrees(np.arccos(cosine)))
                if angle < angle_cutoff:
                    continue
                edge = HBondEdge(
                    donor_id=node_ids[donor_node],
                    hydrogen_id=int(hydrogen_ids_array[h_index]),
                    acceptor_id=node_ids[acceptor_node],
                    heavy_distance=heavy_distance,
                    hydrogen_acceptor_distance=float(acceptor_distance),
                    angle_degrees=angle,
                )
                if best is None or edge.angle_degrees > best.angle_degrees:
                    best = edge
            if best is not None:
                adjacency[left].append(right)
                adjacency[right].append(left)
                edge_details[(left, right)] = best

    max_edges = max_bridging_waters + 1
    queue: deque[int] = deque([0])
    parent = {0: -1}
    depth = {0: 0}
    while queue:
        current = queue.popleft()
        if current == target:
            break
        if depth[current] >= max_edges:
            continue
        for neighbor in adjacency[current]:
            if neighbor in parent:
                continue
            parent[neighbor] = current
            depth[neighbor] = depth[current] + 1
            queue.append(neighbor)
    if target not in parent:
        return WireResult(True, False, -1, ()), ()

    path_nodes: list[int] = []
    current = target
    while current != -1:
        path_nodes.append(current)
        current = parent[current]
    path_nodes.reverse()
    oxygen_path = tuple(node_ids[i] for i in path_nodes[1:])
    edges = tuple(
        edge_details[tuple(sorted((left, right)))]
        for left, right in zip(path_nodes, path_nodes[1:])
    )
    return WireResult(True, True, max(0, len(oxygen_path) - 1), oxygen_path), edges


def iter_wire_results(
    traj_path: Path,
    defect_oxygen_ids: Sequence[float],
    nitrogen_id: int,
    solute_atoms: int,
    cutoff: float = 3.5,
    max_bridging_waters: int = 4,
    box: np.ndarray | None = None,
    covalent_cutoff: float = 1.3,
    hydrogen_acceptor_cutoff: float = 2.5,
    angle_cutoff: float = 135.0,
    target_times: Sequence[float] | None = None,
) -> Iterator[FrameResult]:
    """Yield one result per requested aligned frame."""
    symbols = read_xyz_symbols(traj_path)
    finite_defect_ids = [int(x) for x in defect_oxygen_ids if np.isfinite(x)]
    if finite_defect_ids and max(finite_defect_ids) > len(symbols):
        raise ValueError(
            f"Defect atom ID {max(finite_defect_ids)} exceeds the {len(symbols)} atoms "
            f"in {traj_path}; the CSV and trajectory do not describe the same system"
        )
    solvent_oxygen_ids = [
        atom_id
        for atom_id, symbol in enumerate(symbols, start=1)
        if atom_id > solute_atoms and symbol.upper() == "O"
    ]
    if not solvent_oxygen_ids:
        raise ValueError(f"No solvent oxygens found after atom {solute_atoms}")
    hydrogen_ids = [
        atom_id
        for atom_id, symbol in enumerate(symbols, start=1)
        if symbol.upper() == "H"
    ]

    target_limit = None
    target_read_tolerance = 0.0
    if target_times is not None:
        target_array_for_read = np.asarray(target_times, dtype=float)
        finite_targets_for_read = target_array_for_read[
            np.isfinite(target_array_for_read)
        ]
        if finite_targets_for_read.size:
            target_limit = float(np.max(finite_targets_for_read))
            target_diffs_for_read = np.diff(finite_targets_for_read)
            positive_diffs_for_read = target_diffs_for_read[
                target_diffs_for_read > 0
            ]
            if positive_diffs_for_read.size:
                target_read_tolerance = 0.51 * float(
                    np.median(positive_diffs_for_read)
                )

    frames: list[tuple[float | None, np.ndarray]] = []
    for frame_time, coords in iter_xyz_frames(traj_path):
        frames.append((frame_time, coords))
        if (
            target_limit is not None
            and frame_time is not None
            and frame_time > target_limit + target_read_tolerance
        ):
            break
    timed_frames = bool(frames) and all(time is not None for time, _coords in frames)
    trajectory_times = (
        np.asarray([time for time, _coords in frames], dtype=float)
        if timed_frames
        else np.array([], dtype=float)
    )
    if target_times is not None and timed_frames and trajectory_times.size > 1:
        target_array = np.asarray(target_times, dtype=float)
        finite_target = target_array[np.isfinite(target_array)]
        target_diffs = np.diff(finite_target)
        positive_target_diffs = target_diffs[target_diffs > 0]
        trajectory_tolerance = (
            0.51 * float(np.median(positive_target_diffs))
            if positive_target_diffs.size
            else 1.0e-6
        )
    else:
        trajectory_tolerance = 0.0

    for row_index, raw_defect_id in enumerate(defect_oxygen_ids):
        coords = None
        if target_times is not None and timed_frames:
            target_time = float(target_times[row_index])
            insertion = int(np.searchsorted(trajectory_times, target_time))
            candidates = [i for i in (insertion - 1, insertion) if 0 <= i < len(frames)]
            if candidates:
                frame_index = min(candidates, key=lambda i: abs(trajectory_times[i] - target_time))
                if abs(trajectory_times[frame_index] - target_time) <= trajectory_tolerance:
                    coords = frames[frame_index][1]
        elif row_index < len(frames):
            coords = frames[row_index][1]

        if coords is None:
            missing = WireResult(False, False, -1, ())
            yield FrameResult(missing, (), missing, (), float("nan"), float("nan"))
            continue
        if not np.isfinite(raw_defect_id):
            missing = WireResult(False, False, -1, ())
            yield FrameResult(missing, (), missing, (), float("nan"), float("nan"))
            continue
        oxygen = shortest_oxygen_path(
            coords=coords,
            nitrogen_id=nitrogen_id,
            defect_oxygen_id=int(raw_defect_id),
            solvent_oxygen_ids=solvent_oxygen_ids,
            cutoff=cutoff,
            max_bridging_waters=max_bridging_waters,
            box=box,
        )
        hbond, hbond_edges = shortest_hbond_path(
            coords=coords,
            nitrogen_id=nitrogen_id,
            defect_oxygen_id=int(raw_defect_id),
            solvent_oxygen_ids=solvent_oxygen_ids,
            hydrogen_ids=hydrogen_ids,
            covalent_cutoff=covalent_cutoff,
            hydrogen_acceptor_cutoff=hydrogen_acceptor_cutoff,
            angle_cutoff=angle_cutoff,
            max_bridging_waters=max_bridging_waters,
            box=box,
        )
        if hbond.connected:
            mean_angle, loopiness = _hbond_shape_metrics(
                coords, nitrogen_id, hbond.oxygen_path, box
            )
        else:
            mean_angle, loopiness = float("nan"), float("nan")
        yield FrameResult(
            oxygen=oxygen,
            oxygen_edge_distances=_path_edge_distances(
                coords, nitrogen_id, oxygen.oxygen_path, box
            ),
            hbond=hbond,
            hbond_edges=hbond_edges,
            hbond_mean_consecutive_angle_deg=mean_angle,
            hbond_loopiness_lambda=loopiness,
        )


def augment_csv(
    input_csv: Path,
    output_csv: Path,
    traj_path: Path,
    nitrogen_id: int,
    solute_atoms: int,
    cutoff: float = 3.5,
    max_bridging_waters: int = 4,
    box: np.ndarray | None = None,
    covalent_cutoff: float = 1.3,
    hydrogen_acceptor_cutoff: float = 2.5,
    angle_cutoff: float = 135.0,
) -> None:
    """Append frame-aligned oxygen-wire columns to a scatter-plot CSV."""
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No data rows found in {input_csv}")
    if "defect_oxygen_id" not in rows[0]:
        raise ValueError(f"Missing defect_oxygen_id column in {input_csv}")

    defect_ids = np.asarray(
        [float(row["defect_oxygen_id"]) for row in rows], dtype=float
    )
    target_times = (
        np.asarray([float(row["time_ps"]) for row in rows], dtype=float)
        if "time_ps" in rows[0]
        else None
    )
    results = list(
        iter_wire_results(
            traj_path,
            defect_ids,
            nitrogen_id,
            solute_atoms,
            cutoff,
            max_bridging_waters,
            box,
            covalent_cutoff,
            hydrogen_acceptor_cutoff,
            angle_cutoff,
            target_times,
        )
    )

    extra_fields = [
        "frame_index",
        "frame_number",
        "wire_evaluated",
        "wire_connected",
        "bridging_water_count",
        "oxygen_path_atom_ids",
        "oxygen_path_edge_ids",
        "oxygen_path_edge_distances_A",
        "hbond_wire_connected",
        "hbond_bridging_water_count",
        "hbond_oxygen_path_atom_ids",
        "hbond_edge_atom_ids_D-H-A",
        "hbond_heavy_atom_distances_A",
        "hbond_H_acceptor_distances_A",
        "hbond_angles_deg",
        "hbond_mean_consecutive_angle_deg",
        "hbond_loopiness_lambda",
    ]
    fieldnames = list(rows[0]) + [x for x in extra_fields if x not in rows[0]]
    for frame_number, (row, result) in enumerate(zip(rows, results)):
        oxygen = result.oxygen
        hbond = result.hbond
        oxygen_full_path = [nitrogen_id, *oxygen.oxygen_path]
        row["frame_index"] = str(frame_number)
        row["frame_number"] = str(frame_number + 1)
        row["wire_evaluated"] = str(int(oxygen.evaluated))
        row["wire_connected"] = str(int(oxygen.connected))
        row["bridging_water_count"] = str(oxygen.bridging_water_count)
        row["oxygen_path_atom_ids"] = ";".join(map(str, oxygen.oxygen_path))
        row["oxygen_path_edge_ids"] = ";".join(
            f"{left}-{right}"
            for left, right in zip(oxygen_full_path, oxygen_full_path[1:])
        )
        row["oxygen_path_edge_distances_A"] = ";".join(
            f"{value:.6f}" for value in result.oxygen_edge_distances
        )
        row["hbond_wire_connected"] = str(int(hbond.connected))
        row["hbond_bridging_water_count"] = str(hbond.bridging_water_count)
        row["hbond_oxygen_path_atom_ids"] = ";".join(map(str, hbond.oxygen_path))
        row["hbond_edge_atom_ids_D-H-A"] = ";".join(
            f"{edge.donor_id}-{edge.hydrogen_id}-{edge.acceptor_id}"
            for edge in result.hbond_edges
        )
        row["hbond_heavy_atom_distances_A"] = ";".join(
            f"{edge.heavy_distance:.6f}" for edge in result.hbond_edges
        )
        row["hbond_H_acceptor_distances_A"] = ";".join(
            f"{edge.hydrogen_acceptor_distance:.6f}" for edge in result.hbond_edges
        )
        row["hbond_angles_deg"] = ";".join(
            f"{edge.angle_degrees:.3f}" for edge in result.hbond_edges
        )
        row["hbond_mean_consecutive_angle_deg"] = (
            f"{result.hbond_mean_consecutive_angle_deg:.8f}"
        )
        row["hbond_loopiness_lambda"] = f"{result.hbond_loopiness_lambda:.8f}"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _first_sustained_start(
    times: np.ndarray,
    condition: np.ndarray,
    persistence_ps: float,
    start_index: int = 0,
) -> int | None:
    """Return the first index beginning a continuously true time interval."""
    run_start: int | None = None
    for index in range(start_index, len(times)):
        if not (np.isfinite(times[index]) and condition[index]):
            run_start = None
            continue
        if run_start is None:
            run_start = index
        if times[index] - times[run_start] >= persistence_ps:
            return run_start
    return None


def detect_diffusive_start(
    times: np.ndarray,
    coordination: np.ndarray,
    distance: np.ndarray,
    distance_min: float = 4.0,
    deprotonated_s_max: float = 0.05,
    returned_s_min: float = 0.20,
    persistence_ps: float = 0.05,
) -> float:
    """Detect sustained deprotonation followed by sustained labeled-H return."""
    finite_distance = np.isfinite(times) & np.isfinite(distance)
    if np.count_nonzero(finite_distance) < 2:
        raise ValueError("Automatic t_diffuse detection needs at least two distances")
    distance_for_state = np.full(distance.shape, np.nan, dtype=float)
    inside_distance_range = (
        np.isfinite(times)
        & (times >= times[finite_distance][0])
        & (times <= times[finite_distance][-1])
    )
    distance_for_state[inside_distance_range] = np.interp(
        times[inside_distance_range], times[finite_distance], distance[finite_distance]
    )
    finite = (
        np.isfinite(times)
        & np.isfinite(coordination)
        & np.isfinite(distance_for_state)
    )
    escaped = finite & (distance_for_state >= distance_min)
    deprotonated = escaped & (coordination <= deprotonated_s_max)
    deprotonated_index = _first_sustained_start(
        times, deprotonated, persistence_ps
    )
    if deprotonated_index is None:
        raise ValueError(
            "Automatic t_diffuse detection found no sustained deprotonated, "
            "distance-separated state"
        )

    returned = escaped & (coordination >= returned_s_min)
    returned_index = _first_sustained_start(
        times, returned, persistence_ps, start_index=deprotonated_index + 1
    )
    if returned_index is None:
        raise ValueError(
            "Automatic t_diffuse detection found no sustained coordination "
            "recovery after deprotonation"
        )
    return float(times[returned_index])


def load_pka_series(
    fes_path: Path,
    temperature: float = 313.15,
    min1_x: float = 0.0,
    min2_x: float = 1.0,
    half_window: float = 0.1,
    fes_xmin: float = 0.0,
    fes_xmax: float = 1.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the pKa time series using the same logic as plot_pka_grid.py."""
    from plot_pka_grid import (
        PKA_FACTOR,
        deltaf,
        load_biaspot_with_restart,
        load_fes_with_restart,
    )

    cv_path = fes_path.parent
    run_dir = cv_path.parent
    cv_dir = cv_path.name
    biaspot = cv_path / "biaspot"
    if not fes_path.exists():
        raise FileNotFoundError(f"FES file not found: {fes_path}")
    if not biaspot.exists():
        raise FileNotFoundError(f"Biaspot file not found: {biaspot}")

    times = load_biaspot_with_restart(run_dir, cv_dir)
    blocks = load_fes_with_restart(run_dir, cv_dir)
    nblocks = min(len(times), len(blocks))
    if nblocks == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    times = np.asarray(times[:nblocks], dtype=float)
    delta_f = np.asarray(
        [
            deltaf(block, min1_x, min2_x, half_window, fes_xmin, fes_xmax)
            for block in blocks[:nblocks]
        ],
        dtype=float,
    )
    return times, delta_f / (PKA_FACTOR * temperature)


def save_pka_csv(path: Path, times: np.ndarray, pka: np.ndarray) -> None:
    """Save the aligned numerical data used for the pKa panel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack([times, pka]),
        delimiter=",",
        header="time_ps,pka",
        comments="",
    )


def sample_pka_window(
    times: np.ndarray,
    pka: np.ndarray,
    start_ps: float,
    end_ps: float,
    sample_count: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate equally spaced pKa samples inside a time window."""
    if sample_count < 2:
        raise ValueError("pKa window sample count must be at least 2")
    finite = np.isfinite(times) & np.isfinite(pka)
    if np.count_nonzero(finite) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    finite_times = times[finite]
    finite_pka = pka[finite]
    order = np.argsort(finite_times)
    finite_times = finite_times[order]
    finite_pka = finite_pka[order]
    if start_ps < finite_times[0] or end_ps > finite_times[-1]:
        raise ValueError(
            "The blue-to-red sampling window extends outside the pKa time series"
        )
    sample_times = np.linspace(start_ps, end_ps, sample_count)
    return sample_times, np.interp(sample_times, finite_times, finite_pka)


def save_pka_window_csv(
    path: Path,
    times: np.ndarray,
    pka: np.ndarray,
) -> None:
    """Save the equally spaced samples summarized in the pKa panel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mean = float(np.mean(pka))
    std = float(np.std(pka))
    np.savetxt(
        path,
        np.column_stack([times, pka]),
        delimiter=",",
        header=f"mean_pka={mean:.10g},std_pka={std:.10g}\ntime_ps,pka",
        comments="# ",
    )


def plot_wire_csv(
    path: Path,
    out: Path,
    cutoff: float = 3.5,
    split_regime_cutoff: float | None = None,
    covalent_cutoff: float = 1.3,
    hydrogen_acceptor_cutoff: float = 2.5,
    angle_cutoff: float = 135.0,
    diffusive_start: float | None = None,
    probability_tmax: float | None = None,
    probability_window_ps: float = 1.75,
    deprotonated_s_max: float = 0.05,
    returned_s_min: float = 0.20,
    persistence_ps: float = 0.05,
    fes_path: Path | None = None,
    pka_data_out: Path | None = None,
    temperature: float = 313.15,
    min1_x: float = 0.0,
    min2_x: float = 1.0,
    half_window: float = 0.1,
    fes_xmin: float = 0.0,
    fes_xmax: float = 1.25,
    exp_pkas: Sequence[float] = (),
    pka_window_sample_count: int = 10,
    solvation_layer_boundaries: Sequence[float] = (2.0, 3.5, 5.5, 7.5, 9.5),
    figure_width_inches: float = 14.5,
    figure_height_inches: float | None = None,
) -> None:
    """Plot wire connectivity and the aligned N--defect distance against time."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap

    layer_boundaries = np.asarray(solvation_layer_boundaries, dtype=float)
    if layer_boundaries.shape != (5,) or np.any(np.diff(layer_boundaries) <= 0.0):
        raise ValueError(
            "Solvation-layer boundaries must be five strictly increasing distances"
        )

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No data rows found in {path}")

    evaluated = np.asarray([int(row["wire_evaluated"]) == 1 for row in rows])
    connected = np.asarray([int(row["wire_connected"]) for row in rows], dtype=int)
    bridges = np.asarray([int(row["bridging_water_count"]) for row in rows], dtype=int)
    hbond_connected = np.asarray(
        [int(row["hbond_wire_connected"]) for row in rows], dtype=int
    )
    hbond_bridges = np.asarray(
        [int(row["hbond_bridging_water_count"]) for row in rows], dtype=int
    )
    hbond_mean_angle = np.asarray(
        [float(row.get("hbond_mean_consecutive_angle_deg", "nan")) for row in rows],
        dtype=float,
    )
    hbond_loopiness = np.asarray(
        [float(row.get("hbond_loopiness_lambda", "nan")) for row in rows],
        dtype=float,
    )
    if "time_ps" in rows[0]:
        x = np.asarray([float(row["time_ps"]) for row in rows], dtype=float)
        xlabel = "t (ps)"
    else:
        x = np.arange(len(rows), dtype=float)
        xlabel = "aligned frame"

    distance_columns = [
        name for name in rows[0] if name.endswith("_Odefect_distance_A")
    ]
    if not distance_columns:
        raise ValueError(f"No *_Odefect_distance_A column found in {path}")
    distance_column = distance_columns[0]
    distance = np.asarray([float(row[distance_column]) for row in rows], dtype=float)
    nitrogen_label = distance_column.split("_Odefect_distance_A", 1)[0]
    if "coordination_s" not in rows[0]:
        raise ValueError(f"Missing coordination_s column in {path}")
    coordination = np.asarray(
        [float(row["coordination_s"]) for row in rows], dtype=float
    )
    if split_regime_cutoff is not None:
        crossings = np.flatnonzero(
            np.isfinite(distance) & (distance >= split_regime_cutoff)
        )
        split_index = int(crossings[0]) if crossings.size else distance.size
        before_split = np.arange(distance.size) < split_index
        distance[before_split & ~np.isfinite(distance)] = 0.0

    if diffusive_start is None:
        if split_regime_cutoff is None:
            raise ValueError(
                "Automatic t_diffuse detection requires --split-regime-cutoff"
            )
        diffusive_start = detect_diffusive_start(
            x,
            coordination,
            distance,
            distance_min=split_regime_cutoff,
            deprotonated_s_max=deprotonated_s_max,
            returned_s_min=returned_s_min,
            persistence_ps=persistence_ps,
        )
        print(f"Auto-detected t_diffuse = {diffusive_start:.6f} ps")
    if probability_tmax is None:
        probability_tmax = diffusive_start + probability_window_ps

    wire_value = np.full(x.shape, np.nan, dtype=float)
    wire_value[evaluated & (hbond_connected == 0)] = -1.0
    wire_value[evaluated & (hbond_connected == 1)] = hbond_bridges[
        evaluated & (hbond_connected == 1)
    ]

    pka_times = np.array([], dtype=float)
    pka_values = np.array([], dtype=float)
    if fes_path is not None:
        pka_times, pka_values = load_pka_series(
            fes_path,
            temperature=temperature,
            min1_x=min1_x,
            min2_x=min2_x,
            half_window=half_window,
            fes_xmin=fes_xmin,
            fes_xmax=fes_xmax,
        )
        if pka_data_out is not None:
            save_pka_csv(pka_data_out, pka_times, pka_values)

    panel_count = 4 if fes_path is not None else 3
    if figure_height_inches is None:
        figure_height_inches = 12.0 if panel_count == 4 else 9.0
    if figure_width_inches <= 0.0 or figure_height_inches <= 0.0:
        raise ValueError("Figure width and height must be positive")
    fig, axes = plt.subplots(
        panel_count,
        2,
        figsize=(figure_width_inches, figure_height_inches),
        dpi=220,
        sharex=True,
        layout="constrained",
        gridspec_kw={
            "height_ratios": [1.1, 1.15, 1.0, 1.0][:panel_count],
            "hspace": 0.08,
            "wspace": 0.14,
        },
    )
    left_axes = axes[:, 0]
    ax_hbond, ax_distance, ax_coordination = left_axes[:3]
    ax_pka = left_axes[3] if panel_count == 4 else None
    ax_angle = axes[0, 1]
    ax_loopiness = axes[1, 1]
    for unused_axis in axes[2:, 1]:
        unused_axis.set_visible(False)
    active_axes = [*left_axes, ax_angle, ax_loopiness]

    finite_wire = np.isfinite(wire_value)
    wire_cmap = ListedColormap(
        ["#7F7F7F", "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]
    )
    wire_norm = BoundaryNorm(np.arange(-1.5, 5.5, 1.0), wire_cmap.N)
    wire_points = ax_hbond.scatter(
        x[finite_wire],
        wire_value[finite_wire],
        c=wire_value[finite_wire],
        cmap=wire_cmap,
        norm=wire_norm,
        s=16,
        alpha=0.8,
        edgecolors="none",
        rasterized=True,
    )
    wire_colorbar = fig.colorbar(
        wire_points,
        ax=ax_hbond,
        ticks=[-1, 0, 1, 2, 3, 4],
        pad=0.02,
    )
    wire_colorbar.ax.set_yticklabels(["-1", "0", "1", "2", "3", "4"])

    ax_hbond.set_ylabel("H-bond wire state")
    ax_hbond.set_yticks(
        [-1, 0, 1, 2, 3, 4],
        labels=["-1", "0", "1", "2", "3", "4"],
    )
    ax_hbond.set_ylim(-1.4, 4.4)
    ax_hbond.grid(axis="y", alpha=0.25)
    probability_window = evaluated & np.isfinite(x) & (x > diffusive_start)
    if probability_tmax is not None:
        probability_window &= x <= probability_tmax
    post_count = int(np.count_nonzero(probability_window))
    if post_count:
        connected_probability = float(
            np.count_nonzero(hbond_connected[probability_window] == 1) / post_count
        )
        disconnected_probability = 1.0 - connected_probability
        ax_hbond.text(
            0.02,
            0.95,
            (
                rf"$P_{{\rm connected}}={connected_probability:.3f}$; "
                rf"$P_{{\rm disconnected}}={disconnected_probability:.3f}$"
            ),
            transform=ax_hbond.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "none", "edgecolor": "0.75"},
        )

    finite_angle = np.isfinite(hbond_mean_angle)
    ax_angle.plot(
        x,
        hbond_mean_angle,
        color="#6A3D9A",
        linewidth=1.8,
        alpha=0.9,
    )
    ax_angle.scatter(
        x[finite_angle],
        hbond_mean_angle[finite_angle],
        color="#6A3D9A",
        s=20,
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )
    ax_angle.set_ylabel(r"mean consecutive angle, $\theta$ (deg)")
    ax_angle.set_ylim(0.0, 180.0)
    ax_angle.grid(axis="y", alpha=0.25)

    finite_loopiness = np.isfinite(hbond_loopiness)
    ax_loopiness.plot(
        x,
        hbond_loopiness,
        color="#1B7837",
        linewidth=1.8,
        alpha=0.9,
    )
    ax_loopiness.scatter(
        x[finite_loopiness],
        hbond_loopiness[finite_loopiness],
        color="#1B7837",
        s=20,
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )
    ax_loopiness.set_ylabel(r"wire loopiness, $\lambda$")
    ax_loopiness.set_ylim(0.0, 1.0)
    ax_loopiness.grid(axis="y", alpha=0.25)

    finite_distance = np.isfinite(distance)
    layer_colors = ["#DCEEFF", "#E2F3E5", "#FFF0D6", "#E6E6E6"]
    for lower, upper, color in zip(
        layer_boundaries[:-1], layer_boundaries[1:], layer_colors
    ):
        ax_distance.axhspan(
            lower,
            upper,
            color=color,
            alpha=0.75,
            linewidth=0,
            zorder=0,
        )
    layer_point_colors = ["#2F6FA5", "#4F8A62", "#B57621", "#000000"]
    in_any_layer = np.zeros(distance.shape, dtype=bool)
    for layer_index, (lower, upper, point_color) in enumerate(
        zip(layer_boundaries[:-1], layer_boundaries[1:], layer_point_colors)
    ):
        upper_comparison = (
            distance <= upper
            if layer_index == len(layer_point_colors) - 1
            else distance < upper
        )
        in_layer = finite_distance & (distance >= lower) & upper_comparison
        in_any_layer |= in_layer
        ax_distance.scatter(
            x[in_layer],
            distance[in_layer],
            color=point_color,
            s=30,
            alpha=0.95,
            edgecolors=point_color,
            linewidths=0.6,
            rasterized=True,
            zorder=2,
        )
    outside_layers = finite_distance & ~in_any_layer
    ax_distance.scatter(
        x[outside_layers],
        distance[outside_layers],
        color="#3F3F3F",
        s=30,
        alpha=0.95,
        edgecolors="#3F3F3F",
        linewidths=0.6,
        rasterized=True,
        zorder=2,
    )
    layer_cmap = ListedColormap(layer_colors)
    layer_norm = BoundaryNorm(layer_boundaries, layer_cmap.N)
    layer_colorbar = fig.colorbar(
        ScalarMappable(norm=layer_norm, cmap=layer_cmap),
        ax=ax_distance,
        boundaries=layer_boundaries,
        ticks=0.5 * (layer_boundaries[:-1] + layer_boundaries[1:]),
        spacing="proportional",
        pad=0.02,
    )
    layer_colorbar.ax.set_yticklabels(["1", "2", "3", "4"])
    layer_colorbar.set_label("solvation layer")
    ax_distance.set_ylabel(
        rf"{nitrogen_label}--O(defect$^+$) distance ($\AA$)"
    )
    ax_distance.set_ylim(
        bottom=float(layer_boundaries[0]),
        top=float(layer_boundaries[-1]),
    )
    ax_distance.grid(axis="y", alpha=0.25)

    finite_x = x[np.isfinite(x)]
    if finite_x.size:
        reaction_start = float(np.min(finite_x))
        diffusion_end = probability_tmax if probability_tmax is not None else float(np.max(finite_x))
        ax_coordination.axvspan(
            reaction_start,
            diffusive_start,
            color="#D9EEF9",
            alpha=0.65,
            linewidth=0,
            zorder=0,
        )
        ax_coordination.axvspan(
            diffusive_start,
            diffusion_end,
            color="#F8DADA",
            alpha=0.65,
            linewidth=0,
            zorder=0,
        )
        ax_coordination.text(
            reaction_start + 0.03 * (diffusive_start - reaction_start),
            0.10,
            "reaction",
            transform=ax_coordination.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#356A8A",
        )
        ax_coordination.text(
            0.5 * (diffusive_start + diffusion_end),
            0.86,
            "diffusion",
            transform=ax_coordination.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#9B4A4A",
        )
    reaction_coordination = np.where(x <= diffusive_start, coordination, np.nan)
    diffusion_coordination = np.where(x >= diffusive_start, coordination, np.nan)
    ax_coordination.plot(
        x,
        reaction_coordination,
        color="#356A8A",
        linewidth=2.2,
        alpha=0.95,
        zorder=2,
    )
    ax_coordination.plot(
        x,
        diffusion_coordination,
        color="#9B4A4A",
        linewidth=2.2,
        alpha=0.95,
        zorder=2,
    )
    ax_coordination.set_ylabel(r"coordination, $s(t)$")
    ax_coordination.set_ylim(0.0, 1.0)
    ax_coordination.grid(axis="y", alpha=0.25)
    if ax_pka is not None:
        finite_pka = np.isfinite(pka_times) & np.isfinite(pka_values)
        ax_pka.plot(
            pka_times[finite_pka],
            pka_values[finite_pka],
            color="black",
            linewidth=1.6,
        )
        ax_pka.scatter(
            pka_times[finite_pka],
            pka_values[finite_pka],
            color="#FFA500",
            edgecolor=(0.0, 0.0, 0.0, 0.35),
            s=18,
            rasterized=True,
        )
        for index, exp_pka in enumerate(exp_pkas):
            ax_pka.axhline(
                exp_pka,
                color=f"C{index % 10}",
                linewidth=1.2,
                linestyle="--",
                alpha=0.85,
            )
        if probability_tmax is not None:
            sample_times, sample_values = sample_pka_window(
                pka_times,
                pka_values,
                diffusive_start,
                probability_tmax,
                sample_count=pka_window_sample_count,
            )
            if sample_values.size:
                sample_mean = float(np.mean(sample_values))
                sample_std = float(np.std(sample_values))
                ax_pka.scatter(
                    sample_times,
                    sample_values,
                    marker="*",
                    color="#D62728",
                    edgecolor="black",
                    linewidth=0.4,
                    s=65,
                    zorder=6,
                )
                exp_pka_summary = ""
                if exp_pkas:
                    exp_values = ", ".join(f"{value:g}" for value in exp_pkas)
                    exp_pka_summary = rf"; p$K_{{a,\mathrm{{exp}}}}$ = {exp_values}"
                ax_pka.text(
                    0.02,
                    0.06,
                    rf"p$K_a$ = {sample_mean:.3f} $\pm$ {sample_std:.3f}"
                    + exp_pka_summary,
                    transform=ax_pka.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=10,
                    bbox={"facecolor": "none", "edgecolor": "0.75"},
                )
                if pka_data_out is not None:
                    sample_path = pka_data_out.with_name(
                        f"{pka_data_out.stem}_window_samples.csv"
                    )
                    save_pka_window_csv(sample_path, sample_times, sample_values)
        ax_pka.set_ylabel(r"p$K_a$")
        ax_pka.set_ylim(-20.0, 30.0)
        ax_pka.grid(axis="y", alpha=0.25)

    left_axes[-1].set_xlabel(xlabel)
    ax_loopiness.set_xlabel(xlabel)
    ax_loopiness.tick_params(axis="x", which="both", labelbottom=True)
    if probability_tmax is not None:
        finite_x = x[np.isfinite(x)]
        if finite_x.size:
            left_axes[-1].set_xlim(float(np.min(finite_x)), probability_tmax)
    for panel in active_axes:
        panel.axvline(
            diffusive_start,
            color="black",
            linestyle="-",
            linewidth=3.0,
            zorder=5,
        )
        if probability_tmax is not None:
            panel.axvline(
                probability_tmax,
                color="#D62728",
                linestyle="-",
                linewidth=3.0,
                zorder=5,
            )
    for figure_axis in fig.axes:
        figure_axis.tick_params(axis="both", which="major", labelsize=12)
        figure_axis.xaxis.label.set_fontsize(13)
        figure_axis.yaxis.label.set_fontsize(13)
        figure_axis.xaxis.get_offset_text().set_fontsize(12)
        figure_axis.yaxis.get_offset_text().set_fontsize(12)
        for annotation in figure_axis.texts:
            annotation.set_fontsize(12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find shortest N--O...O(defect) paths using a distance graph."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--traj", required=True, type=Path)
    parser.add_argument("--nitrogen-id", required=True, type=int)
    parser.add_argument("--solute-atoms", required=True, type=int)
    parser.add_argument("--cutoff", type=float, default=3.5)
    parser.add_argument("--max-bridging-waters", type=int, default=4)
    parser.add_argument("--covalent-cutoff", type=float, default=1.3)
    parser.add_argument("--hydrogen-acceptor-cutoff", type=float, default=2.5)
    parser.add_argument("--angle-cutoff", type=float, default=135.0)
    parser.add_argument(
        "--diffusive-start",
        default="auto",
        help="Diffusive onset in ps, or 'auto' (default)",
    )
    parser.add_argument(
        "--t-force-diffusion",
        "--t_force_diffusion",
        type=float,
        default=None,
        help=(
            "Force the diffusion onset to this time in ps, overriding automatic "
            "detection (alias: --t_force_diffusion)"
        ),
    )
    parser.add_argument("--probability-tmax", type=float, default=None)
    parser.add_argument("--probability-window-ps", type=float, default=1.75)
    parser.add_argument("--deprotonated-s-max", type=float, default=0.05)
    parser.add_argument("--returned-s-min", type=float, default=0.20)
    parser.add_argument("--persistence-ps", type=float, default=0.05)
    parser.add_argument(
        "--fes",
        type=Path,
        default=None,
        help="fes.dat for the fourth-row pKa series (default: beside --traj)",
    )
    parser.add_argument("--temp", type=float, default=313.15)
    parser.add_argument("--min1-x", type=float, default=0.0)
    parser.add_argument("--min2-x", type=float, default=1.0)
    parser.add_argument("--half-window", type=float, default=0.1)
    parser.add_argument("--fes-xmin", type=float, default=0.0)
    parser.add_argument("--fes-xmax", type=float, default=1.25)
    parser.add_argument("--pka-window-samples", type=int, default=10)
    parser.add_argument(
        "--figure-width-inches",
        type=float,
        default=14.5,
        help="Saved two-column figure width in inches (default: 14.5)",
    )
    parser.add_argument(
        "--figure-height-inches",
        type=float,
        default=None,
        help="Saved figure height in inches (default: 12 for four rows, 9 for three)",
    )
    parser.add_argument(
        "--solvation-layer-boundaries",
        type=float,
        nargs=5,
        default=(2.0, 3.5, 5.5, 7.5, 9.5),
        metavar=("D0", "D1", "D2", "D3", "D4"),
        help=(
            "Five increasing distance boundaries in angstrom defining the "
            "four shaded solvation layers (default: 2 3.5 5.5 7.5 9.5)"
        ),
    )
    parser.add_argument(
        "--exp-pka",
        default=None,
        help="Experimental pKa(s), comma or space separated",
    )
    parser.add_argument(
        "--split-regime-cutoff",
        type=float,
        default=None,
        help="Set missing defect distances before the first crossing to physical d=0",
    )
    parser.add_argument("--dftb-inp", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Connectivity plot path (default: --out with a .png suffix)",
    )
    parser.add_argument(
        "--pka-data-out",
        type=Path,
        default=None,
        help="Aligned pKa CSV path (default: derived from --out)",
    )
    args = parser.parse_args()

    box = None
    if args.dftb_inp is not None:
        box = read_box_lengths_from_dftb_inp(args.dftb_inp)
    augment_csv(
        input_csv=args.input_csv,
        output_csv=args.out,
        traj_path=args.traj,
        nitrogen_id=args.nitrogen_id,
        solute_atoms=args.solute_atoms,
        cutoff=args.cutoff,
        max_bridging_waters=args.max_bridging_waters,
        box=box,
        covalent_cutoff=args.covalent_cutoff,
        hydrogen_acceptor_cutoff=args.hydrogen_acceptor_cutoff,
        angle_cutoff=args.angle_cutoff,
    )
    plot_out = args.plot_out or args.out.with_suffix(".png")
    if args.t_force_diffusion is not None:
        if str(args.diffusive_start).strip().lower() != "auto":
            parser.error(
                "--t-force-diffusion cannot be combined with a numeric "
                "--diffusive-start"
            )
        diffusive_start = args.t_force_diffusion
        print(f"Forced t_diffuse = {diffusive_start:.6f} ps")
    elif str(args.diffusive_start).strip().lower() == "auto":
        diffusive_start = None
    else:
        diffusive_start = float(args.diffusive_start)
    from plot_pka_grid import parse_exp_pkas

    fes_path = args.fes or args.traj.parent / "fes.dat"
    if not fes_path.exists():
        fes_path = None
    pka_data_out = args.pka_data_out or args.out.with_name(
        f"{args.out.stem}_pka.csv"
    )
    plot_wire_csv(
        args.out,
        plot_out,
        cutoff=args.cutoff,
        split_regime_cutoff=args.split_regime_cutoff,
        covalent_cutoff=args.covalent_cutoff,
        hydrogen_acceptor_cutoff=args.hydrogen_acceptor_cutoff,
        angle_cutoff=args.angle_cutoff,
        diffusive_start=diffusive_start,
        probability_tmax=args.probability_tmax,
        probability_window_ps=args.probability_window_ps,
        deprotonated_s_max=args.deprotonated_s_max,
        returned_s_min=args.returned_s_min,
        persistence_ps=args.persistence_ps,
        fes_path=fes_path,
        pka_data_out=pka_data_out if fes_path is not None else None,
        temperature=args.temp,
        min1_x=args.min1_x,
        min2_x=args.min2_x,
        half_window=args.half_window,
        fes_xmin=args.fes_xmin,
        fes_xmax=args.fes_xmax,
        exp_pkas=parse_exp_pkas(args.exp_pka),
        pka_window_sample_count=args.pka_window_samples,
        solvation_layer_boundaries=args.solvation_layer_boundaries,
        figure_width_inches=args.figure_width_inches,
        figure_height_inches=args.figure_height_inches,
    )
    print(f"Wrote {args.out}")
    print(f"Wrote {plot_out}")
    if fes_path is not None:
        print(f"Wrote {pka_data_out}")
        print(
            "Wrote "
            + str(
                pka_data_out.with_name(
                    f"{pka_data_out.stem}_window_samples.csv"
                )
            )
        )


if __name__ == "__main__":
    main()
