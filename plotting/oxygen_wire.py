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

    frames = list(iter_xyz_frames(traj_path))
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
            yield FrameResult(missing, (), missing, ())
            continue
        if not np.isfinite(raw_defect_id):
            missing = WireResult(False, False, -1, ())
            yield FrameResult(missing, (), missing, ())
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
        yield FrameResult(
            oxygen=oxygen,
            oxygen_edge_distances=_path_edge_distances(
                coords, nitrogen_id, oxygen.oxygen_path, box
            ),
            hbond=hbond,
            hbond_edges=hbond_edges,
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
) -> None:
    """Plot wire connectivity and the aligned N--defect distance against time."""
    import matplotlib.pyplot as plt

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

    fig, (ax_hbond, ax_distance, ax_coordination) = plt.subplots(
        3,
        1,
        figsize=(9.0, 7.8),
        dpi=220,
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.1, 1.15, 1.0], "hspace": 0.08},
    )

    finite_wire = np.isfinite(wire_value)
    ax_hbond.scatter(
        x[finite_wire],
        wire_value[finite_wire],
        color="black",
        s=16,
        alpha=0.8,
        edgecolors="none",
        rasterized=True,
    )

    ax_hbond.set_ylabel("H-bond wire state")
    ax_hbond.set_yticks(
        [-1, 0, 1, 2, 3, 4],
        labels=["disconnected", "0", "1", "2", "3", "4"],
    )
    ax_hbond.set_ylim(-1.4, 4.4)
    ax_hbond.grid(axis="y", alpha=0.25)
    ax_hbond.text(
        0.02,
        0.50,
        (
            rf"short H $\leq$ {covalent_cutoff:g} $\AA$; "
            rf"H--acceptor $\leq$ {hydrogen_acceptor_cutoff:g} $\AA$; "
            rf"angle $\geq$ {angle_cutoff:g}$^\circ$"
        ),
        transform=ax_hbond.transAxes,
        ha="left",
        va="center",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    probability_window = evaluated & np.isfinite(x) & (x > diffusive_start)
    if probability_tmax is not None:
        probability_window &= x <= probability_tmax
    post_count = int(np.count_nonzero(probability_window))
    if post_count:
        connected_probability = float(
            np.count_nonzero(hbond_connected[probability_window] == 1) / post_count
        )
        disconnected_probability = 1.0 - connected_probability
        window_label = (
            rf"${diffusive_start:g}<t\leq{probability_tmax:g}$ ps"
            if probability_tmax is not None
            else rf"$t>{diffusive_start:g}$ ps"
        )
        ax_hbond.text(
            0.98,
            0.50,
            (
                window_label + ": "
                rf"$P_{{\rm connected}}={connected_probability:.3f}$; "
                rf"$P_{{\rm disconnected}}={disconnected_probability:.3f}$"
            ),
            transform=ax_hbond.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )

    finite_distance = np.isfinite(distance)
    ax_distance.scatter(
        x[finite_distance],
        distance[finite_distance],
        color="black",
        s=14,
        alpha=0.75,
        edgecolors="none",
        rasterized=True,
    )
    ax_distance.set_ylabel(
        rf"{nitrogen_label}--O(defect$^+$) distance ($\AA$)"
    )
    ax_distance.grid(axis="y", alpha=0.25)

    finite_coordination = np.isfinite(coordination)
    ax_coordination.scatter(
        x[finite_coordination],
        coordination[finite_coordination],
        color="black",
        s=14,
        alpha=0.75,
        edgecolors="none",
        rasterized=True,
    )
    ax_coordination.set_xlabel(xlabel)
    ax_coordination.set_ylabel(r"coordination, $s(t)$")
    ax_coordination.set_ylim(0.0, 1.0)
    ax_coordination.grid(axis="y", alpha=0.25)
    for panel in (ax_hbond, ax_distance, ax_coordination):
        panel.axvline(
            diffusive_start,
            color="#0072B2",
            linestyle="--",
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
    parser.add_argument("--probability-tmax", type=float, default=None)
    parser.add_argument("--probability-window-ps", type=float, default=1.75)
    parser.add_argument("--deprotonated-s-max", type=float, default=0.05)
    parser.add_argument("--returned-s-min", type=float, default=0.20)
    parser.add_argument("--persistence-ps", type=float, default=0.05)
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
    if str(args.diffusive_start).strip().lower() == "auto":
        diffusive_start = None
    else:
        diffusive_start = float(args.diffusive_start)
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
    )
    print(f"Wrote {args.out}")
    print(f"Wrote {plot_out}")


if __name__ == "__main__":
    main()
