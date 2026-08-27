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
from collections import Counter, deque
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
    hbond_orientation_signs: tuple[int, ...]
    hbond_ksi: float
    hbond_topology: str
    closest_solute_heavy_atom_id: int = -1
    closest_solute_heavy_distance_A: float = float("nan")
    defect_hbond_coordination_m: int = -1


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


def closest_solute_heavy_atom(
    coords: np.ndarray,
    defect_oxygen_id: int,
    solute_heavy_atom_ids: Sequence[int],
    box: np.ndarray | None = None,
) -> tuple[int, float]:
    """Return the one-based nearest solute heavy-atom ID and distance."""
    valid_ids = np.asarray(
        [atom_id for atom_id in solute_heavy_atom_ids if 1 <= atom_id <= len(coords)],
        dtype=int,
    )
    if not (1 <= defect_oxygen_id <= len(coords)) or valid_ids.size == 0:
        return -1, float("nan")
    delta = coords[valid_ids - 1] - coords[defect_oxygen_id - 1]
    distances = np.linalg.norm(_minimum_image(delta, box), axis=1)
    closest_index = int(np.argmin(distances))
    return int(valid_ids[closest_index]), float(distances[closest_index])


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


def _hbond_orientation_ksi(
    nitrogen_id: int,
    oxygen_path: Sequence[int],
    edges: Sequence[HBondEdge],
) -> tuple[tuple[int, ...], float]:
    """Return bond-orientation signs and ksi on the defect-to-N path.

    A sign is +1 when the donor is the hydronium-side atom of an ordered
    defect-to-N edge and -1 when it is the nitrogen-side atom.
    """
    defect_to_n_ids = list(reversed([nitrogen_id, *oxygen_path]))
    edge_by_nodes = {
        frozenset((edge.donor_id, edge.acceptor_id)): edge for edge in edges
    }
    signs: list[int] = []
    for hydronium_side, nitrogen_side in zip(
        defect_to_n_ids, defect_to_n_ids[1:]
    ):
        edge = edge_by_nodes.get(frozenset((hydronium_side, nitrogen_side)))
        if edge is None:
            raise ValueError(
                "H-bond edge metadata does not match the ordered oxygen path"
            )
        if edge.donor_id == hydronium_side:
            signs.append(1)
        elif edge.donor_id == nitrogen_side:
            signs.append(-1)
        else:
            raise ValueError("H-bond donor is not an endpoint of its path edge")
    if not signs:
        return (), float("nan")
    return tuple(signs), float(np.mean(signs))


def _water_wire_topology(signs: Sequence[int]) -> str:
    """Classify interior waters from hydronium toward N.

    P is the proper (+,+) relay, D is a double acceptor (+,-), L is a
    double donor (-,+), and R is the reverse-oriented (-,-) relay state.
    """
    labels = {
        (1, 1): "P",
        (1, -1): "D",
        (-1, 1): "L",
        (-1, -1): "R",
    }
    return "".join(labels[(int(left), int(right))] for left, right in zip(signs, signs[1:]))


def _bounded_gaussian_kde(
    values: Sequence[float],
    lower: float,
    upper: float,
    bandwidth: float | None = None,
    grid_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a reflection-corrected Gaussian KDE on a bounded interval."""
    samples = np.asarray(values, dtype=float)
    samples = samples[np.isfinite(samples)]
    if not samples.size:
        return np.array([], dtype=float), np.array([], dtype=float), float("nan")
    if not lower < upper:
        raise ValueError("KDE lower bound must be smaller than upper bound")
    if grid_points < 2:
        raise ValueError("KDE grid_points must be at least 2")
    if bandwidth is None:
        standard_deviation = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
        q25, q75 = np.percentile(samples, [25.0, 75.0])
        robust_scale = float((q75 - q25) / 1.34)
        positive_scales = [x for x in (standard_deviation, robust_scale) if x > 0.0]
        scale = min(positive_scales) if positive_scales else 0.05 * (upper - lower)
        bandwidth = 0.9 * scale * samples.size ** (-0.2)
        bandwidth = max(bandwidth, 0.01 * (upper - lower))
    if bandwidth <= 0.0:
        raise ValueError("KDE bandwidth must be positive")

    grid = np.linspace(lower, upper, grid_points)
    reflected = np.concatenate((samples, 2.0 * lower - samples, 2.0 * upper - samples))
    scaled = (grid[:, None] - reflected[None, :]) / bandwidth
    density = np.sum(np.exp(-0.5 * scaled * scaled), axis=1)
    density /= samples.size * bandwidth * np.sqrt(2.0 * np.pi)
    if hasattr(np, "trapezoid"):
        normalization = float(np.trapezoid(density, grid))
    else:
        normalization = float(np.trapz(density, grid))
    if normalization > 0.0:
        density /= normalization
    return grid, density, float(bandwidth)


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


def defect_hbond_coordination(
    coords: np.ndarray,
    nitrogen_id: int,
    defect_oxygen_id: int,
    solvent_oxygen_ids: Sequence[int],
    hydrogen_ids: Sequence[int],
    covalent_cutoff: float = 1.3,
    hydrogen_acceptor_cutoff: float = 2.5,
    angle_cutoff: float = 135.0,
    box: np.ndarray | None = None,
) -> int:
    """Count unique H-bond graph neighbors of the defect-bearing oxygen.

    This is the degree of the Mulliken-assigned defect oxygen in the same
    geometric H-bond graph used for the water-wire path.  The possible
    partners are the designated solute nitrogen and all solvent oxygens.
    """
    natoms = len(coords)
    if not (1 <= defect_oxygen_id <= natoms):
        return -1
    partner_ids = sorted(
        ({int(nitrogen_id)} | {int(x) for x in solvent_oxygen_ids})
        - {int(defect_oxygen_id)}
    )
    hydrogen_ids_array = np.asarray(hydrogen_ids, dtype=int)
    defect_xyz = coords[defect_oxygen_id - 1]
    hydrogen_xyz = coords[hydrogen_ids_array - 1]
    defect_h_vectors = _minimum_image(
        defect_xyz[None, :] - hydrogen_xyz, box
    )
    defect_h_distances = np.linalg.norm(defect_h_vectors, axis=1)
    coordination = 0
    for partner_id in partner_ids:
        partner_xyz = coords[partner_id - 1]
        heavy_distance = float(
            np.linalg.norm(_minimum_image(defect_xyz - partner_xyz, box))
        )
        if heavy_distance > covalent_cutoff + hydrogen_acceptor_cutoff:
            continue
        partner_h_vectors = _minimum_image(
            partner_xyz[None, :] - hydrogen_xyz, box
        )
        partner_h_distances = np.linalg.norm(partner_h_vectors, axis=1)
        candidates = np.flatnonzero(
            (np.minimum(defect_h_distances, partner_h_distances) <= covalent_cutoff)
            & (
                np.maximum(defect_h_distances, partner_h_distances)
                <= hydrogen_acceptor_cutoff
            )
        )
        connected = False
        for h_index in candidates:
            if defect_h_distances[h_index] <= partner_h_distances[h_index]:
                donor_vector = defect_h_vectors[h_index]
                acceptor_vector = partner_h_vectors[h_index]
            else:
                donor_vector = partner_h_vectors[h_index]
                acceptor_vector = defect_h_vectors[h_index]
            denominator = np.linalg.norm(donor_vector) * np.linalg.norm(
                acceptor_vector
            )
            if denominator == 0.0:
                continue
            cosine = float(
                np.clip(
                    np.dot(donor_vector, acceptor_vector) / denominator,
                    -1.0,
                    1.0,
                )
            )
            if float(np.degrees(np.arccos(cosine))) >= angle_cutoff:
                connected = True
                break
        coordination += int(connected)
    return coordination


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
    solute_heavy_atom_ids = [
        atom_id
        for atom_id, symbol in enumerate(symbols[:solute_atoms], start=1)
        if symbol.upper() != "H"
    ]
    if not solute_heavy_atom_ids:
        raise ValueError(f"No solute heavy atoms found in atoms 1..{solute_atoms}")

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
            yield FrameResult(
                missing,
                (),
                missing,
                (),
                float("nan"),
                float("nan"),
                (),
                float("nan"),
                "",
            )
            continue
        if not np.isfinite(raw_defect_id):
            missing = WireResult(False, False, -1, ())
            yield FrameResult(
                missing,
                (),
                missing,
                (),
                float("nan"),
                float("nan"),
                (),
                float("nan"),
                "",
            )
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
            orientation_signs, ksi = _hbond_orientation_ksi(
                nitrogen_id, hbond.oxygen_path, hbond_edges
            )
            topology = _water_wire_topology(orientation_signs)
        else:
            mean_angle, loopiness = float("nan"), float("nan")
            orientation_signs, ksi = (), float("nan")
            topology = ""
        closest_atom_id, closest_distance = closest_solute_heavy_atom(
            coords,
            int(raw_defect_id),
            solute_heavy_atom_ids,
            box,
        )
        coordination_m = defect_hbond_coordination(
            coords,
            nitrogen_id,
            int(raw_defect_id),
            solvent_oxygen_ids,
            hydrogen_ids,
            covalent_cutoff,
            hydrogen_acceptor_cutoff,
            angle_cutoff,
            box,
        )
        yield FrameResult(
            oxygen=oxygen,
            oxygen_edge_distances=_path_edge_distances(
                coords, nitrogen_id, oxygen.oxygen_path, box
            ),
            hbond=hbond,
            hbond_edges=hbond_edges,
            hbond_mean_consecutive_angle_deg=mean_angle,
            hbond_loopiness_lambda=loopiness,
            hbond_orientation_signs=orientation_signs,
            hbond_ksi=ksi,
            hbond_topology=topology,
            closest_solute_heavy_atom_id=closest_atom_id,
            closest_solute_heavy_distance_A=closest_distance,
            defect_hbond_coordination_m=coordination_m,
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
    symbols = read_xyz_symbols(traj_path)

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
        "hbond_orientation_signs_defect_to_N",
        "hbond_ksi",
        "hbond_topology_defect_to_N",
        "closest_solute_heavy_atom_id",
        "closest_solute_heavy_element",
        "closest_solute_heavy_distance_A",
        "defect_hbond_coordination_m",
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
        row["hbond_orientation_signs_defect_to_N"] = ";".join(
            f"{sign:+d}" for sign in result.hbond_orientation_signs
        )
        row["hbond_ksi"] = f"{result.hbond_ksi:.8f}"
        row["hbond_topology_defect_to_N"] = result.hbond_topology
        closest_atom_id = result.closest_solute_heavy_atom_id
        row["closest_solute_heavy_atom_id"] = (
            "" if closest_atom_id < 1 else str(closest_atom_id)
        )
        row["closest_solute_heavy_element"] = (
            "" if closest_atom_id < 1 else symbols[closest_atom_id - 1]
        )
        row["closest_solute_heavy_distance_A"] = (
            ""
            if not np.isfinite(result.closest_solute_heavy_distance_A)
            else f"{result.closest_solute_heavy_distance_A:.8f}"
        )
        row["defect_hbond_coordination_m"] = (
            ""
            if result.defect_hbond_coordination_m < 0
            else str(result.defect_hbond_coordination_m)
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


def load_fes_snapshot(
    fes_path: Path,
    target_time_ps: float,
) -> tuple[float, np.ndarray]:
    """Return the latest FES block at or before a requested timestamp."""
    from plot_pka_grid import load_biaspot_with_restart, load_fes_with_restart

    run_dir = fes_path.parent.parent
    cv_dir = fes_path.parent.name
    times = load_biaspot_with_restart(run_dir, cv_dir)
    blocks = load_fes_with_restart(run_dir, cv_dir)
    nblocks = min(len(times), len(blocks))
    if nblocks == 0:
        raise ValueError(f"No aligned FES blocks found beside {fes_path}")
    times = np.asarray(times[:nblocks], dtype=float)
    eligible = np.flatnonzero(times <= target_time_ps + 1.0e-9)
    index = int(eligible[-1]) if eligible.size else int(np.argmin(abs(times - target_time_ps)))
    block = np.asarray(blocks[index], dtype=float).copy()
    finite_energy = np.isfinite(block[:, 1])
    if np.any(finite_energy):
        block[:, 1] -= float(np.min(block[finite_energy, 1]))
    return float(times[index]), block


def save_fes_snapshot_csv(path: Path, time_ps: float, block: np.ndarray) -> None:
    """Save the FES curve displayed in the snapshot panel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        block[:, :2],
        delimiter=",",
        header=f"snapshot_time_ps={time_ps:.10g}\ns,free_energy_kcal_mol",
        comments="# ",
    )


def save_oxygen_charge_csv(
    path: Path,
    times: np.ndarray,
    charges: np.ndarray,
    oxygen_ids: Sequence[int],
) -> None:
    """Save all solvent-oxygen Mulliken series displayed in the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "time_ps," + ",".join(f"O{atom_id}" for atom_id in oxygen_ids)
    np.savetxt(
        path,
        np.column_stack([times, charges.T]),
        delimiter=",",
        header=header,
        comments="",
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
    loopiness_kde_bandwidth: float | None = None,
    figure_width_inches: float = 20.0,
    figure_height_inches: float | None = None,
    oxygen_charge_times: np.ndarray | None = None,
    oxygen_charges: np.ndarray | None = None,
    oxygen_ids: Sequence[int] = (),
    fes_snapshot_out: Path | None = None,
) -> None:
    """Plot wire connectivity and the aligned N--defect distance against time."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D

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
    hbond_loopiness = np.asarray(
        [float(row.get("hbond_loopiness_lambda", "nan")) for row in rows],
        dtype=float,
    )
    hbond_ksi = np.asarray(
        [float(row.get("hbond_ksi", "nan")) for row in rows], dtype=float
    )
    hbond_topology = np.asarray(
        [row.get("hbond_topology_defect_to_N", "") for row in rows], dtype=object
    )
    defect_oxygen_ids = np.asarray(
        [float(row.get("defect_oxygen_id") or "nan") for row in rows],
        dtype=float,
    )
    defect_coordination_m = np.asarray(
        [float(row.get("defect_hbond_coordination_m") or "nan") for row in rows],
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
    closest_solute_distance = np.asarray(
        [
            float(row.get("closest_solute_heavy_distance_A") or "nan")
            for row in rows
        ],
        dtype=float,
    )
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
    probability_window = evaluated & np.isfinite(x) & (x > diffusive_start)
    if probability_tmax is not None:
        probability_window &= x <= probability_tmax

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

    if figure_height_inches is None:
        figure_height_inches = 10.0
    if figure_width_inches <= 0.0 or figure_height_inches <= 0.0:
        raise ValueError("Figure width and height must be positive")
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(figure_width_inches, figure_height_inches),
        dpi=220,
        sharex=False,
        layout="constrained",
        gridspec_kw={
            "height_ratios": [1.1, 1.0, 1.0],
            "hspace": 0.08,
            "wspace": 0.16,
        },
    )
    ax_fes_snapshot, ax_hbond, ax_distance = axes[0]
    ax_coordination, ax_ksi, ax_oxygen_charges = axes[1]
    ax_pka, ax_loopiness, ax_m = axes[2]
    time_axes = [
        ax_distance,
        ax_hbond,
        ax_coordination,
        ax_loopiness,
        ax_oxygen_charges,
        ax_m,
    ]
    if fes_path is not None:
        time_axes.append(ax_pka)
    for time_axis in time_axes[1:]:
        time_axis.sharex(ax_distance)
    if fes_path is None:
        ax_pka.set_visible(False)
        active_axes = [
            ax_distance,
            ax_hbond,
            ax_coordination,
            ax_ksi,
            ax_loopiness,
            ax_fes_snapshot,
            ax_oxygen_charges,
            ax_m,
        ]
    else:
        active_axes = list(axes.flat)

    if fes_path is not None:
        snapshot_target = diffusive_start + probability_window_ps
        snapshot_time, snapshot_block = load_fes_snapshot(
            fes_path, snapshot_target
        )
        snapshot_finite = np.isfinite(snapshot_block[:, 0]) & np.isfinite(
            snapshot_block[:, 1]
        )
        ax_fes_snapshot.plot(
            snapshot_block[snapshot_finite, 0],
            snapshot_block[snapshot_finite, 1],
            color="#5B3A8A",
            linewidth=2.2,
        )
        ax_fes_snapshot.set_xlabel(r"coordination, $s$")
        ax_fes_snapshot.set_ylabel(r"$F(s)$ (kcal mol$^{-1}$)")
        ax_fes_snapshot.set_title(rf"$t={snapshot_time:.3f}$ ps")
        ax_fes_snapshot.set_xlim(0.0, 1.0)
        ax_fes_snapshot.grid(axis="y", alpha=0.25)
        if fes_snapshot_out is not None:
            save_fes_snapshot_csv(
                fes_snapshot_out, snapshot_time, snapshot_block
            )
    else:
        ax_fes_snapshot.set_visible(False)

    if (
        oxygen_charge_times is not None
        and oxygen_charges is not None
        and oxygen_charges.size
    ):
        charge_colors = plt.get_cmap("turbo")(
            np.linspace(0.02, 0.98, oxygen_charges.shape[0])
        )
        for series, color in zip(oxygen_charges, charge_colors):
            ax_oxygen_charges.plot(
                oxygen_charge_times,
                series,
                color=color,
                linewidth=0.55,
                alpha=0.72,
                rasterized=True,
            )
        assigned_indices = np.flatnonzero(
            np.isfinite(x) & np.isfinite(defect_oxygen_ids)
        )
        transfer_indices = assigned_indices[1:][
            defect_oxygen_ids[assigned_indices[1:]]
            != defect_oxygen_ids[assigned_indices[:-1]]
        ]
        for transfer_index in transfer_indices:
            ax_oxygen_charges.axvline(
                x[transfer_index],
                color="#5E3C99",
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                zorder=4,
            )
        ax_oxygen_charges.set_ylabel("solvent O Mulliken charge")
        ax_oxygen_charges.grid(axis="y", alpha=0.25)
    else:
        ax_oxygen_charges.text(
            0.5,
            0.5,
            "Mulliken charges unavailable",
            transform=ax_oxygen_charges.transAxes,
            ha="center",
            va="center",
        )

    finite_m = np.isfinite(x) & np.isfinite(defect_coordination_m)
    if np.any(finite_m):
        m_min = int(np.nanmin(defect_coordination_m[finite_m]))
        m_max = int(np.nanmax(defect_coordination_m[finite_m]))
        m_values = np.arange(m_min, m_max + 1)
        m_palette = plt.get_cmap("tab10")(np.arange(len(m_values)) % 10)
        m_cmap = ListedColormap(m_palette)
        m_norm = BoundaryNorm(
            np.arange(m_min - 0.5, m_max + 1.5), m_cmap.N
        )
        m_points = ax_m.scatter(
            x[finite_m],
            defect_coordination_m[finite_m],
            c=defect_coordination_m[finite_m],
            cmap=m_cmap,
            norm=m_norm,
            s=20,
            alpha=0.85,
            edgecolors="none",
            rasterized=True,
        )
        m_colorbar = fig.colorbar(
            m_points, ax=ax_m, ticks=m_values, pad=0.02
        )
        m_colorbar.ax.set_yticklabels([str(value) for value in m_values])
        ax_m.set_yticks(m_values)
        ax_m.set_ylim(m_min - 0.45, m_max + 0.45)
        m_window = probability_window & finite_m
        if np.any(m_window):
            window_values, window_counts = np.unique(
                defect_coordination_m[m_window].astype(int),
                return_counts=True,
            )
            window_probabilities = window_counts / np.sum(window_counts)
            color_by_m = {
                int(value): m_palette[index]
                for index, value in enumerate(m_values)
            }
            m_histogram = ax_m.inset_axes([0.07, 0.57, 0.40, 0.35])
            m_histogram.bar(
                window_values,
                window_probabilities,
                width=0.72,
                color=[color_by_m[int(value)] for value in window_values],
                edgecolor="black",
                linewidth=0.6,
            )
            m_histogram.set_xticks(window_values)
            m_histogram.set_ylim(
                0.0, max(1.0, 1.12 * float(np.max(window_probabilities)))
            )
            m_histogram.text(
                0.96,
                0.92,
                r"$P(m)$",
                transform=m_histogram.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                fontweight="bold",
            )
            m_histogram.text(
                0.50,
                0.06,
                r"$m$",
                transform=m_histogram.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
            m_histogram.tick_params(axis="both", which="major", labelsize=8)
            m_histogram.set_facecolor((1.0, 1.0, 1.0, 0.92))
    ax_m.set_ylabel(r"defect H-bond coordination, $m$")
    ax_m.grid(axis="y", alpha=0.25)

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

    finite_ksi = np.isfinite(hbond_ksi)
    ksi_window = probability_window & finite_ksi
    if np.any(ksi_window):
        rounded_ksi = np.round(hbond_ksi[ksi_window], decimals=8)
        ksi_values, ksi_counts = np.unique(rounded_ksi, return_counts=True)
        ksi_probability = ksi_counts / np.sum(ksi_counts)
        if ksi_values.size > 1:
            minimum_spacing = float(np.min(np.diff(ksi_values)))
            bar_width = min(0.16, 0.75 * minimum_spacing)
        else:
            bar_width = 0.12
        ax_ksi.bar(
            ksi_values,
            ksi_probability,
            width=bar_width,
            color="#B39DDB",
            edgecolor="#4A2A78",
            linewidth=0.8,
        )
        for ksi_value, probability in zip(ksi_values, ksi_probability):
            matching_topologies = hbond_topology[
                ksi_window & np.isclose(hbond_ksi, ksi_value, atol=5.0e-8)
            ]
            topology_counts = Counter(
                str(label) if str(label) else "direct"
                for label in matching_topologies
            )
            topology_label = "/".join(
                label for label, _count in topology_counts.most_common()
            )
            ax_ksi.text(
                ksi_value,
                probability + 0.025,
                topology_label,
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#3B1F5E",
            )
        ax_ksi.set_xticks(ksi_values)
        ax_ksi.set_xticklabels(
            [f"{value:.2f}".rstrip("0").rstrip(".") for value in ksi_values],
            rotation=35,
            ha="right",
        )
        ax_ksi.set_ylim(0.0, max(1.0, 1.12 * float(np.max(ksi_probability))))
    ax_ksi.set_xlim(-1.05, 1.05)
    ax_ksi.set_xlabel(r"wire conductivity, $\xi$")
    ax_ksi.set_ylabel(r"$P(\xi)$")
    ax_ksi.grid(axis="y", alpha=0.25)

    finite_loopiness = np.isfinite(hbond_loopiness)
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

    loopiness_histogram = None
    loopiness_window = probability_window & finite_loopiness
    if np.any(loopiness_window):
        lambda_grid, lambda_density, _lambda_bandwidth = _bounded_gaussian_kde(
            hbond_loopiness[loopiness_window],
            lower=0.0,
            upper=1.0,
            bandwidth=loopiness_kde_bandwidth,
        )
        loopiness_histogram = ax_loopiness.inset_axes([0.07, 0.57, 0.40, 0.35])
        loopiness_histogram.plot(
            lambda_grid,
            lambda_density,
            color="#145A2A",
            linewidth=1.8,
        )
        loopiness_histogram.fill_between(
            lambda_grid,
            0.0,
            lambda_density,
            color="#7FC98B",
            alpha=0.65,
            linewidth=0.0,
        )
        loopiness_histogram.set_xlim(0.0, 1.0)
        loopiness_histogram.text(
            0.96,
            0.92,
            r"$P(\lambda)$",
            transform=loopiness_histogram.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
        loopiness_histogram.text(
            0.50,
            0.06,
            r"$\lambda$",
            transform=loopiness_histogram.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        loopiness_histogram.tick_params(axis="both", which="major", labelsize=8)
        loopiness_histogram.set_facecolor((1.0, 1.0, 1.0, 0.92))

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
    finite_closest_solute = np.isfinite(x) & np.isfinite(closest_solute_distance)
    ax_distance.scatter(
        x[finite_closest_solute],
        closest_solute_distance[finite_closest_solute],
        color="#7B3294",
        s=20,
        alpha=0.95,
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )
    ax_distance.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#4F8A62",
                marker="o",
                linestyle="none",
                markersize=7,
                label=rf"{nitrogen_label}--defect",
            ),
            Line2D(
                [],
                [],
                color="#7B3294",
                marker="o",
                linestyle="none",
                markersize=6,
                label="nearest solute heavy atom--defect",
            ),
        ],
        loc="upper left",
        frameon=False,
        fontsize=11,
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
        bottom=0.0,
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
    if fes_path is not None:
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

    left_bottom_axis = ax_pka if fes_path is not None else ax_coordination
    left_bottom_axis.set_xlabel(xlabel)
    if fes_path is None:
        ax_coordination.tick_params(axis="x", which="both", labelbottom=True)
    ax_loopiness.set_xlabel(xlabel)
    ax_m.set_xlabel(xlabel)
    ax_loopiness.tick_params(axis="x", which="both", labelbottom=True)
    ax_distance.tick_params(axis="x", which="both", labelbottom=False)
    ax_hbond.tick_params(axis="x", which="both", labelbottom=False)
    ax_oxygen_charges.tick_params(axis="x", which="both", labelbottom=False)
    if fes_path is not None:
        ax_coordination.tick_params(axis="x", which="both", labelbottom=False)
    if probability_tmax is not None:
        finite_x = x[np.isfinite(x)]
        if finite_x.size:
            ax_distance.set_xlim(float(np.min(finite_x)), probability_tmax)
    for panel in time_axes:
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
        if figure_axis is loopiness_histogram:
            continue
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
        help="fes.dat for the lower-left pKa series (default: beside --traj)",
    )
    parser.add_argument("--temp", type=float, default=313.15)
    parser.add_argument("--min1-x", type=float, default=0.0)
    parser.add_argument("--min2-x", type=float, default=1.0)
    parser.add_argument("--half-window", type=float, default=0.1)
    parser.add_argument("--fes-xmin", type=float, default=0.0)
    parser.add_argument("--fes-xmax", type=float, default=1.25)
    parser.add_argument("--pka-window-samples", type=int, default=10)
    parser.add_argument(
        "--loopiness-kde-bandwidth",
        type=float,
        default=None,
        help=(
            "Gaussian KDE bandwidth for P(lambda), in dimensionless lambda "
            "units (default: automatic Silverman-style estimate)"
        ),
    )
    parser.add_argument(
        "--figure-width-inches",
        type=float,
        default=20.0,
        help="Saved three-column figure width in inches (default: 20)",
    )
    parser.add_argument(
        "--figure-height-inches",
        type=float,
        default=None,
        help="Saved three-row figure height in inches (default: 10)",
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
    parser.add_argument(
        "--mulliken",
        type=Path,
        default=None,
        help="Mulliken file for the solvent-oxygen charge panel (default: beside --traj)",
    )
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
    parser.add_argument(
        "--oxygen-charge-data-out",
        type=Path,
        default=None,
        help="CSV for plotted solvent-oxygen charges (default: derived from --out)",
    )
    parser.add_argument(
        "--fes-snapshot-data-out",
        type=Path,
        default=None,
        help="CSV for FES(s) at t_diffuse + window (default: derived from --out)",
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
    charge_data_out = args.oxygen_charge_data_out or args.out.with_name(
        f"{args.out.stem}_oxygen_charges.csv"
    )
    fes_snapshot_out = args.fes_snapshot_data_out or args.out.with_name(
        f"{args.out.stem}_fes_snapshot.csv"
    )
    mulliken_path = args.mulliken or args.traj.parent / "mulliken"
    oxygen_charge_times: np.ndarray | None = None
    oxygen_charges: np.ndarray | None = None
    oxygen_ids: Sequence[int] = ()
    if mulliken_path.exists():
        if diffusive_start is not None:
            from plot_coord_prod_grid import parse_mulliken_limited

            charge_result = parse_mulliken_limited(
                mulliken_path,
                args.solute_atoms,
                diffusive_start + args.probability_window_ps,
            )
        else:
            from defect_identification import parse_mulliken

            charge_result = parse_mulliken(
                mulliken_path,
                args.solute_atoms,
            )
        oxygen_charge_times, oxygen_charges, oxygen_ids, _elements = charge_result
        save_oxygen_charge_csv(
            charge_data_out,
            oxygen_charge_times,
            oxygen_charges,
            oxygen_ids,
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
        loopiness_kde_bandwidth=args.loopiness_kde_bandwidth,
        figure_width_inches=args.figure_width_inches,
        figure_height_inches=args.figure_height_inches,
        oxygen_charge_times=oxygen_charge_times,
        oxygen_charges=oxygen_charges,
        oxygen_ids=oxygen_ids,
        fes_snapshot_out=fes_snapshot_out if fes_path is not None else None,
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
        print(f"Wrote {fes_snapshot_out}")
    if oxygen_charge_times is not None:
        print(f"Wrote {charge_data_out}")


if __name__ == "__main__":
    main()
