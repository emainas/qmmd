#!/usr/bin/env python3
"""Detect water-mediated solute tautomerization and make a 3x2 report.

The product tautomer is assigned from persistent covalent protonation of a
configured solute acceptor while the metadynamics donor N--H bond is broken.
Mulliken charges support and illustrate the assignment; they do not define it.
All atom IDs exposed by the CLI and written to reports are one-based.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from oxygen_wire import (
    _hbond_orientation_ksi,
    _minimum_image,
    _water_wire_topology,
    iter_xyz_frames,
    read_box_lengths_from_dftb_inp,
    read_xyz_symbols,
    shortest_hbond_path,
)
from plot_pka_grid import (
    deltaf,
    load_biaspot_with_restart,
    load_fes_with_restart,
)


TIME_RE = re.compile(r"T=\s*([0-9.+\-EeDd]+)\s*FSEC", re.IGNORECASE)
RUN_RE = re.compile(r"^run-(\d+)$")
BV_DONOR_IDS = {
    "meta-hia": (1, 2),
    "meta-hib": (19, 20),
    "meta-hic": (31, 32),
    "meta-hid": (43, 44),
}
BV_LACTAM_LABELS = {11: "A", 53: "D"}
BV_LACTAM_IDS = tuple(BV_LACTAM_LABELS)
SYSTEM_SPECS = {
    "BV": {
        "solute_atoms": 78,
        "donors": BV_DONOR_IDS,
        "lactams": BV_LACTAM_LABELS,
        "acceptors": {11: "lactam-A", 53: "lactam-D"},
        "mulliken_roles": {},
    },
    "APP": {
        "solute_atoms": 77,
        "donors": {"meta-hib": (18, 19)},
        "lactams": {10: "A", 52: "D"},
        "acceptors": {1: "N-A", 10: "lactam-A", 52: "lactam-D"},
        "mulliken_roles": {
            1: "N$_A$", 18: "N$_B$", 30: "N$_C$", 42: "N$_D$",
            19: "H$_B$", 31: "H$_C$", 43: "H$_D$",
            10: "O$_A$", 52: "O$_D$",
        },
    },
    "BPP": {
        "solute_atoms": 77,
        "donors": {"meta-hic": (30, 31)},
        "lactams": {11: "A", 52: "D"},
        "acceptors": {19: "N-B", 11: "lactam-A", 52: "lactam-D"},
        "mulliken_roles": {
            1: "N$_A$", 19: "N$_B$", 30: "N$_C$", 42: "N$_D$",
            2: "H$_A$", 31: "H$_C$", 43: "H$_D$",
            11: "O$_A$", 52: "O$_D$",
        },
    },
    "CPP": {
        "solute_atoms": 77,
        "donors": {"meta-hib": (19, 20)},
        "lactams": {11: "A", 52: "D"},
        "acceptors": {31: "N-C", 11: "lactam-A", 52: "lactam-D"},
        "mulliken_roles": {
            1: "N$_A$", 19: "N$_B$", 31: "N$_C$", 42: "N$_D$",
            2: "H$_A$", 20: "H$_B$", 43: "H$_D$",
            11: "O$_A$", 52: "O$_D$",
        },
    },
    "DPP": {
        "solute_atoms": 77,
        "donors": {"meta-hic": (31, 32)},
        "lactams": {11: "A", 52: "D"},
        "acceptors": {43: "N-D", 11: "lactam-A", 52: "lactam-D"},
        "mulliken_roles": {
            1: "N$_A$", 19: "N$_B$", 31: "N$_C$", 43: "N$_D$",
            2: "H$_A$", 20: "H$_B$", 32: "H$_C$",
            11: "O$_A$", 52: "O$_D$",
        },
    },
}


@dataclass(frozen=True)
class ProtonationEvent:
    acceptor_id: int
    hydrogen_id: int
    start_ps: float
    end_ps: float
    duration_ps: float


@dataclass(frozen=True)
class FrameResult:
    time_ps: float
    trajectory_frame: int
    donor_h_distance_A: float
    acceptor_h_distance_A: float
    acceptor_hydrogen_id: int
    acceptor_protonated: bool
    wire_connected: bool
    bridging_waters: int
    ksi: float
    topology: str
    path_atom_ids: tuple[int, ...]
    edge_atom_ids: tuple[tuple[int, int, int], ...]


def parse_run_ids(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    result: set[int] = set()
    for token in re.split(r"[,\s]+", raw.strip()):
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            if start < 1 or stop < start:
                raise argparse.ArgumentTypeError(f"invalid run range: {token}")
            result.update(range(start, stop + 1))
        else:
            value = int(token)
            if value < 1:
                raise argparse.ArgumentTypeError("run IDs must be positive")
            result.add(value)
    return result


def parse_atom_ids(raw: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in re.split(r"[,\s]+", raw.strip()) if item)
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("atom IDs must be positive one-based integers")
    return values


def infer_bv_donor(cv_dir: str) -> tuple[int, int]:
    try:
        return BV_DONOR_IDS[cv_dir]
    except KeyError as exc:
        raise ValueError(
            f"No BV donor mapping for {cv_dir!r}; provide --donor-id and "
            "--designated-h-id"
        ) from exc


def infer_system(runs_path: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit.upper()
    upper_parts = {part.upper() for part in runs_path.parts}
    matches = [name for name in SYSTEM_SPECS if name in upper_parts]
    if len(matches) != 1:
        raise ValueError("Cannot infer system from --runs-path; provide --system")
    return matches[0]


def contiguous_events(
    times: np.ndarray,
    mask: np.ndarray,
    hydrogen_ids: np.ndarray,
    acceptor_id: int,
) -> list[ProtonationEvent]:
    events: list[ProtonationEvent] = []
    start: int | None = None
    extended = np.r_[mask, False]
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    for index, active in enumerate(extended):
        if active and start is None:
            start = index
        elif not active and start is not None:
            segment_ids = hydrogen_ids[start:index]
            finite_ids = segment_ids[segment_ids > 0]
            if finite_ids.size:
                ids, counts = np.unique(finite_ids, return_counts=True)
                hydrogen_id = int(ids[int(np.argmax(counts))])
            else:
                hydrogen_id = -1
            events.append(
                ProtonationEvent(
                    acceptor_id=acceptor_id,
                    hydrogen_id=hydrogen_id,
                    start_ps=float(times[start]),
                    end_ps=float(times[index - 1]),
                    duration_ps=float(times[index - 1] - times[start] + dt),
                )
            )
            start = None
    return events


def scan_geometry(
    traj: Path,
    donor_id: int,
    designated_h_id: int,
    acceptor_ids: Sequence[int],
    box: np.ndarray | None,
    donor_break_A: float,
    acceptor_bond_A: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    list[ProtonationEvent],
]:
    symbols = read_xyz_symbols(traj)
    hydrogen_ids = np.asarray(
        [index for index, symbol in enumerate(symbols, start=1) if symbol.upper() == "H"],
        dtype=int,
    )
    times: list[float] = []
    donor_distances: list[float] = []
    acceptor_distances = {atom_id: [] for atom_id in acceptor_ids}
    nearest_hydrogens = {atom_id: [] for atom_id in acceptor_ids}
    for time_ps, coords in iter_xyz_frames(traj):
        if time_ps is None:
            raise ValueError(f"Trajectory frame lacks a timestamp: {traj}")
        times.append(float(time_ps))
        donor_delta = _minimum_image(
            coords[designated_h_id - 1] - coords[donor_id - 1], box
        )
        donor_distances.append(float(np.linalg.norm(donor_delta)))
        h_xyz = coords[hydrogen_ids - 1]
        for acceptor_id in acceptor_ids:
            delta = _minimum_image(h_xyz - coords[acceptor_id - 1], box)
            distances = np.linalg.norm(delta, axis=1)
            closest = int(np.argmin(distances))
            acceptor_distances[acceptor_id].append(float(distances[closest]))
            nearest_hydrogens[acceptor_id].append(int(hydrogen_ids[closest]))
    time_array = np.asarray(times, dtype=float)
    donor_array = np.asarray(donor_distances, dtype=float)
    distance_arrays = {
        atom_id: np.asarray(values, dtype=float)
        for atom_id, values in acceptor_distances.items()
    }
    hydrogen_arrays = {
        atom_id: np.asarray(values, dtype=int)
        for atom_id, values in nearest_hydrogens.items()
    }
    donor_broken = donor_array >= donor_break_A
    events: list[ProtonationEvent] = []
    for acceptor_id in acceptor_ids:
        mask = donor_broken & (distance_arrays[acceptor_id] <= acceptor_bond_A)
        events.extend(
            contiguous_events(
                time_array, mask, hydrogen_arrays[acceptor_id], acceptor_id
            )
        )
    return time_array, donor_array, distance_arrays, hydrogen_arrays, events


def select_event(
    events: Sequence[ProtonationEvent], minimum_lifetime_ps: float
) -> ProtonationEvent | None:
    persistent = [event for event in events if event.duration_ps >= minimum_lifetime_ps]
    return max(persistent, key=lambda event: event.duration_ps, default=None)


def analyze_wire(
    traj: Path,
    donor_id: int,
    acceptor_id: int,
    solute_atoms: int,
    box: np.ndarray | None,
    covalent_cutoff_A: float,
    hydrogen_acceptor_cutoff_A: float,
    angle_cutoff_deg: float,
    max_bridging_waters: int,
    protonation_distances: np.ndarray,
    protonation_hydrogens: np.ndarray,
    acceptor_bond_A: float,
) -> list[FrameResult]:
    symbols = read_xyz_symbols(traj)
    solvent_oxygen_ids = [
        atom_id
        for atom_id, symbol in enumerate(symbols, start=1)
        if atom_id > solute_atoms and symbol.upper() == "O"
    ]
    hydrogen_ids = [
        atom_id
        for atom_id, symbol in enumerate(symbols, start=1)
        if symbol.upper() == "H"
    ]
    results: list[FrameResult] = []
    for frame_index, ((time_ps, coords), acceptor_distance, acceptor_h_id) in enumerate(
        zip(iter_xyz_frames(traj), protonation_distances, protonation_hydrogens)
    ):
        if time_ps is None:
            raise ValueError(f"Trajectory frame lacks a timestamp: {traj}")
        wire, edges = shortest_hbond_path(
            coords=coords,
            nitrogen_id=donor_id,
            defect_oxygen_id=acceptor_id,
            solvent_oxygen_ids=solvent_oxygen_ids,
            hydrogen_ids=hydrogen_ids,
            covalent_cutoff=covalent_cutoff_A,
            hydrogen_acceptor_cutoff=hydrogen_acceptor_cutoff_A,
            angle_cutoff=angle_cutoff_deg,
            max_bridging_waters=max_bridging_waters,
            box=box,
        )
        donor_h_distance = float("nan")
        if edges:
            try:
                signs, ksi = _hbond_orientation_ksi(
                    donor_id, wire.oxygen_path, edges
                )
                topology = _water_wire_topology(signs)
            except ValueError:
                ksi = float("nan")
                topology = ""
        else:
            ksi = float("nan")
            topology = ""
        results.append(
            FrameResult(
                time_ps=float(time_ps),
                trajectory_frame=frame_index,
                donor_h_distance_A=donor_h_distance,
                acceptor_h_distance_A=float(acceptor_distance),
                acceptor_hydrogen_id=int(acceptor_h_id),
                acceptor_protonated=bool(acceptor_distance <= acceptor_bond_A),
                wire_connected=wire.connected,
                bridging_waters=wire.bridging_water_count if wire.connected else -1,
                ksi=ksi,
                topology=topology,
                path_atom_ids=(donor_id, *wire.oxygen_path) if wire.connected else (),
                edge_atom_ids=tuple(
                    (edge.donor_id, edge.hydrogen_id, edge.acceptor_id)
                    for edge in edges
                ),
            )
        )
    return results


def parse_bias_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    values: list[float] = []
    current_time: float | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = TIME_RE.search(line)
            if match:
                current_time = float(match.group(1).replace("D", "E")) / 1000.0
            elif "Coordinate" in line and "=" in line and current_time is not None:
                values.append(float(line.rsplit("=", 1)[1]))
                times.append(current_time)
                current_time = None
    return np.asarray(times), np.asarray(values)


def load_bias_coordinates_with_restart(
    run_dir: Path, cv_dir: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load s(t) with the same restart time convention as the FES series."""
    base_path = run_dir / cv_dir / "biaspot"
    base_times, base_values = parse_bias_coordinates(base_path)
    restart_path = run_dir / cv_dir / "metad-restart" / "biaspot"
    if not restart_path.exists() or base_times.size == 0:
        return base_times, base_values
    restart_times, restart_values = parse_bias_coordinates(restart_path)
    if restart_times.size == 0:
        return base_times, base_values
    last_base = float(base_times[-1])
    if float(restart_times[-1]) > last_base:
        keep = restart_times > last_base
        restart_times = restart_times[keep]
        restart_values = restart_values[keep]
    else:
        restart_times = restart_times + last_base
    return (
        np.concatenate([base_times, restart_times]),
        np.concatenate([base_values, restart_values]),
    )


def parse_target_mulliken(
    path: Path, target_ids: Sequence[int]
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    targets = tuple(dict.fromkeys(map(int, target_ids)))
    times: list[float] = []
    values = {atom_id: [] for atom_id in targets}
    current_time: float | None = None
    current = {atom_id: 0.0 for atom_id in targets}

    def finish() -> None:
        if current_time is None:
            return
        times.append(current_time)
        for atom_id in targets:
            values[atom_id].append(current[atom_id])

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = TIME_RE.search(line)
            if match:
                finish()
                current_time = float(match.group(1).replace("D", "E")) / 1000.0
                current = {atom_id: 0.0 for atom_id in targets}
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    atom_id = int(parts[0])
                    value = float(parts[3].replace("D", "E"))
                except ValueError:
                    continue
                if atom_id in targets:
                    current[atom_id] += value
    finish()
    return np.asarray(times), {
        atom_id: np.asarray(atom_values, dtype=float)
        for atom_id, atom_values in values.items()
    }


def initial_negative_defect_oxygen(
    path: Path,
    symbols: Sequence[str],
    solute_atoms: int,
) -> int:
    """Return the one-based solvent O ID with the lowest initial Mulliken charge."""
    solvent_oxygen_ids = {
        atom_id
        for atom_id, symbol in enumerate(symbols, start=1)
        if atom_id > solute_atoms and symbol.upper() == "O"
    }
    charges = {atom_id: 0.0 for atom_id in solvent_oxygen_ids}
    in_first_frame = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if TIME_RE.search(line):
                if in_first_frame:
                    break
                in_first_frame = True
                continue
            if not in_first_frame:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                atom_id = int(parts[0])
                value = float(parts[3].replace("D", "E"))
            except ValueError:
                continue
            if atom_id in charges:
                charges[atom_id] += value
    if not in_first_frame or not charges:
        raise ValueError(f"Cannot identify the initial negative defect in {path}")
    return min(charges, key=charges.__getitem__)


def positive_defect_oxygen_ids(
    path: Path,
    symbols: Sequence[str],
    solute_atoms: int,
    charge_min: float,
    charge_max: float,
) -> tuple[int, ...]:
    """Return solvent O IDs entering the configured hydronium charge window."""
    oxygen_ids = {
        atom_id
        for atom_id, symbol in enumerate(symbols, start=1)
        if atom_id > solute_atoms and symbol.upper() == "O"
    }
    detected: set[int] = set()
    current = {atom_id: 0.0 for atom_id in oxygen_ids}
    in_frame = False

    def finish_frame() -> None:
        if not in_frame:
            return
        detected.update(
            atom_id
            for atom_id, charge in current.items()
            if charge_min <= charge <= charge_max
        )

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if TIME_RE.search(line):
                finish_frame()
                in_frame = True
                current = {atom_id: 0.0 for atom_id in oxygen_ids}
                continue
            if not in_frame:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                atom_id = int(parts[0])
                value = float(parts[3].replace("D", "E"))
            except ValueError:
                continue
            if atom_id in current:
                current[atom_id] += value
    finish_frame()
    return tuple(sorted(detected))


def charge_shift_at(
    times: np.ndarray,
    values: np.ndarray,
    event_start_ps: float,
    window_ps: float,
) -> float:
    """Return a post-minus-pre median Mulliken shift at an event."""
    pre = (times >= event_start_ps - window_ps) & (times < event_start_ps)
    post = (times >= event_start_ps) & (times <= event_start_ps + window_ps)
    if not np.any(pre) or not np.any(post):
        return float("nan")
    return float(np.median(values[post]) - np.median(values[pre]))


def first_persistent_start(
    times: np.ndarray, mask: np.ndarray, minimum_lifetime_ps: float
) -> float | None:
    dummy_hydrogens = np.ones(times.shape, dtype=int)
    events = contiguous_events(times, mask, dummy_hydrogens, acceptor_id=-1)
    persistent = [event for event in events if event.duration_ps >= minimum_lifetime_ps]
    return persistent[0].start_ps if persistent else None


def detect_diffusive_regime(
    times: np.ndarray,
    coordination: np.ndarray,
    deprotonated_s_max: float,
    returned_s_min: float,
    deprotonation_persistence_ps: float,
    recovery_persistence_ps: float,
) -> tuple[float, float] | None:
    """Require sustained low s followed by sustained recovery, as in acid-base scans."""
    low_start = first_persistent_start(
        times,
        coordination <= deprotonated_s_max,
        deprotonation_persistence_ps,
    )
    if low_start is None:
        return None
    after_low = times > low_start
    recovery_start = first_persistent_start(
        times,
        after_low & (coordination >= returned_s_min),
        recovery_persistence_ps,
    )
    if recovery_start is None:
        return None
    return low_start, recovery_start


def free_energy_data(
    run_dir: Path,
    cv_dir: str,
    min1_x: float,
    min2_x: float,
    half_window: float,
    fes_xmin: float,
    fes_xmax: float,
    snapshot_target_ps: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    times = load_biaspot_with_restart(run_dir, cv_dir)
    blocks = load_fes_with_restart(run_dir, cv_dir)
    count = min(len(times), len(blocks))
    if count == 0:
        raise ValueError(f"No aligned bias/FES blocks in {run_dir / cv_dir}")
    delta_f = np.asarray(
        [
            deltaf(block, min1_x, min2_x, half_window, fes_xmin, fes_xmax)
            for block in blocks[:count]
        ]
    )
    aligned_times = np.asarray(times[:count], dtype=float)
    eligible = np.flatnonzero(aligned_times <= snapshot_target_ps + 1.0e-9)
    snapshot_index = (
        int(eligible[-1])
        if eligible.size
        else int(np.argmin(np.abs(aligned_times - snapshot_target_ps)))
    )
    snapshot = np.asarray(blocks[snapshot_index], dtype=float).copy()
    finite = np.isfinite(snapshot[:, 1])
    if np.any(finite):
        snapshot[:, 1] -= float(np.min(snapshot[finite, 1]))
    return (
        aligned_times,
        delta_f,
        float(aligned_times[snapshot_index]),
        snapshot[:, 0],
        snapshot[:, 1],
    )


def write_frame_csv(path: Path, frames: Sequence[FrameResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "time_ps", "trajectory_frame", "acceptor_h_distance_A",
        "acceptor_hydrogen_id", "acceptor_protonated", "wire_connected",
        "bridging_waters", "ksi", "topology", "path_atom_ids", "edge_atom_ids_D-H-A",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in frames:
            writer.writerow(
                {
                    "time_ps": f"{frame.time_ps:.8f}",
                    "trajectory_frame": frame.trajectory_frame,
                    "acceptor_h_distance_A": f"{frame.acceptor_h_distance_A:.8f}",
                    "acceptor_hydrogen_id": frame.acceptor_hydrogen_id,
                    "acceptor_protonated": int(frame.acceptor_protonated),
                    "wire_connected": int(frame.wire_connected),
                    "bridging_waters": frame.bridging_waters,
                    "ksi": "" if not np.isfinite(frame.ksi) else f"{frame.ksi:.8f}",
                    "topology": frame.topology,
                    "path_atom_ids": ";".join(map(str, frame.path_atom_ids)),
                    "edge_atom_ids_D-H-A": ";".join(
                        f"{d}-{h}-{a}" for d, h, a in frame.edge_atom_ids
                    ),
                }
            )


def save_numeric_csv(path: Path, header: str, columns: Sequence[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=header,
        comments="",
    )


def sample_delta_f_window(
    times: np.ndarray,
    values: np.ndarray,
    start_ps: float,
    end_ps: float,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate equally spaced Delta-F samples in the diffusion window."""
    if sample_count < 2:
        raise ValueError("Delta-F window sample count must be at least 2")
    finite = np.isfinite(times) & np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    finite_times = times[finite]
    finite_values = values[finite]
    order = np.argsort(finite_times)
    finite_times = finite_times[order]
    finite_values = finite_values[order]
    if start_ps < finite_times[0] or end_ps > finite_times[-1]:
        raise ValueError(
            "The diffusion sampling window extends outside the Delta-F series"
        )
    sample_times = np.linspace(start_ps, end_ps, sample_count)
    return sample_times, np.interp(sample_times, finite_times, finite_values)


def make_report(
    output: Path,
    run_label: str,
    donor_id: int,
    designated_h_id: int,
    acceptor_id: int,
    negative_defect_oxygen_id: int,
    lactam_labels: dict[int, str],
    mulliken_role_labels: dict[int, str],
    positive_defect_oxygen_ids: Sequence[int],
    event: ProtonationEvent,
    diffusive_start_ps: float,
    cv_times: np.ndarray,
    cv_values: np.ndarray,
    df_times: np.ndarray,
    delta_f: np.ndarray,
    snapshot_time_ps: float,
    fes_s: np.ndarray,
    fes_values: np.ndarray,
    frames: Sequence[FrameResult],
    mulliken_times: np.ndarray,
    mulliken: dict[int, np.ndarray],
    symbols: Sequence[str],
    probability_window_ps: float,
    delta_f_sample_count: int,
    figure_width_inches: float,
    figure_height_inches: float,
) -> tuple[np.ndarray, np.ndarray]:
    probability_tmax = diffusive_start_ps + probability_window_ps
    fig, axes = plt.subplots(
        3,
        2,
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
    ax_fes, ax_mull = axes[0]
    ax_cv, ax_wire = axes[1]
    ax_df, ax_ksi = axes[2]
    time_axes = [ax_mull, ax_cv, ax_wire, ax_df]
    for axis in time_axes[1:]:
        axis.sharex(ax_mull)

    ax_fes.plot(fes_s, fes_values, color="#5B3A8A", linewidth=2.2)
    ax_fes.set_xlabel(r"coordination, $s$")
    ax_fes.set_ylabel(r"$F(s)$ (kcal mol$^{-1}$)")
    ax_fes.set_xlim(0.0, 1.0)
    ax_fes.set_title(rf"$t={snapshot_time_ps:.3f}$ ps")
    ax_fes.grid(axis="y", alpha=0.25)

    charge_ids = list(mulliken)
    special_ids = {
        donor_id,
        designated_h_id,
        *lactam_labels,
        *mulliken_role_labels,
        negative_defect_oxygen_id,
        *positive_defect_oxygen_ids,
    }
    water_ids = [
        atom_id
        for atom_id in charge_ids
        if atom_id not in special_ids
    ]
    water_colors = iter(
        plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(1, len(water_ids))))
    )
    special_colors = {
        **{
            atom_id: plt.get_cmap("tab10")(index % 10)
            for index, atom_id in enumerate(mulliken_role_labels)
        },
        donor_id: "#0072B2",
        designated_h_id: "#E69F00",
        **{
            atom_id: color
            for atom_id, color in zip(lactam_labels, ("#009E73", "#CC79A7"))
        },
        negative_defect_oxygen_id: "#000000",
        **{atom_id: "#D55E00" for atom_id in positive_defect_oxygen_ids},
    }
    for atom_id in charge_ids:
        role = f" {mulliken_role_labels[atom_id]}" if atom_id in mulliken_role_labels else ""
        if atom_id == donor_id:
            role += " donor"
        elif atom_id == designated_h_id:
            role += " biased"
        if atom_id == acceptor_id:
            role += " acceptor"
        elif atom_id == negative_defect_oxygen_id:
            role += " initial negative-defect O"
        elif atom_id in positive_defect_oxygen_ids:
            role += " positive-defect O"
        color = (
            special_colors[atom_id]
            if atom_id in special_colors
            else next(water_colors)
        )
        ax_mull.plot(
            mulliken_times,
            mulliken[atom_id],
            linewidth=1.7 if role else 0.8,
            alpha=0.95 if role else 0.72,
            color=color,
            label=f"{symbols[atom_id - 1]}{atom_id}{role}",
            rasterized=not bool(role),
        )
    ax_mull.set_ylabel("Mulliken charge (s+p)")
    ax_mull.legend(loc="upper left", fontsize=9, ncol=2, frameon=False)
    ax_mull.grid(axis="y", alpha=0.25)

    finite_cv = np.isfinite(cv_times) & np.isfinite(cv_values)
    reaction = finite_cv & (cv_times <= diffusive_start_ps)
    diffusion = finite_cv & (cv_times >= diffusive_start_ps)
    time_min = float(np.min(cv_times[finite_cv]))
    ax_cv.axvspan(
        time_min, diffusive_start_ps, color="#D9EEF9", alpha=0.65,
        linewidth=0, zorder=0,
    )
    ax_cv.axvspan(
        diffusive_start_ps, probability_tmax, color="#F8DADA", alpha=0.65,
        linewidth=0, zorder=0,
    )
    ax_cv.plot(
        cv_times[reaction], cv_values[reaction], color="#356A8A",
        linewidth=2.2, alpha=0.95,
    )
    ax_cv.plot(
        cv_times[diffusion], cv_values[diffusion], color="#9B4A4A",
        linewidth=2.2, alpha=0.95,
    )
    ax_cv.text(
        time_min + 0.03 * (diffusive_start_ps - time_min), 0.10, "reaction",
        transform=ax_cv.get_xaxis_transform(), ha="left", va="bottom",
        fontsize=10, fontweight="bold", color="#356A8A",
    )
    ax_cv.text(
        0.5 * (diffusive_start_ps + probability_tmax), 0.86, "diffusion",
        transform=ax_cv.get_xaxis_transform(), ha="center", va="top",
        fontsize=10, fontweight="bold", color="#9B4A4A",
    )
    ax_cv.set_ylabel(r"coordination, $s(t)$")
    ax_cv.set_ylim(0.0, 1.0)
    ax_cv.grid(axis="y", alpha=0.25)

    frame_times = np.asarray([frame.time_ps for frame in frames])
    bridge_values = np.asarray([frame.bridging_waters for frame in frames])
    finite_wire = np.isfinite(frame_times) & np.isfinite(bridge_values)
    wire_min = -1
    wire_max = max(0, int(np.max(bridge_values[finite_wire])))
    wire_values = np.arange(wire_min, wire_max + 1)
    base_colors = ["#7F7F7F", "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]
    if len(wire_values) > len(base_colors):
        base_colors.extend(
            plt.get_cmap("viridis")(np.linspace(0.2, 0.9, len(wire_values) - len(base_colors)))
        )
    wire_cmap = ListedColormap(base_colors[: len(wire_values)])
    wire_norm = BoundaryNorm(
        np.arange(wire_min - 0.5, wire_max + 1.5), wire_cmap.N
    )
    wire_points = ax_wire.scatter(
        frame_times[finite_wire], bridge_values[finite_wire],
        c=bridge_values[finite_wire], cmap=wire_cmap, norm=wire_norm,
        s=16, alpha=0.8, edgecolors="none", rasterized=True,
    )
    wire_colorbar = fig.colorbar(
        wire_points, ax=ax_wire, ticks=wire_values, pad=0.02
    )
    wire_colorbar.ax.set_yticklabels([str(value) for value in wire_values])
    ax_wire.set_ylabel("H-bond wire state")
    ax_wire.set_yticks(wire_values)
    ax_wire.set_ylim(wire_min - 0.4, wire_max + 0.4)
    ax_wire.grid(axis="y", alpha=0.25)
    wire_window = (
        finite_wire
        & (frame_times > diffusive_start_ps)
        & (frame_times <= probability_tmax)
    )
    if np.any(wire_window):
        p_connected = float(np.mean(bridge_values[wire_window] >= 0))
        ax_wire.text(
            0.02, 0.95,
            rf"$P_{{\rm connected}}={p_connected:.3f}$; "
            rf"$P_{{\rm disconnected}}={1.0-p_connected:.3f}$",
            transform=ax_wire.transAxes, ha="left", va="top", fontsize=10,
            bbox={"facecolor": "none", "edgecolor": "0.75"},
        )

    finite_df = np.isfinite(df_times) & np.isfinite(delta_f)
    ax_df.plot(df_times[finite_df], delta_f[finite_df], color="black", linewidth=1.6)
    ax_df.scatter(
        df_times[finite_df], delta_f[finite_df], color="#FFA500",
        edgecolor=(0.0, 0.0, 0.0, 0.35), s=18, rasterized=True,
    )
    ax_df.axhline(0.0, color="0.5", linestyle="--", linewidth=1.2)
    sample_times, sample_values = sample_delta_f_window(
        df_times,
        delta_f,
        diffusive_start_ps,
        probability_tmax,
        delta_f_sample_count,
    )
    if sample_values.size:
        sample_mean = float(np.mean(sample_values))
        sample_std = float(np.std(sample_values))
        ax_df.scatter(
            sample_times,
            sample_values,
            marker="*",
            color="#D62728",
            edgecolor="black",
            linewidth=0.4,
            s=65,
            zorder=6,
        )
        ax_df.text(
            0.02,
            0.06,
            rf"$\Delta F = {sample_mean:.3f} \pm {sample_std:.3f}$ kcal mol$^{{-1}}$",
            transform=ax_df.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            bbox={"facecolor": "none", "edgecolor": "0.75"},
        )
    ax_df.set_xlabel(r"$t$ (ps)")
    ax_df.set_ylabel(r"$\Delta F(t)$ (kcal mol$^{-1}$)")
    ax_df.grid(axis="y", alpha=0.25)

    ksi = np.asarray([frame.ksi for frame in frames])
    topology = np.asarray([frame.topology for frame in frames], dtype=object)
    ksi_window = (
        np.isfinite(ksi)
        & (frame_times > diffusive_start_ps)
        & (frame_times <= probability_tmax)
    )
    if np.any(ksi_window):
        ksi_values, ksi_counts = np.unique(np.round(ksi[ksi_window], 8), return_counts=True)
        probabilities = ksi_counts / np.sum(ksi_counts)
        spacing = float(np.min(np.diff(ksi_values))) if ksi_values.size > 1 else 0.16
        bar_width = min(0.16, 0.75 * spacing)
        ax_ksi.bar(
            ksi_values, probabilities, width=bar_width, color="#B39DDB",
            edgecolor="#4A2A78", linewidth=0.8,
        )
        for value, probability in zip(ksi_values, probabilities):
            labels, counts = np.unique(
                topology[ksi_window & np.isclose(ksi, value, atol=5.0e-8)],
                return_counts=True,
            )
            order = np.argsort(counts)[::-1]
            label = "/".join(str(labels[index]) or "direct" for index in order)
            ax_ksi.text(
                value, probability + 0.025, label, rotation=90,
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#3B1F5E",
            )
        ax_ksi.set_xticks(ksi_values)
        ax_ksi.set_xticklabels(
            [f"{value:.2f}".rstrip("0").rstrip(".") for value in ksi_values],
            rotation=35, ha="right",
        )
        ax_ksi.set_ylim(0.0, max(1.0, 1.12 * float(np.max(probabilities))))
    ax_ksi.set_xlim(-1.05, 1.05)
    ax_ksi.set_xlabel(r"wire conductivity, $\xi$")
    ax_ksi.set_ylabel(r"$P(\xi)$")
    ax_ksi.grid(axis="y", alpha=0.25)

    ax_mull.set_xlim(time_min, probability_tmax)
    for axis in time_axes:
        axis.axvline(
            diffusive_start_ps, color="black", linestyle="-", linewidth=3.0, zorder=5
        )
        axis.axvline(
            probability_tmax, color="#D62728", linestyle="-", linewidth=3.0, zorder=5
        )
    ax_mull.tick_params(axis="x", which="both", labelbottom=False)
    ax_cv.tick_params(axis="x", which="both", labelbottom=False)
    ax_wire.tick_params(axis="x", which="both", labelbottom=False)
    for figure_axis in fig.axes:
        figure_axis.tick_params(axis="both", which="major", labelsize=12)
        figure_axis.xaxis.label.set_fontsize(13)
        figure_axis.yaxis.label.set_fontsize(13)
        figure_axis.xaxis.get_offset_text().set_fontsize(12)
        figure_axis.yaxis.get_offset_text().set_fontsize(12)
        for annotation in figure_axis.texts:
            annotation.set_fontsize(12)
    fig.suptitle(
        f"{run_label}: {symbols[donor_id - 1]}{donor_id} → "
        f"{symbols[acceptor_id - 1]}{acceptor_id} tautomerization "
        f"({event.duration_ps:.2f} ps)",
        fontsize=15,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return sample_times, sample_values


def discover_run_dirs(
    runs_path: Path, cv_dir: str, run_ids: set[int] | None
) -> Iterator[tuple[int, Path]]:
    for candidate in sorted(
        runs_path.glob("run-*"),
        key=lambda path: int(path.name.split("-")[-1])
        if path.name.split("-")[-1].isdigit()
        else 10**9,
    ):
        match = RUN_RE.match(candidate.name)
        if not match:
            continue
        run_id = int(match.group(1))
        if run_ids is not None and run_id not in run_ids:
            continue
        target = candidate / cv_dir
        if (target / "traject").exists() and (target / "mulliken").exists():
            yield run_id, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-path", required=True, type=Path)
    parser.add_argument("--cv-dir", required=True)
    parser.add_argument("--run-ids", default=None)
    parser.add_argument("--system", choices=tuple(SYSTEM_SPECS), default=None)
    parser.add_argument("--solute-atoms", type=int, default=None)
    parser.add_argument("--donor-id", type=int, default=None)
    parser.add_argument("--designated-h-id", type=int, default=None)
    parser.add_argument("--acceptor-ids", type=parse_atom_ids, default=None)
    parser.add_argument("--donor-break", type=float, default=1.35)
    parser.add_argument("--acceptor-bond", type=float, default=1.20)
    parser.add_argument("--minimum-lifetime", type=float, default=0.25)
    parser.add_argument("--deprotonated-s-max", type=float, default=0.05)
    parser.add_argument("--returned-s-min", type=float, default=0.20)
    parser.add_argument("--deprotonation-persistence", type=float, default=0.25)
    parser.add_argument("--recovery-persistence", type=float, default=0.05)
    parser.add_argument(
        "--charge-window",
        type=float,
        default=0.50,
        help="Pre/post-event Mulliken median window in ps (default: 0.50)",
    )
    parser.add_argument(
        "--minimum-charge-shift",
        type=float,
        default=0.08,
        help=(
            "Minimum magnitude for the expected donor decrease and acceptor "
            "increase in Mulliken charge (default: 0.08)"
        ),
    )
    parser.add_argument("--covalent-cutoff", type=float, default=1.30)
    parser.add_argument("--hydrogen-acceptor-cutoff", type=float, default=2.50)
    parser.add_argument("--angle-cutoff", type=float, default=135.0)
    parser.add_argument("--max-bridging-waters", type=int, default=4)
    parser.add_argument("--positive-defect-charge-min", type=float, default=-0.625)
    parser.add_argument("--positive-defect-charge-max", type=float, default=-0.525)
    parser.add_argument("--min1-x", type=float, default=0.0)
    parser.add_argument("--min2-x", type=float, default=1.0)
    parser.add_argument("--half-window", type=float, default=0.1)
    parser.add_argument("--fes-xmin", type=float, default=0.0)
    parser.add_argument("--fes-xmax", type=float, default=1.25)
    parser.add_argument("--probability-window-ps", type=float, default=1.75)
    parser.add_argument("--delta-f-window-samples", type=int, default=10)
    parser.add_argument("--figure-width-inches", type=float, default=14.0)
    parser.add_argument("--figure-height-inches", type=float, default=11.0)
    parser.add_argument(
        "--style",
        type=Path,
        default=Path(__file__).resolve().parent / "prl.mplstyle",
    )
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Write only summary PNGs; do not write aligned CSV or scan files",
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--categorize-transitions",
        action="store_true",
        help="Place figures/data in <cv-dir>_to_<acceptor-site> subdirectories",
    )
    parser.add_argument(
        "--scan-out",
        type=Path,
        default=None,
        help="Scan-table path (default: reports-dir/<system>_solv_4.0_<cv>_tautomer-scan.csv)",
    )
    args = parser.parse_args()

    try:
        system_name = infer_system(args.runs_path, args.system)
    except ValueError as exc:
        parser.error(str(exc))
    system_spec = SYSTEM_SPECS[system_name]
    solute_atoms = (
        args.solute_atoms
        if args.solute_atoms is not None
        else int(system_spec["solute_atoms"])
    )
    lactam_labels = dict(system_spec["lactams"])
    acceptor_labels = dict(system_spec["acceptors"])
    acceptor_ids = (
        args.acceptor_ids
        if args.acceptor_ids is not None
        else tuple(acceptor_labels)
    )
    mulliken_role_labels = dict(system_spec["mulliken_roles"])

    if args.style.exists():
        plt.style.use(args.style)

    if args.deprotonated_s_max >= args.returned_s_min:
        parser.error("--deprotonated-s-max must be less than --returned-s-min")
    if args.positive_defect_charge_min >= args.positive_defect_charge_max:
        parser.error(
            "--positive-defect-charge-min must be less than "
            "--positive-defect-charge-max"
        )
    if min(
        args.minimum_lifetime,
        args.deprotonation_persistence,
        args.recovery_persistence,
        args.charge_window,
        args.minimum_charge_shift,
    ) < 0.0:
        parser.error("persistence times, charge window, and charge shift must be non-negative")
    if (
        args.probability_window_ps <= 0.0
        or args.figure_width_inches <= 0.0
        or args.figure_height_inches <= 0.0
    ):
        parser.error("probability window and figure dimensions must be positive")
    if args.delta_f_window_samples < 2:
        parser.error("--delta-f-window-samples must be at least 2")

    run_ids = parse_run_ids(args.run_ids)
    if (args.donor_id is None) != (args.designated_h_id is None):
        parser.error("provide both --donor-id and --designated-h-id")
    donor_id, designated_h_id = (
        (args.donor_id, args.designated_h_id)
        if args.donor_id is not None
        else system_spec["donors"].get(args.cv_dir, (None, None))
    )
    if donor_id is None or designated_h_id is None:
        parser.error(
            f"No donor mapping for {system_name} {args.cv_dir}; provide "
            "--donor-id and --designated-h-id"
        )
    rows: list[dict[str, object]] = []
    for run_id, cv_path in discover_run_dirs(args.runs_path, args.cv_dir, run_ids):
        cv_times, cv_values = load_bias_coordinates_with_restart(
            cv_path.parent, args.cv_dir
        )
        diffusion = detect_diffusive_regime(
            cv_times,
            cv_values,
            args.deprotonated_s_max,
            args.returned_s_min,
            args.deprotonation_persistence,
            args.recovery_persistence,
        )
        if diffusion is None:
            rows.append(
                {
                    "run_id": run_id,
                    "cv_dir": args.cv_dir,
                    "classification": "no_diffusive_regime",
                    "donor_id": donor_id,
                    "designated_h_id": designated_h_id,
                    "deprotonated_start_ps": "",
                    "diffusive_start_ps": "",
                    "acceptor_id": "",
                    "acceptor_hydrogen_id": "",
                    "event_start_ps": "",
                    "event_end_ps": "",
                    "event_duration_ps": "",
                    "donor_charge_shift": "",
                    "acceptor_charge_shift": "",
                }
            )
            print(f"run {run_id}: no diffusive s(t) regime")
            continue
        deprotonated_start_ps, diffusive_start_ps = diffusion
        probability_tmax = diffusive_start_ps + args.probability_window_ps
        if cv_times.size == 0 or float(np.max(cv_times)) < probability_tmax - 1.0e-9:
            rows.append(
                {
                    "run_id": run_id,
                    "cv_dir": args.cv_dir,
                    "classification": "insufficient_post_diffusion_window",
                    "donor_id": donor_id,
                    "designated_h_id": designated_h_id,
                    "deprotonated_start_ps": f"{deprotonated_start_ps:.8f}",
                    "diffusive_start_ps": f"{diffusive_start_ps:.8f}",
                    "acceptor_id": "",
                    "acceptor_hydrogen_id": "",
                    "event_start_ps": "",
                    "event_end_ps": "",
                    "event_duration_ps": "",
                    "donor_charge_shift": "",
                    "acceptor_charge_shift": "",
                }
            )
            print(
                f"run {run_id}: insufficient data through "
                f"t_diffuse + {args.probability_window_ps:g} ps"
            )
            continue
        box_path = cv_path / "dftb.inp"
        box = read_box_lengths_from_dftb_inp(box_path) if box_path.exists() else None
        geometry = scan_geometry(
            cv_path / "traject",
            donor_id,
            designated_h_id,
            acceptor_ids,
            box,
            args.donor_break,
            args.acceptor_bond,
        )
        times, donor_distances, acceptor_distances, acceptor_hydrogens, events = geometry
        event = select_event(events, args.minimum_lifetime)
        donor_charge_shift = float("nan")
        acceptor_charge_shift = float("nan")
        if event is not None:
            screening_times, screening_charges = parse_target_mulliken(
                cv_path / "mulliken", (donor_id, event.acceptor_id)
            )
            donor_event_start = first_persistent_start(
                times, donor_distances >= args.donor_break, args.minimum_lifetime
            )
            donor_charge_shift = (
                charge_shift_at(
                    screening_times,
                    screening_charges[donor_id],
                    donor_event_start,
                    args.charge_window,
                )
                if donor_event_start is not None
                else float("nan")
            )
            acceptor_charge_shift = charge_shift_at(
                screening_times,
                screening_charges[event.acceptor_id],
                event.start_ps,
                args.charge_window,
            )
            charge_confirmed = (
                donor_charge_shift <= -args.minimum_charge_shift
                and acceptor_charge_shift >= args.minimum_charge_shift
            )
        else:
            charge_confirmed = False
        classification = (
            "tautomer"
            if event is not None and charge_confirmed
            else "geometry_only_candidate"
            if event is not None
            else "no_persistent_lactam_tautomer"
        )
        row = {
            "run_id": run_id,
            "cv_dir": args.cv_dir,
            "classification": classification,
            "donor_id": donor_id,
            "designated_h_id": designated_h_id,
            "deprotonated_start_ps": f"{deprotonated_start_ps:.8f}",
            "diffusive_start_ps": f"{diffusive_start_ps:.8f}",
            "acceptor_id": event.acceptor_id if event else "",
            "acceptor_hydrogen_id": event.hydrogen_id if event else "",
            "event_start_ps": f"{event.start_ps:.8f}" if event else "",
            "event_end_ps": f"{event.end_ps:.8f}" if event else "",
            "event_duration_ps": f"{event.duration_ps:.8f}" if event else "",
            "donor_charge_shift": (
                f"{donor_charge_shift:.8f}" if np.isfinite(donor_charge_shift) else ""
            ),
            "acceptor_charge_shift": (
                f"{acceptor_charge_shift:.8f}"
                if np.isfinite(acceptor_charge_shift)
                else ""
            ),
        }
        rows.append(row)
        print(
            f"run {run_id}: "
            + (
                f"{classification}: N{donor_id} -> "
                f"{mulliken_role_labels.get(event.acceptor_id, f'atom-{event.acceptor_id}')}, "
                f"{event.start_ps:.3f}-{event.end_ps:.3f} ps "
                f"({event.duration_ps:.3f} ps), "
                f"dq(N)={donor_charge_shift:+.3f}, "
                f"dq(acceptor)={acceptor_charge_shift:+.3f}"
                if event
                else "no persistent lactam tautomer"
            )
        )
        if args.scan_only or event is None or not charge_confirmed:
            continue

        frames = analyze_wire(
            cv_path / "traject",
            donor_id,
            event.acceptor_id,
            solute_atoms,
            box,
            args.covalent_cutoff,
            args.hydrogen_acceptor_cutoff,
            args.angle_cutoff,
            args.max_bridging_waters,
            acceptor_distances[event.acceptor_id],
            acceptor_hydrogens[event.acceptor_id],
            args.acceptor_bond,
        )
        connected_frames = [frame for frame in frames if frame.wire_connected]
        transition_wire = min(
            connected_frames,
            key=lambda frame: abs(frame.time_ps - event.start_ps),
            default=None,
        )
        wire_water_ids = (
            [
                atom_id
                for atom_id in transition_wire.path_atom_ids
                if atom_id > solute_atoms
            ]
            if transition_wire is not None
            else []
        )
        symbols = read_xyz_symbols(cv_path / "traject")
        negative_defect_oxygen_id = initial_negative_defect_oxygen(
            cv_path / "mulliken", symbols, solute_atoms
        )
        positive_oxygen_ids = positive_defect_oxygen_ids(
            cv_path / "mulliken",
            symbols,
            solute_atoms,
            args.positive_defect_charge_min,
            args.positive_defect_charge_max,
        )
        charge_ids = list(
            dict.fromkeys(
                [
                    donor_id,
                    designated_h_id,
                    *mulliken_role_labels,
                    negative_defect_oxygen_id,
                    *positive_oxygen_ids,
                    *wire_water_ids,
                ]
            )
        )
        mull_times, mulliken = parse_target_mulliken(cv_path / "mulliken", charge_ids)
        snapshot_target_ps = diffusive_start_ps + args.probability_window_ps
        df_times, df_values, snapshot_time_ps, fes_s, fes_values = free_energy_data(
            cv_path.parent,
            args.cv_dir,
            args.min1_x,
            args.min2_x,
            args.half_window,
            args.fes_xmin,
            args.fes_xmax,
            snapshot_target_ps,
        )
        stem = f"{system_name}_solv_4.0_{args.cv_dir}_run-{run_id}_tautomer-summary"
        transition_dir = args.reports_dir
        if args.categorize_transitions:
            acceptor_label = acceptor_labels.get(
                event.acceptor_id, f"atom-{event.acceptor_id}"
            )
            transition_dir = (
                args.reports_dir / f"{args.cv_dir}_to_{acceptor_label}"
            )
        if not args.png_only:
            frame_out = transition_dir / f"{stem}.csv"
            write_frame_csv(frame_out, frames)
            save_numeric_csv(
                transition_dir / f"{stem}_delta-f.csv",
                "time_ps,delta_f_kcal_mol",
                (df_times, df_values),
            )
            save_numeric_csv(
                transition_dir / f"{stem}_fes.csv",
                "s,free_energy_kcal_mol",
                (fes_s, fes_values),
            )
            save_numeric_csv(
                transition_dir / f"{stem}_coordination.csv",
                "time_ps,s",
                (cv_times, cv_values),
            )
            charge_columns = [mull_times, *[mulliken[atom_id] for atom_id in charge_ids]]
            save_numeric_csv(
                transition_dir / f"{stem}_mulliken.csv",
                "time_ps," + ",".join(f"atom_{atom_id}" for atom_id in charge_ids),
                charge_columns,
            )
        sample_times, sample_values = make_report(
            transition_dir / f"{stem}.png",
            f"{system_name} {args.cv_dir} run {run_id}",
            donor_id,
            designated_h_id,
            event.acceptor_id,
            negative_defect_oxygen_id,
            lactam_labels,
            mulliken_role_labels,
            positive_oxygen_ids,
            event,
            diffusive_start_ps,
            cv_times,
            cv_values,
            df_times,
            df_values,
            snapshot_time_ps,
            fes_s,
            fes_values,
            frames,
            mull_times,
            mulliken,
            symbols,
            args.probability_window_ps,
            args.delta_f_window_samples,
            args.figure_width_inches,
            args.figure_height_inches,
        )
        if sample_values.size and not args.png_only:
            save_numeric_csv(
                transition_dir / f"{stem}_delta-f_window_samples.csv",
                (
                    f"mean_delta_f_kcal_mol={float(np.mean(sample_values)):.10g},"
                    f"std_delta_f_kcal_mol={float(np.std(sample_values)):.10g}"
                    "\ntime_ps,delta_f_kcal_mol"
                ),
                (sample_times, sample_values),
            )

    scan_out = args.scan_out or (
        args.reports_dir / f"{system_name}_solv_4.0_{args.cv_dir}_tautomer-scan.csv"
    )
    if not args.png_only:
        scan_out.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with scan_out.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(f"Wrote {scan_out}")


if __name__ == "__main__":
    main()
