#!/usr/bin/env python3
"""Copied HIST/PRD geometry helpers for CPP NB19 and NC31 comparison."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from acid_base_CPP import (
    _minimum_image,
    _signed_dihedral_deg,
    iter_xyz_frames,
    read_xyz_symbols,
    shortest_hbond_path,
)


HIST_DIHEDRAL_ATOM_NAMES = {
    "chi1": ("N", "CA", "CB", "CG"),
    "chi2": ("CA", "CB", "CG", "ND1"),
}
HIST_COMPETITOR_LABELS = ("NB", "NC")


def classify_histidine_conformers(
    chi1_deg: np.ndarray, chi2_deg: np.ndarray
) -> np.ndarray:
    """Assign the six established ``tp/tm/gpp/gpm/gmp/gmm`` rotamers.

    Chi1 is trans for absolute angles of at least 120 degrees, gauche-plus
    from 0 to 120 degrees, and gauche-minus from -120 to 0 degrees.  The final
    ``p`` or ``m`` records the sign of chi2.
    """
    chi1 = (np.asarray(chi1_deg, dtype=float) + 180.0) % 360.0 - 180.0
    chi2 = (np.asarray(chi2_deg, dtype=float) + 180.0) % 360.0 - 180.0
    if chi1.shape != chi2.shape:
        raise ValueError("chi1 and chi2 must have matching shapes")
    labels = np.full(chi1.shape, "", dtype=object)
    finite = np.isfinite(chi1) & np.isfinite(chi2)
    chi1_state = np.full(chi1.shape, "", dtype=object)
    chi1_state[finite & (np.abs(chi1) >= 120.0)] = "t"
    chi1_state[finite & (chi1 >= 0.0) & (chi1 < 120.0)] = "gp"
    chi1_state[finite & (chi1 < 0.0) & (chi1 > -120.0)] = "gm"
    chi2_state = np.where(chi2 >= 0.0, "p", "m")
    labels[finite] = np.char.add(
        chi1_state[finite].astype(str), chi2_state[finite].astype(str)
    )
    return labels


def read_amber_atom_names(parm7: Path, solute_atoms: int) -> list[str]:
    """Read the first ``solute_atoms`` Amber atom names in one-based order."""
    names: list[str] = []
    in_atom_names = False
    for line in parm7.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("%FLAG "):
            if in_atom_names:
                break
            in_atom_names = line.strip() == "%FLAG ATOM_NAME"
            continue
        if not in_atom_names or line.startswith("%FORMAT"):
            continue
        names.extend(
            line[index : index + 4].strip()
            for index in range(0, len(line), 4)
            if line[index : index + 4].strip()
        )
    names = names[:solute_atoms]
    if len(names) != solute_atoms:
        raise ValueError(
            f"Could not read {solute_atoms} histidine atom names from {parm7}"
        )
    return names


def resolve_unique_atom_ids(
    parm7: Path, solute_atoms: int, required_names: Sequence[str]
) -> dict[str, int]:
    """Resolve unique one-based solute atom IDs by Amber atom name."""
    names = read_amber_atom_names(parm7, solute_atoms)
    resolved: dict[str, int] = {}
    for name in required_names:
        matches = [index + 1 for index, value in enumerate(names) if value == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one histidine atom named {name}; found {matches}")
        resolved[name] = matches[0]
    return resolved


def histidine_definitions(
    parm7: Path, solute_atoms: int
) -> tuple[dict[str, tuple[int, ...]], dict[str, int]]:
    """Return dihedral quartets and all histidine N/O competitor endpoints."""
    required = {
        name for quartet in HIST_DIHEDRAL_ATOM_NAMES.values() for name in quartet
    } | set(HIST_COMPETITOR_LABELS)
    atom_ids = resolve_unique_atom_ids(parm7, solute_atoms, sorted(required))
    quartets = {
        label: tuple(atom_ids[name] for name in names)
        for label, names in HIST_DIHEDRAL_ATOM_NAMES.items()
    }
    endpoints = {label: atom_ids[label] for label in HIST_COMPETITOR_LABELS}
    return quartets, endpoints


def _time_tolerance(target_times: np.ndarray) -> float:
    # Targets are actual XYZ timestamps, not nearest-neighbor time bins.
    return 1.0e-8


def calculate_histidine_dihedrals(
    traj_path: Path,
    target_times: np.ndarray,
    quartets: dict[str, tuple[int, ...]],
    box: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Calculate timestamp-aligned histidine chi1 and chi2 dihedrals."""
    series = {
        label: np.full(target_times.shape, np.nan, dtype=float) for label in quartets
    }
    tolerance = _time_tolerance(target_times)
    target_index = 0
    for frame_time, coords in iter_xyz_frames(traj_path):
        if frame_time is None:
            raise ValueError(f"Trajectory frame in {traj_path} has no timestamp")
        while (
            target_index < len(target_times)
            and target_times[target_index] < frame_time - tolerance
        ):
            target_index += 1
        if target_index >= len(target_times):
            break
        if abs(target_times[target_index] - frame_time) > tolerance:
            continue
        for label, atom_ids in quartets.items():
            indices = np.asarray(atom_ids, dtype=int) - 1
            series[label][target_index] = _signed_dihedral_deg(coords[indices], box)
        target_index += 1
    return series


def rational_coordination(
    distances: np.ndarray,
    refdist: float = 1.6,
    nexp: int = 6,
    mexp: int = 12,
) -> float:
    """Return the DCDFTBMD rational coordination sum for one heavy atom."""
    scaled = distances / refdist
    numerator = 1.0 - np.power(scaled, nexp)
    denominator = 1.0 - np.power(scaled, mexp)
    values = np.empty_like(scaled, dtype=float)
    singular = np.isclose(denominator, 0.0)
    values[~singular] = numerator[~singular] / denominator[~singular]
    values[singular] = nexp / mexp
    return float(np.sum(values))


def calculate_histidine_competitor_geometry(
    traj_path: Path,
    target_times: np.ndarray,
    defect_oxygen_ids: np.ndarray,
    endpoint_ids: dict[str, int],
    solute_atoms: int,
    box: np.ndarray | None,
    max_bridging_waters: int = 3,
    covalent_cutoff: float = 1.3,
    hydrogen_acceptor_cutoff: float = 2.5,
    angle_cutoff: float = 135.0,
    coordination_refdist: float = 1.6,
    coordination_nexp: int = 6,
    coordination_mexp: int = 12,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Calculate defect wires and all-H coordination for every histidine N/O."""
    symbols = read_xyz_symbols(traj_path)
    solvent_oxygen_ids = [
        atom_id
        for atom_id, symbol in enumerate(symbols, start=1)
        if atom_id > solute_atoms and symbol.upper() == "O"
    ]
    hydrogen_ids = np.asarray(
        [
            atom_id
            for atom_id, symbol in enumerate(symbols, start=1)
            if symbol.upper() == "H"
        ],
        dtype=int,
    )
    wires = {
        label: np.full(target_times.shape, np.nan, dtype=float)
        for label in HIST_COMPETITOR_LABELS
    }
    coordination = {
        label: np.full(target_times.shape, np.nan, dtype=float)
        for label in HIST_COMPETITOR_LABELS
    }
    tolerance = _time_tolerance(target_times)
    target_index = 0
    for frame_time, coords in iter_xyz_frames(traj_path):
        if frame_time is None:
            raise ValueError(f"Trajectory frame in {traj_path} has no timestamp")
        while (
            target_index < len(target_times)
            and target_times[target_index] < frame_time - tolerance
        ):
            target_index += 1
        if target_index >= len(target_times):
            break
        if abs(target_times[target_index] - frame_time) > tolerance:
            continue
        hydrogen_coords = coords[hydrogen_ids - 1]
        for label, endpoint_id in endpoint_ids.items():
            displacements = _minimum_image(
                hydrogen_coords - coords[endpoint_id - 1], box
            )
            coordination[label][target_index] = rational_coordination(
                np.linalg.norm(displacements, axis=1),
                coordination_refdist,
                coordination_nexp,
                coordination_mexp,
            )
            raw_defect_id = defect_oxygen_ids[target_index]
            if not np.isfinite(raw_defect_id):
                continue
            result, _edges = shortest_hbond_path(
                coords=coords,
                nitrogen_id=endpoint_id,
                defect_oxygen_id=int(raw_defect_id),
                solvent_oxygen_ids=solvent_oxygen_ids,
                hydrogen_ids=hydrogen_ids,
                covalent_cutoff=covalent_cutoff,
                hydrogen_acceptor_cutoff=hydrogen_acceptor_cutoff,
                angle_cutoff=angle_cutoff,
                max_bridging_waters=max_bridging_waters,
                box=box,
            )
            wires[label][target_index] = (
                float(result.bridging_water_count) if result.connected else -1.0
            )
        target_index += 1
    return wires, coordination


def calculate_histidine_charges(
    mulliken_path: Path,
    endpoint_ids: dict[str, int],
    solute_atoms: int,
    tmax: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return Mulliken charge series for all histidine N and O atoms."""
    from plot_coord_prod_grid import parse_mulliken_limited

    requested_ids = sorted(endpoint_ids.values())
    times, charges, target_ids, _elements = parse_mulliken_limited(
        mulliken_path,
        solute_atoms,
        tmax,
        extra_ids=requested_ids,
    )
    index_by_id = {atom_id: index for index, atom_id in enumerate(target_ids)}
    return times, {
        label: charges[index_by_id[atom_id]]
        for label, atom_id in endpoint_ids.items()
    }


def save_histidine_dihedral_data(
    path: Path, times: np.ndarray, series: dict[str, np.ndarray]
) -> None:
    """Save the aligned angular data rendered in summary column four."""
    labels = tuple(HIST_DIHEDRAL_ATOM_NAMES)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack([times, *[series[label] for label in labels]]),
        delimiter=",",
        header=",".join(["time_ps", *[f"{label}_deg" for label in labels]]),
        comments="",
    )


def save_histidine_competitor_data(
    path: Path,
    times: np.ndarray,
    defect_oxygen_ids: np.ndarray,
    wires: dict[str, np.ndarray],
    coordination: dict[str, np.ndarray],
) -> None:
    """Save aligned N/O wire and all-H coordination observables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["time_ps", "defect_oxygen_id"]
    fields += [
        f"{label}_defect_hbond_bridging_waters"
        for label in HIST_COMPETITOR_LABELS
    ]
    fields += [f"{label}_all_H_coordination" for label in HIST_COMPETITOR_LABELS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, time_ps in enumerate(times):
            row: dict[str, str | int] = {"time_ps": f"{time_ps:.10g}"}
            row["defect_oxygen_id"] = (
                ""
                if not np.isfinite(defect_oxygen_ids[index])
                else int(defect_oxygen_ids[index])
            )
            for label in HIST_COMPETITOR_LABELS:
                wire = wires[label][index]
                coord = coordination[label][index]
                row[f"{label}_defect_hbond_bridging_waters"] = (
                    "" if not np.isfinite(wire) else f"{wire:.10g}"
                )
                row[f"{label}_all_H_coordination"] = (
                    "" if not np.isfinite(coord) else f"{coord:.10g}"
                )
            writer.writerow(row)


def save_histidine_charge_data(
    path: Path, times: np.ndarray, charges: dict[str, np.ndarray]
) -> None:
    """Save aligned Mulliken charge series for all histidine N/O atoms."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack(
            [times, *[charges[label] for label in HIST_COMPETITOR_LABELS]]
        ),
        delimiter=",",
        header=",".join(
            ["time_ps", *[f"{label}_charge" for label in HIST_COMPETITOR_LABELS]]
        ),
        comments="",
    )
