#!/usr/bin/env python3
"""Frame-aligned BV competitor observables for acid--base summaries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from bv_ring_defect_distances import (
    identify_bv_rings,
    identify_terminal_ring_oxygens,
)
from acid_base_BV import (
    _minimum_image,
    iter_xyz_frames,
    read_xyz_symbols,
    shortest_hbond_path,
)


COMPETITOR_LABELS = (
    "NA", "NB", "NC", "ND", "OA", "OD", "T1-O1", "T1-O2", "T2-O1", "T2-O2"
)
OXYGEN_LABELS = COMPETITOR_LABELS[4:]
CHARGE_LABELS = (
    "ring A",
    "ring B",
    "ring C",
    "ring D",
    *OXYGEN_LABELS,
)
RING_LABELS = ("A", "B", "C", "D")
RING_NITROGEN_IDS = (1, 19, 31, 43)


def competitor_definitions(
    parm7: Path, solute_atoms: int
) -> tuple[dict[str, tuple[int, ...]], dict[str, int]]:
    """Return ring atom groups and chemically meaningful endpoint atom IDs."""
    rings = identify_bv_rings(parm7, solute_atoms, RING_NITROGEN_IDS)
    terminal_oxygens = identify_terminal_ring_oxygens(parm7, solute_atoms, rings)
    from bv_ring_defect_distances import read_solute_bond_graph

    _adjacency, masses = read_solute_bond_graph(parm7, solute_atoms)
    terminal_ids = set(terminal_oxygens.values())
    tail_oxygen_ids = [
        atom_id
        for atom_id, mass in enumerate(masses, start=1)
        if 15.5 <= mass <= 16.5 and atom_id not in terminal_ids
    ]
    if len(tail_oxygen_ids) != 4:
        raise ValueError(
            "Expected four BV tail oxygens in addition to OA and OD; "
            f"found {tail_oxygen_ids}"
        )
    endpoints = {
        **dict(zip(COMPETITOR_LABELS[:4], RING_NITROGEN_IDS)),
        "OA": terminal_oxygens["A"],
        "OD": terminal_oxygens["D"],
        **dict(zip(OXYGEN_LABELS[2:], tail_oxygen_ids)),
    }
    return rings, endpoints


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


def calculate_competitor_geometry(
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
    """Calculate defect-wire lengths and endpoint--all-H coordination."""
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
        for label in COMPETITOR_LABELS
    }
    coordination = {
        label: np.full(target_times.shape, np.nan, dtype=float)
        for label in COMPETITOR_LABELS
    }
    finite_times = target_times[np.isfinite(target_times)]
    positive_steps = np.diff(finite_times)
    positive_steps = positive_steps[positive_steps > 0.0]
    tolerance = 0.51 * float(np.median(positive_steps)) if positive_steps.size else 1e-6
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


def calculate_competitor_charges(
    mulliken_path: Path,
    rings: dict[str, tuple[int, ...]],
    endpoint_ids: dict[str, int],
    solute_atoms: int,
    tmax: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return ring-mean and terminal-oxygen Mulliken charge time series."""
    from plot_coord_prod_grid import parse_mulliken_limited

    requested_ids = sorted(
        {atom_id for atom_ids in rings.values() for atom_id in atom_ids}
        | set(endpoint_ids.values())
    )
    times, charges, target_ids, _elements = parse_mulliken_limited(
        mulliken_path,
        solute_atoms,
        tmax,
        extra_ids=requested_ids,
    )
    index_by_id = {atom_id: index for index, atom_id in enumerate(target_ids)}
    series: dict[str, np.ndarray] = {}
    for label, ring_label in zip(CHARGE_LABELS[:4], RING_LABELS):
        indices = [index_by_id[atom_id] for atom_id in rings[ring_label]]
        series[label] = np.nanmean(charges[indices], axis=0)
    for label in OXYGEN_LABELS:
        series[label] = charges[index_by_id[endpoint_ids[label]]]
    return times, series


def save_competitor_data(
    path: Path,
    times: np.ndarray,
    defect_oxygen_ids: np.ndarray,
    wires: dict[str, np.ndarray],
    coordination: dict[str, np.ndarray],
) -> None:
    """Save trajectory-aligned competitor wire and coordination observables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["time_ps", "defect_oxygen_id"]
    fields += [f"{label}_defect_hbond_bridging_waters" for label in COMPETITOR_LABELS]
    fields += [f"{label}_all_H_coordination" for label in COMPETITOR_LABELS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, time_ps in enumerate(times):
            row: dict[str, str | float | int] = {"time_ps": f"{time_ps:.10g}"}
            row["defect_oxygen_id"] = (
                "" if not np.isfinite(defect_oxygen_ids[index])
                else int(defect_oxygen_ids[index])
            )
            for label in COMPETITOR_LABELS:
                wire = wires[label][index]
                coord = coordination[label][index]
                row[f"{label}_defect_hbond_bridging_waters"] = (
                    "" if not np.isfinite(wire) else f"{wire:.10g}"
                )
                row[f"{label}_all_H_coordination"] = (
                    "" if not np.isfinite(coord) else f"{coord:.10g}"
                )
            writer.writerow(row)


def save_competitor_charge_data(
    path: Path, times: np.ndarray, charges: dict[str, np.ndarray]
) -> None:
    """Save Mulliken-time-aligned competitor charge observables."""
    data = np.column_stack([times, *[charges[label] for label in CHARGE_LABELS]])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=",".join(["time_ps", *[f"{label}_charge" for label in CHARGE_LABELS]]),
        comments="",
    )
