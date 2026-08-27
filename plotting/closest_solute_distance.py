#!/usr/bin/env python3
"""Append frame-aligned closest-solute-heavy-atom defect distances to a CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from oxygen_wire import (
    closest_solute_heavy_atom,
    iter_xyz_frames,
    read_box_lengths_from_dftb_inp,
    read_xyz_symbols,
)


def augment_closest_distances(
    input_csv: Path,
    output_csv: Path,
    trajectory: Path,
    solute_atoms: int,
    dftb_input: Path,
) -> None:
    """Write minimum-image closest-heavy-atom distances at matched timestamps."""
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {input_csv}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    required = {"time_ps", "defect_oxygen_id"}
    if not rows or not required.issubset(fieldnames):
        raise ValueError(f"{input_csv} must contain {sorted(required)}")

    symbols = read_xyz_symbols(trajectory)
    heavy_ids = [
        atom_id
        for atom_id, symbol in enumerate(symbols[:solute_atoms], start=1)
        if symbol.upper() != "H"
    ]
    box = read_box_lengths_from_dftb_inp(dftb_input)
    target_times = np.asarray([float(row["time_ps"]) for row in rows])
    maximum_time = float(np.nanmax(target_times))
    frames: list[tuple[float, np.ndarray]] = []
    for time_ps, coordinates in iter_xyz_frames(trajectory):
        if time_ps is None:
            raise ValueError(f"Trajectory timestamps are required in {trajectory}")
        frames.append((time_ps, coordinates))
        if time_ps > maximum_time:
            break
    frame_times = np.asarray([time for time, _ in frames])
    positive_steps = np.diff(target_times)
    positive_steps = positive_steps[positive_steps > 0.0]
    tolerance = 0.51 * float(np.median(positive_steps)) if positive_steps.size else 1e-6

    extra_fields = [
        "closest_solute_heavy_atom_id",
        "closest_solute_heavy_element",
        "closest_solute_heavy_distance_A",
    ]
    fieldnames.extend(field for field in extra_fields if field not in fieldnames)
    for row, target_time in zip(rows, target_times):
        defect_text = row.get("defect_oxygen_id", "")
        try:
            defect_id = int(float(defect_text))
        except (TypeError, ValueError, OverflowError):
            defect_id = -1
        insertion = int(np.searchsorted(frame_times, target_time))
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(frames)
        ]
        frame_index = min(candidates, key=lambda i: abs(frame_times[i] - target_time))
        if defect_id < 1 or abs(frame_times[frame_index] - target_time) > tolerance:
            atom_id, distance = -1, float("nan")
        else:
            atom_id, distance = closest_solute_heavy_atom(
                frames[frame_index][1], defect_id, heavy_ids, box
            )
        row["closest_solute_heavy_atom_id"] = "" if atom_id < 1 else str(atom_id)
        row["closest_solute_heavy_element"] = (
            "" if atom_id < 1 else symbols[atom_id - 1]
        )
        row["closest_solute_heavy_distance_A"] = (
            "" if not math.isfinite(distance) else f"{distance:.8f}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--traj", required=True, type=Path)
    parser.add_argument("--solute-atoms", required=True, type=int)
    parser.add_argument("--dftb-inp", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    augment_closest_distances(
        args.input_csv, args.out, args.traj, args.solute_atoms, args.dftb_inp
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
