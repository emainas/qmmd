#!/usr/bin/env python3
"""Extract connected tautomer-wire frames and generate a VMD movie bundle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from extract_hbond_wire_movie import (
    SelectedFrame,
    _parse_edges,
    _parse_ids,
    match_xyz_frames,
    write_outputs,
)
from oxygen_wire import read_box_lengths_from_dftb_inp, read_xyz_symbols


def read_tautomer_frames(
    path: Path,
    donor_id: int,
    time_min: float,
    time_max: float,
    stride: int,
    acceptor_id: int | None = None,
    include_disconnected: bool = False,
) -> list[SelectedFrame]:
    """Read connected frames in the open post-diffusion interval."""
    selected: list[SelectedFrame] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            time_ps = float(row["time_ps"])
            if not time_min < time_ps < time_max:
                continue
            connected = row["wire_connected"] == "1"
            if not connected and not include_disconnected:
                continue
            path_ids = _parse_ids(row["path_atom_ids"])
            edges = _parse_edges(row["edge_atom_ids_D-H-A"])
            if not connected:
                if acceptor_id is None:
                    raise ValueError("acceptor_id is required for disconnected frames")
                path_ids = (donor_id, acceptor_id)
                edges = ()
            elif not path_ids or not edges:
                continue
            if path_ids[0] != donor_id:
                raise ValueError(
                    f"Wire path starts at atom {path_ids[0]}, expected donor {donor_id}"
                )
            selected.append(
                SelectedFrame(
                    analyzed_time_ps=time_ps,
                    defect_oxygen_id=path_ids[-1],
                    bridging_water_count=int(row["bridging_waters"]),
                    path_atom_ids=path_ids,
                    edge_atom_ids=edges,
                )
            )
    return selected[::stride]


def relabel_outputs(csv_path: Path, vmd_path: Path, transition: str) -> None:
    """Replace acid-base defect terminology with tautomer endpoint terminology."""
    csv_text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(
        csv_text.replace("defect_oxygen_id", "acceptor_atom_id", 1),
        encoding="utf-8",
    )
    vmd_text = vmd_path.read_text(encoding="utf-8")
    replacements = {
        "Dynamic hydrogen-bond wire movie": "Dynamic tautomer hydrogen-bond wire movie",
        'set defect_id [lindex $row 4]': 'set acceptor_id [lindex $row 4]',
        'defect O$defect_id   bridges = $bridges': (
            f"{transition}   acceptor atom $acceptor_id   bridges = $bridges"
        ),
        "H-bond-connected frames": "tautomer-wire-connected frames",
    }
    for old, new in replacements.items():
        vmd_text = vmd_text.replace(old, new)
    vmd_path.write_text(vmd_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-csv", required=True, type=Path)
    parser.add_argument("--traj", required=True, type=Path)
    parser.add_argument("--dftb-inp", required=True, type=Path)
    parser.add_argument("--donor-id", required=True, type=int)
    parser.add_argument("--solute-atoms", required=True, type=int)
    parser.add_argument("--t-diffusion", required=True, type=float)
    parser.add_argument("--t-stop", type=float, default=1.75)
    parser.add_argument("--transition", required=True)
    parser.add_argument("--out-prefix", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--time-tolerance", type=float, default=None)
    args = parser.parse_args()
    if args.donor_id < 1 or args.solute_atoms < 1 or args.stride < 1:
        parser.error("atom IDs/counts and stride must be positive")
    if args.t_stop <= 0.0:
        parser.error("--t-stop must be positive")

    selected = read_tautomer_frames(
        args.frame_csv,
        args.donor_id,
        args.t_diffusion,
        args.t_diffusion + args.t_stop,
        args.stride,
    )
    if not selected:
        parser.error("No connected tautomer-wire frames satisfy the requested interval")
    outputs = write_outputs(
        args.out_prefix,
        read_xyz_symbols(args.traj),
        match_xyz_frames(args.traj, selected, args.time_tolerance),
        args.solute_atoms,
        read_box_lengths_from_dftb_inp(args.dftb_inp),
    )
    relabel_outputs(outputs[1], outputs[2], args.transition)
    print(f"Extracted {len(selected)} connected frames")
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
