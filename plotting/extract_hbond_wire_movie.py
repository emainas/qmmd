#!/usr/bin/env python3
"""Extract H-bond-connected frames and generate a dynamic VMD wire movie.

The input CSV must be an augmented output from ``oxygen_wire.py``.  Frame
selection and path membership are taken directly from that CSV, while XYZ
frames are matched by timestamp.  Atom IDs remain one-based in all outputs;
the generated VMD script performs the conversion to zero-based indices.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from oxygen_wire import (
    iter_xyz_frames,
    read_box_lengths_from_dftb_inp,
    read_xyz_symbols,
)


@dataclass(frozen=True)
class SelectedFrame:
    analyzed_time_ps: float
    defect_oxygen_id: int
    bridging_water_count: int
    path_atom_ids: tuple[int, ...]
    edge_atom_ids: tuple[tuple[int, int, int], ...]


def _parse_ids(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(";") if item)


def _parse_edges(value: str) -> tuple[tuple[int, int, int], ...]:
    edges: list[tuple[int, int, int]] = []
    for item in value.split(";"):
        if not item:
            continue
        donor, hydrogen, acceptor = (int(atom_id) for atom_id in item.split("-"))
        edges.append((donor, hydrogen, acceptor))
    return tuple(edges)


def read_connected_frames(
    path: Path,
    nitrogen_id: int,
    time_min: float | None,
    time_max: float | None,
    stride: int,
    open_interval: bool = False,
) -> list[SelectedFrame]:
    """Read every ``stride``-th H-bond-connected analyzed frame."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "time_ps",
        "defect_oxygen_id",
        "hbond_wire_connected",
        "hbond_bridging_water_count",
        "hbond_oxygen_path_atom_ids",
        "hbond_edge_atom_ids_D-H-A",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")

    connected: list[SelectedFrame] = []
    for row in rows:
        if row["hbond_wire_connected"] != "1":
            continue
        time_ps = float(row["time_ps"])
        if time_min is not None:
            if open_interval and time_ps <= time_min:
                continue
            if not open_interval and time_ps < time_min:
                continue
        if time_max is not None:
            if open_interval and time_ps >= time_max:
                continue
            if not open_interval and time_ps > time_max:
                continue
        defect_value = float(row["defect_oxygen_id"])
        if not np.isfinite(defect_value):
            continue
        oxygen_path = _parse_ids(row["hbond_oxygen_path_atom_ids"])
        edges = _parse_edges(row["hbond_edge_atom_ids_D-H-A"])
        if not oxygen_path or not edges:
            continue
        connected.append(
            SelectedFrame(
                analyzed_time_ps=time_ps,
                defect_oxygen_id=int(defect_value),
                bridging_water_count=int(row["hbond_bridging_water_count"]),
                path_atom_ids=(nitrogen_id, *oxygen_path),
                edge_atom_ids=edges,
            )
        )
    return connected[::stride]


def match_xyz_frames(
    traj_path: Path,
    selected: list[SelectedFrame],
    tolerance_ps: float | None,
) -> Iterator[tuple[SelectedFrame, int, float | None, np.ndarray]]:
    """Yield selected frames aligned to the closest trajectory timestamp."""
    if not selected:
        return
    targets = np.asarray([item.analyzed_time_ps for item in selected], dtype=float)
    if tolerance_ps is None:
        tolerance_ps = 1.0e-4

    target_index = 0
    previous: tuple[int, float | None, np.ndarray] | None = None
    for trajectory_index, (time_ps, coords) in enumerate(iter_xyz_frames(traj_path)):
        if time_ps is None:
            raise ValueError(
                "Trajectory comments have no DCDFTBMD timestamps; timestamp "
                "alignment is required for this extraction"
            )
        current = (trajectory_index, time_ps, coords)
        while target_index < len(selected) and targets[target_index] <= time_ps:
            candidates = [current]
            if previous is not None:
                candidates.append(previous)
            best = min(candidates, key=lambda item: abs(float(item[1]) - targets[target_index]))
            error = abs(float(best[1]) - targets[target_index])
            if error > tolerance_ps:
                raise ValueError(
                    f"No trajectory frame within {tolerance_ps:g} ps of analyzed "
                    f"time {targets[target_index]:g} ps"
                )
            yield selected[target_index], best[0], best[1], best[2]
            target_index += 1
        previous = current
        if target_index == len(selected):
            return
    if target_index < len(selected):
        raise ValueError(
            f"Trajectory ended before analyzed time {targets[target_index]:g} ps"
        )


def write_outputs(
    out_prefix: Path,
    symbols: list[str],
    matched: Iterator[tuple[SelectedFrame, int, float | None, np.ndarray]],
    solute_atoms: int,
    box: np.ndarray | None,
) -> tuple[Path, Path, Path]:
    """Write extracted XYZ, aligned metadata CSV, and dynamic VMD script."""
    xyz_path = out_prefix.parent / f"{out_prefix.name}.xyz"
    csv_path = out_prefix.with_name(f"{out_prefix.name}_frames.csv")
    vmd_path = out_prefix.parent / f"{out_prefix.name}.vmd"
    xyz_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "movie_frame_index",
        "original_trajectory_frame_index",
        "trajectory_time_ps",
        "analyzed_time_ps",
        "defect_oxygen_id",
        "hbond_bridging_water_count",
        "hbond_path_atom_ids",
        "hbond_edge_atom_ids_D-H-A",
    ]
    records: list[dict[str, str]] = []
    with xyz_path.open("w", encoding="utf-8") as xyz_handle:
        for movie_index, (item, original_index, trajectory_time, coords) in enumerate(matched):
            if len(coords) != len(symbols):
                raise ValueError("Atom count changes between XYZ frames")
            path_text = ";".join(map(str, item.path_atom_ids))
            edge_text = ";".join("-".join(map(str, edge)) for edge in item.edge_atom_ids)
            xyz_handle.write(f"{len(symbols)}\n")
            xyz_handle.write(
                f"movie_frame={movie_index} original_frame={original_index} "
                f"time_ps={float(trajectory_time):.8f} path={path_text}\n"
            )
            for symbol, xyz in zip(symbols, coords):
                xyz_handle.write(
                    f"{symbol:<2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n"
                )
            records.append(
                {
                    "movie_frame_index": str(movie_index),
                    "original_trajectory_frame_index": str(original_index),
                    "trajectory_time_ps": f"{float(trajectory_time):.8f}",
                    "analyzed_time_ps": f"{item.analyzed_time_ps:.8f}",
                    "defect_oxygen_id": str(item.defect_oxygen_id),
                    "hbond_bridging_water_count": str(item.bridging_water_count),
                    "hbond_path_atom_ids": path_text,
                    "hbond_edge_atom_ids_D-H-A": edge_text,
                }
            )
    if not records:
        raise ValueError("No connected frames were extracted")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    write_vmd_script(vmd_path, xyz_path.name, csv_path.name, solute_atoms, box)
    return xyz_path, csv_path, vmd_path


def write_vmd_script(
    path: Path,
    xyz_name: str,
    csv_name: str,
    solute_atoms: int,
    box: np.ndarray | None,
) -> None:
    """Write a VMD script that redraws the per-frame H-bond wire."""
    template = r'''# Dynamic hydrogen-bond wire movie generated by extract_hbond_wire_movie.py
set script_dir [file dirname [file normalize [info script]]]
set trajectory [file join $script_dir "__XYZ__"]
set metadata [file join $script_dir "__CSV__"]
set wire_box {__BOX__}

if {![file exists $trajectory] || ![file exists $metadata]} {
    puts stderr "ERROR: movie XYZ or frame metadata is missing beside this script."
    quit
}

set wire_rows {}
set handle [open $metadata r]
gets $handle header
while {[gets $handle line] >= 0} {
    if {[string trim $line] eq ""} { continue }
    lappend wire_rows [split $line ","]
}
close $handle

mol new $trajectory type xyz waitfor all
set wire_molid [molinfo top]
color Display Background white
display projection Orthographic
axes location Off

mol delrep 0 $wire_molid
mol representation DynamicBonds 1.6 0.10 12.0
mol color Name
mol selection all
mol material Transparent
mol addrep $wire_molid

mol representation Licorice 0.20 16.0 16.0
mol color Name
mol selection "index 0 to __SOLUTE_LAST__"
mol material Opaque
mol addrep $wire_molid

proc wire_nearest_image {raw reference box} {
    set adjusted {}
    foreach value $raw origin $reference length $box {
        if {$length > 0.0} {
            lappend adjusted [expr {$value - $length * round(($value - $origin) / $length)}]
        } else {
            lappend adjusted $value
        }
    }
    return $adjusted
}

proc wire_draw_frame {name element op} {
    global wire_molid wire_rows wire_box
    set frame $::vmd_frame($wire_molid)
    if {$frame < 0 || $frame >= [llength $wire_rows]} { return }
    graphics $wire_molid delete all
    set row [lindex $wire_rows $frame]
    set time_ps [lindex $row 2]
    set defect_id [lindex $row 4]
    set bridges [lindex $row 5]
    set path_ids [split [lindex $row 6] ";"]
    set edge_text [lindex $row 7]

    # Reconstruct the ordered path in a single periodic image, anchored at N.
    # The trajectory itself is unchanged; only these VMD graphics are unwrapped.
    array unset wire_xyz
    set previous_xyz {}
    foreach atom_id $path_ids {
        set atom_index [expr {$atom_id - 1}]
        set selection [atomselect $wire_molid "index $atom_index" frame $frame]
        set raw_xyz [lindex [$selection get {x y z}] 0]
        $selection delete
        if {[llength $previous_xyz] == 0} {
            set adjusted_xyz $raw_xyz
        } else {
            set adjusted_xyz [wire_nearest_image $raw_xyz $previous_xyz $wire_box]
        }
        set wire_xyz($atom_index) $adjusted_xyz
        set previous_xyz $adjusted_xyz
        graphics $wire_molid color red
        graphics $wire_molid sphere $adjusted_xyz radius 0.42 resolution 18
    }

    foreach edge [split $edge_text ";"] {
        lassign [split $edge "-"] donor_id hydrogen_id acceptor_id
        set donor_index [expr {$donor_id - 1}]
        set hydrogen_index [expr {$hydrogen_id - 1}]
        set acceptor_index [expr {$acceptor_id - 1}]
        set hydrogen_sel [atomselect $wire_molid "index $hydrogen_index" frame $frame]
        set raw_hydrogen_xyz [lindex [$hydrogen_sel get {x y z}] 0]
        set donor_xyz $wire_xyz($donor_index)
        set acceptor_xyz $wire_xyz($acceptor_index)
        set hydrogen_xyz [wire_nearest_image $raw_hydrogen_xyz $donor_xyz $wire_box]
        graphics $wire_molid color yellow
        graphics $wire_molid sphere $hydrogen_xyz radius 0.26 resolution 18
        graphics $wire_molid color orange
        graphics $wire_molid cylinder $donor_xyz $hydrogen_xyz radius 0.09 resolution 16
        graphics $wire_molid color cyan
        graphics $wire_molid cylinder $hydrogen_xyz $acceptor_xyz radius 0.12 resolution 16
        $hydrogen_sel delete
    }
    graphics $wire_molid color black
    graphics $wire_molid text {-5 -5 0} "t = $time_ps ps   defect O$defect_id   bridges = $bridges" size 1.3 thickness 2
}

trace add variable ::vmd_frame($wire_molid) write wire_draw_frame
animate goto 0
wire_draw_frame {} {} {}
display resetview
puts "Loaded [llength $wire_rows] H-bond-connected frames from $trajectory"
puts "Orange: covalent donor-H segment; cyan: H...acceptor segment."
puts "Use VMD animation controls to play the extracted connected-frame movie."
'''
    text = template.replace("__XYZ__", xyz_name)
    text = text.replace("__CSV__", csv_name)
    text = text.replace("__SOLUTE_LAST__", str(solute_atoms - 1))
    box_text = "0.0 0.0 0.0" if box is None else " ".join(f"{x:.10g}" for x in box)
    text = text.replace("__BOX__", box_text)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract connected water-wire frames and generate a VMD movie."
    )
    parser.add_argument("--wire-csv", required=True, type=Path)
    parser.add_argument("--traj", required=True, type=Path)
    parser.add_argument("--nitrogen-id", required=True, type=int)
    parser.add_argument("--solute-atoms", required=True, type=int)
    parser.add_argument("--out-prefix", required=True, type=Path)
    parser.add_argument(
        "--dftb-inp",
        type=Path,
        default=None,
        help="DFTB input containing TV box vectors for periodic wire unwrapping",
    )
    parser.add_argument("--time-min", type=float, default=None, help="Minimum time in ps")
    parser.add_argument("--time-max", type=float, default=None, help="Maximum time in ps")
    parser.add_argument(
        "--t-diffusion",
        type=float,
        default=None,
        help=(
            "Diffusion onset in ps; extracts the open interval "
            "t_diffusion < t < t_diffusion + t_stop"
        ),
    )
    parser.add_argument(
        "--t-stop",
        type=float,
        default=1.75,
        help="Duration after --t-diffusion in ps (default: 1.75)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth connected frame (default: 1)",
    )
    parser.add_argument(
        "--time-tolerance",
        type=float,
        default=None,
        help="Maximum CSV-to-XYZ timestamp difference in ps (default: 1e-4)",
    )
    args = parser.parse_args()
    if args.nitrogen_id < 1 or args.solute_atoms < 1:
        parser.error("--nitrogen-id and --solute-atoms must be positive one-based counts")
    if args.stride < 1:
        parser.error("--stride must be positive")
    if args.t_stop <= 0:
        parser.error("--t-stop must be positive")
    if args.t_diffusion is not None and (
        args.time_min is not None or args.time_max is not None
    ):
        parser.error("--t-diffusion cannot be combined with --time-min/--time-max")

    time_min = args.time_min
    time_max = args.time_max
    open_interval = False
    if args.t_diffusion is not None:
        time_min = args.t_diffusion
        time_max = args.t_diffusion + args.t_stop
        open_interval = True

    selected = read_connected_frames(
        args.wire_csv,
        args.nitrogen_id,
        time_min,
        time_max,
        args.stride,
        open_interval,
    )
    if not selected:
        parser.error("No H-bond-connected frames satisfy the requested selection")
    symbols = read_xyz_symbols(args.traj)
    box = (
        read_box_lengths_from_dftb_inp(args.dftb_inp)
        if args.dftb_inp is not None
        else None
    )
    outputs = write_outputs(
        args.out_prefix,
        symbols,
        match_xyz_frames(args.traj, selected, args.time_tolerance),
        args.solute_atoms,
        box,
    )
    print(f"Extracted {len(selected)} connected frames")
    for output in outputs:
        print(f"Wrote {output}")
    if args.t_diffusion is not None:
        print(
            f"Interval: {args.t_diffusion:.8f} < t < "
            f"{args.t_diffusion + args.t_stop:.8f} ps"
        )
    print(f"Open with: vmd -e {outputs[2]}")


if __name__ == "__main__":
    main()
