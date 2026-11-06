#!/usr/bin/env python3
"""Append trajectory-derived DFTB torsions to an umbrella report CSV.

This utility is intended for manually appended umbrella windows whose stage
snapshots differ from the original ``us-prod-report`` configuration.  It uses
the same DFTB-XYZ conversion, imaging, cpptraj torsion calculation, time
alignment, and branch selection as the package production reporter.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

from qmmd.dihedral import convert_dftb_xyz_to_cpptraj
from qmmd.us_equil_report import EquilSample, write_samples_csv
from qmmd.us_prod_report import (
    read_dftb_trajectory_times,
    render_prod_cpptraj_input,
)
from qmmd.us_pull_report import (
    amber_residue_for_atoms,
    angle_near_target,
    read_cpptraj_dihedrals,
    run_cpptraj,
)


def parse_window(value: str) -> tuple[int, float]:
    """Parse ``INDEX:CENTER_DEG`` from the command line."""
    try:
        index_text, center_text = value.split(":", maxsplit=1)
        index, center = int(index_text), float(center_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "window must have the form INDEX:CENTER_DEG"
        ) from exc
    if index < 0:
        raise argparse.ArgumentTypeError("window index must be nonnegative")
    return index, center


def read_existing(path: Path) -> list[EquilSample]:
    if not path.is_file():
        return []
    samples: list[EquilSample] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "window_index", "target_deg", "time_ps", "step",
            "dihedral_raw_deg", "dihedral_branch_deg",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing torsion columns in {path}")
        for row in reader:
            samples.append(
                EquilSample(
                    window_index=int(row["window_index"]),
                    target_deg=float(row["target_deg"]),
                    time_ps=float(row["time_ps"]),
                    step=int(row["step"]),
                    dihedral_raw_deg=float(row["dihedral_raw_deg"]),
                    dihedral_branch_deg=float(row["dihedral_branch_deg"]),
                )
            )
    return samples


def collect_window(
    root: Path,
    stage_name: str,
    topology: Path,
    atoms: tuple[int, int, int, int],
    module: str,
    index: int,
    center: float,
) -> list[EquilSample]:
    stage = root / f"window-{index:03d}" / stage_name
    trajectory = stage / "traject"
    dftb_input = stage / "dftb.inp"
    if not trajectory.is_file() or not dftb_input.is_file():
        raise RuntimeError(f"Missing trajectory/input in {stage}")
    records = read_dftb_trajectory_times(trajectory)
    residue, _ = amber_residue_for_atoms(topology, atoms)
    data_path = stage / "dihedral.dat"
    input_path = stage / "dihedral.cpptraj.in"
    log_path = stage / "dihedral.cpptraj.log"
    with tempfile.TemporaryDirectory(prefix=f"qmmd-us-stage-{index:03d}-") as tmp:
        converted = Path(tmp) / "traject.cpptraj.xyz"
        convert_dftb_xyz_to_cpptraj(trajectory, converted, dftb_input)
        input_path.write_text(
            render_prod_cpptraj_input(topology, converted, atoms, data_path, residue)
        )
        run_cpptraj(input_path, log_path, module)
    angles = read_cpptraj_dihedrals(data_path)
    if len(angles) != len(records):
        raise ValueError(
            f"cpptraj wrote {len(angles)} angles for {len(records)} frames in {stage}"
        )
    start = records[0][1]
    return [
        EquilSample(
            window_index=index,
            target_deg=center,
            time_ps=time_ps - start,
            step=step,
            dihedral_raw_deg=angle,
            dihedral_branch_deg=angle_near_target(angle, center),
        )
        for (step, time_ps), angle in zip(records, angles)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--atoms", type=int, nargs=4, required=True)
    parser.add_argument("--window", action="append", type=parse_window, required=True)
    parser.add_argument("--base-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amber-module", default="amber/26")
    args = parser.parse_args()

    output = args.output.resolve()
    samples = read_existing(args.base_csv.resolve()) if args.base_csv else []
    replaced = {index for index, _ in args.window}
    samples = [sample for sample in samples if sample.window_index not in replaced]
    atoms = tuple(args.atoms)
    for index, center in args.window:
        samples.extend(
            collect_window(
                args.root.resolve(), args.stage, args.topology.resolve(), atoms,
                args.amber_module, index, center,
            )
        )
    samples.sort(key=lambda sample: (sample.window_index, sample.time_ps))
    write_samples_csv(output, samples)
    print(f"OK: wrote {len(samples)} aligned trajectory-derived samples to {output}")


if __name__ == "__main__":
    main()
