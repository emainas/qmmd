#!/usr/bin/env python3
"""Generate one timestamp-matched BV acid-base summary at a supplied grid tdiff."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from acid_base_BV import (
    _minimum_image,
    iter_xyz_frames,
    read_box_lengths_from_dftb_inp,
    read_xyz_symbols,
)
from plot_cv_grid import infer_offset_ps
from prn_acid_base_analysis import aligned_fes, bias_samples, mulliken_frames


ROOT = Path(__file__).resolve().parents[1]


def snapshot(path: Path) -> dict[str, object]:
    """Return immutable-source metadata for provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--tdiff", type=float, required=True, help="Grid-clock tdiff (ps)")
    parser.add_argument(
        "--runs-path", type=Path,
        default=ROOT / "systems/BV/solv_5.5/dftb/N1T64C1",
    )
    parser.add_argument("--cv-dir", default="meta-hib")
    parser.add_argument("--bench-tag", default="N1T64C1")
    parser.add_argument("--nitrogen-id", type=int, default=19)
    parser.add_argument("--hydrogen-id", type=int, default=20)
    parser.add_argument("--solute-atoms", type=int, default=78)
    parser.add_argument("--window-ps", type=float, default=1.75)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--charge-min", type=float, default=-0.625)
    parser.add_argument("--charge-max", type=float, default=-0.525)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if (
        not np.isfinite(args.tdiff)
        or args.window_ps <= 0.0
        or args.samples < 2
        or args.stride < 1
        or args.charge_min > args.charge_max
    ):
        parser.error("Invalid tdiff, window, sample, stride, or charge-window setting")

    source = (args.runs_path / f"run-{args.run}" / args.cv_dir).resolve()
    required = ("traject", "mulliken", "biaspot", "fes.dat", "dftb.inp")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        parser.error(f"Missing source files in {source}: {', '.join(missing)}")
    output = args.out or (
        ROOT / f"reports/BV/solv_5.5/{args.cv_dir}/acid_base"
        / args.bench_tag / f"run-{args.run}"
    )
    if output.exists():
        parser.error(f"Output already exists; refusing to overwrite: {output}")

    symbols = read_xyz_symbols(source / "traject")
    if len(symbols) <= args.solute_atoms:
        raise ValueError(f"Trajectory has no solvent atoms: {source / 'traject'}")
    if symbols[args.nitrogen_id - 1].upper() != "N":
        raise ValueError(f"Atom {args.nitrogen_id} is not nitrogen")
    if symbols[args.hydrogen_id - 1].upper() != "H":
        raise ValueError(f"Atom {args.hydrogen_id} is not hydrogen")

    bias_times, coordination, _ = bias_samples(source / "biaspot")
    offset = infer_offset_ps(source.parent, source.name, float(bias_times[0]))
    raw_tdiff = args.tdiff - offset
    raw_stop = raw_tdiff + args.window_ps
    fes_times, _ = aligned_fes(source / "fes.dat")
    if raw_tdiff < bias_times[0] or raw_stop > bias_times[-1] + 1.0e-8:
        raise ValueError("Requested analysis window is outside bias-potential coverage")
    if raw_stop > fes_times[-1] + 1.0e-8:
        raise ValueError("Requested analysis window is outside FES coverage")

    mulliken_times, charges = mulliken_frames(source / "mulliken", symbols)
    if np.any(np.diff(mulliken_times) <= 0.0) or np.any(np.diff(bias_times) <= 0.0):
        raise ValueError("Nonmonotonic timestamps require explicit restart stitching")
    mulliken_lookup = {
        round(float(time), 8): index for index, time in enumerate(mulliken_times)
    }
    solvent_oxygens = np.asarray(
        [
            index
            for index, symbol in enumerate(symbols)
            if index >= args.solute_atoms and symbol.upper() == "O"
        ],
        dtype=int,
    )
    if not solvent_oxygens.size:
        raise ValueError("No solvent oxygens found after the solute atoms")
    box = read_box_lengths_from_dftb_inp(source / "dftb.inp")
    rows: list[list[float]] = []
    for frame, (time_ps, coordinates) in enumerate(iter_xyz_frames(source / "traject")):
        if time_ps is None:
            raise ValueError("Trajectory frame lacks a timestamp")
        if time_ps > raw_stop + 1.0e-8:
            break
        if frame % args.stride:
            continue
        mulliken_index = mulliken_lookup.get(round(time_ps, 8))
        defect_id = float("nan")
        distance = float("nan")
        if mulliken_index is not None:
            oxygen_charges = charges[mulliken_index, solvent_oxygens]
            valid = (oxygen_charges >= args.charge_min) & (
                oxygen_charges <= args.charge_max
            )
            if valid.any():
                defect_index = int(
                    solvent_oxygens[
                        np.argmax(np.where(valid, oxygen_charges, -np.inf))
                    ]
                )
                defect_id = float(defect_index + 1)
                displacement = _minimum_image(
                    coordinates[defect_index]
                    - coordinates[args.nitrogen_id - 1],
                    box,
                )
                distance = float(np.linalg.norm(displacement))
        rows.append(
            [
                time_ps,
                float(
                    np.interp(
                        time_ps, bias_times, coordination,
                        left=np.nan, right=np.nan,
                    )
                ),
                defect_id,
                distance,
            ]
        )
    values = np.asarray(rows, dtype=float)
    if not len(values) or raw_stop - values[-1, 0] > 0.021:
        raise ValueError("Trajectory sampling does not cover the requested endpoint")
    if np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError("Sampled trajectory timestamps are not strictly increasing")

    output.mkdir(parents=True, exist_ok=False)
    wire_input = output / "wire_input.csv"
    np.savetxt(
        wire_input,
        values,
        delimiter=",",
        comments="",
        header=(
            "time_ps,coordination_s,defect_oxygen_id,"
            f"N{args.nitrogen_id}_Odefect_distance_A"
        ),
    )
    topology = ROOT / "systems/BV/solv_5.5/salt/ready.parm7"
    command = [
        sys.executable,
        str(ROOT / "plotting/acid_base_BV.py"),
        "--input-csv", str(wire_input),
        "--traj", str(source / "traject"),
        "--nitrogen-id", str(args.nitrogen_id),
        "--solute-atoms", str(args.solute_atoms),
        "--dftb-inp", str(source / "dftb.inp"),
        "--bv-parm", str(topology),
        "--mulliken", str(source / "mulliken"),
        "--fes", str(source / "fes.dat"),
        "--diffusive-start", str(raw_tdiff),
        "--probability-tmax", str(raw_stop),
        "--probability-window-ps", str(args.window_ps),
        "--pka-window-samples", str(args.samples),
        "--temp", str(args.temperature),
        "--out", str(output / "summary.csv"),
        "--plot-out", str(output / "summary.png"),
    ]
    with (output / "analysis.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    provenance = {
        "system": "BV",
        "run": args.run,
        "bench_tag": args.bench_tag,
        "cv_dir": args.cv_dir,
        "source": str(source),
        "grid_tdiff_ps": args.tdiff,
        "raw_tdiff_ps": raw_tdiff,
        "clock_offset_ps": offset,
        "analysis_end_raw_ps": raw_stop,
        "analysis_end_grid_ps": args.tdiff + args.window_ps,
        "temperature_K": args.temperature,
        "nitrogen_id": args.nitrogen_id,
        "hydrogen_id": args.hydrogen_id,
        "solute_atoms": args.solute_atoms,
        "charge_window_e": [args.charge_min, args.charge_max],
        "geometry_stride": args.stride,
        "snapshots": {name: snapshot(source / name) for name in required},
        "command": command,
        "note": (
            "User-selected manual diffusion marker on the synchronized grid clock; "
            "not independent evidence of bulk diffusion. Missing defect assignments "
            "remain missing."
        ),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        f"# BV solv 5.5 {args.cv_dir} run {args.run}\n\n"
        f"User-selected synchronized-clock tdiff: {args.tdiff:g} ps. Raw DFTB "
        f"tdiff: {raw_tdiff:g} ps; raw analysis endpoint: {raw_stop:g} ps. "
        f"NB atom ID: {args.nitrogen_id}; labeled H atom ID: {args.hydrogen_id}; "
        f"solute atoms: {args.solute_atoms}. Geometry uses every {args.stride} "
        "trajectory frame, timestamp-matched Mulliken assignments, interpolated "
        "bias coordination, and periodic minimum-image distances. Hydrogen-bond "
        "criteria retain the acid_base_BV defaults. Raw simulation files were "
        "read only.\n",
        encoding="utf-8",
    )
    print(output / "summary.png")


if __name__ == "__main__":
    main()
