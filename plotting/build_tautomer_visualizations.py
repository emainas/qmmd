#!/usr/bin/env python3
"""Build VMD movie bundles for confirmed tautomer summary analyses."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from extract_hbond_wire_movie import match_xyz_frames, write_outputs
from extract_tautomer_wire_movie import read_tautomer_frames, relabel_outputs
from oxygen_wire import read_box_lengths_from_dftb_inp, read_xyz_symbols
from tautomers import SYSTEM_SPECS, detect_diffusive_regime


STEM_RE = re.compile(
    r"^(?P<system>[A-Z]+)_solv_4\.0_(?P<cv>meta-hi[abcd])_run-(?P<run>\d+)"
    r"_tautomer-summary$"
)
BENCH_TAGS = {"BV": "N1T48C1", "APP": "N1T64C1", "BPP": "N1T64C1",
              "CPP": "N1T64C1", "DPP": "N1T64C1"}
SITE_NAMES = {
    "N-A": "N_A", "N-B": "N_B", "N-C": "N_C", "N-D": "N_D",
    "lactam-A": "lactam_O_A", "lactam-D": "lactam_O_D",
}


def load_coordination(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return values[:, 0], values[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--systems-root", type=Path, default=Path("systems"))
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--t-stop", type=float, default=1.75)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    if args.t_stop <= 0.0 or args.stride < 1:
        parser.error("--t-stop and --stride must be positive")

    frame_csvs = sorted(args.analysis_root.glob("**/*_tautomer-summary.csv"))
    if not frame_csvs:
        parser.error(f"No tautomer frame CSVs found below {args.analysis_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped: list[str] = []
    for frame_csv in frame_csvs:
        match = STEM_RE.match(frame_csv.stem)
        if match is None:
            skipped.append(f"{frame_csv}: unrecognized filename")
            continue
        system = match.group("system")
        cv_dir = match.group("cv")
        run_id = int(match.group("run"))
        spec = SYSTEM_SPECS[system]
        donor_id = int(spec["donors"][cv_dir][0])
        category = frame_csv.parent.name
        acceptor_key = category.split("_to_", 1)[1]
        acceptor_id = next(
            atom_id
            for atom_id, label in spec["acceptors"].items()
            if label == acceptor_key
        )
        donor_site = cv_dir[-1].upper()
        transition = f"N_{donor_site} -> {SITE_NAMES.get(acceptor_key, acceptor_key)}"
        coordination_path = frame_csv.with_name(
            f"{frame_csv.stem}_coordination.csv"
        )
        times, coordination = load_coordination(coordination_path)
        diffusion = detect_diffusive_regime(times, coordination, 0.05, 0.20, 0.25, 0.05)
        if diffusion is None:
            skipped.append(f"{frame_csv}: diffusion onset unavailable")
            continue
        t_diffusion = diffusion[1]
        bench_tag = BENCH_TAGS[system]
        run_root = (
            args.systems_root / system / "solv_4.0" / "dftb" / bench_tag
            / f"run-{run_id}" / cv_dir
        )
        selected = read_tautomer_frames(
            frame_csv,
            donor_id,
            t_diffusion,
            t_diffusion + args.t_stop,
            args.stride,
        )
        if not selected:
            selected = read_tautomer_frames(
                frame_csv,
                donor_id,
                t_diffusion,
                t_diffusion + args.t_stop,
                args.stride,
                acceptor_id=acceptor_id,
                include_disconnected=True,
            )
            if not selected:
                skipped.append(f"{frame_csv}: no analyzed frames in diffusion window")
                continue
        name = (
            f"{system}_solv_4.0_{bench_tag}_{cv_dir}_run-{run_id}_"
            f"{acceptor_key}-wire-movie"
        )
        outputs = write_outputs(
            args.out_dir / name,
            read_xyz_symbols(run_root / "traject"),
            match_xyz_frames(run_root / "traject", selected, None),
            int(spec["solute_atoms"]),
            read_box_lengths_from_dftb_inp(run_root / "dftb.inp"),
        )
        relabel_outputs(outputs[1], outputs[2], transition)
        written += 1
        print(f"{system} {cv_dir} run {run_id}: {len(selected)} frames")
    print(f"Wrote {written} visualization bundles to {args.out_dir}")
    for message in skipped:
        print(f"SKIPPED: {message}")


if __name__ == "__main__":
    main()
