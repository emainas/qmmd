#!/usr/bin/env python3
"""Plot the negative proton-defect distance for BV torsional US windows.

The negative defect is assigned independently in each Mulliken frame as the
most-negative solvent oxygen.  Its distance is the periodic minimum distance
to any BV atom (atoms 1..``solute_atoms``), including hydrogens.  Mulliken and
trajectory records are matched by their explicit DFTB timestamps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml

from acid_base_BV import read_box_lengths_from_dftb_inp, read_xyz_symbols
from defect_distance_check import nearest_solute_for_frame


TIME_RE = re.compile(r"\*\*\*\s+AT\s+T=\s*([+\-0-9.EeDd]+)\s*FSEC", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DefectFrame:
    time_ps: float
    oxygen_id: int
    charge_e: float
    second_charge_e: float


@dataclass(frozen=True, slots=True)
class WindowSeries:
    index: int
    center_deg: float
    times_ps: np.ndarray
    defect_ids: np.ndarray
    defect_charges_e: np.ndarray
    second_charges_e: np.ndarray
    closest_atom_ids: np.ndarray
    closest_atom_symbols: tuple[str, ...]
    distances_A: np.ndarray
    source_last_time_ps: float


def window_centers(pull_spec: Path) -> list[float]:
    """Read inclusive umbrella centers without assuming their direction."""
    data = yaml.safe_load(pull_spec.read_text())
    spec = data["windows"]
    start = float(spec["start_deg"])
    stop = float(spec["stop_deg"])
    spacing = float(spec["spacing_deg"])
    if spacing <= 0.0:
        raise ValueError("windows.spacing_deg must be positive")
    direction = 1.0 if stop >= start else -1.0
    count_float = abs(stop - start) / spacing
    count = int(round(count_float))
    if not math.isclose(count_float, count, abs_tol=1.0e-10):
        raise ValueError("Window range is not exactly divisible by spacing")
    return [start + direction * spacing * index for index in range(count + 1)]


def iter_complete_xyz_frames(path: Path) -> Iterator[tuple[float, np.ndarray]]:
    """Yield complete timestamped XYZ frames, ignoring only a partial tail."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        previous_time = -np.inf
        while True:
            count_line = handle.readline()
            if not count_line:
                return
            if not count_line.strip():
                continue
            try:
                natoms = int(count_line)
            except ValueError as exc:
                raise ValueError(f"Invalid XYZ atom-count line in {path}: {count_line!r}") from exc
            comment = handle.readline()
            if not comment:
                return
            match = TIME_RE.search(comment)
            if match is None:
                raise ValueError(f"Missing timestamp in XYZ frame comment: {comment.rstrip()}")
            time_ps = float(match.group(1).replace("D", "E").replace("d", "e")) / 1000.0
            rows: list[list[float]] = []
            complete = True
            for _ in range(natoms):
                line = handle.readline()
                if not line or not line.endswith("\n"):
                    complete = False
                    break
                fields = line.split()
                if len(fields) < 4:
                    raise ValueError(f"Malformed XYZ coordinate in {path}: {line.rstrip()}")
                rows.append([float(value) for value in fields[1:4]])
            if not complete:
                return
            if time_ps <= previous_time:
                raise ValueError(f"Non-increasing trajectory timestamps in {path}")
            previous_time = time_ps
            yield time_ps, np.asarray(rows, dtype=float)


def iter_negative_defect_frames(
    path: Path, solvent_oxygen_ids: Sequence[int]
) -> Iterator[DefectFrame]:
    """Yield the most-negative solvent oxygen from every complete charge frame."""
    oxygen_ids = tuple(int(atom_id) for atom_id in solvent_oxygen_ids)
    oxygen_set = set(oxygen_ids)
    if not oxygen_ids:
        raise ValueError("At least one solvent oxygen ID is required")

    current_time: float | None = None
    charges: dict[int, float] = {}
    contributions: dict[int, int] = {}
    previous_time = -np.inf

    def finish(*, allow_partial: bool) -> DefectFrame | None:
        if current_time is None:
            return None
        missing = [atom_id for atom_id in oxygen_ids if contributions.get(atom_id, 0) == 0]
        if missing:
            if allow_partial:
                return None
            raise ValueError(
                f"Incomplete interior Mulliken frame at {current_time:g} ps in {path}; "
                f"missing {len(missing)} solvent oxygens"
            )
        ordered = sorted((charges[atom_id], atom_id) for atom_id in oxygen_ids)
        return DefectFrame(
            time_ps=current_time,
            oxygen_id=ordered[0][1],
            charge_e=ordered[0][0],
            second_charge_e=ordered[1][0] if len(ordered) > 1 else float("nan"),
        )

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = TIME_RE.search(line)
            if match:
                completed = finish(allow_partial=False)
                if completed is not None:
                    yield completed
                current_time = (
                    float(match.group(1).replace("D", "E").replace("d", "e")) / 1000.0
                )
                if current_time <= previous_time:
                    raise ValueError(f"Non-increasing Mulliken timestamps in {path}")
                previous_time = current_time
                charges = {}
                contributions = {}
                continue

            if current_time is None:
                continue
            fields = line.split()
            if len(fields) < 4:
                continue
            try:
                atom_id = int(fields[0])
            except ValueError:
                continue
            if atom_id not in oxygen_set:
                continue
            try:
                contribution = float(fields[3].replace("D", "E").replace("d", "e"))
            except ValueError as exc:
                raise ValueError(f"Invalid Mulliken value in {path}: {line.rstrip()}") from exc
            charges[atom_id] = charges.get(atom_id, 0.0) + contribution
            contributions[atom_id] = contributions.get(atom_id, 0) + 1

    completed = finish(allow_partial=True)
    if completed is not None:
        yield completed


def analyze_window(
    window_dir: Path,
    index: int,
    center_deg: float,
    solute_atoms: int,
) -> WindowSeries:
    """Calculate one timestamp-matched defect-distance series."""
    stage = window_dir / "equil"
    trajectory = stage / "traject"
    mulliken = stage / "mulliken"
    dftb_input = stage / "dftb.inp"
    for path in (trajectory, mulliken, dftb_input):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty equilibration input: {path}")

    symbols = read_xyz_symbols(trajectory)
    if not 1 <= solute_atoms < len(symbols):
        raise ValueError(f"Invalid solute atom count {solute_atoms} for {trajectory}")
    solvent_oxygen_ids = [
        atom_id
        for atom_id, symbol in enumerate(symbols, start=1)
        if atom_id > solute_atoms and symbol.upper() == "O"
    ]
    if not solvent_oxygen_ids:
        raise ValueError(f"No solvent oxygen atoms found in {trajectory}")
    box = read_box_lengths_from_dftb_inp(dftb_input)

    rows: list[tuple[float, int, float, float, int, str, float]] = []
    trajectory_frames = iter_complete_xyz_frames(trajectory)
    defect_frames = iter_negative_defect_frames(mulliken, solvent_oxygen_ids)
    for pair_index, pair in enumerate(zip_longest(trajectory_frames, defect_frames)):
        trajectory_frame, defect_frame = pair
        if trajectory_frame is None or defect_frame is None:
            # A concurrently written final frame may exist in only one stream.
            break
        time_ps, coordinates = trajectory_frame
        if len(coordinates) != len(symbols):
            raise ValueError(f"Trajectory atom count changes in {trajectory}")
        if not math.isclose(time_ps, defect_frame.time_ps, abs_tol=1.0e-8):
            raise ValueError(
                f"Timestamp mismatch at aligned frame {pair_index}: trajectory "
                f"{time_ps:g} ps, Mulliken {defect_frame.time_ps:g} ps"
            )
        closest_id, distance = nearest_solute_for_frame(
            coordinates,
            defect_frame.oxygen_id,
            solute_atoms,
            box,
        )
        rows.append(
            (
                time_ps,
                defect_frame.oxygen_id,
                defect_frame.charge_e,
                defect_frame.second_charge_e,
                closest_id,
                symbols[closest_id - 1],
                distance,
            )
        )
    if not rows:
        raise ValueError(f"No aligned complete trajectory/Mulliken frames in {stage}")

    return WindowSeries(
        index=index,
        center_deg=center_deg,
        times_ps=np.asarray([row[0] for row in rows]),
        defect_ids=np.asarray([row[1] for row in rows], dtype=int),
        defect_charges_e=np.asarray([row[2] for row in rows]),
        second_charges_e=np.asarray([row[3] for row in rows]),
        closest_atom_ids=np.asarray([row[4] for row in rows], dtype=int),
        closest_atom_symbols=tuple(row[5] for row in rows),
        distances_A=np.asarray([row[6] for row in rows]),
        source_last_time_ps=float(rows[-1][0]),
    )


def write_csv(path: Path, series: Sequence[WindowSeries]) -> None:
    """Write all frame-aligned values represented in the plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "window_index",
                "window_center_deg",
                "time_ps",
                "negative_defect_oxygen_id",
                "negative_defect_charge_e",
                "second_most_negative_oxygen_charge_e",
                "closest_BV_atom_id",
                "closest_BV_atom_element",
                "defect_closest_BV_atom_distance_A",
            ]
        )
        for item in series:
            for row in zip(
                item.times_ps,
                item.defect_ids,
                item.defect_charges_e,
                item.second_charges_e,
                item.closest_atom_ids,
                item.closest_atom_symbols,
                item.distances_A,
            ):
                time_ps, defect_id, defect_charge, second_charge, atom_id, symbol, distance = row
                writer.writerow(
                    [
                        item.index,
                        f"{item.center_deg:g}",
                        f"{time_ps:.10g}",
                        int(defect_id),
                        f"{defect_charge:.10g}",
                        f"{second_charge:.10g}",
                        int(atom_id),
                        symbol,
                        f"{distance:.10g}",
                    ]
                )


def plot_series(path: Path, series: Sequence[WindowSeries], style: Path | None) -> None:
    """Render one vertically stacked distance subplot per umbrella window."""
    if style is not None and style.is_file():
        plt.style.use(style)
    colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, len(series)))
    fig, axes = plt.subplots(
        len(series),
        1,
        figsize=(9.0, max(15.0, 1.15 * len(series))),
        dpi=220,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes_array = np.atleast_1d(axes)
    maximum_time = max(float(item.times_ps[-1]) for item in series)
    for axis, color, item in zip(axes_array, colors, series):
        axis.plot(item.times_ps, item.distances_A, color=color, linewidth=1.15)
        axis.text(
            0.012,
            0.90,
            f"w{item.index:03d}  ·  {item.center_deg:g}°  ·  {len(item.times_ps)} frames",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=color,
        )
        axis.set_ylim(bottom=0.0)
        axis.tick_params(axis="both", labelsize=8)
    axes_array[-1].set_xlim(0.0, maximum_time)
    axes_array[-1].set_xlabel("DFTB equilibration time (ps)")
    fig.supylabel(r"O(defect$^-$)–nearest BV atom distance ($\mathrm{\AA}$)")
    fig.suptitle(
        r"BV solv 4.0 · $\psi_{cd}$ umbrella equilibration · negative-defect distance",
        fontsize=15,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.root.resolve()
    centers = window_centers(root / "pull_spec.yaml")
    series: list[WindowSeries] = []
    for index, center in enumerate(centers):
        item = analyze_window(
            root / f"window-{index:03d}", index, center, args.solute_atoms
        )
        series.append(item)
        print(
            f"window-{index:03d}: {len(item.times_ps)} frames through "
            f"{item.source_last_time_ps:g} ps",
            flush=True,
        )

    output = args.out or root / "equil-defect-distance.png"
    data_output = args.data_out or root / "equil_defect_distance.csv"
    provenance_output = output.with_name(f"{output.stem}_provenance.json")
    write_csv(data_output, series)
    plot_series(output, series, args.style)
    provenance = {
        "system": "BV",
        "umbrella_root": str(root),
        "solute_atom_ids": [1, args.solute_atoms],
        "distance_definition": (
            "periodic minimum distance from the dynamically assigned negative-defect "
            "oxygen to any BV atom, including hydrogens"
        ),
        "defect_assignment": (
            "most-negative solvent-oxygen summed Mulliken charge independently at "
            "each explicit timestamp; no identity carry-forward and no charge threshold"
        ),
        "time_alignment": "exact explicit DFTB timestamps in traject and mulliken",
        "windows": [
            {
                "index": item.index,
                "center_deg": item.center_deg,
                "frames": len(item.times_ps),
                "last_time_ps": item.source_last_time_ps,
                "unique_defect_oxygen_ids": int(len(np.unique(item.defect_ids))),
            }
            for item in series
        ],
        "figure": str(output),
        "aligned_data": str(data_output),
    }
    provenance_output.write_text(json.dumps(provenance, indent=2) + "\n")
    return output, data_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--solute-atoms", type=int, default=78)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--data-out", type=Path, default=None)
    parser.add_argument(
        "--style",
        type=Path,
        default=Path(__file__).resolve().with_name("lefteris.mplstyle"),
    )
    args = parser.parse_args()
    if args.solute_atoms < 1:
        parser.error("--solute-atoms must be positive")
    output, data_output = run(args)
    print(f"Wrote {output}")
    print(f"Wrote {data_output}")


if __name__ == "__main__":
    main()
