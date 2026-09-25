#!/usr/bin/env python3
"""Measure the MAH C2-C1-O1-HO torsion with cpptraj and plot every DFTB run."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CPPTRAJ_DEFAULT = Path("/nas/sycamore/apps/amber/26/ambertools26/bin/cpptraj")
ATOM_IDS = (5, 3, 1, 2)  # C2-C1-O1-HO, one-based
ATOM_SYMBOLS = ("C", "C", "O", "H")
TIME_RE = re.compile(r"\bT\s*=\s*([-+0-9.EeDd]+)\s*FSEC", re.IGNORECASE)


def read_box(path: Path) -> np.ndarray:
    """Return the positive orthorhombic box lengths from a DFTB input."""
    vectors = np.asarray(
        [list(map(float, line.split()[1:4])) for line in path.read_text().splitlines() if line.startswith("TV")],
        dtype=float,
    )
    if vectors.shape != (3, 3) or not np.allclose(vectors, np.diag(np.diag(vectors)), atol=1.0e-8):
        raise ValueError(f"Expected an orthorhombic box in {path}")
    box = np.diag(vectors)
    if np.any(box <= 0):
        raise ValueError(f"Non-positive box length in {path}")
    return box


def minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return delta - box * np.rint(delta / box)


def dihedral(points: np.ndarray, box: np.ndarray) -> float:
    """Signed periodic dihedral in degrees, matching cpptraj's convention."""
    a, b, c, d = points
    axis = minimum_image(c - b, box)
    axis /= np.linalg.norm(axis)
    first = minimum_image(a - b, box)
    last = minimum_image(d - c, box)
    first -= np.dot(first, axis) * axis
    last -= np.dot(last, axis) * axis
    return float(
        np.degrees(
            np.arctan2(np.dot(np.cross(axis, first), last), np.dot(first, last))
        )
    )


def read_complete_frames(path: Path, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Snapshot all complete frames present at open time and audit the requested torsion."""
    times: list[float] = []
    angles: list[float] = []
    with path.open("rb") as stream:
        limit = os.fstat(stream.fileno()).st_size

        def readline() -> bytes:
            remaining = limit - stream.tell()
            if remaining <= 0:
                return b""
            line = stream.readline(remaining)
            return line if line.endswith(b"\n") else b""

        while count_line := readline():
            if not count_line.strip():
                continue
            atom_count = int(count_line)
            if atom_count < max(ATOM_IDS):
                raise ValueError(f"Atom IDs exceed the {atom_count}-atom trajectory in {path}")
            comment = readline()
            if not comment:
                break
            match = TIME_RE.search(comment.decode(errors="replace"))
            if match is None:
                raise ValueError(f"Missing timestamp in {path}")
            selected: dict[int, tuple[str, np.ndarray]] = {}
            complete = True
            for atom_id in range(1, atom_count + 1):
                line = readline()
                if not line:
                    complete = False
                    break
                if atom_id in ATOM_IDS:
                    fields = line.split()
                    selected[atom_id] = (
                        fields[0].decode(),
                        np.asarray([float(value) for value in fields[1:4]], dtype=float),
                    )
            if not complete:
                break
            symbols = tuple(selected[atom_id][0] for atom_id in ATOM_IDS)
            if symbols != ATOM_SYMBOLS:
                raise ValueError(f"Unexpected MAH atom order in {path}: {symbols}")
            time_ps = float(match.group(1).replace("D", "E").replace("d", "e")) / 1000.0
            if times and time_ps <= times[-1]:
                raise ValueError(f"Non-increasing timestamps in {path}")
            points = np.asarray([selected[atom_id][1] for atom_id in ATOM_IDS])
            times.append(time_ps)
            angles.append(dihedral(points, box))
    if len(times) < 2:
        raise ValueError(f"Fewer than two complete frames in {path}")
    return np.asarray(times), np.asarray(angles)


def render_cpptraj_input(
    topology: Path, trajectory: Path, frame_count: int, box: np.ndarray
) -> str:
    a, b, c, d = ATOM_IDS
    return "\n".join(
        [
            f"parm {topology.resolve()}",
            f"trajin {trajectory.resolve()} 1 {frame_count}",
            f"box x {box[0]:.10f} y {box[1]:.10f} z {box[2]:.10f} alpha 90 beta 90 gamma 90",
            "fiximagedbonds :1",
            "autoimage",
            f"dihedral MAH_OH @{a} @{b} @{c} @{d} out oh-dihedral.dat",
            "run",
            "quit",
            "",
        ]
    )


def read_cpptraj_data(path: Path, expected: int) -> np.ndarray:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) != expected or [int(row[0]) for row in rows] != list(range(1, expected + 1)):
        raise ValueError(f"cpptraj frame mismatch in {path}")
    values = np.asarray([float(row[1]) for row in rows], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite cpptraj torsion in {path}")
    return (values + 180.0) % 360.0 - 180.0


def break_at_seam(times: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seams = np.flatnonzero(np.abs(np.diff(angles)) > 180.0) + 1
    return np.insert(times, seams, np.nan), np.insert(angles, seams, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "systems/MAH/solv_5.5/dftb/N1T48C1",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        default=ROOT / "systems/MAH/solv_5.5/salt/ready.parm7",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/MAH/solv_5.5/equil_oh_dihedral",
    )
    parser.add_argument("--cpptraj", type=Path, default=CPPTRAJ_DEFAULT)
    args = parser.parse_args()

    if not args.topology.is_file() or not args.cpptraj.is_file():
        raise FileNotFoundError("MAH topology or cpptraj executable is missing")
    run_dirs = sorted(
        args.runs_root.glob("run-*/equil"),
        key=lambda path: int(path.parent.name.split("-")[1]),
    )
    if len(run_dirs) != 10:
        raise ValueError(f"Expected 10 MAH equilibration runs, found {len(run_dirs)}")
    args.output.mkdir(parents=True, exist_ok=True)

    results: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    for stage in run_dirs:
        run = int(stage.parent.name.split("-")[1])
        trajectory = stage / "traject"
        box = read_box(stage / "dftb.inp")
        times, independently_calculated = read_complete_frames(trajectory, box)
        input_path = stage / "oh-dihedral.cpptraj.in"
        data_path = stage / "oh-dihedral.dat"
        log_path = stage / "oh-dihedral.cpptraj.log"
        input_path.write_text(render_cpptraj_input(args.topology, trajectory, len(times), box))
        if data_path.exists():
            data_path.unlink()
        with log_path.open("w") as log:
            subprocess.run(
                [str(args.cpptraj), "-i", input_path.name],
                cwd=stage,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        angles = read_cpptraj_data(data_path, len(times))
        errors = np.abs((angles - independently_calculated + 180.0) % 360.0 - 180.0)
        maximum_error = float(np.max(errors))
        if maximum_error > 0.002:
            raise ValueError(f"cpptraj/MIC audit failed for run-{run}: {maximum_error:g} degrees")
        results.append((run, times, angles, maximum_error))
        print(f"run-{run}: {len(times)} frames through {times[-1]:.2f} ps")

    csv_path = args.output / "mah_oh_dihedral.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "time_ps", "C2_C1_O1_HO_dihedral_deg"])
        for run, times, angles, _ in results:
            writer.writerows((run, f"{time:.8f}", f"{angle:.10f}") for time, angle in zip(times, angles))

    plt.style.use(Path(__file__).with_name("lefteris.mplstyle"))
    fig, axes = plt.subplots(5, 2, figsize=(8.2, 11.2), sharex=True, sharey=True)
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(results)))
    end_time = max(times[-1] for _, times, _, _ in results)
    for axis, color, (run, times, angles, _) in zip(axes.flat, colors, results):
        x, y = break_at_seam(times, angles)
        axis.plot(x, y, color=color, linewidth=0.65)
        axis.axhline(0.0, color="0.65", linewidth=0.55, linestyle="--")
        axis.set_title(f"run-{run}", fontsize=10)
        axis.set_xlim(0.0, max(40.0, end_time))
        axis.set_ylim(-180.0, 180.0)
        axis.set_yticks([-180, -90, 0, 90, 180])
        axis.text(
            0.98,
            0.08,
            f"{len(times)} frames · {times[-1]:.2f} ps",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
        )
    fig.supxlabel("DFTB equilibration time (ps)")
    fig.supylabel("C2–C1–O1–HO dihedral (degrees)")
    fig.suptitle("MAH O–H rotation · cpptraj · atoms 5–3–1–2", y=0.995)
    fig.tight_layout(rect=(0.025, 0.02, 1.0, 0.985))
    png_path = args.output / "mah_oh_dihedral.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"figure: {png_path}")
    print(f"data: {csv_path}")


if __name__ == "__main__":
    main()
