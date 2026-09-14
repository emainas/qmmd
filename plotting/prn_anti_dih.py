#!/usr/bin/env python3
"""Read live PRN trajectories without modifying runs; save a PNG and aligned CSV.

Example: python plotting/prn_anti_dih.py --runs 2,4,6-7,9,12-13,16,19-20
Plots dihedrals around 180 degrees for anti, or around 0 for PRN-syn paths.
Panels show circular mean +/- circular standard deviation over all finite frames.
Circular SD = sqrt(-2 ln R), converted to degrees; R is mean resultant length.
This describes angular spread, not the uncertainty of the mean.
Rerun to refresh the snapshot. The fixed atom quartet does not track proton hops.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
TIME = re.compile(rf"\bT\s*=\s*({NUMBER})\s*FSEC", re.IGNORECASE)


def run_ids(value: str) -> list[int]:
    result: set[int] = set()
    for part in value.split(","):
        bounds = part.strip().split("-")
        try:
            first, last = (int(bounds[0]), int(bounds[-1]))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Use run IDs such as 2,4,6-10") from exc
        if len(bounds) > 2 or first < 1 or last < first:
            raise argparse.ArgumentTypeError("Run IDs must be positive; ranges ascending")
        result.update(range(first, last + 1))
    return sorted(result)


def box_from_input(path: Path) -> np.ndarray:
    box = np.array([list(map(float, line[2:].split())) for line in path.read_text().splitlines()
                    if line.startswith("TV")])
    if box.shape != (3, 3) or not np.allclose(box, np.diag(np.diag(box))) or np.any(np.diag(box) <= 0):
        raise ValueError("Expected a positive orthorhombic box in dftb.inp")
    return np.diag(box)


def minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return delta - box * np.round(delta / box)


def dihedral(points: np.ndarray, box: np.ndarray) -> float:
    a, b, c, d = points
    axis = minimum_image(c - b, box)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return float("nan")
    axis /= norm
    v, w = minimum_image(a - b, box), minimum_image(d - c, box)
    v -= np.dot(v, axis) * axis
    w -= np.dot(w, axis) * axis
    if min(np.linalg.norm(v), np.linalg.norm(w)) < 1e-12:
        return float("nan")
    return float(np.degrees(np.arctan2(np.dot(np.cross(axis, v), w), np.dot(v, w))))


def circular_mean_deg(values: np.ndarray) -> float:
    """Return the mean direction in [0, 360), or NaN if undefined."""
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return float("nan")
    resultant = np.mean(np.exp(1j * np.radians(finite)))
    if abs(resultant) < 1e-12:
        return float("nan")
    mean = float(np.degrees(np.angle(resultant)) % 360)
    return 0.0 if np.isclose(mean, 360, rtol=0, atol=1e-10) else mean


def wrapped_plot_series(times: np.ndarray, signed: np.ndarray, center: float = 180) -> tuple[np.ndarray, np.ndarray]:
    """Break lines at the selected angular seam without discarding samples."""
    wrapped = (signed - center + 180) % 360 + center - 180
    seams = np.flatnonzero(np.abs(np.diff(wrapped)) > 180) + 1
    return np.insert(times, seams, np.nan), np.insert(wrapped, seams, np.nan)


def circular_std_deg(values: np.ndarray) -> float:
    """Circular SD sqrt(-2 log R) in degrees (the scipy.stats.circstd definition)."""
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return float("nan")
    length = float(np.clip(abs(np.mean(np.exp(1j * np.radians(finite)))), 0, 1))
    if 1 - length <= 4 * np.finfo(float).eps:
        return 0.0
    if length == 0:
        return float("inf")
    return float(np.degrees(np.sqrt(-2 * np.log(length))))


def read_series(path: Path, ids: list[int], box: np.ndarray) -> np.ndarray:
    """Read complete frames up to the initial file size, ignoring an incomplete tail.

    Columns: time_ps, signed_dihedral_deg, folded_dihedral_deg, O1_H11_distance_A.
    Full malformed frames and restarted/nonmonotonic timestamps are rejected.
    """
    rows = []
    with path.open("rb") as handle:
        limit = os.fstat(handle.fileno()).st_size

        def line() -> bytes:
            remaining = limit - handle.tell()
            if remaining <= 0:
                return b""
            value = handle.readline(remaining)
            return value if value.endswith(b"\n") else b""

        while count_line := line():
            if not count_line.strip():
                continue
            count = int(count_line)
            if count < max(ids) or min(ids) < 1:
                raise ValueError(f"Atom IDs outside trajectory atom count {count}")
            comment = line()
            if not comment:
                break
            frame = []
            for _ in range(count):
                atom = line()
                if not atom:
                    break
                frame.append(atom)
            if len(frame) != count:
                break
            match = TIME.search(comment.decode())
            if match is None:
                raise ValueError("Missing trajectory timestamp")
            time = float(match[1].replace("D", "E").replace("d", "e")) / 1000
            if rows and time <= rows[-1][0]:
                raise ValueError("Non-increasing timestamps: analyze restart segments separately")
            points = np.array([[float(v.replace(b"D", b"E").replace(b"d", b"e")) for v in frame[i-1].split()[1:4]] for i in ids])
            if points.shape != (4, 3) or not np.isfinite(points).all():
                raise ValueError("Invalid dihedral coordinates")
            phi = dihedral(points, box)
            distance = np.linalg.norm(minimum_image(points[3] - points[2], box))
            rows.append((time, phi, abs(phi), distance))
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-path", type=Path, default=ROOT / "systems/PRN-anti/solv_100/dftb/N1T48C1")
    parser.add_argument("--runs", type=run_ids, help="Comma-separated IDs and inclusive ranges; default: all existing runs")
    parser.add_argument("--atoms", nargs=4, type=int, default=[5, 3, 4, 11], metavar=("O2", "CG", "O1", "H11"), help="One-based quartet, default 5 3 4 11")
    parser.add_argument("--anti-window", "--state-window", type=float, default=30, help="Shade within this many degrees of the selected center (default 30)")
    parser.add_argument("--center", type=int, choices=[0, 180], help="Angular center; default 0 for PRN-syn paths, otherwise 180")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/PRN-anti/prn_anti_dih.png")
    args = parser.parse_args()
    center = args.center if args.center is not None else (0 if "PRN-syn" in args.runs_path.resolve().parts else 180)
    state = "syn" if center == 0 else "anti"
    if not 0 < args.anti_window < 180 or min(args.atoms) < 1 or len(set(args.atoms)) != 4:
        parser.error("Require four distinct positive atom IDs and 0 < anti-window < 180")
    if args.out.suffix.lower() != ".png":
        parser.error("--out must end in .png")
    out = args.out.resolve()
    if out.is_relative_to((ROOT / "systems").resolve()) or out.is_relative_to(args.runs_path.resolve()):
        parser.error("Save plots outside simulation directories")
    selected = args.runs if args.runs is not None else sorted(int(p.name[4:]) for p in args.runs_path.glob("run-*") if p.name[4:].isdigit())
    series = {}
    for run in selected:
        folder = args.runs_path / f"run-{run}/equil"
        if not (folder / "traject").exists():
            print(f"run-{run}: no trajectory yet; skipped")
            continue
        data = read_series(folder / "traject", args.atoms, box_from_input(folder / "dftb.inp"))
        if data.size:
            series[run] = data
            print(f"run-{run}: {len(data)} frames, through {data[-1, 0]:.3f} ps; circular mean={circular_mean_deg(data[:, 1]):.2f} degrees", flush=True)
    if not series:
        raise SystemExit("No complete trajectory frames found")
    cols = min(3, len(series))
    fig, axes = plt.subplots(math.ceil(len(series) / cols), cols, figsize=(5 * cols, 2.8 * math.ceil(len(series) / cols)), squeeze=False, sharex=True, sharey=True)
    for ax, (run, data) in zip(axes.flat, series.items()):
        ax.axhspan(center - args.anti_window, center + args.anti_window, color="tab:blue", alpha=.10)
        times, angles = wrapped_plot_series(data[:, 0], data[:, 1], center)
        ax.plot(times, angles, lw=.8, color="tab:blue", marker=".", markersize=1.5)
        mean = circular_mean_deg(data[:, 1])
        mean = (mean - center + 180) % 360 + center - 180
        std = circular_std_deg(data[:, 1])
        label = f"Circular: {mean:.1f}° ± {std:.1f}°" if np.isfinite(mean) else "Circular mean: undefined"
        ax.text(.97, .82, label, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
        ax.set(title=f"run-{run}", ylim=(center - 180, center + 180), yticks=np.arange(center - 180, center + 181, 90), xlabel="Time (ps)", ylabel="Dihedral (degrees)")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(series):]:
        ax.set_visible(False)
    reference = "syn = 0°, anti = ±180°" if center == 0 else "anti = 180°, syn = 0°/360°"
    fig.suptitle(f"PRN O2–CG–O1–H11: {reference}\nCircular mean ± SD over all sampled frames; shading: within {args.anti_window:g}° of {state}",
                 y=1 - 0.15 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / fig.get_figheight()))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    with out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "time_ps", "signed_dihedral_deg", "folded_dihedral_deg", "O1_H11_distance_A", "dihedral_0_360_deg", "circular_mean_deg", "circular_std_deg", "plotted_dihedral_deg", "plotted_circular_mean_deg"])
        for run, data in series.items():
            mean = circular_mean_deg(data[:, 1])
            std = circular_std_deg(data[:, 1])
            writer.writerows((run, *row, row[1] % 360, mean, std,
                              (row[1] - center + 180) % 360 + center - 180,
                              (mean - center + 180) % 360 + center - 180) for row in data)
    print(f"Saved {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
