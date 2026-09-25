#!/usr/bin/env python3
"""Build a read-only, first-10-ps CPP LCOD WHAM preview from live windows."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


TIME_RE = re.compile(
    r"AT\s+T=\s*([-+0-9.EeDd]+)\s+FSEC.*STEP\s+NO\.=\s*(\d+)",
    re.IGNORECASE,
)
HEADER_RE = re.compile(r"^\s*(\d+)\s*$")
KJ_PER_KCAL = 4.184


@dataclass(frozen=True, slots=True)
class Frame:
    time_ps: float
    step: int
    symbols: tuple[str, ...]
    coordinates: np.ndarray


def repository_root(start: Path) -> Path:
    for path in (start.resolve(), *start.resolve().parents):
        if (path / "pyproject.toml").is_file():
            return path
    raise RuntimeError("Could not locate repository root")


def read_box_lengths(path: Path) -> tuple[float, float, float]:
    vectors: list[list[float]] = []
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[0] == "TV":
            vectors.append([float(value) for value in fields[1:]])
    if len(vectors) != 3:
        raise ValueError(f"Expected three TV vectors in {path}")
    box = np.asarray(vectors)
    if not np.allclose(box, np.diag(np.diag(box)), atol=1.0e-8):
        raise ValueError("This preview currently requires an orthorhombic box")
    return tuple(float(value) for value in np.diag(box))


def read_initial_frames(path: Path, end_ps: float) -> list[Frame]:
    """Take an immutable in-memory snapshot of complete frames through end_ps."""
    frames: list[Frame] = []
    with path.open(errors="replace") as stream:
        while True:
            count_line = stream.readline()
            if not count_line:
                break
            if not count_line.strip():
                continue
            count_match = HEADER_RE.match(count_line)
            if count_match is None:
                raise ValueError(f"Malformed XYZ atom count in {path}: {count_line!r}")
            atom_count = int(count_match.group(1))
            comment = stream.readline()
            time_match = TIME_RE.search(comment)
            if time_match is None:
                raise ValueError(f"Missing time/step record in {path}: {comment!r}")
            rows: list[tuple[str, float, float, float]] = []
            complete = True
            for _ in range(atom_count):
                line = stream.readline()
                if not line:
                    complete = False
                    break
                fields = line.split()
                if len(fields) < 4:
                    raise ValueError(f"Malformed XYZ coordinate in {path}: {line!r}")
                rows.append(
                    (fields[0], float(fields[1]), float(fields[2]), float(fields[3]))
                )
            if not complete:
                break
            time_ps = float(
                time_match.group(1).replace("D", "E").replace("d", "e")
            ) / 1000.0
            if time_ps > end_ps + 1.0e-10:
                break
            frames.append(
                Frame(
                    time_ps=time_ps,
                    step=int(time_match.group(2)),
                    symbols=tuple(row[0] for row in rows),
                    coordinates=np.asarray([row[1:] for row in rows], dtype=float),
                )
            )
    if len(frames) < 2 or frames[0].time_ps > 1.0e-10 or frames[-1].time_ps < end_ps - 1.0e-8:
        raise RuntimeError(
            f"{path} does not yet contain the complete 0-{end_ps:g} ps interval"
        )
    times = np.asarray([frame.time_ps for frame in frames])
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Non-increasing frame times in {path}")
    return frames


def write_cpptraj_xyz(
    path: Path, frames: list[Frame], box: tuple[float, float, float]
) -> None:
    with path.open("w") as stream:
        for frame in frames:
            stream.write(f"{len(frame.symbols)}\n")
            stream.write(
                f"Box X: {box[0]:.8f} 0.0 0.0 Y: 0.0 {box[1]:.8f} 0.0 "
                f"Z: 0.0 0.0 {box[2]:.8f}\n"
            )
            for symbol, xyz in zip(frame.symbols, frame.coordinates):
                stream.write(
                    f"{symbol:<2s} {xyz[0]:18.10f} {xyz[1]:18.10f} {xyz[2]:18.10f}\n"
                )


def cpptraj_path(path: Path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def run_cpptraj_window(
    executable: Path,
    topology: Path,
    trajectory: Path,
    output_dir: Path,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    nhb_path = output_dir / f"window-{index:03d}-NB19-H20.dat"
    nhc_path = output_dir / f"window-{index:03d}-NC31-H20.dat"
    input_path = output_dir / f"window-{index:03d}.in"
    log_path = output_dir / f"window-{index:03d}.log"
    input_path.write_text(
        "\n".join(
            [
                f"parm {cpptraj_path(topology)}",
                f"trajin {cpptraj_path(trajectory)}",
                "fiximagedbonds :1",
                "autoimage",
                f"distance NB19_H20 @19 @20 out {cpptraj_path(nhb_path)}",
                f"distance NC31_H20 @31 @20 out {cpptraj_path(nhc_path)}",
                "run",
                "quit",
                "",
            ]
        )
    )
    result = subprocess.run(
        [str(executable), "-i", str(input_path)],
        cwd=output_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"cpptraj failed for window {index}; see {log_path}")

    def read_distance(path: Path) -> np.ndarray:
        values: list[float] = []
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            values.append(float(fields[1]))
        if not values or not np.all(np.isfinite(values)):
            raise ValueError(f"No finite cpptraj distances in {path}")
        return np.asarray(values)

    return read_distance(nhb_path), read_distance(nhc_path)


def write_wham_series(path: Path, values: np.ndarray) -> None:
    with path.open("w") as stream:
        stream.write("#Frame LCOD_A\n")
        for frame, value in enumerate(values, start=1):
            stream.write(f"{frame:8d} {value:14.8f}\n")


def read_wham(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinate: list[float] = []
    free_energy: list[float] = []
    probability: list[float] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        coordinate.append(float(fields[0]))
        free_energy.append(float(fields[1]))
        probability.append(float(fields[3]))
    return np.asarray(coordinate), np.asarray(free_energy), np.asarray(probability)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-ps", type=float, default=10.0)
    parser.add_argument(
        "--cpptraj",
        type=Path,
        default=Path("/nas/sycamore/apps/amber/26/ambertools26/bin/cpptraj"),
    )
    parser.add_argument(
        "--wham",
        type=Path,
        default=Path("/nas/sycamore/home/emainas/software/wham/wham/wham"),
    )
    args = parser.parse_args()
    if not math.isfinite(args.end_ps) or args.end_ps <= 0:
        parser.error("--end-ps must be positive and finite")
    root = repository_root(Path(__file__))
    equil_yaml = root / "configs/CPP/us/equil.yaml"
    pull_yaml = root / "configs/CPP/us/pull.yaml"
    equil = yaml.safe_load(equil_yaml.read_text())
    pull = yaml.safe_load(pull_yaml.read_text())
    us_root = root / "systems/CPP/solv_4.0/us-lcod"
    output = us_root / f"wham-preview-{args.end_ps:g}ps"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing preview: {output}")
    output.mkdir(parents=True)
    cpptraj_dir = output / "cpptraj"
    series_dir = output / "windows"
    cpptraj_dir.mkdir()
    series_dir.mkdir()

    if not args.cpptraj.is_file() or not args.wham.is_file():
        raise RuntimeError("cpptraj and WHAM executables must exist")
    topology = root / "systems/CPP/solv_4.0/salt/ready.parm7"
    if not topology.is_file():
        raise RuntimeError(f"Missing topology: {topology}")
    window_cfg = pull["windows"]
    centers = np.arange(
        float(window_cfg["start_angstrom"]),
        float(window_cfg["stop_angstrom"]) + 0.5 * float(window_cfg["spacing_angstrom"]),
        float(window_cfg["spacing_angstrom"]),
    )
    if len(centers) != 44:
        raise ValueError(f"Expected 44 LCOD windows, found {len(centers)}")

    all_rows: list[tuple[int, float, float, int, float, float, float]] = []
    summaries: list[tuple[int, float, int, float, float, float]] = []
    with tempfile.TemporaryDirectory(prefix="cpp-lcod-wham-") as tmp_name:
        tmp = Path(tmp_name)
        for index, center in enumerate(centers):
            stage = us_root / f"window-{index:03d}" / "equil"
            spec = stage / "equil_spec.yaml"
            if not spec.is_file() or spec.read_text() != equil_yaml.read_text():
                raise RuntimeError(f"Prepared snapshot mismatch in {stage}")
            source = stage / "traject"
            dftb_input = stage / "dftb.inp"
            frames = read_initial_frames(source, args.end_ps)
            converted = tmp / f"window-{index:03d}.xyz"
            write_cpptraj_xyz(converted, frames, read_box_lengths(dftb_input))
            nhb, nhc = run_cpptraj_window(
                args.cpptraj, topology, converted, cpptraj_dir, index
            )
            if len(nhb) != len(frames) or len(nhc) != len(frames):
                raise ValueError(
                    f"cpptraj/frame mismatch for window {index}: "
                    f"{len(nhb)}, {len(nhc)}, {len(frames)}"
                )
            lcod = nhb - nhc
            write_wham_series(series_dir / f"window-{index:03d}.dat", lcod)
            for frame, first, second, value in zip(frames, nhb, nhc, lcod):
                all_rows.append(
                    (index, center, frame.time_ps, frame.step, first, second, value)
                )
            summaries.append(
                (
                    index,
                    center,
                    len(lcod),
                    float(np.mean(lcod)),
                    float(np.std(lcod, ddof=1)),
                    frames[-1].time_ps,
                )
            )

    with (output / "lcod_timeseries.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "window_index",
                "center_A",
                "time_ps",
                "step",
                "r_NB19_H20_A",
                "r_NC31_H20_A",
                "lcod_A",
            ]
        )
        writer.writerows(all_rows)
    with (output / "window_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window_index", "center_A", "samples", "mean_A", "std_A", "end_ps"])
        writer.writerows(summaries)

    coefficient_kj = float(equil["wall"]["coefficient_kcal_mol_angstrom2"])
    force_kcal = 2.0 * coefficient_kj / KJ_PER_KCAL
    metadata = output / "metadata.dat"
    with metadata.open("w") as stream:
        for index, center in enumerate(centers):
            stream.write(
                f"windows/window-{index:03d}.dat {center:.8f} {force_kcal:.12g}\n"
            )
    result_path = output / "wham_result.dat"
    command = [
        str(args.wham),
        "-2.3",
        "2.6",
        "245",
        "1e-6",
        "300",
        "0",
        metadata.name,
        result_path.name,
    ]
    (output / "wham_command.txt").write_text(" ".join(command) + "\n")
    result = subprocess.run(
        command,
        cwd=output,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (output / "wham.log").write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"WHAM failed; see {output / 'wham.log'}")

    grid, raw_pmf, probability = read_wham(result_path)
    finite = np.isfinite(raw_pmf)
    if not np.any(finite):
        raise RuntimeError("WHAM returned no finite PMF bins")
    pmf = raw_pmf - float(np.nanmin(np.where(finite, raw_pmf, np.nan)))
    with (output / "pmf.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["lcod_A", "pmf_kcal_mol", "probability", "finite"])
        for x, free, prob, valid in zip(grid, pmf, probability, finite):
            writer.writerow([f"{x:.8f}", f"{free:.8f}", f"{prob:.12g}", int(valid)])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = root / "plotting/lefteris.mplstyle"
    if style.is_file():
        plt.style.use(style)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    ax.plot(grid[finite], pmf[finite], color="#315f9b", linewidth=2.2)
    ax.scatter(grid[finite], pmf[finite], color="#315f9b", s=9, zorder=3)
    ax.set(
        xlabel=r"LCOD = $r(\mathrm{N_B-H}) - r(\mathrm{N_C-H})$ (Å)",
        ylabel=r"PMF (kcal mol$^{-1}$)",
        title=f"CPP LCOD WHAM preview: first {args.end_ps:g} ps of equilibration",
        xlim=(-2.15, 2.45),
    )
    ax.grid(False)
    ax.text(
        0.02,
        0.98,
        f"Exploratory nonequilibrium estimate\n"
        f"{len(centers)} windows × {summaries[0][2]} frames; no bootstrap",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.savefig(output / "pmf-lcod-preview.png", dpi=300)
    plt.close(fig)

    manifest = {
        "source": "read-only live equilibration trajectories",
        "time_interval_ps": [0.0, args.end_ps],
        "frames_per_window": summaries[0][2],
        "windows": len(centers),
        "temperature_k": 300.0,
        "dcdftb_wall_coefficient_kj_mol_A2": coefficient_kj,
        "wham_half_harmonic_force_kcal_mol_A2": force_kcal,
        "wham_bins": 245,
        "bootstrap": False,
        "warning": (
            f"First {args.end_ps:g} ps of equilibration are not equilibrium "
            "production data."
        ),
    }
    (output / "analysis.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"OK: wrote exploratory PMF to {output / 'pmf-lcod-preview.png'}")
    print(f"OK: wrote aligned cpptraj LCOD data to {output / 'lcod_timeseries.csv'}")


if __name__ == "__main__":
    main()
