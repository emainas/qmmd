"""Report trajectory-derived LCOD traces from restrained CPP equilibration."""

from __future__ import annotations

import csv
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qmmd.us_lcod import (
    LCODEquilConfig,
    lcod_centers,
    load_equil_config,
    pull_root,
    read_box_vectors,
)
from qmmd.us_pull import find_repo_root


_TIME_RE = re.compile(
    r"AT\s+T=\s*([-+0-9.EeDd]+)\s+FSEC.*STEP\s+NO\.=\s*(\d+)",
    re.IGNORECASE,
)
def equilibration_duration_ps(header_lines: list[str]) -> float:
    """Return the configured DCDFTBMD endpoint from NSTEP and DELTAT."""
    values: dict[str, str] = {}
    for key in ("NSTEP", "DELTAT"):
        pattern = re.compile(rf"\b{key}\s*=\s*([^\s,)]+)", re.IGNORECASE)
        for line in header_lines:
            match = pattern.search(line)
            if match:
                values[key] = match.group(1)
                break
    if set(values) != {"NSTEP", "DELTAT"}:
        raise ValueError("Equilibration NSTEP and DELTAT must be explicit")
    nstep = int(values["NSTEP"])
    timestep_seconds = float(
        values["DELTAT"].replace("D", "E").replace("d", "e")
    )
    if nstep <= 0 or not math.isfinite(timestep_seconds) or timestep_seconds <= 0:
        raise ValueError("Equilibration NSTEP and DELTAT must be positive")
    return nstep * timestep_seconds * 1.0e12


@dataclass(frozen=True, slots=True)
class LCODEquilSample:
    window_index: int
    target_angstrom: float
    time_ps: float
    step: int
    lcod_angstrom: float


@dataclass(frozen=True, slots=True)
class PullSeed:
    window_index: int
    target_angstrom: float
    selected_angstrom: float


@dataclass(frozen=True, slots=True)
class DensityTrace:
    window_index: int
    target_angstrom: float
    histogram_centers_angstrom: np.ndarray
    histogram_density_per_angstrom: np.ndarray
    gaussian_x_angstrom: np.ndarray
    gaussian_density_per_angstrom: np.ndarray
    empirical_mean_angstrom: float
    empirical_sd_angstrom: float


def read_pull_seeds(root: Path, centers: list[float], pull_yaml_text: str) -> list[PullSeed]:
    """Read and validate the handmade seed-selection table."""
    spec = root / "pull_spec.yaml"
    if not spec.is_file() or spec.read_text() != pull_yaml_text:
        raise RuntimeError(f"Prepared LCOD pull snapshot does not match config: {spec}")
    seeds: list[PullSeed] = []
    with (root / "windows.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            seeds.append(
                PullSeed(
                    window_index=int(row["window_index"]),
                    target_angstrom=float(row["target_lcod_A"]),
                    selected_angstrom=float(row["selected_lcod_A"]),
                )
            )
    if len(seeds) != len(centers) or [seed.window_index for seed in seeds] != list(range(len(centers))):
        raise ValueError("Handmade LCOD seed table does not contain the complete ordered window set")
    if any(not math.isclose(seed.target_angstrom, center, abs_tol=1.0e-8)
           for seed, center in zip(seeds, centers)):
        raise ValueError("Handmade LCOD seed targets do not match pull.yaml")
    if any(not math.isfinite(seed.selected_angstrom) for seed in seeds):
        raise ValueError("Handmade LCOD seed coordinates must be finite")
    return seeds


def read_complete_trajectory_times(path: Path) -> list[tuple[int, float]]:
    """Read (step, ps) for every complete frame, ignoring a growing-file tail."""
    records: list[tuple[int, float]] = []
    with path.open(errors="replace") as stream:
        while True:
            count_line = stream.readline()
            if not count_line:
                break
            if not count_line.strip():
                continue
            try:
                atom_count = int(count_line)
            except ValueError as exc:
                raise ValueError(f"Malformed XYZ atom count in {path}: {count_line!r}") from exc
            comment = stream.readline()
            match = _TIME_RE.search(comment)
            if match is None:
                if not comment:
                    break
                raise ValueError(f"Missing DFTB time/step in {path}: {comment!r}")
            for _ in range(atom_count):
                if not stream.readline():
                    return records
            step = int(match.group(2))
            time_ps = float(match.group(1).replace("D", "E").replace("d", "e")) / 1000.0
            if records and (step <= records[-1][0] or time_ps <= records[-1][1]):
                raise ValueError(f"Non-increasing trajectory step/time in {path}")
            records.append((step, time_ps))
    return records


def render_lcod_cpptraj_input(
    topology: Path,
    trajectory: Path,
    frame_count: int,
    box: np.ndarray,
    atoms: tuple[int, int, int, int],
) -> str:
    """Render a rerunnable cpptraj script, including the DFTB fixed box."""
    if frame_count < 2:
        raise ValueError("At least two complete trajectory frames are required")
    if box.shape != (3, 3) or not np.allclose(box, np.diag(np.diag(box)), atol=1.0e-8):
        raise ValueError("CPP LCOD cpptraj report requires an orthorhombic box")
    if np.any(np.diag(box) <= 0):
        raise ValueError("Box lengths must be positive")
    a, b, c, d = atoms
    return "\n".join(
        [
            f"parm {topology}",
            f"trajin {trajectory} 1 {frame_count}",
            ("box x {0:.10f} y {1:.10f} z {2:.10f} "
             "alpha 90 beta 90 gamma 90").format(*np.diag(box)),
            "fiximagedbonds :1",
            "autoimage",
            f"distance NBH @{a} @{b}",
            f"distance NCH @{c} @{d}",
            "run",
            "calc LCOD = NBH - NCH",
            "writedata lcod.dat LCOD prec 18.10",
            "quit",
            "",
        ]
    )


def read_cpptraj_lcod(path: Path, expected_frames: int) -> np.ndarray:
    frames: list[int] = []
    values: list[float] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 2:
            raise ValueError(f"Malformed cpptraj LCOD row in {path}: {line!r}")
        frames.append(int(fields[0]))
        values.append(float(fields[1]))
    array = np.asarray(values, dtype=float)
    if frames != list(range(1, expected_frames + 1)) or not np.all(np.isfinite(array)):
        raise ValueError(f"cpptraj LCOD frames do not match complete trajectory in {path}")
    return array


def resolve_cpptraj() -> Path:
    found = shutil.which("cpptraj")
    if found:
        return Path(found).resolve()
    installed = Path("/nas/sycamore/apps/amber/26/ambertools26/bin/cpptraj")
    if installed.is_file():
        return installed
    raise RuntimeError("cpptraj not found; load Amber before running this report")


def collect_lcod_equil_samples(
    cfg: LCODEquilConfig, yaml_text: str, repo_root: Path
) -> tuple[list[LCODEquilSample], list[int]]:
    """Measure LCOD with cpptraj from every complete equilibration XYZ frame."""
    root = pull_root(cfg.pull, repo_root)
    topology = repo_root / "systems" / cfg.pull.system / f"{cfg.pull.prefix}_{cfg.pull.buffer:.1f}" / "salt" / "ready.parm7"
    if not topology.is_file():
        raise RuntimeError(f"Missing CPP Amber topology: {topology}")
    cpptraj = resolve_cpptraj()
    samples: list[LCODEquilSample] = []
    missing: list[int] = []
    for index, center in enumerate(lcod_centers(cfg.pull.windows)):
        stage = root / f"window-{index:03d}" / cfg.stage_dirname
        spec = stage / "equil_spec.yaml"
        trajectory = stage / "traject"
        dftb_input = stage / "dftb.inp"
        if not spec.is_file() or spec.read_text() != yaml_text:
            missing.append(index)
            continue
        if not trajectory.is_file() or not dftb_input.is_file():
            missing.append(index)
            continue
        times = read_complete_trajectory_times(trajectory)
        if len(times) < 2:
            missing.append(index)
            continue
        input_path = stage / "lcod.cpptraj.in"
        data_path = stage / "lcod.dat"
        log_path = stage / "lcod.cpptraj.log"
        script = render_lcod_cpptraj_input(
            topology, trajectory, len(times), read_box_vectors(dftb_input), cfg.cv.atoms
        )
        input_path.write_text(script)
        with log_path.open("w") as log:
            subprocess.run(
                [str(cpptraj), "-i", input_path.name],
                cwd=stage,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        values = read_cpptraj_lcod(data_path, len(times))
        samples.extend(
            LCODEquilSample(index, center, time_ps, step, float(value))
            for (step, time_ps), value in zip(times, values)
        )
    if not samples:
        raise RuntimeError(f"No complete LCOD trajectory frames found under {root}")
    return samples, missing


def write_lcod_samples_csv(path: Path, samples: list[LCODEquilSample]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "window_index",
                "target_lcod_A",
                "time_ps",
                "step",
                "lcod_A",
                "deviation_A",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.window_index,
                    f"{sample.target_angstrom:.8f}",
                    f"{sample.time_ps:.8f}",
                    sample.step,
                    f"{sample.lcod_angstrom:.8f}",
                    f"{sample.lcod_angstrom - sample.target_angstrom:.8f}",
                ]
            )


def build_density_traces(
    samples: list[LCODEquilSample], bins: int = 30
) -> list[DensityTrace]:
    """Build normalized per-window histograms and Gaussian visual guides."""
    if bins < 2:
        raise ValueError("Density histogram needs at least two bins")
    grouped: dict[int, list[float]] = {}
    targets: dict[int, float] = {}
    for sample in samples:
        grouped.setdefault(sample.window_index, []).append(sample.lcod_angstrom)
        previous = targets.setdefault(sample.window_index, sample.target_angstrom)
        if not math.isclose(previous, sample.target_angstrom, abs_tol=1.0e-8):
            raise ValueError(f"Inconsistent center for window {sample.window_index}")
    traces: list[DensityTrace] = []
    for index, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        if len(array) < 2 or not np.all(np.isfinite(array)):
            raise ValueError(f"Window {index} needs at least two finite LCOD samples")
        mean = float(np.mean(array))
        sd = float(np.std(array, ddof=1))
        if sd <= 0 or not math.isfinite(sd):
            raise ValueError(f"Window {index} has zero or non-finite LCOD spread")
        histogram, edges = np.histogram(array, bins=bins, density=True)
        gaussian_x = np.linspace(mean - 4.0 * sd, mean + 4.0 * sd, 240)
        gaussian_y = np.exp(-0.5 * ((gaussian_x - mean) / sd) ** 2)
        gaussian_y /= sd * math.sqrt(2.0 * math.pi)
        traces.append(
            DensityTrace(
                window_index=index,
                target_angstrom=targets[index],
                histogram_centers_angstrom=0.5 * (edges[:-1] + edges[1:]),
                histogram_density_per_angstrom=histogram,
                gaussian_x_angstrom=gaussian_x,
                gaussian_density_per_angstrom=gaussian_y,
                empirical_mean_angstrom=mean,
                empirical_sd_angstrom=sd,
            )
        )
    return traces


def write_density_csv(path: Path, traces: list[DensityTrace]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["window_index", "target_lcod_A", "kind", "lcod_A",
             "probability_density_per_A", "empirical_mean_A", "empirical_sd_A"]
        )
        for trace in traces:
            for kind, coordinates, densities in (
                ("histogram", trace.histogram_centers_angstrom,
                 trace.histogram_density_per_angstrom),
                ("gaussian_guide", trace.gaussian_x_angstrom,
                 trace.gaussian_density_per_angstrom),
            ):
                for coordinate, density in zip(coordinates, densities):
                    writer.writerow(
                        [trace.window_index, f"{trace.target_angstrom:.8f}", kind,
                         f"{coordinate:.8f}", f"{density:.12g}",
                         f"{trace.empirical_mean_angstrom:.8f}",
                         f"{trace.empirical_sd_angstrom:.8f}"]
                    )


def plot_lcod_samples(
    path: Path,
    samples: list[LCODEquilSample],
    seeds: list[PullSeed],
    density_traces: list[DensityTrace],
    centers: list[float],
    expected_time_ps: float,
    style_path: Path | None = None,
) -> None:
    """Plot pull seeds, vertical equilibration rays, and normalized densities."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if style_path is not None and style_path.is_file():
        plt.style.use(style_path)

    grouped: dict[int, list[LCODEquilSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.window_index, []).append(sample)

    cmap = plt.get_cmap("turbo")
    colors = {index: cmap(index / max(1, len(centers) - 1))
              for index in range(len(centers))}
    fig, (density_ax, equil_ax, pull_ax) = plt.subplots(
        3, 1, figsize=(9.0, 12.2), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 1.55, 0.95)},
    )
    for density in density_traces:
        color = colors[density.window_index]
        density_ax.scatter(
            density.histogram_centers_angstrom,
            density.histogram_density_per_angstrom,
            color=color, s=7, alpha=0.58, linewidths=0, zorder=2,
        )
        density_ax.plot(
            density.gaussian_x_angstrom,
            density.gaussian_density_per_angstrom,
            color=color,
            linewidth=1.2, alpha=0.9,
        )
    for index, trace in sorted(grouped.items()):
        ordered = sorted(trace, key=lambda sample: (sample.time_ps, sample.step))
        equil_ax.plot(
            [sample.lcod_angstrom for sample in ordered],
            [sample.time_ps for sample in ordered],
            color=colors[index], linewidth=0.95, alpha=0.87,
        )
    for seed in seeds:
        color = colors[seed.window_index]
        pull_ax.plot(
            [seed.target_angstrom, seed.selected_angstrom],
            [seed.window_index, seed.window_index],
            color=color, linewidth=1.4,
        )
        pull_ax.scatter(
            [seed.selected_angstrom], [seed.window_index],
            color=[color], s=12, zorder=3,
        )
    for axis in (density_ax, equil_ax, pull_ax):
        for index, center in enumerate(centers):
            axis.axvline(
                center, color=colors[index], linewidth=0.68,
                linestyle="--", alpha=0.38, zorder=0,
            )
        axis.grid(False)
    density_ax.set(
        ylabel=r"$P(\mathrm{LCOD})$ (Å$^{-1}$)",
        title=f"CPP LCOD umbrella equilibration ({len(grouped)}/{len(centers)} windows with data)",
    )
    density_ax.set_ylim(bottom=0.0)
    equil_ax.set(
        ylabel="Equilibration time (ps)",
        ylim=(0.0, expected_time_ps * 1.01),
    )
    pull_ax.set(
        ylabel="Seed window index",
        xlabel=r"LCOD = $r(\mathrm{N_B-H}) - r(\mathrm{N_C-H})$ (Å)",
        ylim=(-1.0, len(centers)),
    )
    pull_ax.set_yticks(np.arange(0, len(centers), 5))
    pull_ax.set_xlim(
        min(centers[0], *(seed.selected_angstrom for seed in seeds)) - 0.13,
        max(centers[-1], *(seed.selected_angstrom for seed in seeds)) + 0.13,
    )
    pull_ax.text(
        0.01, 0.99, "Handpicked source frames; no pull-MD time series",
        transform=pull_ax.transAxes, ha="left", va="top", fontsize=8.5,
    )
    density_ax.text(
        0.99, 0.98, "Points: normalized histogram\nLines: Gaussian guides",
        transform=density_ax.transAxes, ha="right", va="top", fontsize=8.5,
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)


def create_lcod_equil_report(
    cfg: LCODEquilConfig, yaml_text: str, repo_root: Path
) -> tuple[Path, Path, int, list[int]]:
    samples, missing = collect_lcod_equil_samples(cfg, yaml_text, repo_root)
    destination = pull_root(cfg.pull, repo_root)
    centers = lcod_centers(cfg.pull.windows)
    seeds = read_pull_seeds(destination, centers, cfg.pull_yaml.read_text())
    density_traces = build_density_traces(samples)
    csv_path = destination / "equil_lcod.csv"
    density_path = destination / "equil_density.csv"
    figure_path = destination / "equil-report.png"
    write_lcod_samples_csv(csv_path, samples)
    write_density_csv(density_path, density_traces)
    expected_time_ps = equilibration_duration_ps(cfg.dftb.header_lines)
    plot_lcod_samples(
        figure_path,
        samples,
        seeds,
        density_traces,
        centers,
        expected_time_ps,
        repo_root / "plotting" / "lefteris.mplstyle",
    )
    (destination / "equil_report_spec.yaml").write_text(yaml_text)
    return figure_path, csv_path, len(samples), missing


def run_us_lcod_equil_report(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    yaml_text = resolved.read_text()
    cfg = load_equil_config(resolved)
    figure, data, count, missing = create_lcod_equil_report(
        cfg, yaml_text, find_repo_root(resolved)
    )
    print(f"OK: wrote {count} trajectory-derived LCOD equilibration samples to {data}")
    print(f"OK: wrote normalized LCOD densities to {data.with_name('equil_density.csv')}")
    print(f"OK: wrote LCOD equilibration progress plot to {figure}")
    if missing:
        names = ", ".join(f"window-{index:03d}" for index in missing)
        print(f"NOTE: no usable samples yet for {names}")
