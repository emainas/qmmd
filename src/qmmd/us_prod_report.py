"""Report dihedral traces from restrained DCDFTBMD production windows."""

from __future__ import annotations

import csv
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from qmmd.dihedral import convert_dftb_xyz_to_cpptraj
from qmmd.us_equil_report import (
    EquilSample,
    collect_equil_samples,
    write_samples_csv,
)
from qmmd.us_prod import USProdConfig, load_config
from qmmd.us_pull import find_repo_root, pull_dir, source_paths, window_centers
from qmmd.us_pull_report import (
    _cpptraj_path,
    amber_residue_for_atoms,
    angle_near_target,
    read_cpptraj_dihedrals,
    run_cpptraj,
)


@dataclass(frozen=True, slots=True)
class DensityTrace:
    window_index: int
    target_deg: float
    histogram_edges_deg: np.ndarray
    histogram_density_per_deg: np.ndarray
    fit_mean_deg: float
    fit_sigma_deg: float
    gaussian_dihedral_deg: np.ndarray
    gaussian_density_per_deg: np.ndarray


def read_pull_report_samples(path: Path) -> list[EquilSample]:
    """Read the aligned local-window series written by ``us-pull-report``."""
    samples: list[EquilSample] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "window_index",
            "target_deg",
            "local_time_ps",
            "frame_in_window",
            "dihedral_raw_deg",
            "dihedral_branch_deg",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing pull-report columns in {path}")
        for row in reader:
            samples.append(
                EquilSample(
                    window_index=int(row["window_index"]),
                    target_deg=float(row["target_deg"]),
                    time_ps=float(row.get("time_ps") or row["local_time_ps"]),
                    step=int(row["frame_in_window"]),
                    dihedral_raw_deg=float(row["dihedral_raw_deg"]),
                    dihedral_branch_deg=float(row["dihedral_branch_deg"]),
                )
            )
    if not samples:
        raise RuntimeError(f"No pull samples found in {path}")
    return samples


_TRAJECT_TIME_RE = re.compile(
    r"AT T=\s*([+-]?[0-9.eEdD+-]+)\s+FSEC,\s*THIS RUN'S STEP NO\.=\s*(\d+)"
)


def read_dftb_trajectory_times(path: Path) -> list[tuple[int, float]]:
    """Return ``(step, time_ps)`` metadata for every complete DFTB XYZ frame."""
    records: list[tuple[int, float]] = []
    with path.open(errors="replace") as stream:
        while True:
            atom_count_line = stream.readline()
            if not atom_count_line:
                break
            if not atom_count_line.strip():
                continue
            try:
                atom_count = int(atom_count_line)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed DFTB trajectory atom count in {path}: "
                    f"{atom_count_line.strip()!r}"
                ) from exc
            comment = stream.readline()
            match = _TRAJECT_TIME_RE.search(comment)
            if match is None:
                raise ValueError(f"Missing DFTB time/step metadata in {path}: {comment.strip()}")
            complete = True
            for _ in range(atom_count):
                if not stream.readline():
                    complete = False
                    break
            if not complete:
                break
            time_fs = float(match.group(1).replace("D", "E").replace("d", "e"))
            records.append((int(match.group(2)), time_fs / 1000.0))
    if not records:
        raise ValueError(f"No complete coordinate frames found in {path}")
    return records


def render_prod_cpptraj_input(
    topology: Path,
    trajectory: Path,
    atoms: tuple[int, int, int, int],
    data_path: Path,
    solute_residue: int,
) -> str:
    """Render the cpptraj calculation used for a production window."""
    return "\n".join(
        [
            f"parm {_cpptraj_path(topology)}",
            f"trajin {_cpptraj_path(trajectory)}",
            # Repair any solute bonds split by DFTB's wrapped XYZ coordinates
            # before centering/imaging the system.
            f"fiximagedbonds :{solute_residue}",
            "autoimage",
            "dihedral production "
            + " ".join(f"@{atom}" for atom in atoms)
            + f" out {_cpptraj_path(data_path)}",
            "run",
            "quit",
            "",
        ]
    )


def collect_prod_samples(
    cfg: USProdConfig, yaml_text: str, repo_root: Path
) -> tuple[list[EquilSample], list[int]]:
    """Calculate production torsions from coordinate trajectories with cpptraj."""
    root = pull_dir(cfg.run.pull, repo_root)
    topology, _, _ = source_paths(cfg.run.pull, repo_root)
    solute_residue, _ = amber_residue_for_atoms(
        topology, cfg.run.pull.restraint.atoms
    )
    samples: list[EquilSample] = []
    missing: list[int] = []

    for index, center in enumerate(window_centers(cfg.run.pull.windows)):
        stage = root / f"window-{index:03d}" / cfg.run.stage_dirname
        spec = stage / "prod_spec.yaml"
        if not spec.is_file() or spec.read_text() != yaml_text:
            missing.append(index)
            continue
        trajectory = stage / "traject"
        dftb_input = stage / "dftb.inp"
        if (
            not trajectory.is_file()
            or trajectory.stat().st_size == 0
            or not dftb_input.is_file()
        ):
            missing.append(index)
            continue
        frame_records = read_dftb_trajectory_times(trajectory)
        dihedral_data = stage / "dihedral.dat"
        cpptraj_input = stage / "dihedral.cpptraj.in"
        cpptraj_log = stage / "dihedral.cpptraj.log"
        with tempfile.TemporaryDirectory(prefix=f"qmmd-us-prod-{index:03d}-") as tmp:
            converted = Path(tmp) / "traject.cpptraj.xyz"
            convert_dftb_xyz_to_cpptraj(trajectory, converted, dftb_input)
            cpptraj_input.write_text(
                render_prod_cpptraj_input(
                    topology,
                    converted,
                    cfg.run.pull.restraint.atoms,
                    dihedral_data,
                    solute_residue,
                )
            )
            run_cpptraj(cpptraj_input, cpptraj_log, cfg.run.pull.runtime.module)
        angles = read_cpptraj_dihedrals(dihedral_data)
        if len(angles) != len(frame_records):
            raise ValueError(
                f"cpptraj wrote {len(angles)} angles for {len(frame_records)} complete "
                f"frames in {trajectory}"
            )
        start_time_ps = frame_records[0][1]
        samples.extend(
            EquilSample(
                window_index=index,
                target_deg=center,
                time_ps=time_ps - start_time_ps,
                step=step,
                dihedral_raw_deg=angle,
                dihedral_branch_deg=angle_near_target(angle, center),
            )
            for (step, time_ps), angle in zip(frame_records, angles)
        )

    if not samples:
        raise RuntimeError(f"No complete production trajectories found under {root}")
    return samples, missing


def build_density_traces(samples: list[EquilSample]) -> list[DensityTrace]:
    """Build normalized histograms and fitted normal densities per window."""
    grouped: dict[int, list[EquilSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.window_index, []).append(sample)

    traces: list[DensityTrace] = []
    for window_index, window_samples in sorted(grouped.items()):
        values = np.asarray(
            [sample.dihedral_branch_deg for sample in window_samples], dtype=float
        )
        bins = 50
        histogram, edges = np.histogram(values, bins=bins, density=True)
        fit_mean = float(np.mean(values))
        fit_sigma = float(np.std(values, ddof=0))
        if not math.isfinite(fit_sigma) or fit_sigma <= 1.0e-12:
            raise ValueError(f"Cannot fit Gaussian for window {window_index}")
        gaussian_x = np.linspace(
            float(values.min()) - 4.0 * fit_sigma,
            float(values.max()) + 4.0 * fit_sigma,
            240,
        )
        gaussian_y = np.exp(-0.5 * ((gaussian_x - fit_mean) / fit_sigma) ** 2)
        gaussian_y /= fit_sigma * math.sqrt(2.0 * math.pi)
        traces.append(
            DensityTrace(
                window_index=window_index,
                target_deg=window_samples[0].target_deg,
                histogram_edges_deg=edges,
                histogram_density_per_deg=histogram,
                fit_mean_deg=fit_mean,
                fit_sigma_deg=fit_sigma,
                gaussian_dihedral_deg=gaussian_x,
                gaussian_density_per_deg=gaussian_y,
            )
        )
    return traces


def write_density_csv(path: Path, traces: list[DensityTrace]) -> None:
    """Write every histogram and fitted-Gaussian point used in the plot."""
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "window_index",
                "target_deg",
                "series",
                "dihedral_deg",
                "bin_left_deg",
                "bin_right_deg",
                "probability_density_per_deg",
                "fit_mean_deg",
                "fit_sigma_deg",
            ]
        )
        for trace in traces:
            centers = 0.5 * (
                trace.histogram_edges_deg[:-1] + trace.histogram_edges_deg[1:]
            )
            for coordinate, left, right, density in zip(
                centers,
                trace.histogram_edges_deg[:-1],
                trace.histogram_edges_deg[1:],
                trace.histogram_density_per_deg,
            ):
                writer.writerow(
                    [
                        trace.window_index,
                        f"{trace.target_deg:.8f}",
                        "histogram",
                        f"{coordinate:.8f}",
                        f"{left:.8f}",
                        f"{right:.8f}",
                        f"{density:.12g}",
                        f"{trace.fit_mean_deg:.8f}",
                        f"{trace.fit_sigma_deg:.8f}",
                    ]
                )
            for coordinate, density in zip(
                trace.gaussian_dihedral_deg, trace.gaussian_density_per_deg
            ):
                writer.writerow(
                    [
                        trace.window_index,
                        f"{trace.target_deg:.8f}",
                        "gaussian",
                        f"{coordinate:.8f}",
                        "",
                        "",
                        f"{density:.12g}",
                        f"{trace.fit_mean_deg:.8f}",
                        f"{trace.fit_sigma_deg:.8f}",
                    ]
                )


def build_pull_target_ladder(
    samples: list[EquilSample],
) -> tuple[np.ndarray, np.ndarray]:
    """Build an exact target staircase against stitched pull time."""
    grouped: dict[int, list[EquilSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.window_index, []).append(sample)
    target_points: list[float] = []
    time_points: list[float] = []
    previous_end = 0.0
    for position, window_index in enumerate(sorted(grouped)):
        window_samples = sorted(grouped[window_index], key=lambda item: item.time_ps)
        target = window_samples[0].target_deg
        end = window_samples[-1].time_ps
        start = previous_end
        target_points.extend((target, target))
        time_points.extend((start, end))
        if position + 1 < len(grouped):
            next_target = grouped[window_index + 1][0].target_deg
            target_points.append(next_target)
            time_points.append(end)
        previous_end = end
    return np.asarray(target_points), np.asarray(time_points)


def plot_prod_samples(
    path: Path,
    pull_samples: list[EquilSample],
    equil_samples: list[EquilSample],
    prod_samples: list[EquilSample],
    density_traces: list[DensityTrace],
    atoms: tuple[int, int, int, int],
    total_windows: int,
    pull_spring_kcal_mol_rad2: float,
    equil_spring_kcal_mol_deg2: float,
    prod_spring_kcal_mol_deg2: float,
) -> None:
    """Plot pull, equilibration, production, and density on one shared CV axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def group(samples: list[EquilSample]) -> dict[int, list[EquilSample]]:
        grouped_samples: dict[int, list[EquilSample]] = {}
        for sample in samples:
            grouped_samples.setdefault(sample.window_index, []).append(sample)
        return grouped_samples

    pull_grouped = group(pull_samples)
    equil_grouped = group(equil_samples)
    prod_grouped = group(prod_samples)
    empirical_mean_by_window = {
        trace.window_index: trace.fit_mean_deg for trace in density_traces
    }

    colors = plt.get_cmap("turbo")
    color_by_window = {
        window_index: colors(color_index / max(1, total_windows - 1))
        for color_index, window_index in enumerate(range(total_windows))
    }
    fig, (density_ax, prod_ax, equil_ax, pull_ax) = plt.subplots(
        4,
        1,
        figsize=(12, 11),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 1.7, 1.7, 0.9)},
    )

    for trace in density_traces:
        color = color_by_window[trace.window_index]
        histogram_centers = 0.5 * (
            trace.histogram_edges_deg[:-1] + trace.histogram_edges_deg[1:]
        )
        density_ax.scatter(
            histogram_centers,
            trace.histogram_density_per_deg,
            color=color,
            s=11,
            alpha=0.68,
            edgecolors="none",
        )
        density_ax.plot(
            trace.gaussian_dihedral_deg,
            trace.gaussian_density_per_deg,
            color=color,
            linewidth=1.8,
        )

    def plot_stage(
        axis: object,
        grouped_samples: dict[int, list[EquilSample]],
        *,
        add_labels: bool = False,
    ) -> None:
        for window_index, window_samples in sorted(grouped_samples.items()):
            target = window_samples[0].target_deg
            axis.plot(
                [sample.dihedral_branch_deg for sample in window_samples],
                [sample.time_ps for sample in window_samples],
                color=color_by_window[window_index],
                linewidth=0.95,
                marker=".",
                markersize=2.3,
                label=(
                    f"w{window_index:03d}: {target:g}°, "
                    f"⟨φ⟩={empirical_mean_by_window[window_index]:.2f}°"
                    if add_labels
                    else None
                ),
            )

    plot_stage(equil_ax, equil_grouped)
    plot_stage(prod_ax, prod_grouped, add_labels=True)
    ordered_pull = sorted(pull_samples, key=lambda sample: sample.time_ps)
    for window_index, window_samples in sorted(pull_grouped.items()):
        window_samples = sorted(window_samples, key=lambda sample: sample.time_ps)
        pull_ax.plot(
            [sample.dihedral_branch_deg for sample in window_samples],
            [sample.time_ps for sample in window_samples],
            color=color_by_window[window_index],
            linewidth=1.5,
            marker=".",
            markersize=3.0,
            label="Measured pull (window colors)" if window_index == 0 else None,
        )
    pull_targets, pull_times = build_pull_target_ladder(ordered_pull)
    pull_ax.plot(
        pull_targets,
        pull_times,
        color="#d94b5b",
        linewidth=1.2,
        linestyle="--",
        label="Pull target",
    )

    centers = window_centers_from_samples(prod_samples)
    for axis in (density_ax, prod_ax, equil_ax, pull_ax):
        for window_index, center in enumerate(centers):
            axis.axvline(
                center,
                color=color_by_window[window_index],
                linewidth=0.75,
                linestyle="--",
                alpha=0.38,
                zorder=0,
            )

    all_angles = np.asarray(
        [
            sample.dihedral_branch_deg
            for sample in (*pull_samples, *equil_samples, *prod_samples)
        ],
        dtype=float,
    )
    padding = max(5.0, 0.025 * float(np.ptp(all_angles)))
    pull_ax.set_xlim(float(all_angles.min()) - padding, float(all_angles.max()) + padding)
    pull_ax.set_ylim(bottom=0.0)
    equil_ax.set_ylim(bottom=0.0)
    prod_ax.set_ylim(bottom=0.0)
    density_ax.set_ylim(bottom=0.0)
    density_ax.set_ylabel(r"$P(\phi)$ (degree$^{-1}$)")
    density_ax.set_title(
        "Torsional umbrella sampling: "
        + "-".join(str(atom) for atom in atoms)
        + f" ({len(prod_grouped)}/{total_windows} windows with data)"
    )
    prod_ax.set_ylabel("Production\ntime (ps)")
    equil_ax.set_ylabel("Equilibration\ntime (ps)")
    pull_ax.set_ylabel("Pull\ntime (ps)")
    pull_ax.set_xlabel("Dihedral (degrees; branch nearest target)")
    for axis in (density_ax, prod_ax, equil_ax, pull_ax):
        axis.grid(alpha=0.18)
    legend_title = (
        "Spring constants\n"
        f"Pull: k = {pull_spring_kcal_mol_rad2:g} kcal mol⁻¹ rad⁻²\n"
        f"Equil: κ = {equil_spring_kcal_mol_deg2:.8g} kcal mol⁻¹ deg⁻²\n"
        f"Prod: κ = {prod_spring_kcal_mol_deg2:g} kcal mol⁻¹ deg⁻²"
    )
    prod_ax.legend(
        frameon=False,
        ncol=1,
        fontsize=7.5,
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        title=legend_title,
        title_fontsize=8,
    )
    pull_ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def window_centers_from_samples(samples: list[EquilSample]) -> list[float]:
    """Return ordered unique targets while checking window-index continuity."""
    targets: dict[int, float] = {}
    for sample in samples:
        previous = targets.setdefault(sample.window_index, sample.target_deg)
        if not math.isclose(previous, sample.target_deg, abs_tol=1.0e-9):
            raise ValueError(f"Target changed within window {sample.window_index}")
    expected = list(range(len(targets)))
    if sorted(targets) != expected:
        raise ValueError("Production window indices are not contiguous from zero")
    return [targets[index] for index in expected]


def create_us_prod_report(
    cfg: USProdConfig, yaml_text: str, repo_root: Path
) -> tuple[Path, Path, int, list[int]]:
    samples, missing = collect_prod_samples(cfg, yaml_text, repo_root)
    destination = pull_dir(cfg.run.pull, repo_root)
    csv_path = destination / "prod_dihedral.csv"
    density_path = destination / "prod_density.csv"
    figure_path = destination / "prod-report.png"
    write_samples_csv(csv_path, samples)
    density_traces = build_density_traces(samples)
    write_density_csv(density_path, density_traces)
    pull_csv = destination / "pull_dihedral.csv"
    pull_spec = destination / "report_spec.yaml"
    pull_yaml_text = cfg.run.pull_yaml.read_text()
    if not pull_spec.is_file() or pull_spec.read_text() != pull_yaml_text:
        raise RuntimeError(
            f"{pull_spec} does not match {cfg.run.pull_yaml}; run us-pull-report first"
        )
    pull_samples = read_pull_report_samples(pull_csv)
    equil_samples, _ = collect_equil_samples(
        cfg.equil, cfg.equil_yaml.read_text(), repo_root
    )
    plot_prod_samples(
        figure_path,
        pull_samples,
        equil_samples,
        samples,
        density_traces,
        cfg.run.pull.restraint.atoms,
        len(window_centers(cfg.run.pull.windows)),
        cfg.run.pull.restraint.force_constant,
        cfg.equil.wall.coefficient_kcal_mol_deg2,
        cfg.run.wall.coefficient_kcal_mol_deg2,
    )
    (destination / "prod_report_spec.yaml").write_text(yaml_text)
    return figure_path, csv_path, len(samples), missing


def run_us_prod_report(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    yaml_text = resolved.read_text()
    cfg = load_config(resolved)
    figure, data, count, missing = create_us_prod_report(
        cfg, yaml_text, find_repo_root(resolved)
    )
    print(f"OK: wrote {count} production dihedral samples to {data}")
    print(f"OK: wrote production density data to {data.with_name('prod_density.csv')}")
    print(f"OK: wrote US production plot to {figure}")
    if missing:
        names = ", ".join(f"window-{index:03d}" for index in missing)
        print(f"NOTE: no usable production samples for {names}")
