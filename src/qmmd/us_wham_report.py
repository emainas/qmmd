"""Compose pull, production-density, and smooth-PMF umbrella report panels."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from qmmd.us_prod_report import build_pull_target_ladder, read_pull_report_samples
from qmmd.us_pull import find_repo_root, pull_dir, window_centers
from qmmd.us_wham import load_config


def read_density_rows(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "window_index",
            "target_deg",
            "series",
            "dihedral_deg",
            "probability_density_per_deg",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing production-density columns in {path}")
        for row in reader:
            rows.append(
                {
                    "window_index": int(row["window_index"]),
                    "target_deg": float(row["target_deg"]),
                    "series": row["series"],
                    "dihedral_deg": float(row["dihedral_deg"]),
                    "density": float(row["probability_density_per_deg"]),
                }
            )
    if not rows:
        raise RuntimeError(f"No production-density data found in {path}")
    return rows


def read_smooth_pmf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"dihedral_deg", "smoothed_pmf_kcal_mol"}.issubset(rows[0]):
        raise ValueError(f"Missing smooth-PMF columns in {path}")
    grid = np.asarray([float(row["dihedral_deg"]) for row in rows])
    pmf = np.asarray([float(row["smoothed_pmf_kcal_mol"]) for row in rows])
    if np.any(np.diff(grid) <= 0) or not np.all(np.isfinite(pmf)):
        raise ValueError(f"Smooth PMF is not finite on an increasing grid: {path}")
    return grid, pmf


def read_thermodynamics(path: Path) -> dict[str, float]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = {row["metric"]: float(row["original"]) for row in rows}
    required = {
        "syn_min_dihedral",
        "syn_min_pmf",
        "anti_min_dihedral",
        "anti_min_pmf",
        "transition_max_dihedral",
        "transition_max_pmf",
        "delta_pmf_anti_minus_syn",
        "barrier_syn_to_anti",
    }
    if not required.issubset(values):
        raise ValueError(f"Missing thermodynamic metrics in {path}")
    return values


def nearest_window_indices(grid: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each coordinate to its nearest umbrella center."""
    return np.argmin(np.abs(grid[:, None] - centers[None, :]), axis=1)


def write_report_data(
    path: Path,
    pull_samples: list[object],
    density_rows: list[dict[str, float | int | str]],
    pmf_grid: np.ndarray,
    pmf: np.ndarray,
    pmf_windows: np.ndarray,
    centers: np.ndarray,
    thermodynamics: dict[str, float],
) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["panel", "series", "window_index", "target_deg", "x", "y"])
        for sample in pull_samples:
            writer.writerow(
                [
                    "pull",
                    "measured",
                    sample.window_index,
                    f"{sample.target_deg:.8f}",
                    f"{sample.dihedral_branch_deg:.8f}",
                    f"{sample.time_ps:.8f}",
                ]
            )
        for row in density_rows:
            writer.writerow(
                [
                    "density",
                    row["series"],
                    row["window_index"],
                    f"{float(row['target_deg']):.8f}",
                    f"{float(row['dihedral_deg']):.8f}",
                    f"{float(row['density']):.12g}",
                ]
            )
        for coordinate, value, window_index in zip(pmf_grid, pmf, pmf_windows):
            writer.writerow(
                [
                    "pmf",
                    "penalized_smooth",
                    int(window_index),
                    f"{centers[window_index]:.8f}",
                    f"{coordinate:.8f}",
                    f"{value:.10g}",
                ]
            )
        for metric in (
            "syn_min_pmf",
            "anti_min_pmf",
            "transition_max_pmf",
            "delta_pmf_anti_minus_syn",
            "barrier_syn_to_anti",
        ):
            writer.writerow(["annotation", metric, "", "", "", f"{thermodynamics[metric]:.10g}"])


def plot_wham_report(
    path: Path,
    pull_samples: list[object],
    density_rows: list[dict[str, float | int | str]],
    pmf_grid: np.ndarray,
    pmf: np.ndarray,
    centers: np.ndarray,
    system: str,
    thermodynamics: dict[str, float],
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    style = Path(__file__).resolve().parents[2] / "plotting" / "lefteris.mplstyle"
    if not style.is_file():
        raise RuntimeError(f"Missing plotting style: {style}")
    plt.style.use(style)

    cmap = plt.get_cmap("turbo")
    color_by_window = {
        index: cmap(index / max(1, len(centers) - 1)) for index in range(len(centers))
    }
    fig, (pmf_ax, density_ax, pull_ax) = plt.subplots(
        3,
        1,
        figsize=(8.2, 11.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.45, 1.0, 0.9)},
    )

    segment_windows = nearest_window_indices(
        0.5 * (pmf_grid[:-1] + pmf_grid[1:]), centers
    )
    segments = np.stack(
        (
            np.column_stack((pmf_grid[:-1], pmf[:-1])),
            np.column_stack((pmf_grid[1:], pmf[1:])),
        ),
        axis=1,
    )
    pmf_ax.add_collection(
        LineCollection(
            segments,
            colors=[color_by_window[int(index)] for index in segment_windows],
            linewidths=4.6,
            capstyle="round",
            joinstyle="round",
        )
    )
    pmf_ax.set_xlim(float(pmf_grid.min()), float(pmf_grid.max()))
    pmf_padding = max(0.25, 0.04 * float(np.ptp(pmf)))
    pmf_ax.set_ylim(float(pmf.min()) - pmf_padding, float(pmf.max()) + pmf_padding)

    extrema_x = [
        thermodynamics["syn_min_dihedral"],
        thermodynamics["transition_max_dihedral"],
        thermodynamics["anti_min_dihedral"],
    ]
    extrema_y = [
        thermodynamics["syn_min_pmf"],
        thermodynamics["transition_max_pmf"],
        thermodynamics["anti_min_pmf"],
    ]
    pmf_ax.scatter(
        extrema_x,
        extrema_y,
        s=34,
        facecolor="white",
        edgecolor="#202124",
        linewidth=1.1,
        zorder=5,
    )
    annotation_box = {
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.86,
    }
    arrow_style = {"arrowstyle": "<->", "color": "#202124", "linewidth": 1.5}
    barrier_x = 96.0
    pmf_ax.annotate(
        "",
        xy=(barrier_x, thermodynamics["transition_max_pmf"]),
        xytext=(barrier_x, thermodynamics["syn_min_pmf"]),
        arrowprops=arrow_style,
        zorder=4,
    )
    pmf_ax.text(
        barrier_x + 2.5,
        0.5 * (thermodynamics["transition_max_pmf"] + thermodynamics["syn_min_pmf"]),
        f"ΔG‡ syn→anti\n{thermodynamics['barrier_syn_to_anti']:.2f} kcal mol⁻¹",
        va="center",
        ha="left",
        fontsize=10,
        bbox=annotation_box,
        zorder=6,
    )
    gap_x = 174.0
    pmf_ax.annotate(
        "",
        xy=(gap_x, thermodynamics["anti_min_pmf"]),
        xytext=(gap_x, thermodynamics["syn_min_pmf"]),
        arrowprops=arrow_style,
        zorder=4,
    )
    pmf_ax.text(
        gap_x - 3.0,
        0.5 * (thermodynamics["anti_min_pmf"] + thermodynamics["syn_min_pmf"]),
        f"ΔG anti−syn\n{thermodynamics['delta_pmf_anti_minus_syn']:.2f} kcal mol⁻¹",
        va="center",
        ha="right",
        fontsize=10,
        bbox=annotation_box,
        zorder=6,
    )

    grouped_density: dict[tuple[int, str], list[dict[str, float | int | str]]] = {}
    for row in density_rows:
        grouped_density.setdefault((int(row["window_index"]), str(row["series"])), []).append(row)
    for window_index in range(len(centers)):
        hist = grouped_density.get((window_index, "histogram"), [])
        fit = grouped_density.get((window_index, "gaussian"), [])
        color = color_by_window[window_index]
        if hist:
            density_ax.scatter(
                [float(row["dihedral_deg"]) for row in hist],
                [float(row["density"]) for row in hist],
                color=color,
                s=9,
                alpha=0.48,
                edgecolors="none",
            )
        if fit:
            density_ax.plot(
                [float(row["dihedral_deg"]) for row in fit],
                [float(row["density"]) for row in fit],
                color=color,
                linewidth=1.7,
            )

    grouped_pull: dict[int, list[object]] = {}
    for sample in pull_samples:
        grouped_pull.setdefault(sample.window_index, []).append(sample)
    for window_index, samples in sorted(grouped_pull.items()):
        samples = sorted(samples, key=lambda sample: sample.time_ps)
        pull_ax.plot(
            [sample.dihedral_branch_deg for sample in samples],
            [sample.time_ps for sample in samples],
            color=color_by_window[window_index],
            linewidth=1.5,
            marker=".",
            markersize=3.0,
        )
    pull_targets, pull_times = build_pull_target_ladder(
        sorted(pull_samples, key=lambda sample: sample.time_ps)
    )
    pull_ax.plot(
        pull_targets,
        pull_times,
        color="#32343a",
        linewidth=1.1,
        linestyle="--",
        alpha=0.75,
        label="Pull target",
    )

    for axis in (pmf_ax, density_ax, pull_ax):
        for window_index, center in enumerate(centers):
            axis.axvline(
                center,
                color=color_by_window[window_index],
                linewidth=0.65,
                linestyle="--",
                alpha=0.20,
                zorder=0,
            )
        axis.grid(False)
    density_ax.set_ylim(bottom=0.0)
    pull_ax.set_ylim(bottom=0.0)
    pmf_ax.set_ylabel("PMF\n(kcal mol⁻¹)")
    density_ax.set_ylabel("$P(\\phi)$\n(degree$^{-1}$)")
    pull_ax.set_ylabel("Pull time\n(ps)")
    pull_ax.set_xlabel("Dihedral (degrees; branch nearest target)")
    pull_ax.legend(frameon=False, loc="upper right", fontsize=8)
    pmf_ax.set_title(f"{system} umbrella sampling: pull, production densities, and PMF")

    scalar = ScalarMappable(norm=Normalize(vmin=float(centers.min()), vmax=float(centers.max())), cmap="turbo_r")
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=[pmf_ax, density_ax, pull_ax], pad=0.012, fraction=0.025)
    colorbar.set_label("Umbrella center (degrees)")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return nearest_window_indices(pmf_grid, centers)


def create_us_wham_report(yaml_path: Path) -> tuple[Path, Path]:
    resolved = yaml_path.resolve()
    yaml_text = resolved.read_text()
    cfg = load_config(resolved)
    repo_root = find_repo_root(resolved)
    root = pull_dir(cfg.prod.run.pull, repo_root)
    output = root / cfg.output_dirname
    spec = output / "wham_spec.yaml"
    if not spec.is_file() or spec.read_text() != yaml_text:
        raise RuntimeError(f"{spec} does not exactly match {resolved}; run us-wham first")

    pull_path = root / "pull_dihedral.csv"
    density_path = root / "prod_density.csv"
    smooth_path = output / "pmf_smooth.csv"
    thermodynamics_path = output / "thermodynamics.csv"
    for required in (pull_path, density_path, smooth_path, thermodynamics_path):
        if not required.is_file() or required.stat().st_size == 0:
            raise RuntimeError(f"Missing/empty report input: {required}")

    pull_samples = read_pull_report_samples(pull_path)
    density_rows = read_density_rows(density_path)
    pmf_grid, pmf = read_smooth_pmf(smooth_path)
    thermodynamics = read_thermodynamics(thermodynamics_path)
    centers = np.asarray(window_centers(cfg.prod.run.pull.windows), dtype=float)
    figure = output / "wham-report.png"
    report_data = output / "wham-report-data.csv"
    pmf_windows = plot_wham_report(
        figure,
        pull_samples,
        density_rows,
        pmf_grid,
        pmf,
        centers,
        cfg.prod.run.pull.system,
        thermodynamics,
    )
    write_report_data(
        report_data,
        pull_samples,
        density_rows,
        pmf_grid,
        pmf,
        pmf_windows,
        centers,
        thermodynamics,
    )
    return figure, report_data


def run_us_wham_report(path: Path) -> None:
    figure, data = create_us_wham_report(path)
    print(f"OK: wrote WHAM summary plot to {figure}")
    print(f"OK: wrote WHAM summary plot data to {data}")
