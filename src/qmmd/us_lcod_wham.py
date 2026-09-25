"""WHAM analysis of trajectory-derived CPP LCOD equilibration windows."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from qmmd.us_lcod import LCODEquilConfig, lcod_centers, load_equil_config, pull_root
from qmmd.us_lcod_equil_report import read_pull_seeds
from qmmd.us_pull import find_repo_root
from qmmd.us_wham import (
    dcdftb_wall_to_wham_force,
    read_wham_result,
    resolve_executable,
    run_wham,
    zero_pmf,
)


@dataclass(frozen=True, slots=True)
class LCODWhamConfig:
    equil_yaml: Path
    equil: LCODEquilConfig
    output_dirname: str
    discard_ps: float
    minimum_angstrom: float
    maximum_angstrom: float
    bins: int
    report_minimum_angstrom: float
    report_maximum_angstrom: float
    zero_reference_angstrom: tuple[float, float]
    executable: str
    tolerance: float
    temperature_k: float
    padding: int
    smoothing_penalty: float
    plot_xlim_angstrom: tuple[float, float]
    negative_basin_angstrom: tuple[float, float]
    positive_basin_angstrom: tuple[float, float]


def load_config(path: Path) -> LCODWhamConfig:
    resolved = path.resolve()
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("LCOD WHAM config must be a YAML mapping")
    equil_value = Path(str(data.get("equil_yaml", "equil.yaml")))
    equil_yaml = (resolved.parent / equil_value).resolve() if not equil_value.is_absolute() else equil_value
    equil = load_equil_config(equil_yaml)
    coordinate = data["coordinate"]
    runtime = data["wham"]
    plot = data.get("plot", {})
    basins = data["basins"]
    reference = data["zero_reference_angstrom"]
    def interval(name: str) -> tuple[float, float]:
        value = basins[name]
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"basins.{name} must contain two Å bounds")
        return float(value[0]), float(value[1])

    cfg = LCODWhamConfig(
        equil_yaml=equil_yaml,
        equil=equil,
        output_dirname=str(data.get("output_dirname", "wham-equil")),
        discard_ps=float(data.get("discard_ps", 0.0)),
        minimum_angstrom=float(coordinate["minimum_angstrom"]),
        maximum_angstrom=float(coordinate["maximum_angstrom"]),
        bins=int(coordinate["bins"]),
        report_minimum_angstrom=float(coordinate["report_minimum_angstrom"]),
        report_maximum_angstrom=float(coordinate["report_maximum_angstrom"]),
        zero_reference_angstrom=(float(reference[0]), float(reference[1])),
        executable=str(runtime.get("executable", "wham")),
        tolerance=float(runtime.get("tolerance", 1e-6)),
        temperature_k=float(runtime.get("temperature_k", 300.0)),
        padding=int(runtime.get("padding", 0)),
        smoothing_penalty=float(plot.get("smoothing_penalty", 4.0)),
        plot_xlim_angstrom=tuple(float(value) for value in plot.get(
            "xlim_angstrom", [coordinate["report_minimum_angstrom"], coordinate["report_maximum_angstrom"]]
        )),
        negative_basin_angstrom=interval("negative_lcod_angstrom"),
        positive_basin_angstrom=interval("positive_lcod_angstrom"),
    )
    if Path(cfg.output_dirname).name != cfg.output_dirname or cfg.output_dirname in {".", ".."}:
        raise ValueError("output_dirname must be a single directory name")
    if not (cfg.minimum_angstrom < cfg.report_minimum_angstrom < cfg.report_maximum_angstrom < cfg.maximum_angstrom):
        raise ValueError("WHAM coordinate/report bounds are invalid")
    if cfg.bins < 3 or cfg.discard_ps < 0 or cfg.tolerance <= 0 or cfg.temperature_k <= 0 or cfg.padding < 0:
        raise ValueError("WHAM bins, discard, tolerance, temperature, or padding are invalid")
    if not cfg.minimum_angstrom <= cfg.zero_reference_angstrom[0] < cfg.zero_reference_angstrom[1] <= cfg.maximum_angstrom:
        raise ValueError("zero_reference_angstrom must lie inside WHAM grid")
    if cfg.smoothing_penalty < 0 or cfg.equil.wall.exponent != 2:
        raise ValueError("Smoothing penalty must be nonnegative and METAWALL quadratic")
    if len(cfg.plot_xlim_angstrom) != 2 or not (
        cfg.report_minimum_angstrom <= cfg.plot_xlim_angstrom[0]
        < cfg.plot_xlim_angstrom[1] <= cfg.report_maximum_angstrom
    ):
        raise ValueError("plot.xlim_angstrom must lie inside the WHAM report range")
    if not (cfg.report_minimum_angstrom <= cfg.negative_basin_angstrom[0]
            < cfg.negative_basin_angstrom[1] < 0 < cfg.positive_basin_angstrom[0]
            < cfg.positive_basin_angstrom[1] <= cfg.report_maximum_angstrom):
        raise ValueError("The negative and positive LCOD basin ranges must be ordered within the report")
    return cfg


def read_samples(path: Path, centers: list[float], discard_ps: float) -> dict[int, np.ndarray]:
    grouped: dict[int, list[tuple[float, float]]] = {index: [] for index in range(len(centers))}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"window_index", "target_lcod_A", "time_ps", "lcod_A"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing LCOD report columns in {path}")
        for row in reader:
            index = int(row["window_index"])
            if index not in grouped or not math.isclose(float(row["target_lcod_A"]), centers[index], abs_tol=1e-7):
                raise ValueError(f"LCOD sample has an unknown/inconsistent center: {row}")
            time, value = float(row["time_ps"]), float(row["lcod_A"])
            if not math.isfinite(time) or not math.isfinite(value):
                raise ValueError(f"Non-finite LCOD sample: {row}")
            if time >= discard_ps:
                grouped[index].append((time, value))
    result = {index: np.asarray(values, dtype=float) for index, values in grouped.items()}
    for index, samples in result.items():
        if len(samples) < 2 or np.any(np.diff(samples[:, 0]) <= 0):
            raise ValueError(f"window-{index:03d} lacks two time-ordered LCOD samples")
    return result


def write_wham_inputs(output: Path, centers: list[float], samples: dict[int, np.ndarray], force: float) -> Path:
    metadata = output / "metadata.dat"
    with metadata.open("w") as stream:
        for index, center in enumerate(centers):
            filename = f"window-{index:03d}.dat"
            # Grossfield WHAM reads the CV from column 2, after a frame/time column.
            np.savetxt(output / filename, samples[index], fmt="%.8f %.10f", header="time_ps LCOD_A")
            stream.write(f"{filename} {center:.8f} {force:.12g}\n")
    with (output / "window_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window_index", "center_A", "samples", "start_ps", "end_ps", "mean_A", "std_A"])
        for index, center in enumerate(centers):
            values = samples[index]
            writer.writerow([index, f"{center:.8f}", len(values), f"{values[0, 0]:.8f}",
                             f"{values[-1, 0]:.8f}", f"{np.mean(values[:, 1]):.8f}",
                             f"{np.std(values[:, 1], ddof=1):.8f}"])
    return metadata


def calculate_wham(cfg: LCODWhamConfig, yaml_text: str, repo_root: Path) -> tuple[Path, Path]:
    root = pull_root(cfg.equil.pull, repo_root)
    report_spec = root / "equil_report_spec.yaml"
    if not report_spec.is_file() or report_spec.read_text() != cfg.equil_yaml.read_text():
        raise RuntimeError("Equilibration report snapshot differs from equil.yaml; regenerate the cpptraj report")
    centers = lcod_centers(cfg.equil.pull.windows)
    samples = read_samples(root / "equil_lcod.csv", centers, cfg.discard_ps)
    output = root / cfg.output_dirname
    output.mkdir(parents=True, exist_ok=True)
    # DCD's V=kappa*(s-s0)^2 is kJ/mol; Grossfield expects half-K in kcal/mol.
    force = dcdftb_wall_to_wham_force(cfg.equil.wall.coefficient_kcal_mol_angstrom2)
    metadata = write_wham_inputs(output, centers, samples, force)
    result_path = output / "wham_result.dat"
    command = [str(resolve_executable(cfg.executable)), f"{cfg.minimum_angstrom:g}",
               f"{cfg.maximum_angstrom:g}", str(cfg.bins), f"{cfg.tolerance:g}",
               f"{cfg.temperature_k:g}", str(cfg.padding), str(metadata), str(result_path)]
    (output / "wham_command.txt").write_text(" ".join(command) + "\n")
    run_wham(command, output, output / "wham.log")
    grid, raw = read_wham_result(result_path)
    mask = (grid >= cfg.report_minimum_angstrom) & (grid <= cfg.report_maximum_angstrom)
    if not np.any(mask) or not np.all(np.isfinite(raw[mask])):
        raise ValueError("WHAM PMF contains empty/non-finite bins inside report bounds")
    pmf = zero_pmf(grid, raw, cfg.zero_reference_angstrom)
    table = output / "pmf.csv"
    with table.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["lcod_A", "pmf_kcal_mol", "in_report_range"])
        writer.writerows((f"{x:.8f}", f"{y:.10g}" if math.isfinite(y) else "nan", int(keep))
                         for x, y, keep in zip(grid, pmf, mask))
    (output / "wham_spec.yaml").write_text(yaml_text)
    return table, output


def pmf_features(
    coordinates: np.ndarray,
    energies: np.ndarray,
    negative_basin: tuple[float, float],
    positive_basin: tuple[float, float],
) -> dict[str, tuple[float, float] | float]:
    """Locate smooth-curve basin minima and intervening maximum in Å/kcal mol⁻¹."""
    if coordinates.ndim != 1 or energies.shape != coordinates.shape or len(coordinates) < 3:
        raise ValueError("PMF feature arrays must be aligned one-dimensional vectors")
    if np.any(np.diff(coordinates) <= 0) or not np.all(np.isfinite(energies)):
        raise ValueError("PMF coordinates/energies must increase and be finite")
    negative = np.where((coordinates >= negative_basin[0]) & (coordinates <= negative_basin[1]))[0]
    positive = np.where((coordinates >= positive_basin[0]) & (coordinates <= positive_basin[1]))[0]
    if not len(negative) or not len(positive):
        raise ValueError("No smooth PMF points in one of the basin ranges")
    left = int(negative[np.argmin(energies[negative])])
    right = int(positive[np.argmin(energies[positive])])
    if left >= right:
        raise ValueError("Negative basin must precede positive basin")
    top = left + int(np.argmax(energies[left:right + 1]))
    return {
        "negative_minimum": (float(coordinates[left]), float(energies[left])),
        "positive_minimum": (float(coordinates[right]), float(energies[right])),
        "barrier_maximum": (float(coordinates[top]), float(energies[top])),
        "gap_positive_minus_negative": float(energies[right] - energies[left]),
        "barrier_negative_to_positive": float(energies[top] - energies[left]),
        "barrier_positive_to_negative": float(energies[top] - energies[right]),
    }


def plot_pmf_zero(
    raw_x: np.ndarray, raw_y: np.ndarray, smooth_x: np.ndarray,
    smooth_y: np.ndarray, limits: tuple[float, float]
) -> float:
    """Return the lowest value displayed in the PMF panel, in kcal/mol."""
    raw_mask = (raw_x >= limits[0]) & (raw_x <= limits[1])
    smooth_mask = (smooth_x >= limits[0]) & (smooth_x <= limits[1])
    if not np.any(raw_mask) or not np.any(smooth_mask):
        raise ValueError("PMF plot limits contain no raw or smooth points")
    return float(min(np.min(raw_y[raw_mask]), np.min(smooth_y[smooth_mask])))


def plot_report(cfg: LCODWhamConfig, output: Path, repo_root: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from qmmd.us_wham import dense_hermite_curve, smooth_pmf

    style = repo_root / "plotting" / "lefteris.mplstyle"
    if style.is_file():
        plt.style.use(style)
    root = pull_root(cfg.equil.pull, repo_root)
    centers = lcod_centers(cfg.equil.pull.windows)
    samples = read_samples(root / "equil_lcod.csv", centers, cfg.discard_ps)
    seeds = read_pull_seeds(root, centers, cfg.equil.pull_yaml.read_text())
    with (output / "pmf.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grid = np.asarray([float(row["lcod_A"]) for row in rows if int(row["in_report_range"])])
    pmf = np.asarray([float(row["pmf_kcal_mol"]) for row in rows if int(row["in_report_range"])])
    if len(grid) < 3 or not np.all(np.isfinite(pmf)):
        raise ValueError("Reported WHAM PMF is incomplete")
    smooth_x, smooth_y = dense_hermite_curve(grid, smooth_pmf(pmf, cfg.smoothing_penalty))
    features = pmf_features(smooth_x, smooth_y, cfg.negative_basin_angstrom,
                            cfg.positive_basin_angstrom)
    zero = plot_pmf_zero(grid, pmf, smooth_x, smooth_y, cfg.plot_xlim_angstrom)
    plotted_raw = pmf - zero
    plotted_smooth = smooth_y - zero
    with (output / "pmf_smooth.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["lcod_A", "smoothed_pmf_kcal_mol"])
        writer.writerows((f"{x:.8f}", f"{y:.10g}") for x, y in zip(smooth_x, smooth_y))
    with (output / "pmf_features.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["feature", "lcod_A", "pmf_or_difference_kcal_mol"])
        for name, value in features.items():
            if isinstance(value, tuple):
                writer.writerow([name, f"{value[0]:.8f}", f"{value[1]:.8f}"])
            else:
                writer.writerow([name, "", f"{value:.8f}"])
    with (output / "pmf_plot.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["series", "lcod_A", "plotted_pmf_kcal_mol", "subtracted_zero_kcal_mol"])
        writer.writerows(("raw_wham", f"{x:.8f}", f"{y:.10g}", f"{zero:.10g}")
                         for x, y in zip(grid, plotted_raw))
        writer.writerows(("penalized_smooth", f"{x:.8f}", f"{y:.10g}", f"{zero:.10g}")
                         for x, y in zip(smooth_x, plotted_smooth))
    cmap = plt.get_cmap("turbo")
    colors = [cmap(i / (len(centers) - 1)) for i in range(len(centers))]
    fig, (pmf_ax, density_ax, equil_ax, seed_ax) = plt.subplots(
        4, 1, figsize=(9.0, 14.0), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": (1.25, 1.0, 1.55, 0.8)},
    )
    nearest = np.argmin(np.abs(smooth_x[:, None] - np.asarray(centers)[None, :]), axis=1)
    for index in range(len(centers)):
        chosen = nearest == index
        pmf_ax.plot(np.where(chosen, smooth_x, np.nan), np.where(chosen, plotted_smooth, np.nan),
                    color=colors[index], linewidth=2.4)
        values = samples[index]
        histogram, edges = np.histogram(values[:, 1], bins=24, density=True)
        density_ax.plot(0.5 * (edges[1:] + edges[:-1]), histogram,
                        color=colors[index], linewidth=1.0, alpha=0.8)
        equil_ax.plot(values[:, 1], values[:, 0], color=colors[index], linewidth=0.75, alpha=0.83)
    pmf_ax.scatter(grid, plotted_raw, s=5, color="0.25", alpha=0.55, zorder=3)
    for name in ("negative_minimum", "positive_minimum", "barrier_maximum"):
        coordinate, energy = features[name]
        pmf_ax.scatter([coordinate], [energy - zero], s=35,
                       facecolor="white", edgecolor="#202124", linewidth=1.1, zorder=5)
    for seed in seeds:
        color = colors[seed.window_index]
        seed_ax.plot([seed.target_angstrom, seed.selected_angstrom],
                     [seed.window_index] * 2, color=color, linewidth=1.4)
        seed_ax.scatter([seed.selected_angstrom], [seed.window_index], color=[color], s=10)
    for axis in (pmf_ax, density_ax, equil_ax, seed_ax):
        for index, center in enumerate(centers):
            axis.axvline(center, color=colors[index], linestyle="--", linewidth=0.65, alpha=0.35, zorder=0)
        axis.grid(False)
    pmf_ax.set(ylabel="PMF (minimum = 0; kcal mol⁻¹)",
               title="CPP LCOD WHAM from equilibration trajectories (exploratory)")
    pmf_ax.set_ylim(bottom=0.0)
    density_ax.set(ylabel="P(LCOD) (Å⁻¹)", ylim=(0, None))
    equil_ax.set(ylabel="Equilibration time (ps)", ylim=(0, 40))
    seed_ax.set(ylabel="Seed window", xlabel="LCOD = r(NB–H) − r(NC–H) (Å)", ylim=(-1, len(centers)))
    seed_ax.set_yticks(np.arange(0, len(centers), 5))
    seed_ax.set_xlim(*cfg.plot_xlim_angstrom)
    negative_x, negative_y = features["negative_minimum"]
    positive_x, positive_y = features["positive_minimum"]
    barrier_x, barrier_y = features["barrier_maximum"]
    annotation_box = {
        "boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.86,
    }
    pmf_ax.annotate(
        "", xy=(0.12, barrier_y - zero), xytext=(0.12, negative_y - zero),
        arrowprops={"arrowstyle": "<->", "color": "#202124", "linewidth": 1.35}, zorder=4,
    )
    pmf_ax.text(
        0.22, 0.5 * (barrier_y + negative_y) - zero,
        f"ΔG‡ −→+\n{features['barrier_negative_to_positive']:.2f} kcal mol⁻¹",
        ha="left", va="center", fontsize=10, bbox=annotation_box, zorder=6,
    )
    pmf_ax.text(
        0.02, 0.96,
        f"ΔG (+ vs −) = {features['gap_positive_minus_negative']:+.3f} kcal mol⁻¹\n"
        f"minima: {negative_x:+.2f}, {positive_x:+.2f} Å",
        transform=pmf_ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=annotation_box, zorder=6,
    )
    path = output / "wham-report.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def run_us_lcod_wham(path: Path) -> None:
    resolved = path.resolve()
    cfg = load_config(resolved)
    repo_root = find_repo_root(resolved)
    table, output = calculate_wham(cfg, resolved.read_text(), repo_root)
    figure = plot_report(cfg, output, repo_root)
    print(f"OK: wrote equilibration-derived LCOD WHAM PMF to {table}")
    print(f"OK: wrote four-tier exploratory WHAM report to {figure}")
