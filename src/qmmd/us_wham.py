"""Construct a one-dimensional umbrella-sampling PMF with WHAM."""

from __future__ import annotations

import csv
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from qmmd.us_prod import USProdConfig, load_config as load_prod_config
from qmmd.us_pull import find_repo_root, pull_dir, window_centers


KJ_PER_KCAL = 4.184


@dataclass(frozen=True, slots=True)
class CoordinateConfig:
    minimum_deg: float
    maximum_deg: float
    bins: int
    periodic: bool
    report_minimum_deg: float
    report_maximum_deg: float


@dataclass(frozen=True, slots=True)
class WhamRuntimeConfig:
    executable: str
    tolerance: float
    temperature_k: float
    padding: int


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    enabled: bool
    replicas: int
    block_length_ps: float
    seed: int
    confidence_level: float
    minimum_success_fraction: float


@dataclass(frozen=True, slots=True)
class ThermodynamicsConfig:
    syn_range_deg: tuple[float, float]
    anti_range_deg: tuple[float, float]
    transition_range_deg: tuple[float, float]


@dataclass(frozen=True, slots=True)
class PlotConfig:
    smoothing_penalty: float


@dataclass(frozen=True, slots=True)
class USWhamConfig:
    prod_yaml: Path
    prod: USProdConfig
    output_dirname: str
    discard_ps: float
    coordinate: CoordinateConfig
    wham: WhamRuntimeConfig
    bootstrap: BootstrapConfig
    zero_reference_deg: tuple[float, float]
    thermodynamics: ThermodynamicsConfig
    plot: PlotConfig


@dataclass(frozen=True, slots=True)
class WindowSeries:
    index: int
    center_deg: float
    times_ps: np.ndarray
    angles_deg: np.ndarray


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a YAML mapping")
    return value


def load_config(path: Path) -> USWhamConfig:
    resolved = path.resolve()
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("US WHAM configuration must be a YAML mapping")
    prod_value = Path(str(data.get("prod_yaml", "prod.yaml")))
    prod_yaml = prod_value if prod_value.is_absolute() else (resolved.parent / prod_value).resolve()
    if not prod_yaml.is_file():
        raise RuntimeError(f"Missing production configuration: {prod_yaml}")
    prod = load_prod_config(prod_yaml)

    coordinate_data = _mapping(data.get("coordinate"), "coordinate")
    coordinate = CoordinateConfig(
        minimum_deg=float(coordinate_data["minimum_deg"]),
        maximum_deg=float(coordinate_data["maximum_deg"]),
        bins=int(coordinate_data["bins"]),
        periodic=bool(coordinate_data.get("periodic", False)),
        report_minimum_deg=float(coordinate_data.get("report_minimum_deg", coordinate_data["minimum_deg"])),
        report_maximum_deg=float(coordinate_data.get("report_maximum_deg", coordinate_data["maximum_deg"])),
    )
    wham_data = _mapping(data.get("wham"), "wham")
    wham = WhamRuntimeConfig(
        executable=str(wham_data.get("executable", "wham")),
        tolerance=float(wham_data.get("tolerance", 1.0e-6)),
        temperature_k=float(wham_data.get("temperature_k", 300.0)),
        padding=int(wham_data.get("padding", 0)),
    )
    bootstrap_data = _mapping(data.get("bootstrap"), "bootstrap")
    bootstrap = BootstrapConfig(
        enabled=bool(bootstrap_data.get("enabled", True)),
        replicas=int(bootstrap_data.get("replicas", 500)),
        block_length_ps=float(bootstrap_data.get("block_length_ps", 2.0)),
        seed=int(bootstrap_data.get("seed", 20260920)),
        confidence_level=float(bootstrap_data.get("confidence_level", 0.95)),
        minimum_success_fraction=float(bootstrap_data.get("minimum_success_fraction", 0.9)),
    )
    reference = data.get("zero_reference_deg", [0.0, 5.0])
    if not isinstance(reference, list) or len(reference) != 2:
        raise ValueError("zero_reference_deg must contain [minimum, maximum]")
    thermo_data = _mapping(data.get("thermodynamics"), "thermodynamics")
    plot_data = data.get("plot", {})
    if not isinstance(plot_data, dict):
        raise ValueError("plot must be a YAML mapping")

    def interval(name: str) -> tuple[float, float]:
        value = thermo_data.get(name)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"thermodynamics.{name} must contain [minimum, maximum]")
        result = (float(value[0]), float(value[1]))
        if not result[0] < result[1]:
            raise ValueError(f"thermodynamics.{name} bounds must increase")
        return result

    cfg = USWhamConfig(
        prod_yaml=prod_yaml,
        prod=prod,
        output_dirname=str(data.get("output_dirname", "wham")),
        discard_ps=float(data.get("discard_ps", 0.0)),
        coordinate=coordinate,
        wham=wham,
        bootstrap=bootstrap,
        zero_reference_deg=(float(reference[0]), float(reference[1])),
        thermodynamics=ThermodynamicsConfig(
            syn_range_deg=interval("syn_range_deg"),
            anti_range_deg=interval("anti_range_deg"),
            transition_range_deg=interval("transition_range_deg"),
        ),
        plot=PlotConfig(
            smoothing_penalty=float(plot_data.get("smoothing_penalty", 10.0)),
        ),
    )
    if Path(cfg.output_dirname).name != cfg.output_dirname:
        raise ValueError("output_dirname must be one directory name")
    if not coordinate.minimum_deg < coordinate.maximum_deg or coordinate.bins < 2:
        raise ValueError("coordinate bounds/bins are invalid")
    if not coordinate.minimum_deg <= coordinate.report_minimum_deg < coordinate.report_maximum_deg <= coordinate.maximum_deg:
        raise ValueError("report bounds must lie inside coordinate bounds")
    if wham.tolerance <= 0 or wham.temperature_k <= 0 or wham.padding < 0:
        raise ValueError("WHAM tolerance/temperature must be positive and padding nonnegative")
    if cfg.discard_ps < 0 or bootstrap.block_length_ps <= 0:
        raise ValueError("discard_ps must be nonnegative and block_length_ps positive")
    if bootstrap.enabled and bootstrap.replicas < 2:
        raise ValueError("bootstrap.replicas must be at least 2")
    if not 0 < bootstrap.confidence_level < 1 or not 0 < bootstrap.minimum_success_fraction <= 1:
        raise ValueError("bootstrap confidence/success fractions must lie in (0, 1]")
    if not cfg.zero_reference_deg[0] < cfg.zero_reference_deg[1]:
        raise ValueError("zero_reference_deg bounds must increase")
    if cfg.plot.smoothing_penalty < 0:
        raise ValueError("plot.smoothing_penalty must be nonnegative")
    if prod.run.wall.exponent != 2:
        raise ValueError("WHAM currently requires a quadratic production METAWALL")
    return cfg


def resolve_executable(value: str) -> Path:
    expanded = Path(value).expanduser()
    if expanded.parent != Path(".") or expanded.is_absolute():
        if not expanded.is_file():
            raise RuntimeError(f"WHAM executable does not exist: {expanded}")
        return expanded.resolve()
    found = shutil.which(value)
    if found is None:
        raise RuntimeError(f"WHAM executable not found in PATH: {value}")
    return Path(found).resolve()


def read_window_series(cfg: USWhamConfig, yaml_text: str, repo_root: Path) -> list[WindowSeries]:
    root = pull_dir(cfg.prod.run.pull, repo_root)
    report_spec = root / "prod_report_spec.yaml"
    if not report_spec.is_file() or report_spec.read_text() != cfg.prod_yaml.read_text():
        raise RuntimeError(f"{report_spec} does not match {cfg.prod_yaml}; run us-prod-report first")
    source = root / "prod_dihedral.csv"
    grouped: dict[int, list[tuple[float, float, float]]] = {}
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"window_index", "target_deg", "time_ps", "dihedral_branch_deg"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required production columns in {source}")
        for row in reader:
            grouped.setdefault(int(row["window_index"]), []).append(
                (float(row["time_ps"]), float(row["dihedral_branch_deg"]), float(row["target_deg"]))
            )
    centers = window_centers(cfg.prod.run.pull.windows)
    if sorted(grouped) != list(range(len(centers))):
        raise RuntimeError("Production CSV does not contain the complete window set")
    series: list[WindowSeries] = []
    for index, center in enumerate(centers):
        kept = [row for row in grouped[index] if row[0] >= cfg.discard_ps - 1.0e-12]
        if len(kept) < 2:
            raise ValueError(f"Window {index} has fewer than two samples after discard")
        if any(not math.isclose(row[2], center, abs_tol=1.0e-8) for row in kept):
            raise ValueError(f"Target mismatch in production window {index}")
        times = np.asarray([row[0] for row in kept], dtype=float)
        angles = np.asarray([row[1] for row in kept], dtype=float)
        deltas = np.diff(times)
        if np.any(deltas <= 0) or not np.allclose(deltas, deltas[0], rtol=1.0e-7, atol=1.0e-9):
            raise ValueError(f"Production sampling times are not uniform in window {index}")
        series.append(WindowSeries(index, center, times, angles))
    return series


def write_series(path: Path, values: np.ndarray) -> None:
    with path.open("w") as stream:
        stream.write("#Frame dihedral_deg\n")
        for frame, value in enumerate(values, start=1):
            stream.write(f"{frame:8d} {value:14.8f}\n")


def write_metadata(path: Path, series: list[WindowSeries], force_constant: float) -> None:
    with path.open("w") as stream:
        for window in series:
            stream.write(
                f"window-{window.index:03d}.dat {window.center_deg:.8f} {force_constant:.12g}\n"
            )


def dcdftb_wall_to_wham_force(coefficient_kj_mol_coordinate2: float) -> float:
    """Convert DCDFTBMD V=kappa*dx^2 in kJ/mol to WHAM's half-K kcal form."""
    return 2.0 * coefficient_kj_mol_coordinate2 / KJ_PER_KCAL


def wham_command(cfg: USWhamConfig, executable: Path, metadata: Path, result: Path) -> list[str]:
    command = [str(executable)]
    if cfg.coordinate.periodic:
        command.append("P")
    command.extend(
        [
            f"{cfg.coordinate.minimum_deg:g}",
            f"{cfg.coordinate.maximum_deg:g}",
            str(cfg.coordinate.bins),
            f"{cfg.wham.tolerance:g}",
            f"{cfg.wham.temperature_k:g}",
            str(cfg.wham.padding),
            str(metadata),
            str(result),
        ]
    )
    return command


def run_wham(command: list[str], cwd: Path, log_path: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    log_path.write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"WHAM failed with return code {result.returncode}; see {log_path}")


def read_wham_result(path: Path) -> tuple[np.ndarray, np.ndarray]:
    coordinates: list[float] = []
    pmf: list[float] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        try:
            coordinate, free_energy = float(fields[0]), float(fields[1])
        except (IndexError, ValueError):
            continue
        if not math.isfinite(coordinate):
            raise ValueError(f"Non-finite coordinate in {path}: {line}")
        coordinates.append(coordinate)
        pmf.append(free_energy if math.isfinite(free_energy) else math.nan)
    if not coordinates:
        raise ValueError(f"No PMF grid found in {path}")
    return np.asarray(coordinates), np.asarray(pmf)


def zero_pmf(grid: np.ndarray, pmf: np.ndarray, reference: tuple[float, float]) -> np.ndarray:
    mask = (grid >= reference[0]) & (grid <= reference[1])
    if not np.any(mask):
        raise ValueError("No WHAM grid point lies in zero_reference_deg")
    return pmf - float(np.mean(pmf[mask]))


def circular_block_resample(values: np.ndarray, block_frames: int, rng: np.random.Generator) -> np.ndarray:
    if block_frames < 1 or block_frames > len(values):
        raise ValueError("Bootstrap block length must be between 1 and the series length")
    blocks = math.ceil(len(values) / block_frames)
    starts = rng.integers(0, len(values), size=blocks)
    offsets = np.arange(block_frames)
    indices = np.concatenate([(start + offsets) % len(values) for start in starts])
    return values[indices[: len(values)]]


def write_window_summary(path: Path, series: list[WindowSeries], block_frames: int) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window_index", "center_deg", "samples", "start_ps", "end_ps", "mean_deg", "std_deg", "block_frames"])
        for window in series:
            writer.writerow(
                [window.index, f"{window.center_deg:.8f}", len(window.angles_deg), f"{window.times_ps[0]:.8f}",
                 f"{window.times_ps[-1]:.8f}", f"{np.mean(window.angles_deg):.8f}",
                 f"{np.std(window.angles_deg, ddof=1):.8f}", block_frames]
            )


def write_overlap(path: Path, series: list[WindowSeries], cfg: USWhamConfig) -> None:
    ordered = sorted(series, key=lambda item: item.center_deg)
    edges = np.linspace(cfg.coordinate.minimum_deg, cfg.coordinate.maximum_deg, cfg.coordinate.bins + 1)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["left_window", "left_center_deg", "right_window", "right_center_deg", "histogram_overlap"])
        for left, right in zip(ordered[:-1], ordered[1:]):
            a, _ = np.histogram(left.angles_deg, bins=edges)
            b, _ = np.histogram(right.angles_deg, bins=edges)
            pa = a / a.sum()
            pb = b / b.sum()
            writer.writerow([left.index, f"{left.center_deg:.8f}", right.index, f"{right.center_deg:.8f}", f"{np.minimum(pa, pb).sum():.8f}"])


def compute_thermodynamics(
    grid: np.ndarray, pmf: np.ndarray, cfg: USWhamConfig
) -> dict[str, tuple[float, str]]:
    """Calculate basin minima, barriers, and integrated two-state populations."""
    thermo = cfg.thermodynamics

    def extrema(bounds: tuple[float, float], find_maximum: bool) -> tuple[float, float]:
        mask = (grid >= bounds[0]) & (grid <= bounds[1])
        if not np.any(mask):
            raise ValueError(f"No PMF grid point lies inside thermodynamic range {bounds}")
        indices = np.flatnonzero(mask)
        local = int(np.argmax(pmf[mask]) if find_maximum else np.argmin(pmf[mask]))
        index = int(indices[local])
        return float(grid[index]), float(pmf[index])

    syn_x, syn_f = extrema(thermo.syn_range_deg, False)
    anti_x, anti_f = extrema(thermo.anti_range_deg, False)
    transition_x, transition_f = extrema(thermo.transition_range_deg, True)
    beta = 1.0 / (0.00198720425864083 * cfg.wham.temperature_k)
    shifted = pmf - float(np.min(pmf))
    weights = np.exp(-beta * shifted)
    syn_weight = float(np.sum(weights[grid <= transition_x]))
    anti_weight = float(np.sum(weights[grid > transition_x]))
    total = syn_weight + anti_weight
    syn_fraction = syn_weight / total
    anti_fraction = anti_weight / total
    population_delta_g = -math.log(anti_fraction / syn_fraction) / beta
    return {
        "syn_min_dihedral": (syn_x, "degree"),
        "syn_min_pmf": (syn_f, "kcal mol^-1"),
        "anti_min_dihedral": (anti_x, "degree"),
        "anti_min_pmf": (anti_f, "kcal mol^-1"),
        "transition_max_dihedral": (transition_x, "degree"),
        "transition_max_pmf": (transition_f, "kcal mol^-1"),
        "delta_pmf_anti_minus_syn": (anti_f - syn_f, "kcal mol^-1"),
        "barrier_syn_to_anti": (transition_f - syn_f, "kcal mol^-1"),
        "barrier_anti_to_syn": (transition_f - anti_f, "kcal mol^-1"),
        "syn_molar_fraction": (syn_fraction, "fraction"),
        "anti_molar_fraction": (anti_fraction, "fraction"),
        "population_delta_g_anti_minus_syn": (population_delta_g, "kcal mol^-1"),
    }


def write_thermodynamics(
    csv_path: Path,
    text_path: Path,
    original: dict[str, tuple[float, str]],
    bootstrap: list[dict[str, tuple[float, str]]],
    confidence_level: float,
) -> None:
    alpha = 1.0 - confidence_level
    rows: list[tuple[str, str, float, float, float, float, float, float, float]] = []
    for metric, (value, unit) in original.items():
        samples = np.asarray([replica[metric][0] for replica in bootstrap], dtype=float)
        rows.append(
            (
                metric,
                unit,
                value,
                float(np.mean(samples)),
                float(np.std(samples, ddof=1)),
                float(np.quantile(samples, alpha / 2.0)),
                float(np.quantile(samples, 1.0 - alpha / 2.0)),
                float(np.min(samples)),
                float(np.max(samples)),
            )
        )
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "unit", "original", "bootstrap_mean", "bootstrap_std", "ci_lower", "ci_upper", "bootstrap_min", "bootstrap_max", "bootstrap_replicates"])
        for row in rows:
            writer.writerow([row[0], row[1], *(f"{value:.10g}" for value in row[2:]), len(bootstrap)])

    values = {metric: value for metric, _, value, *_ in rows}
    intervals = {metric: (low, high) for metric, _, _, _, _, low, high, _, _ in rows}
    extrema = {metric: (minimum, maximum) for metric, _, _, _, _, _, _, minimum, maximum in rows}
    confidence_percent = 100.0 * confidence_level
    text_path.write_text(
        "WHAM thermodynamic summary\n"
        "==========================\n"
        f"Delta PMF (anti - syn minima): {values['delta_pmf_anti_minus_syn']:.3f} kcal/mol "
        f"({confidence_percent:g}% CI {intervals['delta_pmf_anti_minus_syn'][0]:.3f} to {intervals['delta_pmf_anti_minus_syn'][1]:.3f}; "
        f"bootstrap min-max {extrema['delta_pmf_anti_minus_syn'][0]:.3f} to {extrema['delta_pmf_anti_minus_syn'][1]:.3f})\n"
        f"Syn molar fraction:  {100 * values['syn_molar_fraction']:.6f}% "
        f"(bootstrap min-max {100 * extrema['syn_molar_fraction'][0]:.6f}% to {100 * extrema['syn_molar_fraction'][1]:.6f}%)\n"
        f"Anti molar fraction: {100 * values['anti_molar_fraction']:.6f}% "
        f"(bootstrap min-max {100 * extrema['anti_molar_fraction'][0]:.6f}% to {100 * extrema['anti_molar_fraction'][1]:.6f}%)\n"
        f"Population Delta G (anti - syn): {values['population_delta_g_anti_minus_syn']:.3f} kcal/mol\n"
        f"Syn minimum:  phi={values['syn_min_dihedral']:.1f} deg, PMF={values['syn_min_pmf']:.3f} kcal/mol\n"
        f"Anti minimum: phi={values['anti_min_dihedral']:.1f} deg, PMF={values['anti_min_pmf']:.3f} kcal/mol\n"
        f"PMF maximum:  phi={values['transition_max_dihedral']:.1f} deg, PMF={values['transition_max_pmf']:.3f} kcal/mol\n"
        f"Barrier syn -> anti: {values['barrier_syn_to_anti']:.3f} kcal/mol "
        f"({confidence_percent:g}% CI {intervals['barrier_syn_to_anti'][0]:.3f} to {intervals['barrier_syn_to_anti'][1]:.3f}; "
        f"bootstrap min-max {extrema['barrier_syn_to_anti'][0]:.3f} to {extrema['barrier_syn_to_anti'][1]:.3f})\n"
        f"Barrier anti -> syn: {values['barrier_anti_to_syn']:.3f} kcal/mol "
        f"({confidence_percent:g}% CI {intervals['barrier_anti_to_syn'][0]:.3f} to {intervals['barrier_anti_to_syn'][1]:.3f}; "
        f"bootstrap min-max {extrema['barrier_anti_to_syn'][0]:.3f} to {extrema['barrier_anti_to_syn'][1]:.3f})\n"
        "Molar fractions integrate exp[-PMF/(RT)] on either side of the PMF maximum.\n"
    )


def smooth_pmf(values: np.ndarray, penalty: float) -> np.ndarray:
    """Suppress bin-scale noise with a second-difference penalized fit."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("PMF smoothing requires at least three one-dimensional values")
    if penalty < 0:
        raise ValueError("PMF smoothing penalty must be nonnegative")
    second_difference = np.diff(np.eye(values.size), n=2, axis=0)
    system = np.eye(values.size) + penalty * second_difference.T @ second_difference
    return np.linalg.solve(system, values)


def dense_hermite_curve(
    grid: np.ndarray,
    values: np.ndarray,
    points_per_interval: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate already-smoothed values without adding external dependencies."""
    if grid.ndim != 1 or values.ndim != 1 or grid.size != values.size or grid.size < 2:
        raise ValueError("Curve grid and values must be aligned one-dimensional arrays")
    if points_per_interval < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("Curve grid must increase and use at least two points per interval")
    slopes = np.gradient(values, grid)
    dense_grid: list[np.ndarray] = []
    dense_values: list[np.ndarray] = []
    for index in range(grid.size - 1):
        include_endpoint = index == grid.size - 2
        parameter = np.linspace(0.0, 1.0, points_per_interval, endpoint=include_endpoint)
        width = grid[index + 1] - grid[index]
        h00 = 2.0 * parameter**3 - 3.0 * parameter**2 + 1.0
        h10 = parameter**3 - 2.0 * parameter**2 + parameter
        h01 = -2.0 * parameter**3 + 3.0 * parameter**2
        h11 = parameter**3 - parameter**2
        dense_grid.append(grid[index] + parameter * width)
        dense_values.append(
            h00 * values[index]
            + h10 * width * slopes[index]
            + h01 * values[index + 1]
            + h11 * width * slopes[index + 1]
        )
    return np.concatenate(dense_grid), np.concatenate(dense_values)


def plot_pmf(path: Path, grid: np.ndarray, original: np.ndarray, lower: np.ndarray, upper: np.ndarray, cfg: USWhamConfig, successful: int, thermodynamics: dict[str, tuple[float, str]], thermodynamic_std: dict[str, float], thermodynamic_interval: dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = (grid >= cfg.coordinate.report_minimum_deg) & (grid <= cfg.coordinate.report_maximum_deg)
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    lower_error = np.maximum(0.0, original[mask] - lower[mask])
    upper_error = np.maximum(0.0, upper[mask] - original[mask])
    smoothed = smooth_pmf(original[mask], cfg.plot.smoothing_penalty)
    smooth_grid, smooth_values = dense_hermite_curve(grid[mask], smoothed)
    ax.plot(
        smooth_grid,
        smooth_values,
        color="#d1495b",
        linewidth=2.2,
        label="Penalized smooth guide",
        zorder=2,
    )
    ax.errorbar(
        grid[mask],
        original[mask],
        yerr=np.vstack((lower_error, upper_error)),
        fmt="o",
        linestyle="none",
        markersize=3.8,
        color="#173f5f",
        ecolor="#6f98b8",
        elinewidth=1.0,
        capsize=2.0,
        capthick=0.9,
        zorder=3,
        label=(
            "Original-data WHAM with "
            f"{cfg.bootstrap.confidence_level * 100:g}% bootstrap interval"
        ),
    )
    ax.set_xlabel("Dihedral (degrees)")
    ax.set_ylabel("PMF (kcal mol⁻¹)")
    ax.set_title(
        f"{cfg.prod.run.pull.system} torsional PMF "
        f"({successful} bootstrap replicas)"
    )
    ax.grid(alpha=0.24)
    ax.legend(frameon=False)
    syn_fraction = thermodynamics["syn_molar_fraction"][0]
    syn_lower, syn_upper = thermodynamic_interval["syn_molar_fraction"]
    anti_fraction = thermodynamics["anti_molar_fraction"][0]
    anti_lower, anti_upper = thermodynamic_interval["anti_molar_fraction"]
    summary = (
        f"ΔPMF anti−syn = {thermodynamics['delta_pmf_anti_minus_syn'][0]:.2f} ± "
        f"{thermodynamic_std['delta_pmf_anti_minus_syn']:.2f} kcal mol⁻¹\n"
        f"x(syn) = {100 * syn_fraction:.2f}% "
        f"(+{100 * (syn_upper - syn_fraction):.2f}/−{100 * (syn_fraction - syn_lower):.2f})\n"
        f"x(anti) = {100 * anti_fraction:.2f}% "
        f"(+{100 * (anti_upper - anti_fraction):.2f}/−{100 * (anti_fraction - anti_lower):.2f})\n"
        f"ΔG‡ syn→anti = {thermodynamics['barrier_syn_to_anti'][0]:.2f} kcal mol⁻¹\n"
        f"ΔG‡ anti→syn = {thermodynamics['barrier_anti_to_syn'][0]:.2f} kcal mol⁻¹"
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"})
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return smooth_grid, smooth_values


def create_us_wham(cfg: USWhamConfig, yaml_text: str, repo_root: Path) -> tuple[Path, Path, int]:
    executable = resolve_executable(cfg.wham.executable)
    series = read_window_series(cfg, yaml_text, repo_root)
    root = pull_dir(cfg.prod.run.pull, repo_root)
    output = root / cfg.output_dirname
    output.mkdir(parents=True, exist_ok=True)
    for window in series:
        write_series(output / f"window-{window.index:03d}.dat", window.angles_deg)

    # DCDFTBMD METAWALL uses V=kappa*delta^2 with kappa in kJ/mol/CV^2.
    # Grossfield WHAM uses V=(1/2)*K*delta^2 with K in kcal/mol/CV^2.
    # The legacy production-field name says kcal, but the completed input and
    # exact YAML snapshots must remain untouched; its numeric value is kJ/mol.
    wham_force = dcdftb_wall_to_wham_force(
        cfg.prod.run.wall.coefficient_kcal_mol_deg2
    )
    metadata = output / "metadata.dat"
    write_metadata(metadata, series, wham_force)
    result_path = output / "wham_result.dat"
    command = wham_command(cfg, executable, metadata, result_path)
    (output / "wham_command.txt").write_text(" ".join(command) + "\n")
    run_wham(command, output, output / "wham.log")
    full_grid, full_original_raw = read_wham_result(result_path)
    report_mask = (
        (full_grid >= cfg.coordinate.report_minimum_deg)
        & (full_grid <= cfg.coordinate.report_maximum_deg)
    )
    grid = full_grid[report_mask]
    original_raw = full_original_raw[report_mask]
    if grid.size == 0 or not np.all(np.isfinite(original_raw)):
        raise ValueError("WHAM produced an empty or non-finite value inside the reporting domain")
    original = zero_pmf(grid, original_raw, cfg.zero_reference_deg)

    interval = float(np.diff(series[0].times_ps[:2])[0])
    block_frames = int(round(cfg.bootstrap.block_length_ps / interval))
    if not math.isclose(block_frames * interval, cfg.bootstrap.block_length_ps, rel_tol=1.0e-7, abs_tol=1.0e-9):
        raise ValueError("bootstrap.block_length_ps must be divisible by the sampling interval")
    if block_frames > len(series[0].angles_deg) // 2:
        raise ValueError("Bootstrap block length leaves fewer than two blocks per window")
    write_window_summary(output / "window_summary.csv", series, block_frames)
    write_overlap(output / "overlap.csv", series, cfg)

    bootstrap_pmfs: list[np.ndarray] = []
    if cfg.bootstrap.enabled:
        rng = np.random.default_rng(cfg.bootstrap.seed)
        with tempfile.TemporaryDirectory(prefix="qmmd-us-wham-") as tmp:
            workspace = Path(tmp)
            for replica in range(cfg.bootstrap.replicas):
                for window in series:
                    values = circular_block_resample(window.angles_deg, block_frames, rng)
                    write_series(workspace / f"window-{window.index:03d}.dat", values)
                replica_meta = workspace / "metadata.dat"
                replica_result = workspace / "result.dat"
                replica_log = workspace / "wham.log"
                write_metadata(replica_meta, series, wham_force)
                replica_command = wham_command(cfg, executable, replica_meta, replica_result)
                try:
                    run_wham(replica_command, workspace, replica_log)
                    replica_full_grid, replica_full_raw = read_wham_result(replica_result)
                    if not np.array_equal(replica_full_grid, full_grid):
                        raise ValueError("Bootstrap WHAM grid differs from original grid")
                    replica_grid = replica_full_grid[report_mask]
                    replica_raw = replica_full_raw[report_mask]
                    if not np.all(np.isfinite(replica_raw)):
                        raise ValueError("Bootstrap PMF has a non-finite value inside the reporting domain")
                    bootstrap_pmfs.append(zero_pmf(replica_grid, replica_raw, cfg.zero_reference_deg))
                except (RuntimeError, ValueError) as exc:
                    print(f"WARNING: bootstrap replica {replica + 1} failed: {exc}")
        minimum_successes = math.ceil(cfg.bootstrap.replicas * cfg.bootstrap.minimum_success_fraction)
        if len(bootstrap_pmfs) < minimum_successes:
            raise RuntimeError(f"Only {len(bootstrap_pmfs)}/{cfg.bootstrap.replicas} bootstrap replicas succeeded")

    if bootstrap_pmfs:
        boot = np.vstack(bootstrap_pmfs)
        mean = np.mean(boot, axis=0)
        std = np.std(boot, axis=0, ddof=1)
        alpha = 1.0 - cfg.bootstrap.confidence_level
        lower = np.quantile(boot, alpha / 2.0, axis=0)
        upper = np.quantile(boot, 1.0 - alpha / 2.0, axis=0)
        with (output / "bootstrap_pmfs.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["bootstrap_replica", "dihedral_deg", "pmf_kcal_mol"])
            for replica_index, replica_pmf in enumerate(boot, start=1):
                writer.writerows(
                    (replica_index, f"{coordinate:.8f}", f"{value:.10g}")
                    for coordinate, value in zip(grid, replica_pmf)
                )
    else:
        mean = original.copy()
        std = np.zeros_like(original)
        lower = original.copy()
        upper = original.copy()

    original_thermodynamics = compute_thermodynamics(grid, original, cfg)
    bootstrap_thermodynamics = [
        compute_thermodynamics(grid, replica, cfg) for replica in bootstrap_pmfs
    ]
    thermodynamic_std = {
        metric: (
            float(
                np.std(
                    [replica[metric][0] for replica in bootstrap_thermodynamics],
                    ddof=1,
                )
            )
            if bootstrap_thermodynamics
            else 0.0
        )
        for metric in original_thermodynamics
    }
    alpha = 1.0 - cfg.bootstrap.confidence_level
    thermodynamic_interval = {
        metric: (
            (
                float(np.quantile([replica[metric][0] for replica in bootstrap_thermodynamics], alpha / 2.0)),
                float(np.quantile([replica[metric][0] for replica in bootstrap_thermodynamics], 1.0 - alpha / 2.0)),
            )
            if bootstrap_thermodynamics
            else (original_thermodynamics[metric][0], original_thermodynamics[metric][0])
        )
        for metric in original_thermodynamics
    }
    if bootstrap_thermodynamics:
        write_thermodynamics(
            output / "thermodynamics.csv",
            output / "thermodynamics.txt",
            original_thermodynamics,
            bootstrap_thermodynamics,
            cfg.bootstrap.confidence_level,
        )

    table = output / "pmf.csv"
    with table.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dihedral_deg", "pmf_kcal_mol", "bootstrap_mean_kcal_mol", "bootstrap_std_kcal_mol", "ci_lower_kcal_mol", "ci_upper_kcal_mol", "bootstrap_replicates"])
        for values in zip(grid, original, mean, std, lower, upper):
            writer.writerow([*(f"{value:.10g}" for value in values), len(bootstrap_pmfs)])
    figure = output / "pmf.png"
    smooth_grid, smooth_values = plot_pmf(
        figure,
        grid,
        original,
        lower,
        upper,
        cfg,
        len(bootstrap_pmfs),
        original_thermodynamics,
        thermodynamic_std,
        thermodynamic_interval,
    )
    with (output / "pmf_smooth.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dihedral_deg", "smoothed_pmf_kcal_mol"])
        writer.writerows(
            (f"{coordinate:.10g}", f"{value:.10g}")
            for coordinate, value in zip(smooth_grid, smooth_values)
        )
    (output / "wham_spec.yaml").write_text(yaml_text)
    return figure, table, len(bootstrap_pmfs)


def run_us_wham(path: Path) -> None:
    resolved = path.resolve()
    text = resolved.read_text()
    cfg = load_config(resolved)
    figure, table, replicas = create_us_wham(cfg, text, find_repo_root(resolved))
    print(f"OK: wrote WHAM PMF data to {table}")
    print(f"OK: wrote WHAM PMF plot to {figure}")
    print(f"OK: {replicas} bootstrap replicas contributed")
