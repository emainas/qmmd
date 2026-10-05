"""Analyze a completed REFEP Hamiltonian replica-exchange calculation."""

from __future__ import annotations

import csv
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from qmmd.cphmd_dgref import find_repo_root
from qmmd.refep_equil_report import (
    DihedralSample,
    RefepDihedralConfig,
    angle_near_target,
    load_refep_equil_report_config,
    plot_dihedral_timeseries,
    read_cpptraj_dihedral,
    render_dihedral_cpptraj_input,
    run_cpptraj,
    write_dihedral_csv,
)
from qmmd.refep_prod import refep_prod_dir
from qmmd.refep_prod_post import (
    RefepProdPostConfig,
    load_refep_prod_post_config,
    refep_prod_post_dir,
    render_energy_mdin,
    render_energy_run_script,
    render_local_run_script,
    render_post_provenance,
    render_post_run_script,
    render_post_slurm_script,
    render_task_manifest,
    validate_completed_production,
    validate_stripped_inputs,
)


GAS_CONSTANT_KCAL = 0.00198720425864083
_ENERGY_RE = re.compile(r"ENE=\s*([-+\d.eE]+)")
_FRAME_COUNT_RE = re.compile(r"TRAJENE: Frames in trajectory=\s*(\d+)")
_FAILURE_RE = re.compile(
    r"\b(?:nan|infinity|fatal|segmentation)\b|shake cannot|coordinate resetting|"
    r"vlimit exceeded|bomb|terminated abnormally",
    re.IGNORECASE,
)
_EXCHANGE_RE = re.compile(r"^# exchange\s+(\d+)\s*$")


@dataclass(frozen=True, slots=True)
class RefepReportConfig:
    post: RefepProdPostConfig
    output_dir: str
    reference_state: str | None
    style: Path
    temperature_k: float
    bootstrap_samples: int
    bootstrap_block_frames: int
    bootstrap_seed: int
    histogram_bins: int
    convergence_min_frames: int
    convergence_stride_frames: int
    mbar_tolerance: float
    mbar_max_iterations: int
    dihedral: RefepDihedralConfig | None


@dataclass(frozen=True, slots=True)
class EnergyGrid:
    lambdas: np.ndarray
    energies: np.ndarray
    time_ps: np.ndarray

    @property
    def windows(self) -> int:
        return int(self.energies.shape[0])

    @property
    def frames(self) -> int:
        return int(self.energies.shape[2])


@dataclass(frozen=True, slots=True)
class FreeEnergyAnalysis:
    forward_fep_edges: np.ndarray
    reverse_fep_edges: np.ndarray
    bar_edges: np.ndarray
    neighbor_overlap: np.ndarray
    ti_integrand: np.ndarray
    ti_std: np.ndarray
    ti_sem: np.ndarray
    ti_fit_rms: np.ndarray
    mbar_free_energies: np.ndarray
    mbar_overlap: np.ndarray
    totals: dict[str, float]


@dataclass(frozen=True, slots=True)
class ReplicaDiagnostics:
    exchanges: np.ndarray
    walker_positions: np.ndarray
    edge_pairs: tuple[tuple[int, int], ...]
    edge_attempts: np.ndarray
    edge_accepts: np.ndarray
    residence_percent: np.ndarray
    roundtrip_counts: np.ndarray
    roundtrip_mean_ps: np.ndarray
    reached_both_endpoints: np.ndarray


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a YAML mapping")
    return value


def _child_path(value: object, field: str, default: str) -> str:
    text = default if value is None else str(value).strip()
    path = Path(text)
    if not text or path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError(f"{field} must be a child path")
    return text


def load_refep_report_config(yaml_path: Path) -> RefepReportConfig:
    resolved = yaml_path.resolve()
    post = load_refep_prod_post_config(resolved)
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("REFEP production configuration must be a YAML mapping")
    raw = _mapping(data.get("report", {}), "report")
    labels = (
        post.prod.equil.prep.lambda0_label,
        post.prod.equil.prep.lambda1_label,
    )
    raw_reference = raw.get("reference_state")
    reference_state = None
    if raw_reference is not None:
        reference_state = str(raw_reference).strip()
        if reference_state not in labels:
            raise ValueError(
                "report.reference_state must match an endpoint label: "
                + ", ".join(labels)
            )

    root = find_repo_root(resolved)
    style = Path(str(raw.get("style", "plotting/prl.mplstyle")))
    if not style.is_absolute():
        style = (root / style).resolve()
    if not style.is_file():
        raise FileNotFoundError(f"Missing matplotlib style: {style}")

    temperature = float(raw.get("temperature_k", post.prod.cntrl["temp0"]))
    simulation_temperature = float(post.prod.cntrl["temp0"])
    if temperature <= 0.0:
        raise ValueError("report.temperature_k must be positive")
    if not math.isclose(temperature, simulation_temperature, abs_tol=1.0e-8):
        raise ValueError(
            "report.temperature_k must match remd_mdin production temp0 "
            f"({simulation_temperature:g} K)"
        )

    bootstrap_samples = int(raw.get("bootstrap_samples", 200))
    bootstrap_block_frames = int(raw.get("bootstrap_block_frames", 5))
    bootstrap_seed = int(raw.get("bootstrap_seed", 20260929))
    histogram_bins = int(raw.get("histogram_bins", 20))
    convergence_min_frames = int(raw.get("convergence_min_frames", 10))
    convergence_stride_frames = int(raw.get("convergence_stride_frames", 5))
    mbar_tolerance = float(raw.get("mbar_tolerance", 1.0e-10))
    mbar_max_iterations = int(raw.get("mbar_max_iterations", 10000))
    expected_frames = (
        int(post.prod.cntrl["nstlim"])
        * int(post.prod.cntrl["numexchg"])
        // int(post.prod.cntrl["ntwx"])
    )
    if bootstrap_samples < 2:
        raise ValueError("report.bootstrap_samples must be at least 2")
    if bootstrap_block_frames < 1 or bootstrap_block_frames > expected_frames:
        raise ValueError(
            f"report.bootstrap_block_frames must be between 1 and {expected_frames}"
        )
    if histogram_bins < 5:
        raise ValueError("report.histogram_bins must be at least 5")
    if not 2 <= convergence_min_frames <= expected_frames:
        raise ValueError(
            f"report.convergence_min_frames must be between 2 and {expected_frames}"
        )
    if convergence_stride_frames < 1:
        raise ValueError("report.convergence_stride_frames must be positive")
    if mbar_tolerance <= 0.0 or mbar_max_iterations < 1:
        raise ValueError("report MBAR tolerance and iteration limit must be positive")

    equil_report = load_refep_equil_report_config(post.prod.equil_yaml)

    return RefepReportConfig(
        post=post,
        output_dir=_child_path(raw.get("output_dir"), "report.output_dir", "report"),
        reference_state=reference_state,
        style=style,
        temperature_k=temperature,
        bootstrap_samples=bootstrap_samples,
        bootstrap_block_frames=bootstrap_block_frames,
        bootstrap_seed=bootstrap_seed,
        histogram_bins=histogram_bins,
        convergence_min_frames=convergence_min_frames,
        convergence_stride_frames=convergence_stride_frames,
        mbar_tolerance=mbar_tolerance,
        mbar_max_iterations=mbar_max_iterations,
        dihedral=equil_report.dihedral,
    )


def refep_report_dir(cfg: RefepReportConfig) -> Path:
    return refep_prod_dir(cfg.post.prod) / cfg.output_dir


def reporting_labels(cfg: RefepReportConfig) -> tuple[str, str]:
    labels = (
        cfg.post.prod.equil.prep.lambda0_label,
        cfg.post.prod.equil.prep.lambda1_label,
    )
    if cfg.reference_state is None or cfg.reference_state == labels[1]:
        return labels
    return labels[1], labels[0]


def orient_grid_for_reporting(
    cfg: RefepReportConfig, grid: EnergyGrid
) -> EnergyGrid:
    labels = (
        cfg.post.prod.equil.prep.lambda0_label,
        cfg.post.prod.equil.prep.lambda1_label,
    )
    if cfg.reference_state is None or cfg.reference_state == labels[1]:
        return grid
    return EnergyGrid(
        lambdas=1.0 - grid.lambdas[::-1],
        energies=grid.energies[::-1, ::-1, :],
        time_ps=grid.time_ps,
    )


def orient_replica_for_reporting(
    cfg: RefepReportConfig, replica: ReplicaDiagnostics
) -> ReplicaDiagnostics:
    labels = (
        cfg.post.prod.equil.prep.lambda0_label,
        cfg.post.prod.equil.prep.lambda1_label,
    )
    if cfg.reference_state is None or cfg.reference_state == labels[1]:
        return replica
    count = cfg.post.prod.equil.prep.windows
    adjacent = count - 1
    attempts = replica.edge_attempts.copy()
    accepts = replica.edge_accepts.copy()
    attempts[:adjacent] = attempts[:adjacent][::-1]
    accepts[:adjacent] = accepts[:adjacent][::-1]
    return ReplicaDiagnostics(
        exchanges=replica.exchanges,
        walker_positions=(count - 1) - replica.walker_positions,
        edge_pairs=replica.edge_pairs,
        edge_attempts=attempts,
        edge_accepts=accepts,
        residence_percent=replica.residence_percent[:, ::-1],
        roundtrip_counts=replica.roundtrip_counts,
        roundtrip_mean_ps=replica.roundtrip_mean_ps,
        reached_both_endpoints=replica.reached_both_endpoints,
    )


def _config_without_report(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("REFEP configuration snapshot must be a YAML mapping")
    payload = dict(data)
    payload.pop("report", None)
    return payload


def _read_lambdas(path: Path, windows: int) -> np.ndarray:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != windows:
        raise ValueError(f"Expected {windows} lambda rows in {path}; found {len(rows)}")
    indices = [int(row["window"]) for row in rows]
    if indices != list(range(windows)):
        raise ValueError(f"Lambda manifest is not ordered from 0 to {windows - 1}")
    lambdas = np.asarray([float(row["lambda"]) for row in rows], dtype=float)
    if not np.all(np.diff(lambdas) > 0.0):
        raise ValueError("Lambda values must be strictly increasing")
    if not np.isclose(lambdas[0], 0.0) or not np.isclose(lambdas[-1], 1.0):
        raise ValueError("REFEP lambda ladder must span exactly 0 to 1")
    return lambdas


def load_completed_energy_grid(cfg: RefepReportConfig) -> EnergyGrid:
    windows = validate_completed_production(cfg.post)
    post_dir = refep_prod_post_dir(cfg.post)
    spec = post_dir / "refep-prod-post-spec.yaml"
    if not spec.is_file():
        raise FileNotFoundError(f"Missing REFEP post specification: {spec}")
    if _config_without_report(spec.read_text()) != _config_without_report(
        cfg.post.yaml_path.read_text()
    ):
        raise ValueError(f"{spec} does not match the single-point configuration")

    expected_inputs = {
        "energy.mdin": render_energy_mdin(cfg.post),
        "task-manifest.csv": render_task_manifest(cfg.post, windows),
        "provenance.yaml": render_post_provenance(cfg.post, windows),
        "run-energy.sh": render_energy_run_script(cfg.post),
        "run.sh": render_post_run_script(cfg.post),
        "slurm.sh": render_post_slurm_script(cfg.post),
    }
    if cfg.post.keep_mask is not None:
        expected_inputs["run-local.sh"] = render_local_run_script(cfg.post)
    for name, expected in expected_inputs.items():
        path = post_dir / name
        if not path.is_file() or path.read_text() != expected:
            raise ValueError(f"Prepared single-point input differs from config: {path}")
    validate_stripped_inputs(cfg.post, windows, post_dir)

    count = len(windows)
    lambdas = _read_lambdas(refep_prod_dir(cfg.post.prod) / "lambda-manifest.csv", count)
    frames = windows[0].frame_count
    energies = np.empty((count, count, frames), dtype=float)
    energy_dir = post_dir / cfg.post.energy_dirname
    expected_names = {
        f"refep-j{sample:03d}-k{evaluation:03d}.mdout"
        for sample in range(count)
        for evaluation in range(count)
    }
    actual_names = {path.name for path in energy_dir.glob("refep-j*-k*.mdout")}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            "Incomplete or unexpected REFEP energy grid; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    for sample in range(count):
        for evaluation in range(count):
            stem = f"refep-j{sample:03d}-k{evaluation:03d}"
            mdout = energy_dir / f"{stem}.mdout"
            restart = energy_dir / f"{stem}.rst7"
            if not restart.is_file() or restart.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty single-point restart: {restart}")
            text = mdout.read_text(errors="replace")
            frame_match = _FRAME_COUNT_RE.search(text)
            if frame_match is None or int(frame_match.group(1)) != frames:
                raise ValueError(f"Unexpected trajectory frame count in {mdout}")
            if "TRAJENE: Trajene complete." not in text or "5.  TIMINGS" not in text:
                raise ValueError(f"Single-point trajectory evaluation is incomplete: {mdout}")
            failure = _FAILURE_RE.search(text)
            if failure is not None:
                raise ValueError(f"Amber failure marker {failure.group(0)!r} in {mdout}")
            values = np.asarray([float(value) for value in _ENERGY_RE.findall(text)])
            if len(values) != frames or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Expected {frames} finite ENE records in {mdout}; found {len(values)}"
                )
            energies[sample, evaluation] = values

    frame_interval_ps = int(cfg.post.prod.cntrl["ntwx"]) * float(
        cfg.post.prod.cntrl["dt"]
    )
    time_ps = np.arange(1, frames + 1, dtype=float) * frame_interval_ps
    return EnergyGrid(lambdas=lambdas, energies=energies, time_ps=time_ps)


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _logmeanexp(values: np.ndarray) -> float:
    return float(_logsumexp(values) - math.log(values.size))


def _fermi(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(values, -700.0, 700.0)))


def bar_free_energy(
    forward_work_kbt: np.ndarray,
    reverse_work_kbt: np.ndarray,
    tolerance: float = 1.0e-10,
    max_iterations: int = 1000,
) -> float:
    """Return equal-sample Bennett Δf for A→B in dimensionless units."""
    forward = np.asarray(forward_work_kbt, dtype=float)
    reverse = np.asarray(reverse_work_kbt, dtype=float)
    if forward.ndim != 1 or reverse.ndim != 1 or not len(forward) or not len(reverse):
        raise ValueError("BAR work arrays must be nonempty and one-dimensional")

    def equation(delta: float) -> float:
        return float(np.sum(_fermi(forward - delta)) - np.sum(_fermi(reverse + delta)))

    bound = max(20.0, float(np.max(np.abs(np.concatenate((forward, reverse))))) + 10.0)
    low, high = -bound, bound
    f_low, f_high = equation(low), equation(high)
    for _ in range(12):
        if f_low * f_high <= 0.0:
            break
        low *= 2.0
        high *= 2.0
        f_low, f_high = equation(low), equation(high)
    else:
        raise ValueError("Could not bracket the BAR solution")
    midpoint = 0.0
    for _ in range(max_iterations):
        midpoint = 0.5 * (low + high)
        f_mid = equation(midpoint)
        if abs(f_mid) <= tolerance or 0.5 * (high - low) <= tolerance:
            return midpoint
        if f_low * f_mid > 0.0:
            low, f_low = midpoint, f_mid
        else:
            high = midpoint
    raise ValueError("BAR solver did not converge")


def solve_mbar(
    energies: np.ndarray,
    beta: float,
    tolerance: float = 1.0e-10,
    max_iterations: int = 10000,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the equal-count MBAR equations and return f_k and overlap matrix."""
    if energies.ndim != 3 or energies.shape[0] != energies.shape[1]:
        raise ValueError("MBAR energies must have shape (sample, evaluation, frame)")
    states, _, frames = energies.shape
    reduced = np.concatenate([beta * energies[j] for j in range(states)], axis=1)
    counts = np.full(states, float(frames))
    free = np.zeros(states) if initial is None else np.asarray(initial, dtype=float).copy()
    if free.shape != (states,):
        raise ValueError("MBAR initial free energies have the wrong shape")
    free -= free[0]
    for _ in range(max_iterations):
        log_denominator = _logsumexp(
            np.log(counts)[:, None] + free[:, None] - reduced, axis=0
        )
        updated = -_logsumexp(-reduced - log_denominator[None, :], axis=1)
        updated -= updated[0]
        if float(np.max(np.abs(updated - free))) < tolerance:
            free = updated
            break
        free = updated
    else:
        raise ValueError(f"MBAR did not converge within {max_iterations} iterations")

    log_denominator = _logsumexp(
        np.log(counts)[:, None] + free[:, None] - reduced, axis=0
    )
    weights = np.exp((free[:, None] - reduced - log_denominator[None, :]).T)
    overlap = (weights.T @ weights) * counts[None, :]
    if not np.allclose(overlap.sum(axis=1), 1.0, atol=1.0e-7, rtol=0.0):
        raise ValueError("MBAR overlap matrix is not row stochastic")
    return free, overlap


def _trapezoid(values: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(x)))


def ti_charge_derivative(
    grid: EnergyGrid, block_frames: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit U(q,lambda) quadratically and integrate ensemble-mean dU/dlambda."""
    lambdas = grid.lambdas
    design = np.column_stack((lambdas**2, lambdas, np.ones_like(lambdas)))
    projector = np.linalg.pinv(design)
    derivative = np.empty((grid.windows, grid.frames), dtype=float)
    fit_rms = np.empty(grid.windows, dtype=float)
    for sample, lam in enumerate(lambdas):
        coefficients = projector @ grid.energies[sample]
        derivative[sample] = 2.0 * coefficients[0] * lam + coefficients[1]
        residual = design @ coefficients - grid.energies[sample]
        fit_rms[sample] = float(np.sqrt(np.mean(residual**2)))
    means = derivative.mean(axis=1)
    std = derivative.std(axis=1, ddof=1)
    sem = np.empty(grid.windows, dtype=float)
    for state in range(grid.windows):
        blocks = [
            derivative[state, start : start + block_frames].mean()
            for start in range(0, grid.frames, block_frames)
            if len(derivative[state, start : start + block_frames]) == block_frames
        ]
        sem[state] = (
            float(np.std(blocks, ddof=1) / math.sqrt(len(blocks)))
            if len(blocks) > 1
            else float(std[state] / math.sqrt(grid.frames))
        )
    ti = _trapezoid(means, lambdas)
    legacy_gap = (grid.energies[:, -1] - grid.energies[:, 0]).mean(axis=1)
    legacy_ti = _trapezoid(legacy_gap, lambdas)
    return means, std, sem, fit_rms, ti, legacy_ti


def _histogram_overlap(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    low = float(min(a.min(), b.min()))
    high = float(max(a.max(), b.max()))
    if math.isclose(low, high):
        return 1.0
    edges = np.linspace(low, high, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return float(np.sum(np.sqrt(pa * pb)))


def analyze_energy_grid(
    cfg: RefepReportConfig, grid: EnergyGrid
) -> FreeEnergyAnalysis:
    kbt = GAS_CONSTANT_KCAL * cfg.temperature_k
    beta = 1.0 / kbt
    edges = grid.windows - 1
    forward = np.empty(edges)
    reverse = np.empty(edges)
    bar = np.empty(edges)
    overlap = np.empty(edges)
    for index in range(edges):
        forward_work = grid.energies[index, index + 1] - grid.energies[index, index]
        reverse_work = (
            grid.energies[index + 1, index]
            - grid.energies[index + 1, index + 1]
        )
        forward[index] = -kbt * _logmeanexp(-beta * forward_work)
        reverse[index] = kbt * _logmeanexp(-beta * reverse_work)
        bar[index] = kbt * bar_free_energy(beta * forward_work, beta * reverse_work)
        overlap[index] = _histogram_overlap(
            forward_work, -reverse_work, cfg.histogram_bins
        )
    initial = np.concatenate(([0.0], np.cumsum(bar / kbt)))
    mbar_free, mbar_overlap = solve_mbar(
        grid.energies,
        beta,
        cfg.mbar_tolerance,
        cfg.mbar_max_iterations,
        initial,
    )
    ti_mean, ti_std, ti_sem, ti_rms, ti, legacy_ti = ti_charge_derivative(
        grid, cfg.bootstrap_block_frames
    )
    totals = {
        "FEP forward": float(forward.sum()),
        "FEP reverse": float(reverse.sum()),
        "BAR": float(bar.sum()),
        "TI charge derivative": ti,
        "TI legacy endpoint gap": legacy_ti,
        "MBAR": float((mbar_free[-1] - mbar_free[0]) * kbt),
    }
    return FreeEnergyAnalysis(
        forward_fep_edges=forward,
        reverse_fep_edges=reverse,
        bar_edges=bar,
        neighbor_overlap=overlap,
        ti_integrand=ti_mean,
        ti_std=ti_std,
        ti_sem=ti_sem,
        ti_fit_rms=ti_rms,
        mbar_free_energies=mbar_free * kbt,
        mbar_overlap=mbar_overlap,
        totals=totals,
    )


def _circular_block_indices(
    rng: np.random.Generator, frames: int, block_frames: int
) -> np.ndarray:
    blocks = math.ceil(frames / block_frames)
    starts = rng.integers(0, frames, size=blocks)
    return np.concatenate(
        [(start + np.arange(block_frames)) % frames for start in starts]
    )[:frames]


def bootstrap_free_energies(
    cfg: RefepReportConfig, grid: EnergyGrid
) -> tuple[dict[str, float], np.ndarray]:
    rng = np.random.default_rng(cfg.bootstrap_seed)
    methods = tuple(analyze_energy_grid(cfg, grid).totals)
    values = np.empty((cfg.bootstrap_samples, len(methods)))
    neighbor_bar = np.empty((cfg.bootstrap_samples, grid.windows - 1))
    for sample in range(cfg.bootstrap_samples):
        resampled = np.empty_like(grid.energies)
        for state in range(grid.windows):
            indices = _circular_block_indices(
                rng, grid.frames, cfg.bootstrap_block_frames
            )
            resampled[state] = grid.energies[state][:, indices]
        result = analyze_energy_grid(
            cfg,
            EnergyGrid(grid.lambdas, resampled, grid.time_ps),
        )
        values[sample] = [result.totals[name] for name in methods]
        neighbor_bar[sample] = result.bar_edges
    uncertainty = {
        name: float(values[:, index].std(ddof=1))
        for index, name in enumerate(methods)
    }
    return uncertainty, neighbor_bar.std(axis=0, ddof=1)


def convergence_series(
    cfg: RefepReportConfig, grid: EnergyGrid
) -> list[tuple[int, float, dict[str, float]]]:
    endpoints = list(
        range(
            cfg.convergence_min_frames,
            grid.frames + 1,
            cfg.convergence_stride_frames,
        )
    )
    if endpoints[-1] != grid.frames:
        endpoints.append(grid.frames)
    rows: list[tuple[int, float, dict[str, float]]] = []
    for frames in endpoints:
        subset = EnergyGrid(
            grid.lambdas,
            grid.energies[:, :, :frames],
            grid.time_ps[:frames],
        )
        rows.append((frames, float(grid.time_ps[frames - 1]), analyze_energy_grid(cfg, subset).totals))
    return rows


def parse_rem_log(cfg: RefepReportConfig) -> ReplicaDiagnostics:
    path = refep_prod_dir(cfg.post.prod) / "rem.log"
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty replica-exchange log: {path}")
    blocks: list[tuple[int, list[tuple[int, int, bool]]]] = []
    exchange: int | None = None
    rows: list[tuple[int, int, bool]] = []
    for line in path.read_text().splitlines():
        match = _EXCHANGE_RE.match(line)
        if match:
            if exchange is not None:
                blocks.append((exchange, rows))
            exchange = int(match.group(1))
            rows = []
            continue
        if exchange is None or line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if len(fields) == 9 and fields[-2] in {"T", "F"}:
            rows.append((int(fields[0]) - 1, int(fields[1]) - 1, fields[-2] == "T"))
    if exchange is not None:
        blocks.append((exchange, rows))

    count = cfg.post.prod.equil.prep.windows
    expected_exchanges = int(cfg.post.prod.cntrl["numexchg"])
    if [value for value, _ in blocks] != list(range(1, expected_exchanges + 1)):
        raise ValueError("Replica-exchange log is incomplete or non-sequential")
    if any(len(block) != count for _, block in blocks):
        raise ValueError("Replica-exchange log contains an incomplete exchange block")

    replica_to_walker = np.arange(count, dtype=int)
    walker_positions = np.empty((expected_exchanges, count), dtype=int)
    attempts: dict[tuple[int, int], int] = {}
    accepts: dict[tuple[int, int], int] = {}
    for row_index, (_, block) in enumerate(blocks):
        proposals: dict[tuple[int, int], list[bool]] = {}
        for source, target, accepted in block:
            pair = tuple(sorted((source, target)))
            proposals.setdefault(pair, []).append(accepted)
        for pair, flags in proposals.items():
            if len(flags) != 2 or flags[0] != flags[1]:
                raise ValueError(f"Inconsistent reciprocal exchange record for pair {pair}")
            attempts[pair] = attempts.get(pair, 0) + 1
            if flags[0]:
                accepts[pair] = accepts.get(pair, 0) + 1
                first, second = pair
                replica_to_walker[first], replica_to_walker[second] = (
                    replica_to_walker[second],
                    replica_to_walker[first],
                )
        inverse = np.empty(count, dtype=int)
        inverse[replica_to_walker] = np.arange(count)
        walker_positions[row_index] = inverse

    adjacent = [(index, index + 1) for index in range(count - 1)]
    wrap = (0, count - 1)
    edge_pairs = tuple(adjacent + ([wrap] if wrap in attempts else []))
    edge_attempts = np.asarray([attempts.get(pair, 0) for pair in edge_pairs], dtype=int)
    edge_accepts = np.asarray([accepts.get(pair, 0) for pair in edge_pairs], dtype=int)
    residence = np.empty((count, count), dtype=float)
    for walker in range(count):
        residence[walker] = [
            100.0 * np.mean(walker_positions[:, walker] == replica)
            for replica in range(count)
        ]

    exchange_ps = int(cfg.post.prod.cntrl["nstlim"]) * float(cfg.post.prod.cntrl["dt"])
    roundtrip_counts = np.zeros(count, dtype=int)
    roundtrip_mean = np.full(count, np.nan)
    reached_both = np.zeros(count, dtype=bool)
    for walker in range(count):
        series = walker_positions[:, walker]
        reached_both[walker] = bool(np.any(series == 0) and np.any(series == count - 1))
        events: list[tuple[int, int]] = []
        for step, position in enumerate(series, start=1):
            if position not in {0, count - 1}:
                continue
            if not events or position != events[-1][1]:
                events.append((step, int(position)))
        durations: list[float] = []
        start = 0
        while start + 2 < len(events):
            first, middle, last = events[start : start + 3]
            if first[1] == last[1] and first[1] != middle[1]:
                durations.append((last[0] - first[0]) * exchange_ps)
                start += 2
            else:
                start += 1
        roundtrip_counts[walker] = len(durations)
        if durations:
            roundtrip_mean[walker] = float(np.mean(durations))

    return ReplicaDiagnostics(
        exchanges=np.arange(1, expected_exchanges + 1),
        walker_positions=walker_positions,
        edge_pairs=edge_pairs,
        edge_attempts=edge_attempts,
        edge_accepts=edge_accepts,
        residence_percent=residence,
        roundtrip_counts=roundtrip_counts,
        roundtrip_mean_ps=roundtrip_mean,
        reached_both_endpoints=reached_both,
    )


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_numerical_outputs(
    cfg: RefepReportConfig,
    work: Path,
    grid: EnergyGrid,
    analysis: FreeEnergyAnalysis,
    uncertainties: dict[str, float],
    bar_uncertainty: np.ndarray,
    convergence: list[tuple[int, float, dict[str, float]]],
    replica: ReplicaDiagnostics,
) -> None:
    kbt = GAS_CONSTANT_KCAL * cfg.temperature_k
    rtln10 = math.log(10.0) * kbt
    labels = reporting_labels(cfg)
    _write_csv(
        work / "energy-matrix.csv",
        [
            "sampling_window",
            "sampling_lambda",
            "evaluation_window",
            "evaluation_lambda",
            "frame",
            "production_time_ps",
            "potential_energy_kcal_mol",
        ],
        [
            [
                sample,
                f"{grid.lambdas[sample]:.10f}",
                evaluation,
                f"{grid.lambdas[evaluation]:.10f}",
                frame + 1,
                f"{grid.time_ps[frame]:.6f}",
                f"{grid.energies[sample, evaluation, frame]:.8f}",
            ]
            for sample in range(grid.windows)
            for evaluation in range(grid.windows)
            for frame in range(grid.frames)
        ],
    )
    _write_csv(
        work / "neighbor-free-energies.csv",
        [
            "edge",
            "lambda_i",
            "lambda_j",
            "fep_forward_kcal_mol",
            "fep_reverse_kcal_mol",
            "fep_hysteresis_kcal_mol",
            "bar_kcal_mol",
            "bar_block_bootstrap_sd_kcal_mol",
            "work_histogram_bhattacharyya",
        ],
        [
            [
                index,
                f"{grid.lambdas[index]:.10f}",
                f"{grid.lambdas[index + 1]:.10f}",
                f"{analysis.forward_fep_edges[index]:.8f}",
                f"{analysis.reverse_fep_edges[index]:.8f}",
                f"{analysis.forward_fep_edges[index] - analysis.reverse_fep_edges[index]:.8f}",
                f"{analysis.bar_edges[index]:.8f}",
                f"{bar_uncertainty[index]:.8f}",
                f"{analysis.neighbor_overlap[index]:.8f}",
            ]
            for index in range(grid.windows - 1)
        ],
    )
    _write_csv(
        work / "ti-integrand.csv",
        [
            "window",
            "lambda",
            "mean_dU_dlambda_kcal_mol",
            "sample_sd_kcal_mol",
            "block_sem_kcal_mol",
            "quadratic_fit_rms_kcal_mol",
        ],
        [
            [
                index,
                f"{lam:.10f}",
                f"{analysis.ti_integrand[index]:.8f}",
                f"{analysis.ti_std[index]:.8f}",
                f"{analysis.ti_sem[index]:.8f}",
                f"{analysis.ti_fit_rms[index]:.8f}",
            ]
            for index, lam in enumerate(grid.lambdas)
        ],
    )
    notes = {
        "FEP forward": "Zwanzig sum over adjacent lambda windows, sampled from lambda_i",
        "FEP reverse": "sign-corrected reverse Zwanzig sum, reported lambda0 to lambda1",
        "BAR": "sum of adjacent equal-sample Bennett estimates",
        "TI charge derivative": "quadratic U(lambda) fit per configuration, integrated mean dU/dlambda",
        "TI legacy endpoint gap": "legacy integral of mean[U(1)-U(0)]; diagnostic for comparison",
        "MBAR": "full 16x16 cross-Hamiltonian energy matrix",
    }
    _write_csv(
        work / "free-energy-summary.csv",
        [
            "method",
            "direction",
            "delta_g_kcal_mol",
            "block_bootstrap_sd_kcal_mol",
            "delta_g_kBT",
            "delta_g_over_RTln10",
            "definition",
        ],
        [
            [
                method,
                f"{labels[0]} -> {labels[1]}",
                f"{value:.8f}",
                f"{uncertainties[method]:.8f}",
                f"{value / kbt:.8f}",
                f"{value / rtln10:.8f}",
                notes[method],
            ]
            for method, value in analysis.totals.items()
        ],
    )
    method_names = list(analysis.totals)
    _write_csv(
        work / "convergence.csv",
        ["frames", "production_time_ps", *method_names],
        [
            [frames, f"{time_ps:.6f}", *[f"{totals[name]:.8f}" for name in method_names]]
            for frames, time_ps, totals in convergence
        ],
    )
    _write_csv(
        work / "mbar-overlap-matrix.csv",
        ["sampling_window", "sampling_lambda", *[f"to_window_{i}" for i in range(grid.windows)]],
        [
            [index, f"{grid.lambdas[index]:.10f}", *[f"{value:.10f}" for value in row]]
            for index, row in enumerate(analysis.mbar_overlap)
        ],
    )
    _write_csv(
        work / "walker-replica.csv",
        ["exchange", "exchange_time_ps", "walker", "replica", "lambda"],
        [
            [
                int(replica.exchanges[exchange]),
                f"{replica.exchanges[exchange] * int(cfg.post.prod.cntrl['nstlim']) * float(cfg.post.prod.cntrl['dt']):.6f}",
                walker + 1,
                int(replica.walker_positions[exchange, walker]) + 1,
                f"{grid.lambdas[replica.walker_positions[exchange, walker]]:.10f}",
            ]
            for exchange in range(len(replica.exchanges))
            for walker in range(grid.windows)
        ],
    )
    _write_csv(
        work / "replica-acceptance.csv",
        ["replica_i", "lambda_i", "replica_j", "lambda_j", "wrap_edge", "attempts", "accepted", "acceptance_fraction"],
        [
            [
                first + 1,
                f"{grid.lambdas[first]:.10f}",
                second + 1,
                f"{grid.lambdas[second]:.10f}",
                first == 0 and second == grid.windows - 1,
                int(replica.edge_attempts[index]),
                int(replica.edge_accepts[index]),
                f"{replica.edge_accepts[index] / replica.edge_attempts[index]:.8f}",
            ]
            for index, (first, second) in enumerate(replica.edge_pairs)
        ],
    )
    _write_csv(
        work / "replica-summary.csv",
        ["walker", "reached_both_endpoints", "round_trips", "mean_roundtrip_time_ps", *[f"residence_window_{i}" for i in range(grid.windows)]],
        [
            [
                walker + 1,
                bool(replica.reached_both_endpoints[walker]),
                int(replica.roundtrip_counts[walker]),
                "" if not np.isfinite(replica.roundtrip_mean_ps[walker]) else f"{replica.roundtrip_mean_ps[walker]:.6f}",
                *[f"{value:.8f}" for value in replica.residence_percent[walker]],
            ]
            for walker in range(grid.windows)
        ],
    )
    if cfg.post.keep_mask is None:
        scientific_scope = [
            "Reported delta G is the raw periodic explicit-solvent charge-mutation free energy.",
            "No net-charge finite-size correction or proton standard-state term is applied.",
            "delta_g_over_RTln10 is an energy-equivalent shift, not an absolute pKa.",
            "The legacy endpoint-gap TI value is retained only for comparison with the earlier histidine analysis.",
        ]
    else:
        scientific_scope = [
            "Energies were rescored after retaining atoms selected by the configured keep_mask.",
            "The conformations were sampled under explicit periodic PME, but evaluated under the configured implicit-solvent Hamiltonian.",
            "BAR, MBAR, FEP, and TI values are diagnostic rescoring estimates, not rigorously sampled implicit-solvent free energies.",
            "No proton standard-state term is applied.",
            "delta_g_over_RTln10 is an energy-equivalent shift, not an absolute pKa.",
        ]
    summary = {
        "system": cfg.post.prod.equil.prep.system,
        "direction": f"{labels[0]} -> {labels[1]}",
        "reference_state": cfg.reference_state,
        "free_energy_definition": f"G({labels[1]}) - G({labels[0]})",
        "temperature_K": cfg.temperature_k,
        "windows": grid.windows,
        "frames_per_window": grid.frames,
        "production_time_ps": float(grid.time_ps[-1]),
        "bootstrap": {
            "samples": cfg.bootstrap_samples,
            "method": "independent circular moving-block resampling within each lambda ensemble",
            "block_frames": cfg.bootstrap_block_frames,
            "block_time_ps": cfg.bootstrap_block_frames * float(grid.time_ps[0]),
            "seed": cfg.bootstrap_seed,
        },
        "free_energies_kcal_mol": {
            name: {"estimate": value, "bootstrap_sd": uncertainties[name]}
            for name, value in analysis.totals.items()
        },
        "replica_exchange": {
            "walkers_reaching_both_endpoints": int(replica.reached_both_endpoints.sum()),
            "walkers": grid.windows,
            "complete_round_trips": int(replica.roundtrip_counts.sum()),
            "adjacent_acceptance_min": float(
                np.min(replica.edge_accepts[: grid.windows - 1] / replica.edge_attempts[: grid.windows - 1])
            ),
            "adjacent_acceptance_max": float(
                np.max(replica.edge_accepts[: grid.windows - 1] / replica.edge_attempts[: grid.windows - 1])
            ),
        },
        "scientific_scope": scientific_scope,
    }
    (work / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))


def _plot_free_energy_summary(
    cfg: RefepReportConfig,
    path: Path,
    grid: EnergyGrid,
    analysis: FreeEnergyAnalysis,
    uncertainties: dict[str, float],
    bar_uncertainty: np.ndarray,
    convergence: list[tuple[int, float, dict[str, float]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    edge_x = np.arange(grid.windows - 1)
    ax = axes[0, 0]
    ax.plot(edge_x, analysis.forward_fep_edges, "o-", label="FEP forward")
    ax.plot(edge_x, analysis.reverse_fep_edges, "s-", label="FEP reverse")
    ax.errorbar(edge_x, analysis.bar_edges, yerr=bar_uncertainty, fmt="^-", capsize=3, label="BAR ± block-bootstrap SD")
    ax.set(xlabel="Neighbor edge i → i+1", ylabel="ΔG (kcal/mol)", title="Neighbor free energies")
    ax.set_xticks(edge_x)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.errorbar(grid.lambdas, analysis.ti_integrand, yerr=analysis.ti_sem, fmt="o-", capsize=3, label="Quadratic charge-path derivative")
    ax.set(xlabel="λ", ylabel="⟨∂U/∂λ⟩ (kcal/mol)", title="Thermodynamic-integration integrand")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    selected = ["FEP forward", "FEP reverse", "BAR", "TI charge derivative", "MBAR"]
    for name in selected:
        ax.plot([row[1] for row in convergence], [row[2][name] for row in convergence], marker="o", ms=3.5, label=name)
    ax.set(xlabel="Production time used (ps)", ylabel="Cumulative ΔG (kcal/mol)", title="Estimator convergence")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    methods = list(analysis.totals)
    values = [analysis.totals[name] for name in methods]
    errors = [uncertainties[name] for name in methods]
    positions = np.arange(len(methods))
    ax.bar(positions, values, yerr=errors, capsize=4)
    ax.set_xticks(positions, [name.replace(" ", "\n", 1) for name in methods], rotation=25, ha="right")
    ax.set(ylabel="ΔG (kcal/mol)", title="Total λ0 → λ1 free energy")
    fig.suptitle(f"{cfg.post.prod.equil.prep.system} REFEP free-energy report", fontsize=16)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_neighbor_overlap(
    cfg: RefepReportConfig, path: Path, grid: EnergyGrid, analysis: FreeEnergyAnalysis
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    columns = 4
    rows = math.ceil((grid.windows - 1) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14.5, 2.9 * rows), constrained_layout=True)
    flat = np.atleast_1d(axes).ravel()
    for index in range(grid.windows - 1):
        ax = flat[index]
        forward = grid.energies[index, index + 1] - grid.energies[index, index]
        reverse_same = -(
            grid.energies[index + 1, index]
            - grid.energies[index + 1, index + 1]
        )
        low = min(float(forward.min()), float(reverse_same.min()))
        high = max(float(forward.max()), float(reverse_same.max()))
        bins = np.linspace(low, high, cfg.histogram_bins + 1)
        ax.hist(forward, bins=bins, density=True, alpha=0.55, label="Forward work")
        ax.hist(reverse_same, bins=bins, density=True, alpha=0.55, label="−reverse work")
        ax.set_title(
            f"λ {grid.lambdas[index]:.3f}–{grid.lambdas[index+1]:.3f}\nBC={analysis.neighbor_overlap[index]:.2f}"
        )
        ax.set_xlabel("Work (kcal/mol)")
        if index % columns == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    for ax in flat[grid.windows - 1 :]:
        ax.set_visible(False)
    fig.suptitle("Neighbor forward/reverse work overlap", fontsize=16)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_mbar_overlap(
    cfg: RefepReportConfig, path: Path, grid: EnergyGrid, analysis: FreeEnergyAnalysis
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    fig, ax = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    image = ax.imshow(analysis.mbar_overlap, origin="lower", cmap="magma", vmin=0.0)
    labels = [f"{value:.2f}" for value in grid.lambdas]
    ax.set_xticks(range(grid.windows), labels, rotation=45, ha="right")
    ax.set_yticks(range(grid.windows), labels)
    ax.set(xlabel="Destination λ", ylabel="Source λ", title="MBAR overlap matrix")
    fig.colorbar(image, ax=ax, label="Transition probability")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_replica_health(
    cfg: RefepReportConfig,
    path: Path,
    grid: EnergyGrid,
    replica: ReplicaDiagnostics,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, grid.windows))
    exchange_ps = int(cfg.post.prod.cntrl["nstlim"]) * float(cfg.post.prod.cntrl["dt"])
    time = replica.exchanges * exchange_ps
    ax = axes[0, 0]
    for walker in range(grid.windows):
        ax.plot(time, grid.lambdas[replica.walker_positions[:, walker]], color=colors[walker], lw=0.75, alpha=0.72)
    ax.set(xlabel="Production time (ps)", ylabel="λ", title="Walker diffusion across λ ladder")

    ax = axes[0, 1]
    image = ax.imshow(replica.residence_percent, aspect="auto", origin="lower", cmap="magma", vmin=0.0)
    ax.set_xticks(range(grid.windows), [f"{value:.2f}" for value in grid.lambdas], rotation=45, ha="right")
    ax.set_yticks(range(grid.windows), range(1, grid.windows + 1))
    ax.set(xlabel="λ replica", ylabel="Walker", title=f"Residence occupancy (ideal {100/grid.windows:.2f}%)")
    fig.colorbar(image, ax=ax, label="Time (%)")

    ax = axes[1, 0]
    acceptance = replica.edge_accepts / replica.edge_attempts
    labels = [f"{a+1}↔{b+1}" + (" wrap" if a == 0 and b == grid.windows - 1 else "") for a, b in replica.edge_pairs]
    colors_accept = ["#C43C39" if "wrap" in label else "#1F3A5F" for label in labels]
    ax.bar(np.arange(len(labels)), 100.0 * acceptance, color=colors_accept)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set(xlabel="Replica edge", ylabel="Accepted attempts (%)", title="Exchange acceptance")
    ax.set_ylim(0.0, max(70.0, 105.0 * float(acceptance.max())))

    ax = axes[1, 1]
    walkers = np.arange(1, grid.windows + 1)
    ax.bar(walkers, replica.roundtrip_counts, color=colors)
    ax.set_xticks(walkers)
    ax.set(
        xlabel="Walker",
        ylabel="Complete endpoint round trips",
        title=(
            f"Round trips: {int(replica.roundtrip_counts.sum())}; "
            f"both endpoints: {int(replica.reached_both_endpoints.sum())}/{grid.windows}"
        ),
    )
    fig.suptitle(f"{cfg.post.prod.equil.prep.system} H-REMD replica health", fontsize=16)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def collect_production_dihedrals(
    cfg: RefepReportConfig,
    work: Path,
) -> list[DihedralSample]:
    """Measure an optional equil.yaml dihedral in every production lambda ensemble."""
    if cfg.dihedral is None:
        return []
    prod = cfg.post.prod
    source = refep_prod_dir(prod)
    windows = prod.equil.prep.windows
    total_steps = int(prod.cntrl["nstlim"]) * int(prod.cntrl["numexchg"])
    expected_frames = total_steps // int(prod.cntrl["ntwx"])
    frame_dt = int(prod.cntrl["ntwx"]) * float(prod.cntrl["dt"])
    samples: list[DihedralSample] = []
    combined_inputs: list[str] = []
    combined_logs: list[str] = []
    for index in range(windows):
        stem = f"lambda-{index:03d}"
        input_path = work / f"prod-dihedral-{index:03d}.cpptraj.in"
        output_path = work / f"prod-dihedral-{index:03d}.dat"
        log_path = work / f"prod-dihedral-{index:03d}.cpptraj.log"
        input_text = render_dihedral_cpptraj_input(
            source / f"{stem}.parm7",
            source / f"{stem}.nc",
            output_path,
            cfg.dihedral,
        )
        input_path.write_text(input_text)
        run_cpptraj(
            input_path,
            log_path,
            prod.runtime.module,
            cfg.post.runtime.cpptraj_executable,
        )
        values = read_cpptraj_dihedral(output_path)
        if len(values) != expected_frames:
            raise ValueError(
                f"Production window {index:03d} has {len(values)} dihedral frames; "
                f"expected {expected_frames}"
            )
        lambda_value = index / (windows - 1)
        for frame, angle in values:
            samples.append(
                DihedralSample(
                    window=index,
                    lambda_value=lambda_value,
                    frame=frame,
                    time_ps=frame * frame_dt,
                    raw_deg=angle,
                    branch_deg=angle_near_target(angle, cfg.dihedral.target_deg),
                )
            )
        combined_inputs.extend([f"# {stem}", input_text])
        combined_logs.extend([f"===== {stem} =====", log_path.read_text()])
    (work / "prod-dihedral-cpptraj.in").write_text("\n".join(combined_inputs))
    (work / "prod-dihedral-cpptraj.log").write_text("\n".join(combined_logs))
    return samples


def generate_refep_report(
    cfg: RefepReportConfig, output_dir: Path | None = None
) -> tuple[Path, FreeEnergyAnalysis, dict[str, float], ReplicaDiagnostics]:
    grid = orient_grid_for_reporting(cfg, load_completed_energy_grid(cfg))
    analysis = analyze_energy_grid(cfg, grid)
    uncertainties, bar_uncertainty = bootstrap_free_energies(cfg, grid)
    convergence = convergence_series(cfg, grid)
    replica = orient_replica_for_reporting(cfg, parse_rem_log(cfg))
    destination = output_dir.resolve() if output_dir else refep_report_dir(cfg)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".refep-report-", dir=destination.parent) as tmp:
        work = Path(tmp)
        dihedrals = collect_production_dihedrals(cfg, work)
        _write_numerical_outputs(
            cfg,
            work,
            grid,
            analysis,
            uncertainties,
            bar_uncertainty,
            convergence,
            replica,
        )
        _plot_free_energy_summary(
            cfg,
            work / "free-energy-summary.png",
            grid,
            analysis,
            uncertainties,
            bar_uncertainty,
            convergence,
        )
        _plot_neighbor_overlap(
            cfg, work / "neighbor-work-overlap.png", grid, analysis
        )
        _plot_mbar_overlap(cfg, work / "mbar-overlap.png", grid, analysis)
        _plot_replica_health(
            cfg, work / "replica-exchange-health.png", grid, replica
        )
        if cfg.dihedral is not None:
            write_dihedral_csv(work / "prod-dihedral.csv", dihedrals)
            plot_dihedral_timeseries(
                work / "prod-dihedral.png",
                cfg.style,
                cfg.post.prod.equil.prep.system,
                "production",
                cfg.dihedral,
                cfg.post.prod.equil.prep.windows,
                dihedrals,
                scatter=True,
            )
            summary_path = work / "summary.yaml"
            summary = yaml.safe_load(summary_path.read_text())
            summary["dihedral"] = {
                "label": cfg.dihedral.label,
                "atom_ids_one_based": list(cfg.dihedral.atom_ids),
                "target_deg": cfg.dihedral.target_deg,
                "samples": len(dihedrals),
                "source": "fixed-lambda production trajectories",
            }
            summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
        (work / "refep-report-spec.yaml").write_text(cfg.post.yaml_path.read_text())
        destination.mkdir(parents=True, exist_ok=True)
        for source in work.iterdir():
            source.replace(destination / source.name)
    return destination, analysis, uncertainties, replica


def run_refep_prod_report(yaml_path: Path) -> None:
    cfg = load_refep_report_config(yaml_path)
    destination, analysis, uncertainties, replica = generate_refep_report(cfg)
    labels = reporting_labels(cfg)
    print(f"OK: wrote REFEP report to {destination}")
    print(f"Direction: {labels[0]} -> {labels[1]}")
    for name in ("BAR", "TI charge derivative", "MBAR"):
        print(
            f"  {name}: {analysis.totals[name]:+.4f} ± "
            f"{uncertainties[name]:.4f} kcal/mol (block-bootstrap SD)"
        )
    adjacent = replica.edge_accepts[: cfg.post.prod.equil.prep.windows - 1] / replica.edge_attempts[: cfg.post.prod.equil.prep.windows - 1]
    print(
        f"Replica health: adjacent acceptance {adjacent.min():.3f}–{adjacent.max():.3f}; "
        f"{int(replica.reached_both_endpoints.sum())}/{len(replica.reached_both_endpoints)} "
        f"walkers reached both endpoints; {int(replica.roundtrip_counts.sum())} round trips"
    )
    if cfg.post.keep_mask is None:
        print("NOTE: raw charge-mutation ΔG; no net-charge finite-size/pKa-cycle correction")
    else:
        print(
            "NOTE: implicit-solvent rescoring of explicit-solvent ensembles; "
            "diagnostic, not a rigorously sampled implicit-solvent ΔG"
        )
    if cfg.dihedral is not None:
        print(
            f"OK: plotted {cfg.dihedral.label} production time series for "
            f"{cfg.post.prod.equil.prep.windows} lambda windows"
        )
