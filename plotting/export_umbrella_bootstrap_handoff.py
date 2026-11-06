"""Export block-bootstrap uncertainty data into the Elisa PMF handoff folders."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qmmd.us_lcod import lcod_centers
from qmmd.us_lcod_wham import (
    equivalent_delta_pka,
    load_config as load_lcod_config,
    pmf_features,
)
from qmmd.us_wham import (
    circular_block_resample,
    compute_thermodynamics,
    dense_hermite_curve,
    load_config as load_torsion_config,
    read_wham_result,
    resolve_executable,
    run_wham,
    smooth_pmf,
    zero_pmf,
)


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "elisa" / "umbrella-sampling"


@dataclass(frozen=True, slots=True)
class LCODExport:
    package: str
    config: Path
    source: Path


@dataclass(frozen=True, slots=True)
class TorsionExport:
    package: str
    config: Path
    source: Path


LCOD_EXPORTS = (
    LCODExport("MAL-LCOD", ROOT / "configs/MAH/us/wham.yaml", ROOT / "systems/MAH/solv_5.5/us-lcod/wham-equil"),
    LCODExport("APP-BPP-LCOD", ROOT / "configs/APP/us/wham.yaml", ROOT / "systems/APP/solv_4.0/us-lcod/wham-equil"),
    LCODExport("BPP-CPP-LCOD", ROOT / "configs/CPP/us/wham.yaml", ROOT / "systems/CPP/solv_4.0/us-lcod/wham-equil"),
    LCODExport("CPP-DPP-LCOD", ROOT / "configs/DPP/us/wham.yaml", ROOT / "systems/DPP/solv_4.0/us-lcod/wham-equil"),
)

TORSION_EXPORTS = (
    TorsionExport(
        "PRN-SYN-ANTI",
        ROOT / "configs/PRN-anti/us/wham.yaml",
        ROOT / "systems/PRN-anti/solv_5.5/us-pull/wham",
    ),
    TorsionExport(
        "BV_rot",
        ROOT / "configs/BV/us/wham-dih.yaml",
        ROOT / "systems/BV/solv_4.0/us-single15/wham-dih",
    ),
    TorsionExport(
        "APP_rot",
        ROOT / "configs/APP/us/wham-dih.yaml",
        ROOT / "systems/APP/solv_4.0/us-single15/wham-dih-equil",
    ),
    TorsionExport(
        "CPP_rot",
        ROOT / "configs/CPP/us/wham-dih.yaml",
        ROOT / "systems/CPP/solv_4.0/us-single15/wham-dih",
    ),
    TorsionExport(
        "BPP_rot",
        ROOT / "configs/BPP/us/wham-dih.yaml",
        ROOT / "systems/BPP/solv_4.0/us-single15/wham-dih-equil",
    ),
    TorsionExport(
        "DPP_rot",
        ROOT / "configs/DPP/us/wham-dih-partial-w000.yaml",
        ROOT / "systems/DPP/solv_4.0/us-single15/wham-dih-equil-partial-w000",
    ),
)


def write_pmf_exports(
    package: Path,
    coordinate_name: str,
    grid: np.ndarray,
    original: np.ndarray,
    replicas: list[np.ndarray],
    confidence: float,
) -> None:
    boot = np.vstack(replicas)
    alpha = 1.0 - confidence
    lower = np.quantile(boot, alpha / 2.0, axis=0)
    upper = np.quantile(boot, 1.0 - alpha / 2.0, axis=0)
    data = package / "data"
    with (data / "bootstrap_pmfs.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bootstrap_replica", coordinate_name, "pmf_kcal_mol"])
        for index, pmf in enumerate(boot, start=1):
            writer.writerows(
                (index, f"{coordinate:.10g}", f"{energy:.10g}")
                for coordinate, energy in zip(grid, pmf, strict=True)
            )
    with (data / "pmf_bootstrap_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                coordinate_name,
                "pmf_kcal_mol",
                "bootstrap_mean_kcal_mol",
                "bootstrap_std_kcal_mol",
                "ci_lower_kcal_mol",
                "ci_upper_kcal_mol",
                "bootstrap_replicates",
            ]
        )
        for values in zip(
            grid,
            original,
            np.mean(boot, axis=0),
            np.std(boot, axis=0, ddof=1),
            lower,
            upper,
            strict=True,
        ):
            writer.writerow([*(f"{value:.10g}" for value in values), len(boot)])


def write_feature_exports(
    package: Path,
    original: dict[str, tuple[float, str]],
    replicas: list[dict[str, tuple[float, str]]],
    confidence: float,
) -> None:
    data = package / "data"
    names = list(original)
    with (data / "bootstrap_features.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bootstrap_replica", *names])
        for index, replica in enumerate(replicas, start=1):
            writer.writerow([index, *(f"{replica[name][0]:.10g}" for name in names)])
    alpha = 1.0 - confidence
    with (data / "bootstrap_feature_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "feature",
                "unit",
                "original",
                "bootstrap_mean",
                "bootstrap_std",
                "ci_lower",
                "ci_upper",
                "bootstrap_replicates",
            ]
        )
        for name in names:
            values = np.asarray([replica[name][0] for replica in replicas])
            writer.writerow(
                [
                    name,
                    original[name][1],
                    f"{original[name][0]:.10g}",
                    f"{np.mean(values):.10g}",
                    f"{np.std(values, ddof=1):.10g}",
                    f"{np.quantile(values, alpha / 2.0):.10g}",
                    f"{np.quantile(values, 1.0 - alpha / 2.0):.10g}",
                    len(values),
                ]
            )


def lcod_features(
    grid: np.ndarray, pmf: np.ndarray, cfg: object
) -> dict[str, tuple[float, str]]:
    dense_x, dense_y = dense_hermite_curve(grid, smooth_pmf(pmf, cfg.smoothing_penalty))
    values = pmf_features(
        dense_x,
        dense_y,
        cfg.negative_basin_angstrom,
        cfg.positive_basin_angstrom,
    )
    negative_x, negative_y = values["negative_minimum"]
    positive_x, positive_y = values["positive_minimum"]
    barrier_x, barrier_y = values["barrier_maximum"]
    return {
        "negative_minimum_lcod": (negative_x, "angstrom"),
        "negative_minimum_pmf": (negative_y, "kcal mol^-1"),
        "positive_minimum_lcod": (positive_x, "angstrom"),
        "positive_minimum_pmf": (positive_y, "kcal mol^-1"),
        "barrier_maximum_lcod": (barrier_x, "angstrom"),
        "barrier_maximum_pmf": (barrier_y, "kcal mol^-1"),
        "gap_positive_minus_negative": (
            float(values["gap_positive_minus_negative"]),
            "kcal mol^-1",
        ),
        "barrier_negative_to_positive": (
            float(values["barrier_negative_to_positive"]),
            "kcal mol^-1",
        ),
        "barrier_positive_to_negative": (
            float(values["barrier_positive_to_negative"]),
            "kcal mol^-1",
        ),
        "equivalent_delta_pka_positive_minus_negative": (
            equivalent_delta_pka(
                float(values["gap_positive_minus_negative"]), cfg.temperature_k
            ),
            "pKa unit",
        ),
    }


def read_lcod_original(source: Path, cfg: object) -> tuple[np.ndarray, np.ndarray]:
    grid: list[float] = []
    pmf: list[float] = []
    with (source / "pmf.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["in_report_range"]):
                grid.append(float(row["lcod_A"]))
                pmf.append(float(row["pmf_kcal_mol"]))
    result_grid = np.asarray(grid)
    result_pmf = np.asarray(pmf)
    if len(result_grid) < 3 or not np.all(np.isfinite(result_pmf)):
        raise ValueError(f"Invalid original LCOD PMF in {source}")
    return result_grid, result_pmf


def bootstrap_lcod(
    item: LCODExport,
    replicas: int,
    block_length_ps: float,
    seed: int,
    confidence: float,
    minimum_success_fraction: float,
) -> int:
    cfg = load_lcod_config(item.config)
    centers = lcod_centers(cfg.equil.pull.windows)
    series = [np.loadtxt(item.source / f"window-{index:03d}.dat") for index in range(len(centers))]
    intervals = np.asarray([np.median(np.diff(values[:, 0])) for values in series])
    if not np.allclose(intervals, intervals[0], rtol=1.0e-7, atol=1.0e-9):
        raise ValueError(f"Inconsistent sampling intervals in {item.source}")
    block_frames = int(round(block_length_ps / intervals[0]))
    if not math.isclose(block_frames * intervals[0], block_length_ps, abs_tol=1.0e-8):
        raise ValueError("Block length is not divisible by the LCOD sampling interval")
    if any(block_frames > len(values) // 2 for values in series):
        raise ValueError(f"Block length leaves fewer than two blocks in {item.package}")

    force = [float(line.split()[2]) for line in (item.source / "metadata.dat").read_text().splitlines()]
    if len(force) != len(centers) or not np.allclose(force, force[0]):
        raise ValueError(f"Inconsistent WHAM force constants in {item.source}")
    original_grid, original = read_lcod_original(item.source, cfg)
    full_grid, _ = read_wham_result(item.source / "wham_result.dat")
    mask = (full_grid >= cfg.report_minimum_angstrom) & (full_grid <= cfg.report_maximum_angstrom)
    if not np.array_equal(full_grid[mask], original_grid):
        raise ValueError(f"Original/report WHAM grids differ in {item.source}")

    executable = resolve_executable(cfg.executable)
    rng = np.random.default_rng(seed)
    successful: list[np.ndarray] = []
    with tempfile.TemporaryDirectory(prefix=f"qmmd-{item.package.lower()}-") as temporary:
        work = Path(temporary)
        metadata = work / "metadata.dat"
        for replica in range(replicas):
            with metadata.open("w") as stream:
                for index, (center, values) in enumerate(zip(centers, series, strict=True)):
                    resampled = circular_block_resample(values[:, 1], block_frames, rng)
                    name = f"window-{index:03d}.dat"
                    np.savetxt(
                        work / name,
                        np.column_stack((values[:, 0], resampled)),
                        fmt="%.8f %.10f",
                    )
                    stream.write(f"{name} {center:.8f} {force[0]:.12g}\n")
            result = work / "result.dat"
            command = [
                str(executable),
                f"{cfg.minimum_angstrom:g}",
                f"{cfg.maximum_angstrom:g}",
                str(cfg.bins),
                f"{cfg.tolerance:g}",
                f"{cfg.temperature_k:g}",
                str(cfg.padding),
                str(metadata),
                str(result),
            ]
            try:
                run_wham(command, work, work / "wham.log")
                replica_grid, replica_raw = read_wham_result(result)
                if not np.array_equal(replica_grid, full_grid):
                    raise ValueError("Bootstrap WHAM grid differs from original")
                reported = replica_raw[mask]
                if not np.all(np.isfinite(reported)):
                    raise ValueError("non-finite reporting bin")
                successful.append(zero_pmf(original_grid, reported, cfg.zero_reference_angstrom))
            except (RuntimeError, ValueError) as exc:
                print(f"WARNING {item.package} replica {replica + 1}: {exc}")
    required = math.ceil(replicas * minimum_success_fraction)
    if len(successful) < required:
        raise RuntimeError(f"Only {len(successful)}/{replicas} replicas succeeded for {item.package}")

    package = HANDOFF / item.package
    write_pmf_exports(package, "lcod_A", original_grid, original, successful, confidence)
    original_features = lcod_features(original_grid, original, cfg)
    bootstrap_features = [lcod_features(original_grid, pmf, cfg) for pmf in successful]
    write_feature_exports(package, original_features, bootstrap_features, confidence)
    update_metadata(
        package,
        replicas,
        len(successful),
        block_length_ps,
        seed,
        confidence,
        cfg.zero_reference_angstrom,
        "independent circular moving-block resampling within each umbrella window",
    )
    print(f"{item.package}: {len(successful)}/{replicas} LCOD bootstrap replicas")
    return len(successful)


def export_torsion(item: TorsionExport, confidence: float) -> int:
    package = HANDOFF / item.package
    source = item.source
    cfg = load_torsion_config(item.config)
    shutil.copy2(source / "bootstrap_pmfs.csv", package / "data/bootstrap_pmfs.csv")
    shutil.copy2(source / "pmf.csv", package / "data/pmf.csv")
    shutil.copy2(source / "pmf_smooth.csv", package / "data/pmf_smooth.csv")
    shutil.copy2(source / "thermodynamics.csv", package / "data/thermodynamics.csv")
    grouped: dict[int, list[tuple[float, float]]] = {}
    with (source / "bootstrap_pmfs.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault(int(row["bootstrap_replica"]), []).append(
                (float(row["dihedral_deg"]), float(row["pmf_kcal_mol"]))
            )
    ordered = [grouped[index] for index in sorted(grouped)]
    grid = np.asarray([value[0] for value in ordered[0]])
    boot = [np.asarray([value[1] for value in replica]) for replica in ordered]
    with (source / "pmf.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    source_grid = np.asarray([float(row["dihedral_deg"]) for row in rows])
    original = np.asarray([float(row["pmf_kcal_mol"]) for row in rows])
    if not np.array_equal(grid, source_grid):
        raise ValueError(f"{item.package} bootstrap/original grids differ")
    write_pmf_exports(package, "dihedral_deg", grid, original, boot, confidence)
    original_features = compute_thermodynamics(grid, original, cfg)
    bootstrap_features = [compute_thermodynamics(grid, pmf, cfg) for pmf in boot]
    write_feature_exports(package, original_features, bootstrap_features, confidence)
    update_metadata(
        package,
        cfg.bootstrap.replicas,
        len(boot),
        cfg.bootstrap.block_length_ps,
        cfg.bootstrap.seed,
        confidence,
        cfg.zero_reference_deg,
        "independent circular moving-block resampling within each umbrella window",
    )
    print(f"{item.package}: exported {len(boot)} existing bootstrap replicas")
    return len(boot)


def update_metadata(
    package: Path,
    requested: int,
    successful: int,
    block_length_ps: float,
    seed: int,
    confidence: float,
    zero_reference: tuple[float, float],
    method: str,
) -> None:
    path = package / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["bootstrap"] = {
        "method": method,
        "block_length_ps": block_length_ps,
        "requested_replicas": requested,
        "successful_replicas": successful,
        "seed": seed,
        "confidence_level": confidence,
        "pmf_zero_reference": list(zero_reference),
        "pmf_zero_note": "Every replica was shifted independently to this same coordinate interval before pointwise statistics.",
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=500)
    parser.add_argument("--block-length-ps", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-success-fraction", type=float, default=0.90)
    parser.add_argument("--only", choices=["all", "torsion", "lcod"], default="all")
    parser.add_argument(
        "--package",
        choices=[item.package for item in (*TORSION_EXPORTS, *LCOD_EXPORTS)],
        help="Process one handoff package (useful for independent runs).",
    )
    args = parser.parse_args()
    if args.replicas < 2 or args.block_length_ps <= 0:
        parser.error("replicas must be >=2 and block length must be positive")
    if not 0 < args.confidence < 1 or not 0 < args.minimum_success_fraction <= 1:
        parser.error("confidence and minimum success fraction must lie in (0,1]")
    return args


def main() -> None:
    args = parse_args()
    if args.only in {"all", "torsion"}:
        for item in TORSION_EXPORTS:
            if args.package and item.package != args.package:
                continue
            export_torsion(item, args.confidence)
    if args.only in {"all", "lcod"}:
        for item in LCOD_EXPORTS:
            if args.package and item.package != args.package:
                continue
            bootstrap_lcod(
                item,
                args.replicas,
                args.block_length_ps,
                args.seed,
                args.confidence,
                args.minimum_success_fraction,
            )


if __name__ == "__main__":
    main()
