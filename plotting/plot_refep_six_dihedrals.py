#!/usr/bin/env python3
"""Audit the six bilin bridge dihedrals across REFEP lambda ensembles.

The configured atom order makes ZZZsss correspond to approximately zero
degrees for all six torsions.  Every plotted point is retained in the aligned
CSV so the conformational audit is reproducible rather than figure-only.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmmd.refep_prod import load_refep_prod_config, refep_prod_dir


DIHEDRALS: tuple[tuple[str, tuple[str, str, str, str]], ...] = (
    ("double5", ("NA", "C4A", "C5", "C1B")),
    ("single5", ("NB", "C1B", "C5", "C4A")),
    ("single10", ("NB", "C4B", "C10", "C1C")),
    ("double10", ("C4B", "C10", "C1C", "NC")),
    ("single15", ("NC", "C4C", "C15", "C1D")),
    ("double15", ("C4C", "C15", "C1D", "ND")),
)


@dataclass(frozen=True, slots=True)
class Sample:
    calculation: str
    yaml_path: Path
    window: int
    lambda_value: float
    frame: int
    time_ps: float
    angles_deg: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prod-yaml",
        type=Path,
        action="append",
        required=True,
        help="REFEP prod.yaml; repeat once per transformation",
    )
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="Short transformation label matching each --prod-yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--residue",
        type=int,
        default=1,
        help="One-based TPS/BLA topology residue number (default: 1)",
    )
    parser.add_argument(
        "--target-deg",
        type=float,
        default=0.0,
        help="ZZZsss target for every ordered torsion in degrees (default: 0)",
    )
    parser.add_argument(
        "--retention-cutoff-deg",
        type=float,
        default=45.0,
        help="Maximum circular distance from target counted as retained (default: 45)",
    )
    parser.add_argument(
        "--cpptraj",
        type=Path,
        default=Path("/nas/sycamore/apps/amber/26/ambertools26/bin/cpptraj"),
    )
    parser.add_argument(
        "--style",
        type=Path,
        default=ROOT / "plotting/lefteris.mplstyle",
    )
    return parser.parse_args()


def circular_branch(values_deg: np.ndarray, target_deg: float) -> np.ndarray:
    """Return angles on the target-centered branch [target-180, target+180)."""
    return target_deg + (values_deg - target_deg + 180.0) % 360.0 - 180.0


def circular_mean_deg(values_deg: np.ndarray) -> float:
    radians = np.deg2rad(values_deg)
    return float(
        np.rad2deg(
            np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
        )
    )


def cpptraj_input(
    topology: Path,
    trajectory: Path,
    output: Path,
    residue: int,
) -> str:
    lines = [f"parm {topology}", f"trajin {trajectory}"]
    for label, names in DIHEDRALS:
        masks = " ".join(f":{residue}@{name}" for name in names)
        lines.append(f"dihedral {label} {masks} out {output}")
    lines.extend(["run", "quit", ""])
    return "\n".join(lines)


def read_cpptraj_output(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#", ndmin=2)
    expected_columns = len(DIHEDRALS) + 1
    if data.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} cpptraj columns in {path}, found {data.shape[1]}"
        )
    return data[:, 0].astype(int), np.asarray(data[:, 1:], dtype=float)


def collect_one(
    yaml_path: Path,
    label: str,
    output_dir: Path,
    residue: int,
    cpptraj: Path,
) -> list[Sample]:
    cfg = load_refep_prod_config(yaml_path.resolve())
    source = refep_prod_dir(cfg)
    windows = cfg.equil.prep.windows
    frame_dt_ps = float(cfg.cntrl["ntwx"]) * float(cfg.cntrl["dt"])
    expected_frames = (
        int(cfg.cntrl["nstlim"])
        * int(cfg.cntrl["numexchg"])
        // int(cfg.cntrl["ntwx"])
    )
    work = output_dir / "cpptraj" / label.lower()
    work.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    for window in range(windows):
        stem = f"lambda-{window:03d}"
        topology = source / f"{stem}.parm7"
        trajectory = source / f"{stem}.nc"
        if not topology.is_file() or not trajectory.is_file():
            raise FileNotFoundError(
                f"Missing REFEP production pair: {topology}, {trajectory}"
            )
        data_path = work / f"{stem}-six-dihedrals.dat"
        input_path = work / f"{stem}-six-dihedrals.in"
        log_path = work / f"{stem}-six-dihedrals.log"
        input_path.write_text(
            cpptraj_input(topology, trajectory, data_path, residue),
            encoding="utf-8",
        )
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                [str(cpptraj), "-i", input_path.name],
                cwd=work,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        frames, angles = read_cpptraj_output(data_path)
        if len(frames) != expected_frames:
            raise ValueError(
                f"{label} window {window:03d} has {len(frames)} frames; "
                f"expected {expected_frames}"
            )
        lambda_value = window / (windows - 1)
        samples.extend(
            Sample(
                calculation=label,
                yaml_path=yaml_path.resolve(),
                window=window,
                lambda_value=lambda_value,
                frame=int(frame),
                time_ps=float(frame) * frame_dt_ps,
                angles_deg=tuple(float(value) for value in row),
            )
            for frame, row in zip(frames, angles, strict=True)
        )
    return samples


def write_samples(path: Path, samples: list[Sample], target_deg: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        labels = [label for label, _ in DIHEDRALS]
        writer.writerow(
            [
                "calculation",
                "prod_yaml",
                "window",
                "lambda",
                "frame",
                "time_ps",
                *[f"{label}_raw_deg" for label in labels],
                *[f"{label}_target_branch_deg" for label in labels],
            ]
        )
        for sample in samples:
            raw = np.asarray(sample.angles_deg)
            branch = circular_branch(raw, target_deg)
            writer.writerow(
                [
                    sample.calculation,
                    sample.yaml_path,
                    sample.window,
                    f"{sample.lambda_value:.8f}",
                    sample.frame,
                    f"{sample.time_ps:.8f}",
                    *[f"{value:.8f}" for value in raw],
                    *[f"{value:.8f}" for value in branch],
                ]
            )


def write_summary(
    path: Path,
    samples: list[Sample],
    labels: list[str],
    target_deg: float,
    cutoff_deg: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "calculation",
                "dihedral",
                "window",
                "lambda",
                "frames",
                "circular_mean_deg",
                "mean_abs_target_deviation_deg",
                "retained_within_cutoff_fraction",
                "target_deg",
                "cutoff_deg",
            ]
        )
        for calculation in labels:
            selected = [sample for sample in samples if sample.calculation == calculation]
            windows = sorted({sample.window for sample in selected})
            for column, (dihedral, _names) in enumerate(DIHEDRALS):
                for window in windows:
                    rows = [sample for sample in selected if sample.window == window]
                    raw = np.asarray([sample.angles_deg[column] for sample in rows])
                    branch = circular_branch(raw, target_deg)
                    deviations = np.abs(branch - target_deg)
                    writer.writerow(
                        [
                            calculation,
                            dihedral,
                            window,
                            f"{rows[0].lambda_value:.8f}",
                            len(rows),
                            f"{circular_mean_deg(raw):.8f}",
                            f"{deviations.mean():.8f}",
                            f"{np.mean(deviations <= cutoff_deg):.8f}",
                            f"{target_deg:.8f}",
                            f"{cutoff_deg:.8f}",
                        ]
                    )


def plot_samples(
    destination: Path,
    samples: list[Sample],
    labels: list[str],
    target_deg: float,
    cutoff_deg: float,
    style: Path,
) -> None:
    plt.style.use(style)
    figure, axes = plt.subplots(
        len(DIHEDRALS),
        len(labels),
        figsize=(4.2 * len(labels), 2.25 * len(DIHEDRALS)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    rng = np.random.default_rng(20261027)
    for column, calculation in enumerate(labels):
        selected = [sample for sample in samples if sample.calculation == calculation]
        for row, (dihedral, names) in enumerate(DIHEDRALS):
            axis = axes[row, column]
            raw = np.asarray([sample.angles_deg[row] for sample in selected])
            branch = circular_branch(raw, target_deg)
            lambdas = np.asarray([sample.lambda_value for sample in selected])
            jitter = rng.uniform(-0.006, 0.006, size=len(lambdas))
            axis.scatter(
                lambdas + jitter,
                branch,
                s=2.0,
                alpha=0.16,
                color="#1F3A5F",
                linewidths=0,
                rasterized=True,
            )
            windows = sorted({sample.window for sample in selected})
            lambda_grid: list[float] = []
            means: list[float] = []
            retained: list[float] = []
            for window in windows:
                values = np.asarray(
                    [
                        sample.angles_deg[row]
                        for sample in selected
                        if sample.window == window
                    ]
                )
                branch_values = circular_branch(values, target_deg)
                lambda_grid.append(window / (len(windows) - 1))
                means.append(float(np.mean(branch_values)))
                retained.append(
                    float(np.mean(np.abs(branch_values - target_deg) <= cutoff_deg))
                )
            axis.plot(lambda_grid, means, color="#C43C39", marker="o", ms=2.7, lw=1.1)
            axis.axhspan(
                target_deg - cutoff_deg,
                target_deg + cutoff_deg,
                color="#2E9E7E",
                alpha=0.09,
            )
            axis.axhline(target_deg, color="#2E9E7E", ls="--", lw=0.9)
            if row == 0:
                axis.set_title(calculation, fontweight="bold")
            if column == 0:
                axis.set_ylabel(
                    f"{dihedral} (°)\n" + "–".join(names),
                    fontsize=9,
                )
            if row == len(DIHEDRALS) - 1:
                axis.set_xlabel("λ")
            axis.set_xlim(-0.03, 1.03)
            axis.set_ylim(target_deg - 185.0, target_deg + 185.0)
            axis.set_yticks([target_deg - 180.0, target_deg, target_deg + 180.0])
            axis.text(
                0.985,
                0.91,
                f"min retained: {100.0 * min(retained):.1f}%",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.0,
            )
    figure.suptitle(
        f"REFEP locked-helix audit: six ZZZsss torsions\n"
        f"target {target_deg:g}°; shaded retention band ±{cutoff_deg:g}°",
        fontsize=16,
    )
    figure.savefig(destination, dpi=300)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if len(args.prod_yaml) != len(args.label):
        raise ValueError("Provide exactly one --label for every --prod-yaml")
    if args.residue < 1:
        raise ValueError("--residue is one-based and must be positive")
    if not 0.0 < args.retention_cutoff_deg <= 180.0:
        raise ValueError("--retention-cutoff-deg must be in (0, 180]")
    if not args.cpptraj.is_file() and shutil.which(str(args.cpptraj)) is None:
        raise FileNotFoundError(f"cpptraj executable not found: {args.cpptraj}")
    if not args.style.is_file():
        raise FileNotFoundError(f"Matplotlib style not found: {args.style}")

    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    all_samples: list[Sample] = []
    for yaml_path, label in zip(args.prod_yaml, args.label, strict=True):
        print(f"Auditing {label}: {yaml_path}", flush=True)
        all_samples.extend(
            collect_one(
                yaml_path,
                label,
                destination,
                args.residue,
                args.cpptraj,
            )
        )
    write_samples(destination / "refep-six-dihedrals.csv", all_samples, args.target_deg)
    write_summary(
        destination / "refep-six-dihedrals-summary.csv",
        all_samples,
        args.label,
        args.target_deg,
        args.retention_cutoff_deg,
    )
    plot_samples(
        destination / "refep-six-dihedrals.png",
        all_samples,
        args.label,
        args.target_deg,
        args.retention_cutoff_deg,
        args.style,
    )
    print(f"Wrote REFEP six-dihedral audit to {destination}")


if __name__ == "__main__":
    main()
