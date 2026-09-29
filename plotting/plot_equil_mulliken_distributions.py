#!/usr/bin/env python3
"""Plot pooled atomic Mulliken-charge distributions from replicate DFTB runs.

Each ``--dataset`` supplies a label, a directory containing ``run-*``
subdirectories, a one-based target atom ID, and a display atom label.  The
script reads ``<runs_path>/run-N/<run_dir>/mulliken`` for the selected runs,
sums all orbital contributions reported for the target atom in every available
Mulliken frame, and pools the replicate samples into one normalized P(q).  An
optional comparison system overlays the same one-based atoms from a second set
of replicate trajectories.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


TIME_RE = re.compile(
    r"\*\*\*\s+AT\s+T=\s*([+\-0-9.EeDd]+)\s*FSEC", re.IGNORECASE
)


@dataclass(frozen=True)
class Dataset:
    label: str
    runs_path: Path
    atom_id: int
    atom_label: str


@dataclass(frozen=True)
class ChargeSamples:
    dataset: Dataset
    run_ids: np.ndarray
    times_ps: np.ndarray
    charges_e: np.ndarray
    poisoned_run_id: int | None = None
    poisoned_keep_until_ps: float | None = None
    excluded_frames: int = 0


@dataclass(frozen=True)
class PoisonedRun:
    dataset_label: str
    run_id: int
    keep_until_ps: float


@dataclass(frozen=True)
class ComparisonSystem:
    label: str
    runs_path: Path


DEFAULT_DATASETS = (
    "APP,systems/APP/solv_4.0/dftb/N1T64C1,1,NA",
    "BPP,systems/BPP/solv_4.0/dftb/N1T64C1,19,NB",
    "DPP,systems/DPP/solv_4.0/dftb/N1T64C1,43,ND",
    "CPP,systems/CPP/solv_4.0/dftb/N1T64C1,31,NC",
)
DEFAULT_POISONED_RUN = "APP,8,20.0"
DEFAULT_COMPARISON_SYSTEM = "BV,systems/BV/solv_4.0/dftb/N1T48C1"


def parse_dataset(raw: str) -> Dataset:
    """Parse LABEL,RUNS_PATH,ATOM_ID,ATOM_LABEL from the command line."""
    parts = [part.strip() for part in raw.split(",", maxsplit=3)]
    if len(parts) != 4 or not all(parts):
        raise argparse.ArgumentTypeError(
            "dataset must be LABEL,RUNS_PATH,ATOM_ID,ATOM_LABEL"
        )
    try:
        atom_id = int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ATOM_ID must be an integer") from exc
    if atom_id < 1:
        raise argparse.ArgumentTypeError("ATOM_ID is one-based and must be positive")
    return Dataset(parts[0], Path(parts[1]), atom_id, parts[3])


def parse_run_ids(raw: str) -> tuple[int, ...]:
    """Parse comma-separated run IDs and inclusive ranges such as 1-10,15."""
    result: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", maxsplit=1)
            start, stop = int(left), int(right)
            if start < 1 or stop < start:
                raise argparse.ArgumentTypeError(f"invalid run range: {token}")
            result.extend(range(start, stop + 1))
        else:
            run_id = int(token)
            if run_id < 1:
                raise argparse.ArgumentTypeError("run IDs must be positive")
            result.append(run_id)
    if not result:
        raise argparse.ArgumentTypeError("at least one run ID is required")
    return tuple(dict.fromkeys(result))


def parse_poisoned_run(raw: str) -> PoisonedRun:
    """Parse DATASET_LABEL,RUN_ID,KEEP_UNTIL_PS truncation metadata."""
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "poisoned run must be DATASET_LABEL,RUN_ID,KEEP_UNTIL_PS"
        )
    try:
        run_id = int(parts[1])
        keep_until_ps = float(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "poisoned RUN_ID and KEEP_UNTIL_PS must be numeric"
        ) from exc
    if run_id < 1 or keep_until_ps < 0.0:
        raise argparse.ArgumentTypeError(
            "poisoned RUN_ID must be positive and KEEP_UNTIL_PS non-negative"
        )
    return PoisonedRun(parts[0], run_id, keep_until_ps)


def parse_comparison_system(raw: str) -> ComparisonSystem:
    """Parse LABEL,RUNS_PATH for a shared protonated comparison system."""
    parts = [part.strip() for part in raw.split(",", maxsplit=1)]
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            "comparison system must be LABEL,RUNS_PATH"
        )
    return ComparisonSystem(parts[0], Path(parts[1]))


def read_atoms_mulliken(
    path: Path, atom_ids: Sequence[int]
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Return times and summed orbital Mulliken charges for selected atoms."""
    targets = tuple(dict.fromkeys(int(atom_id) for atom_id in atom_ids))
    if not targets or any(atom_id < 1 for atom_id in targets):
        raise ValueError("atom IDs must be a non-empty sequence of positive integers")
    times: list[float] = []
    charges = {atom_id: [] for atom_id in targets}
    current_time: float | None = None
    current_charge = {atom_id: 0.0 for atom_id in targets}
    contributions = {atom_id: 0 for atom_id in targets}
    elements: dict[int, str] = {}

    def finish_frame() -> None:
        if current_time is None:
            return
        missing = [atom_id for atom_id in targets if contributions[atom_id] == 0]
        if missing:
            raise ValueError(f"atoms {missing} are absent from a frame in {path}")
        times.append(current_time)
        for atom_id in targets:
            charges[atom_id].append(current_charge[atom_id])

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            time_match = TIME_RE.search(line)
            if time_match:
                finish_frame()
                current_time = float(time_match.group(1).replace("D", "E")) / 1000.0
                current_charge = {atom_id: 0.0 for atom_id in targets}
                contributions = {atom_id: 0 for atom_id in targets}
                continue

            fields = line.split()
            if current_time is None or len(fields) < 4:
                continue
            try:
                line_atom_id = int(fields[0])
            except ValueError:
                continue
            if line_atom_id not in current_charge:
                continue
            try:
                value = float(fields[3].replace("D", "E"))
            except ValueError as exc:
                raise ValueError(f"invalid Mulliken value in {path}: {line.rstrip()}") from exc
            if line_atom_id not in elements:
                elements[line_atom_id] = fields[1]
            elif fields[1] != elements[line_atom_id]:
                raise ValueError(f"atom {line_atom_id} changes element in {path}")
            current_charge[line_atom_id] += value
            contributions[line_atom_id] += 1

    finish_frame()
    if not times:
        raise ValueError(f"no Mulliken frames found in {path}")
    for atom_id in targets:
        element = elements.get(atom_id)
        if element is not None and element.upper() != "N":
            raise ValueError(f"atom {atom_id} in {path} is {element}, expected N")
    return np.asarray(times, dtype=float), {
        atom_id: np.asarray(charges[atom_id], dtype=float) for atom_id in targets
    }


def read_atom_mulliken(path: Path, atom_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return times and summed orbital Mulliken charges for one one-based atom."""
    times, charges = read_atoms_mulliken(path, (atom_id,))
    return times, charges[atom_id]


def collect_dataset(
    dataset: Dataset,
    run_ids: Sequence[int],
    run_dir: str,
    poisoned_run: PoisonedRun | None = None,
) -> ChargeSamples:
    all_runs: list[np.ndarray] = []
    all_times: list[np.ndarray] = []
    all_charges: list[np.ndarray] = []
    excluded_frames = 0
    for run_id in run_ids:
        path = dataset.runs_path / f"run-{run_id}" / run_dir / "mulliken"
        if not path.is_file():
            raise FileNotFoundError(f"missing Mulliken file: {path}")
        times, charges = read_atom_mulliken(path, dataset.atom_id)
        if times.size != charges.size or not np.all(np.isfinite(charges)):
            raise ValueError(f"invalid charge series in {path}")
        if poisoned_run is not None and run_id == poisoned_run.run_id:
            keep = times <= poisoned_run.keep_until_ps + 1.0e-9
            excluded_frames = int(np.count_nonzero(~keep))
            times = times[keep]
            charges = charges[keep]
            if times.size == 0:
                raise ValueError(
                    f"poisoned-run cutoff removes every frame from {path}"
                )
        all_runs.append(np.full(times.size, run_id, dtype=int))
        all_times.append(times)
        all_charges.append(charges)
    return ChargeSamples(
        dataset=dataset,
        run_ids=np.concatenate(all_runs),
        times_ps=np.concatenate(all_times),
        charges_e=np.concatenate(all_charges),
        poisoned_run_id=None if poisoned_run is None else poisoned_run.run_id,
        poisoned_keep_until_ps=(
            None if poisoned_run is None else poisoned_run.keep_until_ps
        ),
        excluded_frames=excluded_frames,
    )


def collect_comparison_system(
    comparison: ComparisonSystem,
    datasets: Sequence[Dataset],
    run_ids: Sequence[int],
    run_dir: str,
) -> list[ChargeSamples]:
    """Read all comparison atoms together so each large file is scanned once."""
    atom_ids = tuple(dataset.atom_id for dataset in datasets)
    runs_by_atom = {atom_id: [] for atom_id in atom_ids}
    times_by_atom = {atom_id: [] for atom_id in atom_ids}
    charges_by_atom = {atom_id: [] for atom_id in atom_ids}
    for run_id in run_ids:
        path = comparison.runs_path / f"run-{run_id}" / run_dir / "mulliken"
        if not path.is_file():
            raise FileNotFoundError(f"missing comparison Mulliken file: {path}")
        times, charges = read_atoms_mulliken(path, atom_ids)
        for atom_id in atom_ids:
            atom_charges = charges[atom_id]
            if times.size != atom_charges.size or not np.all(np.isfinite(atom_charges)):
                raise ValueError(f"invalid comparison charge series in {path}")
            runs_by_atom[atom_id].append(np.full(times.size, run_id, dtype=int))
            times_by_atom[atom_id].append(times)
            charges_by_atom[atom_id].append(atom_charges)

    return [
        ChargeSamples(
            dataset=Dataset(
                comparison.label,
                comparison.runs_path,
                dataset.atom_id,
                dataset.atom_label,
            ),
            run_ids=np.concatenate(runs_by_atom[dataset.atom_id]),
            times_ps=np.concatenate(times_by_atom[dataset.atom_id]),
            charges_e=np.concatenate(charges_by_atom[dataset.atom_id]),
        )
        for dataset in datasets
    ]


def common_edges(samples: Sequence[ChargeSamples], bins: int) -> np.ndarray:
    values = np.concatenate([sample.charges_e for sample in samples])
    lower, upper = float(values.min()), float(values.max())
    padding = max(0.02 * (upper - lower), 1.0e-3)
    return np.linspace(lower - padding, upper + padding, bins + 1)


def write_samples(path: Path, samples: Sequence[ChargeSamples]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["system", "atom_label", "atom_id", "run_id", "time_ps", "charge_e"]
        )
        for sample in samples:
            for run_id, time_ps, charge in zip(
                sample.run_ids, sample.times_ps, sample.charges_e, strict=True
            ):
                writer.writerow(
                    [
                        sample.dataset.label,
                        sample.dataset.atom_label,
                        sample.dataset.atom_id,
                        int(run_id),
                        f"{time_ps:.8f}",
                        f"{charge:.10f}",
                    ]
                )


def write_distributions(
    path: Path, samples: Sequence[ChargeSamples], edges: np.ndarray
) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    densities = [np.histogram(sample.charges_e, bins=edges, density=True)[0] for sample in samples]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "charge_e",
                *[
                    f"P_{sample.dataset.label}_{sample.dataset.atom_label}_per_e"
                    for sample in samples
                ],
            ]
        )
        for row in zip(centers, *densities, strict=True):
            writer.writerow([f"{row[0]:.10f}", *[f"{value:.10f}" for value in row[1:]]])


def write_summary(path: Path, samples: Sequence[ChargeSamples]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "system", "atom_label", "atom_id", "runs", "samples",
                "mean_charge_e", "sample_sd_e", "median_charge_e", "min_charge_e",
                "max_charge_e",
                "poisoned_run_id", "poisoned_keep_until_ps", "excluded_frames",
            ]
        )
        for sample in samples:
            q = sample.charges_e
            writer.writerow(
                [
                    sample.dataset.label,
                    sample.dataset.atom_label,
                    sample.dataset.atom_id,
                    np.unique(sample.run_ids).size,
                    q.size,
                    f"{np.mean(q):.10f}",
                    f"{np.std(q, ddof=1):.10f}",
                    f"{np.median(q):.10f}",
                    f"{np.min(q):.10f}",
                    f"{np.max(q):.10f}",
                    "" if sample.poisoned_run_id is None else sample.poisoned_run_id,
                    (
                        ""
                        if sample.poisoned_keep_until_ps is None
                        else f"{sample.poisoned_keep_until_ps:.8f}"
                    ),
                    sample.excluded_frames,
                ]
            )


def plot_distributions(
    path: Path,
    samples: Sequence[ChargeSamples],
    comparisons: Sequence[ChargeSamples] | None,
    edges: np.ndarray,
) -> None:
    colors = ("#D62728", "#1F77B4", "#111111", "#2CA02C")
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), sharex=True, sharey=True)
    comparison_items: Sequence[ChargeSamples | None] = (
        comparisons if comparisons is not None else [None] * len(samples)
    )
    for ax, sample, comparison, color in zip(
        axes.flat, samples, comparison_items, colors, strict=True
    ):
        density, _ = np.histogram(sample.charges_e, bins=edges, density=True)
        mean = float(np.mean(sample.charges_e))
        ax.fill_between(centers, density, step="mid", color=color, alpha=0.20)
        ax.plot(
            centers,
            density,
            color=color,
            linewidth=2.8,
            label=f"{sample.dataset.label} deprotonated; mean {mean:+.3f} e",
        )
        ax.axvline(mean, color=color, linestyle=":", linewidth=1.2, alpha=0.8)
        ax.set_title(
            f"{sample.dataset.label}: {sample.dataset.atom_label} "
            f"(atom {sample.dataset.atom_id})"
        )
        if comparison is not None:
            comparison_density, _ = np.histogram(
                comparison.charges_e, bins=edges, density=True
            )
            comparison_mean = float(np.mean(comparison.charges_e))
            ax.plot(
                centers,
                comparison_density,
                color=color,
                linewidth=2.5,
                linestyle="--",
                label=(
                    f"{comparison.dataset.label} protonated; "
                    f"mean {comparison_mean:+.3f} e"
                ),
            )
            ax.axvline(
                comparison_mean,
                color=color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
            )
            ax.text(
                0.97,
                0.73,
                f"{sample.dataset.label}: {sample.charges_e.size:,} frames\n"
                f"{comparison.dataset.label}: {comparison.charges_e.size:,} frames",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color="0.25",
            )
            ax.legend(loc="upper right", frameon=False, fontsize=8.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(charge) (e⁻¹)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Mulliken charge, s + p (e)")
    fig.suptitle("DFTB equilibration: deprotonated vs protonated ring-N charge")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        help=(
            "repeatable LABEL,RUNS_PATH,ATOM_ID,ATOM_LABEL specification; "
            "defaults to APP/NA, BPP/NB, DPP/ND, and CPP/NC"
        ),
    )
    parser.add_argument(
        "--runs", type=parse_run_ids, default=parse_run_ids("1-10"),
        help="run IDs/ranges (default: 1-10)",
    )
    parser.add_argument(
        "--poisoned-run",
        action="append",
        type=parse_poisoned_run,
        help=(
            "repeatable DATASET_LABEL,RUN_ID,KEEP_UNTIL_PS truncation; "
            "the default four-system report uses APP,8,20.0"
        ),
    )
    parser.add_argument(
        "--comparison-system",
        type=parse_comparison_system,
        help=(
            "protonated LABEL,RUNS_PATH whose matching one-based atom IDs are "
            "overlaid; the default four-system report uses BV at N1T48C1"
        ),
    )
    parser.add_argument(
        "--comparison-runs",
        type=parse_run_ids,
        help="comparison run IDs/ranges (default: use --runs)",
    )
    parser.add_argument("--run-dir", default="equil", help="directory inside each run")
    parser.add_argument("--bins", type=int, default=100, help="common histogram bins")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/tetrapyrrole_mulliken_equil"),
    )
    parser.add_argument("--style", type=Path, default=Path("lefteris.mplstyle"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bins < 5:
        raise SystemExit("--bins must be at least 5")
    using_default_datasets = args.dataset is None
    datasets = args.dataset or [parse_dataset(raw) for raw in DEFAULT_DATASETS]
    if len(datasets) != 4:
        raise SystemExit("exactly four --dataset values are required for the 2x2 report")
    comparison_system = args.comparison_system
    if comparison_system is None and using_default_datasets:
        comparison_system = parse_comparison_system(DEFAULT_COMPARISON_SYSTEM)
    comparison_runs = args.comparison_runs or args.runs
    poisoned_runs = args.poisoned_run
    if poisoned_runs is None:
        poisoned_runs = (
            [parse_poisoned_run(DEFAULT_POISONED_RUN)] if using_default_datasets else []
        )
    dataset_labels = {dataset.label for dataset in datasets}
    selected_runs = set(args.runs)
    poisoned_by_label: dict[str, PoisonedRun] = {}
    for poisoned in poisoned_runs:
        if poisoned.dataset_label not in dataset_labels:
            raise SystemExit(
                f"poisoned-run dataset is not selected: {poisoned.dataset_label}"
            )
        if poisoned.run_id not in selected_runs:
            raise SystemExit(
                f"poisoned run-{poisoned.run_id} is not included by --runs"
            )
        if poisoned.dataset_label in poisoned_by_label:
            raise SystemExit(
                f"only one poisoned run is allowed per dataset: {poisoned.dataset_label}"
            )
        poisoned_by_label[poisoned.dataset_label] = poisoned
    if args.style.is_file():
        plt.style.use(args.style)

    samples = [
        collect_dataset(
            dataset,
            args.runs,
            args.run_dir,
            poisoned_by_label.get(dataset.label),
        )
        for dataset in datasets
    ]
    comparisons = (
        None
        if comparison_system is None
        else collect_comparison_system(
            comparison_system,
            datasets,
            comparison_runs,
            args.run_dir,
        )
    )
    all_samples = [*samples, *(comparisons or [])]
    edges = common_edges(all_samples, args.bins)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.out_dir / "mulliken_charge_distributions.png"
    sample_path = args.out_dir / "mulliken_charge_samples.csv"
    distribution_path = args.out_dir / "mulliken_charge_distributions.csv"
    summary_path = args.out_dir / "mulliken_charge_summary.csv"
    plot_distributions(figure_path, samples, comparisons, edges)
    write_samples(sample_path, all_samples)
    write_distributions(distribution_path, all_samples, edges)
    write_summary(summary_path, all_samples)
    print(f"figure: {figure_path.resolve()}")
    print(f"samples: {sample_path.resolve()}")
    print(f"distributions: {distribution_path.resolve()}")
    print(f"summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
