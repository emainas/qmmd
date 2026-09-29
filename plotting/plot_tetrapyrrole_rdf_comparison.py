#!/usr/bin/env python3
"""Compare deprotonated and protonated tetrapyrrole hydration RDFs.

Each ``--pair`` provides a deprotonated RDF YAML, a protonated reference RDF
YAML, and a site label.  Per-run cpptraj RDFs are combined with frame-count
weights read from their preserved cpptraj logs.  The plotted bands are the
weighted run-to-run standard deviations, not frame-level confidence intervals.
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

from qmmd.rdf import RDFConfig, _sanitize_mask, find_repo_root, load_config, run_dir


FRAME_RE = re.compile(r"RADIAL:\s+(\d+)\s+frames", re.IGNORECASE)


@dataclass(frozen=True)
class PairSpec:
    deprotonated_yaml: Path
    protonated_yaml: Path
    site: str


@dataclass(frozen=True)
class EnsembleRDF:
    system: str
    site: str
    state: str
    pair_label: str
    radii_A: np.ndarray
    per_run: np.ndarray
    run_ids: np.ndarray
    frame_counts: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    sem: np.ndarray


DEFAULT_PAIRS = (
    "configs/APP/analysis/rdf-deprot-water-h.yaml,"
    "configs/BV/analysis/rdf-prot-na-water-o.yaml,NA",
    "configs/BPP/analysis/rdf-deprot-water-h.yaml,"
    "configs/BV/analysis/rdf-prot-nb-water-o.yaml,NB",
    "configs/DPP/analysis/rdf-deprot-water-h.yaml,"
    "configs/BV/analysis/rdf-prot-nd-water-o.yaml,ND",
    "configs/CPP/analysis/rdf-deprot-water-h.yaml,"
    "configs/BV/analysis/rdf-prot-nc-water-o.yaml,NC",
)


def parse_pair(raw: str) -> PairSpec:
    parts = [part.strip() for part in raw.split(",", maxsplit=2)]
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "pair must be DEPROTONATED_YAML,PROTONATED_YAML,SITE"
        )
    return PairSpec(Path(parts[0]), Path(parts[1]), parts[2])


def rdf_filename(cfg: RDFConfig) -> str:
    mask1 = _sanitize_mask(cfg.mask1)
    mask2 = _sanitize_mask(cfg.mask2) if cfg.mask2 else "all"
    return f"rdf_{mask1}_{mask2}.dat"


def log_filename(cfg: RDFConfig) -> str:
    suffix = f"_{_sanitize_mask(cfg.analysis_name)}" if cfg.analysis_name else ""
    return f"cpptraj{suffix}.out"


def read_rdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing RDF: {path}")
    values = np.loadtxt(path, comments="#", usecols=(0, 1), ndmin=2)
    if values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid RDF table: {path}")
    return values[:, 0], values[:, 1]


def read_frame_count(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"missing cpptraj RDF log: {path}")
    matches = FRAME_RE.findall(path.read_text(errors="ignore"))
    if not matches:
        raise ValueError(f"could not find RADIAL frame count in {path}")
    count = int(matches[-1])
    if count < 1:
        raise ValueError(f"invalid frame count in {path}: {count}")
    return count


def load_ensemble(yaml_path: Path, site: str, state: str) -> EnsembleRDF:
    yaml_path = yaml_path.resolve()
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)
    radii: np.ndarray | None = None
    curves: list[np.ndarray] = []
    frame_counts: list[int] = []
    for run_id in cfg.run_ids:
        analysis = run_dir(cfg, repo_root, run_id) / "analysis"
        run_radii, curve = read_rdf(analysis / rdf_filename(cfg))
        if radii is None:
            radii = run_radii
        elif run_radii.shape != radii.shape or not np.allclose(
            run_radii, radii, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError(f"RDF grid mismatch in run-{run_id}: {yaml_path}")
        curves.append(curve)
        frame_counts.append(read_frame_count(analysis / log_filename(cfg)))
    if radii is None:
        raise ValueError(f"no runs configured in {yaml_path}")

    matrix = np.vstack(curves)
    weights = np.asarray(frame_counts, dtype=float)
    mean = np.average(matrix, axis=0, weights=weights)
    variance = np.average((matrix - mean) ** 2, axis=0, weights=weights)
    sd = np.sqrt(np.maximum(variance, 0.0))
    effective_runs = weights.sum() ** 2 / np.square(weights).sum()
    sem = sd / np.sqrt(effective_runs)
    atom = site[-1].upper()
    pair_label = f"N{atom}···H(water)" if state == "deprotonated" else f"H{atom}···O(water)"
    return EnsembleRDF(
        system=cfg.system,
        site=site,
        state=state,
        pair_label=pair_label,
        radii_A=radii,
        per_run=matrix,
        run_ids=np.asarray(cfg.run_ids, dtype=int),
        frame_counts=weights.astype(int),
        mean=mean,
        sd=sd,
        sem=sem,
    )


def write_mean_data(path: Path, ensembles: Sequence[EnsembleRDF]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "site", "state", "system", "pair", "r_A", "g_mean",
                "g_run_sd", "g_run_sem", "runs", "total_frames",
            ]
        )
        for ensemble in ensembles:
            for radius, mean, sd, sem in zip(
                ensemble.radii_A,
                ensemble.mean,
                ensemble.sd,
                ensemble.sem,
                strict=True,
            ):
                writer.writerow(
                    [
                        ensemble.site,
                        ensemble.state,
                        ensemble.system,
                        ensemble.pair_label,
                        f"{radius:.8f}",
                        f"{mean:.10f}",
                        f"{sd:.10f}",
                        f"{sem:.10f}",
                        ensemble.run_ids.size,
                        int(ensemble.frame_counts.sum()),
                    ]
                )


def write_run_data(path: Path, ensembles: Sequence[EnsembleRDF]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["site", "state", "system", "pair", "run_id", "frames", "r_A", "g_r"]
        )
        for ensemble in ensembles:
            for run_index, run_id in enumerate(ensemble.run_ids):
                for radius, value in zip(
                    ensemble.radii_A, ensemble.per_run[run_index], strict=True
                ):
                    writer.writerow(
                        [
                            ensemble.site,
                            ensemble.state,
                            ensemble.system,
                            ensemble.pair_label,
                            int(run_id),
                            int(ensemble.frame_counts[run_index]),
                            f"{radius:.8f}",
                            f"{value:.10f}",
                        ]
                    )


def dimensionless_wr(g_r: np.ndarray, minimum_g: float) -> np.ndarray:
    """Return -ln(g(r)), masking unsupported/near-zero RDF values."""
    result = np.full(g_r.shape, np.nan, dtype=float)
    supported = np.isfinite(g_r) & (g_r >= minimum_g)
    result[supported] = -np.log(g_r[supported])
    return result


def write_wr_data(
    path: Path, ensembles: Sequence[EnsembleRDF], minimum_g: float
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["site", "state", "system", "pair", "r_A", "g_mean", "minus_ln_g"]
        )
        for ensemble in ensembles:
            wr = dimensionless_wr(ensemble.mean, minimum_g)
            for radius, mean, value in zip(
                ensemble.radii_A, ensemble.mean, wr, strict=True
            ):
                writer.writerow(
                    [
                        ensemble.site,
                        ensemble.state,
                        ensemble.system,
                        ensemble.pair_label,
                        f"{radius:.8f}",
                        f"{mean:.10f}",
                        "" if not np.isfinite(value) else f"{value:.10f}",
                    ]
                )


def plot(
    path: Path,
    ensemble_pairs: Sequence[tuple[EnsembleRDF, EnsembleRDF]],
    x_min: float,
    x_max: float,
) -> None:
    colors = ("#D62728", "#1F77B4", "#111111", "#2CA02C")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), sharex=True, sharey=True)
    for ax, (deprotonated, protonated), color in zip(
        axes.flat, ensemble_pairs, colors, strict=True
    ):
        ax.fill_between(
            deprotonated.radii_A,
            np.maximum(deprotonated.mean - deprotonated.sd, 0.0),
            deprotonated.mean + deprotonated.sd,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            deprotonated.radii_A,
            deprotonated.mean,
            color=color,
            linewidth=2.8,
            label=f"{deprotonated.system}: {deprotonated.pair_label}",
        )
        ax.fill_between(
            protonated.radii_A,
            np.maximum(protonated.mean - protonated.sd, 0.0),
            protonated.mean + protonated.sd,
            color=color,
            alpha=0.08,
            linewidth=0,
        )
        ax.plot(
            protonated.radii_A,
            protonated.mean,
            color=color,
            linewidth=2.5,
            linestyle="--",
            label=f"{protonated.system}: {protonated.pair_label}",
        )
        ax.set_title(f"Site {deprotonated.site[-1].upper()}")
        ax.set_xlim(x_min, x_max)
        ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("g(r)")
    for ax in axes[-1, :]:
        ax.set_xlabel("r (Å)")
    fig.suptitle("Tetrapyrrole hydration: deprotonated vs protonated sites")
    fig.text(
        0.5,
        0.012,
        "Curves are frame-weighted means over runs 1–10; shading is weighted run-to-run SD.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_wr(
    path: Path,
    ensemble_pairs: Sequence[tuple[EnsembleRDF, EnsembleRDF]],
    x_min: float,
    x_max: float,
    minimum_g: float,
) -> None:
    colors = ("#D62728", "#1F77B4", "#111111", "#2CA02C")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), sharex=True, sharey=True)
    for ax, (deprotonated, protonated), color in zip(
        axes.flat, ensemble_pairs, colors, strict=True
    ):
        deprotonated_wr = dimensionless_wr(deprotonated.mean, minimum_g)
        protonated_wr = dimensionless_wr(protonated.mean, minimum_g)
        ax.plot(
            deprotonated.radii_A,
            deprotonated_wr,
            color=color,
            linewidth=2.8,
            label=f"{deprotonated.system}: {deprotonated.pair_label}",
        )
        ax.plot(
            protonated.radii_A,
            protonated_wr,
            color=color,
            linewidth=2.5,
            linestyle="--",
            label=f"{protonated.system}: {protonated.pair_label}",
        )
        ax.axhline(0.0, color="0.55", linewidth=0.9, linestyle=":")
        ax.set_title(f"Site {deprotonated.site[-1].upper()}")
        ax.set_xlim(x_min, x_max)
        ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$w(r)=-\ln g(r)$")
    for ax in axes[-1, :]:
        ax.set_xlabel("r (Å)")
    fig.suptitle("Dimensionless radial PMF from hydration RDFs")
    fig.text(
        0.5,
        0.012,
        f"Frame-weighted mean g(r); values with g(r) < {minimum_g:g} are masked.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        type=parse_pair,
        help="repeatable DEPROTONATED_YAML,PROTONATED_YAML,SITE specification",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/tetrapyrrole_mulliken_equil"),
    )
    parser.add_argument("--x-min", type=float, default=1.0)
    parser.add_argument("--x-max", type=float, default=6.0)
    parser.add_argument(
        "--w-min-g",
        type=float,
        default=0.01,
        help="mask g(r) below this value when evaluating -ln(g(r))",
    )
    parser.add_argument("--style", type=Path, default=Path("lefteris.mplstyle"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.x_max <= args.x_min:
        raise SystemExit("--x-max must be greater than --x-min")
    if not 0.0 < args.w_min_g < 1.0:
        raise SystemExit("--w-min-g must be between 0 and 1")
    pairs = args.pair or [parse_pair(raw) for raw in DEFAULT_PAIRS]
    if len(pairs) != 4:
        raise SystemExit("exactly four --pair values are required for the 2x2 report")
    if args.style.is_file():
        plt.style.use(args.style)

    ensemble_pairs = [
        (
            load_ensemble(pair.deprotonated_yaml, pair.site, "deprotonated"),
            load_ensemble(pair.protonated_yaml, pair.site, "protonated"),
        )
        for pair in pairs
    ]
    ensembles = [ensemble for pair in ensemble_pairs for ensemble in pair]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.out_dir / "rdf_charge_state_comparison.png"
    wr_figure_path = args.out_dir / "rdf_charge_state_wr.png"
    mean_path = args.out_dir / "rdf_charge_state_mean.csv"
    runs_path = args.out_dir / "rdf_charge_state_runs.csv"
    wr_path = args.out_dir / "rdf_charge_state_wr.csv"
    plot(figure_path, ensemble_pairs, args.x_min, args.x_max)
    plot_wr(
        wr_figure_path,
        ensemble_pairs,
        args.x_min,
        args.x_max,
        args.w_min_g,
    )
    write_mean_data(mean_path, ensembles)
    write_run_data(runs_path, ensembles)
    write_wr_data(wr_path, ensembles, args.w_min_g)
    print(f"figure: {figure_path.resolve()}")
    print(f"w(r) figure: {wr_figure_path.resolve()}")
    print(f"mean data: {mean_path.resolve()}")
    print(f"run data: {runs_path.resolve()}")
    print(f"w(r) data: {wr_path.resolve()}")


if __name__ == "__main__":
    main()
