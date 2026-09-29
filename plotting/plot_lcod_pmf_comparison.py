#!/usr/bin/env python3
"""Compare the APP/BPP, CPP/BPP, and CPP/DPP LCOD umbrella PMFs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qmmd.us_lcod_wham import equivalent_delta_pka, load_config


ROOT = Path(__file__).resolve().parents[1]
R_KCAL_MOL_K = 0.00198720425864083


@dataclass(frozen=True, slots=True)
class Comparison:
    pair: str
    system: str
    color: str
    config: Path
    output: Path


@dataclass(frozen=True, slots=True)
class Result:
    case: Comparison
    negative_state: str
    positive_state: str
    negative_site: str
    positive_site: str
    temperature_k: float
    x: np.ndarray
    y: np.ndarray
    negative_minimum: tuple[float, float]
    positive_minimum: tuple[float, float]
    barrier_maximum: tuple[float, float]
    gap: float
    delta_pka: float
    barrier_negative_to_positive: float
    barrier_positive_to_negative: float
    negative_fraction: float
    positive_fraction: float


CASES = (
    Comparison(
        "A ⇌ B",
        "APP/BPP",
        "#D62728",
        ROOT / "configs/APP/us/wham.yaml",
        ROOT / "systems/APP/solv_4.0/us-lcod/wham-equil",
    ),
    Comparison(
        "B ⇌ C",
        "BPP/CPP",
        "#1F5AA6",
        ROOT / "configs/CPP/us/wham.yaml",
        ROOT / "systems/CPP/solv_4.0/us-lcod/wham-equil",
    ),
    Comparison(
        "C ⇌ D",
        "CPP/DPP",
        "#111111",
        ROOT / "configs/DPP/us/wham.yaml",
        ROOT / "systems/DPP/solv_4.0/us-lcod/wham-equil",
    ),
)


def read_smooth(path: Path, limits: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, float]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    x = np.asarray([float(row["lcod_A"]) for row in rows], dtype=float)
    y = np.asarray([float(row["smoothed_pmf_kcal_mol"]) for row in rows], dtype=float)
    keep = (x >= limits[0]) & (x <= limits[1]) & np.isfinite(y)
    if np.count_nonzero(keep) < 3:
        raise ValueError(f"Too few finite smooth-PMF points in {path}")
    x, y = x[keep], y[keep]
    zero = float(np.min(y))
    return x, y - zero, zero


def read_features(path: Path) -> dict[str, tuple[float, float] | float]:
    result: dict[str, tuple[float, float] | float] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            coordinate = row["lcod_A"].strip()
            value = float(row["pmf_or_difference_kcal_mol"])
            result[row["feature"]] = (float(coordinate), value) if coordinate else value
    required = {
        "negative_minimum",
        "positive_minimum",
        "barrier_maximum",
        "gap_positive_minus_negative",
        "barrier_negative_to_positive",
        "barrier_positive_to_negative",
    }
    missing = required.difference(result)
    if missing:
        raise ValueError(f"Missing PMF features in {path}: {sorted(missing)}")
    return result


def two_state_fractions(gap_kcal_mol: float, temperature_k: float) -> tuple[float, float]:
    """Return negative/positive fractions implied by G(+) - G(-)."""
    ratio = math.exp(-gap_kcal_mol / (R_KCAL_MOL_K * temperature_k))
    positive = ratio / (1.0 + ratio)
    return 1.0 - positive, positive


def load_result(case: Comparison, limits: tuple[float, float]) -> Result:
    cfg = load_config(case.config)
    x, y, zero = read_smooth(case.output / "pmf_smooth.csv", limits)
    features = read_features(case.output / "pmf_features.csv")
    negative = features["negative_minimum"]
    positive = features["positive_minimum"]
    barrier = features["barrier_maximum"]
    if not isinstance(negative, tuple) or not isinstance(positive, tuple) or not isinstance(barrier, tuple):
        raise TypeError(f"Malformed coordinate features for {case.system}")
    gap = float(features["gap_positive_minus_negative"])
    negative_fraction, positive_fraction = two_state_fractions(gap, cfg.temperature_k)
    return Result(
        case=case,
        negative_state=cfg.negative_basin_state,
        positive_state=cfg.positive_basin_state,
        negative_site=cfg.negative_basin_site,
        positive_site=cfg.positive_basin_site,
        temperature_k=cfg.temperature_k,
        x=x,
        y=y,
        negative_minimum=(negative[0], negative[1] - zero),
        positive_minimum=(positive[0], positive[1] - zero),
        barrier_maximum=(barrier[0], barrier[1] - zero),
        gap=gap,
        delta_pka=equivalent_delta_pka(gap, cfg.temperature_k),
        barrier_negative_to_positive=float(features["barrier_negative_to_positive"]),
        barrier_positive_to_negative=float(features["barrier_positive_to_negative"]),
        negative_fraction=negative_fraction,
        positive_fraction=positive_fraction,
    )


def write_curves(path: Path, results: list[Result]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pair", "system", "color", "lcod_A", "shifted_pmf_kcal_mol"])
        for result in results:
            writer.writerows(
                (result.case.pair, result.case.system, result.case.color, f"{x:.8f}", f"{y:.10g}")
                for x, y in zip(result.x, result.y)
            )


def write_summary(path: Path, results: list[Result]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "pair",
                "negative_state",
                "negative_site",
                "positive_state",
                "positive_site",
                "negative_minimum_A",
                "positive_minimum_A",
                "delta_g_positive_minus_negative_kcal_mol",
                "equivalent_delta_pka_positive_minus_negative",
                "barrier_negative_to_positive_kcal_mol",
                "barrier_positive_to_negative_kcal_mol",
                "negative_fraction",
                "positive_fraction",
                "temperature_K",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.case.pair,
                    result.negative_state,
                    result.negative_site,
                    result.positive_state,
                    result.positive_site,
                    f"{result.negative_minimum[0]:.8f}",
                    f"{result.positive_minimum[0]:.8f}",
                    f"{result.gap:.8f}",
                    f"{result.delta_pka:.8f}",
                    f"{result.barrier_negative_to_positive:.8f}",
                    f"{result.barrier_positive_to_negative:.8f}",
                    f"{result.negative_fraction:.10f}",
                    f"{result.positive_fraction:.10f}",
                    f"{result.temperature_k:.2f}",
                ]
            )


def plot(path: Path, results: list[Result], limits: tuple[float, float]) -> None:
    style = Path(__file__).with_name("lefteris.mplstyle")
    if style.is_file():
        plt.style.use(style)
    fig = plt.figure(figsize=(14.0, 9.0))
    grid = fig.add_gridspec(2, 1, height_ratios=(2.25, 1.05))
    axis = fig.add_subplot(grid[0])
    table_axis = fig.add_subplot(grid[1])
    table_axis.axis("off")

    for result in results:
        axis.plot(
            result.x,
            result.y,
            color=result.case.color,
            linewidth=3.4,
            label=f"{result.case.pair}: {result.negative_state} (−) ⇌ {result.positive_state} (+)",
            zorder=2,
        )
        axis.scatter(
            [result.negative_minimum[0], result.positive_minimum[0]],
            [result.negative_minimum[1], result.positive_minimum[1]],
            s=42,
            facecolor="white",
            edgecolor=result.case.color,
            linewidth=1.7,
            zorder=4,
        )
        axis.scatter(
            [result.barrier_maximum[0]],
            [result.barrier_maximum[1]],
            marker="^",
            s=58,
            facecolor=result.case.color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
    axis.axvline(0.0, color="0.55", linewidth=0.8, linestyle="--", zorder=0)
    axis.set(
        xlim=limits,
        ylim=(0.0, None),
        xlabel="LCOD (Å; negative/positive basin assignments are listed below)",
        ylabel="PMF (each minimum = 0; kcal mol⁻¹)",
        title="Tetrapyrrole proton-tautomer LCOD PMFs",
    )
    axis.grid(False)
    axis.legend(frameon=False, loc="upper right")

    columns = [
        "Pair",
        "Negative basin",
        "Positive basin",
        "ΔG(+−−)\n(kcal mol⁻¹)",
        "equiv. ΔpKa\n(+−−)",
        "ΔG‡ −→+ / +→−\n(kcal mol⁻¹)",
        "Two-state fractions",
    ]
    cells = []
    for result in results:
        cells.append(
            [
                result.case.pair,
                f"{result.negative_state}\n{result.negative_site}",
                f"{result.positive_state}\n{result.positive_site}",
                f"{result.gap:+.3f}",
                f"{result.delta_pka:+.3f}",
                f"{result.barrier_negative_to_positive:.2f} / "
                f"{result.barrier_positive_to_negative:.2f}",
                f"{result.negative_state} {100.0 * result.negative_fraction:.1f}%\n"
                f"{result.positive_state} {100.0 * result.positive_fraction:.1f}%",
            ]
        )
    table = table_axis.table(
        cellText=cells,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.24, 1.0, 0.72],
        colWidths=[0.07, 0.15, 0.15, 0.12, 0.11, 0.18, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    for column in range(len(columns)):
        table[(0, column)].set_facecolor("#ECEFF1")
        table[(0, column)].set_text_props(weight="bold")
    for row, result in enumerate(results, start=1):
        table[(row, 0)].set_text_props(color=result.case.color, weight="bold")
    table_axis.text(
        0.5,
        0.02,
        "ΔG = G(+) − G(−) from smooth-curve minima; equivalent ΔpKa = −ΔG/(RT ln 10). "
        "Fractions are the two-state populations implied by that ΔG at 300 K.\n"
        "All three PMFs are exploratory and use available equilibration trajectories.",
        transform=table_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.8,
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.055, hspace=0.22)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/tetrapyrrole_lcod_pmf",
    )
    parser.add_argument("--xmin", type=float, default=-2.0)
    parser.add_argument("--xmax", type=float, default=2.0)
    args = parser.parse_args()
    if not args.xmin < args.xmax:
        parser.error("--xmin must be less than --xmax")
    limits = (args.xmin, args.xmax)
    results = [load_result(case, limits) for case in CASES]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = args.output_dir / "three_pmf_curves.csv"
    summary = args.output_dir / "three_pmf_summary.csv"
    figure = args.output_dir / "three_pmf_comparison.png"
    write_curves(curves, results)
    write_summary(summary, results)
    plot(figure, results, limits)
    print(f"figure: {figure}")
    print(f"curves: {curves}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
