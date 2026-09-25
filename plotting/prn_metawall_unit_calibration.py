"""Plot the PRN-anti DCDFTBMD METAWALL unit calibration."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WHAM_DIR = ROOT / "systems/PRN-anti/solv_5.5/us-pull/wham"
SUMMARY = WHAM_DIR / "window_summary.csv"
FIGURE = WHAM_DIR / "metawall-unit-calibration.png"
DATA = WHAM_DIR / "metawall-unit-calibration.csv"

TEMPERATURE_K = 300.0
KAPPA_INPUT = 0.05
KJ_PER_KCAL = 4.184
R_KJ_MOL_K = 0.008314462618
R_KCAL_MOL_K = R_KJ_MOL_K / KJ_PER_KCAL
WHAM_K_USED = 2.0 * KAPPA_INPUT
WHAM_K_CORRECTED = 2.0 * KAPPA_INPUT / KJ_PER_KCAL
RAW_BARRIER = 31.556315
CORRECTED_BARRIER = 7.524

# Differential three-step calibration. The integrator accumulates a factor of
# 3.5 for this setup; the observed response was 3.499639766 times the one-step
# kJ prediction. A kcal interpretation predicts 4.184 times too much force.
INTEGRATOR_WEIGHT = 3.5
OBSERVED_OVER_KJ_ONE_STEP = 3.4996397663393854
RESPONSE_KJ = OBSERVED_OVER_KJ_ONE_STEP / INTEGRATOR_WEIGHT
RESPONSE_KCAL = RESPONSE_KJ / KJ_PER_KCAL


def read_window_widths() -> tuple[np.ndarray, np.ndarray]:
    with SUMMARY.open() as stream:
        rows = list(csv.DictReader(stream))
    centers = np.array([float(row["center_deg"]) for row in rows])
    widths = np.array([float(row["std_deg"]) for row in rows])
    order = np.argsort(centers)
    return centers[order], widths[order]


def harmonic_sigma(energy_unit: str) -> float:
    gas_constant = R_KJ_MOL_K if energy_unit == "kJ" else R_KCAL_MOL_K
    return math.sqrt(gas_constant * TEMPERATURE_K / (2.0 * KAPPA_INPUT))


def write_data(centers: np.ndarray, widths: np.ndarray, sigma_kj: float, sigma_kcal: float) -> None:
    with DATA.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["category", "label", "value", "unit"])
        writer.writerows(
            ("window_std", f"center_{center:g}_deg", f"{width:.10g}", "degree")
            for center, width in zip(centers, widths)
        )
        writer.writerows(
            [
                ("predicted_std", "DCDFTBMD_kJ_interpretation", f"{sigma_kj:.10g}", "degree"),
                ("predicted_std", "incorrect_kcal_interpretation", f"{sigma_kcal:.10g}", "degree"),
                ("force_test", "kJ_hypothesis", f"{RESPONSE_KJ:.10g}", "observed_over_expected"),
                ("force_test", "kcal_hypothesis", f"{RESPONSE_KCAL:.10g}", "observed_over_expected"),
                ("wham_K", "used", f"{WHAM_K_USED:.10g}", "kcal/mol/degree^2"),
                ("wham_K", "corrected", f"{WHAM_K_CORRECTED:.10g}", "kcal/mol/degree^2"),
                ("barrier", "previous_incorrect_WHAM", f"{RAW_BARRIER:.10g}", "kcal/mol"),
                ("barrier", "corrected_WHAM", f"{CORRECTED_BARRIER:.10g}", "kcal/mol"),
            ]
        )


def label_bars(ax: plt.Axes, bars: object, digits: int) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def main() -> None:
    centers, widths = read_window_widths()
    sigma_kj = harmonic_sigma("kJ")
    sigma_kcal = harmonic_sigma("kcal")
    write_data(centers, widths, sigma_kj, sigma_kcal)

    blue = "#235789"
    green = "#2a9d65"
    red = "#d1495b"
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=True)
    fig.suptitle(
        "DCDFTBMD METAWALL unit calibration: κ input = 0.05",
        fontsize=17,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.scatter(centers, widths, color=blue, s=34, label="19 production windows", zorder=3)
    ax.axhline(sigma_kj, color=green, linewidth=2.2, label=f"kJ prediction: {sigma_kj:.2f}°")
    ax.axhline(sigma_kcal, color=red, linewidth=2.2, linestyle="--", label=f"kcal prediction: {sigma_kcal:.2f}°")
    ax.fill_between([centers.min(), centers.max()], widths.min(), widths.max(), color=blue, alpha=0.08)
    ax.set_title("A. Equilibrium window widths")
    ax.set_xlabel("Umbrella center (degrees)")
    ax.set_ylabel("Observed standard deviation (degrees)")
    ax.text(
        0.03,
        0.32,
        f"Observed mean = {widths.mean():.2f}°\nObserved range = {widths.min():.2f}–{widths.max():.2f}°",
        transform=ax.transAxes,
        fontsize=9.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    response_bars = ax.bar(["kJ hypothesis", "kcal hypothesis"], [RESPONSE_KJ, RESPONSE_KCAL], color=[green, red], width=0.58)
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="Exact match = 1")
    label_bars(ax, response_bars, 4)
    ax.set_ylim(0.0, 1.15)
    ax.set_title("B. Direct short-time force calibration")
    ax.set_ylabel("Measured response / predicted response")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1, 0]
    k_bars = ax.bar(["Used in WHAM", "Correct value"], [WHAM_K_USED, WHAM_K_CORRECTED], color=[red, green], width=0.58)
    label_bars(ax, k_bars, 5)
    ax.set_ylim(0.0, 0.115)
    ax.set_title("C. WHAM harmonic coefficient")
    ax.set_ylabel("K (kcal mol⁻¹ deg⁻²)")
    ax.text(
        0.5,
        0.91,
        f"Used value is {WHAM_K_USED / WHAM_K_CORRECTED:.3f}× too large",
        transform=ax.transAxes,
        ha="center",
        color=red,
        fontweight="bold",
    )

    ax = axes[1, 1]
    barrier_bars = ax.bar(["Previous WHAM", "Corrected WHAM"], [RAW_BARRIER, CORRECTED_BARRIER], color=[red, green], width=0.58)
    label_bars(ax, barrier_bars, 2)
    ax.set_ylim(0.0, 36.0)
    ax.set_title("D. Consequence for syn→anti barrier")
    ax.set_ylabel("Barrier (kcal mol⁻¹)")
    ax.text(
        0.5,
        0.06,
        "Corrected using K = 0.0239006\nand 500 bootstrap replicas",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(FIGURE, dpi=300)
    plt.close(fig)
    print(f"Wrote {FIGURE}")
    print(f"Wrote {DATA}")


if __name__ == "__main__":
    main()
