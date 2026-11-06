#!/usr/bin/env python3
"""Plot the Mainas et al. spectral offset as a conventional two-panel figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "reports" / "Mainas2024" / "spectra-offset"
STYLE = ROOT / "plotting" / "lefteris.mplstyle"

EXPERIMENT = "#2B73B6"
CALCULATION = "#C43C39"
METHOD = "#D8481A"
QM_REGION = "#2E9E7E"
PROTONATION = "#7570B3"
HC_EV_NM = 1239.841984


def gaussian(x: np.ndarray, center: float, width: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)


def plot_spectra(axis: plt.Axes) -> None:
    energy = np.linspace(1.65, 4.18, 1600)
    experimental = gaussian(energy, 1.96, 0.095, 0.58) + gaussian(
        energy, 3.26, 0.135, 1.00
    )
    calculated = gaussian(energy, 2.43, 0.115, 0.58) + gaussian(
        energy, 3.84, 0.150, 1.00
    )

    axis.fill_between(energy, experimental, color=EXPERIMENT, alpha=0.10)
    axis.fill_between(energy, calculated, color=CALCULATION, alpha=0.10)
    axis.plot(energy, experimental, color=EXPERIMENT, linewidth=2.6, label="Experiment")
    axis.plot(energy, calculated, color=CALCULATION, linewidth=2.6, label="Calculation")

    for start, end, ypos, label in (
        (1.96, 2.43, 0.72, r"$+0.47$ eV"),
        (3.26, 3.84, 1.10, r"$+0.58$ eV"),
    ):
        axis.annotate(
            "",
            xy=(end, ypos),
            xytext=(start, ypos),
            arrowprops={"arrowstyle": "-|>", "color": "0.18", "lw": 1.6},
        )
        axis.text(
            0.5 * (start + end),
            ypos + 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="0.18",
        )

    labels = (
        (1.96, 0.61, "Q  1.96", EXPERIMENT),
        (2.43, 0.61, "Q  2.43", CALCULATION),
        (3.26, 1.02, "Soret  3.26", EXPERIMENT),
        (3.84, 1.02, "Soret  3.84", CALCULATION),
    )
    for x, y, text, color in labels:
        axis.text(x, y, text, ha="center", va="bottom", fontsize=9.7, color=color, fontweight="bold")

    axis.set_xlim(1.70, 4.15)
    axis.set_ylim(0.0, 1.20)
    axis.set_xlabel("Energy (eV)")
    axis.set_ylabel("Normalized absorbance")
    axis.set_yticks([])
    axis.set_title("A  Published protein-spectrum offset", loc="left", fontweight="bold")
    axis.legend(loc="upper left", ncol=2, fontsize=10.5)
    axis.grid(axis="x", alpha=0.15)
    axis.text(
        0.98,
        0.025,
        "schematic line shapes; peak positions from Figure 8",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.6,
        color="0.38",
    )


def plot_effect_scales(axis: plt.Axes) -> None:
    observed = 0.57
    td_dft_range = (0.08, 0.15)
    qm_range = (0.02, 0.08)
    protonation_bv_ssa = HC_EV_NM / 648.0 - HC_EV_NM / 662.0
    protonation_pcb_asa = HC_EV_NM / 528.0 - HC_EV_NM / 590.0

    axis.axvline(
        observed,
        color=CALCULATION,
        linestyle=(0, (4, 3)),
        linewidth=1.3,
    )
    axis.text(
        0.645,
        2.57,
        r"observed offset  $\approx0.57$ eV",
        color=CALCULATION,
        fontsize=9.4,
        fontweight="bold",
        ha="right",
        va="center",
    )

    axis.plot(
        td_dft_range,
        [2.0, 2.0],
        color=METHOD,
        linewidth=10,
        solid_capstyle="round",
    )
    axis.scatter(
        td_dft_range,
        [2.0, 2.0],
        s=58,
        facecolor=METHOD,
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
    )
    axis.text(
        td_dft_range[1] + 0.018,
        2.0,
        "0.08–0.15 eV",
        color=METHOD,
        fontsize=10.0,
        fontweight="bold",
        va="center",
    )

    axis.plot(
        [qm_range[0], qm_range[1]],
        [1.0, 1.0],
        color=QM_REGION,
        linewidth=10,
        solid_capstyle="round",
    )
    axis.scatter(
        qm_range,
        [1.0, 1.0],
        s=58,
        facecolor=QM_REGION,
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
    )
    axis.text(
        qm_range[1] + 0.018,
        1.0,
        "0.02–0.08 eV",
        color=QM_REGION,
        fontsize=10.0,
        fontweight="bold",
        va="center",
    )

    axis.add_patch(
        Rectangle(
            (protonation_bv_ssa, -0.13),
            protonation_pcb_asa - protonation_bv_ssa,
            0.26,
            facecolor=PROTONATION,
            edgecolor=PROTONATION,
            linewidth=1.0,
            alpha=0.18,
            hatch="///",
        )
    )
    axis.plot(
        [protonation_bv_ssa, protonation_pcb_asa],
        [0.0, 0.0],
        color=PROTONATION,
        linewidth=5.0,
        alpha=0.62,
        solid_capstyle="round",
    )
    axis.scatter(
        [protonation_bv_ssa, protonation_pcb_asa],
        [0.0, 0.0],
        s=62,
        facecolor=PROTONATION,
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
    )
    axis.text(
        protonation_pcb_asa + 0.018,
        0.0,
        "0.04–0.25 eV",
        va="center",
        color=PROTONATION,
        fontsize=10.0,
        fontweight="bold",
    )

    axis.set_xlim(0.0, 0.66)
    axis.set_ylim(-0.82, 2.72)
    axis.set_yticks(
        [2.0, 1.0, 0.0],
        ["TD-DFT", "QM region", "Protonation"],
    )
    axis.set_xlabel("Reported energy scale (eV)")
    axis.set_title("B  Benchmarks in related bilin studies", loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=0.15)
    axis.text(
        0.02,
        -0.67,
        "BPP → PPP in Sandercyanin remains unknown",
        color=PROTONATION,
        fontsize=9.4,
        fontweight="bold",
        ha="left",
        va="center",
    )


def write_data(path: Path) -> None:
    protonation_bv_ssa = HC_EV_NM / 648.0 - HC_EV_NM / 662.0
    protonation_pcb_asa = HC_EV_NM / 528.0 - HC_EV_NM / 590.0
    rows = (
        ("observed_offset", 0.57, 0.57, "Mainas et al. 2024"),
        ("td_dft", 0.08, 0.15, "Falklof and Durbeej 2013"),
        ("qm_region_size", 0.02, 0.08, "Falklof and Durbeej 2013"),
        ("fixed_geometry_protonation", protonation_bv_ssa, protonation_pcb_asa, "Matute et al. 2008"),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "lower_eV", "upper_eV", "source"])
        writer.writerows(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.style.use(STYLE)
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 5.9), gridspec_kw={"wspace": 0.29})
    plot_spectra(axes[0])
    plot_effect_scales(axes[1])

    figure.suptitle(
        "Possible origins of the Sandercyanin spectral offset",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.936,
        "Calculated versus experimental absorption in Mainas et al. (2024)",
        ha="center",
        va="center",
        fontsize=12.5,
        color="0.25",
    )
    figure.text(
        0.5,
        0.025,
        "These are benchmarks from related systems, not an additive decomposition.  TD-DFT/QM-region "
        "values: Falklöf & Durbeej (2013); fixed-geometry protonation: Matute et al. (2008).",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.35",
    )
    figure.subplots_adjust(top=0.82, bottom=0.17)

    destination = OUTDIR / "mainas-2024-spectra-offset-figure.png"
    figure.savefig(destination, dpi=300)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)
    write_data(OUTDIR / "mainas-2024-spectra-offset-figure-data.csv")


if __name__ == "__main__":
    main()
