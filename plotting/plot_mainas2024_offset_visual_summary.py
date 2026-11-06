#!/usr/bin/env python3
"""Make a compact visual summary of the Mainas et al. (2024) spectral offset."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "reports" / "Mainas2024" / "spectra-offset"

NAVY = "#17324D"
BLUE = "#247BA0"
RED = "#E76F51"
PURPLE = "#745296"
TEAL = "#2A9D8F"
GOLD = "#D99A27"
INK = "#18232E"
MUTED = "#61717D"
GRID = "#D9E2E7"

HC_EV_NM = 1239.841984


def gaussian(x: np.ndarray, center: float, width: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)


def rounded_panel(fig: plt.Figure, xy: tuple[float, float], width: float, height: float) -> None:
    fig.add_artist(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            transform=fig.transFigure,
            facecolor="#F7F9FA",
            edgecolor="#D9E2E7",
            linewidth=1.2,
            zorder=-10,
        )
    )


def build_figure() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    observed_offset = 0.57
    method_range = (0.30, 0.60)
    qm_region_range = (0.02, 0.08)
    protonation_bv = HC_EV_NM / 648.0 - HC_EV_NM / 662.0
    protonation_pcb = HC_EV_NM / 528.0 - HC_EV_NM / 590.0

    fig = plt.figure(figsize=(15.5, 7.8), dpi=180)
    fig.text(
        0.055,
        0.935,
        "Where could the ≈0.57 eV blue shift come from?",
        fontsize=25,
        color=NAVY,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.056,
        0.884,
        "Sandercyanin absorption spectrum · Mainas et al. (2024)",
        fontsize=13.5,
        color=MUTED,
        va="top",
    )

    rounded_panel(fig, (0.045, 0.145), 0.505, 0.680)
    rounded_panel(fig, (0.575, 0.145), 0.380, 0.680)

    # Left: schematic spectra using the published peak positions.
    ax = fig.add_axes([0.080, 0.245, 0.435, 0.480])
    energy = np.linspace(1.65, 4.18, 1600)
    experimental = gaussian(energy, 1.96, 0.095, 0.58) + gaussian(energy, 3.26, 0.135, 1.00)
    calculated = gaussian(energy, 2.43, 0.115, 0.58) + gaussian(energy, 3.84, 0.150, 1.00)

    ax.fill_between(energy, experimental, color=BLUE, alpha=0.12)
    ax.fill_between(energy, calculated, color=RED, alpha=0.12)
    ax.plot(energy, experimental, color=BLUE, linewidth=3.0, label="Experiment")
    ax.plot(energy, calculated, color=RED, linewidth=3.0, label="Calculation")
    ax.set_xlim(1.70, 4.15)
    ax.set_ylim(0.0, 1.20)
    ax.set_xlabel("Energy (eV)", fontsize=12, color=INK)
    ax.set_ylabel("Normalized absorbance", fontsize=12, color=INK)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10.5, colors=MUTED)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#AAB8C2")
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.75)
    ax.legend(loc="upper left", frameon=False, fontsize=11.5, ncol=2, handlelength=2.4)

    ax.text(1.96, 0.63, "Q", color=BLUE, fontsize=11, fontweight="bold", ha="center")
    ax.text(2.43, 0.63, "Q", color=RED, fontsize=11, fontweight="bold", ha="center")
    ax.text(3.26, 1.055, "Soret", color=BLUE, fontsize=11, fontweight="bold", ha="center")
    ax.text(3.84, 1.055, "Soret", color=RED, fontsize=11, fontweight="bold", ha="center")

    for start, end, ypos, label in [
        (1.96, 2.43, 0.79, "+0.47 eV"),
        (3.26, 3.84, 1.15, "+0.58 eV"),
    ]:
        ax.annotate(
            "",
            xy=(end, ypos),
            xytext=(start, ypos),
            arrowprops={"arrowstyle": "-|>", "color": NAVY, "lw": 2.0, "mutation_scale": 13},
        )
        ax.text((start + end) / 2, ypos + 0.025, label, color=NAVY, fontsize=10.5, fontweight="bold", ha="center")

    fig.text(0.073, 0.790, "PUBLISHED OFFSET", fontsize=10.5, color=BLUE, fontweight="bold")
    fig.text(0.073, 0.755, "Calculated peaks are shifted to higher energy", fontsize=16.5, color=INK, fontweight="bold")
    fig.text(
        0.297,
        0.180,
        "Schematic line shapes · peak positions from Figure 8",
        fontsize=9.2,
        color=MUTED,
        ha="center",
    )

    # Right: comparison scales, deliberately not an additive decomposition.
    bx = fig.add_axes([0.615, 0.245, 0.300, 0.480])
    bx.set_xlim(0.0, 0.66)
    bx.set_ylim(-0.70, 2.75)
    bx.set_yticks([2, 1, 0], ["TD-DFT\nmethod", "QM-region\nsize", "Protonation\nstate  ?"])
    bx.set_xlabel("Possible peak shift (eV)", fontsize=12, color=INK)
    bx.set_xticks(np.arange(0.0, 0.7, 0.1))
    bx.tick_params(axis="x", labelsize=10.5, colors=MUTED)
    bx.tick_params(axis="y", labelsize=11.5, colors=INK, length=0, pad=10)
    bx.spines[["top", "right", "left"]].set_visible(False)
    bx.spines["bottom"].set_color("#AAB8C2")
    bx.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)

    # Observed reference line.
    bx.axvline(observed_offset, color=RED, linewidth=2.2, linestyle=(0, (5, 4)), zorder=1)
    bx.text(
        observed_offset,
        2.52,
        "observed\n≈0.57",
        color=RED,
        fontsize=10.2,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    # Method and QM-region ranges from Falklof & Durbeej.
    for ypos, (low, high), color, text in [
        (2, method_range, GOLD, "0.30–0.60"),
        (1, qm_region_range, TEAL, "0.02–0.08"),
    ]:
        bx.plot([low, high], [ypos, ypos], color=color, linewidth=14, solid_capstyle="round", zorder=3)
        bx.scatter([low, high], [ypos, ypos], s=78, color=color, edgecolor="white", linewidth=1.2, zorder=4)
        bx.text(high + 0.018, ypos, text, color=color, fontsize=10.8, fontweight="bold", va="center")

    # Protonation: reported bilin range plus a faint unresolved residual envelope.
    bx.add_patch(
        Rectangle(
            (0.0, -0.17),
            protonation_pcb,
            0.34,
            facecolor=PURPLE,
            edgecolor=PURPLE,
            linewidth=1.6,
            alpha=0.18,
            hatch="///",
            zorder=2,
        )
    )
    bx.plot([protonation_bv, protonation_pcb], [0, 0], color=PURPLE, linewidth=5.5, alpha=0.55, solid_capstyle="round", zorder=3)
    bx.scatter([protonation_bv], [0], s=95, color=PURPLE, edgecolor="white", linewidth=1.2, zorder=4)
    bx.text(protonation_bv, 0.22, "BV ≈0.04", color=PURPLE, fontsize=9.8, fontweight="bold", ha="center")
    bx.text(protonation_pcb + 0.015, 0, "≤0.25 ?", color=PURPLE, fontsize=11.0, fontweight="bold", va="center")

    fig.text(0.603, 0.790, "PLAUSIBLE EFFECT SCALES", fontsize=10.5, color=BLUE, fontweight="bold")
    fig.text(0.603, 0.755, "How large can each contribution be?", fontsize=16.5, color=INK, fontweight="bold")

    # Minimal interpretation and sources.
    fig.text(
        0.500,
        0.102,
        "Ranges show scale—not an additive decomposition.  The Sandercyanin BPP → PPP shift remains the missing measurement.",
        fontsize=10.5,
        color=NAVY,
        fontweight="bold",
        ha="center",
    )
    fig.text(
        0.500,
        0.061,
        "Mainas et al. 2024 · Falklöf & Durbeej 2013 (DrBphP) · Matute et al. 2008 (BV/PCB protonation)",
        fontsize=8.8,
        color=MUTED,
        ha="center",
    )

    png_path = OUTDIR / "mainas-2024-spectra-offset-visual-summary.png"
    pdf_path = OUTDIR / "mainas-2024-spectra-offset-visual-summary.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    csv_path = OUTDIR / "mainas-2024-spectra-offset-visual-summary-data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "lower_eV", "upper_eV", "interpretation", "source"])
        writer.writerow(["observed_offset", observed_offset, observed_offset, "global visual shift used by Mainas et al.", "Mainas et al. 2024"])
        writer.writerow(["td_dft_method", *method_range, "long-range-corrected hybrid benchmark scale", "Falklof and Durbeej 2013"])
        writer.writerow(["qm_region_size", *qm_region_range, "reduction in blue shift on QM-region expansion", "Falklof and Durbeej 2013"])
        writer.writerow(["protonation_bv_zzzssa", protonation_bv, protonation_bv, "648 to 662 nm Q-band shift", "Matute et al. 2008"])
        writer.writerow(["protonation_related_bilin_range", protonation_bv, protonation_pcb, "BV-ZZZssa to PCB-ZZZasa Q-band scale; transfer uncertain", "Matute et al. 2008"])


if __name__ == "__main__":
    build_figure()
