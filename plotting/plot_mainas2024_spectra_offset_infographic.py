#!/usr/bin/env python3
"""Build a literature-grounded infographic for the Mainas et al. spectral offset."""

from __future__ import annotations

import csv
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "reports" / "Mainas2024" / "spectra-offset"

NAVY = "#17324D"
BLUE = "#247BA0"
CYAN = "#56B4D3"
ORANGE = "#E76F51"
AMBER = "#E9A23B"
PURPLE = "#745296"
TEAL = "#2A9D8F"
GREEN = "#4E8F70"
INK = "#17212B"
MUTED = "#596773"
PALE = "#F4F7F9"
WHITE = "#FFFFFF"
RED = "#C94C4C"


def rounded_box(
    fig: plt.Figure,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = "none",
    linewidth: float = 1.0,
    radius: float = 0.018,
    zorder: int = 0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        transform=fig.transFigure,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    fig.add_artist(patch)
    return patch


def add_wrapped_text(
    fig: plt.Figure,
    x: float,
    y: float,
    text: str,
    *,
    width: int,
    fontsize: float,
    color: str = INK,
    weight: str = "normal",
    linespacing: float = 1.22,
    va: str = "top",
    ha: str = "left",
) -> None:
    fig.text(
        x,
        y,
        fill(text, width=width),
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        linespacing=linespacing,
        va=va,
        ha=ha,
    )


def draw_bilin_icon(fig: plt.Figure, x: float, y: float, scale: float = 1.0) -> None:
    """Draw a compact four-ring bilin cartoon, highlighting ring B and its proton."""
    centers = [x + i * 0.031 * scale for i in range(4)]
    for i, cx in enumerate(centers):
        radius = 0.014 * scale
        points = []
        for k in range(5):
            angle = 1.5708 + 2.0 * 3.14159 * k / 5.0
            points.append((cx + radius * __import__("math").cos(angle), y + radius * __import__("math").sin(angle)))
        fig.add_artist(
            Polygon(
                points,
                closed=True,
                transform=fig.transFigure,
                facecolor="#FFF4E8" if i == 1 else WHITE,
                edgecolor=ORANGE if i == 1 else NAVY,
                linewidth=2.2,
                zorder=3,
            )
        )
        fig.text(cx, y - 0.002, "N", ha="center", va="center", fontsize=8.4 * scale, color=NAVY, fontweight="bold")
        if i < 3:
            fig.add_artist(
                FancyArrowPatch(
                    (cx + 0.012 * scale, y),
                    (centers[i + 1] - 0.012 * scale, y),
                    transform=fig.transFigure,
                    arrowstyle="-",
                    color=NAVY,
                    linewidth=2.0,
                )
            )
    bx = centers[1]
    fig.text(bx, y + 0.027 * scale, "+H", color=ORANGE, fontsize=10 * scale, ha="center", fontweight="bold")
    fig.add_artist(
        FancyArrowPatch(
            (bx, y + 0.023 * scale),
            (bx, y + 0.012 * scale),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=9,
            color=ORANGE,
            linewidth=1.5,
        )
    )


def draw_method_icon(fig: plt.Figure, x: float, y: float) -> None:
    fig.text(x, y, "ω", fontsize=30, color=BLUE, fontweight="bold", ha="center", va="center")
    fig.add_artist(
        FancyArrowPatch(
            (x - 0.025, y - 0.026),
            (x + 0.029, y - 0.026),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=11,
            color=CYAN,
            linewidth=2.2,
        )
    )
    fig.text(x + 0.002, y - 0.043, "exact exchange →", fontsize=7.8, color=MUTED, ha="center")


def draw_qm_icon(fig: plt.Figure, x: float, y: float) -> None:
    fig.add_artist(Circle((x, y), 0.023, transform=fig.transFigure, facecolor="#E7F4EF", edgecolor=TEAL, linewidth=2.0))
    fig.add_artist(Circle((x, y), 0.010, transform=fig.transFigure, facecolor=TEAL, edgecolor="none"))
    for dx, dy in [(-0.036, 0.022), (0.036, 0.018), (-0.032, -0.028), (0.033, -0.027)]:
        fig.add_artist(Circle((x + dx, y + dy), 0.005, transform=fig.transFigure, facecolor=GREEN, edgecolor="none"))
        fig.add_artist(
            FancyArrowPatch(
                (x + 0.7 * dx, y + 0.7 * dy),
                (x + dx, y + dy),
                transform=fig.transFigure,
                arrowstyle="-",
                color=GREEN,
                linewidth=1.2,
                linestyle=(0, (2, 2)),
            )
        )
    fig.text(x, y, "QM", color=WHITE, fontsize=7.8, fontweight="bold", ha="center", va="center")


def draw_number_badge(fig: plt.Figure, x: float, y: float, number: str, color: str) -> None:
    fig.add_artist(Circle((x, y), 0.016, transform=fig.transFigure, facecolor=color, edgecolor="none", zorder=8))
    fig.text(x, y - 0.001, number, color=WHITE, fontsize=12, fontweight="bold", ha="center", va="center", zorder=9)


def build_infographic() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(16, 10), dpi=180)

    # Header
    fig.text(
        0.045,
        0.955,
        "Explaining the spectral offset in Mainas et al. (2024)",
        fontsize=26,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    fig.text(
        0.046,
        0.913,
        "Why were the calculated Sandercyanin absorption bands blue-shifted by ≈0.57 eV?",
        fontsize=14.5,
        color=MUTED,
        va="top",
    )
    fig.add_artist(
        FancyArrowPatch(
            (0.046, 0.894),
            (0.955, 0.894),
            transform=fig.transFigure,
            arrowstyle="-",
            color="#D9E1E6",
            linewidth=1.5,
        )
    )

    # Observed discrepancy panel.
    rounded_box(fig, (0.04, 0.700), 0.605, 0.175, facecolor="#F6F9FB", edgecolor="#DCE5EA", linewidth=1.1)
    fig.text(0.058, 0.855, "THE OBSERVATION", fontsize=10.5, fontweight="bold", color=BLUE, va="top")
    fig.text(0.058, 0.830, "A nearly rigid blue offset in the protein spectrum", fontsize=16, fontweight="bold", color=INK, va="top")

    ax = fig.add_axes([0.067, 0.715, 0.555, 0.095])
    ax.set_xlim(1.75, 4.06)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([1, 0], ["Q band", "Soret band"])
    ax.set_xlabel("Peak energy (eV)", fontsize=10.5, color=MUTED, labelpad=2)
    ax.set_xticks([2.0, 2.5, 3.0, 3.5, 4.0])
    ax.tick_params(axis="both", labelsize=9.5, colors=MUTED, length=3)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#AAB7C1")
    ax.grid(axis="x", color="#DCE5EA", linewidth=0.8, alpha=0.8)

    peaks = {
        "Q band": (1.96, 2.43, 1),
        "Soret band": (3.26, 3.84, 0),
    }
    for label, (experimental, calculated, ypos) in peaks.items():
        ax.annotate(
            "",
            xy=(calculated, ypos),
            xytext=(experimental, ypos),
            arrowprops={"arrowstyle": "-|>", "lw": 3.0, "color": ORANGE, "shrinkA": 4, "shrinkB": 4},
        )
        ax.scatter([experimental], [ypos], s=78, color=TEAL, edgecolor=WHITE, linewidth=1.2, zorder=5)
        ax.scatter([calculated], [ypos], s=78, color=ORANGE, edgecolor=WHITE, linewidth=1.2, zorder=5)
        ax.text(experimental, ypos + 0.22, f"exp. {experimental:.2f}", ha="center", va="bottom", fontsize=8.8, color=TEAL, fontweight="bold")
        ax.text(calculated, ypos + 0.22, f"calc. {calculated:.2f}", ha="center", va="bottom", fontsize=8.8, color=ORANGE, fontweight="bold")
        ax.text(
            (experimental + calculated) / 2,
            ypos - 0.25,
            f"Δ = +{calculated - experimental:.2f} eV",
            ha="center",
            va="top",
            fontsize=9.2,
            color=RED,
            fontweight="bold",
        )

    # Peer-review panel.
    rounded_box(fig, (0.665, 0.700), 0.295, 0.175, facecolor="#FFF8F0", edgecolor="#F2D6B5", linewidth=1.1)
    fig.text(0.683, 0.855, "THE REVIEWERS NOTICED", fontsize=10.5, fontweight="bold", color=ORANGE, va="top")
    add_wrapped_text(
        fig,
        0.683,
        0.824,
        "Reviewer 1: the 0.57 eV blue shift was unusually large relative to comparable biliverdin TD-DFT studies.",
        width=46,
        fontsize=10.4,
        weight="bold",
        linespacing=1.25,
    )
    add_wrapped_text(
        fig,
        0.683,
        0.763,
        "Reviewer 2: questioned the deprotonated B ring and proposed that this protonation choice could contribute to the discrepancy.",
        width=46,
        fontsize=10.4,
        weight="bold",
        linespacing=1.25,
    )

    # Three explanatory cards.
    card_y = 0.333
    card_h = 0.335
    card_w = 0.287
    card_xs = [0.04, 0.3565, 0.673]
    card_colors = [PURPLE, BLUE, TEAL]
    card_fills = ["#F7F1FA", "#EFF7FB", "#EFF8F5"]
    for x, edge, face in zip(card_xs, card_colors, card_fills, strict=True):
        rounded_box(fig, (x, card_y), card_w, card_h, facecolor=face, edgecolor=edge, linewidth=1.4)

    # Card 1: protonation.
    draw_number_badge(fig, 0.066, 0.636, "1", PURPLE)
    fig.text(0.090, 0.646, "PROTONATION STATE", fontsize=10.5, color=PURPLE, fontweight="bold", va="top")
    fig.text(0.090, 0.621, "A newly identified model mismatch", fontsize=14.0, color=INK, fontweight="bold", va="top")
    draw_bilin_icon(fig, 0.100, 0.570, scale=1.0)
    fig.text(0.244, 0.578, "BPP  →  PPP", fontsize=13, color=ORANGE, fontweight="bold", ha="center")
    fig.text(0.254, 0.556, "restore B-ring proton", fontsize=8.4, color=MUTED, ha="center")
    add_wrapped_text(
        fig,
        0.060,
        0.523,
        "Published spectra used a B-ring-deprotonated tetrapyrrole core (BPP). Our new CpHMD sampling instead strongly favors fully protonated PPP inside Sandercyanin.",
        width=47,
        fontsize=10.0,
        linespacing=1.25,
    )
    add_wrapped_text(
        fig,
        0.060,
        0.431,
        "This supports Reviewer 2's concern, but the protein spectral effect remains unmeasured.",
        width=53,
        fontsize=9.1,
        color=PURPLE,
        weight="bold",
    )
    rounded_box(fig, (0.060, 0.342), 0.246, 0.042, facecolor=WHITE, edgecolor="#D8C5E1", linewidth=0.9, radius=0.012)
    fig.text(0.183, 0.372, "DECISIVE TEST", fontsize=8.1, color=PURPLE, fontweight="bold", ha="center", va="top")
    fig.text(0.183, 0.353, "Same protein snapshots: BPP vs PPP", fontsize=9.2, color=INK, fontweight="bold", ha="center", va="top")

    # Card 2: method.
    draw_number_badge(fig, 0.3825, 0.636, "2", BLUE)
    fig.text(0.407, 0.646, "EXCITED-STATE METHOD", fontsize=10.5, color=BLUE, fontweight="bold", va="top")
    fig.text(0.407, 0.621, "A credible source of systematic blue bias", fontsize=14.0, color=INK, fontweight="bold", va="top")
    draw_method_icon(fig, 0.408, 0.560)
    fig.text(0.454, 0.579, "Published protocol", fontsize=8.6, color=MUTED, fontweight="bold")
    fig.text(0.454, 0.552, "ωB97X(ω=0.3)/6-31G*", fontsize=11.2, color=BLUE, fontweight="bold")
    add_wrapped_text(
        fig,
        0.377,
        0.518,
        "Falklöf & Durbeej found that long-range-corrected hybrids blue-shifted the Q band by ≈0.5 eV and the Soret band by ≈0.3–0.6 eV; pure GGAs were within ≈0.15 eV.",
        width=47,
        fontsize=9.9,
        linespacing=1.24,
    )
    add_wrapped_text(
        fig,
        0.377,
        0.421,
        "ωB97X was not tested there. Still, its range-separated form makes method bias a plausible major contributor.",
        width=53,
        fontsize=9.0,
        color=BLUE,
        weight="bold",
    )
    rounded_box(fig, (0.377, 0.342), 0.246, 0.042, facecolor=WHITE, edgecolor="#BDD8E5", linewidth=0.9, radius=0.012)
    fig.text(0.500, 0.372, "DECISIVE TEST", fontsize=8.1, color=BLUE, fontweight="bold", ha="center", va="top")
    fig.text(0.500, 0.353, "Same snapshots: functional benchmark", fontsize=9.2, color=INK, fontweight="bold", ha="center", va="top")

    # Card 3: QM region.
    draw_number_badge(fig, 0.699, 0.636, "3", TEAL)
    fig.text(0.723, 0.646, "QM-REGION DEFINITION", fontsize=10.5, color=TEAL, fontweight="bold", va="top")
    fig.text(0.723, 0.621, "Important to test; likely a smaller term", fontsize=14.0, color=INK, fontweight="bold", va="top")
    draw_qm_icon(fig, 0.723, 0.559)
    fig.text(0.764, 0.578, "Boundary + nearby residues", fontsize=10.5, color=TEAL, fontweight="bold")
    fig.text(0.764, 0.554, "can alter polarization and state mixing", fontsize=8.6, color=MUTED)
    add_wrapped_text(
        fig,
        0.693,
        0.518,
        "Reviewer 2 questioned the QM-region composition, particularly inclusion of the nonconjugated propionates. The published work also found dark low-lying transitions involving aromatic side chains.",
        width=47,
        fontsize=9.9,
        linespacing=1.24,
    )
    add_wrapped_text(
        fig,
        0.693,
        0.421,
        "Benchmark QM-region effect: only 0.02–0.08 eV. Likely secondary; verify explicitly in Sandercyanin.",
        width=53,
        fontsize=9.0,
        color=TEAL,
        weight="bold",
    )
    rounded_box(fig, (0.693, 0.342), 0.246, 0.042, facecolor=WHITE, edgecolor="#B9DCD2", linewidth=0.9, radius=0.012)
    fig.text(0.816, 0.372, "DECISIVE TEST", fontsize=8.1, color=TEAL, fontweight="bold", ha="center", va="top")
    fig.text(0.816, 0.353, "Nested QM regions on identical frames", fontsize=9.2, color=INK, fontweight="bold", ha="center", va="top")

    # Synthesis panel.
    rounded_box(fig, (0.04, 0.080), 0.92, 0.220, facecolor=NAVY, edgecolor=NAVY, linewidth=1.0)
    fig.text(0.065, 0.275, "MOST DEFENSIBLE WORKING PICTURE", fontsize=10.5, color="#9ED9EA", fontweight="bold", va="top")
    fig.text(0.065, 0.243, "The 0.57 eV offset is probably not a single-cause failure", fontsize=18, color=WHITE, fontweight="bold", va="top")

    # Equation-like synthesis.
    terms = [
        (0.090, "systematic\nTD-DFT bias", CYAN),
        (0.304, "BPP → PPP\nmodel mismatch", "#C6A1D5"),
        (0.518, "QM-region +\ngeometry effects", "#8DD0B9"),
    ]
    for x, label, color in terms:
        rounded_box(fig, (x, 0.132), 0.168, 0.065, facecolor="#24435D", edgecolor=color, linewidth=1.3, radius=0.013)
        fig.text(x + 0.084, 0.165, label, fontsize=10.8, color=WHITE, fontweight="bold", ha="center", va="center", linespacing=1.15)
    fig.text(0.278, 0.165, "+", fontsize=22, color=WHITE, fontweight="bold", ha="center", va="center")
    fig.text(0.492, 0.165, "+", fontsize=22, color=WHITE, fontweight="bold", ha="center", va="center")
    fig.text(0.710, 0.165, "=", fontsize=22, color=WHITE, fontweight="bold", ha="center", va="center")
    rounded_box(fig, (0.742, 0.132), 0.168, 0.065, facecolor=ORANGE, edgecolor="#FFD6C9", linewidth=1.3, radius=0.013)
    fig.text(0.826, 0.165, "observed\n≈0.57 eV", fontsize=12.0, color=WHITE, fontweight="bold", ha="center", va="center")

    fig.text(
        0.065,
        0.100,
        "Priority experiment: a factorial comparison on matched protein snapshots — protonation state × functional × QM-region size.",
        fontsize=10.8,
        color="#DDEAF1",
        fontweight="bold",
        va="bottom",
    )

    # References and interpretation boundary.
    fig.text(
        0.045,
        0.048,
        "Sources: Mainas et al., J. Phys. Chem. B (2024), DOI 10.1021/acs.jpcb.4c06722; anonymous reviewer reports; "
        "Falklöf & Durbeej, J. Comput. Chem. 34, 1363–1374 (2013), DOI 10.1002/jcc.23265.",
        fontsize=8.2,
        color=MUTED,
        va="bottom",
    )
    fig.text(
        0.045,
        0.025,
        "Interpretation boundary: Falklöf studied DrBphP, not Sandercyanin; its numerical effects motivate tests but do not partition the Sandercyanin error.",
        fontsize=8.2,
        color=RED,
        fontweight="bold",
        va="bottom",
    )

    png = OUTDIR / "mainas-2024-spectra-offset-infographic.png"
    pdf = OUTDIR / "mainas-2024-spectra-offset-infographic.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    csv_path = OUTDIR / "mainas-2024-spectra-offset-evidence.csv"
    rows = [
        ("Mainas 2024", "protein Q band", "experimental_peak_eV", 1.96, "Figure 8B"),
        ("Mainas 2024", "protein Q band", "calculated_peak_eV", 2.43, "Figure 8B"),
        ("Mainas 2024", "protein Q band", "blue_shift_eV", 0.47, "calculated - experimental"),
        ("Mainas 2024", "protein Soret band", "experimental_peak_eV", 3.26, "Figure 8B"),
        ("Mainas 2024", "protein Soret band", "calculated_peak_eV", 3.84, "Figure 8B"),
        ("Mainas 2024", "protein Soret band", "blue_shift_eV", 0.58, "calculated - experimental"),
        ("Falklof 2013", "Q band", "LRC_hybrid_error_approx_eV", 0.50, "M06-HF/LC-BLYP/CAM-B3LYP/LC-wPBE class"),
        ("Falklof 2013", "Soret band", "LRC_hybrid_error_upper_eV", 0.60, "up to 0.5-0.6 eV; CAM-B3LYP about 0.3 eV"),
        ("Falklof 2013", "Q and Soret", "best_functional_error_upper_eV", 0.15, "pure GGA Q; non-LRC Soret"),
        ("Falklof 2013", "QM region expansion", "blue_shift_reduction_lower_eV", 0.02, "QM1 to QM2"),
        ("Falklof 2013", "QM region expansion", "blue_shift_reduction_upper_eV", 0.08, "QM1 to QM2"),
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "quantity", "metric", "value_eV", "note"])
        writer.writerows(rows)


if __name__ == "__main__":
    build_infographic()
