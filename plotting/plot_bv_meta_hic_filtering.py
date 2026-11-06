#!/usr/bin/env python3
"""Render the hierarchical BV meta-hic run-filtering funnel."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
WINNERS = ((39, -17.0, 3.9342), (99, 7.2, 4.4517))


@dataclass(frozen=True)
class FilterStage:
    stage: str
    remaining: int
    removed: int
    removed_category: str


STAGES = (
    FilterStage("Initial ensemble", 200, 0, ""),
    FilterStage("Sustained CV diffusion", 108, 92, "Non-diffusive"),
    FilterStage("No counterion interference", 100, 8, "Counterion interference"),
    FilterStage("No tautomer side-reaction", 31, 69, "Tautomer side-reaction"),
    FilterStage("Mechanistic finalists", 4, 27, "Poor sampling"),
)


def save_stage_data(path: Path) -> None:
    """Save the exact counts displayed in the funnel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stage",
                "runs_remaining",
                "runs_removed_at_stage",
                "removed_category",
                "fraction_of_initial_remaining",
                "winner_runs",
                "pre_reaction_mean_helicity_deg",
                "endpoint_apparent_pka",
            ]
        )
        for stage in STAGES:
            writer.writerow(
                [
                    stage.stage,
                    stage.remaining,
                    stage.removed,
                    stage.removed_category,
                    f"{stage.remaining / STAGES[0].remaining:.6f}",
                    "",
                    "",
                    "",
                ]
            )
        writer.writerow(
            [
                "Conformational co-winners: runs 39 and 99",
                2,
                2,
                "Other finalists: 57;142",
                "0.010000",
                ";".join(str(run) for run, _, _ in WINNERS),
                ";".join(f"{helicity:.1f}" for _, helicity, _ in WINNERS),
                ";".join(f"{pka:.4f}" for _, _, pka in WINNERS),
            ]
        )


def add_box(
    axis,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
    title: str,
    count_text: str,
    title_color: str = "white",
    count_color: str = "white",
) -> None:
    """Add one rounded stage box and its two-line label."""
    patch = FancyBboxPatch(
        (center_x - width / 2.0, center_y - height / 2.0),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        linewidth=2.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        zorder=3,
    )
    axis.add_patch(patch)
    axis.text(
        center_x,
        center_y + 0.13,
        title,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=title_color,
        zorder=4,
    )
    axis.text(
        center_x,
        center_y - 0.17,
        count_text,
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=count_color,
        zorder=4,
    )


def add_winner_box(
    axis,
    center_x: float,
    run: int,
    color: str,
    helicity_deg: float,
    pka: float,
) -> None:
    """Add a conformational co-winner box with its helicity and pKa."""
    patch = FancyBboxPatch(
        (center_x - 1.78, -0.08),
        3.56,
        1.18,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        linewidth=3.0,
        facecolor="#FFF9E8",
        edgecolor=color,
        zorder=3,
    )
    axis.add_patch(patch)
    axis.text(
        center_x,
        0.77,
        f"CO-WINNER: RUN {run}",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=color,
        zorder=4,
    )
    axis.text(
        center_x,
        0.35,
        rf"mean helicity = {helicity_deg:+.1f}$^\circ$"
        "\n"
        rf"apparent p$K_a$ = {pka:.2f}",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#332600",
        linespacing=1.25,
        zorder=4,
    )


def render(output: Path, data_output: Path) -> None:
    """Render the portrait filtering diagram and save its source table."""
    plt.style.use(Path(__file__).with_name("lefteris.mplstyle"))
    fig, axis = plt.subplots(figsize=(9, 13), layout="constrained")
    axis.set_xlim(0.0, 10.0)
    axis.set_ylim(-0.65, 11.5)
    axis.axis("off")

    stage_y = (9.7, 7.9, 6.1, 4.3, 2.5)
    stage_width = (7.3, 6.0, 5.7, 4.2, 3.2)
    stage_colors = ("#244C66", "#2F6684", "#3B7F9C", "#4A96AF", "#58A9BE")
    stage_titles = (
        "Initial BV meta-hic ensemble",
        "Sustained diffusion in s(t)",
        "No counterion interference",
        "No tautomer side-reaction",
        "Mechanistic finalists",
    )

    for index, (stage, y, width, color, title) in enumerate(
        zip(STAGES, stage_y, stage_width, stage_colors, stage_titles)
    ):
        add_box(
            axis,
            4.35,
            y,
            width,
            1.05,
            color,
            "#173548",
            title,
            f"N = {stage.remaining}",
        )
        if index == 0:
            continue
        previous_y = stage_y[index - 1]
        arrow = FancyArrowPatch(
            (4.35, previous_y - 0.56),
            (4.35, y + 0.56),
            arrowstyle="-|>",
            mutation_scale=21,
            linewidth=2.2,
            color="#233746",
            zorder=2,
        )
        axis.add_patch(arrow)
        removed_fraction = 100.0 * stage.removed / STAGES[0].remaining
        axis.text(
            7.62,
            0.5 * (previous_y + y),
            f"−{stage.removed}  {stage.removed_category}\n({removed_fraction:.1f}% of 200)",
            ha="left",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color="#8A2E2E",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#F7E3E1",
                "edgecolor": "#C77D75",
                "linewidth": 1.5,
            },
        )

    branch_y = 1.48
    axis.plot(
        [4.35, 4.35],
        [stage_y[-1] - 0.56, branch_y],
        color="#7A5A00",
        linewidth=2.6,
        zorder=2,
    )
    for winner_x, winner_color in ((2.25, "green"), (6.45, "red")):
        axis.add_patch(
            FancyArrowPatch(
                (4.35, branch_y),
                (winner_x, 1.11),
                arrowstyle="-|>",
                mutation_scale=23,
                linewidth=2.6,
                color=winner_color,
                connectionstyle="arc3,rad=0.0",
                zorder=2,
            )
        )
    axis.text(
        7.62,
        1.65,
        "2 other finalists\nruns 57 and 142",
        ha="left",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#4F4F4F",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#ECECEC",
            "edgecolor": "#888888",
            "linewidth": 1.5,
        },
    )
    axis.text(
        4.35,
        1.58,
        "CONFORMATIONAL CONTROL OF p$K_a$",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#6B4F00",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5},
        zorder=5,
    )
    add_winner_box(
        axis, 2.25, WINNERS[0][0], "green", WINNERS[0][1], WINNERS[0][2]
    )
    add_winner_box(
        axis, 6.45, WINNERS[1][0], "red", WINNERS[1][1], WINNERS[1][2]
    )
    axis.text(
        4.35,
        -0.18,
        "Different BV helicities create different proton-binding environments",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#332600",
    )

    axis.set_title(
        "Hierarchical filtering of 200 BV meta-hic trajectories",
        fontsize=20,
        fontweight="bold",
        pad=18,
    )
    axis.text(
        4.35,
        10.65,
        "From the full ensemble to conformer-dependent acid–base behavior",
        ha="center",
        va="center",
        fontsize=14,
        color="#34495E",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)
    save_stage_data(data_output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/BV/meta-hic/hierarchical-filtering.png",
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=ROOT / "reports/BV/meta-hic/hierarchical-filtering.csv",
    )
    args = parser.parse_args()
    render(args.output, args.data_output)
    print(args.output)
    print(args.output.with_suffix(".svg"))
    print(args.data_output)


if __name__ == "__main__":
    main()
