#!/usr/bin/env python3
"""Draw the BV microscopic deprotonation network from LCOD PMF gaps."""

from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


R_KCAL_MOL_K = 0.00198720425864083


@dataclass(frozen=True)
class Edge:
    left: str
    right: str
    delta_g_right_minus_left: float


def read_edges(path: Path) -> tuple[list[Edge], float]:
    edges: list[Edge] = []
    temperatures: set[float] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            left = row["negative_state"].strip()
            right = row["positive_state"].strip()
            edges.append(
                Edge(
                    left,
                    right,
                    float(row["delta_g_positive_minus_negative_kcal_mol"]),
                )
            )
            temperatures.add(float(row["temperature_K"]))
    if not edges:
        raise ValueError(f"no PMF edges found in {path}")
    if len(temperatures) != 1:
        raise ValueError(f"PMF comparisons do not share one temperature: {temperatures}")
    return edges, temperatures.pop()


def relative_free_energies(edges: list[Edge], anchor: str) -> dict[str, float]:
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for edge in edges:
        adjacency.setdefault(edge.left, []).append(
            (edge.right, edge.delta_g_right_minus_left)
        )
        adjacency.setdefault(edge.right, []).append(
            (edge.left, -edge.delta_g_right_minus_left)
        )
    if anchor not in adjacency:
        raise ValueError(f"anchor state {anchor!r} is absent from PMF network")
    energies = {anchor: 0.0}
    queue = deque([anchor])
    while queue:
        state = queue.popleft()
        for neighbor, difference in adjacency[state]:
            candidate = energies[state] + difference
            if neighbor in energies:
                if not math.isclose(energies[neighbor], candidate, abs_tol=1.0e-6):
                    raise ValueError(f"thermodynamic cycle does not close at {neighbor}")
                continue
            energies[neighbor] = candidate
            queue.append(neighbor)
    if set(energies) != set(adjacency):
        raise ValueError("PMF network is disconnected")
    return energies


def microscopic_pkas(
    energies: dict[str, float], anchor: str, anchor_pka: float, temperature: float
) -> dict[str, float]:
    conversion = R_KCAL_MOL_K * temperature * math.log(10.0)
    return {
        state: anchor_pka + (energy - energies[anchor]) / conversion
        for state, energy in energies.items()
    }


def site_letter(state: str) -> str:
    if len(state) != 3 or not state.endswith("PP"):
        raise ValueError(f"cannot infer deprotonated site from state {state!r}")
    return state[0]


def draw_node(
    ax: plt.Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
    text: str,
    textcolor: str = "black",
) -> None:
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2.0, y - height / 2.0),
        width,
        height,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2.2,
        zorder=4,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=11,
        color=textcolor,
        zorder=5,
        linespacing=1.25,
    )


def plot_network(
    output: Path,
    edges: list[Edge],
    energies: dict[str, float],
    pkas: dict[str, float],
    anchor: str,
    anchor_pka: float,
    temperature: float,
) -> None:
    order = ("APP", "BPP", "CPP", "DPP")
    missing = set(order) - set(pkas)
    if missing:
        raise ValueError(f"missing expected microstates: {sorted(missing)}")
    positions = {
        "APP": (1.6, 1.35),
        "BPP": (4.65, 1.35),
        "CPP": (7.70, 1.35),
        "DPP": (10.75, 1.35),
    }
    colors = {
        "APP": ("#FDE2E1", "#D62728", "black"),
        "BPP": ("#DFECF7", "#1F77B4", "black"),
        "CPP": ("#E3F3E2", "#2CA02C", "black"),
        "DPP": ("#E5E5E5", "#111111", "black"),
    }

    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax.set_xlim(0.0, 12.35)
    ax.set_ylim(0.0, 7.0)
    ax.axis("off")

    bv_center = (6.175, 5.85)
    draw_node(
        ax,
        bv_center,
        2.65,
        0.95,
        "#FFF2C7",
        "#8A6D1D",
        "BV\nfully N-protonated parent",
    )

    for state in order:
        site = site_letter(state)
        x, y = positions[state]
        face, edgecolor, textcolor = colors[state]
        anchor_mark = "  (anchor)" if state == anchor else ""
        draw_node(
            ax,
            (x, y),
            2.15,
            1.15,
            face,
            edgecolor,
            f"{state}\nN{site}-deprotonated{anchor_mark}\n"
            f"micro-pKₐ,{site} = {pkas[state]:.2f}",
            textcolor,
        )
        arrow = FancyArrowPatch(
            (bv_center[0], bv_center[1] - 0.50),
            (x, y + 0.62),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color=edgecolor,
            connectionstyle=f"arc3,rad={0.04 * (x - bv_center[0]):.3f}",
            zorder=2,
        )
        ax.add_patch(arrow)
        fraction = 0.57
        label_x = bv_center[0] + fraction * (x - bv_center[0])
        label_y = (bv_center[1] - 0.50) + fraction * (y + 0.62 - (bv_center[1] - 0.50))
        ax.text(
            label_x,
            label_y,
            f"−H⁺ at N{site}",
            ha="center",
            va="center",
            fontsize=9.5,
            color=edgecolor,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
            zorder=3,
        )

    edge_lookup = {(edge.left, edge.right): edge for edge in edges}
    for left, right in zip(order[:-1], order[1:], strict=True):
        edge = edge_lookup.get((left, right))
        sign = 1.0
        if edge is None:
            edge = edge_lookup.get((right, left))
            sign = -1.0
        if edge is None:
            raise ValueError(f"missing LCOD edge between {left} and {right}")
        delta_g = sign * edge.delta_g_right_minus_left
        delta_pka = pkas[right] - pkas[left]
        x_left = positions[left][0] + 1.10
        x_right = positions[right][0] - 1.10
        y = positions[left][1]
        ax.add_patch(
            FancyArrowPatch(
                (x_left, y),
                (x_right, y),
                arrowstyle="<->",
                mutation_scale=12,
                linewidth=1.25,
                color="#555555",
                zorder=3,
            )
        )
        ax.text(
            0.5 * (x_left + x_right),
            y + 0.82,
            f"LCOD {left} ⇌ {right}\n"
            f"G({right})−G({left}) = {delta_g:+.3f} kcal mol⁻¹\n"
            f"pKₐ,{site_letter(right)}−pKₐ,{site_letter(left)} = {delta_pka:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8.4,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#C8C8C8",
                "alpha": 0.95,
            },
            zorder=6,
        )

    ax.text(
        6.175,
        6.72,
        "BV microscopic deprotonation network from LCOD thermodynamic cycles",
        ha="center",
        va="center",
        fontsize=17,
    )
    ax.text(
        6.175,
        0.17,
        f"Anchor: pKₐ,C = {anchor_pka:.2f} at {temperature:.0f} K.  "
        "For the common BV parent: pKₐ,X−pKₐ,C = "
        "[G(XPP)−G(CPP)]/(RT ln 10).",
        ha="center",
        va="bottom",
        fontsize=9.3,
    )
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_results(
    path: Path,
    energies: dict[str, float],
    pkas: dict[str, float],
    anchor: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "microstate",
                "deprotonated_site",
                f"delta_g_vs_{anchor}_kcal_mol",
                f"delta_pka_vs_{anchor}",
                "microscopic_pka",
            ]
        )
        for state in ("APP", "BPP", "CPP", "DPP"):
            writer.writerow(
                [
                    state,
                    site_letter(state),
                    f"{energies[state] - energies[anchor]:.8f}",
                    f"{pkas[state] - pkas[anchor]:.8f}",
                    f"{pkas[state]:.8f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pmf-summary",
        type=Path,
        default=Path("reports/tetrapyrrole_lcod_pmf/three_pmf_summary.csv"),
    )
    parser.add_argument("--anchor-state", default="CPP")
    parser.add_argument("--anchor-pka", type=float, default=4.1)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("reports/tetrapyrrole_lcod_pmf")
    )
    parser.add_argument("--style", type=Path, default=Path("lefteris.mplstyle"))
    args = parser.parse_args()

    if args.style.is_file():
        plt.style.use(args.style)
    edges, temperature = read_edges(args.pmf_summary)
    energies = relative_free_energies(edges, args.anchor_state)
    pkas = microscopic_pkas(
        energies, args.anchor_state, args.anchor_pka, temperature
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.out_dir / "network.png"
    data_path = args.out_dir / "network_micro_pkas.csv"
    plot_network(
        figure_path,
        edges,
        energies,
        pkas,
        args.anchor_state,
        args.anchor_pka,
        temperature,
    )
    write_results(data_path, energies, pkas, args.anchor_state)
    print(f"figure: {figure_path.resolve()}")
    print(f"micro-pKas: {data_path.resolve()}")


if __name__ == "__main__":
    main()
