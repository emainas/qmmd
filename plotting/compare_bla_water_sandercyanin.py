#!/usr/bin/env python3
"""Compare BLA/TPS protonation populations in water and Sandercyanin.

The two inputs are the aligned ``tps-titration-tautomers.csv`` tables written
by the aqueous BLA and Sandercyanin CpHMD reporters.  The figure separates the
overall PPP/deprotonated equilibrium from the tautomer distribution conditional
on being deprotonated.  It also writes the exact long-form and summary tables
used for plotting.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


STATE_ORDER = ("ppp", "app", "bpp", "cpp", "dpp")
DEPROTONATED_STATES = STATE_ORDER[1:]
STATE_LABELS = {
    "ppp": "PPP (protonated)",
    "app": "APP",
    "bpp": "BPP",
    "cpp": "CPP",
    "dpp": "DPP",
}
STATE_COLORS = {
    "ppp": "#202020",
    "app": "#D95F02",
    "bpp": "#7570B3",
    "cpp": "#159D82",
    "dpp": "#D45A9E",
}
STATE_MARKERS = {"ppp": "o", "app": "^", "bpp": "D", "cpp": "v", "dpp": "P"}


@dataclass(frozen=True)
class PopulationSeries:
    name: str
    path: Path
    ph: np.ndarray
    records: np.ndarray
    fractions: dict[str, np.ndarray]
    sem: dict[str, np.ndarray]
    conditional: dict[str, np.ndarray]


def _float(row: dict[str, str], key: str) -> float:
    value = row[key]
    return float(value) if value.lower() != "nan" else np.nan


def read_population_table(path: Path, name: str) -> PopulationSeries:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No population rows in {path}")

    fractions = {"ppp": np.asarray([_float(row, "ppp_fraction") for row in rows])}
    sem = {"ppp": np.asarray([_float(row, "ppp_sem") for row in rows])}
    conditional: dict[str, np.ndarray] = {}
    for state in DEPROTONATED_STATES:
        fractions[state] = np.asarray(
            [_float(row, f"{state}_fraction_total") for row in rows]
        )
        sem[state] = np.asarray([_float(row, f"{state}_sem_total") for row in rows])
        conditional[state] = np.asarray(
            [_float(row, f"{state}_fraction_of_deprotonated") for row in rows]
        )
    return PopulationSeries(
        name=name,
        path=path,
        ph=np.asarray([_float(row, "pH") for row in rows]),
        records=np.asarray([_float(row, "records") for row in rows]),
        fractions=fractions,
        sem=sem,
        conditional=conditional,
    )


def fit_apparent_pka(series: PopulationSeries) -> float:
    """Fit a one-proton Henderson-Hasselbalch midpoint by equal-pH SSE."""
    grid = np.linspace(float(series.ph.min()) - 3.0, float(series.ph.max()) + 3.0, 120_001)
    predicted = 1.0 / (1.0 + 10.0 ** (series.ph[:, None] - grid[None, :]))
    objective = np.sum((predicted - series.fractions["ppp"][:, None]) ** 2, axis=0)
    return float(grid[int(np.argmin(objective))])


def pooled_conditional_populations(series: PopulationSeries) -> dict[str, float]:
    counts = {
        state: float(np.sum(series.records * series.fractions[state]))
        for state in DEPROTONATED_STATES
    }
    total = sum(counts.values())
    if total <= 0.0:
        raise ValueError(f"{series.name} contains no deprotonated TPS samples")
    return {state: count / total for state, count in counts.items()}


def write_long_table(path: Path, series_collection: tuple[PopulationSeries, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "environment",
                "source_csv",
                "pH",
                "records",
                "state",
                "fraction_total",
                "block_sem",
                "fraction_conditional_on_deprotonated",
            ]
        )
        for series in series_collection:
            for index, ph in enumerate(series.ph):
                for state in STATE_ORDER:
                    conditional = (
                        ""
                        if state == "ppp" or not np.isfinite(series.conditional[state][index])
                        else f"{series.conditional[state][index]:.10f}"
                    )
                    writer.writerow(
                        [
                            series.name,
                            series.path,
                            f"{ph:.4f}",
                            int(series.records[index]),
                            state.upper(),
                            f"{series.fractions[state][index]:.10f}",
                            f"{series.sem[state][index]:.10f}",
                            conditional,
                        ]
                    )


def write_summary_table(
    path: Path,
    series_collection: tuple[PopulationSeries, ...],
    pkas: dict[str, float],
    pooled: dict[str, dict[str, float]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "environment",
                "apparent_pKa_equal_pH_SSE",
                "APP_conditional_fraction",
                "BPP_conditional_fraction",
                "CPP_conditional_fraction",
                "DPP_conditional_fraction",
            ]
        )
        for series in series_collection:
            writer.writerow(
                [
                    series.name,
                    f"{pkas[series.name]:.6f}",
                    *(f"{pooled[series.name][state]:.10f}" for state in DEPROTONATED_STATES),
                ]
            )


def plot_microstates(axis: plt.Axes, series: PopulationSeries, panel: str, pka: float) -> None:
    for state in STATE_ORDER:
        axis.errorbar(
            series.ph,
            series.fractions[state],
            yerr=series.sem[state],
            color=STATE_COLORS[state],
            marker=STATE_MARKERS[state],
            markersize=4.2,
            linewidth=1.7 if state != "ppp" else 2.4,
            elinewidth=0.9,
            capsize=2.0,
            alpha=0.98,
            label=STATE_LABELS[state],
        )
    axis.axhline(0.5, color="0.68", linestyle="--", linewidth=0.9, zorder=0)
    axis.axvline(pka, color="#B13A3A", linestyle=(0, (4, 3)), linewidth=1.2)
    midpoint = 0.5 * (float(series.ph.min()) + float(series.ph.max()))
    annotation_x = pka - 0.10 if pka > midpoint else pka
    axis.text(
        annotation_x,
        0.54,
        rf"apparent p$K_a$ = {pka:.2f}",
        color="#8E2828",
        fontsize=10.5,
        ha="right" if pka > midpoint else "center",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    axis.set_xlim(float(series.ph.min()) - 0.15, float(series.ph.max()) + 0.15)
    axis.set_ylim(-0.035, 1.045)
    axis.set_xlabel("pH")
    axis.set_ylabel("Molar fraction")
    axis.set_title(f"{panel}  {series.name}", loc="left", fontweight="bold")
    axis.grid(alpha=0.15)


def plot_midpoint_shift(
    axis: plt.Axes,
    water: PopulationSeries,
    protein: PopulationSeries,
    water_pka: float,
    protein_pka: float,
) -> None:
    colors = {water.name: "#2B73B6", protein.name: "#C43C39"}
    for series, pka in ((water, water_pka), (protein, protein_pka)):
        axis.errorbar(
            series.ph,
            series.fractions["ppp"],
            yerr=series.sem["ppp"],
            fmt="o",
            ms=4.0,
            capsize=2.0,
            elinewidth=0.8,
            color=colors[series.name],
            alpha=0.85,
        )
        dense_ph = np.linspace(float(series.ph.min()), float(series.ph.max()), 500)
        fitted = 1.0 / (1.0 + 10.0 ** (dense_ph - pka))
        axis.plot(dense_ph, fitted, color=colors[series.name], linewidth=2.6, label=series.name)
        axis.axvline(pka, color=colors[series.name], linestyle=(0, (3, 3)), linewidth=1.0)

    delta_pka = protein_pka - water_pka
    delta_g = 2.302585093 * 0.001987204258 * 300.0 * delta_pka
    axis.annotate(
        "",
        xy=(protein_pka, 0.50),
        xytext=(water_pka, 0.50),
        arrowprops={"arrowstyle": "<->", "color": "#333333", "lw": 1.7},
    )
    axis.text(
        (water_pka + protein_pka) / 2.0,
        0.55,
        rf"protein shift: $\Delta$p$K_a$ = +{delta_pka:.2f}",
        ha="center",
        va="bottom",
        fontsize=11.0,
        fontweight="bold",
    )
    axis.text(
        (water_pka + protein_pka) / 2.0,
        0.43,
        rf"equivalent to {delta_g:.1f} kcal mol$^{{-1}}$ at 300 K",
        ha="center",
        va="top",
        fontsize=9.7,
        color="0.25",
    )
    axis.set_xlim(min(water.ph.min(), protein.ph.min()) - 0.25, max(water.ph.max(), protein.ph.max()) + 0.25)
    axis.set_ylim(-0.035, 1.045)
    axis.set_xlabel("pH")
    axis.set_ylabel("PPP fraction")
    axis.set_title("C  Protein entry stabilizes protonated BLA", loc="left", fontweight="bold")
    axis.legend(loc="upper right", fontsize=10.5)
    axis.grid(alpha=0.15)


def plot_conditional_composition(
    axis: plt.Axes,
    water: PopulationSeries,
    protein: PopulationSeries,
    pooled: dict[str, dict[str, float]],
) -> None:
    environments = (water.name, protein.name)
    y_positions = (1.0, 0.0)
    for y_position, environment in zip(y_positions, environments):
        left = 0.0
        for state in DEPROTONATED_STATES:
            fraction = pooled[environment][state]
            axis.barh(
                y_position,
                fraction,
                left=left,
                height=0.56,
                color=STATE_COLORS[state],
                edgecolor="white",
                linewidth=1.0,
            )
            if fraction >= 0.045:
                axis.text(
                    left + fraction / 2.0,
                    y_position,
                    f"{state.upper()}\n{100.0 * fraction:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if state in {"bpp", "cpp", "dpp"} else "black",
                    fontsize=9.2,
                    fontweight="bold",
                )
            left += fraction
    protein_bpp = 100.0 * pooled[protein.name]["bpp"]
    axis.text(
        0.985,
        -0.42,
        f"residual BPP = {protein_bpp:.1f}%",
        ha="right",
        va="center",
        fontsize=9.2,
        color=STATE_COLORS["bpp"],
    )
    axis.annotate(
        "broad tautomer ensemble",
        xy=(0.50, 1.31),
        xytext=(0.50, 1.31),
        ha="center",
        va="bottom",
        fontsize=10.0,
        color="0.25",
    )
    axis.annotate(
        "CPP selected by the pocket",
        xy=(0.50, -0.31),
        xytext=(0.50, -0.31),
        ha="center",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=STATE_COLORS["cpp"],
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.65, 1.65)
    axis.set_yticks(y_positions, ("Water", "Protein"))
    axis.set_xticks(np.linspace(0.0, 1.0, 6), [f"{value:.0%}" for value in np.linspace(0.0, 1.0, 6)])
    axis.set_xlabel("Fraction within deprotonated TPS")
    axis.set_title("D  The pocket selects one tautomer", loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=0.15)


def make_figure(
    destination: Path,
    style: Path,
    water: PopulationSeries,
    protein: PopulationSeries,
    pkas: dict[str, float],
    pooled: dict[str, dict[str, float]],
) -> None:
    plt.style.use(style)
    figure = plt.figure(figsize=(14.2, 9.2))
    grid = GridSpec(2, 2, figure=figure, height_ratios=(1.0, 0.92), hspace=0.38, wspace=0.28)
    water_axis = figure.add_subplot(grid[0, 0])
    protein_axis = figure.add_subplot(grid[0, 1])
    shift_axis = figure.add_subplot(grid[1, 0])
    composition_axis = figure.add_subplot(grid[1, 1])

    plot_microstates(water_axis, water, "A", pkas[water.name])
    plot_microstates(protein_axis, protein, "B", pkas[protein.name])
    plot_midpoint_shift(shift_axis, water, protein, pkas[water.name], pkas[protein.name])
    plot_conditional_composition(composition_axis, water, protein, pooled)

    handles, labels = water_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.892),
        ncol=5,
        fontsize=10.5,
        handletextpad=0.5,
        columnspacing=1.25,
    )
    figure.suptitle(
        "The Sandercyanin pocket rewires biliverdin protonation",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.956,
        "BLA/TPS in water (fixed deprotonated PRX tails)  →  BLA/TPS inside the protein",
        ha="center",
        va="center",
        fontsize=12.5,
        color="0.25",
    )
    figure.subplots_adjust(top=0.825)
    figure.savefig(destination, dpi=300)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--water", type=Path, required=True)
    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--style", type=Path, default=Path("plotting/lefteris.mplstyle"))
    args = parser.parse_args()

    water = read_population_table(args.water.resolve(), "Water")
    protein = read_population_table(args.protein.resolve(), "Sandercyanin")
    series_collection = (water, protein)
    pkas = {series.name: fit_apparent_pka(series) for series in series_collection}
    pooled = {
        series.name: pooled_conditional_populations(series) for series in series_collection
    }

    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    write_long_table(destination / "bla-environment-populations.csv", series_collection)
    write_summary_table(
        destination / "bla-environment-summary.csv", series_collection, pkas, pooled
    )
    make_figure(
        destination / "bla-water-to-sandercyanin.png",
        args.style.resolve(),
        water,
        protein,
        pkas,
        pooled,
    )
    print(f"OK: wrote BLA environment comparison to {destination}")
    print(
        f"Apparent pKa: water {pkas[water.name]:.2f}; "
        f"Sandercyanin {pkas[protein.name]:.2f}; "
        f"shift {pkas[protein.name] - pkas[water.name]:+.2f}"
    )
    print(
        "Conditional CPP population: "
        f"water {100.0 * pooled[water.name]['cpp']:.1f}%; "
        f"Sandercyanin {100.0 * pooled[protein.name]['cpp']:.1f}%"
    )


if __name__ == "__main__":
    main()
