#!/usr/bin/env python3
"""Draw BV/deprotonated-state torsion–deprotonation thermodynamic cycles."""

from __future__ import annotations

import csv
import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmmd.us_wham import compute_thermodynamics, load_config


R_KCAL_MOL_K = 0.00198720425864083
TEMPERATURE_K = 300.0
OUTPUT_DIR = ROOT / "reports" / "tetrapyrrole_torsion_pmf"
LOCKED_PKA = {"APP": 5.0, "BPP": 4.5, "CPP": 4.5, "DPP": 5.0}
REPORTED_OPEN_PKA = {"BPP": 7.0, "CPP": 7.0}


@dataclass(frozen=True, slots=True)
class PMFSource:
    state: str
    config: Path
    output: Path


@dataclass(frozen=True, slots=True)
class RotationResult:
    state: str
    minimum_gap_kcal_mol: float
    basin_gap_kcal_mol: float
    bootstrap_basin_gaps_kcal_mol: np.ndarray


SOURCES = {
    "BV": PMFSource(
        "BV",
        ROOT / "configs" / "BV" / "us" / "wham-dih.yaml",
        ROOT / "systems" / "BV" / "solv_4.0" / "us-single15" / "wham-dih",
    ),
    "BPP": PMFSource(
        "BPP",
        ROOT / "configs" / "BPP" / "us" / "wham-dih.yaml",
        ROOT / "systems" / "BPP" / "solv_4.0" / "us-single15" / "wham-dih-equil",
    ),
    "APP": PMFSource(
        "APP",
        ROOT / "configs" / "APP" / "us" / "wham-dih.yaml",
        ROOT / "systems" / "APP" / "solv_4.0" / "us-single15" / "wham-dih-equil",
    ),
    "CPP": PMFSource(
        "CPP",
        ROOT / "configs" / "CPP" / "us" / "wham-dih.yaml",
        ROOT / "systems" / "CPP" / "solv_4.0" / "us-single15" / "wham-dih",
    ),
    "DPP": PMFSource(
        "DPP",
        ROOT / "configs" / "DPP" / "us" / "wham-dih-partial-w000.yaml",
        ROOT
        / "systems"
        / "DPP"
        / "solv_4.0"
        / "us-single15"
        / "wham-dih-equil-partial-w000",
    ),
}


def read_original_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = {row["metric"]: float(row["original"]) for row in rows}
    required = {"delta_pmf_anti_minus_syn", "population_delta_g_anti_minus_syn"}
    if not required.issubset(values):
        raise ValueError(f"Missing thermodynamic metrics in {path}")
    return values


def read_bootstrap_basin_gaps(source: PMFSource) -> np.ndarray:
    cfg = load_config(source.config)
    grouped: dict[int, list[tuple[float, float]]] = {}
    path = source.output / "bootstrap_pmfs.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            grouped.setdefault(int(row["bootstrap_replica"]), []).append(
                (float(row["dihedral_deg"]), float(row["pmf_kcal_mol"]))
            )
    results: list[float] = []
    for replica, rows in sorted(grouped.items()):
        array = np.asarray(rows, dtype=float)
        if np.any(np.diff(array[:, 0]) <= 0.0):
            raise ValueError(f"Bootstrap grid is not increasing for {source.state} replica {replica}")
        metrics = compute_thermodynamics(array[:, 0], array[:, 1], cfg)
        results.append(metrics["population_delta_g_anti_minus_syn"][0])
    result = np.asarray(results, dtype=float)
    if result.size < 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"Insufficient finite bootstrap PMFs for {source.state}")
    return result


def load_rotation(source: PMFSource) -> RotationResult:
    metrics = read_original_metrics(source.output / "thermodynamics.csv")
    return RotationResult(
        state=source.state,
        minimum_gap_kcal_mol=metrics["delta_pmf_anti_minus_syn"],
        basin_gap_kcal_mol=metrics["population_delta_g_anti_minus_syn"],
        bootstrap_basin_gaps_kcal_mol=read_bootstrap_basin_gaps(source),
    )


def shift_distribution(
    protonated: RotationResult,
    deprotonated: RotationResult,
    temperature_k: float,
) -> tuple[float, float, float, float]:
    factor = R_KCAL_MOL_K * temperature_k * math.log(10.0)
    central = (deprotonated.basin_gap_kcal_mol - protonated.basin_gap_kcal_mol) / factor
    # The three WHAM calculations used independent bootstrap streams. The full
    # Cartesian difference is therefore the deterministic empirical convolution.
    samples = (
        deprotonated.bootstrap_basin_gaps_kcal_mol[:, None]
        - protonated.bootstrap_basin_gaps_kcal_mol[None, :]
    ).ravel() / factor
    return (
        central,
        float(np.std(samples, ddof=1)),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def draw_state(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    width, height = 2.55, 0.78
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2.0, y - height / 2.0),
            width,
            height,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=2.0,
            zorder=3,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=10.2, zorder=4)


def double_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#333333",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="<->",
            mutation_scale=12,
            linewidth=1.5,
            color=color,
            zorder=2,
        )
    )


def draw_cycle(
    ax: plt.Axes,
    protonated: RotationResult,
    deprotonated: RotationResult,
    shift: tuple[float, float, float, float],
    site: str,
    state_color: tuple[str, str],
    displayed_open_pka: float,
) -> None:
    locked_x, open_x = 2.0, 8.0
    protonated_y, deprotonated_y = 4.3, 1.65
    central, standard_deviation, lower, upper = shift
    locked_pka = LOCKED_PKA[deprotonated.state]

    draw_state(ax, locked_x, protonated_y, "BV · ZZZsss\nlocked helix", "#fff1bf", "#997515")
    draw_state(ax, open_x, protonated_y, "BV · ZZZssa\nopen helix", "#fff1bf", "#997515")
    draw_state(
        ax,
        locked_x,
        deprotonated_y,
        f"{deprotonated.state} · ZZZsss\nlocked helix {site}",
        state_color[0],
        state_color[1],
    )
    draw_state(
        ax,
        open_x,
        deprotonated_y,
        f"{deprotonated.state} · ZZZssa\nopen helix {site}",
        state_color[0],
        state_color[1],
    )

    double_arrow(ax, (3.35, protonated_y), (6.65, protonated_y), "#997515")
    double_arrow(ax, (3.35, deprotonated_y), (6.65, deprotonated_y), state_color[1])
    for x in (locked_x, open_x):
        double_arrow(ax, (x, protonated_y - 0.45), (x, deprotonated_y + 0.45))

    ax.text(
        5.0,
        protonated_y + 0.43,
        rf"$\Delta G_{{\mathrm{{rot}}}}^{{\mathrm{{BV}}}}$ = {protonated.basin_gap_kcal_mol:.3f} kcal mol$^{{-1}}$",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    ax.text(
        5.0,
        deprotonated_y - 0.48,
        rf"$\Delta G_{{\mathrm{{rot}}}}^{{\mathrm{{{deprotonated.state}}}}}$ = {deprotonated.basin_gap_kcal_mol:.3f} kcal mol$^{{-1}}$",
        ha="center",
        va="top",
        fontsize=10,
    )
    ax.text(
        locked_x - 0.16,
        2.98,
        f"deprotonate N{site}\n"
        + rf"p$K_{{a,{site}}}^{{\mathrm{{locked}}}}$ = {locked_pka:.2f}",
        ha="right",
        va="center",
        fontsize=9.5,
    )
    ax.text(
        open_x + 0.16,
        2.98,
        f"deprotonate N{site}\n"
        + rf"p$K_{{a,{site}}}^{{\mathrm{{open}}}}$ $\sim$ {displayed_open_pka:g}",
        ha="left",
        va="center",
        fontsize=9.5,
    )
    ax.text(
        5.0,
        0.33,
        rf"p$K_{{a,{site}}}^{{\mathrm{{open}}}}$ − p$K_{{a,{site}}}^{{\mathrm{{locked}}}}$ "
        rf"= $(\Delta G_{{\mathrm{{rot}}}}^{{\mathrm{{{deprotonated.state}}}}} - "
        rf"\Delta G_{{\mathrm{{rot}}}}^{{\mathrm{{BV}}}})/(RT\ln 10)$"
        "\n"
        rf"= {central:+.2f} pK units (95% bootstrap CI {lower:+.2f} to {upper:+.2f}; SD {standard_deviation:.2f})",
        ha="center",
        va="center",
        fontsize=9.4,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(-0.15, 5.15)
    ax.axis("off")


def write_table(
    path: Path,
    results: dict[str, RotationResult],
    shifts: dict[str, tuple[float, float, float, float]],
    state_sites: tuple[tuple[str, str], ...],
) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "cycle",
                "deprotonated_site",
                "temperature_K",
                "bv_rotation_minimum_gap_kcal_mol",
                "deprotonated_rotation_minimum_gap_kcal_mol",
                "bv_rotation_basin_delta_g_kcal_mol",
                "deprotonated_rotation_basin_delta_g_kcal_mol",
                "pka_open_minus_locked",
                "locked_pka",
                "inferred_open_pka",
                "bootstrap_std_pka",
                "bootstrap_95ci_lower_pka",
                "bootstrap_95ci_upper_pka",
                "inferred_open_pka_95ci_lower",
                "inferred_open_pka_95ci_upper",
                "protonated_pmf",
                "deprotonated_pmf",
            ]
        )
        for state, site in state_sites:
            shift = shifts[state]
            writer.writerow(
                [
                    f"BV/{state}",
                    site,
                    f"{TEMPERATURE_K:.1f}",
                    f"{results['BV'].minimum_gap_kcal_mol:.8f}",
                    f"{results[state].minimum_gap_kcal_mol:.8f}",
                    f"{results['BV'].basin_gap_kcal_mol:.8f}",
                    f"{results[state].basin_gap_kcal_mol:.8f}",
                    f"{shift[0]:.8f}",
                    f"{LOCKED_PKA[state]:.8f}",
                    f"{LOCKED_PKA[state] + shift[0]:.8f}",
                    f"{shift[1]:.8f}",
                    f"{shift[2]:.8f}",
                    f"{shift[3]:.8f}",
                    f"{LOCKED_PKA[state] + shift[2]:.8f}",
                    f"{LOCKED_PKA[state] + shift[3]:.8f}",
                    str(SOURCES["BV"].output / "pmf.csv"),
                    str(SOURCES[state].output / "pmf.csv"),
                ]
            )


def write_pka_summary(
    path: Path,
    shifts: dict[str, tuple[float, float, float, float]],
    state_sites: tuple[tuple[str, str], ...],
    displayed_open_pkas: dict[str, float],
) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "microstate",
                "site",
                "locked_pka",
                "open_minus_locked_pka",
                "inferred_open_pka",
                "inferred_open_pka_95ci_lower",
                "inferred_open_pka_95ci_upper",
                "reported_open_pka_approx",
                "status",
            ]
        )
        for state, site in state_sites:
            central, _, lower, upper = shifts[state]
            writer.writerow(
                [
                    state,
                    site,
                    f"{LOCKED_PKA[state]:.8f}",
                    f"{central:.8f}",
                    f"{LOCKED_PKA[state] + central:.8f}",
                    f"{LOCKED_PKA[state] + lower:.8f}",
                    f"{LOCKED_PKA[state] + upper:.8f}",
                    f"{displayed_open_pkas[state]:.8f}",
                    "direct BV/deprotonated torsional cycle",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-set",
        choices=("bpp-cpp", "app-dpp"),
        default="bpp-cpp",
        help="deprotonated-state pair to render (default: bpp-cpp)",
    )
    args = parser.parse_args()
    if args.pair_set == "bpp-cpp":
        state_sites = (("BPP", "B"), ("CPP", "C"))
        colors = (("#dcebf7", "#2166ac"), ("#e1f2df", "#2b8c3e"))
        suffix = ""
        anchor_text = "BPP 4.5 · CPP 4.5"
    else:
        state_sites = (("APP", "A"), ("DPP", "D"))
        colors = (("#f8d9d7", "#c62828"), ("#e3e3e3", "#202124"))
        suffix = "_app_dpp"
        anchor_text = "APP 5.0 · DPP 5.0; DPP w000 snapshot = 17.5 ps"

    selected_sources = {"BV": SOURCES["BV"]} | {
        state: SOURCES[state] for state, _ in state_sites
    }
    temperatures = {
        load_config(source.config).wham.temperature_k
        for source in selected_sources.values()
    }
    if temperatures != {TEMPERATURE_K}:
        raise ValueError(f"Expected all PMFs at {TEMPERATURE_K:g} K; found {temperatures}")
    results = {
        state: load_rotation(source) for state, source in selected_sources.items()
    }
    shifts = {
        state: shift_distribution(results["BV"], results[state], TEMPERATURE_K)
        for state, _ in state_sites
    }
    displayed_open_pkas = {
        state: (
            REPORTED_OPEN_PKA[state]
            if state in REPORTED_OPEN_PKA
            else round(LOCKED_PKA[state] + shifts[state][0], 2)
        )
        for state, _ in state_sites
    }
    style = ROOT / "plotting" / "lefteris.mplstyle"
    if style.is_file():
        plt.style.use(style)
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 9.2), constrained_layout=True)
    for axis, (state, site), color in zip(axes, state_sites, colors, strict=True):
        draw_cycle(
            axis,
            results["BV"],
            results[state],
            shifts[state],
            site,
            color,
            displayed_open_pkas[state],
        )
        axis.set_title(f"{site}-site deprotonation cycle", fontsize=14)
    fig.suptitle(
        "ZZZsss → ZZZssa torsion coupled to BV deprotonation\n"
        f"locked-helix references: {anchor_text}",
        fontsize=15,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure = OUTPUT_DIR / f"torsion_deprotonation_cycles{suffix}.png"
    table = OUTPUT_DIR / f"torsion_deprotonation_cycles{suffix}.csv"
    pka_table = OUTPUT_DIR / f"locked_open_pkas{suffix}.csv"
    fig.savefig(figure, dpi=300)
    plt.close(fig)
    write_table(table, results, shifts, state_sites)
    write_pka_summary(pka_table, shifts, state_sites, displayed_open_pkas)
    print(f"figure: {figure}")
    print(f"data: {table}")
    print(f"pKas: {pka_table}")
    for state, _ in state_sites:
        central, _, lower, upper = shifts[state]
        print(
            f"{state}: pKa(open) = {LOCKED_PKA[state] + central:.4f} "
            f"({LOCKED_PKA[state] + lower:.4f}, {LOCKED_PKA[state] + upper:.4f})"
        )


if __name__ == "__main__":
    main()
