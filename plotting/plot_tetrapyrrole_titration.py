#!/usr/bin/env python3
"""Plot the BV one-proton titration and its four conjugate-base microstates."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STATE_ORDER = ("APP", "BPP", "CPP", "DPP")
COLORS = {
    "APP": "#D62728",
    "BPP": "#1F77B4",
    "CPP": "#2CA02C",
    "DPP": "#111111",
}


def read_micro_pkas(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["microstate"].strip()] = float(row["microscopic_pka"])
    missing = set(STATE_ORDER) - set(result)
    if missing:
        raise ValueError(f"missing micro-pKa values for {sorted(missing)}")
    return {state: result[state] for state in STATE_ORDER}


def titration_fractions(
    ph: np.ndarray, pkas: dict[str, float]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return BV and conjugate-base fractions for four parallel microequilibria."""
    weights = {
        state: np.power(10.0, ph - pkas[state]) for state in STATE_ORDER
    }
    partition = 1.0 + sum(weights.values())
    bv = 1.0 / partition
    products = {state: weights[state] / partition for state in STATE_ORDER}
    return bv, products


def conditional_product_ratios(pkas: dict[str, float]) -> dict[str, float]:
    weights = {state: 10.0 ** (-pkas[state]) for state in STATE_ORDER}
    total = sum(weights.values())
    return {state: weights[state] / total for state in STATE_ORDER}


def macroscopic_pka(pkas: dict[str, float]) -> float:
    return -math.log10(sum(10.0 ** (-pkas[state]) for state in STATE_ORDER))


def write_curves(
    path: Path,
    ph: np.ndarray,
    bv: np.ndarray,
    products: dict[str, np.ndarray],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pH",
                "fraction_BV_protonated",
                "fraction_total_deprotonated",
                "fraction_APP",
                "fraction_BPP",
                "fraction_CPP",
                "fraction_DPP",
            ]
        )
        total_deprotonated = sum(products.values())
        for index, ph_value in enumerate(ph):
            writer.writerow(
                [
                    f"{ph_value:.6f}",
                    f"{bv[index]:.10f}",
                    f"{total_deprotonated[index]:.10f}",
                    *[f"{products[state][index]:.10f}" for state in STATE_ORDER],
                ]
            )


def write_summary(
    path: Path,
    pkas: dict[str, float],
    ratios: dict[str, float],
    macro_pka: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["microstate", "site", "microscopic_pka", "conditional_deprotonated_fraction"]
        )
        for state in STATE_ORDER:
            writer.writerow(
                [state, state[0], f"{pkas[state]:.8f}", f"{ratios[state]:.10f}"]
            )
        writer.writerow(["combined", "all", f"{macro_pka:.8f}", "1.0000000000"])


def plot(
    path: Path,
    ph: np.ndarray,
    bv: np.ndarray,
    products: dict[str, np.ndarray],
    pkas: dict[str, float],
    ratios: dict[str, float],
    macro_pka: float,
) -> None:
    total_deprotonated = sum(products.values())
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.18)},
    )
    ax_total, ax_micro = axes

    ax_total.plot(
        ph,
        bv,
        color="#5B3A9E",
        linewidth=3.0,
        label="BV: fully protonated",
    )
    ax_total.plot(
        ph,
        total_deprotonated,
        color="#D26A1B",
        linewidth=3.0,
        label="APP + BPP + CPP + DPP: deprotonated ensemble",
    )
    ax_total.axhline(0.5, color="0.65", linewidth=1.0, linestyle=":")
    ax_total.axvline(macro_pka, color="0.25", linewidth=1.3, linestyle="--")
    ax_total.text(
        macro_pka + 0.08,
        0.52,
        f"macroscopic pKₐ = {macro_pka:.2f}",
        ha="left",
        va="bottom",
        fontsize=9.5,
    )
    ax_total.set_ylabel("total population fraction")
    ax_total.set_ylim(-0.02, 1.02)
    ax_total.legend(
        loc="center left", bbox_to_anchor=(0.02, 0.36), frameon=False
    )
    ax_total.set_title("Overall one-proton titration")

    micro_values = [products[state] for state in STATE_ORDER]
    ax_micro.stackplot(
        ph,
        *micro_values,
        colors=[COLORS[state] for state in STATE_ORDER],
        alpha=0.13,
    )
    for state in STATE_ORDER:
        ax_micro.plot(
            ph,
            products[state],
            color=COLORS[state],
            linewidth=2.5,
            label=(
                f"{state} (N{state[0]} deprotonated): "
                f"pKₐ={pkas[state]:.2f}; {100.0 * ratios[state]:.1f}%"
            ),
        )
    ax_micro.plot(
        ph,
        total_deprotonated,
        color="0.35",
        linewidth=1.5,
        linestyle="--",
        label="total deprotonated",
    )
    ax_micro.set_xlabel("pH")
    ax_micro.set_ylabel("microstate population fraction")
    ax_micro.set_ylim(-0.02, 1.02)
    ax_micro.legend(loc="upper left", frameon=False, fontsize=8.8)
    ax_micro.set_title(
        "Deprotonated ensemble split into four microstates\n"
        "(equivalently, the four reverse protonation channels)"
    )

    fig.suptitle("BV titration network constrained by LCOD ΔpKₐ values", fontsize=16)
    fig.text(
        0.5,
        0.012,
        "Model includes the first proton loss only. Conditional APP:BPP:CPP:DPP ratios "
        "are pH-independent because the four products have the same proton count.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--micro-pkas",
        type=Path,
        default=Path("reports/tetrapyrrole_lcod_pmf/network_micro_pkas.csv"),
    )
    parser.add_argument("--ph-min", type=float, default=0.0)
    parser.add_argument("--ph-max", type=float, default=8.0)
    parser.add_argument("--points", type=int, default=801)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("reports/tetrapyrrole_lcod_pmf")
    )
    parser.add_argument("--style", type=Path, default=Path("lefteris.mplstyle"))
    args = parser.parse_args()

    if args.ph_max <= args.ph_min or args.points < 2:
        raise SystemExit("invalid pH grid")
    if args.style.is_file():
        plt.style.use(args.style)
    pkas = read_micro_pkas(args.micro_pkas)
    ph = np.linspace(args.ph_min, args.ph_max, args.points)
    bv, products = titration_fractions(ph, pkas)
    ratios = conditional_product_ratios(pkas)
    macro_pka = macroscopic_pka(pkas)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.out_dir / "titration.png"
    curves_path = args.out_dir / "titration_curves.csv"
    summary_path = args.out_dir / "titration_summary.csv"
    plot(figure_path, ph, bv, products, pkas, ratios, macro_pka)
    write_curves(curves_path, ph, bv, products)
    write_summary(summary_path, pkas, ratios, macro_pka)
    print(f"figure: {figure_path.resolve()}")
    print(f"curves: {curves_path.resolve()}")
    print(f"summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
