#!/usr/bin/env python3
"""Predict Sandercyanin--BLA binding versus pH from proton linkage.

The calculation applies Wyman's linkage relation to the current one-proton
TPS ring titration (PPP versus the combined APP/BPP/CPP/DPP ensemble).  The
binding curve is anchored to the measured Kd at one reference pH.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT / "plotting" / "lefteris.mplstyle"
DEFAULT_WATER = (
    ROOT
    / "systems/BLA/solv_10.0/cphmd/titr-tps-ti-tails-deprot-10ns/post/report"
    / "tps-titration-tautomers.csv"
)
DEFAULT_PROTEIN = Path(
    "/work/users/e/m/emainas/projects/Sandercyanin_pHREMD/monomer/"
    "pH-REMD-rings-nearby-ph12-v3/post/report/tps-titration-tautomers.csv"
)
DEFAULT_OUTPUT = ROOT / "reports/BLA/water-to-sandercyanin"

R_KCAL = 0.001987204258
LN10 = np.log(10.0)
WATER_COLOR = "#2B73B6"
PROTEIN_COLOR = "#C43C39"
BINDING_COLOR = "#745296"


@dataclass(frozen=True)
class TitrationData:
    name: str
    source: Path
    ph: np.ndarray
    protonated: np.ndarray
    sem: np.ndarray


@dataclass(frozen=True)
class HillFit:
    pka: float
    hill: float
    sse: float


def load_titration(path: Path, name: str) -> TitrationData:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No titration records in {path}")
    return TitrationData(
        name=name,
        source=path,
        ph=np.asarray([float(row["pH"]) for row in rows]),
        protonated=np.asarray([float(row["ppp_fraction"]) for row in rows]),
        sem=np.asarray([float(row["ppp_sem"]) for row in rows]),
    )


def protonated_fraction(ph: np.ndarray | float, pka: float, hill: float) -> np.ndarray:
    ph_array = np.asarray(ph, dtype=float)
    return 1.0 / (1.0 + np.power(10.0, hill * (ph_array - pka)))


def fit_hill_curve(data: TitrationData) -> HillFit:
    """Fit pKa and Hill coefficient by hierarchical equal-pH SSE search."""
    pka_low = float(data.ph.min()) - 2.0
    pka_high = float(data.ph.max()) + 2.0
    hill_low, hill_high = 0.15, 2.0
    best_pka = 0.5 * (pka_low + pka_high)
    best_hill = 1.0
    best_sse = np.inf

    for _ in range(5):
        pka_grid = np.linspace(pka_low, pka_high, 241)
        hill_grid = np.linspace(hill_low, hill_high, 241)
        predicted = 1.0 / (
            1.0
            + np.power(
                10.0,
                hill_grid[:, None, None]
                * (data.ph[None, None, :] - pka_grid[None, :, None]),
            )
        )
        sse = np.sum((predicted - data.protonated[None, None, :]) ** 2, axis=2)
        hill_index, pka_index = np.unravel_index(int(np.argmin(sse)), sse.shape)
        best_pka = float(pka_grid[pka_index])
        best_hill = float(hill_grid[hill_index])
        best_sse = float(sse[hill_index, pka_index])

        pka_step = float(pka_grid[1] - pka_grid[0])
        hill_step = float(hill_grid[1] - hill_grid[0])
        pka_low, pka_high = best_pka - 3.0 * pka_step, best_pka + 3.0 * pka_step
        hill_low = max(0.01, best_hill - 3.0 * hill_step)
        hill_high = best_hill + 3.0 * hill_step

    return HillFit(pka=best_pka, hill=best_hill, sse=best_sse)


def protonated_antiderivative(ph: np.ndarray | float, fit: HillFit) -> np.ndarray:
    """Antiderivative of 1/[1 + 10**(n*(pH-pKa))]."""
    ph_array = np.asarray(ph, dtype=float)
    scale = fit.hill * LN10
    return ph_array - np.logaddexp(0.0, scale * (ph_array - fit.pka)) / scale


def binding_curve(
    ph: np.ndarray,
    free_fit: HillFit,
    bound_fit: HillFit,
    *,
    temperature: float,
    reference_ph: float,
    reference_kd_molar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Delta-nH, Delta-Gbind, and Kd for a one-proton linkage."""
    free_fraction = protonated_fraction(ph, free_fit.pka, free_fit.hill)
    bound_fraction = protonated_fraction(ph, bound_fit.pka, bound_fit.hill)
    delta_nh = bound_fraction - free_fraction

    bound_integral = protonated_antiderivative(ph, bound_fit) - protonated_antiderivative(
        reference_ph, bound_fit
    )
    free_integral = protonated_antiderivative(ph, free_fit) - protonated_antiderivative(
        reference_ph, free_fit
    )
    reference_dg = R_KCAL * temperature * np.log(reference_kd_molar)
    delta_g = reference_dg + R_KCAL * temperature * LN10 * (
        bound_integral - free_integral
    )
    kd_molar = np.exp(delta_g / (R_KCAL * temperature))
    return delta_nh, delta_g, kd_molar


def format_kd(kd_molar: float) -> str:
    if kd_molar < 1.0e-6:
        return f"{kd_molar * 1.0e9:.1f} nM"
    if kd_molar < 1.0e-3:
        return f"{kd_molar * 1.0e6:.1f} µM"
    return f"{kd_molar * 1.0e3:.1f} mM"


def plot_titration_panel(
    axis: plt.Axes,
    water: TitrationData,
    protein: TitrationData,
    water_fit: HillFit,
    protein_fit: HillFit,
    dense_ph: np.ndarray,
) -> None:
    for data, fit, color in (
        (water, water_fit, WATER_COLOR),
        (protein, protein_fit, PROTEIN_COLOR),
    ):
        axis.errorbar(
            data.ph,
            data.protonated,
            yerr=data.sem,
            fmt="o",
            markersize=4.0,
            capsize=2.0,
            elinewidth=0.8,
            color=color,
            alpha=0.82,
        )
        axis.plot(
            dense_ph,
            protonated_fraction(dense_ph, fit.pka, fit.hill),
            color=color,
            linewidth=2.7,
            label=rf"{data.name}: p$K_a$ = {fit.pka:.2f}, $n$ = {fit.hill:.2f}",
        )
        axis.axvline(fit.pka, color=color, linestyle=(0, (3, 3)), linewidth=1.0)

    axis.axhline(0.5, color="0.65", linestyle="--", linewidth=0.9)
    axis.annotate(
        "",
        xy=(protein_fit.pka, 0.53),
        xytext=(water_fit.pka, 0.53),
        arrowprops={"arrowstyle": "<->", "color": "0.20", "lw": 1.5},
    )
    axis.text(
        0.5 * (water_fit.pka + protein_fit.pka),
        0.575,
        rf"$\Delta$p$K_a$ = {protein_fit.pka - water_fit.pka:+.2f}",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
    )
    axis.set_xlim(float(dense_ph.min()), float(dense_ph.max()))
    axis.set_ylim(-0.035, 1.045)
    axis.set_xlabel("pH")
    axis.set_ylabel("PPP fraction")
    axis.set_title("A  Binding stabilizes protonated TPS", loc="left", fontweight="bold")
    axis.legend(loc="upper right", fontsize=9.8)
    axis.grid(alpha=0.15)


def plot_binding_panel(
    axis: plt.Axes,
    ph: np.ndarray,
    delta_g: np.ndarray,
    kd_molar: np.ndarray,
    *,
    temperature: float,
    reference_ph: float,
    reference_kd_molar: float,
) -> None:
    reference_dg = R_KCAL * temperature * np.log(reference_kd_molar)
    axis.plot(ph, delta_g, color=BINDING_COLOR, linewidth=3.0)
    axis.fill_between(ph, delta_g, reference_dg, color=BINDING_COLOR, alpha=0.10)
    axis.axvline(reference_ph, color="#2E9E7E", linestyle=(0, (4, 3)), linewidth=1.2)
    axis.axhline(reference_dg, color="#2E9E7E", linestyle=(0, (4, 3)), linewidth=1.2)
    axis.scatter(
        [reference_ph],
        [reference_dg],
        s=58,
        color="#2E9E7E",
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
    )
    axis.text(
        reference_ph + 0.16,
        reference_dg - 0.16,
        rf"experiment: $K_d$ = {reference_kd_molar * 1.0e6:.1f} µM at pH {reference_ph:g}",
        color="#247D69",
        fontsize=9.6,
        ha="left",
        va="top",
    )

    for index, ha, dx in ((0, "left", 0.12), (-1, "right", -0.12)):
        axis.scatter(
            [ph[index]],
            [delta_g[index]],
            s=42,
            color=BINDING_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        axis.text(
            ph[index] + dx,
            delta_g[index] + (0.22 if index == 0 else -0.22),
            rf"$K_d$ ≈ {format_kd(float(kd_molar[index]))}",
            color=BINDING_COLOR,
            fontsize=9.6,
            fontweight="bold",
            ha=ha,
            va="bottom" if index == 0 else "top",
        )

    axis.set_xlim(float(ph.min()), float(ph.max()))
    axis.set_ylim(-12.55, -1.75)
    axis.set_xlabel("pH")
    axis.set_ylabel(r"$\Delta G^\circ_{\mathrm{bind}}$ (kcal mol$^{-1}$)")
    axis.set_title("B  Proton linkage predicts pH-dependent binding", loc="left", fontweight="bold")
    axis.grid(alpha=0.15)
    axis.text(
        0.03,
        0.05,
        "stronger binding",
        transform=axis.transAxes,
        color="0.35",
        fontsize=9.3,
        fontweight="bold",
        rotation=90,
        ha="left",
        va="bottom",
    )

    kd_ticks = np.asarray([1.0e-9, 1.0e-7, 1.0e-5, 1.0e-3, 1.0e-2])
    kd_tick_positions = R_KCAL * temperature * np.log(kd_ticks)
    visible = (kd_tick_positions >= axis.get_ylim()[0]) & (kd_tick_positions <= axis.get_ylim()[1])
    right_axis = axis.twinx()
    right_axis.set_ylim(axis.get_ylim())
    right_axis.set_yticks(kd_tick_positions[visible])
    right_axis.set_yticklabels([format_kd(float(value)) for value in kd_ticks[visible]])
    right_axis.set_ylabel(r"Equivalent $K_d$")
    right_axis.grid(False)


def write_outputs(
    output_dir: Path,
    ph: np.ndarray,
    water_fit: HillFit,
    protein_fit: HillFit,
    delta_nh: np.ndarray,
    delta_g: np.ndarray,
    kd_molar: np.ndarray,
    *,
    temperature: float,
    reference_ph: float,
    reference_kd_molar: float,
    water_source: Path,
    protein_source: Path,
) -> None:
    with (output_dir / "sandercyanin-ph-dependent-binding.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pH",
                "free_PPP_fraction_fit",
                "bound_PPP_fraction_fit",
                "delta_nH_bound_minus_free",
                "delta_G_bind_kcal_per_mol",
                "Kd_molar",
            ]
        )
        for values in zip(
            ph,
            protonated_fraction(ph, water_fit.pka, water_fit.hill),
            protonated_fraction(ph, protein_fit.pka, protein_fit.hill),
            delta_nh,
            delta_g,
            kd_molar,
            strict=True,
        ):
            writer.writerow([f"{float(value):.10g}" for value in values])

    with (output_dir / "sandercyanin-ph-dependent-binding-fit-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "value", "units_or_note"])
        writer.writerow(["free_pKa", f"{water_fit.pka:.8f}", "BLA/TPS in water"])
        writer.writerow(["free_Hill_n", f"{water_fit.hill:.8f}", "equal-pH SSE fit"])
        writer.writerow(["bound_pKa", f"{protein_fit.pka:.8f}", "BLA/TPS in Sandercyanin"])
        writer.writerow(["bound_Hill_n", f"{protein_fit.hill:.8f}", "equal-pH SSE fit"])
        writer.writerow(["delta_pKa", f"{protein_fit.pka - water_fit.pka:.8f}", "bound - free"])
        writer.writerow(["reference_pH", f"{reference_ph:.8f}", "anchor"])
        writer.writerow(["reference_Kd", f"{reference_kd_molar:.10g}", "M"])
        writer.writerow(["temperature", f"{temperature:.4f}", "K"])
        writer.writerow(["linked_proton_count", "1", "PPP to deprotonated tautomer ensemble"])
        writer.writerow(["water_source", str(water_source), "input CSV"])
        writer.writerow(["protein_source", str(protein_source), "input CSV"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--water", type=Path, default=DEFAULT_WATER)
    parser.add_argument("--protein", type=Path, default=DEFAULT_PROTEIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in K")
    parser.add_argument("--reference-ph", type=float, default=7.5)
    parser.add_argument("--reference-kd", type=float, default=6.4e-6, help="Reference Kd in M")
    args = parser.parse_args()

    water = load_titration(args.water.resolve(), "Water")
    protein = load_titration(args.protein.resolve(), "Sandercyanin")
    water_fit = fit_hill_curve(water)
    protein_fit = fit_hill_curve(protein)
    ph = np.linspace(
        min(float(water.ph.min()), float(protein.ph.min())),
        max(float(water.ph.max()), float(protein.ph.max())),
        1001,
    )
    delta_nh, delta_g, kd_molar = binding_curve(
        ph,
        water_fit,
        protein_fit,
        temperature=args.temperature,
        reference_ph=args.reference_ph,
        reference_kd_molar=args.reference_kd,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(
        output_dir,
        ph,
        water_fit,
        protein_fit,
        delta_nh,
        delta_g,
        kd_molar,
        temperature=args.temperature,
        reference_ph=args.reference_ph,
        reference_kd_molar=args.reference_kd,
        water_source=water.source,
        protein_source=protein.source,
    )

    plt.style.use(STYLE)
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 5.9), gridspec_kw={"wspace": 0.34})
    plot_titration_panel(axes[0], water, protein, water_fit, protein_fit, ph)
    plot_binding_panel(
        axes[1],
        ph,
        delta_g,
        kd_molar,
        temperature=args.temperature,
        reference_ph=args.reference_ph,
        reference_kd_molar=args.reference_kd,
    )
    figure.suptitle(
        "Proton linkage makes biliverdin binding strongly pH dependent",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.938,
        "One-proton TPS ring equilibrium · Wyman linkage anchored to the measured affinity at pH 7.5",
        ha="center",
        va="center",
        fontsize=12.5,
        color="0.25",
    )
    figure.text(
        0.5,
        0.025,
        "Ligand-only linkage model: protein residues and propionic-tail protonation are held fixed.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="0.35",
    )
    figure.subplots_adjust(top=0.82, bottom=0.17)
    destination = output_dir / "sandercyanin-ph-dependent-binding.png"
    figure.savefig(destination, dpi=300)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)

    print(
        f"Water fit: pKa={water_fit.pka:.3f}, n={water_fit.hill:.3f}; "
        f"Sandercyanin fit: pKa={protein_fit.pka:.3f}, n={protein_fit.hill:.3f}"
    )
    print(
        f"Binding prediction: Kd={format_kd(float(kd_molar[0]))} at pH {ph[0]:.1f}; "
        f"Kd={format_kd(float(kd_molar[-1]))} at pH {ph[-1]:.1f}"
    )
    print(f"OK: wrote pH-dependent binding report to {output_dir}")


if __name__ == "__main__":
    main()
