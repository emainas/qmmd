"""Summarize the four BLA TPS REFEP mutations with fixed deprotonated PRX tails."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


STATES = ("APP", "BPP", "CPP", "DPP")
STATE_COLORS = {
    "APP": "#3B6FB6",
    "BPP": "#D97932",
    "CPP": "#3A9363",
    "DPP": "#8B6BB1",
}
TITRATING_PAIRS = {
    "APP": ("NA", "HA"),
    "BPP": ("NB", "HB"),
    "CPP": ("NC", "HC"),
    "DPP": ("ND", "HD"),
}
METHODS = ("TI charge derivative", "BAR", "MBAR")
METHOD_LABELS = {
    "TI charge derivative": "TI",
    "BAR": "BAR",
    "MBAR": "MBAR",
}
def _report_dir(root: Path, state: str) -> Path:
    return (
        root
        / "systems/BLA/solv_10.0/refep/tps-tails-deprot"
        / state.lower()
        / "prod-5ns/report-igb2"
    )


def _load_summary(path: Path, state: str) -> dict[str, Any]:
    summary = yaml.safe_load(path.read_text())
    expected_direction = f"deprotonated-{state.lower()} -> protonated-ppp"
    expected_definition = f"G(protonated-ppp) - G(deprotonated-{state.lower()})"
    if summary.get("system") != "BLA":
        raise ValueError(f"Expected BLA data in {path}")
    if summary.get("direction") != expected_direction:
        raise ValueError(
            f"Unexpected direction {summary.get('direction')!r} in {path}; "
            f"expected {expected_direction!r}"
        )
    if summary.get("free_energy_definition") != expected_definition:
        raise ValueError(f"Unexpected free-energy definition in {path}")
    for method in METHODS:
        if method not in summary["free_energies_kcal_mol"]:
            raise ValueError(f"Missing {method!r} estimate in {path}")
    return summary


def collect(
    root: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    rows: list[dict[str, object]] = []
    charge_rows: list[dict[str, object]] = []
    work_rows: list[dict[str, object]] = []
    expected_duration: float | None = None
    expected_windows: int | None = None

    for state in STATES:
        report_dir = _report_dir(root, state)
        summary_path = report_dir / "summary.yaml"
        charge_path = report_dir.parent.parent / "topology/charge-interpolation.csv"
        work_path = report_dir / "ti-integrand.csv"
        summary = _load_summary(summary_path, state)
        duration = float(summary["production_time_ps"])
        windows = int(summary["windows"])
        if expected_duration is None:
            expected_duration = duration
            expected_windows = windows
        elif duration != expected_duration or windows != expected_windows:
            raise ValueError("The four REFEP reports do not share a common duration/window count")

        energies = summary["free_energies_kcal_mol"]
        health = summary["replica_exchange"]
        row: dict[str, object] = {
            "tautomer": state,
            "direction": summary["direction"],
            "definition": summary["free_energy_definition"],
            "production_time_ps_per_window": duration,
            "windows": windows,
            "frames_per_window": int(summary["frames_per_window"]),
        }
        for method in METHODS:
            slug = METHOD_LABELS[method].lower()
            row[f"{slug}_kcal_mol"] = float(energies[method]["estimate"])
            row[f"{slug}_bootstrap_sd_kcal_mol"] = float(
                energies[method]["bootstrap_sd"]
            )
        row.update(
            {
                "bar_minus_ti_kcal_mol": float(energies["BAR"]["estimate"])
                - float(energies["TI charge derivative"]["estimate"]),
                "mbar_minus_ti_kcal_mol": float(energies["MBAR"]["estimate"])
                - float(energies["TI charge derivative"]["estimate"]),
                "walkers_reaching_both_endpoints": int(
                    health["walkers_reaching_both_endpoints"]
                ),
                "walkers": int(health["walkers"]),
                "complete_round_trips": int(health["complete_round_trips"]),
                "adjacent_acceptance_min": float(health["adjacent_acceptance_min"]),
                "adjacent_acceptance_max": float(health["adjacent_acceptance_max"]),
                "source_summary": str(summary_path.relative_to(root)),
                "source_charge_interpolation": str(charge_path.relative_to(root)),
                "source_ti_integrand": str(work_path.relative_to(root)),
            }
        )
        rows.append(row)

        nitrogen, hydrogen = TITRATING_PAIRS[state]
        with charge_path.open(newline="") as handle:
            for source in csv.DictReader(handle):
                charge_rows.append(
                    {
                        "tautomer": state,
                        "window": int(source["window"]),
                        "lambda_ppp_to_tautomer": float(source["lambda"]),
                        "nitrogen_atom": nitrogen,
                        "nitrogen_charge_e": float(source[nitrogen]),
                        "hydrogen_atom": hydrogen,
                        "hydrogen_charge_e": float(source[hydrogen]),
                    }
                )
        with work_path.open(newline="") as handle:
            for source in csv.DictReader(handle):
                work_rows.append(
                    {
                        "tautomer": state,
                        "window": int(source["window"]),
                        "lambda_tautomer_to_ppp": float(source["lambda"]),
                        "mean_dU_dlambda_kcal_mol": float(
                            source["mean_dU_dlambda_kcal_mol"]
                        ),
                        "sample_sd_kcal_mol": float(source["sample_sd_kcal_mol"]),
                        "block_sem_kcal_mol": float(source["block_sem_kcal_mol"]),
                        "quadratic_fit_rms_kcal_mol": float(
                            source["quadratic_fit_rms_kcal_mol"]
                        ),
                    }
                )
    return rows, charge_rows, work_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )


def plot_summary(
    path: Path,
    rows: list[dict[str, object]],
    charge_rows: list[dict[str, object]],
    work_rows: list[dict[str, object]],
    style: Path,
) -> None:
    if style.is_file():
        plt.style.use(style)

    fig, (charge_ax, work_ax) = plt.subplots(
        1, 2, figsize=(12.4, 5.1), constrained_layout=False
    )
    fig.subplots_adjust(
        left=0.075, right=0.975, bottom=0.18, top=0.76, wspace=0.28
    )

    for state in STATES:
        selected = [row for row in charge_rows if row["tautomer"] == state]
        lambdas = np.asarray(
            [float(row["lambda_ppp_to_tautomer"]) for row in selected]
        )
        nitrogen_charge = np.asarray(
            [float(row["nitrogen_charge_e"]) for row in selected]
        )
        hydrogen_charge = np.asarray(
            [float(row["hydrogen_charge_e"]) for row in selected]
        )
        nitrogen, hydrogen = TITRATING_PAIRS[state]
        charge_ax.plot(
            lambdas,
            nitrogen_charge,
            color=STATE_COLORS[state],
            linewidth=1.8,
            marker="o",
            markersize=3.6,
            label=f"{state}: {nitrogen}",
        )
        charge_ax.plot(
            lambdas,
            hydrogen_charge,
            color=STATE_COLORS[state],
            linewidth=1.5,
            linestyle="--",
            marker="^",
            markersize=3.5,
            label=f"{state}: {hydrogen}",
        )
    charge_ax.axhline(0.0, color="0.55", linewidth=0.8)
    charge_ax.set_xlabel(r"Topology $\lambda$ (PPP $\rightarrow$ tautomer)")
    charge_ax.set_ylabel("Atomic charge (e)")
    charge_ax.set_title("Ionizable N–H charge interpolation")
    charge_ax.legend(frameon=False, ncol=2, fontsize=8, loc="best")
    _panel_label(charge_ax, "A")

    for state in STATES:
        selected = [row for row in work_rows if row["tautomer"] == state]
        lambdas = np.asarray(
            [float(row["lambda_tautomer_to_ppp"]) for row in selected]
        )
        work = np.asarray(
            [float(row["mean_dU_dlambda_kcal_mol"]) for row in selected]
        )
        sem = np.asarray([float(row["block_sem_kcal_mol"]) for row in selected])
        slope, intercept = np.polyfit(lambdas, work, deg=1)
        dense_lambda = np.linspace(0.0, 1.0, 101)
        work_ax.plot(
            dense_lambda,
            slope * dense_lambda + intercept,
            color=STATE_COLORS[state],
            linewidth=1.8,
            label=f"{state} ({float(next(row['ti_kcal_mol'] for row in rows if row['tautomer'] == state)):.2f})",
        )
        work_ax.errorbar(
            lambdas,
            work,
            yerr=sem,
            fmt="o",
            markersize=3.4,
            color=STATE_COLORS[state],
            capsize=1.6,
            linewidth=0.8,
            zorder=3,
        )
    work_ax.set_xlabel(r"Reporting $\lambda$ (tautomer $\rightarrow$ PPP)")
    work_ax.set_ylabel(r"$\langle\partial U/\partial\lambda\rangle$ (kcal mol$^{-1}$)")
    work_ax.set_title("Linear TI work profiles")
    work_ax.legend(
        frameon=False,
        fontsize=8.5,
        title=r"Tautomer ($\Delta G_{\mathrm{TI}}$, kcal mol$^{-1}$)",
        title_fontsize=8.5,
        loc="best",
    )
    _panel_label(work_ax, "B")

    fig.suptitle(
        "BLA TPS REFEP with both PRX tails fixed deprotonated",
        x=0.5,
        y=0.96,
        fontsize=17,
    )
    fig.text(
        0.5,
        0.875,
        r"5 ns window$^{-1}$ explicit-solvent H-REMD; GB2 rescoring; "
        r"$\Delta G_{\mathrm{mut}}=G(\mathrm{PPP})-G(\mathrm{tautomer})$",
        ha="center",
        va="center",
        fontsize=10.5,
        color="0.25",
    )
    fig.text(
        0.5,
        0.035,
        "Work-profile uncertainties are block SEMs (smaller than most markers). "
        "GB2 rescoring of explicit-solvent ensembles is a conditional diagnostic, not implicit-solvent sampling.",
        ha="center",
        va="bottom",
        fontsize=8.7,
        color="0.32",
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "systems/BLA/solv_10.0/refep/tps-tails-deprot/summary"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    rows, charge_rows, work_rows = collect(root)
    write_csv(output / "bla-tps-refep-summary.csv", rows)
    write_csv(output / "bla-tps-refep-charge-interpolation.csv", charge_rows)
    write_csv(output / "bla-tps-refep-ti-integrand.csv", work_rows)
    plot_summary(
        output / "bla-tps-refep-summary.png",
        rows,
        charge_rows,
        work_rows,
        root / "plotting/prl.mplstyle",
    )
    print(f"OK: wrote BLA TPS REFEP summary to {output}")


if __name__ == "__main__":
    main()
