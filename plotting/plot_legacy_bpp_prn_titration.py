#!/usr/bin/env python3
"""Plot the archived BPP(old-charge) + two-PRN BLA titration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import numpy as np
import yaml

from qmmd.cphmd_titr_report import fit_two_site_model, two_site_probabilities


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "systems/BLA/legacy_bpp_prn_titration"
DEFAULT_STYLE = REPO_ROOT / "plotting/lefteris.mplstyle"


def load_legacy_populations(data_dir: Path) -> list[dict[str, float]]:
    """Load and validate the four legacy joint tail populations."""
    titration = np.genfromtxt(data_dir / "titration.dat", names=True)
    ph = np.atleast_1d(titration["pH"]).astype(float)
    populations = {
        "00": np.atleast_1d(np.loadtxt(data_dir / "xdd.dat")).astype(float),
        "10": np.atleast_1d(np.loadtxt(data_dir / "xpd.dat")).astype(float),
        "01": np.atleast_1d(np.loadtxt(data_dir / "xdp.dat")).astype(float),
        "11": np.atleast_1d(np.loadtxt(data_dir / "xpp.dat")).astype(float),
    }
    if any(len(values) != len(ph) for values in populations.values()):
        raise ValueError("Legacy pH and joint-population tables have different lengths")
    normalization = sum(populations.values())
    if not np.allclose(normalization, 1.0, atol=2.0e-6):
        raise ValueError(
            "Legacy joint populations do not sum to one; "
            f"maximum deviation is {np.max(np.abs(normalization - 1.0)):.3g}"
        )

    rows: list[dict[str, float]] = []
    for index, ph_value in enumerate(ph):
        p00 = float(populations["00"][index])
        p10 = float(populations["10"][index])
        p01 = float(populations["01"][index])
        p11 = float(populations["11"][index])
        rows.append(
            {
                "pH": float(ph_value),
                "p00_fraction": p00,
                "p10_fraction": p10,
                "p01_fraction": p01,
                "p11_fraction": p11,
                "P0H_fraction": p00,
                "P1H_fraction": p10 + p01,
                "P2H_fraction": p11,
                "mean_tail_protons": p10 + p01 + 2.0 * p11,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def load_syn_anti_populations(
    data_dir: Path, ph_values: np.ndarray
) -> list[dict[str, float]]:
    """Load PRN syn/anti populations from the archived cphstats tables.

    PRN states 1 and 3 are O1-syn and O2-syn; states 2 and 4 are O1-anti
    and O2-anti. Fractions and ratios are conditional on the residue being
    protonated. A ratio is NaN when no anti population was observed.
    """
    rows: list[dict[str, float]] = []
    for ph in ph_values:
        path = data_dir / "raw" / f"pH{ph:.1f}_populations.dat"
        residue_states: dict[int, np.ndarray] = {}
        for line in path.read_text().splitlines():
            match = re.match(r"Residue:\s+PRN\s+(\d+)\s+(.*)", line)
            if match is None:
                continue
            values = np.asarray(
                [float(value) for value in re.findall(r"([-+]?\d*\.\d+)\s+\([01]\)", match.group(2))]
            )
            if len(values) != 5:
                raise ValueError(f"Expected five PRN states in {path}: {line}")
            residue_states[int(match.group(1))] = values
        if set(residue_states) != {2, 3}:
            raise ValueError(f"Expected PRN residues 2 and 3 in {path}")

        row: dict[str, float] = {"pH": float(ph)}
        pooled_syn = 0.0
        pooled_anti = 0.0
        for residue_id in (2, 3):
            states = residue_states[residue_id]
            syn = float(states[1] + states[3])
            anti = float(states[2] + states[4])
            protonated = syn + anti
            row[f"prn{residue_id}_protonated_fraction"] = protonated
            row[f"prn{residue_id}_syn_fraction_of_protonated"] = (
                syn / protonated if protonated else np.nan
            )
            row[f"prn{residue_id}_anti_fraction_of_protonated"] = (
                anti / protonated if protonated else np.nan
            )
            row[f"prn{residue_id}_syn_to_anti_ratio"] = (
                syn / anti if anti else (np.inf if syn else np.nan)
            )
            pooled_syn += syn
            pooled_anti += anti

        pooled_protonated = pooled_syn + pooled_anti
        row["pooled_protonated_fraction_sum"] = pooled_protonated
        row["pooled_syn_fraction_of_protonated"] = (
            pooled_syn / pooled_protonated if pooled_protonated else np.nan
        )
        row["pooled_anti_fraction_of_protonated"] = (
            pooled_anti / pooled_protonated if pooled_protonated else np.nan
        )
        row["pooled_syn_to_anti_ratio"] = (
            pooled_syn / pooled_anti
            if pooled_anti
            else (np.inf if pooled_syn else np.nan)
        )
        rows.append(row)
    return rows


def plot_results(data_dir: Path, style: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_legacy_populations(data_dir)
    fit = fit_two_site_model(rows)
    ph = np.asarray([row["pH"] for row in rows])
    syn_anti_rows = load_syn_anti_populations(data_dir, ph)
    dense_ph = np.linspace(float(ph.min()), float(ph.max()), 800)
    p00_fit, p10_fit, p01_fit, p11_fit = two_site_probabilities(
        dense_ph, fit.pk10, fit.pk01, fit.interaction_w
    )

    write_csv(
        data_dir / "microstate-data.csv",
        rows,
        [
            "pH",
            "p00_fraction",
            "p10_fraction",
            "p01_fraction",
            "p11_fraction",
            "P0H_fraction",
            "P1H_fraction",
            "P2H_fraction",
        ],
    )
    write_csv(
        data_dir / "total-titration-data.csv",
        rows,
        ["pH", "mean_tail_protons"],
    )
    write_csv(
        data_dir / "syn-anti-data.csv",
        syn_anti_rows,
        list(syn_anti_rows[0]),
    )
    (data_dir / "two-site-fit.yaml").write_text(
        yaml.safe_dump(
            {
                "model": "Ullmann coupled two-site microstate model",
                "pk10": fit.pk10,
                "pk01": fit.pk01,
                "interaction_W": fit.interaction_w,
                "macroscopic_pKa1": fit.macro_pka1,
                "macroscopic_pKa2": fit.macro_pka2,
                "definitions": {
                    "p00": "both PRN tails deprotonated (P0H)",
                    "p10": "PRN residue 2 protonated, residue 3 deprotonated",
                    "p01": "PRN residue 2 deprotonated, residue 3 protonated",
                    "p11": "both PRN tails protonated (P2H)",
                    "P1H": "p10 + p01",
                },
            },
            sort_keys=False,
        )
    )

    plt.style.use(style)
    colors = ("#0072B2", "#D55E00", "#009E73")

    fig, ax = plt.subplots(figsize=(7.3, 5.6), constrained_layout=True)
    series = (
        ("P0H = $p_{00}$", "P0H_fraction", p00_fit, "P", colors[0]),
        ("P1H = $p_{10}+p_{01}$", "P1H_fraction", p10_fit + p01_fit, "s", colors[1]),
        ("P2H = $p_{11}$", "P2H_fraction", p11_fit, "v", colors[2]),
    )
    for label, key, fitted, marker, color in series:
        ax.scatter(
            ph,
            [row[key] for row in rows],
            marker=marker,
            s=55,
            color=color,
            label=label,
            zorder=3,
        )
        ax.plot(dense_ph, fitted, color=color, lw=2.0, alpha=0.85)
    ax.set(
        xlabel="pH",
        ylabel="Molar fraction",
        xlim=(float(ph.min()) - 0.2, float(ph.max()) + 0.2),
        ylim=(-0.03, 1.03),
        title="Legacy BLA tail microstates: fixed BPP + 2 PRN",
    )
    ax.legend(
        title=(
            rf"pK$_{{10}}$={fit.pk10:.2f}, pK$_{{01}}$={fit.pk01:.2f}, "
            rf"W={fit.interaction_w:.2f}"
        )
    )
    fig.savefig(data_dir / "microstate.png", dpi=300)
    plt.close(fig)

    mean_fit = p10_fit + p01_fit + 2.0 * p11_fit
    fig, ax = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
    ax.scatter(
        ph,
        [row["mean_tail_protons"] for row in rows],
        s=52,
        color=colors[0],
        label="Legacy CpHMD",
        zorder=3,
    )
    ax.plot(
        dense_ph,
        mean_fit,
        color=colors[1],
        lw=2.2,
        label="Coupled two-site fit",
    )
    ax.set(
        xlabel="pH",
        ylabel="Mean tail proton count",
        xlim=(float(ph.min()) - 0.2, float(ph.max()) + 0.2),
        ylim=(-0.05, 2.05),
        title="Legacy BLA total tail titration: fixed BPP + 2 PRN",
    )
    ax.legend(
        title=(
            rf"Macroscopic pK$_1$={fit.macro_pka1:.2f}; "
            rf"pK$_2$={fit.macro_pka2:.2f}"
        )
    )
    fig.savefig(data_dir / "total-titration.png", dpi=300)
    plt.close(fig)

    display_rows = [row for row in syn_anti_rows if 2.5 <= row["pH"] <= 7.0]
    display_ph = np.asarray([row["pH"] for row in display_rows])
    display_syn = np.asarray(
        [row["pooled_syn_fraction_of_protonated"] for row in display_rows]
    )
    display_anti = np.asarray(
        [row["pooled_anti_fraction_of_protonated"] for row in display_rows]
    )
    mean_syn = float(np.mean(display_syn))
    mean_anti = float(np.mean(display_anti))
    fig, fraction_ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
    fraction_ax.plot(
        display_ph,
        display_syn,
        marker="o",
        color=colors[2],
        label="Syn / protonated",
    )
    fraction_ax.plot(
        display_ph,
        display_anti,
        marker="s",
        color=colors[1],
        label="Anti / protonated",
    )
    fraction_ax.axhline(mean_syn, color=colors[2], ls=":", lw=1.4)
    fraction_ax.axhline(mean_anti, color=colors[1], ls=":", lw=1.4)
    fraction_ax.text(
        6.95,
        mean_syn + 0.018,
        rf"$\langle f_{{\mathrm{{syn}}}}\rangle = {mean_syn:.3f}$",
        color=colors[2],
        ha="right",
        va="bottom",
    )
    fraction_ax.text(
        6.95,
        mean_anti - 0.018,
        rf"$\langle f_{{\mathrm{{anti}}}}\rangle = {mean_anti:.3f}$",
        color=colors[1],
        ha="right",
        va="top",
    )
    fraction_ax.axhline(0.5, color="0.5", ls="--", lw=1.0)
    fraction_ax.set(
        xlabel="pH",
        ylabel="Conditional fraction",
        xlim=(2.3, 7.2),
        ylim=(-0.03, 1.03),
        title="Legacy BLA PRN syn/anti balance: fixed BPP + 2 PRN",
    )
    fraction_ax.set_xticks(display_ph)
    fraction_ax.legend()
    fig.savefig(data_dir / "syn-anti.png", dpi=300)
    plt.close(fig)

    print(f"OK: wrote legacy BLA plots and aligned data to {data_dir}")
    print(
        f"Fit: pK10={fit.pk10:.4f}, pK01={fit.pk01:.4f}, "
        f"W={fit.interaction_w:.4f}; "
        f"macroscopic pKa1={fit.macro_pka1:.4f}, "
        f"pKa2={fit.macro_pka2:.4f}"
    )
    print("Syn:anti ratios (pooled PRN tails):")
    for row in syn_anti_rows:
        ratio = row["pooled_syn_to_anti_ratio"]
        if np.isfinite(ratio):
            value = f"{ratio:.4f}"
        elif np.isinf(ratio):
            value = "infinite (syn observed; no anti samples)"
        else:
            value = "undefined (no protonated samples)"
        print(f"  pH {row['pH']:4.1f}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    args = parser.parse_args()
    plot_results(args.data_dir.resolve(), args.style.resolve())


if __name__ == "__main__":
    main()
