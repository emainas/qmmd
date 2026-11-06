#!/usr/bin/env python3
"""Plot representative BV meta-hic failure modes for the filtering report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acid_base_BV import read_xyz_symbols
from prn_acid_base_analysis import bias_samples, mulliken_frames


ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ROOT / "systems/BV/solv_4.0/dftb/N1T48C1"
REPORT = ROOT / "reports/BV/meta-hic"
OUTPUT = REPORT / "failure-examples"


def save_csv(path: Path, names: list[str], columns: list[np.ndarray]) -> None:
    values = np.column_stack(columns)
    np.savetxt(path, values, delimiter=",", header=",".join(names), comments="")


def load_charges(run: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    source = SYSTEMS / f"run-{run}/meta-hic"
    symbols = read_xyz_symbols(source / "traject")
    times, charges = mulliken_frames(source / "mulliken", symbols)
    return times, charges, symbols


def plot_non_diffusing() -> None:
    run = 150
    times, coordination, _ = bias_samples(
        SYSTEMS / f"run-{run}/meta-hic/biaspot"
    )
    fig, axis = plt.subplots(figsize=(7.2, 3.5))
    axis.plot(times, coordination, color="#244C66", linewidth=2.0)
    axis.axhline(0.05, color="#D62728", linestyle="--", linewidth=1.2,
                 label=r"low-$s$ threshold")
    axis.axhline(0.20, color="#E69F00", linestyle=":", linewidth=1.4,
                 label="recovery threshold")
    axis.set(xlabel="Raw DFTB time (ps)", ylabel=r"NC31--H32 coordination, $s(t)$",
             title="Non-diffusive biased coordinate: run 150")
    axis.set_ylim(-0.03, 1.02)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "non-diffusing-run-150.png", dpi=220)
    plt.close(fig)
    save_csv(
        OUTPUT / "non-diffusing-run-150.csv",
        ["raw_time_ps", "coordination_s"], [times, coordination],
    )


def plot_annihilation() -> None:
    run = 8
    times, charges, symbols = load_charges(run)
    solvent_oxygens = np.asarray(
        [i for i, symbol in enumerate(symbols) if i >= 78 and symbol == "O"]
    )
    solvent_charges = charges[:, solvent_oxygens]
    negative_indices = np.argmin(solvent_charges, axis=1)
    negative_charge = solvent_charges[np.arange(len(times)), negative_indices]
    negative_ids = solvent_oxygens[negative_indices] + 1
    positive_indices = np.argmax(solvent_charges, axis=1)
    positive_charge = solvent_charges[np.arange(len(times)), positive_indices]
    positive_ids = solvent_oxygens[positive_indices] + 1
    nc_charge = charges[:, 30]
    keep = (times >= 22.5) & (times <= 27.5)
    nc_shift = 23.24
    negative_loss = 26.96

    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.plot(times[keep], nc_charge[keep], color="#0072B2", linewidth=1.8,
              label="NC31")
    axis.plot(times[keep], negative_charge[keep], color="#D55E00", linewidth=1.8,
              label="most-negative solvent O")
    axis.plot(times[keep], positive_charge[keep], color="#009E73", linewidth=1.8,
              label="most-positive solvent O")
    axis.axhspan(-0.625, -0.525, color="#009E73", alpha=0.10,
                 label="positive-defect window")
    axis.axvline(nc_shift, color="#0072B2", linestyle="--", linewidth=1.2,
                 label="NC charge shift")
    axis.axvline(negative_loss, color="#D55E00", linestyle="--", linewidth=1.2,
                 label="negative-defect loss")
    axis.axhline(-1.10, color="0.4", linestyle=":", linewidth=1.1,
                 label="negative-defect criterion")
    axis.set(xlabel="Raw DFTB time (ps)", ylabel="Mulliken charge (e)",
             title="Counterdefect annihilation: run 8")
    axis.set_ylim(-1.42, 0.02)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUTPUT / "defect-annihilation-run-8.png", dpi=220)
    plt.close(fig)
    save_csv(
        OUTPUT / "defect-annihilation-run-8.csv",
        ["raw_time_ps", "NC31_charge_e", "positive_defect_charge_e",
         "positive_defect_oxygen_id", "negative_defect_charge_e",
         "negative_defect_oxygen_id"],
        [times[keep], nc_charge[keep], positive_charge[keep], positive_ids[keep],
         negative_charge[keep], negative_ids[keep]],
    )


def plot_tautomer_capture() -> None:
    run = 27
    times, charges, _symbols = load_charges(run)
    keep = (times >= 24.5) & (times <= 28.5)
    capture_time = 26.78
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.plot(times[keep], charges[keep, 30], color="#0072B2", linewidth=1.7,
              label="NC31")
    axis.plot(times[keep], charges[keep, 368], color="#009E73", linewidth=1.7,
              label="relay water O369")
    axis.plot(times[keep], charges[keep, 10], color="#CC79A7", linewidth=1.9,
              label="lactam OA11")
    axis.axvline(capture_time, color="black", linestyle="--", linewidth=1.2,
                 label="sustained OA capture")
    axis.set(xlabel="Raw DFTB time (ps)", ylabel="Mulliken charge (e)",
             title="Water-mediated tautomer side-reaction: run 27")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT / "tautomer-side-reaction-run-27.png", dpi=220)
    plt.close(fig)
    save_csv(
        OUTPUT / "tautomer-side-reaction-run-27.csv",
        ["raw_time_ps", "NC31_charge_e", "O369_charge_e", "OA11_charge_e"],
        [times[keep], charges[keep, 30], charges[keep, 368], charges[keep, 10]],
    )


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], name: str) -> float:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def running_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centered running mean without zero-padding the endpoints."""
    kernel = np.ones(window, dtype=float)
    numerator = np.convolve(values, kernel, mode="same")
    denominator = np.convolve(np.ones_like(values), kernel, mode="same")
    return numerator / denominator


def plot_poor_sampling() -> None:
    run = 114
    base = ROOT / "reports/BV/meta-hic_discarded_20260907" / f"run-{run}"
    provenance = json.loads((base / "provenance.json").read_text())
    start = float(provenance["t_diffuse_restart_aligned_ps"])
    stop = float(provenance["analysis_end_restart_aligned_ps"])
    summary = {
        round(number(row, "time_ps"), 8): row
        for row in read_dict_rows(base / "summary.csv")
    }
    competitor = [
        row for row in read_dict_rows(base / "summary_bv_competitors.csv")
        if start - 1.0e-8 <= number(row, "time_ps") <= stop + 1.0e-8
    ]
    times = np.asarray([number(row, "time_ps") for row in competitor])
    defect_ids = np.asarray([number(row, "defect_oxygen_id") for row in competitor])
    bridges = np.asarray([
        number(row, "T2-O1_defect_hbond_bridging_waters") for row in competitor
    ])
    assigned = np.isfinite(defect_ids)
    nc_distance = np.asarray(
        [number(summary[round(time, 8)], "N31_Odefect_distance_A") for time in times]
    )
    closest_distance = np.asarray(
        [number(summary[round(time, 8)], "closest_solute_heavy_distance_A") for time in times]
    )

    pka_header = (base / "summary_pka_window_samples.csv").read_text().splitlines()[0]
    pka_fields = dict(item.split("=") for item in pka_header[2:].split(","))
    mean_pka = float(pka_fields["mean_pka"])
    std_pka = float(pka_fields["std_pka"])

    fig, axes = plt.subplots(
        2, 1, figsize=(7.2, 5.7), sharex=True, layout="constrained"
    )
    axes[0].scatter(times[assigned], bridges[assigned], color="0.55", s=23,
                    label="other path states")
    one_bridge = assigned & (bridges == 1.0)
    axes[0].scatter(times[one_bridge], bridges[one_bridge], color="#E69F00", s=34,
                    label="one bridging water")
    axes[0].set_yticks([-1, 0, 1, 2, 3],
                       ["disconnected", "direct", "1", "2", "3"])
    axes[0].set_ylabel("T2-O1--defect path\nbridging waters")
    axes[0].set_title(
        rf"Near-solute trapping: run 114 (apparent p$K_a$ = {mean_pka:.2f} $\pm$ {std_pka:.2f})"
    )
    axes[0].legend(frameon=False, loc="lower left", ncol=2, fontsize=8.5)
    axes[0].grid(axis="x", alpha=0.15)

    axes[1].scatter(times, nc_distance, color="#0072B2", s=22,
                    label="NC31--defect")
    axes[1].scatter(times, closest_distance, color="#D55E00", s=22,
                    label="nearest solute heavy atom--defect")
    for boundary in (3.5, 5.5, 7.5, 9.5):
        axes[1].axhline(boundary, color="0.65", linewidth=0.8, linestyle=":")
    axes[1].set(xlabel="Restart-aligned time (ps)", ylabel="Distance (Å)",
                xlim=(start, stop))
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].grid(axis="x", alpha=0.15)
    fig.savefig(OUTPUT / "poor-sampling-run-114.png", dpi=220)
    plt.close(fig)
    save_csv(
        OUTPUT / "poor-sampling-run-114.csv",
        ["restart_aligned_time_ps", "defect_oxygen_id",
         "T2_O1_defect_hbond_bridging_waters", "NC31_defect_distance_A",
         "nearest_solute_heavy_defect_distance_A"],
        [times, defect_ids, bridges, nc_distance, closest_distance],
    )


def plot_winner_validation() -> None:
    run = 99
    source = SYSTEMS / f"run-{run}/meta-hic"
    bias_times, coordination, _ = bias_samples(source / "biaspot")
    charge_times, charges, _symbols = load_charges(run)
    rows = read_dict_rows(REPORT / f"acid_base/run-{run}/summary.csv")
    summary_times = np.asarray([number(row, "time_ps") for row in rows])
    closest_distances = np.asarray([
        number(row, "closest_solute_heavy_distance_A") for row in rows
    ])
    tdiff = 30.0
    end = 31.75

    bias_keep = (bias_times >= 20.0) & (bias_times <= end)
    charge_keep = (charge_times >= 20.0) & (charge_times <= end)
    hop_keep = (charge_times >= 20.0) & (charge_times <= end)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), layout="constrained")
    ax_s, ax_hop, ax_charge, ax_distance = axes.flat

    ax_s.plot(bias_times[bias_keep], coordination[bias_keep],
              color="#244C66", linewidth=1.8)
    ax_s.axhline(0.05, color="#D62728", linestyle="--", linewidth=1.0,
                 label=r"low-$s$ threshold")
    ax_s.axhline(0.20, color="#E69F00", linestyle=":", linewidth=1.2,
                 label="recovery threshold")
    ax_s.axvline(tdiff, color="0.25", linestyle="--", linewidth=1.0)
    ax_s.set(xlabel="Raw DFTB time (ps)", ylabel=r"NC31--H32 $s(t)$",
             title="(a) Biased-coordinate check")
    ax_s.legend(frameon=False, fontsize=8)

    ax_hop.axhspan(-0.625, -0.525, color="#D9F0E7", alpha=0.75,
                   label="positive-defect window")
    ax_hop.plot(charge_times[hop_keep], charges[hop_keep, 437],
                color="#0072B2", linewidth=1.4, label="O438")
    ax_hop.plot(charge_times[hop_keep], charges[hop_keep, 314],
                color="#E69F00", linewidth=1.4, label="O315")
    ax_hop.plot(charge_times[hop_keep], charges[hop_keep, 30],
                color="#CC79A7", linewidth=1.4, label="NC31")
    ax_hop.axvline(tdiff, color="0.25", linestyle="--", linewidth=1.0)
    ax_hop.set(xlim=(20.0, end), xlabel="Raw DFTB time (ps)",
               ylabel="Mulliken charge (e)",
               title="(b) Grotthuss charge-swapping check")
    ax_hop.legend(frameon=False, fontsize=8, ncol=2)

    solute_oxygens = (
        ("OA11", 11, "#CC79A7"),
        ("OD53", 53, "#009E73"),
        ("T1-O1 (O62)", 62, "#0072B2"),
        ("T1-O2 (O63)", 63, "#E69F00"),
        ("T2-O1 (O72)", 72, "#D55E00"),
        ("T2-O2 (O73)", 73, "#6A51A3"),
    )
    smoothed_charges: dict[str, np.ndarray] = {}
    for label, atom_id, color in solute_oxygens:
        smoothed = running_mean(charges[charge_keep, atom_id - 1], window=11)
        smoothed_charges[label] = smoothed
        ax_charge.plot(charge_times[charge_keep], smoothed, color=color,
                       linewidth=1.35, label=label)
    ax_charge.axvline(tdiff, color="0.25", linestyle="--", linewidth=1.0)
    ax_charge.set(xlabel="Raw DFTB time (ps)", ylabel="Mulliken charge (e)",
                  ylim=(-0.82, -0.38),
                  title="(c) No lactam or tail-oxygen capture (0.10 ps mean)")
    ax_charge.legend(frameon=False, fontsize=7.2, ncol=2)

    layer_boundaries = np.asarray([2.0, 3.5, 5.5, 7.5, 9.5])
    layer_colors = ["#DCEEFF", "#E2F3E5", "#FFF0D6", "#E6E6E6"]
    for lower, upper, color in zip(
        layer_boundaries[:-1], layer_boundaries[1:], layer_colors
    ):
        ax_distance.axhspan(lower, upper, color=color, alpha=0.75, linewidth=0)
    closest_keep = (
        (summary_times >= 20.0) & (summary_times <= end)
        & np.isfinite(closest_distances)
    )
    ax_distance.scatter(summary_times[closest_keep], closest_distances[closest_keep],
                        color="#7B3294", s=20, zorder=3)
    ax_distance.axvline(tdiff, color="0.25", linestyle="--", linewidth=1.0)
    ax_distance.axvline(end, color="#D62728", linewidth=1.0)
    ax_distance.set(ylim=(0.0, 9.5),
                    xlabel="Raw DFTB time (ps)",
                    ylabel="solute--defect distance (Å)",
                    title=r"(d) Outward $\rightarrow$ inward $\rightarrow$ outward sampling")

    for axis in axes.flat:
        axis.set_xlim(20.0, 32.0)
        axis.grid(alpha=0.16)
    fig.suptitle("Mechanistic validation of selected BV meta-hic run 99",
                 fontsize=16)
    figure_path = REPORT / "winner-run-99-validation.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    data_path = REPORT / "winner-run-99-validation.csv"
    with data_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "series", "raw_time_ps", "value"])
        for time, value in zip(bias_times[bias_keep], coordination[bias_keep]):
            writer.writerow(["biased_coordinate", "coordination_s", time, value])
        for oxygen_id, atom_index in ((438, 437), (315, 314)):
            for time, value in zip(charge_times[hop_keep], charges[hop_keep, atom_index]):
                writer.writerow(["grotthuss_hopping", f"O{oxygen_id}_charge_e", time, value])
        for time, value in zip(charge_times[hop_keep], charges[hop_keep, 30]):
            writer.writerow(["grotthuss_hopping", "NC31_charge_e", time, value])
        for label, _atom_id, _color in solute_oxygens:
            series = label.split(" ")[0].replace("-", "_") + "_charge_e_running_mean"
            for time, value in zip(charge_times[charge_keep], smoothed_charges[label]):
                writer.writerow(["solute_oxygen_charge", series, time, value])
        for time, distance in zip(summary_times[closest_keep], closest_distances[closest_keep]):
            writer.writerow([
                "defect_sampling", "nearest_solute_defect_distance_A", time, distance
            ])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.style.use(ROOT / "plotting/lefteris.mplstyle")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})
    plot_non_diffusing()
    plot_annihilation()
    plot_tautomer_capture()
    plot_poor_sampling()
    plot_winner_validation()
    print(OUTPUT)


if __name__ == "__main__":
    main()
