#!/usr/bin/env python3
"""Build the full-ensemble BV meta-hib hierarchical-filtering report figures."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from acid_base_BV import read_xyz_symbols
from prn_acid_base_analysis import bias_samples
from tautomers import parse_target_mulliken


ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ROOT / "systems/BV/solv_4.0/dftb/N1T48C1"
REPORT = ROOT / "reports/BV/meta-hib"
TAUTOMER_SCAN = REPORT / "tautomer-scan.csv"
SCREEN = REPORT / "correct-run-screen.csv"
COUNTERION_AUDIT = REPORT / "counterion-audit.csv"
WINNERS = (38, 56, 98, 171)


@dataclass(frozen=True)
class FilterStage:
    title: str
    remaining: int
    removed: int
    removed_label: str


STAGES = (
    FilterStage("Initial BV meta-hib ensemble", 200, 0, ""),
    FilterStage("Sustained diffusion in s(t)", 88, 112, "Non-diffusive / incomplete"),
    FilterStage("No counterion interference", 87, 1, "Counterion interference"),
    FilterStage("No lactam side-reaction", 42, 45, "OA/OD proton capture"),
    FilterStage("Four most likely candidates", 4, 38, "Insufficient mechanistic sampling"),
    FilterStage("Final winner: run 171", 1, 3, "Insufficient reversible sampling"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def first_sustained_start(
    times: np.ndarray, mask: np.ndarray, minimum_duration_ps: float
) -> int | None:
    """Return the first index starting a continuously true interval."""
    start: int | None = None
    for index, value in enumerate(mask):
        if not value:
            start = None
            continue
        if start is None:
            start = index
        if times[index] - times[start] >= minimum_duration_ps:
            return start
    return None


def build_counterion_audit() -> list[dict[str, object]]:
    """Find sustained loss of the hydroxide-like negative-defect signature."""
    rows = read_rows(TAUTOMER_SCAN)
    audit: list[dict[str, object]] = []
    for row in rows:
        if row["classification"] in {"no_diffusive_regime", "tautomer"}:
            continue
        run = int(row["run_id"])
        source = SYSTEMS / f"run-{run}/meta-hib"
        symbols = read_xyz_symbols(source / "traject")
        solvent_oxygen_ids = [
            atom_id
            for atom_id, symbol in enumerate(symbols, start=1)
            if atom_id > 78 and symbol.upper() == "O"
        ]
        times, charges = parse_target_mulliken(
            source / "mulliken", [19, *solvent_oxygen_ids]
        )
        solvent = np.column_stack([charges[atom_id] for atom_id in solvent_oxygen_ids])
        minimum_charge = np.min(solvent, axis=1)
        most_negative_index = np.argmin(solvent, axis=1)
        most_negative_id = np.asarray(solvent_oxygen_ids)[most_negative_index]
        maximum_charge = np.max(solvent, axis=1)
        most_positive_index = np.argmax(solvent, axis=1)
        most_positive_id = np.asarray(solvent_oxygen_ids)[most_positive_index]
        negative_present = minimum_charge <= -1.10
        positive_present = (maximum_charge >= -0.625) & (maximum_charge <= -0.525)
        deprotonated = float(row["deprotonated_start_ps"])
        diffusive_start = float(row["diffusive_start_ps"])
        analysis_end = diffusive_start + 1.75
        eligible = times >= deprotonated
        eligible_indices = np.flatnonzero(eligible)
        local_start = first_sustained_start(
            times[eligible], ~negative_present[eligible], 0.25
        )
        loss_index = (
            int(eligible_indices[local_start]) if local_start is not None else None
        )
        prior_negative = bool(
            loss_index is not None and np.any(negative_present[eligible_indices[0] : loss_index])
        )
        loss_time = float(times[loss_index]) if loss_index is not None else np.nan
        interference = (
            loss_index is not None
            and prior_negative
            and loss_time <= analysis_end + 1.0e-8
        )
        prior_positive_indices = np.flatnonzero(
            positive_present & (times >= deprotonated) & (times <= loss_time)
        ) if np.isfinite(loss_time) else np.array([], dtype=int)
        last_positive_time = (
            float(times[prior_positive_indices[-1]])
            if prior_positive_indices.size else np.nan
        )
        audit.append(
            {
                "run": run,
                "source_classification": row["classification"],
                "deprotonated_start_ps": deprotonated,
                "diffusive_start_ps": diffusive_start,
                "analysis_end_ps": analysis_end,
                "counterion_interference": int(interference),
                "negative_defect_loss_ps": loss_time,
                "last_positive_defect_ps_before_loss": last_positive_time,
                "positive_to_negative_loss_lag_ps": (
                    loss_time - last_positive_time
                    if np.isfinite(loss_time) and np.isfinite(last_positive_time)
                    else np.nan
                ),
                "loss_negative_oxygen_id": (
                    int(most_negative_id[loss_index]) if loss_index is not None else ""
                ),
                "loss_negative_charge_e": (
                    float(minimum_charge[loss_index]) if loss_index is not None else np.nan
                ),
                "last_positive_oxygen_id": (
                    int(most_positive_id[prior_positive_indices[-1]])
                    if prior_positive_indices.size else ""
                ),
                "nb19_charge_at_loss_e": (
                    float(charges[19][loss_index]) if loss_index is not None else np.nan
                ),
            }
        )
    return audit


def finalize_existing_counterion_audit() -> list[dict[str, object]]:
    """Apply the finite analysis-window criterion without rereading Mulliken data."""
    scan = {int(row["run_id"]): row for row in read_rows(TAUTOMER_SCAN)}
    finalized: list[dict[str, object]] = []
    for row in read_rows(COUNTERION_AUDIT):
        run = int(row["run"])
        diffusive_start = float(scan[run]["diffusive_start_ps"])
        analysis_end = diffusive_start + 1.75
        try:
            loss_time = float(row["negative_defect_loss_ps"])
        except ValueError:
            loss_time = np.nan
        deprotonated = float(row["deprotonated_start_ps"])
        row["diffusive_start_ps"] = diffusive_start
        row["analysis_end_ps"] = analysis_end
        row["counterion_interference"] = int(
            np.isfinite(loss_time)
            and loss_time > deprotonated + 0.02
            and loss_time <= analysis_end + 1.0e-8
        )
        ordered = {
            "run": row["run"],
            "source_classification": row["source_classification"],
            "deprotonated_start_ps": row["deprotonated_start_ps"],
            "diffusive_start_ps": row["diffusive_start_ps"],
            "analysis_end_ps": row["analysis_end_ps"],
            "counterion_interference": row["counterion_interference"],
            "negative_defect_loss_ps": row["negative_defect_loss_ps"],
            "last_positive_defect_ps_before_loss": row[
                "last_positive_defect_ps_before_loss"
            ],
            "positive_to_negative_loss_lag_ps": row[
                "positive_to_negative_loss_lag_ps"
            ],
            "loss_negative_oxygen_id": row["loss_negative_oxygen_id"],
            "loss_negative_charge_e": row["loss_negative_charge_e"],
            "last_positive_oxygen_id": row["last_positive_oxygen_id"],
            "nb19_charge_at_loss_e": row["nb19_charge_at_loss_e"],
        }
        finalized.append(ordered)
    return finalized


def write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionary rows with stable field ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_classification() -> list[dict[str, object]]:
    """Assign every run to the first applicable hierarchy stage."""
    scan = {int(row["run_id"]): row for row in read_rows(TAUTOMER_SCAN)}
    counterion = {
        int(row["run"])
        for row in read_rows(COUNTERION_AUDIT)
        if int(row["counterion_interference"])
    }
    rows: list[dict[str, object]] = []
    for run in range(1, 201):
        source = scan.get(run)
        if source is None:
            stage = 1
            classification = "incomplete production output"
            decision = "reject"
            evidence = "biaspot, trajectory, and Mulliken outputs are absent"
        elif source["classification"] == "no_diffusive_regime":
            stage = 1
            classification = "non-diffusive biased coordinate"
            decision = "reject"
            evidence = "sustained low-s interval followed by recovery was not observed"
        elif run in counterion:
            stage = 2
            classification = "counterion interference"
            decision = "reject"
            evidence = "sustained negative-defect loss within tdiff + 1.75 ps"
        elif source["classification"] == "tautomer":
            stage = 3
            site = "OA11" if int(source["acceptor_id"]) == 11 else "OD53"
            classification = f"charge-confirmed {site} proton capture"
            decision = "reject"
            evidence = (
                f"persistent geometry; dq(NB)={float(source['donor_charge_shift']):+.3f} e; "
                f"dq({site})={float(source['acceptor_charge_shift']):+.3f} e"
            )
        elif run == 171:
            stage = 6
            classification = "final winner"
            decision = "retain"
            evidence = "repeated outward-inward-outward defect-distance sampling"
        elif run in WINNERS:
            stage = 5
            classification = "most-likely candidate with insufficient reversible sampling"
            decision = "reject"
            evidence = "included in four-run summary but lacks repeated defect-distance traversal"
        else:
            stage = 4
            classification = "insufficient mechanistic or orthogonal sampling"
            decision = "reject"
            evidence = {
                "geometry_only_candidate": "persistent lactam geometry lacks charge confirmation",
                "insufficient_post_diffusion_window": "trajectory does not cover tdiff + 1.75 ps",
                "no_persistent_lactam_tautomer": "not selected by the established correct summary",
            }[source["classification"]]
        rows.append(
            {
                "run_id": run,
                "hierarchical_stage": stage,
                "classification": classification,
                "decision": decision,
                "numerical_evidence": evidence,
            }
        )
    return rows


def add_stage_box(axis, y: float, width: float, stage: FilterStage, color: str) -> None:
    """Draw one rounded hierarchy-stage box."""
    center = 5.0
    patch = FancyBboxPatch(
        (center - width / 2.0, y - 0.48), width, 0.96,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        facecolor=color, edgecolor="#173548", linewidth=2.2, zorder=3,
    )
    axis.add_patch(patch)
    text_color = "#332600" if stage.remaining == 1 else "white"
    axis.text(center, y + 0.12, stage.title, ha="center", va="center",
              color=text_color, fontsize=14, fontweight="bold", zorder=4)
    axis.text(center, y - 0.18, f"N = {stage.remaining}", ha="center", va="center",
              color=text_color, fontsize=16, fontweight="bold", zorder=4)


def render_hierarchy(path: Path) -> None:
    """Render the full-ensemble funnel ending only at run 171."""
    fig, axis = plt.subplots(figsize=(10, 13.5), layout="constrained")
    axis.set(xlim=(0, 10), ylim=(-0.8, 11.3))
    axis.axis("off")
    ys = (10.0, 8.35, 6.7, 5.05, 3.4, 1.75)
    widths = (7.2, 6.1, 5.2, 4.3, 3.6, 3.3)
    colors = ("#244C66", "#2F6684", "#3B7F9C", "#4A96AF", "#58A9BE", "#F3C64E")
    for index, (stage, y, width, color) in enumerate(zip(STAGES, ys, widths, colors)):
        add_stage_box(axis, y, width, stage, color)
        if index == 0:
            continue
        axis.add_patch(FancyArrowPatch(
            (5.0, ys[index - 1] - 0.5), (5.0, y + 0.5), arrowstyle="-|>",
            mutation_scale=21, linewidth=2.2, color="#233746", zorder=2,
        ))
        axis.text(
            7.55, 0.5 * (ys[index - 1] + y),
            f"−{stage.removed}  {stage.removed_label}\n"
            f"({100.0 * stage.removed / 200:.1f}% of 200)",
            ha="left", va="center", fontsize=11.5, fontweight="bold",
            color="#8A2E2E",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F7E3E1",
                  "edgecolor": "#C77D75", "linewidth": 1.4},
        )
    axis.text(5.0, 0.95,
              "Repeated outward → inward → outward defect-distance sampling",
              ha="center", va="center", fontsize=12.5, fontweight="bold",
              color="#332600")
    axis.set_title("Hierarchical filtering of 200 BV meta-hib trajectories",
                   fontsize=20, fontweight="bold", pad=16)
    axis.text(5.0, 10.72,
              "NB19–H20 deprotonation: from the full ensemble to run 171",
              ha="center", va="center", fontsize=13.5, color="#34495E")
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def save_columns(path: Path, names: list[str], columns: list[np.ndarray]) -> None:
    """Save aligned numerical series used by a figure."""
    np.savetxt(path, np.column_stack(columns), delimiter=",",
               header=",".join(names), comments="")


def load_charge_extrema(run: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load NB19 and dynamic positive/negative solvent-oxygen charges."""
    source = SYSTEMS / f"run-{run}/meta-hib"
    symbols = read_xyz_symbols(source / "traject")
    oxygen_ids = [atom_id for atom_id, symbol in enumerate(symbols, start=1)
                  if atom_id > 78 and symbol.upper() == "O"]
    times, charges = parse_target_mulliken(source / "mulliken", [19, *oxygen_ids])
    solvent = np.column_stack([charges[atom_id] for atom_id in oxygen_ids])
    return times, charges[19], np.max(solvent, axis=1), np.min(solvent, axis=1), np.asarray(oxygen_ids)


def plot_non_diffusive(output: Path) -> None:
    run = 4
    times, coordination, _ = bias_samples(SYSTEMS / f"run-{run}/meta-hib/biaspot")
    fig, axis = plt.subplots(figsize=(7.2, 3.5))
    axis.plot(times, coordination, color="#244C66", linewidth=2.0)
    axis.axhline(0.05, color="#D62728", linestyle="--", linewidth=1.2,
                 label=r"low-$s$ threshold")
    axis.axhline(0.20, color="#E69F00", linestyle=":", linewidth=1.4,
                 label="recovery threshold")
    axis.set(xlabel="Raw DFTB time (ps)", ylabel=r"NB19--H20 coordination, $s(t)$",
             title="Non-diffusive biased coordinate: run 4", ylim=(-0.03, 1.02))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "non-diffusing-run-4.png", dpi=220)
    plt.close(fig)
    save_columns(output / "non-diffusing-run-4.csv",
                 ["raw_time_ps", "coordination_s"], [times, coordination])


def plot_counterion(output: Path) -> None:
    run = 54
    times, nb, positive, negative, _oxygen_ids = load_charge_extrema(run)
    keep = (times >= 28.0) & (times <= 33.0)
    loss_time = 30.92
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.plot(times[keep], nb[keep], color="#0072B2", linewidth=1.8, label="NB19")
    axis.plot(times[keep], positive[keep], color="#009E73", linewidth=1.7,
              label="most-positive solvent O")
    axis.plot(times[keep], negative[keep], color="#D55E00", linewidth=1.8,
              label="most-negative solvent O")
    axis.axhspan(-0.625, -0.525, color="#009E73", alpha=0.10,
                 label="positive-defect window")
    axis.axhline(-1.10, color="0.4", linestyle=":", linewidth=1.1,
                 label="negative-defect criterion")
    axis.axvline(loss_time, color="black", linestyle="--", linewidth=1.2,
                 label="negative-defect loss")
    axis.set(xlabel="Raw DFTB time (ps)", ylabel="Mulliken charge (e)",
             title="Counterdefect loss: run 54", ylim=(-1.42, 0.02))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output / "counterion-run-54.png", dpi=220)
    plt.close(fig)
    save_columns(output / "counterion-run-54.csv",
                 ["raw_time_ps", "NB19_charge_e", "positive_defect_charge_e",
                  "negative_defect_charge_e"],
                 [times[keep], nb[keep], positive[keep], negative[keep]])


def plot_tautomer(output: Path) -> None:
    run = 10
    source = SYSTEMS / f"run-{run}/meta-hib"
    symbols = read_xyz_symbols(source / "traject")
    oxygen_ids = [atom_id for atom_id, symbol in enumerate(symbols, start=1)
                  if atom_id > 78 and symbol.upper() == "O"]
    times, charges = parse_target_mulliken(source / "mulliken", [11, 19, *oxygen_ids])
    solvent = np.column_stack([charges[atom_id] for atom_id in oxygen_ids])
    positive = np.max(solvent, axis=1)
    keep = (times >= 23.5) & (times <= 28.5)
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.plot(times[keep], charges[19][keep], color="#0072B2", linewidth=1.8,
              label="NB19")
    axis.plot(times[keep], positive[keep], color="#E69F00", linewidth=1.5,
              label="most-positive relay-water O")
    axis.plot(times[keep], charges[11][keep], color="#CC79A7", linewidth=1.9,
              label="lactam OA11")
    axis.axvline(25.76, color="black", linestyle="--", linewidth=1.2,
                 label="persistent OA capture")
    axis.set(xlabel="Raw DFTB time (ps)", ylabel="Mulliken charge (e)",
             title="Water-mediated OA proton capture: run 10")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output / "tautomer-side-reaction-run-10.png", dpi=220)
    plt.close(fig)
    save_columns(output / "tautomer-side-reaction-run-10.csv",
                 ["raw_time_ps", "NB19_charge_e", "relay_positive_charge_e",
                  "OA11_charge_e"],
                 [times[keep], charges[19][keep], positive[keep], charges[11][keep]])


def number(row: dict[str, str], name: str) -> float:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return np.nan


def plot_poor_sampling(output: Path) -> None:
    run = 133
    base = REPORT / f"acid_base/run-{run}"
    summary_rows = read_rows(base / "summary.csv")
    competitor_rows = read_rows(base / "summary_bv_competitors.csv")
    summary = {round(number(row, "time_ps"), 8): row for row in summary_rows}
    times = np.asarray([number(row, "time_ps") for row in competitor_rows])
    defect_ids = np.asarray([number(row, "defect_oxygen_id") for row in competitor_rows])
    oa_bridges = np.asarray([
        number(row, "OA_defect_hbond_bridging_waters") for row in competitor_rows
    ])
    closest = np.asarray([
        number(summary[round(time, 8)], "closest_solute_heavy_distance_A")
        for time in times
    ])
    nb_distance = np.asarray([
        number(summary[round(time, 8)], "N19_Odefect_distance_A") for time in times
    ])
    keep = (times >= 25.0) & (times <= 30.75)
    assigned = keep & np.isfinite(defect_ids)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.7), sharex=True,
                             layout="constrained")
    axes[0].scatter(times[assigned], oa_bridges[assigned], color="#CC79A7", s=26)
    axes[0].set_yticks([-1, 0, 1, 2, 3],
                       ["disconnected", "direct", "1", "2", "3"])
    axes[0].set_ylabel("OA--defect path\nbridging waters")
    axes[0].set_title(r"Competitor-associated sampling: run 133 "
                      r"(apparent p$K_a=6.50\pm0.19$)")
    axes[0].grid(axis="x", alpha=0.15)
    axes[1].scatter(times[assigned], closest[assigned], color="#D55E00", s=24,
                    label="nearest solute heavy atom--defect")
    axes[1].scatter(times[assigned], nb_distance[assigned], color="#0072B2", s=20,
                    label="NB19--defect")
    for boundary in (3.5, 5.5, 7.5, 9.5):
        axes[1].axhline(boundary, color="0.65", linewidth=0.8, linestyle=":")
    axes[1].set(xlabel="Raw DFTB time (ps)", ylabel="Distance (\u00c5)",
                xlim=(25.0, 30.75))
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].grid(axis="x", alpha=0.15)
    fig.savefig(output / "poor-sampling-run-133.png", dpi=220)
    plt.close(fig)
    save_columns(output / "poor-sampling-run-133.csv",
                 ["raw_time_ps", "defect_oxygen_id", "OA_bridging_waters",
                  "nearest_solute_defect_distance_A", "NB19_defect_distance_A"],
                 [times[keep], defect_ids[keep], oa_bridges[keep], closest[keep],
                  nb_distance[keep]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterion-audit-only", action="store_true")
    parser.add_argument("--finalize-counterion-audit", action="store_true")
    parser.add_argument("--rebuild-counterion-audit", action="store_true")
    args = parser.parse_args()
    if args.finalize_counterion_audit:
        audit = finalize_existing_counterion_audit()
        write_dict_rows(COUNTERION_AUDIT, audit)
    elif args.rebuild_counterion_audit or not COUNTERION_AUDIT.exists():
        audit = build_counterion_audit()
        write_dict_rows(COUNTERION_AUDIT, audit)
    else:
        audit = read_rows(COUNTERION_AUDIT)
    flagged = [
        int(row["run"])
        for row in audit
        if int(row["counterion_interference"])
    ]
    print(f"Counterion-interference candidates ({len(flagged)}): {flagged}")
    if args.counterion_audit_only:
        return

    plt.style.use(ROOT / "plotting/lefteris.mplstyle")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})
    classification = make_classification()
    write_dict_rows(REPORT / "run-1-200-classification.csv", classification)
    write_dict_rows(
        REPORT / "hierarchical-filtering.csv",
        [
            {
                "stage": stage.title,
                "runs_remaining": stage.remaining,
                "runs_removed_at_stage": stage.removed,
                "removed_category": stage.removed_label,
                "fraction_of_initial_remaining": stage.remaining / 200.0,
            }
            for stage in STAGES
        ],
    )
    render_hierarchy(REPORT / "hierarchical-filtering.png")
    failure_dir = REPORT / "failure-examples"
    failure_dir.mkdir(parents=True, exist_ok=True)
    plot_non_diffusive(failure_dir)
    plot_counterion(failure_dir)
    plot_tautomer(failure_dir)
    plot_poor_sampling(failure_dir)
    print(REPORT / "hierarchical-filtering.png")
    print(failure_dir)


if __name__ == "__main__":
    main()
