#!/usr/bin/env python3
"""Compare BV meta-hib runs whose proton terminates at lactam OA or OD."""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from acid_base_BV import (
    iter_xyz_frames,
    load_fes_snapshot,
    read_box_lengths_from_dftb_inp,
    read_xyz_symbols,
    shortest_hbond_path,
)
from correct_summary import ROOT, load_run, save_csv, smooth_helicity


HIB_GROUPS = {
    "OA": (10, 50, 156, 177),
    "OD": (63, 73, 74, 76, 83, 88, 123, 155, 166, 174, 20),
}
HIC_GROUPS = {
    "OA": (27, 80, 95, 126, 163, 185, 189, 198),
    "OD": (16, 25, 29, 38, 61, 133, 137, 147, 174),
}
GROUPS = dict(HIB_GROUPS)
RUN_BENCH_TAGS = {20: "N1T64C1"}
DEFAULT_BENCH_TAG = "N1T48C1"
ENDPOINT_IDS = {"OA": 11, "OD": 53}
DONOR_ID = 19
DONOR_LABEL = "NB"
CV_DIR = "meta-hib"
SOLUTE_ATOMS = 78


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_array(path: Path, values: np.ndarray, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, delimiter=",", header=header, comments="")


def geometry_series(
    record: dict,
    endpoint_id: int,
    donor_id: int,
    donor_label: str,
    output: Path,
    max_bridging_waters: int,
    covalent_cutoff: float,
    hydrogen_acceptor_cutoff: float,
    angle_cutoff: float,
    refresh: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed OA--OD and donor--lactam H-bond paths on report times."""
    label = record["label"]
    opposite_path = output / f"{label}_OA_OD_hbonds.csv"
    wire_path = output / f"{label}_{donor_label}_lactam_hbonds.csv"
    if not refresh and opposite_path.exists() and wire_path.exists():
        opposite = np.atleast_1d(
            np.genfromtxt(opposite_path, delimiter=",", names=True)
        )
        wire = np.atleast_1d(np.genfromtxt(wire_path, delimiter=",", names=True))
        return (
            np.column_stack(
                [opposite["aligned_time_ps"], opposite["number_of_hydrogen_bonds"]]
            ),
            np.column_stack([wire["aligned_time_ps"], wire["number_of_hydrogen_bonds"]]),
        )

    report_rows = read_rows(Path(record["report_folder"]) / "summary.csv")
    stop = record["raw_marker"] + record["window_ps"]
    selected = [row for row in report_rows if float(row["time_ps"]) <= stop + 1.0e-8]
    report_times = {round(float(row["time_ps"]), 8) for row in selected}
    source = Path(record["source"])
    symbols = read_xyz_symbols(source / "traject")
    if len(symbols) <= SOLUTE_ATOMS or symbols[donor_id - 1].upper() != "N":
        raise ValueError(f"Unexpected BV atom ordering in {source}")
    for atom_id in ENDPOINT_IDS.values():
        if symbols[atom_id - 1].upper() != "O":
            raise ValueError(f"Expected lactam oxygen at atom {atom_id} in {source}")
    solvent_oxygen_ids = [
        atom_id
        for atom_id, symbol in enumerate(symbols, start=1)
        if atom_id > SOLUTE_ATOMS and symbol.upper() == "O"
    ]
    hydrogen_ids = [
        atom_id
        for atom_id, symbol in enumerate(symbols, start=1)
        if symbol.upper() == "H"
    ]
    box = read_box_lengths_from_dftb_inp(source / "dftb.inp")
    opposite_wires: list[list[float]] = []
    wires: list[list[float]] = []
    matched: set[float] = set()
    for time_ps, coordinates in iter_xyz_frames(source / "traject"):
        if time_ps is None:
            raise ValueError(f"Trajectory frame without time in {source}")
        if time_ps > stop + 1.0e-8:
            break
        key = round(time_ps, 8)
        if key not in report_times:
            continue
        matched.add(key)
        aligned_time = time_ps + record["display_offset"]
        opposite_result, _opposite_edges = shortest_hbond_path(
            coordinates,
            ENDPOINT_IDS["OA"],
            ENDPOINT_IDS["OD"],
            solvent_oxygen_ids,
            hydrogen_ids,
            covalent_cutoff=covalent_cutoff,
            hydrogen_acceptor_cutoff=hydrogen_acceptor_cutoff,
            angle_cutoff=angle_cutoff,
            max_bridging_waters=max_bridging_waters,
            box=box,
        )
        opposite_bonds = (
            float(opposite_result.bridging_water_count + 1)
            if opposite_result.connected else 0.0
        )
        opposite_wires.append(
            [
                time_ps,
                aligned_time,
                opposite_bonds,
                float(opposite_result.bridging_water_count),
                float(opposite_result.connected),
            ]
        )
        result, _edges = shortest_hbond_path(
            coordinates,
            donor_id,
            endpoint_id,
            solvent_oxygen_ids,
            hydrogen_ids,
            covalent_cutoff=covalent_cutoff,
            hydrogen_acceptor_cutoff=hydrogen_acceptor_cutoff,
            angle_cutoff=angle_cutoff,
            max_bridging_waters=max_bridging_waters,
            box=box,
        )
        number_of_bonds = (
            float(result.bridging_water_count + 1) if result.connected else 0.0
        )
        wires.append(
            [
                time_ps,
                aligned_time,
                number_of_bonds,
                float(result.bridging_water_count),
                float(result.connected),
            ]
        )
    missing = sorted(report_times - matched)
    if missing:
        raise ValueError(f"{label}: {len(missing)} report timestamps lack trajectory frames")
    opposite_values = np.asarray(opposite_wires, dtype=float)
    wire_values = np.asarray(wires, dtype=float)
    if not len(opposite_values) or opposite_values[-1, 0] < stop - 0.03:
        raise ValueError(f"{label}: geometry does not cover the analysis endpoint")
    save_array(
        opposite_path,
        opposite_values,
        "raw_time_ps,aligned_time_ps,number_of_hydrogen_bonds,"
        "bridging_water_count,connected",
    )
    save_array(
        wire_path,
        wire_values,
        "raw_time_ps,aligned_time_ps,number_of_hydrogen_bonds,"
        "bridging_water_count,connected",
    )
    return opposite_values[:, [1, 2]], wire_values[:, [1, 2]]


def calculate_record_geometry(
    record: dict,
    output: Path,
    max_bridging_waters: int,
    covalent_cutoff: float,
    hydrogen_acceptor_cutoff: float,
    angle_cutoff: float,
    refresh: bool,
    donor_id: int,
    donor_label: str,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Process one run independently for bounded parallel execution."""
    opposite_wire, wire = geometry_series(
        record, ENDPOINT_IDS[record["group"]], donor_id, donor_label, output,
        max_bridging_waters, covalent_cutoff, hydrogen_acceptor_cutoff,
        angle_cutoff, refresh,
    )
    return record["label"], opposite_wire, wire


def load_records(base: Path, window_ps: float, samples: int) -> list[dict]:
    entries = {
        (row["bench_tag"], int(row["run"])): row
        for row in read_rows(base / "index.csv")
    }
    records: list[dict] = []
    for group, runs in GROUPS.items():
        for run in runs:
            bench_tag = RUN_BENCH_TAGS.get(run, DEFAULT_BENCH_TAG)
            key = (bench_tag, run)
            if key not in entries:
                raise ValueError(
                    f"Run {run} ({bench_tag}) is absent from {base / 'index.csv'}"
                )
            record = load_run(base, entries[key], samples, window_ps)
            record.update(
                group=group, label=f"{group}-{run}", window_ps=window_ps,
                bench_tag=bench_tag,
            )
            # This summary intentionally shows the FES at tdiff, not at tstop.
            snapshot, fes = load_fes_snapshot(
                Path(record["source"]) / "fes.dat", record["raw_marker"]
            )
            record["snapshot_time"] = snapshot + record["display_offset"]
            record["fes"] = fes
            records.append(record)
    return records


def make_main_axes(fig: plt.Figure) -> tuple[np.ndarray, object, object]:
    outer = fig.add_gridspec(
        2, 3, left=0.055, right=0.985, bottom=0.065, top=0.94,
        wspace=0.24, hspace=0.22,
    )
    axes = np.empty((2, 4), dtype=object)
    for panel in range(4):
        cell = outer[panel // 3, panel % 3]
        tiers = cell.subgridspec(2, 1, hspace=0.0)
        axes[0, panel] = fig.add_subplot(tiers[0])
        axes[1, panel] = fig.add_subplot(
            tiers[1],
            sharex=axes[0, panel] if panel != 2 else None,
            sharey=axes[0, panel],
        )
        axes[0, panel].tick_params(labelbottom=False)
    return axes, outer[1, 1], outer[1, 2]


def plot_wire_stack(
    fig: plt.Figure,
    cell: object,
    records: Sequence[dict],
    series_key: str,
    ylabel: str,
    colors: Sequence,
    maximum_time: float,
    max_bridging_waters: int,
) -> list[plt.Axes]:
    """Plot one hydrogen-bond connectivity strip per run."""
    grid = cell.subgridspec(len(records), 1, hspace=0.02)
    axes: list[plt.Axes] = []
    for index, record in enumerate(records):
        axis = fig.add_subplot(
            grid[index],
            sharex=axes[0] if axes else None,
            sharey=axes[0] if axes else None,
        )
        axes.append(axis)
        within_group = list(GROUPS[record["group"]]).index(record["run"])
        color = colors[within_group]
        series = record[series_key]
        axis.scatter(
            series[:, 0], series[:, 1], color=color,
            s=18, linewidths=0, label=record["label"],
        )
        axis.axvline(record["marker"], color=color, linestyle="--", linewidth=0.9)
        axis.set_xlim(40.0, maximum_time)
        axis.set_ylim(-0.5, max_bridging_waters + 1.5)
        axis.set_yticks([0, 3, 6, 9])
        axis.tick_params(labelbottom=index == len(records) - 1, labelsize=13)
        axis.grid(axis="y", alpha=0.18)
        axis.legend(loc="upper right", fontsize=10)
    axes[len(GROUPS["OA"])].spines["top"].set_linewidth(2.5)
    first_position, last_position = axes[0].get_position(), axes[-1].get_position()
    fig.text(
        first_position.x0 - 0.034,
        (first_position.y1 + last_position.y0) / 2,
        ylabel, rotation=90, ha="center", va="center", fontsize=24,
    )
    axes[-1].set_xlabel("t (ps; metadynamics starts at 40)")
    return axes


def plot_summary(
    records: list[dict],
    output: Path,
    smooth_ps: float,
    samples: int,
    max_bridging_waters: int,
) -> None:
    plt.style.use(ROOT / "plotting/lefteris.mplstyle")
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 23,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 12,
        }
    )
    fig = plt.figure(figsize=(30, 25))
    axes, opposite_cell, wire_cell = make_main_axes(fig)
    colors = [plt.get_cmap("tab10")(index) for index in range(10)] + ["black"]
    maximum_time = max(record["marker"] + record["window_ps"] for record in records)
    statistics: list[list[float]] = []
    for tier, group in enumerate(("OA", "OD")):
        group_records = [record for record in records if record["group"] == group]
        fes_axis, delta_axis, bar_axis, helix_axis = axes[tier]
        for index, record in enumerate(group_records):
            color = colors[index]
            run = record["run"]
            fes_axis.plot(
                record["fes"][:, 0], record["fes"][:, 1], color=color,
                linewidth=2.4, alpha=0.75,
                label=f"Run {run}: t={record['snapshot_time']:.2f} ps",
            )
            delta_axis.plot(
                record["time"], record["df"], color=color, linewidth=2.1,
                label=f"Run {run}: tdiff={record['marker']:.3f} ps",
            )
            delta_axis.axvline(record["marker"], color=color, linestyle="--", linewidth=1.2)
            mean = float(np.mean(record["sample_df"]))
            std = float(np.std(record["sample_df"]))
            pka = mean / record["factor"]
            bar_axis.bar(index, mean, yerr=std, color=color, alpha=0.82, capsize=5)
            bar_axis.annotate(
                f"{mean:.2f} ± {std:.2f}", (index, mean + std),
                xytext=(0, 5), textcoords="offset points", ha="center", fontsize=12,
            )
            smoothed = smooth_helicity(
                record["helix_time"], record["helicity"], smooth_ps
            )
            helix_axis.plot(
                record["helix_time"], smoothed, color=color, linewidth=2.0,
                label=f"Run {run}",
            )
            helix_axis.axvline(record["marker"], color=color, linestyle="--", linewidth=1.0)
            statistics.append(
                [
                    0.0 if group == "OA" else 1.0,
                    float(run), record["raw_marker"], record["marker"],
                    record["marker"] + record["window_ps"],
                    record["snapshot_time"], mean, std, pka,
                ]
            )
        fes_axis.set_xlim(0.0, 1.0)
        delta_axis.set_xlim(40.0, maximum_time)
        helix_axis.set_xlim(40.0, maximum_time)
        bar_axis.set_xticks(range(len(group_records)), [str(r["run"]) for r in group_records])
        bar_axis.tick_params(labelbottom=True)
        means = [float(np.mean(record["sample_df"])) for record in group_records]
        pkas = [mean / record["factor"] for mean, record in zip(means, group_records)]
        bar_axis.text(
            0.98, 0.94,
            f"{group}: {len(group_records)} runs\n"
            f"arithmetic mean ΔF = {np.mean(means):.2f} kcal/mol\n"
            f"arithmetic mean pKa = {np.mean(pkas):.2f}",
            transform=bar_axis.transAxes, ha="right", va="top", fontsize=15,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        for axis in axes[tier]:
            axis.grid(axis="y", alpha=0.18)
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(
                    handles, labels, loc="best", fontsize=11,
                    ncol=2 if len(group_records) > 5 else 1,
                )
        fes_axis.text(
            0.03, 0.91, f"{DONOR_LABEL} → {group}", transform=fes_axis.transAxes,
            fontsize=21, fontweight="bold",
        )

    plot_wire_stack(
        fig, opposite_cell, records, "opposite_wire",
        "OA–OD number of hydrogen bonds", colors,
        maximum_time, max_bridging_waters,
    )
    plot_wire_stack(
        fig, wire_cell, records, "wire",
        f"{DONOR_LABEL}–lactam number of hydrogen bonds", colors,
        maximum_time, max_bridging_waters,
    )

    panel_labels = [
        r"$F(s)$ (kcal mol$^{-1}$)",
        r"$\Delta F$ (kcal mol$^{-1}$)",
        r"$\langle\Delta F\rangle$ (kcal mol$^{-1}$)",
        "Ring-center helicity (deg)",
    ]
    xlabels = [
        "Coordination, s", "t (ps; metadynamics starts at 40)", "Run ID",
        "t (ps; metadynamics starts at 40)",
    ]
    for panel, (ylabel, xlabel) in enumerate(zip(panel_labels, xlabels)):
        upper, lower = axes[:, panel]
        top_position, bottom_position = upper.get_position(), lower.get_position()
        upper.set_ylabel("")
        lower.set_ylabel("")
        lower.set_xlabel(xlabel)
        fig.text(
            top_position.x0 - 0.034,
            (top_position.y1 + bottom_position.y0) / 2,
            ylabel, rotation=90, ha="center", va="center", fontsize=24,
        )
    fig.suptitle(
        f"BV · {CV_DIR} · proton accepted by lactam OA (upper) or OD (lower)",
        fontsize=31, y=0.975,
    )
    figure_path = output.with_suffix(".png")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    save_array(
        output / "statistics.csv", np.asarray(statistics),
        "tier_0_OA_1_OD,run,tdiff_raw_ps,tdiff_aligned_ps,tstop_aligned_ps,"
        "fes_snapshot_aligned_ps,mean_delta_F_kcal_mol,std_delta_F_kcal_mol,mean_pka",
    )


def write_report(output: Path, records: Sequence[dict], args: argparse.Namespace) -> None:
    mapping = [
        (
            record["group"], record["run"], record["raw_start"],
            record["raw_marker"], record["marker"],
            record["marker"] + record["window_ps"], record["snapshot_time"],
            record["source"],
        )
        for record in records
    ]
    save_csv(
        output / "run_mapping.csv",
        [
            "group", "run", "raw_start_ps", "tdiff_raw_ps", "tdiff_aligned_ps",
            "tstop_aligned_ps", "fes_snapshot_aligned_ps", "source",
        ],
        mapping,
    )
    provenance = {
        "settings": vars(args),
        "groups": GROUPS,
        "atom_ids_one_based": {DONOR_LABEL: DONOR_ID, **ENDPOINT_IDS},
        "runs": [
            {
                key: value
                for key, value in record.items()
                if not isinstance(value, np.ndarray) and key != "report_folder"
            }
            for record in records
        ],
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, default=str, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        f"# BV {CV_DIR}: lactam OA versus OD\n\n"
        f"Upper tier: {DONOR_LABEL}-to-OA runs "
        f"{', '.join(map(str, GROUPS['OA']))}. Lower tier: {DONOR_LABEL}-to-OD runs "
        f"{', '.join(map(str, GROUPS['OD']))}. "
        + ("Run 20 is from N1T64C1; all other runs are from N1T48C1. "
           if CV_DIR == "meta-hib" else "All runs are from N1T48C1. ") +
        "Run membership "
        "is user supplied; the script does not reclassify proton destinations.\n\n"
        "Each run uses its existing acid-base index/provenance tdiff and a 1.75 ps "
        "post-tdiff window with ten equally spaced Delta-F samples. Raw trajectory "
        "starts are aligned to 40 ps for display only. F(s) is the latest complete "
        "saved surface at or before tdiff, independently zeroed at its minimum. "
        "Delta F is F(min near s=0)-F(min near s=1). Bars show within-run mean and "
        "population SD; tier annotations are equal-run arithmetic means, not "
        "Boltzmann-weighted ensembles or convergence uncertainties.\n\n"
        "Ring-center helicity uses the existing A-B-C-D center dihedral with a "
        f"{args.smooth_ps:g} ps centered circular mean. The fifth panel finds the "
        "shortest hydrogen-bond path between fixed lactam oxygens OA11 and OD53 "
        "through solvent oxygens. Its upper tier is interpreted as OA(H)-to-OD and "
        "its lower tier as OD(H)-to-OA according to the user-supplied run classes.\n\n"
        "The final panel is fixed-endpoint geometry, not the original donor-to-defect "
        f"wire. It finds the shortest hydrogen-bond path from {DONOR_LABEL}{DONOR_ID} "
        "to OA11 or OD53 "
        "through solvent oxygens. Criteria are D-H <= "
        f"{args.covalent_cutoff:g} A, H-acceptor <= {args.hydrogen_acceptor_cutoff:g} A, "
        f"D-H-A >= {args.angle_cutoff:g} degrees, up to {args.max_bridging_waters} "
        "bridging waters, with periodic minimum-image geometry. These same criteria "
        "are used for both fixed-endpoint panels. Plotted zero means no path within "
        "the cap; one is a direct hydrogen bond. Every plotted series "
        "is saved beside the figure.\n",
        encoding="utf-8",
    )


def main() -> None:
    global CV_DIR, DONOR_ID, DONOR_LABEL, GROUPS, RUN_BENCH_TAGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("hib", "hic"), default="hib")
    parser.add_argument(
        "--acid-base-dir", type=Path,
        default=None,
    )
    parser.add_argument(
        "--out", type=Path,
        default=None,
    )
    parser.add_argument("--window-ps", type=float, default=1.75)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--smooth-ps", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-bridging-waters", type=int, default=8)
    parser.add_argument("--covalent-cutoff", type=float, default=1.3)
    parser.add_argument("--hydrogen-acceptor-cutoff", type=float, default=2.5)
    parser.add_argument("--angle-cutoff", type=float, default=135.0)
    parser.add_argument(
        "--refresh-geometry", action="store_true",
        help="Recompute cached OA--OD and donor--lactam hydrogen-bond paths",
    )
    args = parser.parse_args()
    if args.variant == "hic":
        CV_DIR = "meta-hic"
        DONOR_ID = 31
        DONOR_LABEL = "NC"
        GROUPS = dict(HIC_GROUPS)
        RUN_BENCH_TAGS = {}
    if args.acid_base_dir is None:
        args.acid_base_dir = ROOT / f"reports/BV/{CV_DIR}/acid_base"
    if args.out is None:
        args.out = ROOT / f"reports/BV/joint_summary_{CV_DIR}_OA_and_OD"
    if args.window_ps <= 0.0 or args.samples < 2 or args.smooth_ps <= 0.0:
        parser.error("Window and smoothing must be positive; samples must be at least 2")
    if args.max_bridging_waters < 0 or args.workers < 1:
        parser.error("--max-bridging-waters must be nonnegative; --workers must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    records = load_records(args.acid_base_dir, args.window_ps, args.samples)
    records_by_label = {record["label"]: record for record in records}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                calculate_record_geometry, record, args.out,
                args.max_bridging_waters, args.covalent_cutoff,
                args.hydrogen_acceptor_cutoff, args.angle_cutoff,
                args.refresh_geometry, DONOR_ID, DONOR_LABEL,
            )
            for record in records
        ]
        for future in as_completed(futures):
            label, opposite_wire, wire = future.result()
            records_by_label[label]["opposite_wire"] = opposite_wire
            records_by_label[label]["wire"] = wire
            print(f"Completed geometry for {label}", flush=True)
    for record in records:
        save_array(
            args.out / f"{record['label']}_fes.csv", record["fes"][:, :2],
            "coordination_s,F_kcal_mol",
        )
        save_array(
            args.out / f"{record['label']}_delta_f.csv",
            np.column_stack([record["time"] - record["display_offset"], record["time"], record["df"]]),
            "raw_time_ps,aligned_time_ps,delta_F_kcal_mol",
        )
        save_array(
            args.out / f"{record['label']}_samples.csv",
            np.column_stack(
                [
                    record["sample_time"] - record["display_offset"],
                    record["sample_time"], record["sample_df"],
                    record["sample_df"] / record["factor"],
                ]
            ),
            "raw_time_ps,aligned_time_ps,delta_F_kcal_mol,pka",
        )
        save_array(
            args.out / f"{record['label']}_helicity.csv",
            np.column_stack(
                [
                    record["helix_time"] - record["display_offset"],
                    record["helix_time"], record["helicity"],
                    smooth_helicity(record["helix_time"], record["helicity"], args.smooth_ps),
                ]
            ),
            "raw_time_ps,aligned_time_ps,helicity_deg,smoothed_helicity_deg",
        )
    plot_summary(records, args.out, args.smooth_ps, args.samples, args.max_bridging_waters)
    write_report(args.out, records, args)
    print(args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
