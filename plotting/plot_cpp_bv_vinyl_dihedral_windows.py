#!/usr/bin/env python3
"""Compare the C2A-C3A-CAA-CBA dihedral in all CPP/BV torsional windows."""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_cpp_bv_us_helicity import (
    DIHEDRAL_NAMES,
    SeriesSpec,
    calculate_named_dihedral_with_cpptraj,
    circular_statistics,
    resolve_unique_atom_ids,
)


OUTPUT_ROOT = ROOT / "systems/CPP/solv_4.0/us-single15"
FIGURE = OUTPUT_ROOT / "comparison_cpp_bv_vinyl_dihedral_19_windows.png"
DATA = OUTPUT_ROOT / "comparison_cpp_bv_vinyl_dihedral_19_windows.csv"
PROVENANCE = OUTPUT_ROOT / "comparison_cpp_bv_vinyl_dihedral_19_windows.json"
WINDOW_INDICES = tuple(range(19))
BRANCH_CENTER_DEG = -70.0
BRANCH_LIMITS_DEG = (BRANCH_CENTER_DEG - 180.0, BRANCH_CENTER_DEG + 180.0)

BASE_SPECS = {
    "CPP": SeriesSpec(
        "CPP",
        0,
        0.0,
        0.0,
        ROOT / "systems/CPP/solv_4.0/prep/solv.parm7",
        77,
        "#2166ac",
    ),
    "BV": SeriesSpec(
        "BV",
        0,
        0.0,
        0.0,
        ROOT / "systems/BV/solv_4.0/us-single15/salt/ready.parm7",
        78,
        "#b2182b",
    ),
}


def periodic_branch(values: np.ndarray) -> np.ndarray:
    """Place angles on one common 360-degree branch for both systems."""
    return BRANCH_CENTER_DEG + (values - BRANCH_CENTER_DEG + 180.0) % 360.0 - 180.0


def calculate(spec: SeriesSpec) -> tuple[SeriesSpec, np.ndarray, np.ndarray, dict[str, object]]:
    stage = (
        ROOT
        / f"systems/{spec.system}/solv_4.0/us-single15"
        / f"window-{spec.window_index:03d}/prod-dih"
    )
    atom_ids = resolve_unique_atom_ids(spec, DIHEDRAL_NAMES)
    source_times, raw_values, source = calculate_named_dihedral_with_cpptraj(
        spec, stage, atom_ids
    )
    mean, std = circular_statistics(raw_values)
    source.update(
        system=spec.system,
        window_index=spec.window_index,
        target_deg=spec.target_deg,
        topology=str(spec.topology.resolve()),
        atom_names=list(DIHEDRAL_NAMES),
        atom_ids_one_based=list(atom_ids),
        frames=len(raw_values),
        source_first_ps=float(source_times[0]),
        source_last_ps=float(source_times[-1]),
        circular_mean_deg=mean,
        circular_std_deg=std,
    )
    return spec, source_times - source_times[0], raw_values, source


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    specs = [
        replace(
            BASE_SPECS[system],
            window_index=index,
            target_deg=10.0 * index,
            pmf_minimum_deg=10.0 * index,
        )
        for index in WINDOW_INDICES
        for system in ("CPP", "BV")
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        loaded = list(executor.map(calculate, specs))
    by_key = {(row[0].system, row[0].window_index): row for row in loaded}

    with DATA.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "system",
                "window_index",
                "target_deg",
                "local_time_ps",
                "C2A_C3A_CAA_CBA_cpptraj_deg",
                "common_branch_deg",
            ]
        )
        for spec, times, raw_values, _ in loaded:
            writer.writerows(
                (
                    spec.system,
                    spec.window_index,
                    spec.target_deg,
                    time,
                    raw,
                    branch,
                )
                for time, raw, branch in zip(
                    times, raw_values, periodic_branch(raw_values)
                )
            )

    plt.style.use(Path(__file__).with_name("lefteris.mplstyle"))
    fig, axes = plt.subplots(
        5,
        4,
        figsize=(16.5, 16.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for index, axis in zip(WINDOW_INDICES, axes.flat):
        for system, color, label in (
            ("BV", "#d62728", "BV"),
            ("CPP", "#1f77b4", "CPP"),
        ):
            _, times, raw_values, _ = by_key[(system, index)]
            axis.scatter(
                times,
                periodic_branch(raw_values),
                color=color,
                s=1.15,
                linewidths=0,
                alpha=0.48,
                rasterized=True,
                label=label,
            )
        axis.set_title(f"{10 * index}° window", fontsize=13)
        axis.set_ylim(*BRANCH_LIMITS_DEG)
        axis.grid(False)
        axis.tick_params(labelsize=10)
    axes.flat[-1].set_visible(False)
    axes.flat[0].legend(frameon=False, loc="lower right", fontsize=10)
    fig.supxlabel("Local production time (ps)", fontsize=17)
    fig.supylabel(
        "C2A–C3A–CAA–CBA dihedral (degrees; common periodic branch)",
        fontsize=17,
    )
    fig.suptitle(
        "Vinyl dihedral across CPP and BV single15 umbrella windows\n"
        "BV red · CPP blue · 0°–180°",
        fontsize=19,
    )
    fig.savefig(FIGURE, dpi=300)
    plt.close(fig)

    PROVENANCE.write_text(
        json.dumps(
            {
                "definition": (
                    "cpptraj dihedral C2A-C3A-CAA-CBA after fiximagedbonds "
                    "and autoimage"
                ),
                "atom_ids_are_one_based": True,
                "display_branch_deg": list(BRANCH_LIMITS_DEG),
                "window_centers_deg": [10.0 * index for index in WINDOW_INDICES],
                "series": [row[3] for row in loaded],
                "figure": str(FIGURE.resolve()),
                "data": str(DATA.resolve()),
            },
            indent=2,
        )
        + "\n"
    )
    print(FIGURE)
    print(DATA)


if __name__ == "__main__":
    main()
