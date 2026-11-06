#!/usr/bin/env python3
"""Compare ring-center helicity in every CPP/BV torsional umbrella window."""

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

from plot_bv_equil_helicity_pmf import read_helicity
from plot_cpp_bv_us_helicity import (
    SeriesSpec,
    circular_statistics,
    read_box,
    topology_rings,
)


OUTPUT_ROOT = ROOT / "systems/CPP/solv_4.0/us-single15"
FIGURE = OUTPUT_ROOT / "comparison_cpp_bv_helicity_19_windows.png"
DATA = OUTPUT_ROOT / "comparison_cpp_bv_helicity_19_windows.csv"
PROVENANCE = OUTPUT_ROOT / "comparison_cpp_bv_helicity_19_windows.json"
WINDOW_INDICES = tuple(range(19))

BASE_SPECS = {
    "CPP": SeriesSpec(
        "CPP",
        0,
        0.0,
        0.0,
        ROOT / "systems/CPP/solv_4.0/prep/solv.parm7",
        77,
        "#1f77b4",
    ),
    "BV": SeriesSpec(
        "BV",
        0,
        0.0,
        0.0,
        ROOT / "systems/BV/solv_4.0/us-single15/salt/ready.parm7",
        78,
        "#d62728",
    ),
}


def calculate(spec: SeriesSpec) -> tuple[SeriesSpec, np.ndarray, np.ndarray, dict[str, object]]:
    stage = (
        ROOT
        / f"systems/{spec.system}/solv_4.0/us-single15"
        / f"window-{spec.window_index:03d}/prod-dih"
    )
    trajectory = stage / "traject"
    dftb_input = stage / "dftb.inp"
    for path in (spec.topology, trajectory, dftb_input):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty input: {path}")
    rings = topology_rings(spec)
    source_times, values, source = read_helicity(
        trajectory, rings, read_box(dftb_input)
    )
    mean, std = circular_statistics(values)
    source.update(
        system=spec.system,
        window_index=spec.window_index,
        target_deg=spec.target_deg,
        topology=str(spec.topology.resolve()),
        solute_atoms=spec.solute_atoms,
        rings_one_based=rings,
        frames=len(values),
        source_first_ps=float(source_times[0]),
        source_last_ps=float(source_times[-1]),
        circular_mean_deg=mean,
        circular_std_deg=std,
    )
    return spec, source_times - source_times[0], values, source


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
            ["system", "window_index", "target_deg", "local_time_ps", "helicity_deg"]
        )
        for spec, times, values, _ in loaded:
            writer.writerows(
                (spec.system, spec.window_index, spec.target_deg, time, value)
                for time, value in zip(times, values)
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
            _, times, values, _ = by_key[(system, index)]
            axis.scatter(
                times,
                values,
                color=color,
                s=1.15,
                linewidths=0,
                alpha=0.48,
                rasterized=True,
                label=label,
            )
        axis.set_title(f"{10 * index}° window", fontsize=13)
        axis.set_ylim(-180.0, 180.0)
        axis.grid(False)
        axis.tick_params(labelsize=10)
    axes.flat[-1].set_visible(False)
    axes.flat[0].legend(frameon=False, loc="lower right", fontsize=10)
    fig.supxlabel("Local production time (ps)", fontsize=17)
    fig.supylabel("A–B–C–D ring-center helicity (degrees)", fontsize=17)
    fig.suptitle(
        "Ring-center helicity across CPP and BV single15 umbrella windows\n"
        "BV red · CPP blue · 0°–180°",
        fontsize=19,
    )
    fig.savefig(FIGURE, dpi=300)
    plt.close(fig)

    PROVENANCE.write_text(
        json.dumps(
            {
                "definition": (
                    "Signed dihedral of geometric centers of the five heavy atoms "
                    "of rings A, B, C, and D, using periodic minimum-image ring "
                    "unwrapping and center bonds."
                ),
                "atom_ids_are_one_based": True,
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
