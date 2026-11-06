"""Compare ring-center helicity in the two CPP and BV torsional-US basins."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from cpp_competitor_analysis import read_amber_atom_names
from bv_ring_defect_distances import identify_bv_rings
from plot_bv_equil_helicity_pmf import read_helicity
from qmmd.dihedral import convert_dftb_xyz_to_cpptraj
from qmmd.us_prod_report import read_dftb_trajectory_times
from qmmd.us_pull_report import (
    _cpptraj_path,
    amber_residue_for_atoms,
    read_cpptraj_dihedrals,
)


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    system: str
    window_index: int
    target_deg: float
    pmf_minimum_deg: float
    topology: Path
    solute_atoms: int
    color: str


SPECS = (
    SeriesSpec(
        "CPP",
        0,
        0.0,
        1.0,
        ROOT / "systems/CPP/solv_4.0/prep/solv.parm7",
        77,
        "#2166ac",
    ),
    SeriesSpec(
        "BV",
        1,
        10.0,
        13.0,
        ROOT / "systems/BV/solv_4.0/us-single15/salt/ready.parm7",
        78,
        "#b2182b",
    ),
    SeriesSpec(
        "CPP",
        16,
        160.0,
        179.0,
        ROOT / "systems/CPP/solv_4.0/prep/solv.parm7",
        77,
        "#2166ac",
    ),
    SeriesSpec(
        "BV",
        16,
        160.0,
        161.0,
        ROOT / "systems/BV/solv_4.0/us-single15/salt/ready.parm7",
        78,
        "#b2182b",
    ),
)

OUTPUT_ROOT = ROOT / "systems/CPP/solv_4.0/us-single15"
FIGURE = OUTPUT_ROOT / "comparison_cpp_bv_helicity.png"
TIMESERIES_FIGURE = OUTPUT_ROOT / "comparison_cpp_bv_helicity_timeseries.png"
TIMESERIES = OUTPUT_ROOT / "comparison_cpp_bv_helicity_timeseries.csv"
DENSITIES = OUTPUT_ROOT / "comparison_cpp_bv_helicity_density.csv"
PROVENANCE = OUTPUT_ROOT / "comparison_cpp_bv_helicity_provenance.json"
DIHEDRAL_FIGURE = OUTPUT_ROOT / "comparison_cpp_bv_c2a_c3a_caa_cba_timeseries.png"
DIHEDRAL_ANTI_FIGURE = (
    OUTPUT_ROOT / "comparison_cpp_bv_c2a_c3a_caa_cba_anti_timeseries.png"
)
DIHEDRAL_SYSTEM_FIGURE = (
    OUTPUT_ROOT / "comparison_cpp_bv_c2a_c3a_caa_cba_basin_timeseries.png"
)
DIHEDRAL_DATA = OUTPUT_ROOT / "comparison_cpp_bv_c2a_c3a_caa_cba_timeseries.csv"
DIHEDRAL_PROVENANCE = OUTPUT_ROOT / "comparison_cpp_bv_c2a_c3a_caa_cba_provenance.json"
BIN_WIDTH_DEG = 2.0
DIHEDRAL_NAMES = ("C2A", "C3A", "CAA", "CBA")


def circular_statistics(values_deg: np.ndarray) -> tuple[float, float]:
    radians = np.deg2rad(values_deg)
    mean_sine = float(np.mean(np.sin(radians)))
    mean_cosine = float(np.mean(np.cos(radians)))
    mean = float(np.rad2deg(np.arctan2(mean_sine, mean_cosine)))
    resultant = min(1.0, max(np.finfo(float).tiny, np.hypot(mean_sine, mean_cosine)))
    std = float(np.rad2deg(np.sqrt(-2.0 * np.log(resultant))))
    return mean, std


def topology_rings(spec: SeriesSpec) -> dict[str, list[int]]:
    names = read_amber_atom_names(spec.topology, spec.solute_atoms)
    nitrogen_ids: list[int] = []
    for label in "ABCD":
        name = "N" + label
        if names.count(name) != 1:
            raise ValueError(f"Expected one {name} in {spec.topology}")
        nitrogen_ids.append(names.index(name) + 1)
    rings = identify_bv_rings(spec.topology, spec.solute_atoms, nitrogen_ids)
    for label, ids in rings.items():
        expected = {prefix + label for prefix in ("N", "C1", "C2", "C3", "C4")}
        if {names[index - 1] for index in ids} != expected:
            raise ValueError(f"Ring {label} atom names do not match in {spec.topology}")
    return rings


def read_box(path: Path) -> np.ndarray:
    vectors = np.asarray(
        [
            [float(value) for value in line.split()[1:4]]
            for line in path.read_text(errors="replace").splitlines()
            if line.startswith("TV")
        ]
    )
    if (
        vectors.shape != (3, 3)
        or not np.allclose(vectors, np.diag(np.diag(vectors)), atol=1.0e-8)
        or np.any(np.diag(vectors) <= 0.0)
    ):
        raise ValueError(f"Expected a fixed orthorhombic box in {path}")
    return np.diag(vectors)


def resolve_unique_atom_ids(spec: SeriesSpec, atom_names: tuple[str, ...]) -> tuple[int, ...]:
    """Resolve unique one-based IDs for user-facing atom names."""
    topology_names = read_amber_atom_names(spec.topology, spec.solute_atoms)
    ids: list[int] = []
    for atom_name in atom_names:
        matches = [index + 1 for index, name in enumerate(topology_names) if name == atom_name]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {atom_name} among the first {spec.solute_atoms} "
                f"atoms of {spec.topology}; found {len(matches)}"
            )
        ids.append(matches[0])
    return tuple(ids)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def copy_complete_xyz_snapshot(source: Path, destination: Path) -> tuple[int, bool]:
    """Copy a fixed-size snapshot and discard any incomplete trailing XYZ frame."""
    source_size = source.stat().st_size
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        remaining = source_size
        while remaining:
            block = input_stream.read(min(1024 * 1024, remaining))
            if not block:
                break
            output_stream.write(block)
            remaining -= len(block)
    incomplete_tail = False
    last_complete_offset = 0
    with destination.open("rb+") as stream:
        while True:
            count_line = stream.readline()
            if not count_line:
                break
            if not count_line.strip():
                last_complete_offset = stream.tell()
                continue
            try:
                atom_count = int(count_line)
            except ValueError as exc:
                raise ValueError(f"Malformed XYZ atom count in {source}") from exc
            comment = stream.readline()
            rows = [stream.readline() for _ in range(atom_count)]
            if not comment or any(not row for row in rows) or not rows[-1].endswith(b"\n"):
                incomplete_tail = True
                break
            last_complete_offset = stream.tell()
        stream.truncate(last_complete_offset)
    return last_complete_offset, incomplete_tail


def calculate_named_dihedral_with_cpptraj(
    spec: SeriesSpec, stage: Path, atom_ids: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Calculate a named dihedral with cpptraj from all complete DFTB frames."""
    trajectory = stage / "traject"
    dftb_input = stage / "dftb.inp"
    data_path = stage / "c2a_c3a_caa_cba.dat"
    input_path = stage / "c2a_c3a_caa_cba.cpptraj.in"
    log_path = stage / "c2a_c3a_caa_cba.cpptraj.log"
    residue, _ = amber_residue_for_atoms(spec.topology, atom_ids)
    executable = shutil.which("cpptraj")
    if executable is None:
        installed = Path("/nas/sycamore/apps/amber/26/ambertools26/bin/cpptraj")
        if not installed.is_file():
            raise RuntimeError("cpptraj was not found")
        executable = str(installed)
    with tempfile.TemporaryDirectory(prefix="cpp-bv-dihedral-") as tmp:
        snapshot = Path(tmp) / "traject.snapshot.xyz"
        converted = Path(tmp) / "traject.cpptraj.xyz"
        snapshot_bytes, incomplete_tail = copy_complete_xyz_snapshot(
            trajectory, snapshot
        )
        frame_records = read_dftb_trajectory_times(snapshot)
        snapshot_hash = file_sha256(snapshot)
        convert_dftb_xyz_to_cpptraj(snapshot, converted, dftb_input)
        input_path.write_text(
            "\n".join(
                [
                    f"parm {_cpptraj_path(spec.topology)}",
                    f"trajin {_cpptraj_path(converted)}",
                    f"fiximagedbonds :{residue}",
                    "autoimage",
                    "dihedral C2A_C3A_CAA_CBA "
                    + " ".join(f"@{atom_id}" for atom_id in atom_ids)
                    + f" out {_cpptraj_path(data_path)}",
                    "run",
                    "quit",
                    "",
                ]
            )
        )
        with log_path.open("w") as log:
            subprocess.run(
                [executable, "-i", input_path.name],
                cwd=stage,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                text=True,
            )
    values = np.asarray(read_cpptraj_dihedrals(data_path))
    if len(values) != len(frame_records):
        raise RuntimeError(
            f"cpptraj returned {len(values)} values for {len(frame_records)} "
            f"complete frames in {trajectory}"
        )
    source_times = np.asarray([time_ps for _, time_ps in frame_records])
    return source_times, values, {
        "source": str(trajectory.resolve()),
        "snapshot_bytes": snapshot_bytes,
        "snapshot_sha256": snapshot_hash,
        "incomplete_tail_ignored": incomplete_tail,
        "cpptraj_input": str(input_path.resolve()),
        "cpptraj_log": str(log_path.resolve()),
        "cpptraj_data": str(data_path.resolve()),
        "cpptraj_executable": str(Path(executable).resolve()),
    }


def load_series(spec: SeriesSpec) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
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
    times, helicity, source = read_helicity(
        trajectory, rings, read_box(dftb_input)
    )
    local_times = times - times[0]
    mean, std = circular_statistics(helicity)
    source.update(
        system=spec.system,
        window_index=spec.window_index,
        target_deg=spec.target_deg,
        pmf_minimum_deg=spec.pmf_minimum_deg,
        topology=str(spec.topology.resolve()),
        solute_atoms=spec.solute_atoms,
        rings_one_based=rings,
        frames=len(helicity),
        source_first_ps=float(times[0]),
        source_last_ps=float(times[-1]),
        circular_mean_deg=mean,
        circular_std_deg=std,
    )
    return local_times, helicity, source


def histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    density = counts / (counts.sum() * np.diff(edges))
    if not np.isclose(float(np.sum(density * np.diff(edges))), 1.0):
        raise RuntimeError("Helicity density did not normalize to one")
    return density


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    edges = np.arange(-180.0, 180.0 + BIN_WIDTH_DEG, BIN_WIDTH_DEG)
    centers = 0.5 * (edges[:-1] + edges[1:])
    loaded: list[tuple[SeriesSpec, np.ndarray, np.ndarray, dict[str, object]]] = []
    for spec in SPECS:
        times, helicity, provenance = load_series(spec)
        loaded.append((spec, times, helicity, provenance))

    with TIMESERIES.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["system", "window_index", "target_deg", "local_time_ps", "helicity_deg"]
        )
        for spec, times, helicity, _ in loaded:
            writer.writerows(
                (spec.system, spec.window_index, spec.target_deg, time, value)
                for time, value in zip(times, helicity)
            )

    density_rows: list[tuple[object, ...]] = []
    densities: dict[tuple[str, int], np.ndarray] = {}
    for spec, _, helicity, _ in loaded:
        density = histogram(helicity, edges)
        densities[(spec.system, spec.window_index)] = density
        density_rows.extend(
            (spec.system, spec.window_index, spec.target_deg, center, value)
            for center, value in zip(centers, density)
        )
    with DENSITIES.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "system",
                "window_index",
                "target_deg",
                "helicity_bin_center_deg",
                "probability_density_per_deg",
            ]
        )
        writer.writerows(density_rows)

    comparisons = []
    for cpp_index, bv_index, label in ((0, 1, "near_0_deg"), (16, 16, "near_160_deg")):
        overlap = float(
            np.sum(
                np.minimum(
                    densities[("CPP", cpp_index)], densities[("BV", bv_index)]
                )
                * np.diff(edges)
            )
        )
        comparisons.append({"basin": label, "histogram_overlap": overlap})

    plt.style.use(Path(__file__).with_name("lefteris.mplstyle"))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), sharex=True, sharey=True)
    panels = (
        (axes[0], (("CPP", 0), ("BV", 1)), "Low-angle basin"),
        (axes[1], (("CPP", 16), ("BV", 16)), "High-angle basin"),
    )
    by_key = {(row[0].system, row[0].window_index): row for row in loaded}
    for axis, keys, title in panels:
        for key in keys:
            spec, _, _, provenance = by_key[key]
            mean = float(provenance["circular_mean_deg"])
            std = float(provenance["circular_std_deg"])
            axis.plot(
                centers,
                densities[key],
                color=spec.color,
                linewidth=2.4,
                label=(
                    f"{spec.system} {spec.target_deg:g}° window: "
                    f"{mean:.1f} ± {std:.1f}°"
                ),
            )
        comparison = comparisons[0 if "Low" in title else 1]
        axis.text(
            0.03,
            0.95,
            f"distribution overlap = {comparison['histogram_overlap']:.2f}",
            transform=axis.transAxes,
            va="top",
        )
        axis.set_title(title)
        axis.set_xlim(-75.0, 75.0)
        axis.set_xlabel("A–B–C–D ring-center helicity (degrees)")
        axis.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel(r"Normalized density (degree$^{-1}$)")
    fig.suptitle("CPP versus BV helicity in the single15 torsional basins")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), sharex=True, sharey=True)
    for axis, key in zip(axes, (("CPP", 0), ("BV", 1))):
        spec, times, helicity, provenance = by_key[key]
        mean = float(provenance["circular_mean_deg"])
        axis.plot(times, helicity, color=spec.color, linewidth=0.8, alpha=0.88)
        axis.axhline(
            mean,
            color=spec.color,
            linewidth=1.8,
            linestyle="--",
            label=f"circular mean = {mean:.1f}°",
        )
        axis.set_ylabel("Helicity (degrees)")
        axis.set_ylim(-75.0, 75.0)
        axis.set_title(
            f"{spec.system} window {spec.target_deg:g}° "
            f"(PMF minimum near {spec.pmf_minimum_deg:g}°)"
        )
        axis.legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("Local production time (ps)")
    fig.suptitle("Low-angle-basin ring-center helicity time series")
    fig.tight_layout()
    fig.savefig(TIMESERIES_FIGURE, dpi=300)
    plt.close(fig)

    dihedral_loaded: list[
        tuple[SeriesSpec, np.ndarray, np.ndarray, dict[str, object]]
    ] = []
    for key in (("CPP", 0), ("BV", 1), ("CPP", 16), ("BV", 16)):
        spec, _, _, _ = by_key[key]
        stage = (
            ROOT
            / f"systems/{spec.system}/solv_4.0/us-single15"
            / f"window-{spec.window_index:03d}/prod-dih"
        )
        atom_ids = resolve_unique_atom_ids(spec, DIHEDRAL_NAMES)
        source_times, values, source = calculate_named_dihedral_with_cpptraj(
            spec, stage, atom_ids
        )
        local_times = source_times - source_times[0]
        mean, std = circular_statistics(values)
        source.update(
            system=spec.system,
            window_index=spec.window_index,
            target_deg=spec.target_deg,
            topology=str(spec.topology.resolve()),
            atom_names=list(DIHEDRAL_NAMES),
            atom_ids_one_based=list(atom_ids),
            frames=len(values),
            source_first_ps=float(source_times[0]),
            source_last_ps=float(source_times[-1]),
            circular_mean_deg=mean,
            circular_std_deg=std,
        )
        dihedral_loaded.append((spec, local_times, values, source))

    with DIHEDRAL_DATA.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "system",
                "window_index",
                "target_deg",
                "basin",
                "local_time_ps",
                "C2A_C3A_CAA_CBA_deg",
            ]
        )
        for loaded_index, (spec, times, values, _) in enumerate(dihedral_loaded):
            basin = "low_angle" if loaded_index < 2 else "anti_high_angle"
            writer.writerows(
                (spec.system, spec.window_index, spec.target_deg, basin, time, value)
                for time, value in zip(times, values)
            )

    def plot_dihedral_pair(
        rows: list[tuple[SeriesSpec, np.ndarray, np.ndarray, dict[str, object]]],
        title: str,
        output: Path,
        mean_centered_branch: bool = False,
    ) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), sharex=True, sharey=True)
        plotted_values: list[np.ndarray] = []
        for _, _, values, source in rows:
            mean = float(source["circular_mean_deg"])
            plotted_values.append(
                mean + (values - mean + 180.0) % 360.0 - 180.0
                if mean_centered_branch
                else values
            )
        for axis, (spec, times, values, source) in zip(axes, rows):
            mean = float(source["circular_mean_deg"])
            std = float(source["circular_std_deg"])
            plot_values = (
                mean + (values - mean + 180.0) % 360.0 - 180.0
                if mean_centered_branch
                else values
            )
            axis.plot(times, plot_values, color=spec.color, linewidth=0.8, alpha=0.88)
            axis.axhline(
                mean,
                color=spec.color,
                linewidth=1.8,
                linestyle="--",
                label=f"circular mean = {mean:.1f} ± {std:.1f}°",
            )
            axis.set_title(f"{spec.system} {spec.target_deg:g}° window")
            axis.legend(frameon=False, loc="upper right")
        if mean_centered_branch:
            lower = min(float(np.min(values)) for values in plotted_values) - 10.0
            upper = max(float(np.max(values)) for values in plotted_values) + 10.0
            axes[0].set_ylim(lower, upper)
        else:
            axes[0].set_ylim(-180.0, 180.0)
        axes[-1].set_xlabel("Local production time (ps)")
        fig.supylabel(
            "Periodic dihedral (degrees)"
            if mean_centered_branch
            else "Dihedral (degrees)"
        )
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output, dpi=300)
        plt.close(fig)

    plot_dihedral_pair(
        dihedral_loaded[:2],
        "C2A–C3A–CAA–CBA dihedral in the low-angle basins",
        DIHEDRAL_FIGURE,
    )
    plot_dihedral_pair(
        dihedral_loaded[2:],
        "C2A–C3A–CAA–CBA dihedral in the anti/high-angle basins\n"
        "(continuous periodic branches; raw cpptraj angles retained in CSV)",
        DIHEDRAL_ANTI_FIGURE,
        mean_centered_branch=True,
    )

    by_dihedral_key = {
        (row[0].system, row[0].window_index): row for row in dihedral_loaded
    }
    system_pairs = (
        ("CPP", by_dihedral_key[("CPP", 0)], by_dihedral_key[("CPP", 16)]),
        ("BV", by_dihedral_key[("BV", 1)], by_dihedral_key[("BV", 16)]),
    )
    branch_arrays: list[np.ndarray] = []
    for _, syn_row, anti_row in system_pairs:
        for _, _, values, source in (syn_row, anti_row):
            mean = float(source["circular_mean_deg"])
            branch_arrays.append(mean + (values - mean + 180.0) % 360.0 - 180.0)
    branch_lower = min(float(np.min(values)) for values in branch_arrays) - 10.0
    branch_upper = max(float(np.max(values)) for values in branch_arrays) + 10.0

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), sharex=True, sharey=True)
    for axis, (system, syn_row, anti_row) in zip(axes, system_pairs):
        for basin, color, row in (
            ("syn", "#b2182b", syn_row),
            ("anti", "#2166ac", anti_row),
        ):
            spec, times, values, source = row
            mean = float(source["circular_mean_deg"])
            branch = mean + (values - mean + 180.0) % 360.0 - 180.0
            axis.plot(
                times,
                branch,
                color=color,
                linewidth=0.85,
                alpha=0.86,
                label=f"{basin} ({spec.target_deg:g}° window)",
            )
        axis.set_title(system)
        axis.legend(frameon=False, loc="upper right", fontsize=11)
        axis.set_ylim(branch_lower, branch_upper)
    axes[-1].set_xlabel("Local production time (ps)")
    fig.supylabel("Periodic C2A–C3A–CAA–CBA dihedral (degrees)")
    fig.suptitle("Syn (red) and anti (blue) basin dynamics")
    fig.tight_layout()
    fig.savefig(DIHEDRAL_SYSTEM_FIGURE, dpi=300)
    plt.close(fig)

    DIHEDRAL_PROVENANCE.write_text(
        json.dumps(
            {
                "definition": (
                    "cpptraj dihedral C2A-C3A-CAA-CBA after fiximagedbonds "
                    "and autoimage."
                ),
                "atom_ids_are_one_based": True,
                "series": [row[3] for row in dihedral_loaded],
                "low_angle_figure": str(DIHEDRAL_FIGURE.resolve()),
                "anti_high_angle_figure": str(DIHEDRAL_ANTI_FIGURE.resolve()),
                "within_system_basin_figure": str(DIHEDRAL_SYSTEM_FIGURE.resolve()),
                "timeseries_csv": str(DIHEDRAL_DATA.resolve()),
            },
            indent=2,
        )
        + "\n"
    )

    PROVENANCE.write_text(
        json.dumps(
            {
                "definition": (
                    "Signed dihedral of geometric centers of the five heavy atoms "
                    "of rings A, B, C, and D, with periodic minimum-image ring "
                    "unwrapping and center bonds; identical to acid_base_BV.py."
                ),
                "bin_width_deg": BIN_WIDTH_DEG,
                "density_normalization": "Each trajectory integrates to one.",
                "series": [row[3] for row in loaded],
                "comparisons": comparisons,
                "figure": str(FIGURE.resolve()),
                "timeseries_figure": str(TIMESERIES_FIGURE.resolve()),
                "timeseries_csv": str(TIMESERIES.resolve()),
                "density_csv": str(DENSITIES.resolve()),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"OK: wrote {FIGURE}")
    print(f"OK: wrote {TIMESERIES_FIGURE}")
    print(f"OK: wrote {TIMESERIES}")
    print(f"OK: wrote {DENSITIES}")
    print(f"OK: wrote {DIHEDRAL_FIGURE}")
    print(f"OK: wrote {DIHEDRAL_ANTI_FIGURE}")
    print(f"OK: wrote {DIHEDRAL_SYSTEM_FIGURE}")
    print(f"OK: wrote {DIHEDRAL_DATA}")


if __name__ == "__main__":
    main()
