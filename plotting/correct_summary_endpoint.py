#!/usr/bin/env python3
"""Render the BV meta-hic four-run summary using endpoint pKa values."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acid_base_BV import (
    _minimum_image,
    closest_solute_heavy_atom,
    iter_xyz_frames,
    read_box_lengths_from_dftb_inp,
    read_xyz_symbols,
)
from correct_summary import load_run, read_table, save_csv, smooth_helicity
from ion_pair_pmf import calculate_profile, save_profile_data, save_summary


ROOT = Path(__file__).resolve().parents[1]
COLORS = ["green", "black", "red", "blue"]


def continuous_radial_pmf(profile) -> tuple[np.ndarray, np.ndarray]:
    """Join the modeled repulsive wall to the sampled smooth radial PMF."""
    finite = np.isfinite(profile.kde_pmf_kcal_mol)
    sampled_grid = profile.grid_A[finite]
    sampled_pmf = profile.kde_pmf_kcal_mol[finite]
    if not profile.wall_grid_A.size:
        return sampled_grid, sampled_pmf

    anchor = float(profile.wall_grid_A[-1])
    beyond_anchor = sampled_grid > anchor
    return (
        np.concatenate((profile.wall_grid_A, sampled_grid[beyond_anchor])),
        np.concatenate((profile.wall_pmf_kcal_mol, sampled_pmf[beyond_anchor])),
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def endpoint_free_energy(data: dict, window_ps: float) -> float:
    """Interpolate Delta F at the exact raw tdiff + window timestamp."""
    series = read_table(data["report_folder"] / "summary_pka.csv")
    times = np.asarray(series["time_ps"], dtype=float)
    values = np.asarray(series["pka"], dtype=float) * data["factor"]
    target = data["raw_marker"] + window_ps
    finite = np.isfinite(times) & np.isfinite(values)
    times, values = times[finite], values[finite]
    order = np.argsort(times)
    times, values = times[order], values[order]
    if not len(times) or target < times[0] or target > times[-1]:
        raise ValueError(f"Delta-F series does not cover endpoint for run {data['run']}")
    return float(np.interp(target, times, values))


def auxiliary_series(
    data: dict,
    cache_dir: Path,
    window_ps: float,
    cache_prefix: str,
    solute_atoms: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple], list[tuple]
]:
    """Load wire assignments and carry defect identity through assignment gaps."""
    label = f"{cache_prefix}-{data['run']}"
    endpoint = data["marker"] + window_ps
    wire_rows = read_rows(cache_dir / f"{label}_coordination_wires.csv")
    wire_rows = [
        row for row in wire_rows
        if float(row["aligned_time_ps"]) <= endpoint + 1e-8
    ]

    target_times = {round(float(row["raw_time_ps"]), 8) for row in wire_rows}
    frames: dict[float, np.ndarray] = {}
    raw_endpoint = data["raw_marker"] + window_ps
    for time, coordinates in iter_xyz_frames(Path(data["source"]) / "traject"):
        if time > raw_endpoint + 1e-8:
            break
        key = round(float(time), 8)
        if key in target_times:
            frames[key] = coordinates
    box = read_box_lengths_from_dftb_inp(Path(data["source"]) / "dftb.inp")
    symbols = read_xyz_symbols(Path(data["source"]) / "traject")
    solute_heavy_ids = [
        atom_id
        for atom_id, symbol in enumerate(symbols[:solute_atoms], start=1)
        if symbol.upper() != "H"
    ]

    distance_export: list[tuple] = []
    previous_defect = np.nan
    for row in wire_rows:
        raw_time = float(row["raw_time_ps"])
        aligned_time = float(row["aligned_time_ps"])
        assigned_defect = float(row["defect_oxygen_id"])
        identity_carried = not np.isfinite(assigned_defect) and np.isfinite(previous_defect)
        if np.isfinite(assigned_defect):
            previous_defect = assigned_defect
        effective_defect = previous_defect
        nearest_atom = np.nan
        distance = np.nan
        nearest_heavy_atom = np.nan
        heavy_distance = np.nan
        coordinates = frames.get(round(raw_time, 8))
        if np.isfinite(effective_defect) and coordinates is not None:
            defect_index = int(effective_defect) - 1
            deltas = _minimum_image(
                coordinates[:solute_atoms] - coordinates[defect_index], box
            )
            distances = np.linalg.norm(deltas, axis=1)
            nearest_atom = int(np.argmin(distances)) + 1
            distance = float(distances[nearest_atom - 1])
            heavy_id, heavy_distance = closest_solute_heavy_atom(
                coordinates, int(effective_defect), solute_heavy_ids, box
            )
            nearest_heavy_atom = heavy_id
        distance_export.append(
            (
                data["run"], raw_time, aligned_time, assigned_defect,
                effective_defect, nearest_atom, distance, nearest_heavy_atom,
                heavy_distance, int(identity_carried),
            )
        )

    wire_export: list[tuple] = []
    for row in wire_rows:
        time = float(row["aligned_time_ps"])
        if time <= endpoint + 1e-8:
            bridging = float(row["hbond_bridging_waters"])
            bond_count = bridging + 1.0 if np.isfinite(bridging) and bridging >= 0 else np.nan
            wire_export.append(
                (
                    data["run"], float(row["raw_time_ps"]), time,
                    row["coordination_s"], row["defect_oxygen_id"],
                    row["binding_nitrogen_id"], bridging, bond_count,
                    row["oxygen_path_ids"],
                )
            )

    if not distance_export or distance_export[-1][2] < endpoint - 0.03:
        raise ValueError(f"Distance cache does not cover endpoint for run {data['run']}")
    if not wire_export or wire_export[-1][2] < endpoint - 0.03:
        raise ValueError(f"H-bond cache does not cover endpoint for run {data['run']}")

    distance_time = np.asarray([row[2] for row in distance_export], dtype=float)
    distance = np.asarray([float(row[6]) for row in distance_export], dtype=float)
    heavy_distance = np.asarray([float(row[8]) for row in distance_export], dtype=float)
    wire_time = np.asarray([row[2] for row in wire_export], dtype=float)
    bond_count = np.asarray([float(row[7]) for row in wire_export], dtype=float)
    return (
        distance_time, distance, heavy_distance, wire_time, bond_count,
        distance_export, wire_export,
    )


def render(
    data: list[dict],
    out: Path,
    cache_dir: Path,
    window_ps: float,
    smooth_ps: float,
    cache_prefix: str,
    solute_atoms: int,
    binding_nitrogen_id: int,
    system_label: str,
) -> Path:
    """Render a 3x2 summary and save all aligned numerical series."""
    plt.style.use(Path(__file__).with_name("lefteris.mplstyle"))
    fig = plt.figure(figsize=(15, 16), layout="constrained")
    grid = fig.add_gridspec(3, 2)
    ax_fes = fig.add_subplot(grid[0, 0])
    ax_df = fig.add_subplot(grid[0, 1])
    ax_bar = fig.add_subplot(grid[1, 0])
    ax_helix = fig.add_subplot(grid[1, 1])
    ax_distance = fig.add_subplot(grid[2, 0])
    wire_grid = grid[2, 1].subgridspec(len(data), 1, hspace=0.0)
    wire_axes = []
    radial_profiles = []

    fes_rows: list[tuple] = []
    df_rows: list[tuple] = []
    endpoint_rows: list[tuple] = []
    helicity_rows: list[tuple] = []
    distance_rows: list[tuple] = []
    wire_rows: list[tuple] = []
    endpoint_dfs: list[float] = []
    endpoint_pkas: list[float] = []
    all_fes = []

    xmax = max(d["marker"] + window_ps for d in data)
    for index, (d, color) in enumerate(zip(data, COLORS)):
        run = d["run"]
        endpoint = d["marker"] + window_ps
        endpoint_df = endpoint_free_energy(d, window_ps)
        endpoint_pka = endpoint_df / d["factor"]
        endpoint_dfs.append(endpoint_df)
        endpoint_pkas.append(endpoint_pka)

        ax_fes.plot(
            d["fes"][:, 0], d["fes"][:, 1], color=color, alpha=0.32, lw=1.8,
            label=f"{run}: t={d['snapshot_time']:.2f} ps",
        )
        all_fes.append(d["fes"][:, 1])
        ax_df.plot(d["time"], d["df"], color=color, lw=1.7, label=f"run {run}")
        ax_df.axvline(d["marker"], color=color, ls="--", lw=1.0, alpha=0.65)
        ax_df.scatter(
            [endpoint], [endpoint_df], color=color, edgecolor="black", marker="*",
            linewidth=0.5, s=85, zorder=5,
        )
        ax_bar.bar(
            index, endpoint_df, color=color, alpha=0.85,
            label=rf"{run}: p$K_a$ = {endpoint_pka:.2f}",
        )
        ax_bar.annotate(
            f"{endpoint_df:.2f}", (index, endpoint_df), xytext=(0, 5),
            textcoords="offset points", ha="center", fontsize=11,
        )

        smoothed = smooth_helicity(d["helix_time"], d["helicity"], smooth_ps)
        ax_helix.plot(d["helix_time"], smoothed, color=color, lw=1.6, label=f"run {run}")
        ax_helix.axvline(d["treact"], color=color, ls="--", lw=1.0, alpha=0.65)

        (distance_time, distance, heavy_distance, wire_time, bond_count,
         distance_export, wire_export) = auxiliary_series(
            d, cache_dir, window_ps, cache_prefix, solute_atoms
        )
        ax_distance.scatter(
            distance_time, distance, color=color, s=11, linewidths=0,
            alpha=0.82, label=f"run {run}",
        )
        wire_axis = fig.add_subplot(
            wire_grid[index], sharex=wire_axes[0] if wire_axes else None,
            sharey=wire_axes[0] if wire_axes else None,
        )
        wire_axes.append(wire_axis)
        wire_axis.scatter(wire_time, bond_count, color=color, s=12, linewidths=0)
        wire_axis.text(
            0.985, 0.80, f"run {run}", color=color, ha="right", va="top",
            transform=wire_axis.transAxes, fontsize=10,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            profile = calculate_profile(
                label=f"run-{run}",
                source=out / "defect_distance.csv",
                times_ps=distance_time,
                distances_A=heavy_distance,
                distance_column="closest_solute_heavy_distance_A",
                time_min_ps=None,
                time_max_ps=endpoint,
                radial_min_A=2.0,
                radial_max_A=9.5,
                bin_width_A=0.25,
                temperature_K=d["temperature"],
                kde_bandwidth_factor=0.13,
                kde_grid_points=1000,
                minimum_bin_count=2,
            )
        radial_profiles.append(profile)

        fes_rows.extend((run, d["marker"], d["snapshot_time"], s, f)
                        for s, f in d["fes"][:, :2])
        df_rows.extend((run, t, value, value / d["factor"])
                       for t, value in zip(d["time"], d["df"]))
        endpoint_rows.append(
            (run, d["marker"], endpoint, d["raw_marker"] + window_ps,
             d["temperature"], endpoint_df, endpoint_pka)
        )
        helicity_rows.extend((run, t, raw, smooth)
                             for t, raw, smooth in zip(d["helix_time"], d["helicity"], smoothed))
        distance_rows.extend(distance_export)
        wire_rows.extend(wire_export)

    fes_grid = data[0]["fes"][:, 0]
    for d in data[1:]:
        if d["fes"].shape[0] != len(fes_grid) or not np.allclose(
            d["fes"][:, 0], fes_grid, rtol=0, atol=1e-10
        ):
            raise ValueError("FES grids differ; cannot form a pointwise mean")
    fes_values = np.stack(all_fes)
    fes_mean = np.mean(fes_values, axis=0)
    fes_std = np.std(fes_values, axis=0)
    ax_fes.plot(fes_grid, fes_mean, color="#1F3A5F", lw=3.2,
                label=f"Mean F(s), {len(data)} runs", zorder=5)

    mean_df = float(np.mean(endpoint_dfs))
    mean_pka = float(np.mean(endpoint_pkas))
    ax_bar.text(
        0.03, 0.96,
        rf"Across {len(data)} runs at endpoint:" + "\n"
        + rf"$\langle\Delta F\rangle={mean_df:.2f}$ kcal mol$^{{-1}}$" + "\n"
        + rf"$\langle\mathrm{{p}}K_a\rangle={mean_pka:.2f}$",
        transform=ax_bar.transAxes, ha="left", va="top", fontsize=12,
    )

    ax_fes.set(
        title=rf"FES at $t_{{\rm diff}}+{window_ps:g}$ ps",
        xlabel=r"coordination, $s$", ylabel=r"$F(s)$ (kcal mol$^{-1}$)", xlim=(0, 1),
    )
    ax_df.set(
        title="Free-energy difference", xlabel="t (ps; metadynamics starts at 40)",
        ylabel=r"$\Delta F$ (kcal mol$^{-1}$)", xlim=(40, xmax),
    )
    ax_bar.set(
        title=rf"Single value at $t_{{\rm diff}}+{window_ps:g}$ ps",
        xlabel="Run ID", ylabel=r"$\Delta F(t_{\rm diff}+1.75)$ (kcal mol$^{-1}$)",
        xticks=range(len(data)), xticklabels=[str(d["run"]) for d in data],
    )
    ax_bar.margins(y=0.42)
    ax_helix.set(
        title=rf"Ring-center helicity ({smooth_ps:g} ps smoothing)",
        xlabel="t (ps; metadynamics starts at 40)", ylabel="Helicity (deg)",
        xlim=(40, xmax),
    )
    ax_distance.set(
        title="Defect proximity", xlabel="t (ps; metadynamics starts at 40)",
        ylabel="Defect–nearest solute atom (Å)", xlim=(40, xmax), ylim=(0, None),
    )
    for index, axis in enumerate(wire_axes):
        axis.set_xlim(40, xmax)
        axis.set_ylim(-0.6, 9.6)
        axis.set_yticks([0, 3, 6, 9])
        axis.set_yticks(range(10), minor=True)
        axis.tick_params(labelbottom=index == len(wire_axes) - 1)
        axis.grid(axis="y", alpha=0.2)
    wire_axes[-1].set_xlabel("t (ps; metadynamics starts at 40)")
    middle_wire = wire_axes[len(wire_axes) // 2]
    middle_wire.set_ylabel("H-bond wire length (number of bonds)")
    for axis in (ax_fes, ax_df, ax_bar, ax_helix, ax_distance):
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=10, loc="best")
    fig.suptitle(f"BV · {system_label} · N1T48C1 · solv 4.0", fontsize=20)
    plot = out / "summary.png"
    fig.savefig(plot, dpi=220)
    plt.close(fig)

    save_csv(out / "fes.csv",
             ["run", "tdiff_aligned_ps", "snapshot_aligned_ps", "s", "free_energy_kcal_mol"],
             fes_rows)
    save_csv(out / "fes_mean.csv",
             ["s", "mean_free_energy_kcal_mol", "std_free_energy_kcal_mol"],
             zip(fes_grid, fes_mean, fes_std))
    save_csv(out / "delta_f.csv",
             ["run", "aligned_time_ps", "delta_f_kcal_mol", "pka"], df_rows)
    save_csv(out / "samples.csv",
             ["run", "aligned_time_ps", "delta_f_kcal_mol", "pka"],
             [(row[0], row[2], row[5], row[6]) for row in endpoint_rows])
    save_csv(out / "statistics.csv",
             ["run", "tdiff_aligned_ps", "endpoint_aligned_ps", "endpoint_raw_ps",
              "temperature_K", "endpoint_df_kcal_mol", "endpoint_pka"], endpoint_rows)
    save_csv(out / "ensemble_statistics.csv",
             ["run_count", "mean_endpoint_df_kcal_mol", "mean_endpoint_pka"],
             [(len(data), mean_df, mean_pka)])
    save_csv(out / "helicity.csv",
             ["run", "aligned_time_ps", "helicity_deg", "smoothed_helicity_deg"],
             helicity_rows)
    save_csv(out / "defect_distance.csv",
             ["run", "raw_time_ps", "aligned_time_ps", "assigned_defect_oxygen_id",
              "effective_defect_oxygen_id", "nearest_solute_atom_id", "distance_A",
              "nearest_solute_heavy_atom_id", "closest_solute_heavy_distance_A",
              "identity_carried_forward"],
             distance_rows)
    save_csv(out / "hydrogen_bond_length.csv",
             ["run", "raw_time_ps", "aligned_time_ps", "coordination_s",
              "defect_oxygen_id", "binding_nitrogen_id", "bridging_water_count",
              "hydrogen_bond_count", "oxygen_path_ids"], wire_rows)
    for profile in radial_profiles:
        save_profile_data(profile, out)
    save_summary(radial_profiles, out / "radial_pmf_summary.csv")

    pmf_fig, pmf_axes = plt.subplots(
        2, 2, figsize=(13, 9.5), sharex=True, layout="constrained"
    )
    cip_rows: list[tuple] = []
    for index, (axis, profile, color) in enumerate(
        zip(pmf_axes.flat, radial_profiles, COLORS)
    ):
        finite = np.isfinite(profile.binned_pmf_kcal_mol)
        axis.scatter(
            profile.bin_centers_A[finite],
            profile.binned_pmf_kcal_mol[finite],
            color=color,
            s=22,
            alpha=0.28,
            linewidths=0,
            label="0.25 Å shell bins",
        )
        curve_grid, curve_pmf = continuous_radial_pmf(profile)
        axis.plot(
            curve_grid,
            curve_pmf,
            color=color,
            lw=4.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            label="smooth radial PMF",
        )
        if not profile.minima_indices:
            raise ValueError(f"No contact ion-pair minimum found for {profile.label}")
        cip_index = profile.minima_indices[0]
        cip_distance = float(profile.grid_A[cip_index])
        cip_pmf = float(profile.kde_pmf_kcal_mol[cip_index])
        cip_rows.append(
            (profile.label.removeprefix("run-"), cip_distance, cip_pmf)
        )
        axis.scatter(
            [cip_distance], [cip_pmf], marker="D", s=70, color="white",
            edgecolor=color, linewidth=2.0, zorder=5,
            label="contact ion-pair minimum" if index == 0 else None,
        )
        axis.annotate(
            f"CIP\n{cip_distance:.2f} Å",
            (cip_distance, cip_pmf),
            xytext=(9, 12),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color=color,
            ha="left",
            va="bottom",
        )
        axis.set(
            title=f"run {profile.label.removeprefix('run-')} (N={profile.sample_count})",
            xlim=(2.0, 9.5),
        )
        axis.title.set_fontsize(14)
        axis.tick_params(labelsize=11)
        axis.grid(color="0.88", linewidth=0.8)
        axis.set_facecolor("#FAFAFA")
        if index == 0:
            axis.legend(fontsize=9, loc="upper right", frameon=False)
    pmf_fig.supxlabel(r"defect–nearest solute heavy-atom distance, $r$ (Å)", fontsize=15)
    pmf_fig.supylabel(
        r"sampled radial PMF, $W(r)$ (kcal mol$^{-1}$)", fontsize=15
    )
    pmf_fig.suptitle(
        f"BV {system_label} defect-separation landscapes",
        fontsize=17,
        fontweight="bold",
    )
    pmf_fig.savefig(out / "radial_pmf.png", dpi=220)
    plt.close(pmf_fig)
    save_csv(
        out / "contact_ion_pair_minima.csv",
        ["run", "cip_minimum_distance_A", "cip_minimum_pmf_kcal_mol"],
        cip_rows,
    )
    return plot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, nargs="+", required=True)
    parser.add_argument("--tdiff", type=float, nargs="+", required=True)
    parser.add_argument("--acid-base-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-ps", type=float, default=1.75)
    parser.add_argument("--helicity-smooth-ps", type=float, default=0.2)
    parser.add_argument("--cache-prefix", default="C")
    parser.add_argument("--solute-atoms", type=int, default=78)
    parser.add_argument("--binding-nitrogen-id", type=int, default=31)
    parser.add_argument("--system-label", default="meta-hic")
    args = parser.parse_args()
    if len(args.runs) != len(args.tdiff):
        parser.error("Supply one tdiff value per run")

    with (args.acid_base_dir / "index.csv").open(encoding="utf-8") as handle:
        entries = {
            int(row["run"]): row for row in csv.DictReader(handle)
            if row["bench_tag"] == "N1T48C1"
        }
    data = [
        load_run(args.acid_base_dir, entries[run], 10, args.window_ps, tdiff)
        for run, tdiff in zip(args.runs, args.tdiff)
    ]
    marker_path = args.out_dir / "reaction_markers.csv"
    reaction_markers = {
        int(row["run"]): float(row["treact_raw_ps"])
        for row in read_rows(marker_path)
    }
    for d in data:
        d["treact"] = reaction_markers[d["run"]] + d["display_offset"]

    plot = render(
        data,
        args.out_dir,
        args.cache_dir,
        args.window_ps,
        args.helicity_smooth_ps,
        args.cache_prefix,
        args.solute_atoms,
        args.binding_nitrogen_id,
        args.system_label,
    )
    (args.out_dir / "README.md").write_text(
        f"# Combined BV {args.system_label} endpoint summary\n\n"
        f"Runs: {', '.join(map(str, args.runs))}. Aligned tdiff values: {args.tdiff}.\n\n"
        f"Each pKa is Delta F divided by 0.004576*T at the single exact "
        f"tdiff+{args.window_ps:g} ps timestamp; values are linearly interpolated from "
        "the stored Delta-F series. No time-window mean or error bar is used. The FES "
        "is the latest saved block at or before the same endpoint.\n"
        "Defect distance is the periodic minimum distance from the positive-defect oxygen "
        "to any of the 78 solute atoms. When a frame has no new assignment, the previous "
        "defect oxygen identity is carried forward and its distance is recomputed using "
        "the current trajectory frame; leading unassigned frames remain gaps. H-bond wire length is the "
        f"number of hydrogen bonds from N{args.binding_nitrogen_id} to the assigned defect: "
        "bridging-water count "
        "+ 1; negative or missing paths are plotted as gaps. The four strips correspond "
        "to runs 39, 57, 99, and 142. A separate 2x2 figure contains the per-run sampled radial PMFs "
        "for the defect-to-nearest-solute-heavy-atom coordinate. Histogram probabilities "
        "are divided by exact spherical-shell volumes; the Gaussian KDE uses bandwidth "
        "0.13 times the sample standard deviation and is divided by r^2 before "
        "W(r)=-kBT ln g(r) is shifted to zero. The smooth curve joins directly to the "
        "established r^-12 wall at 2.48 A (sigma=2.42 A; scale=1.55 kcal/mol), and the "
        "entire combined profile is drawn as one continuous run-colored line. "
        "The first local minimum of each smooth profile is marked as the contact ion-pair minimum "
        "and saved in contact_ion_pair_minima.csv. "
        "These profiles describe sampling under coordination-biased metadynamics and are "
        "not bias-reweighted equilibrium PMFs. All plotted numerical data are saved beside the figure.\n",
        encoding="utf-8",
    )
    print(plot)


if __name__ == "__main__":
    main()
