#!/usr/bin/env python3
"""Render a compact paper diagnostic from an acid_base_BV report directory."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def load_named_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    return np.atleast_1d(data)


def finite_before(values: np.ndarray, times: np.ndarray, end: float) -> int:
    indices = np.flatnonzero(np.isfinite(values) & np.isfinite(times) & (times <= end + 1.0e-9))
    if not indices.size:
        raise ValueError(f"No finite value at or before {end:g} ps")
    return int(indices[-1])


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.015,
        0.975,
        f"({label})",
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=20,
    )


def mark_window(axis: plt.Axes, start: float, end: float) -> None:
    axis.axvline(start, color="black", linewidth=1.3, zorder=8)
    axis.axvline(end, color="#C43C39", linewidth=1.3, zorder=8)


def save_main_data(
    path: Path,
    data: np.ndarray,
    offset: float,
    raw_end: float,
) -> None:
    fields = (
        "time_ps",
        "coordination_s",
        "defect_oxygen_id",
        "N31_Odefect_distance_A",
        "closest_solute_heavy_distance_A",
        "hbond_wire_connected",
        "hbond_bridging_water_count",
    )
    mask = np.asarray(data["time_ps"] <= raw_end + 1.0e-9)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("aligned_time_ps",) + fields[1:])
        for row in data[mask]:
            writer.writerow(
                [f"{float(row['time_ps']) + offset:.8f}"]
                + [f"{float(row[name]):.10g}" for name in fields[1:]]
            )


def sample_window(
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
    count: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly sample a finite time series at equally spaced timestamps."""
    finite = np.isfinite(times) & np.isfinite(values)
    source_t = np.asarray(times[finite], dtype=float)
    source_v = np.asarray(values[finite], dtype=float)
    if source_t.size < 2 or start < source_t[0] or end > source_t[-1] + 1.0e-9:
        raise ValueError("Requested pKa sampling window is outside finite coverage")
    requested = np.linspace(start, end, count)
    return requested, np.interp(requested, source_t, source_v)


def save_oxygen_data(
    path: Path,
    data: np.ndarray,
    offset: float,
    raw_end: float,
) -> None:
    names = data.dtype.names
    if names is None:
        raise ValueError("Oxygen-charge CSV has no named columns")
    mask = np.asarray(data["time_ps"] <= raw_end + 1.0e-9)
    matrix = np.column_stack(
        [data["time_ps"][mask] + offset]
        + [np.asarray(data[name][mask], dtype=float) for name in names[1:]]
    )
    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header=",".join(("aligned_time_ps",) + names[1:]),
        comments="",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--raw-tdiff", required=True, type=float)
    parser.add_argument("--aligned-tdiff", required=True, type=float)
    parser.add_argument("--tstop", required=True, type=float)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--title", default="BV ring C, run 99")
    parser.add_argument("--style", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tstop <= 0.0:
        raise ValueError("--tstop must be positive")
    if args.style is not None:
        plt.style.use(args.style)

    source = args.report_dir.resolve()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    main_data = load_named_csv(source / "summary.csv")
    pka_data = load_named_csv(source / "summary_pka.csv")
    oxygen_data = load_named_csv(source / "summary_oxygen_charges.csv")
    fes = np.loadtxt(source / "summary_fes_snapshot.csv", delimiter=",", comments="#")

    offset = args.aligned_tdiff - args.raw_tdiff
    raw_end = args.raw_tdiff + args.tstop
    aligned_end = args.aligned_tdiff + args.tstop

    pka_index = finite_before(pka_data["pka"], pka_data["time_ps"], raw_end)
    endpoint_pka = float(pka_data["pka"][pka_index])
    endpoint_raw_time = float(pka_data["time_ps"][pka_index])
    endpoint_aligned_time = endpoint_raw_time + offset

    time = np.asarray(main_data["time_ps"], dtype=float) + offset
    displayed = np.isfinite(time) & (main_data["time_ps"] <= raw_end + 1.0e-9)
    pka_time = np.asarray(pka_data["time_ps"], dtype=float) + offset
    pka_displayed = np.isfinite(pka_time) & (pka_data["time_ps"] <= raw_end + 1.0e-9)
    oxygen_time = np.asarray(oxygen_data["time_ps"], dtype=float) + offset
    oxygen_displayed = np.isfinite(oxygen_time) & (oxygen_data["time_ps"] <= raw_end + 1.0e-9)
    pka_sample_times, pka_sample_values = sample_window(
        pka_time,
        np.asarray(pka_data["pka"], dtype=float),
        args.aligned_tdiff,
        aligned_end,
    )
    pka_mean = float(np.mean(pka_sample_values))
    pka_std = float(np.std(pka_sample_values))

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12.2, 11.2),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 1.0, 1.0)},
    )
    (
        ax_fes,
        ax_hbond,
        ax_distance,
        ax_coordination,
        ax_charges,
        ax_pka,
    ) = axes.flat

    fes_s = np.asarray(fes[:, 0], dtype=float)
    fes_energy = np.asarray(fes[:, 1], dtype=float)
    fes_energy -= np.nanmin(fes_energy)
    ax_fes.plot(fes_s, fes_energy, color="#5B3F8C", linewidth=2.0)
    ax_fes.set(xlabel=r"coordination, $s$", ylabel=r"$F(s)$ (kcal mol$^{-1}$)")
    ax_fes.set_title(
        rf"FES snapshot at $t={endpoint_aligned_time:.2f}$ ps"
    )
    ax_fes.set_xlim(0.0, 1.0)

    connected = np.asarray(main_data["hbond_wire_connected"], dtype=float)
    bridges = np.asarray(main_data["hbond_bridging_water_count"], dtype=float)
    wire_value = np.full(connected.shape, np.nan, dtype=float)
    wire_value[connected == 0.0] = -1.0
    wire_value[connected == 1.0] = bridges[connected == 1.0]
    finite_wire = displayed & np.isfinite(wire_value)
    window_wire = (
        finite_wire
        & (time >= args.aligned_tdiff - 1.0e-9)
        & (time <= aligned_end + 1.0e-9)
    )
    max_bridge = int(np.nanmax(wire_value[finite_wire]))
    wire_states = np.arange(-1, max_bridge + 1)
    wire_colors = ["#7F7F7F", "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]
    if wire_states.size > len(wire_colors):
        wire_colors.extend(
            plt.get_cmap("tab20")(np.linspace(0.05, 0.95, wire_states.size - len(wire_colors)))
        )
    wire_cmap = ListedColormap(wire_colors[: wire_states.size])
    wire_norm = BoundaryNorm(
        np.arange(-1.5, max_bridge + 1.5, 1.0), wire_cmap.N
    )
    wire_points = ax_hbond.scatter(
        time[finite_wire],
        wire_value[finite_wire],
        c=wire_value[finite_wire],
        cmap=wire_cmap,
        norm=wire_norm,
        s=18,
        edgecolors="none",
        alpha=0.85,
        rasterized=True,
    )
    wire_counts = Counter(wire_value[window_wire].astype(int))
    ax_hbond.set_yticks(
        wire_states,
        [f"{state} (N={wire_counts.get(int(state), 0)})" for state in wire_states],
    )
    ax_hbond.set_ylim(-1.4, max_bridge + 0.4)
    ax_hbond.set_ylabel("number of bridging waters")
    wire_colorbar = fig.colorbar(wire_points, ax=ax_hbond, ticks=wire_states, pad=0.02)
    wire_colorbar.ax.set_yticklabels([str(state) for state in wire_states])

    layer_edges = (2.0, 3.5, 5.5, 7.5, 9.5)
    layer_colors = ("#DCECF8", "#E1F1E3", "#FFF1D8", "#F3E4D7")
    for low, high, color in zip(layer_edges[:-1], layer_edges[1:], layer_colors, strict=True):
        ax_distance.axhspan(low, high, color=color, alpha=0.75, zorder=0)
    ax_distance.scatter(
        time[displayed],
        np.asarray(main_data["closest_solute_heavy_distance_A"], dtype=float)[displayed],
        color="#7651A8",
        s=12,
        edgecolors="none",
        alpha=0.85,
        rasterized=True,
        label="nearest solute heavy atom--defect",
    )
    ax_distance.set(
        ylabel=r"nearest-solute--defect distance ($\AA$)", ylim=(0.0, 9.5)
    )

    coordination = np.asarray(main_data["coordination_s"], dtype=float)
    before_diffusion = displayed & (time <= args.aligned_tdiff)
    after_diffusion = displayed & (time >= args.aligned_tdiff)
    x_min = float(np.nanmin(time[displayed]))
    ax_coordination.axvspan(
        x_min, args.aligned_tdiff, color="#D9EEF9", alpha=0.65, linewidth=0, zorder=0
    )
    ax_coordination.axvspan(
        args.aligned_tdiff, aligned_end, color="#F8DADA", alpha=0.65, linewidth=0, zorder=0
    )
    ax_coordination.plot(
        time[before_diffusion], coordination[before_diffusion], color="#356A8A", linewidth=1.8
    )
    ax_coordination.plot(
        time[after_diffusion], coordination[after_diffusion], color="#9B4A4A", linewidth=1.8
    )
    ax_coordination.set(ylabel=r"coordination, $s(t)$", ylim=(-0.03, 1.03))

    oxygen_names = oxygen_data.dtype.names
    if oxygen_names is None:
        raise ValueError("Oxygen-charge CSV has no named columns")
    oxygen_colors = plt.get_cmap("turbo")(
        np.linspace(0.02, 0.98, len(oxygen_names) - 1)
    )
    for name, color in zip(oxygen_names[1:], oxygen_colors, strict=True):
        values = np.asarray(oxygen_data[name], dtype=float)
        ax_charges.plot(
            oxygen_time[oxygen_displayed],
            values[oxygen_displayed],
            color=color,
            linewidth=0.70,
            alpha=0.55,
            rasterized=True,
        )
    defect_ids = np.asarray(main_data["defect_oxygen_id"], dtype=float)
    last_id: int | None = None
    for current_time, value in zip(time[displayed], defect_ids[displayed], strict=True):
        if not np.isfinite(value):
            continue
        current_id = int(round(float(value)))
        if last_id is not None and current_id != last_id:
            ax_charges.axvline(current_time, color="#7B61A8", linewidth=0.55, alpha=0.6)
        last_id = current_id
    ax_charges.set(ylabel="solvent-O Mulliken charge (e)")

    ax_pka.plot(
        pka_time[pka_displayed],
        np.asarray(pka_data["pka"], dtype=float)[pka_displayed],
        color="black",
        linewidth=1.45,
    )
    ax_pka.scatter(
        pka_sample_times,
        pka_sample_values,
        color="#D62728",
        edgecolor="black",
        linewidth=0.45,
        marker="*",
        s=62,
        zorder=10,
        label=r"p$K_a$ = 4.47",
    )
    ax_pka.set(ylabel=r"p$K_a(t)$", ylim=(-20.0, 30.0))
    ax_pka.legend(frameon=False, loc="best", fontsize=9)

    time_axes = (
        ax_hbond,
        ax_distance,
        ax_coordination,
        ax_charges,
        ax_pka,
    )
    for axis in time_axes:
        mark_window(axis, args.aligned_tdiff, aligned_end)
        axis.set_xlim(x_min, aligned_end)
    for axis in time_axes[:-2]:
        axis.tick_params(labelbottom=False)
    ax_charges.set_xlabel("aligned simulation time (ps)")
    ax_pka.set_xlabel("aligned simulation time (ps)")

    for axis in fig.axes:
        axis.grid(False)

    for axis, label in zip(axes.flat, "abcdef", strict=True):
        panel_label(axis, label)

    fig.suptitle(
        rf"{args.title}: analysis window {args.aligned_tdiff:.2f}--{aligned_end:.2f} ps",
        fontsize=15,
    )

    figure_path = output / "c99-paper-summary.png"
    fig.savefig(figure_path, dpi=300)
    fig.savefig(output / "c99-paper-summary.pdf")
    plt.close(fig)

    save_main_data(output / "c99-paper-summary-timeseries.csv", main_data, offset, raw_end)
    save_oxygen_data(output / "c99-paper-summary-oxygen-charges.csv", oxygen_data, offset, raw_end)
    np.savetxt(
        output / "c99-paper-summary-pka.csv",
        np.column_stack(
            [pka_time[pka_displayed], np.asarray(pka_data["pka"], dtype=float)[pka_displayed]]
        ),
        delimiter=",",
        header="aligned_time_ps,pka",
        comments="",
    )
    np.savetxt(
        output / "c99-paper-summary-pka-window-samples.csv",
        np.column_stack([pka_sample_times, pka_sample_values]),
        delimiter=",",
        header=(
            f"mean_pka={pka_mean:.10g},std_pka={pka_std:.10g}\n"
            "aligned_time_ps,pka"
        ),
        comments="# ",
    )
    np.savetxt(
        output / "c99-paper-summary-fes.csv",
        np.column_stack([fes_s, fes_energy]),
        delimiter=",",
        header=(
            f"requested_endpoint_ps={aligned_end:.8f};"
            f"latest_snapshot_ps={endpoint_aligned_time:.8f};"
            f"endpoint_pka={endpoint_pka:.10f}\ns,free_energy_kcal_mol"
        ),
        comments="# ",
    )
    with (output / "provenance.json").open("w") as handle:
        json.dump(
            {
                "source_report_dir": str(source),
                "raw_tdiff_ps": args.raw_tdiff,
                "aligned_tdiff_ps": args.aligned_tdiff,
                "clock_offset_ps": offset,
                "tstop_ps": args.tstop,
                "requested_endpoint_ps": aligned_end,
                "latest_pka_snapshot_ps": endpoint_aligned_time,
                "endpoint_pka": endpoint_pka,
                "window_mean_pka": pka_mean,
                "window_std_pka": pka_std,
                "panel_order": [
                    "FES",
                    "bridging-water count",
                    "defect distances",
                    "biased coordination",
                    "solvent-oxygen Mulliken charges",
                    "pKa",
                ],
                "excluded_panels": [
                    "wire conductivity",
                    "wire loopiness",
                    "BV-site/defect distances",
                ],
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    print(f"Wrote {figure_path}")
    print(
        f"Endpoint pKa: {endpoint_pka:.6f} from the latest saved snapshot "
        f"at {endpoint_aligned_time:.2f} ps (requested endpoint {aligned_end:.2f} ps)"
    )
    print(f"Window pKa: {pka_mean:.6f} +/- {pka_std:.6f} (10 equally spaced samples)")


if __name__ == "__main__":
    main()
