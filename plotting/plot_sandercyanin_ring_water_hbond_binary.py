#!/usr/bin/env python3
"""Plot binary cpptraj water hydrogen-bond traces for BLA NB and NC."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nb-data", type=Path, required=True)
    parser.add_argument("--nc-data", type=Path, required=True)
    parser.add_argument("--nb-protein-data", type=Path, required=True)
    parser.add_argument("--nc-protein-data", type=Path, required=True)
    parser.add_argument("--nb-protein-series", type=Path, required=True)
    parser.add_argument("--nc-protein-series", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-spacing-ps", type=float, default=1.0)
    parser.add_argument(
        "--state-data",
        type=Path,
        help="Optional cpptraj fixed-pH protonation-state table used to select one microstate",
    )
    parser.add_argument(
        "--state-column",
        default="BLA:170%19",
        help="State-table column to select when --state-data is provided",
    )
    parser.add_argument("--state-value", type=int, default=3)
    parser.add_argument(
        "--state-stride",
        type=int,
        default=5,
        help="CpH records per saved coordinate frame",
    )
    parser.add_argument(
        "--state-offset",
        type=int,
        default=5,
        help="One-based CpH record within each coordinate interval",
    )
    parser.add_argument("--state-label", default="CPP")
    parser.add_argument(
        "--style",
        type=Path,
        default=Path(__file__).resolve().parent / "lefteris.mplstyle",
    )
    return parser.parse_args()


def read_cpptraj_counts(path: Path, count_column: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#", usecols=(0, count_column), ndmin=2)
    return data[:, 0].astype(int), data[:, 1].astype(int)


def partner_residue(label: str) -> str:
    matches = re.findall(r"([A-Za-z0-9]+)_(\d+)@", label)
    partners = [(name, number) for name, number in matches if not (name == "BLA" and number == "170")]
    if not partners:
        return label
    name, number = partners[0]
    return f"{name}{number}"


def read_cpptraj_series(path: Path, expected_frames: np.ndarray) -> dict[str, np.ndarray]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    lines = path.read_text().splitlines()
    header = next((line for line in lines if line.startswith("#Frame")), None)
    if header is None:
        return {}
    labels = header.split()[1:]
    if not labels:
        return {}
    data = np.loadtxt(path, comments="#", ndmin=2)
    frames = data[:, 0].astype(int)
    if not np.array_equal(frames, expected_frames):
        raise ValueError(f"Series frames in {path} do not match count data")
    by_residue: dict[str, np.ndarray] = {}
    for column, label in enumerate(labels, start=1):
        residue = partner_residue(label)
        values = (data[:, column] > 0).astype(int)
        by_residue[residue] = np.maximum(by_residue.get(residue, np.zeros_like(values)), values)
    return by_residue


def read_state_selection(
    path: Path,
    column: str,
    value: int,
    stride: int,
    offset: int,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an aligned coordinate-frame mask and sampled microstate values."""
    if stride <= 0:
        raise ValueError("--state-stride must be positive")
    if not 1 <= offset <= stride:
        raise ValueError("--state-offset must be between 1 and --state-stride")
    lines = path.read_text().splitlines()
    header = next((line for line in lines if line.startswith("#Frame")), None)
    if header is None:
        raise ValueError(f"Missing #Frame header in {path}")
    labels = header.split()[1:]
    if column not in labels:
        raise ValueError(f"State column {column!r} is absent from {path}")
    data = np.loadtxt(path, comments="#", ndmin=2)
    state_records = data[:, labels.index(column) + 1].astype(int)
    sampled = state_records[offset - 1 :: stride]
    aligned = min(frame_count, len(sampled))
    selection = np.zeros(frame_count, dtype=bool)
    selection[:aligned] = sampled[:aligned] == value
    return selection, sampled[:aligned]


def main() -> None:
    args = parse_args()
    nb_frames, nb_counts = read_cpptraj_counts(args.nb_data, 2)
    nc_frames, nc_counts = read_cpptraj_counts(args.nc_data, 2)
    if not np.array_equal(nb_frames, nc_frames):
        raise ValueError("NB and NC cpptraj outputs contain different frames")
    nb_protein_frames, nb_protein_counts = read_cpptraj_counts(args.nb_protein_data, 1)
    nc_protein_frames, nc_protein_counts = read_cpptraj_counts(args.nc_protein_data, 1)
    if not np.array_equal(nb_frames, nb_protein_frames) or not np.array_equal(nb_frames, nc_protein_frames):
        raise ValueError("Protein and water cpptraj outputs contain different frames")
    nb_protein_series = read_cpptraj_series(args.nb_protein_series, nb_frames)
    nc_protein_series = read_cpptraj_series(args.nc_protein_series, nb_frames)

    state_values: np.ndarray | None = None
    if args.state_data is not None:
        selection, state_values = read_state_selection(
            args.state_data,
            args.state_column,
            args.state_value,
            args.state_stride,
            args.state_offset,
            len(nb_frames),
        )
        nb_frames = nb_frames[selection]
        nb_counts = nb_counts[selection]
        nc_counts = nc_counts[selection]
        nb_protein_counts = nb_protein_counts[selection]
        nc_protein_counts = nc_protein_counts[selection]
        nb_protein_series = {
            residue: values[selection] for residue, values in nb_protein_series.items()
        }
        nc_protein_series = {
            residue: values[selection] for residue, values in nc_protein_series.items()
        }
        if not len(nb_frames):
            raise ValueError(
                f"No coordinate frames match {args.state_column} == {args.state_value}"
            )

    time_ns = nb_frames * args.frame_spacing_ps / 1000.0
    nb_binary = (nb_counts > 0).astype(int)
    nc_binary = (nc_counts > 0).astype(int)
    nb_protein_binary = (nb_protein_counts > 0).astype(int)
    nc_protein_binary = (nc_protein_counts > 0).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "time_ps",
                "nb_hb_donor_to_water_binary",
                "nb_hb_donor_to_water_count",
                "water_donor_to_nc_acceptor_binary",
                "water_donor_to_nc_acceptor_count",
                "nb_hb_donor_to_protein_binary",
                "nb_hb_donor_to_protein_count",
                "nb_hb_protein_residues",
                "protein_donor_to_nc_acceptor_binary",
                "protein_donor_to_nc_acceptor_count",
                "nc_protein_residues",
                "microstate",
            ]
        )
        for index, frame in enumerate(nb_frames):
            nb_residues = ";".join(name for name, values in nb_protein_series.items() if values[index])
            nc_residues = ";".join(name for name, values in nc_protein_series.items() if values[index])
            writer.writerow(
                [
                    frame,
                    time_ns[index] * 1000.0,
                    nb_binary[index],
                    nb_counts[index],
                    nc_binary[index],
                    nc_counts[index],
                    nb_protein_binary[index],
                    nb_protein_counts[index],
                    nb_residues,
                    nc_protein_binary[index],
                    nc_protein_counts[index],
                    nc_residues,
                    args.state_label if args.state_data is not None else "all",
                ]
            )

    plt.style.use(args.style)
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.7), sharex=True, sharey=True, constrained_layout=True)
    water_panels = (
        (axes[0, 0], nb_binary, "NB–HB donor → water O", "#1677b8"),
        (axes[0, 1], nc_binary, "water H donor → NC acceptor", "#b2182b"),
    )
    for axis, binary, title, color in water_panels:
        occupancy = 100.0 * binary.mean()
        axis.scatter(time_ns, binary, s=10, color=color, alpha=0.72, linewidths=0, rasterized=True)
        axis.set_title(title)
        axis.set_xlabel("Time (ns)")
        axis.set_yticks([0, 1], labels=["absent", "present"])
        axis.set_ylim(-0.28, 1.28)
        axis.text(
            0.025,
            0.94,
            f"occupancy = {occupancy:.1f}%",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
    protein_panels = (
        (
            axes[1, 0],
            nb_protein_binary,
            nb_protein_series,
            "NB–HB donor → protein acceptor",
            "#2ca25f",
        ),
        (
            axes[1, 1],
            nc_protein_binary,
            nc_protein_series,
            "protein donor → NC acceptor",
            "#d95f0e",
        ),
    )
    residue_colors = ("#d95f0e", "#756bb1", "#31a354", "#e7298a", "#636363")
    for axis, binary, series, title, zero_color in protein_panels:
        occupancy = 100.0 * binary.mean()
        axis.scatter(
            time_ns[binary == 0],
            binary[binary == 0],
            s=10,
            color=zero_color,
            alpha=0.60,
            linewidths=0,
            rasterized=True,
        )
        for color, (residue, values) in zip(residue_colors, sorted(series.items()), strict=False):
            selected = values > 0
            residue_occupancy = 100.0 * selected.mean()
            axis.scatter(
                time_ns[selected],
                np.ones(np.count_nonzero(selected)),
                s=28,
                color=color,
                edgecolors="white",
                linewidths=0.35,
                label=f"{residue} ({residue_occupancy:.1f}%)",
                zorder=3,
            )
        axis.set_title(title)
        axis.set_xlabel("Time (ns)")
        axis.set_yticks([0, 1], labels=["absent", "present"])
        axis.set_ylim(-0.28, 1.28)
        axis.text(
            0.025,
            0.94,
            f"occupancy = {occupancy:.1f}%",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
        if series:
            axis.legend(loc="center right", frameon=False, title="Protein residue")

    axes[0, 0].set_ylabel("Hydrogen bond")
    axes[1, 0].set_ylabel("Hydrogen bond")
    condition = (
        f" ({args.state_label} frames only)" if args.state_data is not None else ""
    )
    figure.suptitle(
        f"Sandercyanin pH 12: ring-nitrogen hydrogen bonds{condition}", fontsize=14
    )
    figure.savefig(args.output, dpi=300)
    plt.close(figure)

    print(f"Wrote {args.output}")
    print(f"Wrote {csv_path}")
    if args.state_data is not None:
        print(
            f"Selected {len(nb_frames)} frames with "
            f"{args.state_column} == {args.state_value}"
        )
    print(f"NB–HB donor → water O occupancy: {100.0 * nb_binary.mean():.2f}%")
    print(f"water H donor → NC acceptor occupancy: {100.0 * nc_binary.mean():.2f}%")
    print(f"NB–HB donor → protein acceptor occupancy: {100.0 * nb_protein_binary.mean():.2f}%")
    print(f"protein donor → NC acceptor occupancy: {100.0 * nc_protein_binary.mean():.2f}%")


if __name__ == "__main__":
    main()
