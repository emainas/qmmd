#!/usr/bin/env python3
"""Create live four-BLA titration and aggregation diagnostics.

The report is intentionally read-only and tolerant of an active RECpHMD run:
truncated terminal CPOUT records are discarded, and the newest trajectory
frame is skipped in case Amber is still flushing it.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import parmed as pmd
from scipy.io import netcdf_file


STATE_NAMES = ("PPP", "APP", "BPP", "CPP", "DPP")
STATE_COLORS = ("#202020", "#D95F02", "#7570B3", "#159D82", "#D45A9E")
STATE_MARKERS = ("o", "^", "D", "v", "P")
SITE_RESIDUES = (1, 4, 7, 10)
PAIR_LABELS = ("1–2", "1–3", "1–4", "2–3", "2–4", "3–4")
RESIDUE_RE = re.compile(
    r"^Residue\s+(?P<residue>\d+)\s+State:\s+(?P<state>\d+)\s+"
    r"pH:\s+(?P<ph>[-+0-9.eE]+)\s*$"
)


@dataclass(frozen=True)
class TitrationFit:
    pka: float
    hill: float
    sse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory", default="replica-01.nc")
    parser.add_argument(
        "--trajectory-stride",
        type=int,
        default=50,
        help="Analyze every Nth completed coordinate frame",
    )
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=4.5,
        help="Intermolecular heavy-atom contact cutoff in Å",
    )
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument(
        "--style",
        type=Path,
        default=Path(__file__).resolve().parent / "lefteris.mplstyle",
    )
    return parser.parse_args()


def read_ph_ladder(path: Path) -> tuple[float, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = tuple(float(row["pH"]) for row in rows)
    if not values or values != tuple(sorted(set(values))):
        raise ValueError(f"Invalid pH ladder in {path}")
    return values


def parse_cpout(path: Path) -> tuple[list[tuple[float, tuple[int, ...]]], int]:
    records: list[tuple[float, tuple[int, ...]]] = []
    current: dict[int, tuple[float, int]] = {}
    discarded = 0
    for line in path.read_text(errors="replace").splitlines():
        match = RESIDUE_RE.match(line)
        if match is None:
            continue
        residue = int(match.group("residue"))
        if residue == 0 and current:
            if set(current) == set(range(4)):
                ph_values = {value[0] for value in current.values()}
                if len(ph_values) != 1:
                    raise ValueError(f"Mixed pH values in one record in {path}")
                records.append(
                    (ph_values.pop(), tuple(current[index][1] for index in range(4)))
                )
            else:
                discarded += 1
            current = {}
        current[residue] = (float(match.group("ph")), int(match.group("state")))
    if current:
        if set(current) == set(range(4)):
            ph_values = {value[0] for value in current.values()}
            if len(ph_values) == 1:
                records.append(
                    (ph_values.pop(), tuple(current[index][1] for index in range(4)))
                )
            else:
                discarded += 1
        else:
            discarded += 1
    return records, discarded


def block_sem(values: np.ndarray, requested_blocks: int) -> float:
    if values.size < 2:
        return math.nan
    block_count = min(requested_blocks, values.size)
    blocks = np.array_split(values, block_count)
    block_means = np.asarray([block.mean() for block in blocks])
    return float(block_means.std(ddof=1) / math.sqrt(block_count))


def fit_titration(ph: np.ndarray, protonated: np.ndarray) -> TitrationFit:
    pka_low, pka_high = float(ph.min()) - 2.0, float(ph.max()) + 2.0
    hill_low, hill_high = 0.15, 2.0
    best_pka = 0.5 * (pka_low + pka_high)
    best_hill = 1.0
    best_sse = math.inf
    for _ in range(5):
        pka_grid = np.linspace(pka_low, pka_high, 241)
        hill_grid = np.linspace(hill_low, hill_high, 241)
        predicted = 1.0 / (
            1.0
            + np.power(
                10.0,
                hill_grid[:, None, None]
                * (ph[None, None, :] - pka_grid[None, :, None]),
            )
        )
        sse = np.sum((predicted - protonated[None, None, :]) ** 2, axis=2)
        hill_index, pka_index = np.unravel_index(int(np.argmin(sse)), sse.shape)
        best_pka = float(pka_grid[pka_index])
        best_hill = float(hill_grid[hill_index])
        best_sse = float(sse[hill_index, pka_index])
        pka_step = float(pka_grid[1] - pka_grid[0])
        hill_step = float(hill_grid[1] - hill_grid[0])
        pka_low, pka_high = best_pka - 3.0 * pka_step, best_pka + 3.0 * pka_step
        hill_low = max(0.01, best_hill - 3.0 * hill_step)
        hill_high = best_hill + 3.0 * hill_step
    return TitrationFit(best_pka, best_hill, best_sse)


def titration_report(
    run_dir: Path,
    output_dir: Path,
    ph_values: tuple[float, ...],
    blocks: int,
    style: Path,
) -> list[TitrationFit]:
    cpouts = sorted(run_dir.glob("replica-*.cpout"))
    if len(cpouts) != len(ph_values):
        raise ValueError(f"Expected {len(ph_values)} CPOUT files; found {len(cpouts)}")
    samples: list[tuple[float, tuple[int, ...]]] = []
    discarded = 0
    for path in cpouts:
        parsed, partial = parse_cpout(path)
        samples.extend(parsed)
        discarded += partial
    if not samples:
        raise ValueError("No complete aggregate CpH records are available")

    rows: list[dict[str, object]] = []
    fits: list[TitrationFit] = []
    plt.style.use(style)
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 9.2), sharex=True, sharey=True)
    for site_index, (residue_id, axis) in enumerate(zip(SITE_RESIDUES, axes.ravel())):
        ph_array = np.asarray(ph_values)
        site_fractions: dict[int, np.ndarray] = {}
        site_sems: dict[int, np.ndarray] = {}
        record_counts: list[int] = []
        for state_index in range(5):
            means: list[float] = []
            sems: list[float] = []
            for ph in ph_values:
                states = np.asarray(
                    [state[site_index] for sample_ph, state in samples if sample_ph == ph],
                    dtype=int,
                )
                if state_index == 0:
                    record_counts.append(int(states.size))
                indicator = (states == state_index).astype(float)
                means.append(float(indicator.mean()))
                sems.append(block_sem(indicator, blocks))
                rows.append(
                    {
                        "BLA": site_index + 1,
                        "topology_TPS_residue": residue_id,
                        "pH": ph,
                        "records": int(states.size),
                        "state": STATE_NAMES[state_index],
                        "fraction": means[-1],
                        "block_sem": sems[-1],
                    }
                )
            site_fractions[state_index] = np.asarray(means)
            site_sems[state_index] = np.asarray(sems)
        fit = fit_titration(ph_array, site_fractions[0])
        fits.append(fit)
        for state_index, state_name in enumerate(STATE_NAMES):
            axis.errorbar(
                ph_array,
                site_fractions[state_index],
                yerr=site_sems[state_index],
                color=STATE_COLORS[state_index],
                marker=STATE_MARKERS[state_index],
                ms=4.0,
                lw=1.6 if state_index else 2.2,
                capsize=2.0,
                label=state_name,
            )
        dense_ph = np.linspace(ph_array.min(), ph_array.max(), 500)
        fit_curve = 1.0 / (1.0 + 10.0 ** (fit.hill * (dense_ph - fit.pka)))
        axis.plot(dense_ph, fit_curve, color="#444444", ls="--", lw=1.1)
        axis.axvline(fit.pka, color="#B13A3A", ls=(0, (4, 3)), lw=1.1)
        axis.text(
            0.96,
            0.96,
            f"apparent p$K_a$ = {fit.pka:.2f}\nHill $n$ = {fit.hill:.2f}",
            transform=axis.transAxes,
            fontsize=10,
            color="#8E2828",
            ha="right",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
        )
        axis.set_title(f"BLA {site_index + 1}  ·  TPS residue {residue_id}")
        axis.set_ylim(-0.035, 1.035)
        axis.grid(alpha=0.16)
    for axis in axes[-1, :]:
        axis.set_xlabel("pH")
    for axis in axes[:, 0]:
        axis.set_ylabel("Molar fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=False,
    )
    figure.suptitle(
        "Live four-BLA aggregate ring titration",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.955,
        f"{len(samples):,} complete CpH records · {discarded} truncated terminal record(s) discarded",
        ha="center",
        fontsize=10.5,
        color="0.32",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    figure.savefig(output_dir / "aggregate-live-titration.png", dpi=300)
    plt.close(figure)

    with (output_dir / "aggregate-live-titration.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "aggregate-live-titration-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["BLA", "topology_TPS_residue", "apparent_pKa", "Hill_n", "fit_SSE"]
        )
        for index, (residue_id, fit) in enumerate(zip(SITE_RESIDUES, fits), start=1):
            writer.writerow([index, residue_id, fit.pka, fit.hill, fit.sse])
    return fits


def heavy_atom_groups(topology: pmd.Structure) -> tuple[np.ndarray, ...]:
    expected = ("TPS", "PRX", "PRX") * 4
    observed = tuple(residue.name for residue in topology.residues[:12])
    if observed != expected:
        raise ValueError(f"Unexpected first 12 residues: {observed}; expected {expected}")
    groups: list[np.ndarray] = []
    for molecule in range(4):
        residues = topology.residues[3 * molecule : 3 * molecule + 3]
        indices = [
            atom.idx
            for residue in residues
            for atom in residue.atoms
            if atom.atomic_number > 1
        ]
        groups.append(np.asarray(indices, dtype=int))
    return tuple(groups)


def largest_component_size(contact_matrix: np.ndarray) -> int:
    unseen = set(range(len(contact_matrix)))
    largest = 0
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = {
                int(index)
                for index in np.flatnonzero(contact_matrix[node])
                if int(index) in unseen
            }
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
        largest = max(largest, size)
    return largest


def aggregation_report(
    run_dir: Path,
    output_dir: Path,
    trajectory_name: str,
    stride: int,
    cutoff: float,
    style: Path,
) -> dict[str, float]:
    if stride <= 0:
        raise ValueError("--trajectory-stride must be positive")
    if cutoff <= 0.0:
        raise ValueError("--contact-cutoff must be positive")
    topology_path = run_dir / "agg4_mbondi2.parm7"
    trajectory_path = run_dir / trajectory_name
    topology = pmd.load_file(str(topology_path))
    groups = heavy_atom_groups(topology)
    pair_indices = tuple((i, j) for i in range(4) for j in range(i + 1, 4))

    records: list[dict[str, object]] = []
    # Keep the growing trajectory memory-mapped rather than copying it. SciPy
    # emits this harmless close warning while variable views still exist.
    warnings.filterwarnings(
        "ignore",
        message="Cannot close a netcdf_file opened with mmap=True",
        category=RuntimeWarning,
    )
    with netcdf_file(trajectory_path, mode="r", mmap=True) as trajectory:
        coordinates = trajectory.variables["coordinates"].data
        lengths = trajectory.variables["cell_lengths"].data
        angles = trajectory.variables["cell_angles"].data
        times = trajectory.variables["time"].data
        available_frames = int(coordinates.shape[0])
        safe_frames = max(available_frames - 1, 1)
        frame_indices = np.arange(0, safe_frames, stride, dtype=int)
        if frame_indices[-1] != safe_frames - 1:
            frame_indices = np.append(frame_indices, safe_frames - 1)
        for frame_index in frame_indices:
            frame = np.asarray(coordinates[frame_index], dtype=float)
            box = np.asarray(lengths[frame_index], dtype=float)
            angle = np.asarray(angles[frame_index], dtype=float)
            if not np.allclose(angle, 90.0, atol=1.0e-3):
                raise ValueError(
                    "Aggregate live contact analysis currently requires an orthorhombic box"
                )
            minima: list[float] = []
            contacts: list[int] = []
            matrix = np.zeros((4, 4), dtype=bool)
            for first, second in pair_indices:
                delta = (
                    frame[groups[first]][:, None, :]
                    - frame[groups[second]][None, :, :]
                )
                delta -= box * np.rint(delta / box)
                minimum = float(np.sqrt(np.min(np.einsum("ijk,ijk->ij", delta, delta))))
                contact = int(minimum < cutoff)
                minima.append(minimum)
                contacts.append(contact)
                matrix[first, second] = matrix[second, first] = bool(contact)
            row: dict[str, object] = {
                "frame": int(frame_index + 1),
                "time_ps": float(times[frame_index]),
                "contact_cutoff_A": cutoff,
                "contact_pair_count": int(sum(contacts)),
                "largest_cluster_BLA_count": largest_component_size(matrix),
            }
            for label, minimum, contact in zip(PAIR_LABELS, minima, contacts):
                key = label.replace("–", "-")
                row[f"BLA_{key}_minimum_heavy_distance_A"] = minimum
                row[f"BLA_{key}_contact"] = contact
            records.append(row)

    with (output_dir / "aggregate-live-contacts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    times_ns = np.asarray([float(row["time_ps"]) for row in records]) / 1000.0
    distance_matrix = np.asarray(
        [
            [
                float(row[f"BLA_{label.replace('–', '-')}_minimum_heavy_distance_A"])
                for label in PAIR_LABELS
            ]
            for row in records
        ]
    )
    contact_matrix = distance_matrix < cutoff
    pair_probabilities = contact_matrix.mean(axis=0)
    contact_counts = contact_matrix.sum(axis=1)
    largest_clusters = np.asarray(
        [int(row["largest_cluster_BLA_count"]) for row in records]
    )

    plt.style.use(style)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for pair_index, (label, color) in enumerate(zip(PAIR_LABELS, colors)):
        axes[0].plot(
            times_ns,
            distance_matrix[:, pair_index],
            marker="o",
            ms=2.8,
            lw=1.0,
            color=color,
            alpha=0.82,
            label=f"BLA {label}",
        )
    axes[0].axhline(cutoff, color="#B13A3A", ls="--", lw=1.2, label=f"{cutoff:g} Å contact")
    axes[0].set_xlabel("Walker time (ns)")
    axes[0].set_ylabel("Minimum heavy-atom distance (Å)")
    axes[0].set_title("A  Pairwise approach")
    axes[0].legend(fontsize=7.5, ncol=2, frameon=False)
    axes[0].grid(alpha=0.15)

    axes[1].step(times_ns, contact_counts, where="mid", color="#2B73B6", lw=1.5)
    axes[1].scatter(times_ns, contact_counts, color="#2B73B6", s=10)
    axes[1].set_xlabel("Walker time (ns)")
    axes[1].set_ylabel("Contacting BLA pairs", color="#2B73B6")
    axes[1].tick_params(axis="y", labelcolor="#2B73B6")
    axes[1].set_ylim(-0.25, 6.25)
    cluster_axis = axes[1].twinx()
    cluster_axis.step(times_ns, largest_clusters, where="mid", color="#D95F02", lw=1.7)
    cluster_axis.set_ylabel("Largest connected cluster", color="#D95F02")
    cluster_axis.tick_params(axis="y", labelcolor="#D95F02")
    cluster_axis.set_ylim(0.75, 4.25)
    cluster_axis.set_yticks([1, 2, 3, 4])
    axes[1].set_title("B  Aggregate size")
    axes[1].grid(alpha=0.15)

    image = axes[2].imshow(
        pair_probabilities[:, None],
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="YlOrRd",
    )
    axes[2].set_yticks(range(6), [f"BLA {label}" for label in PAIR_LABELS])
    axes[2].set_xticks([0], ["Contact fraction"])
    for index, probability in enumerate(pair_probabilities):
        axes[2].text(
            0,
            index,
            f"{probability:.0%}",
            ha="center",
            va="center",
            color="white" if probability > 0.55 else "black",
            fontweight="bold",
        )
    figure.colorbar(image, ax=axes[2], fraction=0.10, pad=0.08)
    axes[2].set_title("C  Contact persistence")
    figure.suptitle(
        "Live four-BLA aggregation diagnostic",
        fontsize=18,
        fontweight="bold",
    )
    figure.savefig(output_dir / "aggregate-live-aggregation.png", dpi=300)
    plt.close(figure)

    summary = {
        "trajectory_frames_available_at_open": float(available_frames),
        "sampled_frames": float(len(records)),
        "trajectory_stride_frames": float(stride),
        "contact_cutoff_A": cutoff,
        "any_contact_fraction": float(np.mean(contact_counts > 0)),
        "all_four_connected_fraction": float(np.mean(largest_clusters == 4)),
        "mean_largest_cluster_size": float(largest_clusters.mean()),
        **{
            f"BLA_{label.replace('–', '-')}_contact_fraction": float(probability)
            for label, probability in zip(PAIR_LABELS, pair_probabilities)
        },
    }
    with (output_dir / "aggregate-live-aggregation-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "value"])
        writer.writerows(summary.items())
    return summary


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    style = args.style.resolve()
    ph_values = read_ph_ladder(run_dir / "ph-ladder.csv")
    fits = titration_report(run_dir, output_dir, ph_values, args.blocks, style)
    aggregation = aggregation_report(
        run_dir,
        output_dir,
        args.trajectory,
        args.trajectory_stride,
        args.contact_cutoff,
        style,
    )
    print(f"OK: wrote live four-BLA report to {output_dir}")
    print("Apparent pKa: " + ", ".join(f"BLA {i}={fit.pka:.2f}" for i, fit in enumerate(fits, 1)))
    print(
        "Aggregation: any contact in "
        f"{aggregation['any_contact_fraction']:.1%} of sampled frames; "
        f"all four connected in {aggregation['all_four_connected_fraction']:.1%}"
    )


if __name__ == "__main__":
    main()
