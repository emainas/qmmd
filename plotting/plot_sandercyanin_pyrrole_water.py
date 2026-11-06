#!/usr/bin/env python3
"""Track the water accepting the Sandercyanin BLA NB-HB hydrogen bond.

For every trajectory frame, the closest water oxygen to HB is selected using
triclinic minimum-image distances.  The script reports the NB--O/HB--O
distances and the distances from NC to both hydrogens of that same water.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytraj as pt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parm7", type=Path, required=True)
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--style",
        type=Path,
        default=Path(__file__).resolve().parent / "lefteris.mplstyle",
    )
    parser.add_argument("--residue", type=int, default=170, help="One-based BLA residue ID")
    return parser.parse_args()


def cell_matrix(box: np.ndarray) -> np.ndarray:
    """Return a row-vector triclinic cell matrix from Amber box parameters."""
    a, b, c, alpha_deg, beta_deg, gamma_deg = map(float, box)
    alpha, beta, gamma = np.radians([alpha_deg, beta_deg, gamma_deg])
    sin_gamma = math.sin(gamma)
    vectors = np.array(
        [
            [a, 0.0, 0.0],
            [b * math.cos(gamma), b * sin_gamma, 0.0],
            [
                c * math.cos(beta),
                c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma,
                0.0,
            ],
        ],
        dtype=float,
    )
    vectors[2, 2] = math.sqrt(max(c * c - np.dot(vectors[2, :2], vectors[2, :2]), 0.0))
    return vectors


NEIGHBOR_SHIFTS = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)


def minimum_image(displacements: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Return shortest Cartesian images for vectors in a triclinic cell."""
    vectors = np.atleast_2d(np.asarray(displacements, dtype=float))
    fractional = vectors @ np.linalg.inv(cell)
    centered = fractional - np.rint(fractional)
    candidates = (centered[:, None, :] + NEIGHBOR_SHIFTS[None, :, :]) @ cell
    best = np.argmin(np.einsum("nij,nij->ni", candidates, candidates), axis=1)
    result = candidates[np.arange(len(vectors)), best]
    return result[0] if np.asarray(displacements).ndim == 1 else result


def single_index(topology: pt.Topology, mask: str) -> int:
    indices = topology.select(mask)
    if len(indices) != 1:
        raise ValueError(f"Expected one atom for {mask}, found {len(indices)}")
    return int(indices[0])


def angle_degrees(first: np.ndarray, vertex: np.ndarray, last: np.ndarray, cell: np.ndarray) -> float:
    left = minimum_image(first - vertex, cell)
    right = minimum_image(last - vertex, cell)
    cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))


def distance(first: np.ndarray, second: np.ndarray, cell: np.ndarray) -> float:
    return float(np.linalg.norm(minimum_image(second - first, cell)))


def analyze(parm7: Path, trajectory: Path, residue: int) -> list[dict[str, float | int | bool]]:
    topology = pt.load_topology(str(parm7))
    nb = single_index(topology, f":{residue}@NB")
    hb = single_index(topology, f":{residue}@HB")
    nc = single_index(topology, f":{residue}@NC")
    water_oxygens = np.asarray(topology.select(":WAT@O"), dtype=int)
    if not len(water_oxygens):
        raise ValueError("Topology contains no :WAT@O atoms")

    rows: list[dict[str, float | int | bool]] = []
    frames = pt.iterload(str(trajectory), str(parm7))
    for frame_number, frame in enumerate(frames, start=1):
        xyz = frame.xyz
        cell = cell_matrix(frame.box.values)
        hb_to_oxygen = minimum_image(xyz[water_oxygens] - xyz[hb], cell)
        nb_to_oxygen = minimum_image(xyz[water_oxygens] - xyz[nb], cell)
        hb_o_all = np.linalg.norm(hb_to_oxygen, axis=1)
        nb_o_all = np.linalg.norm(nb_to_oxygen, axis=1)
        hb_to_nb = minimum_image(xyz[nb] - xyz[hb], cell)
        angle_cosines = np.einsum("ij,j->i", hb_to_oxygen, hb_to_nb)
        angle_cosines /= hb_o_all * np.linalg.norm(hb_to_nb)
        angles = np.degrees(np.arccos(np.clip(angle_cosines, -1.0, 1.0)))
        candidates = np.flatnonzero((nb_o_all <= 3.6) & (angles >= 135.0))
        if len(candidates):
            selected_offset = int(candidates[np.argmin(hb_o_all[candidates])])
        else:
            selected_offset = int(np.argmin(hb_o_all))
        oxygen = int(water_oxygens[selected_offset])
        # Amber water atoms are ordered O, H1, H2; verify the topology rather
        # than silently assuming this for an arbitrary solvent residue.
        h1, h2 = oxygen + 1, oxygen + 2
        names = (topology.atom(oxygen).name, topology.atom(h1).name, topology.atom(h2).name)
        residues = (topology.atom(oxygen).resid, topology.atom(h1).resid, topology.atom(h2).resid)
        if names != ("O", "H1", "H2") or len(set(residues)) != 1:
            raise ValueError(f"Unexpected water ordering around atom {oxygen + 1}: {names}")

        nb_o = distance(xyz[nb], xyz[oxygen], cell)
        hb_o = distance(xyz[hb], xyz[oxygen], cell)
        donor_angle = angle_degrees(xyz[nb], xyz[hb], xyz[oxygen], cell)
        nc_h1 = distance(xyz[nc], xyz[h1], cell)
        nc_h2 = distance(xyz[nc], xyz[h2], cell)
        rows.append(
            {
                "frame": frame_number,
                "time_ps": float(frame.time),
                "water_residue": int(topology.atom(oxygen).resid) + 1,
                "water_oxygen_atom": oxygen + 1,
                "nb_o_A": nb_o,
                "hb_o_A": hb_o,
                "nb_hb_o_angle_deg": donor_angle,
                "nb_hb_o_hbond": nb_o <= 3.6 and donor_angle >= 135.0,
                "nc_h1_A": nc_h1,
                "nc_h2_A": nc_h2,
                "nc_nearest_h_A": min(nc_h1, nc_h2),
            }
        )
    return rows


def write_csv(rows: list[dict[str, float | int | bool]], destination: Path) -> None:
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | int | bool]], destination: Path, style: Path) -> None:
    plt.style.use(style)
    time_ns = np.asarray([float(row["time_ps"]) for row in rows]) / 1000.0
    nb_o = np.asarray([float(row["nb_o_A"]) for row in rows])
    hb_o = np.asarray([float(row["hb_o_A"]) for row in rows])
    nc_h1 = np.asarray([float(row["nc_h1_A"]) for row in rows])
    nc_h2 = np.asarray([float(row["nc_h2_A"]) for row in rows])
    nc_min = np.minimum(nc_h1, nc_h2)
    bonded = np.asarray([bool(row["nb_hb_o_hbond"]) for row in rows])
    identities = np.asarray([int(row["water_residue"]) for row in rows])
    switches = int(np.count_nonzero(identities[1:] != identities[:-1]))

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharex=True, constrained_layout=True)
    left, right = axes

    left.plot(time_ns, nb_o, color="#34495e", lw=0.85, alpha=0.78, label=r"$d$(NB, O$_w$)")
    left.plot(time_ns, hb_o, color="#1677b8", lw=1.05, label=r"$d$(HB, O$_w$)")
    left.axhline(3.6, color="#34495e", ls="--", lw=0.9, alpha=0.7, label="NB···O cutoff")
    left.set_title("NB–HB donates to the selected water")
    left.set_xlabel("Time (ns)")
    left.set_ylabel("Distance (Å)")
    left.legend(loc="upper right", frameon=False)
    left.text(
        0.025,
        0.965,
        f"H-bond geometry: {100.0 * bonded.mean():.1f}%\n"
        f"{len(set(identities))} water IDs; {switches} exchanges",
        transform=left.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    right.plot(time_ns, nc_h1, color="#ef8a62", lw=0.70, alpha=0.58, label=r"$d$(NC, H1$_w$)")
    right.plot(time_ns, nc_h2, color="#b2182b", lw=0.70, alpha=0.58, label=r"$d$(NC, H2$_w$)")
    right.plot(time_ns, nc_min, color="#542788", lw=1.10, label="nearest water H")
    right.set_title("NC to the same water hydrogens")
    right.set_xlabel("Time (ns)")
    right.set_ylabel("Distance (Å)")
    right.legend(loc="lower left", frameon=False)
    right.text(
        0.975,
        0.965,
        f"nearest H: {nc_min.mean():.2f} Å mean\n{nc_min.min():.2f} Å minimum",
        transform=right.transAxes,
        va="top",
        ha="right",
        fontsize=9,
    )

    figure.suptitle("Sandercyanin pH 12 ensemble: water contact geometry", fontsize=14)
    figure.savefig(destination, dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = analyze(args.parm7.resolve(), args.traj.resolve(), args.residue)
    csv_path = args.output.with_suffix(".csv")
    write_csv(rows, csv_path)
    plot(rows, args.output, args.style.resolve())

    identities = np.asarray([int(row["water_residue"]) for row in rows])
    unique, counts = np.unique(identities, return_counts=True)
    order = np.argsort(counts)[::-1]
    bonded = np.asarray([bool(row["nb_hb_o_hbond"]) for row in rows])
    nearest_nc = np.asarray([float(row["nc_nearest_h_A"]) for row in rows])
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_path}")
    print(f"NB-HB···Ow hydrogen-bond occupancy: {100.0 * bonded.mean():.2f}%")
    print(f"Distinct closest-water residues: {len(unique)}")
    print("Most frequent water residues: " + ", ".join(f"{unique[i]} ({counts[i]} frames)" for i in order[:8]))
    print(f"NC-to-nearest-water-H: mean={nearest_nc.mean():.3f} Å, median={np.median(nearest_nc):.3f} Å")


if __name__ == "__main__":
    main()
