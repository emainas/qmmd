"""Plot BLA TPS CpHMD state fractions in the legacy histidine style."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(
    "systems/BLA/solv_10.0/cphmd/titr/post/report/"
    "tps-titration-tautomers.csv"
)
DEFAULT_OUTPUT = Path("legacy_scripts/tps-fractions-cphmd.png")

SERIES = (
    ("PPP", "ppp_fraction", "black"),
    ("APP+BPP+CPP+DPP", "deprotonated_fraction", "#ff100b"),
    ("APP", "app_fraction_total", "#1c0df5"),
    ("BPP", "bpp_fraction_total", "#008000"),
    ("CPP", "cpp_fraction_total", "#ff8c00"),
    ("DPP", "dpp_fraction_total", "#8e44ad"),
)


def read_fractions(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No TPS fraction rows found in {path}")

    required = {"pH", *(column for _label, column, _color in SERIES)}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing TPS fraction columns: {sorted(missing)}")

    data = {
        column: np.asarray([float(row[column]) for row in rows], dtype=float)
        for column in required
    }
    order = np.argsort(data["pH"])
    data = {column: values[order] for column, values in data.items()}

    tautomer_sum = sum(data[column] for _label, column, _color in SERIES[2:])
    if not np.allclose(tautomer_sum, data["deprotonated_fraction"], atol=5e-7):
        raise ValueError("TPS tautomer fractions do not sum to the deprotonated fraction")
    if not np.allclose(
        data["ppp_fraction"] + data["deprotonated_fraction"], 1.0, atol=5e-7
    ):
        raise ValueError("TPS protonated and deprotonated fractions do not sum to one")
    return data


def write_plot_data(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["pH", *(column for _label, column, _color in SERIES)]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for index in range(len(data["pH"])):
            writer.writerow([f"{data[column][index]:.6f}" for column in columns])


def plot_fractions(data: dict[str, np.ndarray], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 18,
            "axes.labelsize": 24,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 16,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
        }
    )
    figure, axis = plt.subplots(figsize=(8.12, 6.24), dpi=100)
    ph = data["pH"]
    for label, column, color in SERIES:
        axis.plot(
            ph,
            data[column],
            color=color,
            marker="o",
            markersize=11,
            linewidth=3.5,
            label=label,
        )

    axis.set_xlabel("pH")
    axis.set_ylabel("Fraction")
    axis.set_xlim(float(ph.min()), float(ph.max()))
    axis.set_ylim(-0.05, 1.05)
    axis.set_xticks(ph)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.legend(loc="center right", frameon=True, handlelength=2.1)
    figure.tight_layout()
    figure.savefig(output, dpi=100, facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = read_fractions(args.input)
    plot_fractions(data, args.output)
    data_path = args.output.with_suffix(".csv")
    write_plot_data(data_path, data)
    print(f"OK: wrote {args.output}")
    print(f"OK: wrote {data_path}")


if __name__ == "__main__":
    main()
