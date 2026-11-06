"""Compare TPS GB2-rescored REFEP free energies with converged DGref values."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from qmmd.cphmd_titr import read_converged_dgref


STATES = ("APP", "BPP", "CPP", "DPP")
METHODS = ("BAR", "TI charge derivative", "MBAR")


def _paths(root: Path, state: str) -> tuple[Path, Path]:
    lower = state.lower()
    dgref_branch = (
        root / "systems/TPS/solv_10.0/cphmd/dgref-mbondi2/cpp"
        if state == "CPP"
        else root / f"systems/TPS/solv_10.0/cphmd/dgref-mbondi2-restart/{lower}"
    )
    refep_report = (
        root
        / f"systems/TPS/solv_10.0/refep/{lower}/prod-5ns/report-igb2/summary.yaml"
    )
    return dgref_branch / "dgref.log", refep_report


def collect(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state in STATES:
        dgref_log, summary_path = _paths(root, state)
        dgref = read_converged_dgref(dgref_log)
        summary = yaml.safe_load(summary_path.read_text())
        if summary.get("free_energy_definition") != f"G(protonated-ppp) - G(deprotonated-{state.lower()})":
            raise ValueError(f"Unexpected sign convention in {summary_path}")
        energies = summary["free_energies_kcal_mol"]
        row: dict[str, object] = {
            "state": state,
            "definition": summary["free_energy_definition"],
            "dgref_kcal_mol": dgref,
            "dgref_log": str(dgref_log.relative_to(root)),
            "refep_summary": str(summary_path.relative_to(root)),
        }
        for method in METHODS:
            slug = method.lower().replace(" ", "_")
            estimate = float(energies[method]["estimate"])
            uncertainty = float(energies[method]["bootstrap_sd"])
            row[f"{slug}_kcal_mol"] = estimate
            row[f"{slug}_bootstrap_sd_kcal_mol"] = uncertainty
            row[f"{slug}_minus_dgref_kcal_mol"] = estimate - dgref
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, rows: list[dict[str, object]], style: Path) -> None:
    if style.is_file():
        plt.style.use(style)
    x = np.arange(len(rows), dtype=float)
    dgref = np.asarray([float(row["dgref_kcal_mol"]) for row in rows])
    colors = ("#1F77B4", "#E67E22", "#2E8B57")
    offsets = (-0.20, 0.0, 0.20)
    fig, (absolute_ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(7.4, 7.0),
        sharex=True,
        height_ratios=(1.55, 1.0),
        constrained_layout=True,
    )
    absolute_ax.scatter(x, dgref, color="black", marker="D", s=52, label="CpHMD DGref", zorder=4)
    for method, color, offset in zip(METHODS, colors, offsets, strict=True):
        slug = method.lower().replace(" ", "_")
        values = np.asarray([float(row[f"{slug}_kcal_mol"]) for row in rows])
        errors = np.asarray(
            [float(row[f"{slug}_bootstrap_sd_kcal_mol"]) for row in rows]
        )
        absolute_ax.errorbar(
            x + offset,
            values,
            yerr=errors,
            fmt="o",
            color=color,
            capsize=3,
            label=f"REFEP {method}",
        )
        residual_ax.errorbar(
            x + offset,
            values - dgref,
            yerr=errors,
            fmt="o",
            color=color,
            capsize=3,
            label=method,
        )
    absolute_ax.set_ylabel(r"$G_{\mathrm{PPP}}-G_{\mathrm{tautomer}}$ (kcal mol$^{-1}$)")
    absolute_ax.set_title("TPS: CpHMD DGref versus GB2-rescored REFEP")
    absolute_ax.legend(ncol=2, fontsize=8)
    residual_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    residual_ax.set_ylabel(r"REFEP $-$ DGref (kcal mol$^{-1}$)")
    residual_ax.set_xticks(x, [str(row["state"]) for row in rows])
    residual_ax.set_xlabel("Deprotonated tautomer")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("systems/TPS/solv_10.0/refep/dgref-comparison"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = collect(root)
    write_csv(output / "dgref-vs-refep.csv", rows)
    plot(output / "dgref-vs-refep.png", rows, root / "plotting/prl.mplstyle")
    (output / "comparison.yaml").write_text(
        yaml.safe_dump(
            {
                "definition": "G(protonated-ppp) - G(deprotonated-tautomer)",
                "refep_scope": "GB2 single-point rescoring of explicit-solvent H-REMD ensembles",
                "states": [row["state"] for row in rows],
                "csv": "dgref-vs-refep.csv",
                "figure": "dgref-vs-refep.png",
            },
            sort_keys=False,
        )
    )
    print(f"OK: wrote TPS DGref/REFEP comparison to {output}")


if __name__ == "__main__":
    main()
