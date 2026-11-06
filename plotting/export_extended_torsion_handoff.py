#!/usr/bin/env python3
"""Export frozen extended torsional PMFs into the Elisa handoff tree."""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "elisa/umbrella-sampling/extended"
R_KCAL = 0.00198720425864083


@dataclass(frozen=True, slots=True)
class Export:
    package: str
    system: str
    source: Path
    config: Path
    stage: str
    note: str


EXPORTS = (
    Export("APP_rot", "APP", ROOT / "systems/APP/solv_4.0/us-single15/wham_extended", ROOT / "configs/APP/us/wham-dih-extended.yaml", "restrained equilibration", "24 windows; 40 ps per window."),
    Export("BPP_rot", "BPP", ROOT / "systems/BPP/solv_4.0/us-single15/wham_extended", ROOT / "configs/BPP/us/wham-dih-extended.yaml", "restrained equilibration", "Available samples; exact counts in window_summary.csv."),
    Export("CPP_rot", "CPP", ROOT / "systems/CPP/solv_4.0/us-single15/wham_prod_extended", ROOT / "configs/CPP/us/wham-dih-prod-extended.yaml", "production", "Complete production data; exact counts are in window_summary.csv."),
    Export("DPP_rot", "DPP", ROOT / "systems/DPP/solv_4.0/us-single15/wham_extended", ROOT / "configs/DPP/us/wham-dih-extended.yaml", "restrained equilibration", "24 windows; 40 ps per window."),
)


def integrated_values(x: np.ndarray, pmf: np.ndarray, temperature: float) -> dict[str, float]:
    weights = np.exp(-(pmf - float(np.min(pmf))) / (R_KCAL * temperature))
    syn = float(np.sum(weights[x < 90.0]))
    anti = float(np.sum(weights[x > 90.0]))
    ratio = anti / syn
    return {
        "population_delta_g_anti_minus_syn": -R_KCAL * temperature * math.log(ratio),
        "syn_molar_fraction": 1.0 / (1.0 + ratio),
        "anti_molar_fraction": ratio / (1.0 + ratio),
        "anti_to_syn_ratio": ratio,
    }


def export(item: Export) -> None:
    package = DESTINATION / item.package
    data = package / "data"
    overlap = data / "window-overlap"
    overlap.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(item.config.read_text())
    for name in ("pmf.csv", "pmf_smooth.csv", "bootstrap_pmfs.csv", "thermodynamics.csv", "window_summary.csv", "overlap.csv"):
        shutil.copy2(item.source / name, data / name)
    shutil.copy2(item.source / "pmf.csv", data / "pmf_bootstrap_summary.csv")
    shutil.copy2(ROOT / "plotting/lefteris.mplstyle", package / "lefteris.mplstyle")
    shutil.copy2(ROOT / "plotting/reproduce_extended_torsion_handoff.py", package / "reproduce.py")
    (package / "requirements.txt").write_text("numpy>=1.24\nmatplotlib>=3.7\n")

    metadata_lines = (item.source / "metadata.dat").read_text().splitlines()
    indices: list[int] = []
    centers: list[float] = []
    forces: list[float] = []
    with (overlap / "metadata.dat").open("w") as stream:
        for line in metadata_lines:
            name, center, force = line.split()[:3]
            index = int(Path(name).stem.split("-")[-1])
            indices.append(index)
            centers.append(float(center))
            forces.append(float(force))
            shutil.copy2(item.source / name, overlap / name)
            stream.write(f"{name} {float(center):.8f} {float(force):.12g}\n")

    pmf_rows = list(csv.DictReader((item.source / "pmf.csv").open()))
    x = np.asarray([float(row["dihedral_deg"]) for row in pmf_rows])
    original = np.asarray([float(row["pmf_kcal_mol"]) for row in pmf_rows])
    bootstrap_rows = list(csv.DictReader((item.source / "bootstrap_pmfs.csv").open()))
    grouped: dict[int, list[tuple[float, float]]] = {}
    for row in bootstrap_rows:
        grouped.setdefault(int(row["bootstrap_replica"]), []).append(
            (float(row["dihedral_deg"]), float(row["pmf_kcal_mol"]))
        )
    replicas: list[dict[str, float]] = []
    for replica in sorted(grouped):
        values = grouped[replica]
        replica_x = np.asarray([value[0] for value in values])
        if not np.array_equal(replica_x, x):
            raise ValueError(f"Bootstrap grid mismatch in {item.package}, replica {replica}")
        replicas.append(integrated_values(x, np.asarray([value[1] for value in values]), 300.0))
    central = integrated_values(x, original, 300.0)
    names = list(central)
    with (data / "integrated_bootstrap.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bootstrap_replica", *names])
        for index, values in enumerate(replicas, start=1):
            writer.writerow([index, *(f"{values[name]:.12g}" for name in names)])
    with (data / "integrated_thermodynamics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["feature", "unit", "original", "bootstrap_mean", "bootstrap_std", "ci_lower", "ci_upper", "bootstrap_replicates"])
        units = {
            "population_delta_g_anti_minus_syn": "kcal mol^-1",
            "syn_molar_fraction": "fraction",
            "anti_molar_fraction": "fraction",
            "anti_to_syn_ratio": "ratio",
        }
        for name in names:
            values = np.asarray([replica[name] for replica in replicas])
            writer.writerow([name, units[name], f"{central[name]:.12g}", f"{np.mean(values):.12g}", f"{np.std(values, ddof=1):.12g}", f"{np.quantile(values, .025):.12g}", f"{np.quantile(values, .975):.12g}", len(values)])

    # Mirror the canonical filenames used by the existing Elisa packages while
    # retaining explicit names for these integrated basin-population results.
    shutil.copy2(data / "integrated_bootstrap.csv", data / "bootstrap_features.csv")
    shutil.copy2(data / "integrated_thermodynamics.csv", data / "bootstrap_feature_summary.csv")

    confidence = float(config["bootstrap"]["confidence_level"])
    meta = {
        "mode": "torsion",
        "system": item.system,
        "title": f"{item.system} extended single15 (ψ_cd) torsional umbrella PMF",
        "source": str(item.source.relative_to(ROOT)),
        "configuration": str(item.config.relative_to(ROOT)),
        "sampling_stage": item.stage,
        "sampling_note": item.note,
        "temperature_K": 300.0,
        "dividing_surface_deg": 90.0,
        "integration_definition": "Raw WHAM-bin sums of exp[-F/(RT)] for phi<90 (syn) and phi>90 (anti); uniform bin width cancels.",
        "xlim_deg": [float(config["coordinate"]["report_minimum_deg"]), float(config["coordinate"]["report_maximum_deg"])],
        "overlap_xlim_deg": [float(config["coordinate"]["minimum_deg"]), float(config["coordinate"]["maximum_deg"])],
        "overlap_bin_width_deg": (float(config["coordinate"]["maximum_deg"]) - float(config["coordinate"]["minimum_deg"])) / int(config["coordinate"]["bins"]),
        "window_indices": indices,
        "window_centers_deg": centers,
        "wham_force_constants_kcal_mol_deg2": forces,
        "bootstrap": {
            "method": "independent circular moving-block resampling within each umbrella window",
            "block_length_ps": float(config["bootstrap"]["block_length_ps"]),
            "requested_replicas": int(config["bootstrap"]["replicas"]),
            "successful_replicas": len(replicas),
            "seed": int(config["bootstrap"]["seed"]),
            "confidence_level": confidence,
            "pmf_zero_reference_deg": config["zero_reference_deg"],
        },
    }
    (package / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(item.package)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for item in EXPORTS:
        export(item)


if __name__ == "__main__":
    main()
