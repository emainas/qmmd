#!/usr/bin/env python3
"""Build a neutral BLA master with fixed BPP TPS and two titratable PRN tails."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import parmed as pmd
import yaml
from parmed.tools import actions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mol2_charges(path: Path) -> dict[str, float]:
    charges: dict[str, float] = {}
    in_atoms = False
    for line in path.read_text().splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and in_atoms:
            break
        if not in_atoms or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 9:
            raise ValueError(f"Malformed MOL2 atom line in {path}: {line}")
        name = fields[1]
        if name in charges:
            raise ValueError(f"Duplicate atom name {name!r} in {path}")
        charges[name] = float(fields[8])
    if not charges:
        raise ValueError(f"No MOL2 charges found in {path}")
    return charges


def change_charge(parm: pmd.amber.AmberParm, atom_id: int, charge: float) -> None:
    actions.change(parm, "CHARGE", f"@{atom_id}", charge).execute()


def minimum_image_distances(
    point: np.ndarray, coordinates: np.ndarray, box_lengths: np.ndarray
) -> np.ndarray:
    delta = coordinates - point
    delta -= box_lengths * np.rint(delta / box_lengths)
    return np.linalg.norm(delta, axis=1)


def select_bulk_water(parm: pmd.amber.AmberParm) -> tuple[int, float]:
    if parm.coordinates is None or parm.box is None:
        raise ValueError("Topology must include coordinates and periodic box lengths")
    box = np.asarray(parm.box[:3], dtype=float)
    solute_coordinates = np.asarray(
        [
            [atom.xx, atom.xy, atom.xz]
            for residue in parm.residues[:3]
            for atom in residue.atoms
        ]
    )
    best: tuple[int, float] | None = None
    for residue_id, residue in enumerate(parm.residues, start=1):
        if residue.name != "WAT" or {atom.name for atom in residue.atoms} != {
            "O",
            "H1",
            "H2",
        }:
            continue
        oxygen = next(atom for atom in residue.atoms if atom.name == "O")
        point = np.asarray([oxygen.xx, oxygen.xy, oxygen.xz], dtype=float)
        distance = float(minimum_image_distances(point, solute_coordinates, box).min())
        if best is None or distance > best[1]:
            best = (residue_id, distance)
    if best is None:
        raise ValueError("No TIP3P water residue is available for Na+ placement")
    return best


def build(args: argparse.Namespace) -> None:
    parm = pmd.load_file(str(args.input_parm7), str(args.input_rst7))
    tps = parm.residues[0]
    if tps.name != "TPS":
        raise ValueError(f"Residue 1 is {tps.name!r}, expected 'TPS'")

    endpoint_charges = read_mol2_charges(args.bpp_mol2)
    omitted = set(args.omit_atoms)
    unexpected = set(endpoint_charges) - {atom.name for atom in tps.atoms} - omitted
    missing = {atom.name for atom in tps.atoms} - set(endpoint_charges)
    if unexpected or missing:
        raise ValueError(
            f"BPP/TPS atom mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for atom in tps.atoms:
        change_charge(parm, atom.idx + 1, endpoint_charges[atom.name])

    for residue_id in args.prn_residues:
        residue = parm.residues[residue_id - 1]
        if residue.name != "PRN":
            raise ValueError(
                f"Residue {residue_id} is {residue.name!r}, expected 'PRN'"
            )

    sodium_atoms = [atom for residue in parm.residues if residue.name == "Na+" for atom in residue.atoms]
    if len(sodium_atoms) != 1:
        raise ValueError(f"Expected one existing Na+ atom, found {len(sodium_atoms)}")
    sodium_template = sodium_atoms[0]
    sodium_properties = {
        "type": sodium_template.type,
        "charge": sodium_template.charge,
        "mass": sodium_template.mass,
        "atomic_number": sodium_template.atomic_number,
        "solvent_radius": sodium_template.solvent_radius,
        "screen": sodium_template.screen,
        "rmin": sodium_template.rmin,
        "epsilon": sodium_template.epsilon,
    }

    water_residue_id, min_solute_distance = select_bulk_water(parm)
    actions.strip(parm, f":{water_residue_id}@H1,H2").execute()
    replacement = next(
        residue
        for residue in parm.residues
        if residue.name == "WAT" and len(residue.atoms) == 1
    )
    sodium = replacement.atoms[0]
    atom_index = sodium.idx
    sodium.name = "Na+"
    sodium.type = str(sodium_properties["type"])
    sodium.charge = float(sodium_properties["charge"])
    sodium.mass = float(sodium_properties["mass"])
    sodium.atomic_number = int(sodium_properties["atomic_number"])
    sodium.solvent_radius = float(sodium_properties["solvent_radius"])
    sodium.screen = float(sodium_properties["screen"])
    for flag, value in (
        ("ATOM_NAME", "Na+"),
        ("AMBER_ATOM_TYPE", sodium.type),
        ("CHARGE", sodium.charge),
        ("MASS", sodium.mass),
        ("ATOMIC_NUMBER", sodium.atomic_number),
        ("RADII", sodium.solvent_radius),
        ("SCREEN", sodium.screen),
    ):
        parm.parm_data[flag][atom_index] = value
    replacement.name = "Na+"
    parm.parm_data["RESIDUE_LABEL"][replacement.idx] = "Na+"
    actions.addLJType(
        parm,
        f"@{atom_index + 1}",
        radius=float(sodium_properties["rmin"]),
        epsilon=float(sodium_properties["epsilon"]),
    ).execute()

    args.output_parm7.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    parm.save(str(args.output_parm7), overwrite=True)
    parm.save(str(args.output_rst7), format="rst7", overwrite=True)

    reloaded = pmd.load_file(str(args.output_parm7), str(args.output_rst7))
    residue_charges = {
        f"{index + 1}:{residue.name}": float(sum(atom.charge for atom in residue.atoms))
        for index, residue in enumerate(reloaded.residues[:3])
    }
    total_charge = float(sum(atom.charge for atom in reloaded.atoms))
    sodium_count = sum(residue.name == "Na+" for residue in reloaded.residues)
    if abs(residue_charges["1:TPS"]) > 1.0e-5:
        raise ValueError("Fixed BPP TPS residue is not neutral")
    if any(
        abs(residue_charges[f"{index}:PRN"] + 1.0) > 1.0e-5
        for index in args.prn_residues
    ):
        raise ValueError("A PRN residue is not in its deprotonated -1 topology state")
    if sodium_count != 2:
        raise ValueError(f"Expected two Na+ counterions, found {sodium_count}")
    if abs(total_charge) > 1.0e-5:
        raise ValueError(f"Master system charge is not neutral: {total_charge:+.8f}")

    provenance = {
        "method": "fixed BPP TPS with two standard PRN tails",
        "input_parm7": str(args.input_parm7.resolve()),
        "input_rst7": str(args.input_rst7.resolve()),
        "input_parm7_sha256": sha256(args.input_parm7),
        "input_rst7_sha256": sha256(args.input_rst7),
        "bpp_charge_source": str(args.bpp_mol2.resolve()),
        "bpp_charge_source_sha256": sha256(args.bpp_mol2),
        "omitted_model_caps": args.omit_atoms,
        "fixed_tps_state": "BPP",
        "titrating_prn_residues": args.prn_residues,
        "new_sodium_source_water_residue": water_residue_id,
        "new_sodium_minimum_solute_distance_angstrom": min_solute_distance,
        "residue_charges": residue_charges,
        "sodium_count": sodium_count,
        "system_charge": total_charge,
        "radius_set": reloaded.parm_data.get("RADIUS_SET", []),
        "output_parm7": str(args.output_parm7.resolve()),
        "output_rst7": str(args.output_rst7.resolve()),
        "output_parm7_sha256": sha256(args.output_parm7),
        "output_rst7_sha256": sha256(args.output_rst7),
    }
    args.provenance.write_text(yaml.safe_dump(provenance, sort_keys=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parm7", type=Path, required=True)
    parser.add_argument("--input-rst7", type=Path, required=True)
    parser.add_argument("--bpp-mol2", type=Path, required=True)
    parser.add_argument("--omit-atoms", nargs="+", default=["H2C", "H3B"])
    parser.add_argument("--prn-residues", type=int, nargs=2, default=[2, 3])
    parser.add_argument("--output-parm7", type=Path, required=True)
    parser.add_argument("--output-rst7", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
