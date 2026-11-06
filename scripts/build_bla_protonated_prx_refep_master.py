#!/usr/bin/env python3
"""Build the matched BLA REFEP master with two fixed protonated PRX tails."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import parmed as pmd
import yaml
from parmed.tools import actions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_state_charges(path: Path, state: str) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    field = f"charge_{state}"
    if not rows or field not in rows[0]:
        raise ValueError(f"Missing {field} in {path}")
    column = rows[0].index(field)
    return {
        row[0]: float(row[column])
        for row in rows[1:]
        if row and row[0] not in {"TOTAL", "PROTON_COUNT"}
    }


def change_charge(parm: pmd.amber.AmberParm, atom_id: int, charge: float) -> None:
    actions.change(parm, "CHARGE", f"@{atom_id}", charge).execute()


def build(args: argparse.Namespace) -> None:
    parm = pmd.load_file(str(args.input_parm7), str(args.input_rst7))
    charges = read_state_charges(args.charge_sets, args.state)

    for residue_id in args.tail_residues:
        residue = parm.residues[residue_id - 1]
        if residue.name != "PRX":
            raise ValueError(
                f"Residue {residue_id} is {residue.name!r}, expected 'PRX'"
            )
        names = tuple(atom.name for atom in residue.atoms)
        if names != tuple(charges):
            raise ValueError(
                f"PRX {residue_id} atom ordering does not match {args.charge_sets}"
            )
        for atom in residue.atoms:
            change_charge(parm, atom.idx + 1, charges[atom.name])

    counterion = parm.residues[args.counterion_residue - 1]
    if counterion.name != "Na+" or len(counterion.atoms) != 1:
        raise ValueError(
            f"Residue {args.counterion_residue} is not the expected one-atom Na+"
        )
    replacement = parm.residues[args.replacement_water_residue - 1]
    if replacement.name != "WAT" or {atom.name for atom in replacement.atoms} != {
        "O",
        "H1",
        "H2",
    }:
        raise ValueError(
            f"Residue {args.replacement_water_residue} is not a three-atom WAT"
        )

    # Delete the obsolete sodium and the two hydrogens of a distant bulk water.
    # The surviving oxygen coordinate becomes the chloride coordinate.
    strip_mask = (
        f":{args.counterion_residue},"
        f":{args.replacement_water_residue}@H1,H2"
    )
    actions.strip(parm, strip_mask).execute()
    chloride_residue = next(
        residue
        for residue in parm.residues
        if residue.name == "WAT" and len(residue.atoms) == 1
    )
    chloride = chloride_residue.atoms[0]
    atom_index = chloride.idx

    chloride.name = "Cl-"
    chloride.type = "Cl-"
    chloride.charge = -1.0
    chloride.mass = 35.45
    chloride.atomic_number = 17
    chloride.solvent_radius = 1.7
    chloride.screen = 0.8
    for flag, value in (
        ("ATOM_NAME", "Cl-"),
        ("AMBER_ATOM_TYPE", "Cl-"),
        ("CHARGE", -1.0),
        ("MASS", 35.45),
        ("ATOMIC_NUMBER", 17),
        ("RADII", 1.7),
        ("SCREEN", 0.8),
    ):
        parm.parm_data[flag][atom_index] = value
    chloride_residue.name = "Cl-"
    parm.parm_data["RESIDUE_LABEL"][chloride_residue.idx] = "Cl-"
    actions.addLJType(
        parm,
        f"@{atom_index + 1}",
        radius=args.chloride_radius,
        epsilon=args.chloride_epsilon,
    ).execute()

    args.output_parm7.parent.mkdir(parents=True, exist_ok=True)
    parm.save(str(args.output_parm7), overwrite=True)
    parm.save(str(args.output_rst7), format="rst7", overwrite=True)

    residue_charges = {
        f"{index + 1}:{residue.name}": float(
            sum(atom.charge for atom in residue.atoms)
        )
        for index, residue in enumerate(parm.residues[:3])
    }
    total_charge = float(sum(atom.charge for atom in parm.atoms))
    if abs(residue_charges["1:TPS"] - 1.0) > 1.0e-5:
        raise ValueError("TPS master residue is not protonated (+1)")
    if any(
        abs(residue_charges[f"{index}:PRX"]) > 1.0e-5 for index in (2, 3)
    ):
        raise ValueError("A fixed protonated PRX residue is not neutral")
    if abs(total_charge) > 1.0e-5:
        raise ValueError(f"Master system charge is not neutral: {total_charge:+.8f}")

    provenance = {
        "method": "fixed protonated-tail BLA REFEP master",
        "input_parm7": str(args.input_parm7.resolve()),
        "input_rst7": str(args.input_rst7.resolve()),
        "input_parm7_sha256": sha256(args.input_parm7),
        "input_rst7_sha256": sha256(args.input_rst7),
        "prx_charge_sets": str(args.charge_sets.resolve()),
        "fixed_prx_state": args.state,
        "fixed_prx_residues": args.tail_residues,
        "removed_counterion_residue": args.counterion_residue,
        "chloride_source_water_residue": args.replacement_water_residue,
        "chloride_lj_radius_angstrom": args.chloride_radius,
        "chloride_lj_epsilon_kcal_mol": args.chloride_epsilon,
        "residue_charges": residue_charges,
        "system_charge": total_charge,
        "output_parm7": str(args.output_parm7.resolve()),
        "output_rst7": str(args.output_rst7.resolve()),
        "output_parm7_sha256": sha256(args.output_parm7),
        "output_rst7_sha256": sha256(args.output_rst7),
    }
    args.provenance.write_text(yaml.safe_dump(provenance, sort_keys=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parm7", type=Path, required=True)
    parser.add_argument("--input-rst7", type=Path, required=True)
    parser.add_argument("--charge-sets", type=Path, required=True)
    parser.add_argument("--state", default="o1_syn")
    parser.add_argument("--tail-residues", type=int, nargs=2, default=[2, 3])
    parser.add_argument("--counterion-residue", type=int, default=4)
    parser.add_argument("--replacement-water-residue", type=int, default=107)
    parser.add_argument("--chloride-radius", type=float, default=2.513)
    parser.add_argument("--chloride-epsilon", type=float, default=0.035591)
    parser.add_argument("--output-parm7", type=Path, required=True)
    parser.add_argument("--output-rst7", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
