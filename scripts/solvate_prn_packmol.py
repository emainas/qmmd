"""Pack a fixed PRN conformer with waters and export PDB and periodic XYZ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import parmed as pmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--waters", type=int, default=50)
    parser.add_argument("--water-density", type=float, default=1.0,
                        help="Water mass / total box volume, g/cm^3 (excludes solute mass)")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Packmol separation in Angstrom")
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if args.waters <= 0 or args.water_density <= 0 or args.tolerance <= 0:
        parser.error("Water count, density and tolerance must be positive")
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    length = (args.waters * 18.01528 / 6.02214076e23 * 1e24 / args.water_density) ** (1/3)
    mol = pmd.load_file(str(source))
    initial = mol.coordinates.copy()
    mol.coordinates = initial - initial.mean(axis=0) + length / 2
    if np.any(mol.coordinates < 0) or np.any(mol.coordinates > length):
        raise ValueError("Solute does not fit inside box")
    mol.save(str(out / "solute.pdb"))
    # Obtain the TIP3P template directly from Amber's water library.
    (out / "water.leap.in").write_text(
        "source leaprc.water.tip3p\nsavepdb TP3 water.pdb\nquit\n")
    with (out / "water.leap.log").open("w") as log:
        subprocess.run(["tleap", "-f", "water.leap.in"], cwd=out, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    water = pmd.load_file(str(out / "water.pdb"))
    assert len(water.atoms) == 3
    packing = f"""tolerance {args.tolerance}
filetype pdb
output packed.pdb
seed {args.seed}
nloop 1000
pbc 0.0 0.0 0.0 {length:.8f} {length:.8f} {length:.8f}
structure solute.pdb
 number 1
 fixed 0.0 0.0 0.0 0.0 0.0 0.0
end structure
structure water.pdb
 number {args.waters}
 inside box 0.0 0.0 0.0 {length:.8f} {length:.8f} {length:.8f}
end structure
"""
    (out / "packmol.inp").write_text(packing)
    with (out / "packmol.log").open("w") as log, (out / "packmol.inp").open() as inp:
        subprocess.run(["packmol"], stdin=inp, cwd=out,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if "Success!" not in (out / "packmol.log").read_text():
        raise RuntimeError("Packmol did not report success")
    solv = pmd.load_file(str(out / "packed.pdb"))
    nsolute = len(mol.atoms)
    assert len(solv.atoms) == nsolute + 3 * args.waters
    xyz = solv.coordinates
    np.testing.assert_allclose(xyz[:nsolute], mol.coordinates, atol=0.0011)
    assert [a.name for a in solv.atoms[:nsolute]] == [a.name for a in mol.atoms]
    assert [a.atomic_number for a in solv.atoms[nsolute:]] == [a.atomic_number for a in water.atoms] * args.waters
    labels = np.concatenate((np.zeros(nsolute, dtype=int), np.repeat(np.arange(1, args.waters + 1), 3)))
    delta = xyz[:, None, :] - xyz[None, :, :]
    delta -= length * np.round(delta / length)
    distances = np.linalg.norm(delta, axis=-1)
    minimum = float(distances[labels[:, None] != labels[None, :]].min())
    if minimum < args.tolerance - 0.01:
        raise ValueError(f"Periodic intermolecular separation too small: {minimum}")
    solv.box = [length, length, length, 90, 90, 90]
    solv.save(str(out / "solvated.pdb"))
    lattice = f"{length:.8f} 0 0 0 {length:.8f} 0 0 0 {length:.8f}"
    lines = [str(len(solv.atoms)), f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"']
    symbols = {1: "H", 6: "C", 8: "O"}
    lines.extend(f"{symbols[a.atomic_number]} {r[0]:.8f} {r[1]:.8f} {r[2]:.8f}" for a, r in zip(solv.atoms, xyz))
    (out / "solvated.xyz").write_text("\n".join(lines) + "\n")
    result = dict(source=str(source), waters=args.waters, atoms=len(solv.atoms),
                  box_length_A=length, water_mass_per_box_volume_g_cm3=args.water_density,
                  minimum_periodic_intermolecular_distance_A=minimum,
                  tolerance_A=args.tolerance, seed=args.seed,
                  note="Packed starting geometry; no solvent minimization or equilibration performed.")
    (out / "packing_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
