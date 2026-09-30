from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from qmmd.cphmd_build import (
    build_dummy_protons,
    load_config,
    read_mol2,
    run_cphmd_build,
)


MOL2 = """@<TRIPOS>MOLECULE
TST
4 3 1 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
1 CB  -1.0000  0.0000  0.0000 c3 1 TST -0.1000
2 CG   0.0000  0.0000  0.0000 c  1 TST  0.7000
3 O1   0.6000  1.0392  0.0000 o  1 TST -0.8000
4 O2   0.6000 -1.0392  0.0000 o  1 TST -0.8000
@<TRIPOS>BOND
1 1 2 1
2 2 3 1
3 2 4 1
@<TRIPOS>SUBSTRUCTURE
1 TST 1 TEMP 0 **** **** 0 ROOT
"""


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


class CphmdBuildTests(unittest.TestCase):
    def _files(self, directory: Path) -> tuple[Path, Path]:
        mol2 = directory / "test.mol2"
        mol2.write_text(MOL2)
        config = directory / "build.yaml"
        config.write_text(
            f"""system: TST
mol2: {mol2}
output_pdb: test_cphmd.pdb
carboxyl:
  carbon: CG
  oxygens: [O1, O2]
  dummy_protons:
    O1: {{syn: H11, anti: H12}}
    O2: {{syn: H21, anti: H22}}
geometry:
  oh_distance_angstrom: 0.96
  dummy_pair_angle_degrees: 134.0
"""
        )
        return mol2, config

    def test_builds_four_in_plane_protons_with_requested_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mol2, config = self._files(Path(tmp))
            cfg = load_config(config)
            structure = read_mol2(mol2)
            protons = build_dummy_protons(cfg, structure)

            self.assertEqual([item.name for item in protons], ["H11", "H12", "H21", "H22"])
            atoms = {atom.name: atom for atom in structure.atoms}
            for proton in protons:
                oxygen = atoms[proton.parent_oxygen]
                self.assertAlmostEqual(
                    _distance(proton.coordinates, oxygen.coordinates),
                    0.96,
                    places=10,
                )
                self.assertAlmostEqual(proton.coordinates[2], 0.0, places=10)

            for first, second in ((protons[0], protons[1]), (protons[2], protons[3])):
                oxygen = atoms[first.parent_oxygen].coordinates
                vector1 = tuple(a - b for a, b in zip(first.coordinates, oxygen, strict=True))
                vector2 = tuple(a - b for a, b in zip(second.coordinates, oxygen, strict=True))
                cosine = sum(a * b for a, b in zip(vector1, vector2, strict=True)) / 0.96**2
                self.assertAlmostEqual(math.degrees(math.acos(cosine)), 134.0, places=10)

    def test_run_writes_pdb_beside_source_without_changing_mol2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mol2, config = self._files(Path(tmp))
            original = mol2.read_text()
            run_cphmd_build(config)

            output = mol2.with_name("test_cphmd.pdb")
            self.assertTrue(output.is_file())
            atom_lines = [line for line in output.read_text().splitlines() if line.startswith("HETATM")]
            self.assertEqual(len(atom_lines), 8)
            for name in ("H11", "H12", "H21", "H22"):
                self.assertTrue(any(name in line for line in atom_lines))
            self.assertEqual(mol2.read_text(), original)

    def test_rejects_unbonded_configured_oxygen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mol2, config = self._files(Path(tmp))
            mol2.write_text(MOL2.replace("3 2 4 1", "3 1 4 1"))
            cfg = load_config(config)
            with self.assertRaisesRegex(ValueError, "CG and O2 are not bonded"):
                build_dummy_protons(cfg, read_mol2(mol2))


if __name__ == "__main__":
    unittest.main()
