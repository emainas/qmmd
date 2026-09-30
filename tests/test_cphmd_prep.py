import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from qmmd.cphmd_prep import (
    AcidPrepConfig,
    load_config,
    map_charges,
    read_full_mol2,
    run_cphmd_prep,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/MEA/cphmd/prep.yaml"


class CpHMDPrepTests(unittest.TestCase):
    def test_real_mea_mapping(self):
        cfg = load_config(CONFIG)
        mapping = map_charges(cfg)

        self.assertEqual(len(mapping), 12)
        self.assertEqual([item.master_atom.name for item in mapping], [
            "C1", "C2", "O", "N", "H1", "H2", "H3", "H4", "HO",
            "HN1", "HN2", "HN3",
        ])
        self.assertIsNone(mapping[9].deprot_atom)
        self.assertEqual(mapping[9].deprot_charge, 0.0)
        self.assertEqual(mapping[10].deprot_atom.name, "HN2")
        self.assertAlmostEqual(mapping[10].deprot_charge, 0.402)

    def test_command_prints_table_and_writes_csv(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            output_file = Path(tmp) / "charge_sets.csv"
            config = Path(tmp) / "prep.yaml"
            config.write_text(
                "system: MEA\n"
                f"input_dir: {ROOT / 'systems/MEA/init'}\n"
                "prot_mol2: mea.mol2\n"
                "proton_count_prot: 3\n"
                "deprot_mol2: mea_deprot.mol2\n"
                "proton_count_deprot: 2\n"
                "master: mea.mol2\n"
                f"output_file: {output_file}\n"
            )
            output = StringIO()
            with redirect_stdout(output):
                run_cphmd_prep(config)

            self.assertIn("atom_master", output.getvalue())
            self.assertIn("HN1", output.getvalue())
            self.assertIn("HN1           0.523667     0.000000", output.getvalue())
            self.assertIn("TOTAL         +1.001001    +0.001000", output.getvalue())
            self.assertIn("PROTON_COUNT  3            2", output.getvalue())
            self.assertIn(f"OK: wrote {output_file}", output.getvalue())
            csv_text = output_file.read_text()
            self.assertIn("HN1,0.523667,0.000000", csv_text)
            self.assertIn("TOTAL,+1.001001,+0.001000", csv_text)
            self.assertIn("PROTON_COUNT,3,2", csv_text)

    def test_rejects_more_than_one_missing_atom(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("")
            (root / "prot.mol2").write_text(
                "@<TRIPOS>ATOM\n"
                "1 C1 0 0 0 c3 1 MOL 0.0\n"
                "2 H1 0 0 0 hc 1 MOL 0.5\n"
                "3 H2 0 0 0 hc 1 MOL 0.5\n"
                "@<TRIPOS>BOND\n"
            )
            (root / "deprot.mol2").write_text(
                "@<TRIPOS>ATOM\n"
                "1 C1 0 0 0 c3 1 MOL 0.0\n"
                "@<TRIPOS>BOND\n"
            )
            config = root / "prep.yaml"
            config.write_text(
                "system: MOL\n"
                "input_dir: .\n"
                "prot_mol2: prot.mol2\n"
                "proton_count_prot: 2\n"
                "deprot_mol2: deprot.mol2\n"
                "proton_count_deprot: 1\n"
                "master: prot.mol2\n"
                "output_file: charges.csv\n"
            )
            with self.assertRaisesRegex(ValueError, "exactly one master atom"):
                map_charges(load_config(config))

    def test_real_pr4_acid_prep_builds_master_and_five_charge_states(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            output_dir = Path(tmp)
            (output_dir / "pyproject.toml").write_text("")
            master = output_dir / "pr4_cphmd.mol2"
            frcmod = output_dir / "pr4_cphmd.frcmod"
            charge_sets = output_dir / "charge_sets.csv"
            config = output_dir / "prep.yaml"
            deprotonated = ROOT / "systems/PR4/init/pr4.mol2"
            geometry = ROOT / "systems/PR4/init/pr4_cphmd.pdb"
            protonated = ROOT / "systems/PRN-anti/init/anti_min.mol2"
            original_deprotonated = deprotonated.read_text()
            config.write_text(
                "system: PR4\n"
                "chemistry: carboxylic_acid\n"
                "input:\n"
                f"  deprotonated_mol2: {deprotonated}\n"
                f"  dummy_geometry_pdb: {geometry}\n"
                f"  protonated_mol2: {protonated}\n"
                "carboxyl:\n"
                "  carbon: CG\n"
                "  oxygens: [O1, O2]\n"
                "  protonated_oxygen: O1\n"
                "  proton_atom: H11\n"
                "  dummy_protons:\n"
                "    O1: {syn: H11, anti: H12}\n"
                "    O2: {syn: H21, anti: H22}\n"
                "output:\n"
                f"  master_mol2: {master}\n"
                f"  frcmod: {frcmod}\n"
                f"  charge_sets: {charge_sets}\n"
            )

            loaded = load_config(config)
            self.assertIsInstance(loaded, AcidPrepConfig)
            output = StringIO()
            with redirect_stdout(output):
                run_cphmd_prep(config)

            structure = read_full_mol2(master)
            self.assertEqual(len(structure.atoms), 14)
            self.assertEqual(len(structure.bonds), 13)
            atoms = {atom.name: atom for atom in structure.atoms}
            self.assertEqual(atoms["O1"].atom_type, "oh")
            self.assertEqual(atoms["O2"].atom_type, "oh")
            for name in ("H11", "H12", "H21", "H22"):
                self.assertEqual(atoms[name].atom_type, "ho")
                self.assertEqual(atoms[name].charge, 0.0)

            rows = [line.split(",") for line in charge_sets.read_text().splitlines()]
            self.assertEqual(
                rows[0],
                [
                    "atom_master",
                    "charge_deprot",
                    "charge_o1_syn",
                    "charge_o1_anti",
                    "charge_o2_syn",
                    "charge_o2_anti",
                ],
            )
            by_name = {row[0]: row[1:] for row in rows[1:]}
            self.assertEqual(by_name["H11"], ["0.000000", "0.457000", "0.000000", "0.000000", "0.000000"])
            self.assertEqual(by_name["H22"], ["0.000000", "0.000000", "0.000000", "0.000000", "0.457000"])
            self.assertEqual(by_name["O1"], ["-0.750000", "-0.616100", "-0.616100", "-0.447000", "-0.447000"])
            self.assertEqual(by_name["TOTAL"], ["-0.998999", "-0.000001", "-0.000001", "-0.000001", "-0.000001"])
            self.assertEqual(by_name["PROTON_COUNT"], ["0", "1", "1", "1", "1"])
            self.assertIn("ho-oh-ho    60.000   134.000", frcmod.read_text())
            self.assertIn(f"OK: wrote acid master MOL2 to {master}", output.getvalue())
            self.assertEqual(deprotonated.read_text(), original_deprotonated)


if __name__ == "__main__":
    unittest.main()
