import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from qmmd.cphmd_prep import load_config, map_charges, run_cphmd_prep


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


if __name__ == "__main__":
    unittest.main()
