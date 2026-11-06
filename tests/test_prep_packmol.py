import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from qmmd.prep import (
    load_config,
    materialize_packed_mol2,
    write_packmol_in,
    write_tleap_in,
)


ROOT = Path(__file__).resolve().parents[1]


class PrepPackmolTests(unittest.TestCase):
    def _configuration(self, root: Path) -> Path:
        init = root / "init"
        init.mkdir()
        shutil.copy2(ROOT / "systems/BLA/init/bla.mol2", init / "bla.mol2")
        for name in ("tps.frcmod", "prx.frcmod"):
            shutil.copy2(ROOT / "systems/BLA/init" / name, init / name)
        config = root / "prep.yaml"
        config.write_text(
            "name: BLA4\n"
            "leaprc_mol: leaprc.gaff2\n"
            "leaprc_sol: leaprc.water.tip3p\n"
            "frcmod:\n"
            "  - tps.frcmod\n"
            "  - prx.frcmod\n"
            "mol2: bla.mol2\n"
            "packing:\n"
            "  copies: 2\n"
            "  tolerance: 2.2\n"
            "  seed: 2026\n"
            "  box: [40.0, 45.0, 50.0]\n"
            "water_model: TIP3PBOX\n"
            "buffer: 8.0\n"
            "counterion: Na+\n"
            "counterion_num: 2\n"
            f"input_dir: {init}\n"
            "prefix: solv\n"
        )
        return config

    def test_default_component_and_three_length_box_are_normalized(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            cfg = load_config(self._configuration(Path(tmp)))
            self.assertIsNotNone(cfg.packing)
            assert cfg.packing is not None
            self.assertEqual(cfg.packing.box, (0.0, 0.0, 0.0, 40.0, 45.0, 50.0))
            self.assertEqual(cfg.packing.structures[0].name, "BLA4")
            self.assertEqual(cfg.packing.structures[0].number, 2)

    def test_packmol_coordinates_feed_topology_preserving_mol2_copies(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            cfg = load_config(self._configuration(Path(tmp)))
            packmol_in = write_packmol_in(cfg)
            packmol_text = packmol_in.read_text()
            self.assertIn("tolerance 2.2", packmol_text)
            self.assertIn("seed 2026", packmol_text)
            self.assertIn("number 2", packmol_text)
            self.assertIn("inside box 0 0 0 40 45 50", packmol_text)

            template_pdb = packmol_in.parent / "packing-template-BLA4.pdb"
            atom_lines = [
                line
                for line in template_pdb.read_text().splitlines()
                if line.startswith(("ATOM  ", "HETATM"))
            ]
            packed_pdb = packmol_in.parent / "packed.pdb"
            packed_pdb.write_text("\n".join([*atom_lines, *atom_lines, "END", ""]))

            outputs = materialize_packed_mol2(cfg, packed_pdb)
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.is_file() for path in outputs))
            with (packmol_in.parent / "packing-map.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["atom_start"], "1")
            self.assertEqual(rows[0]["atom_end"], "84")
            self.assertEqual(rows[0]["residue_start"], "1")
            self.assertEqual(rows[0]["residue_end"], "3")
            self.assertEqual(rows[1]["atom_start"], "85")
            self.assertEqual(rows[1]["residue_start"], "4")
            self.assertEqual(rows[1]["residue_end"], "6")

            tleap_in = write_tleap_in(cfg)
            tleap_text = tleap_in.read_text()
            self.assertIn("pack_BLA4_001 = loadmol2 packed-BLA4-001.mol2", tleap_text)
            self.assertIn("pack_BLA4_002 = loadmol2 packed-BLA4-002.mol2", tleap_text)
            self.assertIn("sys = combine {pack_BLA4_001 pack_BLA4_002}", tleap_text)
            self.assertIn("solvatebox sys TIP3PBOX 8.0", tleap_text)
            self.assertIn("addions sys Na+ 2", tleap_text)

    def test_packing_rejects_auxiliary_bonded_components(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            config = self._configuration(Path(tmp))
            with config.open("a") as handle:
                handle.write(
                    "additional_mol2:\n"
                    "  - unit: cap\n"
                    "    mol2: cap.mol2\n"
                )
            with self.assertRaisesRegex(ValueError, "packing cannot be combined"):
                load_config(config)


if __name__ == "__main__":
    unittest.main()
