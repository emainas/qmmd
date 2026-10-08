import shutil
import tempfile
import unittest
from pathlib import Path

from qmmd.prep import load_config, write_tleap_in


ROOT = Path(__file__).resolve().parents[1]


class PrepComponentTests(unittest.TestCase):
    def test_multiple_frcmods_are_copied_and_loaded_in_order(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            init = root / "init"
            init.mkdir()
            shutil.copy2(ROOT / "systems/BLA/init/bla.mol2", init / "bla.mol2")
            for name in ("tps.frcmod", "prx.frcmod"):
                (init / name).write_text(f"{name}\n")
            config = root / "prep.yaml"
            config.write_text(
                "name: BLA\n"
                "leaprc_mol: leaprc.gaff2\n"
                "leaprc_sol: leaprc.water.tip3p\n"
                "mol2: bla.mol2\n"
                "frcmod:\n"
                "  - tps.frcmod\n"
                "  - prx.frcmod\n"
                "water_model: TIP3PBOX\n"
                "buffer: 10.0\n"
                f"input_dir: {init}\n"
                "prefix: solv\n"
            )

            cfg = load_config(config)
            tleap_in = write_tleap_in(cfg)
            text = tleap_in.read_text()

            self.assertEqual(
                cfg.frcmods,
                (Path("tps.frcmod"), Path("prx.frcmod")),
            )
            self.assertLess(
                text.index("loadamberparams tps.frcmod"),
                text.index("loadamberparams prx.frcmod"),
            )
            self.assertTrue((tleap_in.parent / "tps.frcmod").is_file())
            self.assertTrue((tleap_in.parent / "prx.frcmod").is_file())

    def test_prx_reference_cap_is_combined_and_bonded(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            init = root / "init"
            init.mkdir()
            for name in ("prx_cphmd.mol2", "prx_cap.mol2", "prx_cphmd.frcmod"):
                shutil.copy2(ROOT / "systems" / "PRX" / "init" / name, init / name)
            config = root / "prep.yaml"
            config.write_text(
                "name: PRX\n"
                "leaprc_mol: leaprc.gaff2\n"
                "leaprc_sol: leaprc.water.tip3p\n"
                "frcmod_ion: frcmod.ionsjc_tip3p\n"
                "mol2: prx_cphmd.mol2\n"
                "frcmod: prx_cphmd.frcmod\n"
                "additional_mol2:\n"
                "  - unit: cap\n"
                "    mol2: prx_cap.mol2\n"
                "inter_residue_bonds:\n"
                "  - [1.CA, 2.HA3]\n"
                "water_model: TIP3PBOX\n"
                "buffer: 5.5\n"
                "counterion: Na+\n"
                "counterion_num: 1\n"
                f"input_dir: {init}\n"
                "prefix: solv\n"
            )

            cfg = load_config(config)
            tleap_in = write_tleap_in(cfg)
            text = tleap_in.read_text()

            self.assertIn("solute = loadmol2 prx_cphmd.mol2", text)
            self.assertIn("cap = loadmol2 prx_cap.mol2", text)
            self.assertIn("sys = combine {solute cap}", text)
            self.assertIn("bond sys.1.CA sys.2.HA3", text)
            self.assertTrue((tleap_in.parent / "prx_cap.mol2").is_file())


if __name__ == "__main__":
    unittest.main()
