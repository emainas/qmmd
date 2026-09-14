import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from qmmd.ncoord import (
    load_config, read_symbols_from_dftb_inp, run_dir, run_ncoord,
    select_group_indices, system_base_dir,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/PRN-anti/ncoord/ncoord.yaml"


class NcoordConfigTests(unittest.TestCase):
    def test_legacy_layout(self):
        cfg = load_config(ROOT / "configs/HPD/ncoord/ncoord.yaml")
        self.assertEqual(system_base_dir(cfg, ROOT), ROOT / "systems/HPD/solv_5.5")

    def test_explicit_paths(self):
        cfg = load_config(CONFIG)
        self.assertEqual(cfg.run_ids, list(range(1, 21)))
        self.assertEqual(
            run_dir(cfg, ROOT, 1),
            ROOT / "systems/PRN-anti/solv_100/dftb/N1T48C1/run-1",
        )
        self.assertEqual(
            system_base_dir(replace(cfg, system_dir="/tmp/prn-test"), ROOT),
            Path("/tmp/prn-test"),
        )
        with self.assertRaisesRegex(ValueError, "system_dir or buffer"):
            system_base_dir(replace(cfg, system_dir=None), ROOT)

    def test_real_atoms_and_generation_in_tmp(self):
        cfg = load_config(CONFIG)
        source = run_dir(cfg, ROOT, 1) / "equil/dftb.inp"
        if not source.exists():
            self.skipTest("PRN simulation input unavailable")
        symbols = read_symbols_from_dftb_inp(source)
        self.assertEqual(len(symbols), 311)
        self.assertEqual((symbols[3], symbols[10]), ("O", "H"))
        self.assertEqual(select_group_indices(cfg.group1, symbols), [4])
        self.assertEqual(select_group_indices(cfg.group2, symbols), [11])
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp)
            (base / "equil").mkdir()
            (base / "equil/dftb.inp").symlink_to(source)
            with patch("qmmd.ncoord.load_config", return_value=replace(cfg, run_ids=[1])), \
                    patch("qmmd.ncoord.run_dir", return_value=base):
                run_ncoord(CONFIG)
                output = base / "meta-h/metacv.dat"
                self.assertEqual(output.read_text(),
                    "RATIONALCOORDINATIONNUMBER 0.1 1 1 6 12 1.6 LABELS AVERAGE 0.0 1.0 0.01\n4\n11\n")
                before = output.stat().st_mtime_ns
                run_ncoord(CONFIG)
                self.assertEqual(output.stat().st_mtime_ns, before)
                self.assertEqual((base / "meta-h/spec.yaml").read_text(), CONFIG.read_text())


if __name__ == "__main__":
    unittest.main()
