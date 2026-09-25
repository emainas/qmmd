import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from qmmd import meta, ncoord


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/PRN-anti/meta/meta.yaml"


class MetaConfigTests(unittest.TestCase):
    def test_optional_qos_matches_dftb_prep(self):
        cfg = meta.load_config(ROOT / "configs/PRD/meta/meta.yaml")
        self.assertEqual(cfg.slurm.job.qos, "highpri")
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            script = meta.write_slurm_sh(cfg, Path(tmp)).read_text()
        self.assertIn("#SBATCH --qos=highpri\n", script)
        self.assertIn("#SBATCH --job-name=PRD-N1T48C1\n", script)

        legacy = meta.load_config(ROOT / "configs/PRN-anti/meta/meta.yaml")
        self.assertIsNone(legacy.slurm.job.qos)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            legacy_script = meta.write_slurm_sh(legacy, Path(tmp)).read_text()
        self.assertNotIn("#SBATCH --qos=", legacy_script)

    def test_paths_and_template(self):
        cfg = meta.load_config(CONFIG)
        template = meta.load_config(ROOT / "configs/BV/meta/na.yaml")
        cv = ncoord.load_config(ROOT / "configs/PRN-anti/ncoord/ncoord.yaml")
        self.assertEqual(meta.system_base_dir(template, ROOT), ROOT / "systems/BV/solv_4.0")
        self.assertEqual(meta.run_dir(cfg, ROOT, 1), ncoord.run_dir(cv, ROOT, 1))
        self.assertEqual(cfg.cv_dirname, cv.cv_dirname)
        self.assertEqual(cfg.run_ids, list(range(1, 21)))
        self.assertEqual(cfg.dftb.header_lines, template.dftb.header_lines)
        self.assertEqual([e.symbol for e in cfg.dftb.elements], ["H", "O", "C"])
        self.assertEqual(meta.system_base_dir(replace(cfg, system_dir="/tmp/prn-meta"), ROOT), Path("/tmp/prn-meta"))
        with self.assertRaisesRegex(ValueError, "system_dir or buffer"):
            meta.system_base_dir(replace(cfg, system_dir=None), ROOT)

    def test_render_real_input_in_tmp(self):
        cfg = meta.load_config(CONFIG)
        source = meta.run_dir(cfg, ROOT, 1) / "equil/dftb.inp"
        if not source.exists():
            self.skipTest("PRN simulation input unavailable")
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            out = Path(tmp)
            inp = meta.write_dftb_inp(cfg, ROOT, out, source, seed=123)
            self.assertEqual(meta.read_dftb_inp_coords(inp), meta.read_dftb_inp_coords(source))
            self.assertIn("RANDOMSEED=123", inp.read_text())
            slurm = meta.write_slurm_sh(cfg, out).read_text()
            self.assertIn("#SBATCH --time=1-00:00:00", slurm)
            self.assertIn("#SBATCH --partition=batch", slurm)
            self.assertIn("#SBATCH --job-name=PRN-anti-meta-h-N1T48C1", slurm)
            script = meta.write_run_sh(cfg, out).read_text()
            self.assertIn(cfg.runtime.executable, script)
            self.assertIn("mpirun -np 16", script)


if __name__ == "__main__":
    unittest.main()
