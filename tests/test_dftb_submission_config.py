"""Submission configuration checks; never invoke sbatch."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qmmd import dftb


ROOT = Path(__file__).resolve().parents[1]


class SubmissionConfigTests(unittest.TestCase):
    def test_existing_buffer_layout(self) -> None:
        cfg = dftb.load_config(ROOT / "configs/HPD/dftb/dftb.yaml")
        self.assertEqual(dftb.out_dir(cfg, ROOT), ROOT / "systems/HPD/solv_5.5/dftb/N1T48C1")
        self.assertIsNone(cfg.slurm.job.qos)

    def test_prn_matches_twenty_runs_without_submission(self) -> None:
        for rotamer in ("anti", "syn"):
            path = ROOT / f"configs/PRN-{rotamer}/dftb/dftb.yaml"
            cfg = dftb.load_config(path)
            bench = ROOT / f"systems/PRN-{rotamer}/solv_100/dftb/N1T48C1"
            self.assertEqual(dftb.out_dir(cfg, ROOT), bench)
            if not bench.exists():
                continue  # Simulation directories are not tracked in git.
            self.assertEqual(len(dftb.matching_run_dirs(bench, "equil", path.read_text())), 20)
            with patch("builtins.input", return_value="n") as prompt, patch.object(dftb, "submit_slurm") as submit:
                dftb.run_dftb_submit(path)
                prompt.assert_called_once_with("Proceed to submit 20 jobs? [y/N] ")
                submit.assert_not_called()

    def test_qos_script_generation(self) -> None:
        cfg = dftb.load_config(ROOT / "configs/PRN-anti/dftb/dftb.yaml")
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            script = dftb.write_slurm_sh(cfg, Path(folder)).read_text()
        self.assertIn("#SBATCH --qos=highpri\n", script)
        self.assertIn("#SBATCH --partition=batch\n", script)
        self.assertIn("#SBATCH --time=1-00:00:00\n", script)


if __name__ == "__main__":
    unittest.main()
