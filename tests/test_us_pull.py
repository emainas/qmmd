import tempfile
import unittest
import subprocess
from dataclasses import replace
from pathlib import Path

import yaml

from qmmd.us_pull import (
    WindowConfig,
    load_config,
    prepare_us_pull,
    submit_us_pull,
    window_centers,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/PRN-anti/us/pull.yaml"


class USPullTests(unittest.TestCase):
    def test_prn_window_plan(self):
        cfg = load_config(CONFIG)
        self.assertEqual(cfg.buffer, 5.5)
        self.assertEqual(cfg.restraint.atoms, (5, 3, 4, 11))
        self.assertEqual(cfg.slurm.job.nodes, 2)
        self.assertEqual(window_centers(cfg.windows), [float(x) for x in range(180, -1, -10)])
        self.assertEqual(int(cfg.md.cntrl["nstlim"]) * float(cfg.md.cntrl["dt"]), 1.0)

    def test_window_range_must_be_exact(self):
        with self.assertRaisesRegex(ValueError, "exactly divisible"):
            window_centers(WindowConfig(180.0, 0.0, 13.0))

    def test_small_partition_defaults_to_two_nodes_and_rejects_one(self):
        data = yaml.safe_load(CONFIG.read_text())
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            config = Path(tmp) / "pull.yaml"
            del data["slurm"]["job"]["nodes"]
            config.write_text(yaml.safe_dump(data, sort_keys=False))
            self.assertEqual(load_config(config).slurm.job.nodes, 2)

            data["slurm"]["job"]["nodes"] = 1
            config.write_text(yaml.safe_dump(data, sort_keys=False))
            with self.assertRaisesRegex(ValueError, "requires at least 2 nodes"):
                load_config(config)

    def test_prepare_serial_chain_in_tmp(self):
        cfg = load_config(CONFIG)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp)
            (base / "prep").mkdir()
            (base / "mdequil").mkdir()
            (base / "prep/solv.parm7").write_text("topology\n")
            (base / "mdequil/equil-npt.rst7").write_text("restart\n")
            (base / "mdequil/equil-npt.out").write_text("Final Performance Info:\n")
            test_cfg = replace(cfg, system_dir=str(base))
            output = prepare_us_pull(test_cfg, CONFIG.read_text(), ROOT)

            windows = sorted(output.glob("window-*/pull"))
            self.assertEqual(len(windows), 19)
            self.assertIn("r2=180.0, r3=180.0", (windows[0] / "dihedral.rst").read_text())
            self.assertIn("r2=0.0, r3=0.0", (windows[-1] / "dihedral.rst").read_text())
            self.assertIn("nstlim=1000", (windows[0] / "pull.in").read_text())

            script = (output / "run.sh").read_text()
            self.assertEqual(script.count("sander -O"), 19)
            self.assertIn('window="$pull_root/window-000/pull"', script)
            self.assertIn('input_restart="$window/pull.rst7"', script)
            self.assertIn(str((base / "mdequil/equil-npt.rst7").resolve()), script)
            self.assertIn("#SBATCH -p small", (output / "slurm.sh").read_text())
            subprocess.run(["bash", "-n", output / "run.sh"], check=True)
            subprocess.run(["bash", "-n", output / "slurm.sh"], check=True)

            submissions = []
            self.assertFalse(
                submit_us_pull(
                    test_cfg,
                    CONFIG.read_text(),
                    ROOT,
                    confirm=lambda _prompt: "no",
                    submitter=submissions.append,
                )
            )
            self.assertEqual(submissions, [])
            self.assertTrue(
                submit_us_pull(
                    test_cfg,
                    CONFIG.read_text(),
                    ROOT,
                    confirm=lambda _prompt: "yes",
                    submitter=submissions.append,
                )
            )
            self.assertEqual(submissions, [output / "slurm.sh"])

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                prepare_us_pull(test_cfg, CONFIG.read_text(), ROOT)


if __name__ == "__main__":
    unittest.main()
