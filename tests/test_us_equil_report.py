import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from qmmd.us_equil import load_config
from qmmd.us_equil_report import collect_equil_samples, read_biaspot_samples


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/PRN-anti/us/equil.yaml"


class USEquilReportTests(unittest.TestCase):
    def test_reads_complete_records_and_maps_periodic_branch(self):
        text = """ GAUSSIAN BIAS POTENTIAL: 1
 *** AT T= 100.00 FSEC, THIS RUN'S STEP NO.= 100
     Coordinate = -177.5 Degree
 *** AT T= 200.00 FSEC, THIS RUN'S STEP NO.= 200
"""
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "biaspot"
            path.write_text(text)
            samples = read_biaspot_samples(path, 0, 180.0)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].step, 100)
        self.assertAlmostEqual(samples[0].time_ps, 0.1)
        self.assertAlmostEqual(samples[0].dihedral_raw_deg, -177.5)
        self.assertAlmostEqual(samples[0].dihedral_branch_deg, 182.5)

    def test_collection_accepts_partial_running_window_set(self):
        cfg = load_config(CONFIG)
        yaml_text = CONFIG.read_text()
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp)
            test_cfg = replace(cfg, pull=replace(cfg.pull, system_dir=str(base)))
            stage = base / "us-pull/window-003/equil"
            stage.mkdir(parents=True)
            (stage / "equil_spec.yaml").write_text(yaml_text)
            (stage / "biaspot").write_text(
                " *** AT T= 100.00 FSEC, THIS RUN'S STEP NO.= 100\n"
                " Coordinate = 151.25 Degree\n"
            )
            samples, missing = collect_equil_samples(test_cfg, yaml_text, ROOT)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].window_index, 3)
        self.assertEqual(samples[0].target_deg, 150.0)
        self.assertEqual(len(missing), 18)
        self.assertNotIn(3, missing)


if __name__ == "__main__":
    unittest.main()
