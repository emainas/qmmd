import tempfile
import unittest
from pathlib import Path

import numpy as np

from qmmd.us_equil_report import EquilSample
from qmmd.us_prod_report import (
    build_density_traces,
    build_pull_target_ladder,
    read_dftb_trajectory_times,
    render_prod_cpptraj_input,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/PRN-anti/us/prod.yaml"


class USProdReportTests(unittest.TestCase):
    def test_reads_complete_dftb_trajectory_times(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            trajectory = Path(tmp) / "traject"
            trajectory.write_text(
                "2\n"
                " *** AT T= 40000.00 FSEC, THIS RUN'S STEP NO.= 40000\n"
                "H 0 0 0\nH 1 0 0\n"
                "2\n"
                " *** AT T= 40010.00 FSEC, THIS RUN'S STEP NO.= 40010\n"
                "H 0 0 0\nH 1 0 0\n"
                "2\n"
                " *** AT T= 40020.00 FSEC, THIS RUN'S STEP NO.= 40020\n"
                "H 0 0 0\n"
            )
            self.assertEqual(
                read_dftb_trajectory_times(trajectory),
                [(40000, 40.0), (40010, 40.01)],
            )

    def test_cpptraj_input_uses_one_based_atoms_and_autoimage(self):
        text = render_prod_cpptraj_input(
            Path("top.par7"),
            Path("trajectory.xyz"),
            (5, 3, 4, 11),
            Path("dihedral.dat"),
            1,
        )
        self.assertIn("fiximagedbonds :1", text)
        self.assertIn("autoimage", text)
        self.assertIn("dihedral production @5 @3 @4 @11", text)

    def test_density_histogram_is_normalized_and_gaussian_is_fitted(self):
        samples = [
            EquilSample(0, 180.0, i * 0.01, i, angle, angle)
            for i, angle in enumerate((178.0, 179.0, 180.0, 181.0, 182.0))
        ]
        trace = build_density_traces(samples)[0]
        area = sum(
            density * width
            for density, width in zip(
                trace.histogram_density_per_deg,
                trace.histogram_edges_deg[1:] - trace.histogram_edges_deg[:-1],
            )
        )
        self.assertAlmostEqual(float(area), 1.0)
        self.assertEqual(len(trace.histogram_density_per_deg), 50)
        self.assertAlmostEqual(trace.fit_mean_deg, 180.0)
        self.assertGreater(trace.fit_sigma_deg, 0.0)
        self.assertEqual(len(trace.gaussian_dihedral_deg), 240)
        gaussian_area = np.trapz(
            trace.gaussian_density_per_deg, trace.gaussian_dihedral_deg
        )
        self.assertAlmostEqual(float(gaussian_area), 1.0, places=4)

    def test_pull_target_ladder_has_vertical_legs_and_horizontal_steps(self):
        samples = [
            EquilSample(0, 180.0, 0.1, 1, 179.0, 179.0),
            EquilSample(0, 180.0, 1.0, 10, 181.0, 181.0),
            EquilSample(1, 170.0, 1.1, 1, 171.0, 171.0),
            EquilSample(1, 170.0, 2.0, 10, 169.0, 169.0),
        ]
        targets, times = build_pull_target_ladder(samples)
        self.assertEqual(targets.tolist(), [180.0, 180.0, 170.0, 170.0, 170.0])
        self.assertEqual(times.tolist(), [0.0, 1.0, 1.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
