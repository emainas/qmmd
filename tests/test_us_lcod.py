import unittest
import tempfile
from pathlib import Path

from qmmd.us_lcod import lcod_centers, load_equil_config, load_pull_config, render_lcod_metacv
from qmmd.us_equil import render_slurm_sh
from qmmd.us_lcod_equil_report import (
    equilibration_duration_ps,
    build_density_traces,
    read_pull_seeds,
    read_complete_trajectory_times,
    read_cpptraj_lcod,
    render_lcod_cpptraj_input,
    LCODEquilSample,
)


ROOT = Path(__file__).resolve().parents[1]
PULL = ROOT / "configs/CPP/us/pull.yaml"
EQUIL = ROOT / "configs/CPP/us/equil.yaml"


class USLCODTests(unittest.TestCase):
    def test_dpp_run5_grid_atoms_and_label(self):
        cfg = load_pull_config(ROOT / "configs/DPP/us/pull.yaml")
        centers = lcod_centers(cfg.windows)
        self.assertEqual(len(centers), 41)
        self.assertAlmostEqual(centers[0], -2.0)
        self.assertAlmostEqual(centers[-1], 2.0)
        self.assertEqual(cfg.cv.atoms, (43, 32, 31, 32))
        self.assertIn("ND43", cfg.cv.label)

    def test_app_equil_matches_pull_and_cpp_restraint_setup(self):
        cfg = load_equil_config(ROOT / "configs/APP/us/equil.yaml")
        self.assertEqual(cfg.cv.atoms, (18, 19, 1, 19))
        self.assertEqual(len(lcod_centers(cfg.pull.windows)), 41)
        self.assertEqual(
            render_lcod_metacv(cfg, 0.0),
            "BONDDISTANCEDIFFERENCE 0.1 18 19 1 19\n\n"
            "L 1 500 0 2\nU 1 500 0 2\n",
        )
        self.assertIn("#SBATCH --qos=highpri", render_slurm_sh(cfg, 0))
        self.assertIn("#SBATCH --ntasks=48", render_slurm_sh(cfg, 0))

    def test_app_run5_grid_atoms_and_label(self):
        cfg = load_pull_config(ROOT / "configs/APP/us/pull.yaml")
        centers = lcod_centers(cfg.windows)
        self.assertEqual(len(centers), 41)
        self.assertAlmostEqual(centers[0], -2.0)
        self.assertAlmostEqual(centers[-1], 2.0)
        self.assertEqual(cfg.cv.atoms, (18, 19, 1, 19))
        self.assertIn("NB18", cfg.cv.label)

    def test_cpp_grid_and_atoms(self):
        cfg = load_pull_config(PULL)
        centers = lcod_centers(cfg.windows)
        self.assertEqual(len(centers), 44)
        self.assertAlmostEqual(centers[0], -2.0)
        self.assertAlmostEqual(centers[-1], 2.3)
        self.assertEqual(cfg.cv.atoms, (19, 20, 31, 20))

    def test_metacv_uses_literal_500_coefficient(self):
        cfg = load_equil_config(EQUIL)
        self.assertEqual(
            render_lcod_metacv(cfg, -0.3),
            "BONDDISTANCEDIFFERENCE 0.1 19 20 31 20\n\n"
            "L 1 500 -0.3 2\nU 1 500 -0.3 2\n",
        )

    def test_trajectory_reader_ignores_incomplete_tail(self):
        content = """2
 *** AT T= 0.00 FSEC, THIS RUN'S STEP NO.= 0
 N 0 0 0
 H 1 0 0
2
 *** AT T= 40.00 FSEC, THIS RUN'S STEP NO.= 40
 N 0 0 0
 H 1 0 0
2
 *** AT T= 80.00 FSEC, THIS RUN'S STEP NO.= 80
 N 0 0 0
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traject"
            path.write_text(content)
            times = read_complete_trajectory_times(path)
        self.assertEqual(times, [(0, 0.0), (40, 0.04)])

    def test_cpptraj_input_calculates_lcod_with_box(self):
        import numpy as np

        script = render_lcod_cpptraj_input(
            Path("/tmp/topology.parm7"), Path("/tmp/traject"), 2,
            np.diag([17.869, 21.743, 22.474]), (19, 20, 31, 20)
        )
        self.assertIn("trajin /tmp/traject 1 2", script)
        self.assertIn("box x 17.8690000000 y 21.7430000000 z 22.4740000000", script)
        self.assertIn("distance NBH @19 @20", script)
        self.assertIn("distance NCH @31 @20", script)
        self.assertIn("calc LCOD = NBH - NCH", script)
        self.assertIn("writedata lcod.dat LCOD prec 18.10", script)

    def test_cpptraj_lcod_frame_count_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lcod.dat"
            path.write_text("#Frame LCOD\n1 -1.8\n2 -1.7\n")
            self.assertEqual(read_cpptraj_lcod(path, 2).tolist(), [-1.8, -1.7])
            with self.assertRaises(ValueError):
                read_cpptraj_lcod(path, 3)

    def test_equilibration_duration_comes_from_dftb_header(self):
        cfg = load_equil_config(EQUIL)
        self.assertAlmostEqual(equilibration_duration_ps(cfg.dftb.header_lines), 40.0)

    def test_handmade_seed_table_matches_all_windows(self):
        cfg = load_equil_config(EQUIL)
        root = ROOT / "systems/CPP/solv_4.0/us-lcod"
        seeds = read_pull_seeds(root, lcod_centers(cfg.pull.windows), cfg.pull_yaml.read_text())
        self.assertEqual(len(seeds), 44)
        self.assertAlmostEqual(seeds[0].target_angstrom, -2.0)
        self.assertAlmostEqual(seeds[-1].target_angstrom, 2.3)

    def test_density_traces_are_normalized(self):
        samples = [
            LCODEquilSample(0, 0.0, index * 0.08, index, value)
            for index, value in enumerate((-0.06, -0.02, 0.01, 0.04, 0.07))
        ]
        trace = build_density_traces(samples, bins=5)[0]
        spacing = trace.histogram_centers_angstrom[1] - trace.histogram_centers_angstrom[0]
        self.assertAlmostEqual(sum(trace.histogram_density_per_angstrom) * spacing, 1.0)
        self.assertAlmostEqual(trace.empirical_mean_angstrom, 0.008)


if __name__ == "__main__":
    unittest.main()
