import tempfile
import unittest
from pathlib import Path

import numpy as np

from qmmd.us_wham import (
    WindowSeries,
    circular_block_resample,
    compute_thermodynamics,
    dcdftb_wall_to_wham_force,
    load_config,
    read_wham_result,
    smooth_pmf,
    wham_command,
    write_metadata,
    zero_pmf,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/PRN-anti/us/wham.yaml"


class USWhamTests(unittest.TestCase):
    def test_config_uses_nonperiodic_path_and_correct_force_conversion(self):
        cfg = load_config(CONFIG)
        self.assertFalse(cfg.coordinate.periodic)
        self.assertAlmostEqual(
            dcdftb_wall_to_wham_force(cfg.prod.run.wall.coefficient_kcal_mol_deg2),
            0.02390057361376673,
        )
        command = wham_command(cfg, Path("/bin/wham"), Path("metadata.dat"), Path("result.dat"))
        self.assertNotIn("P", command)

    def test_metadata_order_and_unique_names(self):
        windows = [
            WindowSeries(0, 180.0, np.array([0.0, 0.01]), np.array([179.0, 181.0])),
            WindowSeries(1, 170.0, np.array([0.0, 0.01]), np.array([169.0, 171.0])),
        ]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "metadata.dat"
            write_metadata(path, windows, 0.1)
            self.assertEqual(
                path.read_text().splitlines(),
                ["window-000.dat 180.00000000 0.1", "window-001.dat 170.00000000 0.1"],
            )

    def test_circular_blocks_preserve_sample_count_and_values(self):
        values = np.arange(11, dtype=float)
        sampled = circular_block_resample(values, 4, np.random.default_rng(7))
        self.assertEqual(len(sampled), len(values))
        self.assertTrue(set(sampled).issubset(set(values)))

    def test_result_parser_ignores_window_offsets_and_zeroes_reference(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "result.dat"
            path.write_text(
                "#Coor Free +/- Prob +/-\n"
                "1.0 2.0 0.0 0.1 0.0\n"
                "3.0 4.0 0.0 0.1 0.0\n"
                "5.0 inf -nan 0.0 -nan\n"
                "#Window Free +/-\n#0 1.2 0.0\n"
            )
            grid, pmf = read_wham_result(path)
            shifted = zero_pmf(grid, pmf, (0.0, 2.0))
            np.testing.assert_allclose(grid, [1.0, 3.0, 5.0])
            np.testing.assert_allclose(shifted[:2], [0.0, 2.0])
            self.assertTrue(np.isnan(shifted[2]))

    def test_thermodynamic_minima_barriers_and_populations(self):
        cfg = load_config(CONFIG)
        grid = np.arange(1.0, 180.0, 2.0)
        pmf = 10.0 * np.sin(np.pi * grid / 180.0) ** 2 + 2.0 * grid / 180.0
        result = compute_thermodynamics(grid, pmf, cfg)
        self.assertEqual(result["syn_min_dihedral"][0], 1.0)
        self.assertEqual(result["anti_min_dihedral"][0], 179.0)
        self.assertGreater(result["barrier_syn_to_anti"][0], result["barrier_anti_to_syn"][0])
        self.assertAlmostEqual(
            result["syn_molar_fraction"][0] + result["anti_molar_fraction"][0], 1.0
        )

    def test_smoothing_preserves_linear_shape_and_reduces_bin_noise(self):
        linear = np.linspace(0.0, 8.0, 9)
        np.testing.assert_allclose(smooth_pmf(linear, 10.0), linear, atol=1.0e-12)
        noisy = linear + 0.4 * np.where(np.arange(linear.size) % 2, 1.0, -1.0)
        smoothed = smooth_pmf(noisy, 10.0)
        self.assertLess(np.linalg.norm(np.diff(smoothed, n=2)), np.linalg.norm(np.diff(noisy, n=2)))


if __name__ == "__main__":
    unittest.main()
