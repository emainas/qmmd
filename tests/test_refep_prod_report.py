import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from qmmd.refep_prod_report import (
    EnergyGrid,
    analyze_energy_grid,
    bar_free_energy,
    generate_refep_report,
    load_completed_energy_grid,
    load_refep_report_config,
    orient_grid_for_reporting,
    parse_rem_log,
    reporting_labels,
    solve_mbar,
    ti_charge_derivative,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "MEA" / "refep" / "prod.yaml"


class RefepProdReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_refep_report_config(CONFIG)

    def test_config_and_completed_grid(self):
        grid = load_completed_energy_grid(self.cfg)
        self.assertEqual(grid.energies.shape, (16, 16, 50))
        self.assertAlmostEqual(grid.time_ps[-1], 500.0)
        self.assertEqual(self.cfg.bootstrap_block_frames, 5)

    def test_bar_symmetric_work(self):
        forward = np.asarray([0.7, 1.0, 1.3])
        reverse = -forward
        self.assertAlmostEqual(bar_free_energy(forward, reverse), 1.0, places=8)

    def test_ti_recovers_quadratic_charge_path(self):
        lambdas = np.linspace(0.0, 1.0, 5)
        frames = 12
        energies = np.empty((5, 5, frames))
        offsets = np.linspace(-2.0, 2.0, frames)
        for sample in range(5):
            for evaluation, lam in enumerate(lambdas):
                energies[sample, evaluation] = 3.0 * lam**2 + 2.0 * lam + offsets
        result = ti_charge_derivative(
            EnergyGrid(lambdas, energies, np.arange(1, frames + 1)), 3
        )
        means, _, _, rms, total, legacy = result
        np.testing.assert_allclose(means, 6.0 * lambdas + 2.0, atol=1.0e-12)
        self.assertLess(rms.max(), 1.0e-12)
        self.assertAlmostEqual(total, 5.0, places=12)
        self.assertAlmostEqual(legacy, 5.0, places=12)

    def test_protonated_reference_reverses_report_direction(self):
        energies = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
        grid = EnergyGrid(np.asarray([0.0, 1.0]), energies, np.arange(1, 4))
        oriented = orient_grid_for_reporting(self.cfg, grid)
        self.assertEqual(reporting_labels(self.cfg), ("deprotonated", "protonated"))
        np.testing.assert_allclose(oriented.energies, energies[::-1, ::-1, :])
        np.testing.assert_allclose(oriented.lambdas, [0.0, 1.0])

    def test_mbar_and_real_analysis(self):
        grid = load_completed_energy_grid(self.cfg)
        analysis = analyze_energy_grid(self.cfg, grid)
        self.assertAlmostEqual(analysis.totals["MBAR"], 18.6970, places=3)
        self.assertAlmostEqual(analysis.totals["TI charge derivative"], 18.5940, places=3)
        self.assertTrue(np.allclose(analysis.mbar_overlap.sum(axis=1), 1.0))
        free, overlap = solve_mbar(
            grid.energies,
            1.0 / (0.00198720425864083 * self.cfg.temperature_k),
        )
        self.assertEqual(free.shape, (16,))
        self.assertEqual(overlap.shape, (16, 16))

    def test_replica_diagnostics(self):
        diagnostic = parse_rem_log(self.cfg)
        self.assertEqual(diagnostic.walker_positions.shape, (500, 16))
        self.assertGreaterEqual(int(diagnostic.reached_both_endpoints.sum()), 1)
        self.assertEqual(len(diagnostic.edge_pairs), 16)

    def test_full_report_writes_figures_and_aligned_data(self):
        small = replace(self.cfg, bootstrap_samples=3)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            destination = Path(tmp) / "report"
            generate_refep_report(small, destination)
            for name in (
                "free-energy-summary.png",
                "neighbor-work-overlap.png",
                "mbar-overlap.png",
                "replica-exchange-health.png",
                "free-energy-summary.csv",
                "energy-matrix.csv",
                "summary.yaml",
            ):
                self.assertTrue((destination / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
