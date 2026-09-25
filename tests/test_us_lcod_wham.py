"""Focused checks for equilibration-derived LCOD WHAM inputs."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qmmd.us_lcod_wham import plot_pmf_zero, pmf_features, read_samples, write_wham_inputs
from qmmd.us_wham import dcdftb_wall_to_wham_force


class TestLCODWham(unittest.TestCase):
    def test_read_samples_aligns_centers_and_discards_by_time(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "equil_lcod.csv"
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["window_index", "target_lcod_A", "time_ps", "lcod_A"])
                writer.writerows([
                    [0, -0.1, 0.0, -0.12], [0, -0.1, 1.0, -0.11],
                    [0, -0.1, 2.0, -0.10], [1, 0.0, 1.0, 0.01],
                    [1, 0.0, 2.0, 0.02],
                ])
            result = read_samples(source, [-0.1, 0.0], 1.0)
            self.assertEqual(result[0][:, 0].tolist(), [1.0, 2.0])
            self.assertEqual(result[1][:, 1].tolist(), [0.01, 0.02])

    def test_wham_input_uses_lcod_in_second_column_and_half_k(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            samples = {0: np.array([[0.0, -0.10], [0.04, -0.08]])}
            force = dcdftb_wall_to_wham_force(500.0)
            metadata = write_wham_inputs(output, [-0.1], samples, force)
            np.testing.assert_allclose(np.loadtxt(output / "window-000.dat"), samples[0])
            self.assertAlmostEqual(float(metadata.read_text().split()[2]), 1000.0 / 4.184, places=8)

    def test_missing_window_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "equil_lcod.csv"
            source.write_text("window_index,target_lcod_A,time_ps,lcod_A\n0,-0.1,0,-0.1\n0,-0.1,1,-0.1\n")
            with self.assertRaisesRegex(ValueError, "window-001"):
                read_samples(source, [-0.1, 0.0], 0.0)

    def test_pmf_features_reports_gap_and_directional_barriers(self):
        coordinate = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        pmf = np.array([1.0, 0.0, 4.0, 0.2, 1.0])
        result = pmf_features(coordinate, pmf, (-1.5, -0.5), (0.5, 1.5))
        self.assertEqual(result["negative_minimum"], (-1.0, 0.0))
        self.assertEqual(result["positive_minimum"], (1.0, 0.2))
        self.assertEqual(result["barrier_maximum"], (0.0, 4.0))
        self.assertAlmostEqual(result["gap_positive_minus_negative"], 0.2)
        self.assertAlmostEqual(result["barrier_negative_to_positive"], 4.0)
        self.assertAlmostEqual(result["barrier_positive_to_negative"], 3.8)

    def test_plot_zero_is_lowest_displayed_raw_or_smooth_value(self):
        raw_x = np.array([-2.1, -1.2, 0.0, 1.2, 2.1])
        raw_y = np.array([-1.0, -0.05, 3.5, 0.02, -2.0])
        smooth_x = np.array([-2.1, -1.2, 0.0, 1.2, 2.1])
        smooth_y = np.array([-0.5, 0.0, 3.6, -0.01, -1.0])
        self.assertAlmostEqual(plot_pmf_zero(raw_x, raw_y, smooth_x, smooth_y, (-2.0, 2.0)), -0.05)
