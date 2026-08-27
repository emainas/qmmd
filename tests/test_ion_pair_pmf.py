from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "plotting" / "ion_pair_pmf.py"
SPEC = importlib.util.spec_from_file_location("ion_pair_pmf", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
ion_pair_pmf = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ion_pair_pmf
SPEC.loader.exec_module(ion_pair_pmf)


class IonPairPmfTests(unittest.TestCase):
    def test_spherical_volume_samples_give_flat_binned_pmf(self) -> None:
        rng = np.random.default_rng(11)
        radial_min = 2.0
        radial_max = 8.0
        uniform_volume = rng.uniform(radial_min**3, radial_max**3, 80_000)
        distances = np.cbrt(uniform_volume)
        times = np.arange(distances.size, dtype=float)
        profile = ion_pair_pmf.calculate_profile(
            "uniform",
            Path("uniform.csv"),
            times,
            distances,
            "N10_Odefect_distance_A",
            radial_min_A=radial_min,
            radial_max_A=radial_max,
            bin_width_A=0.25,
            kde_grid_points=50,
        )
        finite = np.isfinite(profile.binned_pmf_kcal_mol)
        self.assertLess(float(np.std(profile.binned_pmf_kcal_mol[finite])), 0.025)

    def test_two_radial_basins_recover_two_minima_and_barrier(self) -> None:
        rng = np.random.default_rng(7)
        distances = np.concatenate(
            (
                rng.normal(2.65, 0.12, 5000),
                rng.uniform(2.9, 4.3, 500),
                rng.normal(4.55, 0.18, 6000),
            )
        )
        rng.shuffle(distances)
        profile = ion_pair_pmf.calculate_profile(
            "two-well",
            Path("two-well.csv"),
            np.arange(distances.size, dtype=float),
            distances,
            "N10_Odefect_distance_A",
            radial_min_A=2.0,
            radial_max_A=6.0,
            bin_width_A=0.1,
            kde_bandwidth_factor=0.13,
            kde_grid_points=300,
            minimum_bin_count=2,
        )
        minima = np.asarray([profile.grid_A[i] for i in profile.minima_indices])
        maxima = np.asarray([profile.grid_A[i] for i in profile.maxima_indices])
        self.assertTrue(np.any(np.abs(minima - 2.65) < 0.15))
        self.assertTrue(np.any(np.abs(minima - 4.55) < 0.15))
        self.assertTrue(np.any((maxima > 2.8) & (maxima < 4.4)))

    def test_time_window_and_missing_values_are_excluded(self) -> None:
        times = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        distances = np.asarray([2.5, np.nan, 3.0, 3.5, 10.0])
        profile = ion_pair_pmf.calculate_profile(
            "window",
            Path("window.csv"),
            times,
            distances,
            "N10_Odefect_distance_A",
            time_min_ps=1.0,
            time_max_ps=3.0,
        )
        np.testing.assert_allclose(profile.times_ps, [2.0, 3.0])
        np.testing.assert_allclose(profile.distances_A, [3.0, 3.5])

    def test_empty_requested_window_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "No finite defect distances"):
            ion_pair_pmf.calculate_profile(
                "empty",
                Path("empty.csv"),
                np.asarray([0.0]),
                np.asarray([np.nan]),
                "N10_Odefect_distance_A",
            )

    def test_modeled_wall_is_only_added_for_contact_sampling(self) -> None:
        contact = ion_pair_pmf.calculate_profile(
            "contact",
            Path("contact.csv"),
            np.arange(4, dtype=float),
            np.asarray([2.49, 2.55, 2.60, 2.70]),
            "N10_Odefect_distance_A",
        )
        distant = ion_pair_pmf.calculate_profile(
            "distant",
            Path("distant.csv"),
            np.arange(4, dtype=float),
            np.asarray([4.5, 4.6, 4.7, 4.8]),
            "N10_Odefect_distance_A",
        )
        self.assertGreater(contact.wall_grid_A.size, 0)
        self.assertTrue(np.all(np.isfinite(contact.wall_pmf_kcal_mol)))
        self.assertGreater(
            float(contact.wall_pmf_kcal_mol[0]),
            float(contact.wall_pmf_kcal_mol[-1]),
        )
        self.assertEqual(distant.wall_grid_A.size, 0)


if __name__ == "__main__":
    unittest.main()
