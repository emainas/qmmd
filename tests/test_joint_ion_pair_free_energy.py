from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PLOTTING = Path(__file__).parents[1] / "plotting"
sys.path.insert(0, str(PLOTTING))
SPEC = importlib.util.spec_from_file_location(
    "joint_ion_pair_free_energy", PLOTTING / "joint_ion_pair_free_energy.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load joint ion-pair module")
joint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = joint
SPEC.loader.exec_module(joint)


class JointProbabilityTests(unittest.TestCase):
    def test_tiwary_parrinello_weights_use_equation_13_kernel(self) -> None:
        hills = joint.MetadynamicsHills(
            times_ps=np.array([1.0]),
            centers=np.array([0.25]),
            heights_kcal_mol=np.array([1.5]),
            widths=np.array([0.1]),
        )
        bias, offsets, log_weights, weights, effective_size = (
            joint.tiwary_parrinello_weights(
                np.array([2.0, 2.0]),
                np.array([0.25, 0.75]),
                hills,
                temperature_K=300.0,
                s_min=0.0,
                s_max=1.0,
            )
        )
        # The localized eq-13 kernel makes c(t) depend on the instantaneous s.
        self.assertGreater(offsets[0], offsets[1])
        beta = 1.0 / (joint.KB_KCAL_MOL_K * 300.0)
        expected_log_ratio = beta * (
            (bias[0] - offsets[0]) - (bias[1] - offsets[1])
        )
        self.assertAlmostEqual(log_weights[0] - log_weights[1], expected_log_ratio)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        self.assertGreater(effective_size, 1.0)
        self.assertLessEqual(effective_size, 2.0)

    def test_equation_13_zero_bias_has_zero_offset(self) -> None:
        # With V=0 and a domain spanning the Gaussian kernel, eq 13 gives c=0.
        hills = joint.MetadynamicsHills(
            times_ps=np.array([1.0]),
            centers=np.array([0.0]),
            heights_kcal_mol=np.array([0.0]),
            widths=np.array([0.1]),
        )
        bias, offsets, log_weights, weights, _ = joint.tiwary_parrinello_weights(
            np.array([2.0]),
            np.array([0.0]),
            hills,
            temperature_K=300.0,
            s_min=-1.0,
            s_max=1.0,
            integration_points=2001,
        )
        self.assertAlmostEqual(offsets[0], bias[0], places=3)
        self.assertAlmostEqual(log_weights[0], 0.0, places=3)
        self.assertAlmostEqual(weights[0], 1.0)

    def test_conditional_density_normalizes_for_each_s(self) -> None:
        rng = np.random.default_rng(8)
        samples_s = rng.uniform(0.0, 1.0, 3000)
        samples_d = 2.6 + 3.0 * (1.0 - samples_s) + rng.normal(0.0, 0.15, 3000)
        s_grid = np.linspace(0.0, 1.0, 81)
        d_grid = np.linspace(2.0, 6.5, 101)
        ks = joint.gaussian_kernel_matrix(s_grid, samples_s, 0.05)
        kd = joint.gaussian_kernel_matrix(d_grid, samples_d, 0.18)
        sampled_joint = (ks @ kd.T) / samples_s.size
        sampled_s = joint.TRAPEZOID(sampled_joint, d_grid, axis=1)
        conditional = sampled_joint / sampled_s[:, None]
        normalization = joint.TRAPEZOID(conditional, d_grid, axis=1)
        np.testing.assert_allclose(normalization, 1.0, atol=1.0e-10)

    def test_radial_jacobian_is_applied_only_to_distance_axis(self) -> None:
        s_grid = np.linspace(0.0, 1.0, 5)
        d_grid = np.linspace(2.0, 5.0, 7)
        joint_probability = np.ones((s_grid.size, d_grid.size))
        corrected = joint_probability / d_grid[None, :] ** 2
        np.testing.assert_allclose(corrected[0], corrected[-1])
        self.assertGreater(float(corrected[0, 0]), float(corrected[0, -1]))


if __name__ == "__main__":
    unittest.main()
