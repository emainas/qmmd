"""Live-file tail handling and periodic torsion regression checks."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("prn_anti_dih", Path(__file__).resolve().parents[1] / "plotting/prn_anti_dih.py")
dih = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dih)


class DihedralTests(unittest.TestCase):
    def test_circular_std_matches_scipy(self):
        from scipy.stats import circstd
        for values in ([359., 1.], [170., 180., 190.], [10., 10.], [0., 30., 180., np.nan]):
            values = np.asarray(values)
            self.assertAlmostEqual(dih.circular_std_deg(values),
                                   circstd(values, high=360, low=0, nan_policy="omit"), places=6)
        self.assertTrue(np.isnan(dih.circular_std_deg(np.array([np.nan]))))

    def test_circular_mean_and_seam(self):
        self.assertAlmostEqual(dih.circular_mean_deg(np.array([359., 1., np.nan])), 0)
        self.assertAlmostEqual(dih.circular_mean_deg(np.array([179., -179.])), 180)
        self.assertTrue(np.isnan(dih.circular_mean_deg(np.array([0., 180.]))))
        times, angles = dih.wrapped_plot_series(np.array([0., 1.]), np.array([-1., 1.]))
        np.testing.assert_allclose(times, [0, np.nan, 1], equal_nan=True)
        np.testing.assert_allclose(angles, [359, np.nan, 1], equal_nan=True)

    def test_syn_anti_and_periodic_images(self):
        box = np.full(3, 10.)
        points = np.array([[0., 1, 0], [0, 0, 0], [1, 0, 0], [1, -1, 0]])
        self.assertAlmostEqual(abs(dih.dihedral(points, box)), 180)
        points += np.array([[10, 0, 0], [0, -10, 0], [0, 0, 10], [-10, 10, 0]])
        self.assertAlmostEqual(abs(dih.dihedral(points, box)), 180)
        points[3] = [1, 1, 0]
        self.assertAlmostEqual(dih.dihedral(points, box), 0)

    def test_partial_frame_never_enters_series(self):
        complete = b"4\n *** AT T= 10.00 FSEC\nO 0 1 0\nC 0 0 0\nO 1 0 0\nH 1 -1 0\n"
        next_frame = complete.replace(b"10.00", b"20.00")
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            path = Path(folder) / "traject"
            for tail in (b"", next_frame[:1], next_frame[:35], next_frame[:-1]):
                path.write_bytes(complete + tail)
                data = dih.read_series(path, [1, 2, 3, 4], np.full(3, 10.))
                self.assertEqual(data.shape, (1, 4))
                self.assertAlmostEqual(data[0, 0], .01)
                self.assertAlmostEqual(data[0, 2], 180)


if __name__ == "__main__":
    unittest.main()
