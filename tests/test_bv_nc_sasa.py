import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'plotting'))
from bv_nc_sasa import atom_sasa, nc_sasa, sphere_points


class TestSASA(unittest.TestCase):
    def test_isolated_and_buried(self):
        pts = sphere_points(1920)
        area = atom_sasa(np.empty((0, 3)), np.array([]), 1.55, 1.4, pts)
        self.assertAlmostEqual(area, 4*np.pi*2.95**2)
        self.assertEqual(atom_sasa(np.zeros((1, 3)), np.array([4.]), 1.55, 1.4, pts), 0)

    def test_equal_sphere_cap(self):
        # Equal expanded radii R=3, center separation d=3: one quarter blocked.
        area = atom_sasa(np.array([[3., 0, 0]]), np.array([1.6]), 1.6, 1.4, sphere_points(20000))
        self.assertAlmostEqual(area / (4*np.pi*9), .75, delta=.002)

    def test_h_ownership_and_periodicity(self):
        pts = sphere_points(1920)
        xyz = np.array([[0.,0,0], [1.,0,0], [8.,0,0]])
        elements = np.array(['N','H','O']); box = np.array([20.,20.,20.])
        initial = nc_sasa(xyz, elements, box, 0, 2, 1.4, pts, 1.3)
        self.assertEqual(initial[2], 1)
        self.assertLess(initial[0], initial[1])
        shifted = xyz.copy(); shifted[1] += box
        np.testing.assert_allclose(initial, nc_sasa(shifted, elements, box, 0, 2, 1.4, pts, 1.3))
        xyz[1] = [7.,0,0]
        detached = nc_sasa(xyz, elements, box, 0, 2, 1.4, pts, 1.3)
        self.assertEqual(detached[2], 0)
        self.assertEqual(detached[0], detached[1])


if __name__ == '__main__':
    unittest.main()
