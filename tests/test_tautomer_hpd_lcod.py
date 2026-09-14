import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'plotting'))
from tautomer_HPD_lcod import side_minima, signed_impropers, joined_meta_times, circular_stats_deg
from prd_ring_planarity import planar_deviation, verify_ring


class HPDLcodTests(unittest.TestCase):
    def test_circular_stats_seam(self):
        mean, std = circular_stats_deg(np.array([179.,-179.,np.nan]))
        self.assertAlmostEqual(abs(mean),180.)
        self.assertAlmostEqual(std,1.,places=3)
        mean, std = circular_stats_deg(np.array([15.,15.]))
        self.assertAlmostEqual(mean,15.)
        self.assertAlmostEqual(std,0.,places=5)

    def test_joined_times(self):
        meta = np.array([20.,20.005,40.])
        joined = joined_meta_times(np.array([0.,40.]),meta)
        np.testing.assert_allclose(joined,[40.,40.005,60.])
        np.testing.assert_allclose(np.diff(joined),np.diff(meta))
        np.testing.assert_array_equal(meta,[20.,20.005,40.])

    def test_improper_side_rotation_and_periodicity(self):
        coords = np.zeros((12,3))
        theta = np.arange(6)*np.pi/3
        coords[:6,:2] = np.column_stack([np.cos(theta),np.sin(theta)])
        coords[7] = 1.8*coords[0]
        coords[6] = 1.8*coords[1]
        box = np.full(3,10.)
        _, flat = signed_impropers(coords[None],box)
        np.testing.assert_allclose(flat,0,atol=1e-12)
        coords[[7,6],2] = .2
        _, up = signed_impropers(coords[None],box)
        self.assertTrue(np.all(up > 0))
        coords[6,2] = -.2
        _, opposite = signed_impropers(coords[None],box)
        self.assertGreater(opposite[0,0],0)
        self.assertLess(opposite[0,1],0)
        # Proper rotation, translation, and atomwise wrapping preserve signed sides.
        rotated = coords[:,[2,0,1]] + 4.8
        rotated %= box
        _, transformed = signed_impropers(rotated[None],box)
        np.testing.assert_allclose(transformed,opposite,atol=1e-10)

    def test_side_minima_sign_and_split(self):
        block = np.array([[-2,3],[-1,1],[0,-100],[1,5],[2,4]])
        self.assertEqual(side_minima(block,0), (1,4,3))

    def test_missing_side(self):
        with self.assertRaises(ValueError):
            side_minima(np.array([[-2,0],[-1,1]]),0)

    def test_planarity_definition(self):
        np.testing.assert_allclose(planar_deviation(np.array([-180,-90,-10,0,10,90,180])),
                                   [0,90,10,0,10,90,0])

    def test_hpd_ring(self):
        root = Path(__file__).resolve().parents[1]
        ring, names = verify_ring(root/'systems/HPD/init/hpd.mol2')
        self.assertEqual(ring,[1,2,3,4,5,6])
        self.assertEqual(names[7],'O2')


if __name__ == '__main__':
    unittest.main()
