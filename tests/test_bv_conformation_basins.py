import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
from acid_base_BV import _signed_dihedral_deg
from bv_conformation_basins import density_basins, dihedrals, read_torsions, wrap_degrees


class TestConformationBasins(unittest.TestCase):
    def test_geometry_matches_existing_signed_dihedral_and_images(self):
        rng = np.random.default_rng(3)
        points = rng.uniform(0, 10, (30, 3, 4, 3))
        box = np.array([10., 11., 12.])
        actual = dihedrals(points, box)
        expected = np.array([[_signed_dihedral_deg(q, box) for q in frame] for frame in points])
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        shifted = points + rng.integers(-3, 4, points.shape) * box
        np.testing.assert_allclose(wrap_degrees(dihedrals(shifted, box)-actual), 0, atol=1e-10)

    def test_periodic_boundary_is_one_basin(self):
        rng = np.random.default_rng(7)
        points = wrap_degrees(rng.normal([179, -178, 176], 4, (1500, 3)))
        labels, basins, _ = density_basins(points)
        self.assertEqual(len(basins), 1)
        self.assertTrue(np.all(labels == 1))
        self.assertLess(np.linalg.norm(wrap_degrees(np.array(basins[0]['mean_deg'])-[179,-178,176])), 2)
        shifted_labels, _, _ = density_basins(points + np.array([360, -720, 1080]))
        np.testing.assert_array_equal(labels, shifted_labels)

    def test_two_modes_population_and_small_catchment(self):
        rng = np.random.default_rng(11)
        points = np.concatenate([rng.normal([-70,-60,-50], 4, (700,3)),
                                 rng.normal([65,60,55], 4, (290,3)),
                                 rng.normal([0,130,-140], 2, (10,3))])
        labels, basins, density = density_basins(points)
        self.assertEqual(len(basins), 2)
        self.assertAlmostEqual(basins[0]['population'], .7)
        self.assertAlmostEqual(basins[1]['population'], .29)
        self.assertEqual(np.count_nonzero(labels==0), 10)
        self.assertTrue(np.all(density>0))

    def test_complete_frames_and_partial_tail(self):
        frame = '4\n *** AT T= {time:.2f} FSEC\nC 0 0 0\nC 1 0 0\nC 1 1 0\nN 1 1 1\n'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'traject'
            path.write_text(frame.format(time=0)+frame.format(time=10)+'4\n *** AT T= 20.0 FSEC\nC 0 0 0\n')
            t, phi, meta = read_torsions(path, np.array([[1,2,3,4]]), np.ones(3)*10)
            np.testing.assert_allclose(t, [0,.01])
            self.assertEqual(phi.shape, (2,1))
            self.assertTrue(meta['incomplete_tail_ignored'])
            path.write_text(frame.format(time=0)+frame.format(time=0))
            with self.assertRaises(ValueError):read_torsions(path, np.array([[1,2,3,4]]), np.ones(3)*10)

    def test_invalid_parameters(self):
        for kwargs in ({'bin_deg':7},{'bandwidth_deg':0},{'min_population':0},{'merge_deg':-1}):
            with self.assertRaises(ValueError):density_basins(np.zeros((10,3)),**kwargs)

    def test_plot_renders(self):
        import matplotlib
        matplotlib.use('Agg')
        from plot_bv_equil_basins import draw
        points=np.random.default_rng(9).normal(0,5,(100,3))
        labels,basins,_=density_basins(points)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'plot.png'
            draw(path,39,np.arange(100)*.01,points,labels,basins,np.zeros(3),np.array([[-30,30]]*3),
                 dict(bin_deg=10,bandwidth_deg=15,merge_deg=30,min_population=.05))
            self.assertGreater(path.stat().st_size,10000)


if __name__ == '__main__':
    unittest.main()
