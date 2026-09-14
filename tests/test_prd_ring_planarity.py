import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'plotting'))
from prd_ring_planarity import planar_deviation,ring_metrics,verify_ring


class PlanarityTests(unittest.TestCase):
    def test_angles(self):
        np.testing.assert_allclose(planar_deviation(np.array([0,180,-180,90,-90,175,-10,360])),[0,0,0,90,90,5,10,0])

    def test_planar_and_periodic(self):
        a=np.arange(6)*np.pi/3
        c=np.column_stack([np.cos(a),np.sin(a),np.zeros(6)])
        self.assertAlmostEqual(ring_metrics(c[None],np.ones(3)*10)[2][0],0)
        c[1,2]=.4
        expected=ring_metrics(c[None],np.ones(3)*10)[2][0]
        self.assertGreater(expected,0)
        c[2]+=np.array([10,-10,20])
        self.assertAlmostEqual(ring_metrics(c[None],np.ones(3)*10)[2][0],expected)

    def test_topology(self):
        ring,_=verify_ring(Path(__file__).resolve().parents[1]/'systems/PRD/init/prd.mol2')
        self.assertEqual(ring,[1,2,3,4,5,6])


if __name__=='__main__':unittest.main()
