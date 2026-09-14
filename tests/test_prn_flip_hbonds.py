"""Check H11 contact geometry, including solute and periodic solvent acceptors."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
from build_prn_flip_vmd import proton_hbonds


class ProtonHBondTests(unittest.TestCase):
    def setUp(self):
        self.xyz = np.zeros((12, 3))
        self.symbols = ["C"] * 12
        for i in (3, 4, 11):
            self.symbols[i] = "O"
        self.symbols[10] = "H"
        self.xyz[3] = [-1, 0, 0]
        self.xyz[4] = [2, 0, 0]
        self.xyz[11] = [12, .1, 0]  # Periodic image of a solvent acceptor.

    def contacts(self):
        return proton_hbonds(self.xyz, self.symbols, np.full(3, 10.), 1.3, 2.5, 135)

    def test_both_acceptor_types_and_pbc(self):
        contacts = self.contacts()
        self.assertEqual([r[1] for r in contacts], [4, 11])
        self.assertTrue(all(r[0] == 3 for r in contacts))
        self.assertAlmostEqual(contacts[0][4], 180)
        self.assertAlmostEqual(contacts[1][3], np.sqrt(4.01))

    def test_distance_angle_and_donor_required(self):
        self.xyz[4] = [0, 2, 0]  # 90-degree angle.
        self.xyz[11] = [2.6, 0, 0]  # Too far.
        self.assertEqual(self.contacts(), [])

    def test_heavy_atom_cutoff_is_not_hydrogen_cutoff(self):
        # H-A 2.0 and D-A 3.0 passes, but H-A ~2.0025 and D-A >3 does not.
        contacts = proton_hbonds(self.xyz, self.symbols, np.full(3, 10.), 1.3, None, 135, 3.0)
        self.assertEqual([r[1] for r in contacts], [4])
        self.xyz[4] = [2, 0, 0]
        self.xyz[3] = [-1.4, 0, 0]  # No covalent donor within threshold.
        self.assertEqual(self.contacts(), [])


if __name__ == "__main__":
    unittest.main()
