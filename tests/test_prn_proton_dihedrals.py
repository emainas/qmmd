import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
from prn_proton_dihedrals import bond_dihedrals, persistent_ids


class ProtonDihedralTests(unittest.TestCase):
    def fixture(self):
        symbols = ["C", "C", "C", "O", "O", "H", "H", "H", "H", "H", "H", "H", "O"]
        frame = np.ones((13, 3))*7
        frame[0], frame[1] = [6, 6, 6], [7, 6, 6]
        frame[2], frame[3], frame[4] = [1.3, 0, 0], [0, 0, 0], [2, 1.2, 0]
        frame[10], frame[11], frame[12] = [-.5, .9, .1], [3., 1.2, .2], [3.5, 1.2, 0]
        coords = np.repeat(frame[None, :, :], 8, axis=0)
        coords[4:, 10] = [3.5, .3, .1]  # Old H moves to solvent O13.
        coords[4:, 11] = [1.6, 2., .2]  # A different H attaches to O5.
        return np.arange(8)*.02, coords, symbols, np.ones(3)*20

    def test_old_disappears_new_different_h_appears_and_pbc(self):
        t, coords, symbols, box = self.fixture()
        data = bond_dihedrals(t, coords, symbols, box, [5, 3, 4, 11], 5)
        self.assertTrue(np.isfinite(data["old_dihedral_deg"][:4]).all())
        self.assertTrue(np.isnan(data["old_dihedral_deg"][4:]).all())
        self.assertTrue(np.isnan(data["new_dihedral_deg"][:4]).all())
        np.testing.assert_allclose(data["new_hydrogen_id"][4:], 12)
        coords[:, 4] += box
        shifted = bond_dihedrals(t, coords, symbols, box, [5, 3, 4, 11], 5)
        np.testing.assert_allclose(data["new_dihedral_deg"], shifted["new_dihedral_deg"], equal_nan=True)

    def test_shared_hydrogen_is_not_assigned(self):
        t, coords, symbols, box = self.fixture()
        coords[:, 11] = [2.75, 1.2, 0]  # Equidistant between O5 and O13.
        data = bond_dihedrals(t, coords, symbols, box, [5, 3, 4, 11], 5)
        self.assertTrue(np.isnan(data["new_hydrogen_id"]).all())

    def test_new_proton_on_original_oxygen_excludes_original_h(self):
        t, coords, symbols, box = self.fixture()
        coords[4:, 11] = [-.5, -.9, .1]
        data = bond_dihedrals(t, coords, symbols, box, [5, 3, 4, 11], 5)
        self.assertTrue(np.isnan(data['O4_acquired_hydrogen_id'][:4]).all())
        np.testing.assert_allclose(data['O4_acquired_hydrogen_id'][4:], 12)
        self.assertTrue(np.isnan(data['O5_acquired_hydrogen_id']).all())
        self.assertTrue((data['O4_acquired_bond_distance_A'][4:] < 1.4).all())

    def test_persistence_and_unknown_gap(self):
        ids = np.array([11., 11., 11., 11., np.nan, 12., 12.])
        values = persistent_ids(np.arange(7)*.02, ids, .05)
        np.testing.assert_allclose(values, [11, 11, 11, 11, np.nan, np.nan, np.nan], equal_nan=True)
        values = persistent_ids(np.array([0., .02, .2, .22]), np.ones(4)*11, .05)
        self.assertTrue(np.isnan(values).all())


if __name__ == "__main__":
    unittest.main()
