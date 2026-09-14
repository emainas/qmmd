import argparse
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
from prn_acid_base_analysis import xyz_frames, mulliken_frames, aligned_fes, screen, competitor_geometry, save_base, manual_event_times
from acid_base_PRN import _signed_dihedral_deg, _minimum_image, load_pka_series


class PrnAcidBaseTests(unittest.TestCase):
    def test_manual_diffusion_does_not_require_defect_assignment(self):
        args = argparse.Namespace(deprotonated_s_max=.05, persistence_ps=.05)
        data = dict(times=np.arange(11)*.02+20, coordination=np.r_[.9,.9,np.ones(5)*.01,np.ones(4)*.4])
        dep, diffuse = manual_event_times(data, args, 20.16)
        self.assertAlmostEqual(dep, 20.04)
        self.assertAlmostEqual(diffuse, 20.16)
        with self.assertRaises(ValueError):
            manual_event_times(data, args, 36.1)
        no_dep, manual = manual_event_times(data, args, 20.02)
        self.assertTrue(np.isnan(no_dep))
        self.assertAlmostEqual(manual, 20.02)

    def test_syn_dihedral_csv_center_and_truncation(self):
        args = argparse.Namespace(acid_oxygen=4, dihedral_center=0.)
        data = dict(times=np.arange(3.), coordination=np.ones(3), defect=np.ones(3)*12,
                    distance=np.ones(3), state_distance=np.ones(3), matched=np.ones(3),
                    phi=np.array([350., 10., 20.]), deprot=2.)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "aligned.csv"
            save_base(path, data, args, 2.)
            values = np.genfromtxt(path, delimiter=",", names=True)
        np.testing.assert_allclose(values["dihedral_plotted_deg"], [-10., 10., np.nan])
        np.testing.assert_allclose(values["dihedral_raw_deg"], [350., 10., 20.])

    def test_prn_pka_reverses_shared_minima_difference(self):
        from plot_pka_grid import deltaf, PKA_FACTOR
        block = np.array([[0., 4.], [.05, 5.], [.95, 2.], [1., 1.]])
        with patch("prn_acid_base_analysis.aligned_fes", return_value=(np.array([2.]), [block])):
            times, values = load_pka_series(Path("unused"), temperature=300.)
        self.assertEqual(times[0], 2.)
        self.assertAlmostEqual(values[0], -3 / (PKA_FACTOR * 300))
        self.assertAlmostEqual(values[0], -deltaf(block, 0, 1, .1, 0, 1.25) / (PKA_FACTOR * 300))

    def test_competitor_direct_wire_missing_defect_and_all_h(self):
        args = argparse.Namespace(competitor_oxygen=1, solute_atoms=1,
                                  competitor_refdist=1.6, covalent_cutoff=1.3,
                                  hydrogen_acceptor_cutoff=2.5, angle_cutoff=135.,
                                  competitor_max_bridging_waters=3)
        coords = np.array([[0., 0., 0.], [1., 0., 0.], [2.7, 0., 0.]])
        data = dict(symbols=["O", "H", "O"], times=np.array([0., .02]),
                    coords=np.array([coords, coords]), box=np.ones(3)*10,
                    defect=np.array([3., np.nan]), oxygen_ids=np.array([3]),
                    competitor_charge=np.array([-.7, -.8]))
        values = competitor_geometry(data, args, .02)
        np.testing.assert_allclose(values[:, :2], [[0., -.7], [.02, -.8]])
        self.assertEqual(values[0, 2], 0.)
        self.assertTrue(np.isnan(values[1, 2]))
        np.testing.assert_allclose(values[:, 3], 1/(1+(1/1.6)**6))
        self.assertEqual(len(competitor_geometry(data, args, 0)), 1)

    def test_complete_xyz_only(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "traject"
            path.write_text("1\n*** AT T= 1000 FSEC\nH 0 0 0\n"
                            "1\n*** AT T= 1020 FSEC\nH 1 0")
            frames = list(xyz_frames(path))
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0][0], 1.)

    def test_complete_mulliken_orbitals_only(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "mulliken"
            path.write_text("3 2\n*** AT T= 1000 FSEC\n1 O s .2\n1 O p -.8\n2 H s .4\n"
                            "3 2\n*** AT T= 1020 FSEC\n1 O s .2\n")
            times, charges = mulliken_frames(path, ["O", "H"])
            np.testing.assert_allclose(times, [1.])
            np.testing.assert_allclose(charges, [[-.6, .4]])

    def test_fes_uses_hill_ids_not_rows_and_ignores_partial_tail(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            folder = Path(tmp)
            (folder / "biaspot").write_text("".join(
                f"GAUSSIAN BIAS POTENTIAL: {hill}\n*** AT T= {hill*1000} FSEC\n"
                "Coordinate=0.9\nGaussian width=0.1\n" for hill in (1, 2, 3)))
            (folder / "fes.dat").write_text(
                "### FREE ENERGY SURFACE CONSISTING OF 2 GAUSSIANS\n0 0\n1 .1\n"
                "### FREE ENERGY SURFACE CONSISTING OF 3 GAUSSIANS\n0 0\n")
            times, blocks = aligned_fes(folder / "fes.dat")
            np.testing.assert_allclose(times, [2.])
            self.assertEqual(len(blocks), 1)
            self.assertAlmostEqual(blocks[0][1, 1], 62.7509474)

    def test_sustained_event_not_brief_excursion(self):
        args = argparse.Namespace(distance_min=4., deprotonated_s_max=.05,
                                  returned_s_min=.2, persistence_ps=.05)
        t = np.arange(20) * .02
        s = np.ones(20) * .9
        s[2] = .01  # One-frame excursion must not trigger.
        s[7:12] = .01
        d = np.ones(20) * 5.
        dep, diffuse = screen(dict(times=t, coordination=s, distance=d,
                                   state_distance=d), args)
        self.assertAlmostEqual(dep, .14)
        self.assertAlmostEqual(diffuse, .24)

    def test_periodic_dihedral_invariant(self):
        box = np.ones(3) * 10
        points = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [2., 1., 1.]])
        shifted = points.copy()
        shifted[2] += box
        self.assertAlmostEqual(_signed_dihedral_deg(points, box),
                               _signed_dihedral_deg(shifted, box))
        np.testing.assert_allclose(_minimum_image(np.array([9., 0., 0.]), box), [-1., 0., 0.])


if __name__ == "__main__":
    unittest.main()
