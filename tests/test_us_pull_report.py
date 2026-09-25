import tempfile
import unittest
from pathlib import Path

from qmmd.us_pull_report import (
    amber_residue_for_atoms,
    angle_near_target,
    read_cpptraj_dihedrals,
    render_pull_clock_vmd,
    stitch_samples,
)
from qmmd.us_pull import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/PRN-anti/us/pull.yaml"


class USPullReportTests(unittest.TestCase):
    def test_angle_is_mapped_to_branch_nearest_target(self):
        self.assertAlmostEqual(angle_near_target(-179.0, 180.0), 181.0)
        self.assertAlmostEqual(angle_near_target(2.0, 0.0), 2.0)

    def test_read_cpptraj_dihedrals(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            data = Path(tmp) / "dihedral.dat"
            data.write_text("#Frame pulled\n       1  179.5\n       2 -178.0\n")
            self.assertEqual(read_cpptraj_dihedrals(data), [179.5, -178.0])

    def test_stitch_samples_tracks_windows_and_elapsed_time(self):
        samples = stitch_samples(
            [179.0, -179.0, 169.0, 171.0],
            [180.0, 170.0],
            frames_per_window=2,
            sample_interval_ps=0.1,
        )
        self.assertEqual([sample.window_index for sample in samples], [0, 0, 1, 1])
        self.assertEqual([sample.frame_in_window for sample in samples], [1, 2, 1, 2])
        self.assertAlmostEqual(samples[-1].time_ps, 0.4)
        self.assertAlmostEqual(samples[1].dihedral_branch_deg, 181.0)

    def test_stitch_rejects_missing_frames(self):
        with self.assertRaisesRegex(ValueError, "Expected 4 trajectory frames"):
            stitch_samples([1.0, 2.0, 3.0], [180.0, 170.0], 2, 0.1)

    def test_vmd_clock_uses_single_companion_pdb(self):
        cfg = load_config(CONFIG)
        script = render_pull_clock_vmd(cfg, 190)
        self.assertIn("pull-clock.mol2", script)
        self.assertNotIn("pull.nc", script)
        self.assertNotIn("solv.parm7", script)
        self.assertIn('mol selection "index 0 to 10"', script)
        self.assertIn('mol selection "name CG O1 O2 H11"', script)
        self.assertIn('mol selection "name H11"', script)
        self.assertNotIn("mol drawframes", script)
        self.assertIn("mol representation VDW", script)

    def test_amber_residue_for_atoms(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            topology = Path(tmp) / "test.parm7"
            topology.write_text(
                "%FLAG RESIDUE_POINTER\n%FORMAT(10I8)\n       1      12      20\n"
                "%FLAG NEXT_FLAG\n%FORMAT(10I8)\n"
            )
            self.assertEqual(amber_residue_for_atoms(topology, (5, 3, 4, 11)), (1, 1))
            with self.assertRaisesRegex(ValueError, "must belong to one residue"):
                amber_residue_for_atoms(topology, (5, 3, 4, 12))


if __name__ == "__main__":
    unittest.main()
