import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
from prn_equil_meta_dih import align_meta, restart_origin, timestep_ps, departure_index


class StageTimeTests(unittest.TestCase):
    def test_first_crossing_excludes_later_return(self):
        data = np.zeros((4, 4))
        data[:, 3] = [1.0, 1.3, 1.31, 1.0]
        self.assertEqual(departure_index(data, 1.3), 2)
        self.assertEqual(departure_index(data, 1.4), 4)
        self.assertEqual(departure_index(data, .9), 0)
        self.assertEqual(departure_index(data[:0], 1.3), 0)

    def test_changed_timestep_and_early_branch(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "traject"
            path.write_text("311\n *** AT T= 5351.00 FSEC, THIS RUN'S STEP NO.= 10702\n")
            step, origin, raw = restart_origin(path, .001, .0005)
            self.assertEqual(step, 10702)
            self.assertAlmostEqual(origin, 10.702)
            data = np.array([[5.351, 0, 0, 1], [5.36, 2, 2, 1]])
            aligned = align_meta(data, origin, raw)
            np.testing.assert_allclose(aligned[:,0], [10.702, 10.711])
            self.assertEqual(data[0,0], 5.351)
            with self.assertRaises(ValueError):
                restart_origin(path, .001, .001)

    def test_input_timestep(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "dftb.inp"
            path.write_text("MD=(NSTEP=80000 DELTAT=0.5D-15)\n")
            self.assertAlmostEqual(timestep_ps(path), .0005)


if __name__ == "__main__":
    unittest.main()
