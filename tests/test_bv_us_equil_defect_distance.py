import tempfile
import unittest
from pathlib import Path

import numpy as np

from bv_us_equil_defect_distance import (
    iter_negative_defect_frames,
    window_centers,
)


class BVUSEquilDefectDistanceTests(unittest.TestCase):
    def test_window_centers_preserve_direction(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "pull.yaml"
            path.write_text(
                "windows:\n  start_deg: 20\n  stop_deg: 0\n  spacing_deg: 10\n"
            )
            self.assertEqual(window_centers(path), [20.0, 10.0, 0.0])

    def test_negative_defect_is_most_negative_oxygen_per_frame(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "mulliken"
            path.write_text(
                " *** AT T= 0.00 FSEC, THIS RUN'S STEP NO.= 0\n"
                "  3 O s -0.20\n  3 O p -0.70\n"
                "  6 O s -0.10\n  6 O p -0.70\n"
                " *** AT T= 10.00 FSEC, THIS RUN'S STEP NO.= 10\n"
                "  3 O s -0.10\n  3 O p -0.65\n"
                "  6 O s -0.25\n  6 O p -0.80\n"
            )
            frames = list(iter_negative_defect_frames(path, [3, 6]))
            self.assertEqual([frame.oxygen_id for frame in frames], [3, 6])
            np.testing.assert_allclose(
                [frame.charge_e for frame in frames], [-0.90, -1.05]
            )
            np.testing.assert_allclose(
                [frame.second_charge_e for frame in frames], [-0.80, -0.75]
            )


if __name__ == "__main__":
    unittest.main()
