import tempfile
import unittest
from pathlib import Path

import numpy as np

from qmmd.us_wham_report import nearest_window_indices, read_smooth_pmf


class USWhamReportTests(unittest.TestCase):
    def test_nearest_windows_follow_descending_centers(self):
        centers = np.array([180.0, 170.0, 160.0, 10.0, 0.0])
        indices = nearest_window_indices(np.array([179.0, 161.0, 9.0, 1.0]), centers)
        np.testing.assert_array_equal(indices, [0, 2, 3, 4])

    def test_read_smooth_pmf(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "pmf_smooth.csv"
            path.write_text(
                "dihedral_deg,smoothed_pmf_kcal_mol\n"
                "1.0,0.0\n"
                "3.0,0.2\n"
            )
            grid, pmf = read_smooth_pmf(path)
            np.testing.assert_allclose(grid, [1.0, 3.0])
            np.testing.assert_allclose(pmf, [0.0, 0.2])


if __name__ == "__main__":
    unittest.main()
