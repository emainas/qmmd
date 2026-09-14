import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
import plot_cv_grid as cv


class CvTimeTests(unittest.TestCase):
    def test_ylim_validation(self):
        for limits in (("3", "-3"), ("1", "1"), ("nan", "3"), ("0", "inf")):
            with self.subTest(limits=limits), patch.object(sys, "argv", [
                "plot_cv_grid.py", "--runs-path", "/tmp/nonexistent-cv-test",
                "--cv-dir", "meta-lcod", "--ylim", *limits
            ]), self.assertRaises(SystemExit) as error:
                cv.main()
            self.assertEqual(error.exception.code, 2)

    def test_ylim_default_and_override(self):
        from matplotlib.axes import Axes
        original = Axes.set_ylim
        for limits, expected in (([], (0., 2.)), (["--ylim", "-3", "3"], (-3., 3.))):
            with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
                root = Path(tmp)
                self.setup_run(root, 1, 20000)
                observed = []
                def capture(axis, *args, **kwargs):
                    result = original(axis, *args, **kwargs)
                    observed.append(tuple(axis.get_ylim()))
                    return result
                with patch.object(sys, "argv", ["plot_cv_grid.py", "--runs-path", str(root),
                     "--cv-dir", "meta-h", "--out", str(root/"test.png"), *limits]), \
                     patch.object(Axes, "set_ylim", capture):
                    cv.main()
                self.assertIn(expected, observed)

    def setup_run(self, root, run, step, cv_dir="meta-h", raw_start=None):
        folder = root / f"run-{run}"
        parent, child = folder / "equil", folder / cv_dir
        parent.mkdir(parents=True, exist_ok=True)
        child.mkdir(parents=True, exist_ok=True)
        parent.joinpath("dftb.inp").write_text("MD=(DELTAT=1.0D-15)\n")
        parent.joinpath("traject").write_text("1\n*** AT T= 0 FSEC, THIS RUN'S STEP NO.= 0\nH 0 0 0\n")
        raw_start = step*.5 if raw_start is None else raw_start
        child.joinpath("traject").write_text(f"1\n*** AT T= {raw_start} FSEC, THIS RUN'S STEP NO.= {step}\nH 0 0 0\n")
        child.joinpath("meta_spec.yaml").write_text("replica_dirname: equil\n")
        child.joinpath("biaspot").write_text(
            f"*** AT T= {raw_start+40} FSEC\nCoordinate=0.9\n"
            f"*** AT T= {raw_start+80} FSEC\nCoordinate=0.8\n")
        return folder, child

    def test_changed_timestep_and_first_hill_delay(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            folder, child = self.setup_run(Path(tmp), 6, 25118)
            offset = cv.infer_offset_ps(folder, "meta-h", 12.599)
            self.assertAlmostEqual(offset, 12.559)
            self.assertAlmostEqual(12.599 + offset, 25.158)
            # Overwritten binary restart contents must have no influence.
            child.joinpath("restart").write_bytes((987654).to_bytes(4, "little"))
            self.assertAlmostEqual(cv.infer_offset_ps(folder, "meta-h", 12.599), offset)

    def test_explicit_restart_anchor_not_first_hill(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            folder, child = self.setup_run(Path(tmp), 1, 0, raw_start=0.)
            child.joinpath("meta_spec.yaml").write_text("replica_dirname: equil\nrestart_time_ps: 40\n")
            self.assertEqual(cv.infer_offset_ps(folder, "meta-h", .04), 40.)

    def test_same_timestep_zero_offset(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            folder, _ = self.setup_run(Path(tmp), 1, 40000, raw_start=40000.)
            self.assertEqual(cv.infer_offset_ps(folder, "meta-h", 40.04), 0.)

    def test_bias_pairing_exponents_partial_tail(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            p = Path(tmp)/"biaspot"
            p.write_text("*** AT T= 40 FSEC\nCoordinate=1.2D-2\n*** AT T= 80 FSEC\nCoordinate=0.")
            t, y = cv.parse_biaspot(p)
            np.testing.assert_allclose(t, [.04])
            np.testing.assert_allclose(y, [.012])
            p.write_text("*** AT T= 40 FSEC\n*** AT T= 80 FSEC\nCoordinate=.1\n")
            with self.assertRaises(ValueError):
                cv.parse_biaspot(p)

    def test_multiple_runs_and_cv_dirs_shift_actual_data_independently(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            self.setup_run(root, 1, 20000)
            self.setup_run(root, 1, 30000, cv_dir="meta-other")
            self.setup_run(root, 2, 40000)
            out = root / "grid.png"
            argv = ["plot_cv_grid.py", "--runs-path", str(root), "--cv-dir", "meta-h,meta-other",
                    "--out", str(out), "--style", str(root/"no-style")]
            with patch.object(sys, "argv", argv), patch.object(cv.plt, "close"):
                cv.main()
                figure = cv.plt.gcf()
            plotted_starts = [float(line.get_xdata()[0]) for axis in figure.axes for line in axis.lines]
            np.testing.assert_allclose(plotted_starts, [20.04, 30.04, 40.04])
            cv.plt.close(figure)
            with out.with_suffix(".csv").open() as handle:
                rows = list(csv.DictReader(handle))
            first = {(r["run_id"], r["cv_dir"]): float(r["offset_ps"]) for r in rows}
            self.assertEqual(first, {("1", "meta-h"): 10., ("1", "meta-other"): 15., ("2", "meta-h"): 20.})
            for r in rows:
                self.assertAlmostEqual(float(r["plotted_time_ps"]), float(r["raw_time_ps"])+float(r["offset_ps"]))
            self.assertTrue(out.exists())

    def test_missing_anchor_fails_instead_of_guessing(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            with self.assertRaisesRegex(ValueError, "time-axis raw"):
                cv.infer_offset_ps(Path(tmp), "meta-h", 20.)

    def test_reset_continuation_not_concatenated_by_guess(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            folder, child = self.setup_run(Path(tmp), 1, 20000)
            restart = child / "metad-restart"
            restart.mkdir()
            restart.joinpath("biaspot").write_text("*** AT T= 40 FSEC\nCoordinate=.9\n*** AT T= 20000 FSEC\nCoordinate=.8\n")
            with self.assertRaisesRegex(ValueError, "explicit stitching"):
                cv.load_biaspot_with_restart(folder, "meta-h")


if __name__ == "__main__":
    unittest.main()
