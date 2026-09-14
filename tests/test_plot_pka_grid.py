import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
import plot_cv_grid as cv
import plot_pka_grid as pka


class PkaClockTests(unittest.TestCase):
    def make_run(self, root: Path, run: int, step: int) -> None:
        folder = root / f"run-{run}"
        parent, child = folder / "equil", folder / "meta-h"
        parent.mkdir(parents=True)
        child.mkdir()
        (parent / "dftb.inp").write_text("MD=(DELTAT=1D-15)\n")
        (parent / "traject").write_text("1\n*** AT T= 0 FSEC, STEP NO.= 0\nH 0 0 0\n")
        (child / "traject").write_text(f"1\n*** AT T= {step*.5} FSEC, STEP NO.= {step}\nH 0 0 0\n")
        (child / "meta_spec.yaml").write_text("replica_dirname: equil\n")
        (child / "biaspot").write_text("".join(
            f"*** AT T= {step*.5+dt} FSEC\nCoordinate=.9\n" for dt in (40, 80)))
        (child / "fes.dat").write_text("### surface\n0 0.003\n1 0.001\n"*2)

    def run_grid(self, root: Path, mode: str, ids: str):
        out = root / f"{mode}-{ids}.png"
        argv = ["plot_pka_grid.py", "--runs-path", str(root), "--cv-dir", "meta-h",
                "--run-ids", ids, "--out", str(out), "--style", str(root/"absent"),
                "--temp", "300", "--time-axis", mode]
        with patch.object(sys, "argv", argv), patch.object(pka.plt, "close"):
            pka.main()
            figure = pka.plt.gcf()
        plotted = [axis.lines[0].get_xdata().copy() for axis in figure.axes]
        pka.plt.close(figure)
        with out.with_suffix(".csv").open() as handle:
            rows = list(csv.DictReader(handle))
        return rows, plotted

    def test_same_clock_as_cv_and_independent_panels(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            for run, step in [(1, 40000), (2, 25118), (3, 32188)]:
                self.make_run(root, run, step)
            rows, plotted = self.run_grid(root, "restart-aligned", "1-3")
            for run, x in enumerate(plotted, 1):
                raw, _ = cv.parse_biaspot(root/f"run-{run}"/"meta-h/biaspot")
                offset = cv.infer_offset_ps(root/f"run-{run}", "meta-h", raw[0])
                np.testing.assert_allclose(x, raw+offset)
                csv_x = [float(row["plotted_time_ps"]) for row in rows if int(row["run_id"]) == run]
                np.testing.assert_allclose(x, csv_x)
            self.assertAlmostEqual(plotted[0][0], 40.04)

    def test_raw_mode_and_single_panel_preserve_pka_values(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            self.make_run(root, 1, 40000)
            aligned, _ = self.run_grid(root, "restart-aligned", "1")
            raw, plotted = self.run_grid(root, "raw", "1")
            np.testing.assert_allclose(plotted[0], [20.04, 20.08])
            np.testing.assert_allclose([float(r["pka"]) for r in aligned], [float(r["pka"]) for r in raw])
            self.assertAlmostEqual(float(raw[0]["pka"]), .002*pka.EH_TO_KCALMOL/(pka.PKA_FACTOR*300))


if __name__ == "__main__":
    unittest.main()
