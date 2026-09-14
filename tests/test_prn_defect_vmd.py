import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
from build_prn_defect_vmd import image_frame, VMD_SCRIPT, display_defects


class DefectMovieTests(unittest.TestCase):
    def test_forward_fill_preserves_leading_gaps_and_new_assignments(self):
        raw = np.array([np.nan, 12., np.nan, np.nan, 15., np.nan])
        np.testing.assert_allclose(display_defects(raw, True),
                                   [np.nan, 12., 12., 12., 15., 15.])
        np.testing.assert_allclose(display_defects(raw, False), raw)

    def test_periodic_image_and_changing_hydrogen_owner(self):
        symbols = ["C", "O", "O", "H"]
        box = np.ones(3)*10
        coords = np.array([[.5, 1, 0], [.2, 0, 0], [8., 0, 0], [9.5, 0, 0]])
        imaged, bonds = image_frame(coords, symbols, box, anchor_id=1)
        self.assertEqual(bonds, [(2, 4)])
        self.assertAlmostEqual(np.linalg.norm(imaged[1]-imaged[3]), .7)
        np.testing.assert_allclose(imaged[0], 0.)
        coords[3] = [8., .95, 0]
        _, bonds = image_frame(coords, symbols, box, anchor_id=1)
        self.assertEqual(bonds, [(3, 4)])

    @unittest.skipUnless(shutil.which("tclsh"), "Tcl interpreter unavailable")
    def test_vmd_callbacks_with_assigned_and_unassigned_frames(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            folder = Path(tmp)
            (folder / "view.vmd").write_text(VMD_SCRIPT)
            (folder / "frames.csv").write_text(
                "frame,time,elapsed,defect,q,hs,bonds,matched,distance\n"
                "0,8,0,12,-.6,13,12-13;4-11,1,5\n"
                "1,8.02,.02,,,,4-11,1,nan\n")
            prelude = r'''
proc mol {args} { return 0 }
proc molinfo {args} {
    if {[lindex $args 1] eq "get"} { return 2 }
    return 0
}
proc color {args} {}
proc display {args} {}
proc axes {args} {}
proc graphics {args} {}
proc atomselect {args} { return selection }
proc selection {args} {
    if {[lindex $args 0] eq "get"} { return [lrepeat 13 {0 0 0}] }
}
proc animate {args} {
    if {[lindex $args 0] eq "goto"} { set ::vmd_frame(0) [lindex $args 1] }
}
'''
            commands = prelude + f'if {{[catch {{source {{{folder / "view.vmd"}}}; set ::vmd_frame(0) 1}} err]}} {{puts stderr $err; exit 1}}\n'
            result = subprocess.run(["tclsh"], input=commands, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(result.stderr, result.stderr)


if __name__ == "__main__":
    unittest.main()
