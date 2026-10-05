import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qmmd.refep_equil_report import (
    DihedralSample,
    RefepDihedralConfig,
    angle_near_target,
    load_refep_equil_report_config,
    parse_thermo_samples,
    plot_dihedral_timeseries,
    render_dihedral_cpptraj_input,
)


REPO = Path(__file__).resolve().parents[1]
MEA_CONFIG = REPO / "configs" / "MEA" / "refep" / "equil.yaml"
ANTI_CONFIG = REPO / "configs" / "PRN-anti" / "refep" / "equil.yaml"
SYN_CONFIG = REPO / "configs" / "PRN-syn" / "refep" / "equil.yaml"


class RefepEquilReportTests(unittest.TestCase):
    def test_dihedral_is_optional(self):
        cfg = load_refep_equil_report_config(MEA_CONFIG)
        self.assertIsNone(cfg.dihedral)

    def test_prn_dihedral_definitions(self):
        anti = load_refep_equil_report_config(ANTI_CONFIG)
        syn = load_refep_equil_report_config(SYN_CONFIG)
        self.assertIsNotNone(anti.dihedral)
        self.assertIsNotNone(syn.dihedral)
        assert anti.dihedral is not None
        assert syn.dihedral is not None
        self.assertEqual(anti.dihedral.atom_ids, (5, 3, 4, 11))
        self.assertEqual(syn.dihedral.atom_ids, (5, 3, 4, 11))
        self.assertEqual(anti.dihedral.target_deg, 180.0)
        self.assertEqual(syn.dihedral.target_deg, 0.0)

    def test_parses_amber_thermodynamic_block(self):
        text = """
 NSTEP =      500   TIME(PS) =    1151.000  TEMP(K) =   302.50  PRESS =     0.0
 Etot   =     -1234.5000  EKtot   =       250.0000  EPtot      =     -1484.5000
 NSTEP =     1000   TIME(PS) =    1152.000  TEMP(K) =   298.00  PRESS =     0.0
 Etot   =     -1235.5000  EKtot   =       248.0000  EPtot      =     -1483.5000
"""
        samples = parse_thermo_samples(text, 2, 2 / 15, 0.002)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].time_ps, 1.0)
        self.assertEqual(samples[-1].temperature_k, 298.0)
        self.assertEqual(samples[-1].potential_energy_kcal_mol, -1483.5)

    def test_periodic_dihedral_branch_centers(self):
        self.assertAlmostEqual(angle_near_target(-179.0, 180.0), 181.0)
        self.assertAlmostEqual(angle_near_target(179.0, 180.0), 179.0)
        self.assertAlmostEqual(angle_near_target(-179.0, 0.0), -179.0)

    def test_cpptraj_input_uses_one_based_atom_masks(self):
        cfg = load_refep_equil_report_config(ANTI_CONFIG)
        assert cfg.dihedral is not None
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            text = render_dihedral_cpptraj_input(
                root / "system.parm7",
                root / "equil.nc",
                root / "dihedral.dat",
                cfg.dihedral,
            )
        self.assertIn("dihedral REFEP_DIH @5 @3 @4 @11", text)

    def test_dihedral_plot_supports_scatter_rendering(self):
        dihedral = RefepDihedralConfig("O2-CG-O1-H11", (5, 3, 4, 11), 180.0)
        samples = [
            DihedralSample(0, 0.0, 1, 10.0, 175.0, 175.0),
            DihedralSample(1, 1.0, 1, 10.0, -175.0, 185.0),
        ]
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            output = Path(tmp) / "dihedral.png"
            with patch("matplotlib.axes.Axes.scatter", autospec=True) as scatter:
                plot_dihedral_timeseries(
                    output,
                    REPO / "plotting" / "prl.mplstyle",
                    "PRN-anti",
                    "production",
                    dihedral,
                    2,
                    samples,
                    scatter=True,
                )
            self.assertTrue(output.is_file())
        self.assertEqual(scatter.call_count, 2)


if __name__ == "__main__":
    unittest.main()
