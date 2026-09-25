import tempfile
import unittest
from pathlib import Path

import numpy as np

from qmmd.cphmd_titr_report import (
    bhattacharyya_overlap,
    fit_henderson_hasselbalch,
    load_titr_report_config,
    parse_edge_acceptance,
    parse_roundtrip_stats,
    render_remlog_cpptraj_input,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "MEA" / "cphmd" / "cphmd.yaml"


class CpHMDTitrReportTests(unittest.TestCase):
    def test_mea_report_config_and_cpptraj_input(self):
        cfg = load_titr_report_config(CONFIG)
        self.assertEqual(cfg.residue_id, 1)
        self.assertEqual(cfg.reference_pka, 9.0)
        self.assertEqual(cfg.slope_window_exchanges, 10)
        text = render_remlog_cpptraj_input(cfg, Path("/tmp/rem.log"))
        self.assertIn("crdidx stats", text)
        self.assertIn("printtrips", text)
        self.assertIn("reptimeslope 10", text)
        self.assertIn("acceptout exchange-acceptance.dat", text)
        self.assertIn("repidx", text)

    def test_henderson_hasselbalch_fit(self):
        ph = np.arange(7.0, 11.0, 0.5)
        expected_pka = 9.1
        expected_hill = 1.2
        fraction = 1.0 / (1.0 + 10.0 ** (expected_hill * (ph - expected_pka)))
        fit = fit_henderson_hasselbalch(ph, fraction)
        self.assertAlmostEqual(fit.pka, expected_pka, places=10)
        self.assertAlmostEqual(fit.hill, expected_hill, places=10)

    def test_state_overlap(self):
        self.assertAlmostEqual(
            bhattacharyya_overlap(np.array([0.25, 0.75]), np.array([0.25, 0.75])),
            1.0,
        )
        self.assertAlmostEqual(
            bhattacharyya_overlap(np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            0.0,
        )

    def test_parse_cpptraj_exchange_outputs(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            acceptance = root / "accept.dat"
            acceptance.write_text(
                "#Replica %UP %DOWN\n"
                "1 80 10\n"
                "2 70 80\n"
                "3 10 70\n"
            )
            edges = parse_edge_acceptance(acceptance, 3)
            self.assertEqual((edges[0].forward_percent, edges[0].reverse_percent), (80, 80))
            self.assertEqual((edges[-1].forward_percent, edges[-1].reverse_percent), (10, 10))

            stats = root / "stats.dat"
            stats.write_text(
                "#Round-trip stats:\n"
                "#CRDIDX RndTrips AvgExch. SD_Exch. Min Max\n"
                "1 2 10.0 1.0 9 11\n"
                "2 1 12.0 0.0 12 12\n"
            )
            parsed = parse_roundtrip_stats(stats)
            self.assertEqual(parsed.shape, (2, 6))
            self.assertEqual(parsed[:, 1].sum(), 3)


if __name__ == "__main__":
    unittest.main()
