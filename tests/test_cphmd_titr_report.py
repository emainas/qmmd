import tempfile
import unittest
from pathlib import Path

import numpy as np

from qmmd.cphmd_titr_report import (
    ProtonationRecord,
    analyze_syn_anti,
    bhattacharyya_overlap,
    fit_henderson_hasselbalch,
    fit_tps_microstate_model,
    fit_two_site_model,
    load_titr_report_config,
    parse_edge_acceptance,
    parse_roundtrip_stats,
    render_remlog_cpptraj_input,
    tps_microstate_probabilities,
    two_site_probabilities,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "MEA" / "cphmd" / "cphmd.yaml"
PR4_CONFIG = REPO / "configs" / "PR4" / "cphmd" / "cphmd.yaml"
BLA_CONFIG = REPO / "configs" / "BLA" / "cphmd" / "cphmd.yaml"
BLA_PRN_ONLY_CONFIG = (
    REPO / "configs" / "BLA" / "cphmd" / "cphmd-prn-tails-mbondi2.yaml"
)
BLA_TPS_ONLY_CONFIG = (
    REPO / "configs" / "BLA" / "cphmd" / "cphmd-tps-ti-tails-deprot-1ns.yaml"
)


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
        self.assertIsNone(cfg.syn_anti)

    def test_pr4_syn_anti_state_names_map_to_cpin_indices(self):
        cfg = load_titr_report_config(PR4_CONFIG)
        self.assertIsNotNone(cfg.syn_anti)
        assert cfg.syn_anti is not None
        self.assertEqual(cfg.syn_anti.syn_states, (1, 3))
        self.assertEqual(cfg.syn_anti.anti_states, (2, 4))

    def test_bla_composite_report_maps_tps_and_both_prx_tails(self):
        cfg = load_titr_report_config(BLA_CONFIG)
        self.assertIsNotNone(cfg.composite)
        assert cfg.composite is not None
        self.assertIsNotNone(cfg.composite.tps)
        self.assertIsNotNone(cfg.composite.prx)
        assert cfg.composite.tps is not None
        assert cfg.composite.prx is not None
        self.assertEqual(cfg.composite.tps.residue_id, 1)
        self.assertEqual(
            (
                cfg.composite.tps.protonated_state,
                cfg.composite.tps.deprotonated_states,
            ),
            (0, (1, 2, 3, 4)),
        )
        self.assertEqual(cfg.composite.prx.residue_ids, (2, 3))
        self.assertEqual(cfg.composite.prx.deprotonated_state, 0)
        self.assertEqual(cfg.composite.prx.syn_states, (1, 3))
        self.assertEqual(cfg.composite.prx.anti_states, (2, 4))

    def test_bla_composite_report_allows_prn_only_control(self):
        cfg = load_titr_report_config(BLA_PRN_ONLY_CONFIG)
        self.assertIsNotNone(cfg.composite)
        assert cfg.composite is not None
        self.assertIsNone(cfg.composite.tps)
        self.assertIsNotNone(cfg.composite.prx)
        assert cfg.composite.prx is not None
        self.assertEqual(cfg.composite.prx.residue_ids, (2, 3))
        self.assertEqual(cfg.composite.prx.residue_name, "PRN")
        self.assertEqual(cfg.composite.prx.deprotonated_state, 0)
        self.assertEqual(cfg.composite.prx.syn_states, (1, 3))
        self.assertEqual(cfg.composite.prx.anti_states, (2, 4))

    def test_bla_tps_report_loads_configured_microstate_model(self):
        cfg = load_titr_report_config(BLA_TPS_ONLY_CONFIG)
        self.assertIsNotNone(cfg.composite)
        assert cfg.composite is not None
        self.assertIsNotNone(cfg.composite.tps)
        assert cfg.composite.tps is not None
        self.assertEqual(
            cfg.composite.tps.microstate_model_pkas,
            (5.0, 4.5, 4.5, 5.0),
        )

    def test_syn_anti_fractions_are_conditional_on_protonation(self):
        records = [
            ProtonationRecord(i, i * 100, 4.0, "PR4", 1, state, int(state > 0))
            for i, state in enumerate([0, 1, 3, 2, 0, 4, 1], start=1)
        ]
        rows = analyze_syn_anti(
            np.asarray([4.0]),
            {4.0: records},
            (1, 3),
            (2, 4),
        )
        self.assertEqual(rows[0].protonated_count, 5)
        self.assertEqual(rows[0].syn_count, 3)
        self.assertEqual(rows[0].anti_count, 2)
        self.assertAlmostEqual(rows[0].syn_fraction, 0.6)
        self.assertAlmostEqual(rows[0].anti_fraction, 0.4)
        self.assertAlmostEqual(rows[0].syn_to_anti_ratio, 1.5)

    def test_henderson_hasselbalch_fit(self):
        ph = np.arange(7.0, 11.0, 0.5)
        expected_pka = 9.1
        expected_hill = 1.2
        fraction = 1.0 / (1.0 + 10.0 ** (expected_hill * (ph - expected_pka)))
        fit = fit_henderson_hasselbalch(ph, fraction)
        self.assertAlmostEqual(fit.pka, expected_pka, places=10)
        self.assertAlmostEqual(fit.hill, expected_hill, places=10)

    def test_coupled_two_site_fit_recovers_microstate_model(self):
        ph = np.arange(2.5, 7.5, 0.5)
        expected = (4.7, 5.2, 0.8)
        probabilities = two_site_probabilities(ph, *expected)
        rows = [
            {
                "pH": float(value),
                **{
                    f"p{state}_fraction": float(probabilities[index][row])
                    for index, state in enumerate(("00", "10", "01", "11"))
                },
            }
            for row, value in enumerate(ph)
        ]
        fit = fit_two_site_model(rows)
        self.assertEqual(
            tuple(round(value, 8) for value in sorted((fit.pk10, fit.pk01))),
            tuple(round(value, 8) for value in sorted(expected[:2])),
        )
        self.assertAlmostEqual(fit.interaction_w, expected[2], places=8)
        self.assertLess(fit.macro_pka1, fit.macro_pka2)

    def test_tps_fit_recovers_five_state_microstate_model(self):
        cfg = load_titr_report_config(BLA_CONFIG)
        assert cfg.composite is not None
        assert cfg.composite.tps is not None
        tps = cfg.composite.tps
        ph = np.arange(2.5, 7.5, 0.5)
        expected = (4.7, 4.3, 4.5, 5.1)
        probabilities = tps_microstate_probabilities(ph, expected)
        rows = []
        for row, value in enumerate(ph):
            item = {
                "pH": float(value),
                "ppp_fraction": float(probabilities[0][row]),
            }
            for index, name in enumerate(tps.deprotonated_names, start=1):
                item[f"{name}_fraction_total"] = float(probabilities[index][row])
            rows.append(item)

        fit = fit_tps_microstate_model(rows, tps)
        np.testing.assert_allclose(fit.microscopic_pkas, expected, atol=1.0e-8)
        self.assertAlmostEqual(
            fit.macroscopic_pka,
            -np.log10(np.power(10.0, -np.asarray(expected)).sum()),
            places=8,
        )
        self.assertLess(fit.rmse, 1.0e-10)

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
