import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from qmmd.cphmd_dgref_report import (
    DgrefSample,
    dgref_values_after_discard,
    parse_dgref_log,
    plot_dgref_iteration_report,
    summarize_dgref_after_discard,
    write_report_csv,
)


class CpHMDDgrefReportTests(unittest.TestCase):
    def test_partial_log_keeps_only_completed_evaluations(self):
        text = """
 AMBER execution #1: running 1000 MD steps for DELTAGREF = 0.000000 kcal/mol
   The fraction of protonated species is 100.00% for the Residue 'MEA 1'
 AMBER execution #2: running 1000 MD steps for DELTAGREF = -100.000000 kcal/mol
   The fraction of protonated species is   0.00% for the Residue 'MEA 1'
 AMBER execution #3: running 10000 MD steps of equilibration for DELTAGREF = -50.000000 kcal/mol
 AMBER execution #3: running 100000 MD steps of production    for DELTAGREF = -50.000000 kcal/mol
"""
        samples = parse_dgref_log(text)
        self.assertEqual(len(samples), 2)
        self.assertEqual([sample.execution for sample in samples], [1, 2])
        self.assertEqual([sample.dgref_kcal_mol for sample in samples], [0.0, -100.0])
        self.assertEqual(
            [sample.protonated_fraction_percent for sample in samples],
            [100.0, 0.0],
        )

    def test_production_phase_and_csv(self):
        text = """
 AMBER execution #9: running 10000  MD steps of equilibration for DELTAGREF = -26.454900 kcal/mol
 AMBER execution #9: running 100000 MD steps of production    for DELTAGREF = -26.454900 kcal/mol
   The fraction of protonated species is  66.20% for the Residue 'MEA 1'
"""
        samples = parse_dgref_log(text)
        self.assertEqual(samples[0].phase, "production")
        self.assertEqual(samples[0].md_steps, 100000)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "report.csv"
            write_report_csv(path, samples)
            csv_text = path.read_text()
        self.assertIn("9,production,100000,-26.454900,66.2000", csv_text)

    def test_empty_or_unrelated_log_is_valid(self):
        self.assertEqual(parse_dgref_log("Checking cpin file\n"), [])

    def test_iteration_plot_contains_completed_samples(self):
        samples = [
            DgrefSample(1, "search", 1000, 0.0, 100.0),
            DgrefSample(2, "search", 1000, -100.0, 0.0),
        ]
        cfg = SimpleNamespace(
            system="MEA",
            cntrl={"solvph": 9.0},
            report_discard_first=8,
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            output = Path(tmp) / "dgref-iteration.png"
            plot_dgref_iteration_report(
                output,
                cfg,  # type: ignore[arg-type]
                samples,
                Path(tmp) / "missing.mplstyle",
            )
            self.assertTrue(output.is_file())
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_summary_discards_initial_completed_values(self):
        samples = [
            DgrefSample(i, "search", 1000, float(-i), 50.0)
            for i in range(1, 11)
        ]
        self.assertEqual(dgref_values_after_discard(samples, 8), [-9.0, -10.0])
        mean, standard_deviation = summarize_dgref_after_discard(samples, 8)  # type: ignore[misc]
        self.assertEqual(mean, -9.5)
        self.assertAlmostEqual(standard_deviation, 2 ** -0.5)


if __name__ == "__main__":
    unittest.main()
