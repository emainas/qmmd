import tempfile
import unittest
from pathlib import Path

from qmmd.cphmd_titr_post import (
    load_titr_post_config,
    parse_fraction_plot,
    parse_state_table,
    render_protonation_cpptraj_input,
    render_trajectory_cpptraj_input,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "MEA" / "cphmd" / "cphmd.yaml"


class CpHMDTitrPostTests(unittest.TestCase):
    def test_mea_post_config(self):
        cfg = load_titr_post_config(CONFIG)
        self.assertEqual(cfg.output_dir, "post")
        self.assertEqual(cfg.fraction, "protonated")
        self.assertEqual(cfg.cpptraj_executable, "cpptraj")

    def test_trajectory_input_sorts_to_fixed_ph_by_replica_index(self):
        text = render_trajectory_cpptraj_input(load_titr_post_config(CONFIG))
        self.assertIn("ensemble ../replica-01.nc trajnames", text)
        self.assertIn("../replica-10.nc", text)
        self.assertIn("trajout fixed-ph.nc netcdf", text)
        self.assertNotIn("bycrdidx", text)
        self.assertNotIn("remlog", text)

    def test_protonation_input_uses_amber_remd_sort(self):
        text = render_protonation_cpptraj_input(load_titr_post_config(CONFIG))
        self.assertIn("readensembledata ../replica-01.cpout", text)
        self.assertIn("sortensembledata PH", text)
        self.assertIn("runanalysis cphstats PH[*]", text)
        self.assertNotIn(" deprot ", text)
        self.assertIn("writedata protonation-states.nc PH[*] netcdf", text)

    def test_parses_cpptraj_state_and_fraction_tables(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            states = root / "states.dat"
            states.write_text(
                "#Frame MEA:1%0 MEA:1%1\n"
                "1 0 1\n"
                "2 1 1\n"
            )
            columns, rows = parse_state_table(states)
            self.assertEqual(columns[0].residue_id, 1)
            self.assertEqual(columns[1].member, 1)
            self.assertEqual(rows, [[0, 1], [1, 1]])

            fractions = root / "fractions.dat"
            fractions.write_text("#pH MEA:1\n7.0 0.9\n7.5 0.8\n")
            labels, fraction_rows = parse_fraction_plot(fractions)
            self.assertEqual(labels, ["MEA:1"])
            self.assertEqual(fraction_rows[-1], (7.5, [0.8]))


if __name__ == "__main__":
    unittest.main()
