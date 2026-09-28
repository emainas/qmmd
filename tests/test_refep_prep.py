import unittest
from pathlib import Path

import numpy as np

from qmmd.cphmd_prep import read_mol2_atoms
from qmmd.refep_prep import (
    build_ligand_charge_map,
    load_refep_prep_config,
    read_amber_topology,
    render_endpoint_parmed_input,
    render_interpolation_parmed_input,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "MEA" / "refep" / "prep.yaml"


class RefepPrepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_refep_prep_config(CONFIG)
        cls.topology = read_amber_topology(cls.cfg.input_parm7)
        cls.mapping = build_ligand_charge_map(
            cls.cfg,
            cls.topology,
            read_mol2_atoms(cls.cfg.deprot_mol2),
        )

    def test_mea_config_uses_equilibrated_mdequil_restart(self):
        self.assertEqual(self.cfg.windows, 16)
        self.assertEqual(self.cfg.input_parm7.parent.name, "prep")
        self.assertEqual(self.cfg.input_rst7.parent.name, "mdequil")
        self.assertEqual(self.cfg.input_rst7.name, "equil-npt.rst7")

    def test_topology_and_deprotonated_mol2_mapping(self):
        self.assertEqual(self.topology.atom_count, 358)
        self.assertEqual(self.mapping.atom_names[0], "C1")
        self.assertEqual(self.mapping.atom_names[-1], "HN3")
        self.assertEqual(self.mapping.dummy_atoms, ("HN1",))
        self.assertAlmostEqual(float(self.mapping.lambda0_charges.sum()), 1.001001, places=6)
        self.assertAlmostEqual(float(self.mapping.lambda1_charges.sum()), 0.001, places=6)
        dummy_index = self.mapping.atom_names.index("HN1")
        self.assertEqual(self.mapping.lambda1_charges[dummy_index], 0.0)

    def test_generated_parmed_inputs_edit_only_charges_and_interpolate(self):
        endpoint = render_endpoint_parmed_input(self.cfg, self.mapping, "lambda-015.parm7")
        self.assertIn("change charge @10 0.0000000000 quiet", endpoint)
        self.assertNotIn("change atom_type", endpoint.lower())
        self.assertIn("outparm lambda-015.parm7", endpoint)

        interpolation = render_interpolation_parmed_input(self.cfg)
        self.assertIn("parm lambda-015.parm7", interpolation)
        self.assertIn("parm lambda-000.parm7", interpolation)
        self.assertIn("interpolate 14", interpolation)
        self.assertIn("eleconly", interpolation)

    def test_charge_path_is_linear(self):
        lambdas = np.linspace(0.0, 1.0, self.cfg.windows)
        path = np.asarray(
            [
                self.mapping.lambda0_charges
                + value * (self.mapping.lambda1_charges - self.mapping.lambda0_charges)
                for value in lambdas
            ]
        )
        np.testing.assert_allclose(path[0], self.mapping.lambda0_charges)
        np.testing.assert_allclose(path[-1], self.mapping.lambda1_charges)
        increments = np.diff(path, axis=0)
        np.testing.assert_allclose(
            increments,
            np.repeat(increments[:1], len(increments), axis=0),
        )


if __name__ == "__main__":
    unittest.main()
