import tempfile
import unittest
from pathlib import Path

import numpy as np

from qmmd.cphmd_dgref import ChargeSets, TopologyInfo
from qmmd.cphmd_view import (
    identify_dummy_atoms,
    load_cphmd_view_config,
    pin_local_frame_coordinates,
    read_view_frames,
    render_autoimage_cpptraj_input,
    render_vmd_script,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "MEA" / "cphmd" / "cphmd.yaml"


class CpHMDViewTests(unittest.TestCase):
    def setUp(self):
        self.charges = ChargeSets(
            atom_names=("C", "O", "N", "HN1", "HN2"),
            prot_charges=(0.2, -0.4, -0.6, 0.4, 0.4),
            deprot_charges=(0.2, -0.4, -0.2, 0.0, 0.4),
            proton_count_prot=2,
            proton_count_deprot=1,
        )
        self.topology = TopologyInfo(
            residue_number=4,
            first_atom=10,
            atom_names=("C", "O", "N", "HN1", "HN2"),
            first_solvent=15,
        )

    def test_mea_site_and_water_view_settings(self):
        cfg = load_cphmd_view_config(CONFIG)
        self.assertEqual(cfg.focus_atoms, ("N",))
        self.assertEqual(cfg.orientation_atoms, ("N", "C1", "C2"))
        self.assertEqual(cfg.water_sphere_scale, 0.35)
        self.assertEqual(cfg.hbond_hydrogen_atoms, ("HN1", "HN2", "HN3"))
        self.assertEqual(cfg.hbond_line_radius, 0.08)

    def test_identifies_zero_charge_dummy_with_vmd_index(self):
        dummy = identify_dummy_atoms(self.charges, self.topology, 1.0e-8)
        self.assertEqual(dummy[0].name, "HN1")
        self.assertEqual(dummy[0].vmd_index, 12)
        self.assertEqual(dummy[0].hidden_states, (1,))

    def test_reads_only_requested_aligned_ph_and_residue(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            table = Path(tmp) / "coordinate-protonation.csv"
            table.write_text(
                "trajectory_frame,relative_time_ps,pH,trajectory,residue,residue_id,state,proton_count\n"
                "1,1.0,9.0,ph-9.0.nc,MEA,4,0,2\n"
                "1,1.0,9.5,ph-9.5.nc,MEA,4,1,1\n"
                "2,2.0,9.0,ph-9.0.nc,MEA,4,1,1\n"
            )
            frames, trajectory = read_view_frames(table, 9.0, "MEA", 4, 2)
        self.assertEqual(trajectory, "ph-9.0.nc")
        self.assertEqual([frame.state for frame in frames], [0, 1])

    def test_autoimage_centers_residue_and_reimages_solvent(self):
        text = render_autoimage_cpptraj_input(
            Path("/tmp/solv.parm7"),
            Path("/tmp/ph-9.0.nc"),
            "ph-9.0.nc",
            4,
            ("N",),
            ("N", "C", "O"),
        )
        self.assertIn("autoimage anchor :4", text)
        self.assertIn("rms first :4@N,C,O mass", text)
        self.assertIn("center :4@N mass", text)
        self.assertIn("box nobox", text)
        self.assertIn("trajout ph-9.0.nc netcdf", text)

    def test_local_frame_pins_site_bond_and_plane(self):
        frame0 = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.2, 0.3, 1.0]]
        )
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        frame1 = frame0 @ rotation.T + np.asarray([4.0, -2.0, 3.0])
        pinned = pin_local_frame_coordinates(
            np.stack((frame0, frame1)),
            (0,),
            (0, 1, 2),
        )
        np.testing.assert_allclose(pinned[1], pinned[0], atol=1.0e-12)

    def test_script_embeds_state_callback_and_dynamic_bonds(self):
        dummy = identify_dummy_atoms(self.charges, self.topology, 1.0e-8)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            table = Path(tmp) / "coordinate-protonation.csv"
            table.write_text(
                "trajectory_frame,relative_time_ps,pH,trajectory,residue,residue_id,state,proton_count\n"
                "1,1.0,9.0,ph-9.0.nc,MEA,4,0,2\n"
                "2,2.0,9.0,ph-9.0.nc,MEA,4,1,1\n"
            )
            frames, _ = read_view_frames(table, 9.0, "MEA", 4, 2)
        script = render_vmd_script(
            topology_path="solv.parm7",
            trajectory_path="ph-9.0.nc",
            ph=9.0,
            residue_name="MEA",
            residue_id=4,
            topology=self.topology,
            charges=self.charges,
            dummy_atoms=dummy,
            frames=frames,
            focus_atoms=("N",),
            orientation_atoms=("N", "C", "O"),
            zoom=4.0,
            water_sphere_scale=0.35,
            water_sphere_resolution=24,
            hbond_donor_atom="N",
            hbond_hydrogen_atoms=("HN1", "HN2"),
            hbond_water_oxygen_selection="water and name O",
            hbond_distance_cutoff=2.5,
            hbond_angle_cutoff=135.0,
            hbond_line_radius=0.08,
            hbond_dash_count=6,
            hbond_color_rgb=(0.45, 1.0, 0.45),
        )
        self.assertIn("DynamicBonds", script)
        self.assertIn('mol selection "water"', script)
        self.assertIn("mol material Transparent", script)
        self.assertNotIn("mol representation Bonds", script)
        self.assertIn("mol representation VDW 0.350000 24", script)
        self.assertNotIn("mol representation CPK", script)
        self.assertNotIn("mol representation Lines", script)
        self.assertNotIn("measure fit", script)
        self.assertIn("set ::cphmd_focus_indices {11}", script)
        self.assertIn("set ::cphmd_orientation_indices {11 9 10}", script)
        self.assertIn(
            "molinfo $::cphmd_molid set center $initial_focus_center", script
        )
        self.assertIn("cphmd_set_side_view 0", script)
        self.assertNotIn("cphmd_set_side_view $frame", script)
        self.assertIn("set bond_axis [vecnorm", script)
        self.assertNotIn("mol modcolor", script)
        self.assertIn("mol color ColorID 16", script)
        self.assertIn("cphmd_draw_dashed_hbond", script)
        self.assertIn("radius $::cphmd_hbond_line_radius", script)
        self.assertIn("set ::cphmd_hbond_angle_cutoff 135.000000", script)
        self.assertIn("set hbond_count [cphmd_update_hbonds", script)
        self.assertIn("scale by 4.000000", script)
        self.assertIn("trace variable ::vmd_frame", script)
        self.assertIn("set ::cphmd_hidden(1) {12}", script)
        self.assertIn("set ::cphmd_states {0 1}", script)


if __name__ == "__main__":
    unittest.main()
