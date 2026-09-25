import tempfile
import unittest
from pathlib import Path

from qmmd.cphmd_dgref import (
    load_config,
    prepare_dgref,
    read_charge_sets,
    read_topology_info,
    render_cpin,
    render_mdin,
    submit_dgref,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/MEA/cphmd/dgref.yaml"


class CpHMDDgrefTests(unittest.TestCase):
    def test_real_mea_inputs_and_cpin(self):
        cfg = load_config(CONFIG)
        charges = read_charge_sets(cfg.charge_sets)
        topology = read_topology_info(
            ROOT / "systems/MEA/solv_5.5/prep/solv.parm7", "MEA"
        )

        self.assertEqual(topology.atom_names, charges.atom_names)
        self.assertEqual(topology.first_atom, 1)
        self.assertEqual(topology.first_solvent, 13)
        cpin = render_cpin(cfg, charges, topology)
        self.assertIn("natchrg=24", cpin)
        self.assertIn("PROTCNT=3,2,", cpin)
        self.assertIn("STATENE=DELTAGREF,0.0,", cpin)
        self.assertIn("PKA_CORR=9.0000,0.0000,", cpin)
        self.assertIn("CPHFIRST_SOL=13, CPH_IGB=2", cpin)

    def test_mdin_has_explicit_solvent_cphmd_controls(self):
        mdin = render_mdin(load_config(CONFIG))
        for setting in (
            "ntb=1,",
            "ntp=0,",
            "cut=5.0,",
            "icnstph=2,",
            "ntcnstph=100,",
            "solvph=9.0,",
            "ntrelax=100,",
            "saltcon=0.1,",
        ):
            self.assertIn(setting, mdin)

    def test_preparation_writes_complete_bundle_without_running(self):
        cfg = load_config(CONFIG)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            destination = prepare_dgref(cfg, ROOT, Path(tmp) / "dgref")
            expected = {
                "solv.parm7",
                "equil-npt.rst7",
                "dgref.cpin",
                "dgref.mdin",
                "run.sh",
                "slurm.sh",
                "dgref_spec.yaml",
            }
            self.assertEqual({path.name for path in destination.iterdir()}, expected)
            self.assertIn("finddgref.py", (destination / "run.sh").read_text())
            slurm = (destination / "slurm.sh").read_text()
            self.assertIn("#SBATCH -N 2", slurm)
            self.assertNotIn("#SBATCH -n", slurm)
            self.assertIn("#SBATCH -t 5:00:00", slurm)
            self.assertIn("bash run.sh", slurm)
            self.assertFalse((destination / "dgref.out").exists())

    def test_submit_requires_confirmation_and_uses_sbatch_script(self):
        cfg = load_config(CONFIG)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            destination = prepare_dgref(cfg, ROOT, Path(tmp) / "dgref")
            submitted = []
            result = submit_dgref(
                cfg,
                CONFIG.read_text(),
                ROOT,
                confirm=lambda _: "yes",
                submitter=submitted.append,
                output_dir=destination,
            )
            self.assertTrue(result)
            self.assertEqual(submitted, [destination / "slurm.sh"])

    def test_submit_rejects_stale_or_changed_preparation(self):
        cfg = load_config(CONFIG)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            destination = prepare_dgref(cfg, ROOT, Path(tmp) / "dgref")
            (destination / "dgref.mdin").write_text("changed\n")
            submitted = []
            result = submit_dgref(
                cfg,
                CONFIG.read_text(),
                ROOT,
                confirm=lambda _: "yes",
                submitter=submitted.append,
                output_dir=destination,
            )
            self.assertFalse(result)
            self.assertEqual(submitted, [])


if __name__ == "__main__":
    unittest.main()
