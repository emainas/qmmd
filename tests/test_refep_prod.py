import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from qmmd.refep_prod import (
    load_refep_prod_config,
    prepare_refep_prod,
    render_groupfile,
    render_prod_mdin,
    render_provenance,
    render_run_script,
    render_slurm_script,
    submit_refep_prod,
    validate_completed_equilibration,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "MEA" / "refep" / "prod.yaml"
ANTI_CONFIG = REPO / "configs" / "PRN-anti" / "refep" / "prod.yaml"
SYN_CONFIG = REPO / "configs" / "PRN-syn" / "refep" / "prod.yaml"
ANTI_IMPLICIT_CONFIG = (
    REPO / "configs" / "PRN-anti" / "refep" / "prod-implicit.yaml"
)


class RefepProdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_refep_prod_config(CONFIG)

    def test_mea_production_protocol(self):
        self.assertEqual(self.cfg.equil.prep.windows, 16)
        self.assertEqual(self.cfg.runtime.rem_mode, 3)
        self.assertEqual(self.cfg.runtime.processes_per_replica, 2)
        self.assertEqual(self.cfg.slurm.ntasks, 32)
        self.assertEqual(self.cfg.cntrl["ntb"], 1)
        self.assertEqual(self.cfg.cntrl["ntp"], 0)
        self.assertEqual(self.cfg.cntrl["cut"], 5.0)
        self.assertIsNone(self.cfg.restraint)
        total_ps = (
            self.cfg.cntrl["nstlim"]
            * self.cfg.cntrl["numexchg"]
            * self.cfg.cntrl["dt"]
        )
        self.assertEqual(total_ps, 10000.0)

    def test_completed_equilibration_is_accepted(self):
        windows = validate_completed_equilibration(self.cfg)
        self.assertEqual(len(windows), 16)
        self.assertEqual(windows[0].lambda_value, "0.0000000000")
        self.assertEqual(windows[-1].lambda_value, "1.0000000000")
        self.assertTrue(windows[0].restart.name == "equil.rst7")

    def test_rendered_hremd_inputs(self):
        mdin = render_prod_mdin(self.cfg)
        self.assertIn("nstlim=500", mdin)
        self.assertIn("numexchg=10000", mdin)
        self.assertIn("ntb=1", mdin)
        self.assertIn("ntp=0", mdin)
        self.assertNotIn("DISANG", mdin)

        groupfile = render_groupfile(self.cfg)
        lines = groupfile.splitlines()
        self.assertEqual(len(lines), 16)
        self.assertIn("-p lambda-000.parm7", lines[0])
        self.assertIn("-c lambda-000.equil.rst7", lines[0])
        self.assertIn("-x lambda-015.nc", lines[-1])

        run = render_run_script(self.cfg)
        self.assertIn('"$launcher" -np 32 "$mdexec"', run)
        self.assertIn("-ng 16", run)
        self.assertIn("-rem 3", run)
        self.assertIn("-remlog rem.log", run)

        slurm = render_slurm_script(self.cfg)
        self.assertIn("#SBATCH -N 2", slurm)
        self.assertIn("#SBATCH -n 32", slurm)
        self.assertIn("#SBATCH --ntasks-per-node=16", slurm)

        provenance = render_provenance(self.cfg)
        self.assertIn("total_time_ps_per_replica: 10000.0", provenance)

    def test_production_uses_matching_equilibration_restraint(self):
        anti = load_refep_prod_config(ANTI_CONFIG)
        syn = load_refep_prod_config(SYN_CONFIG)
        anti_implicit = load_refep_prod_config(ANTI_IMPLICIT_CONFIG)
        assert anti.restraint is not None
        assert syn.restraint is not None
        self.assertEqual(anti.restraint, anti.equil.restraint)
        self.assertEqual(syn.restraint, syn.equil.restraint)
        self.assertEqual(anti_implicit.restraint, anti.restraint)
        self.assertEqual(anti.restraint.target_deg, 180.0)
        self.assertEqual(syn.restraint.target_deg, 0.0)
        anti_mdin = render_prod_mdin(anti)
        self.assertIn("nmropt=1", anti_mdin)
        self.assertIn("&wt type='END' /", anti_mdin)
        self.assertIn("DISANG=restraint.rst", anti_mdin)
        provenance = render_provenance(anti)
        self.assertIn("atoms_one_based:", provenance)
        self.assertIn("force_constant_kcal_mol_rad2: 50.0", provenance)

    def test_production_rejects_restraint_mismatch(self):
        text = ANTI_CONFIG.read_text().replace(
            "equil_yaml: equil.yaml",
            f"equil_yaml: {ANTI_CONFIG.with_name('equil.yaml').resolve()}",
            1,
        )
        text = text.replace("target_deg: 180.0", "target_deg: 0.0", 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prod.yaml"
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "must exactly match"):
                load_refep_prod_config(path)

    def test_prepare_copies_lambda_specific_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "prod"
            prepare_refep_prod(self.cfg, destination)

            self.assertEqual(len(list(destination.glob("lambda-*.parm7"))), 16)
            self.assertEqual(len(list(destination.glob("lambda-*.equil.rst7"))), 16)
            self.assertEqual(
                (destination / "lambda-000.parm7").read_bytes(),
                (
                    self.cfg.equil.prep.output_directory.parent
                    / "equil"
                    / "lambda-000"
                    / "system.parm7"
                ).read_bytes(),
            )
            self.assertEqual(
                (destination / "lambda-015.equil.rst7").read_bytes(),
                (
                    self.cfg.equil.prep.output_directory.parent
                    / "equil"
                    / "lambda-015"
                    / "equil.rst7"
                ).read_bytes(),
            )
            for name in (
                "prod.mdin",
                "groupfile",
                "lambda-manifest.csv",
                "provenance.yaml",
                "refep-prod-spec.yaml",
                "run.sh",
                "slurm.sh",
            ):
                self.assertTrue((destination / name).is_file())

    def test_prepare_refuses_to_overwrite_existing_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "prod"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_refep_prod(self.cfg, destination)

    def test_submit_preflight_accepts_exact_prepared_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "prod"
            prepare_refep_prod(self.cfg, destination)
            submitted = []
            with redirect_stdout(StringIO()):
                result = submit_refep_prod(
                    self.cfg,
                    CONFIG.read_text(),
                    confirm=lambda _prompt: "yes",
                    submitter=submitted.append,
                    output_dir=destination,
                )
            self.assertTrue(result)
            self.assertEqual(submitted, [destination / "slurm.sh"])

    def test_submit_preflight_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "prod"
            prepare_refep_prod(self.cfg, destination)
            mdin = destination / "prod.mdin"
            mdin.write_text(mdin.read_text() + "! changed\n")
            submitted = []
            with redirect_stdout(StringIO()):
                result = submit_refep_prod(
                    self.cfg,
                    CONFIG.read_text(),
                    confirm=lambda _prompt: "yes",
                    submitter=submitted.append,
                    output_dir=destination,
                )
            self.assertFalse(result)
            self.assertEqual(submitted, [])

    def test_submit_preflight_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "prod"
            prepare_refep_prod(self.cfg, destination)
            (destination / "rem.log").write_text("existing\n")
            submitted = []
            with redirect_stdout(StringIO()):
                result = submit_refep_prod(
                    self.cfg,
                    CONFIG.read_text(),
                    confirm=lambda _prompt: "yes",
                    submitter=submitted.append,
                    output_dir=destination,
                )
            self.assertFalse(result)
            self.assertEqual(submitted, [])


if __name__ == "__main__":
    unittest.main()
