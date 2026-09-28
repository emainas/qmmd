import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from qmmd.refep_equil import (
    load_refep_equil_config,
    prepare_refep_equil,
    render_equil_mdin,
    render_run_script,
    render_slurm_script,
    render_window_run_script,
    submit_refep_equil,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "MEA" / "refep" / "equil.yaml"


class RefepEquilTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_refep_equil_config(CONFIG)

    def test_mea_equilibration_is_independent_nvt(self):
        self.assertEqual(self.cfg.prep.windows, 16)
        self.assertEqual(self.cfg.stage_dirname, "equil")
        self.assertEqual(self.cfg.cntrl["irest"], 1)
        self.assertEqual(self.cfg.cntrl["ntx"], 5)
        self.assertEqual(self.cfg.cntrl["ntb"], 1)
        self.assertEqual(self.cfg.cntrl["ntp"], 0)
        self.assertEqual(self.cfg.cntrl["cut"], 5.0)
        self.assertAlmostEqual(
            self.cfg.cntrl["nstlim"] * self.cfg.cntrl["dt"], 200.0
        )

    def test_rendered_inputs_have_no_replica_exchange_controls(self):
        mdin = render_equil_mdin(self.cfg, 15)
        self.assertIn("lambda=1.0000000000", mdin)
        self.assertIn("irest=1", mdin)
        self.assertIn("ntb=1", mdin)
        self.assertIn("ntp=0", mdin)
        self.assertNotIn("numexchg", mdin)

        run = render_run_script(self.cfg)
        self.assertEqual(run.count("lambda-"), 16)
        self.assertIn('"$launcher" --exclusive --exact --ntasks=1', run)
        self.assertNotIn("-rem", run)

        window_run = render_window_run_script(self.cfg)
        self.assertIn('cd "$script_dir"', window_run)

        slurm = render_slurm_script(self.cfg)
        self.assertIn("#SBATCH -N 2", slurm)
        self.assertIn("#SBATCH -n 16", slurm)
        self.assertIn("#SBATCH --ntasks-per-node=8", slurm)

    def test_prepare_copies_validated_topologies_and_common_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "equil"
            prepare_refep_equil(self.cfg, destination)

            windows = sorted(destination.glob("lambda-[0-9][0-9][0-9]"))
            self.assertEqual(len(windows), 16)
            self.assertEqual(
                (windows[0] / "system.parm7").read_bytes(),
                (
                    self.cfg.prep.output_directory / "lambda-000.parm7"
                ).read_bytes(),
            )
            self.assertEqual(
                (windows[-1] / "start.rst7").read_bytes(),
                (self.cfg.prep.output_directory / "common.rst7").read_bytes(),
            )
            for window in windows:
                for name in ("system.parm7", "start.rst7", "equil.mdin", "run.sh"):
                    self.assertTrue((window / name).is_file())
            self.assertTrue((destination / "equil-manifest.csv").is_file())
            self.assertTrue((destination / "refep-equil-spec.yaml").is_file())
            self.assertTrue((destination / "slurm.sh").is_file())

    def test_prepare_refuses_to_overwrite_existing_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "equil"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_refep_equil(self.cfg, destination)

    def test_submit_preflight_accepts_exact_prepared_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "equil"
            prepare_refep_equil(self.cfg, destination)
            submitted = []
            with redirect_stdout(StringIO()):
                result = submit_refep_equil(
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
            destination = Path(tmp) / "equil"
            prepare_refep_equil(self.cfg, destination)
            mdin = destination / "lambda-007" / "equil.mdin"
            mdin.write_text(mdin.read_text() + "! changed\n")
            submitted = []
            with redirect_stdout(StringIO()):
                result = submit_refep_equil(
                    self.cfg,
                    CONFIG.read_text(),
                    confirm=lambda _prompt: "yes",
                    submitter=submitted.append,
                    output_dir=destination,
                )
            self.assertFalse(result)
            self.assertEqual(submitted, [])

    def test_submit_preflight_rejects_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "equil"
            prepare_refep_equil(self.cfg, destination)
            (destination / "lambda-000" / "equil.mdout").write_text("existing\n")
            submitted = []
            with redirect_stdout(StringIO()):
                result = submit_refep_equil(
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
