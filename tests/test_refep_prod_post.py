import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from qmmd.refep_prod_post import (
    load_refep_prod_post_config,
    prepare_refep_prod_post,
    render_energy_mdin,
    render_energy_run_script,
    render_local_run_script,
    render_post_run_script,
    render_post_slurm_script,
    render_task_manifest,
    submit_refep_prod_post,
    validate_completed_production,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "MEA" / "refep" / "prod.yaml"
RESCORE_CONFIG = REPO / "configs" / "MEA" / "refep" / "prod-igb2-rescore.yaml"
PRN_ANTI_IMPLICIT_CONFIG = (
    REPO / "configs" / "PRN-anti" / "refep" / "prod-implicit.yaml"
)


class RefepProdPostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_refep_prod_post_config(CONFIG)

    def test_mea_single_point_protocol(self):
        self.assertEqual(self.cfg.prod.equil.prep.windows, 16)
        self.assertEqual(self.cfg.cntrl["imin"], 5)
        self.assertEqual(self.cfg.cntrl["maxcyc"], 1)
        self.assertEqual(self.cfg.cntrl["cut"], 5.0)
        self.assertEqual(self.cfg.slurm.ntasks, 256)

    def test_completed_production_is_accepted(self):
        windows = validate_completed_production(self.cfg)
        self.assertEqual(len(windows), 16)
        self.assertEqual({window.frame_count for window in windows}, {1000})

    def test_rendered_grid_inputs(self):
        windows = validate_completed_production(self.cfg)
        mdin = render_energy_mdin(self.cfg)
        self.assertIn("imin=5", mdin)
        self.assertIn("maxcyc=1", mdin)
        self.assertIn("cut=5.0", mdin)

        manifest = render_task_manifest(self.cfg, windows)
        self.assertEqual(len(manifest.splitlines()), 257)
        self.assertIn("refep-j000-k000", manifest)
        self.assertIn("refep-j015-k015", manifest)

        energy_run = render_energy_run_script(self.cfg)
        self.assertIn('-y "../$trajectory"', energy_run)
        self.assertIn('energies/$stem.mdout', energy_run)

        local_run = render_local_run_script(self.cfg)
        self.assertIn("REFEP_LOCAL_WORKERS", local_run)
        self.assertIn("all 256 REFEP single-point calculations completed locally", local_run)

        run = render_post_run_script(self.cfg)
        self.assertIn("--nodes=1 --ntasks=1", run)
        self.assertIn("all 256 REFEP single-point calculations completed", run)

        slurm = render_post_slurm_script(self.cfg)
        self.assertIn("#SBATCH -N 4", slurm)
        self.assertIn("#SBATCH -n 256", slurm)
        self.assertIn("#SBATCH --ntasks-per-node=64", slurm)

    def test_prepare_and_submit_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "post"
            prepare_refep_prod_post(self.cfg, destination)
            self.assertTrue((destination / "energies").is_dir())
            self.assertTrue((destination / "run-local.sh").is_file())
            self.assertEqual(list((destination / "energies").iterdir()), [])
            submitted = []
            with redirect_stdout(StringIO()):
                result = submit_refep_prod_post(
                    self.cfg,
                    CONFIG.read_text(),
                    confirm=lambda _prompt: "yes",
                    submitter=submitted.append,
                    output_dir=destination,
                )
            self.assertTrue(result)
            self.assertEqual(submitted, [destination / "slurm.sh"])

    def test_submit_rejects_existing_energy_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "post"
            prepare_refep_prod_post(self.cfg, destination)
            (destination / "energies" / "refep-j000-k000.mdout").write_text(
                "existing\n"
            )
            submitted = []
            with redirect_stdout(StringIO()):
                result = submit_refep_prod_post(
                    self.cfg,
                    CONFIG.read_text(),
                    confirm=lambda _prompt: "yes",
                    submitter=submitted.append,
                    output_dir=destination,
                )
            self.assertFalse(result)
            self.assertEqual(submitted, [])

    def test_igb2_rescore_configuration(self):
        cfg = load_refep_prod_post_config(RESCORE_CONFIG)
        self.assertEqual(cfg.keep_mask, ":1")
        self.assertEqual(cfg.stripped_dirname, "stripped")
        self.assertEqual(cfg.cntrl["ntb"], 0)
        self.assertEqual(cfg.cntrl["igb"], 2)
        self.assertEqual(cfg.cntrl["saltcon"], 0.1)
        self.assertEqual(cfg.cntrl["cut"], 999.0)
        self.assertEqual(cfg.runtime.local_workers, 16)
        manifest = render_task_manifest(cfg, validate_completed_production(cfg))
        self.assertIn("stripped/lambda-000.nc", manifest)
        self.assertIn("stripped/lambda-015.parm7", manifest)

    def test_slurm_grid_can_run_in_bounded_waves(self):
        cfg = load_refep_prod_post_config(PRN_ANTI_IMPLICIT_CONFIG)
        self.assertEqual(cfg.slurm.ntasks, 96)
        run = render_post_run_script(cfg)
        self.assertIn('workers="${SLURM_NTASKS:-96}"', run)
        self.assertIn("if (( ${#pids[@]} >= workers )); then", run)
        self.assertIn("< /dev/null", run)
        self.assertIn("if (( launched != 256 )); then", run)
        slurm = render_post_slurm_script(cfg)
        self.assertIn("#SBATCH -N 2", slurm)
        self.assertIn("#SBATCH -n 96", slurm)
        self.assertIn("#SBATCH --ntasks-per-node=48", slurm)


if __name__ == "__main__":
    unittest.main()
