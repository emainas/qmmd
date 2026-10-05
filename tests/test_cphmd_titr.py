import tempfile
import unittest
from pathlib import Path

from qmmd.cphmd_titr import (
    load_titr_config,
    prepare_titration,
    read_converged_dgref,
    render_groupfile,
    render_replica_mdin,
    render_run_script,
    render_slurm_script,
    submit_titration,
)
from qmmd.cphmd_dgref import read_charge_sets, read_topology_info


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "MEA" / "cphmd" / "cphmd.yaml"
PR4_CONFIG = REPO / "configs" / "PR4" / "cphmd" / "cphmd.yaml"


class CpHMDTitrPrepTests(unittest.TestCase):
    def test_mea_config_defines_valid_nvt_ladder(self):
        cfg = load_titr_config(CONFIG)
        self.assertEqual(cfg.system, "MEA")
        self.assertEqual(cfg.job_name, "titr")
        self.assertEqual(cfg.ph_values, tuple(7.0 + 0.5 * i for i in range(10)))
        self.assertEqual(cfg.cntrl["ntb"], 1)
        self.assertEqual(cfg.cntrl["ntp"], 0)
        self.assertEqual(cfg.cntrl["cut"], 5.0)

    def test_pr4_config_preserves_all_acid_microstates(self):
        cfg = load_titr_config(PR4_CONFIG)
        charges = read_charge_sets(cfg.charge_sets)
        topology = read_topology_info(
            REPO / "systems" / "PR4" / "solv_5.5" / "prep" / cfg.input_parm7,
            cfg.system,
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            destination, dgref = prepare_titration(cfg, REPO, Path(tmp) / "titr")
            cpin = (destination / "replica-01.cpin").read_text()

        self.assertEqual(len(cfg.pka_corr), 5)
        self.assertEqual(charges.proton_counts, (0, 1, 1, 1, 1))
        self.assertEqual(topology.atom_names, charges.atom_names)
        self.assertIn("ntstates=5", cpin)
        self.assertIn("PROTCNT=0,1,1,1,1,", cpin)
        self.assertEqual(cpin.count(f"{dgref:.6f}"), 4)

    def test_reads_only_successful_final_dgref(self):
        successful = """
The value of DELTAGREF that gives a converged fraction is: DELTAGREF = -26.742626 kcal/mol
The execution of finddgref.py ended with success.
"""
        incomplete = "AMBER execution #33: DELTAGREF = -26.742626 kcal/mol\n"
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            log = Path(tmp) / "dgref.log"
            log.write_text(successful)
            self.assertEqual(read_converged_dgref(log), -26.742626)
            log.write_text(incomplete)
            with self.assertRaisesRegex(ValueError, "did not finish successfully"):
                read_converged_dgref(log)

    def test_replica_inputs_and_launch_are_consistent(self):
        cfg = load_titr_config(CONFIG)
        mdin = render_replica_mdin(cfg, 1, cfg.ph_values[0])
        self.assertIn("solvph=7", mdin)
        self.assertIn("numexchg=50", mdin)

        groupfile = render_groupfile(cfg)
        lines = groupfile.splitlines()
        self.assertEqual(len(lines), len(cfg.ph_values))
        self.assertIn("replica-01.mdin", lines[0])
        self.assertIn("replica-10.cprestrt", lines[-1])

        run_script = render_run_script(cfg)
        self.assertIn('"$launcher" -np 20 "$mdexec"', run_script)
        self.assertIn("-ng 10", run_script)
        self.assertIn("-rem 4", run_script)
        self.assertIn("pmemd.MPI", run_script)

        slurm = render_slurm_script(cfg)
        self.assertIn("#SBATCH -N 2", slurm)
        self.assertIn("#SBATCH -n 20", slurm)

    def test_submit_requires_confirmation_and_uses_sbatch_script(self):
        cfg = load_titr_config(CONFIG)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            destination, _dgref = prepare_titration(
                cfg,
                REPO,
                Path(tmp) / "titr",
            )
            submitted = []
            result = submit_titration(
                cfg,
                CONFIG.read_text(),
                REPO,
                confirm=lambda _: "yes",
                submitter=submitted.append,
                output_dir=destination,
            )
            self.assertTrue(result)
            self.assertEqual(submitted, [destination / "slurm.sh"])

    def test_submit_rejects_changed_input_or_existing_results(self):
        cfg = load_titr_config(CONFIG)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            destination, _dgref = prepare_titration(
                cfg,
                REPO,
                Path(tmp) / "titr",
            )
            (destination / "replica-03.mdin").write_text("changed\n")
            submitted = []
            self.assertFalse(
                submit_titration(
                    cfg,
                    CONFIG.read_text(),
                    REPO,
                    confirm=lambda _: "yes",
                    submitter=submitted.append,
                    output_dir=destination,
                )
            )
            self.assertEqual(submitted, [])

            prepare_titration(cfg, REPO, destination)
            (destination / "replica-01.mdout").write_text("existing output\n")
            self.assertFalse(
                submit_titration(
                    cfg,
                    CONFIG.read_text(),
                    REPO,
                    confirm=lambda _: "yes",
                    submitter=submitted.append,
                    output_dir=destination,
                )
            )
            self.assertEqual(submitted, [])


if __name__ == "__main__":
    unittest.main()
