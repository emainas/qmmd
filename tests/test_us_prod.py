import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from qmmd.us_prod import (
    dftb_terminated_normally,
    load_config,
    prepare_us_prod,
    production_duration_ps,
    submit_us_prod,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/PRN-anti/us/prod.yaml"


class USProdTests(unittest.TestCase):
    def _write_submit_tree(self, base: Path, yaml_text: str) -> None:
        root = base / "us-pull"
        for index in range(19):
            stage = root / f"window-{index:03d}" / "prod"
            stage.mkdir(parents=True)
            (stage / "prod_spec.yaml").write_text(yaml_text)
            for filename in (
                "dftb.inp",
                "metacv.dat",
                "restart",
                "run.sh",
                "slurm.sh",
            ):
                (stage / filename).write_text("prepared\n")

    def test_normal_completion_is_checked_at_output_tail(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "dftb.out"
            path.write_bytes(
                b"Execution of DCDFTBMD terminated normally\n"
                + b"x" * 20000
            )
            self.assertFalse(dftb_terminated_normally(path))
            path.write_bytes(
                b"x" * 20000
                + b"\nExecution of DCDFTBMD terminated normally\n"
            )
            self.assertTrue(dftb_terminated_normally(path))

    def test_config_is_40_ps_tight_restart_production(self):
        cfg = load_config(CONFIG)
        self.assertAlmostEqual(production_duration_ps(cfg), 40.0)
        self.assertEqual(cfg.run.wall.coefficient_kcal_mol_deg2, 0.05)
        self.assertIn("MD=(RESTART=TRUE READVELOCITY=FALSE)", cfg.run.dftb.header_lines)

    def test_prepares_all_windows_from_completed_equil_restarts(self):
        cfg = load_config(CONFIG)
        yaml_text = CONFIG.read_text()
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp)
            pull = replace(cfg.run.pull, system_dir=str(base))
            test_cfg = replace(
                cfg,
                run=replace(cfg.run, pull=pull),
                equil=replace(cfg.equil, pull=replace(cfg.equil.pull, system_dir=str(base))),
            )
            root = base / "us-pull"
            xyz = """11
Lattice="16 0 0 0 16 0 0 0 16"
C 0 0 0
C 1 0 0
C 2 0 0
O 3 0 0
O 4 0 0
H 5 0 0
H 6 0 0
H 7 0 0
H 8 0 0
H 9 0 0
H 10 0 0
"""
            for index in range(19):
                stage = root / f"window-{index:03d}" / "equil"
                stage.mkdir(parents=True)
                (stage / "equil_spec.yaml").write_text(cfg.equil_yaml.read_text())
                (stage / "dftb.out").write_text(
                    "Execution of DCDFTBMD terminated normally\n"
                )
                (stage / "restart").write_bytes(f"restart-{index}".encode())
                (stage / "start.xyz").write_text(xyz)

            outputs = prepare_us_prod(test_cfg, yaml_text, ROOT)
            self.assertEqual(len(outputs), 19)
            first = outputs[0]
            last = outputs[-1]
            self.assertEqual(first, root / "window-000/prod")
            self.assertEqual((first / "restart").read_bytes(), b"restart-0")
            self.assertIn("RESTART=TRUE", (first / "dftb.inp").read_text())
            self.assertIn("L 1 0.05 180 2", (first / "metacv.dat").read_text())
            self.assertIn("U 1 0.05 0 2", (last / "metacv.dat").read_text())
            self.assertTrue((first / "params/H-O.skf").exists())
            self.assertFalse((first / "dftb.out").exists())
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                prepare_us_prod(test_cfg, yaml_text, ROOT)

    def test_submit_complete_set_after_one_confirmation(self):
        cfg = load_config(CONFIG)
        yaml_text = CONFIG.read_text()
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp)
            pull = replace(cfg.run.pull, system_dir=str(base))
            test_cfg = replace(cfg, run=replace(cfg.run, pull=pull))
            self._write_submit_tree(base, yaml_text)
            prompts: list[str] = []
            submissions: list[Path] = []

            submitted = submit_us_prod(
                test_cfg,
                yaml_text,
                ROOT,
                confirm=lambda prompt: prompts.append(prompt) or "yes",
                submitter=submissions.append,
            )

            self.assertTrue(submitted)
            self.assertEqual(prompts, ["Proceed to submit 19 jobs? [y/N] "])
            self.assertEqual(len(submissions), 19)
            self.assertEqual(submissions[0], base / "us-pull/window-000/prod/slurm.sh")
            self.assertEqual(submissions[-1], base / "us-pull/window-018/prod/slurm.sh")

    def test_submit_preflight_is_all_or_nothing(self):
        cfg = load_config(CONFIG)
        yaml_text = CONFIG.read_text()
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp)
            pull = replace(cfg.run.pull, system_dir=str(base))
            test_cfg = replace(cfg, run=replace(cfg.run, pull=pull))
            self._write_submit_tree(base, yaml_text)
            (base / "us-pull/window-009/prod/restart").unlink()
            prompts: list[str] = []
            submissions: list[Path] = []

            submitted = submit_us_prod(
                test_cfg,
                yaml_text,
                ROOT,
                confirm=lambda prompt: prompts.append(prompt) or "yes",
                submitter=submissions.append,
            )

            self.assertFalse(submitted)
            self.assertEqual(prompts, [])
            self.assertEqual(submissions, [])


if __name__ == "__main__":
    unittest.main()
