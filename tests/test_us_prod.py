import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from qmmd.us_prod import (
    dftb_terminated_normally,
    last_dftb_progress,
    load_config,
    prepare_us_prod,
    production_duration_ps,
    read_last_complete_trajectory_frame,
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
                "equil_source.json",
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

    def test_last_progress_is_read_from_output_tail(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "dftb.out"
            path.write_text(
                " *** AT T= 37405.00 FSEC, THIS RUN'S STEP NO.= 37405\n"
            )
            self.assertEqual(last_dftb_progress(path), (37405, 37.405))

    def test_last_complete_trajectory_frame_ignores_partial_tail(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "traject"
            path.write_text(
                "2\n"
                " *** AT T= 10.00 FSEC, THIS RUN'S STEP NO.= 10\n"
                "H 0 0 0\nO 1 0 0\n"
                "2\n"
                " *** AT T= 20.00 FSEC, THIS RUN'S STEP NO.= 20\n"
                "H 2 0 0\n"
            )
            frame = read_last_complete_trajectory_frame(path)
            self.assertEqual(frame.step, 10)
            self.assertAlmostEqual(frame.time_ps, 0.01)
            self.assertEqual(frame.coordinates[-1], ("O", 1.0, 0.0, 0.0))

    def test_bv_config_uses_last_trajectory_frame_for_40_ps(self):
        cfg = load_config(ROOT / "configs/BV/us/prod-dih.yaml")
        self.assertEqual(cfg.equil_source, "last_trajectory_frame")
        self.assertAlmostEqual(production_duration_ps(cfg), 40.0)
        self.assertIn("MD=(RESTART=FALSE READVELOCITY=FALSE)", cfg.run.dftb.header_lines)

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
            self.assertTrue((first / "equil_source.json").is_file())
            self.assertIn("RESTART=TRUE", (first / "dftb.inp").read_text())
            self.assertIn("L 1 0.05 180 2", (first / "metacv.dat").read_text())
            self.assertIn("U 1 0.05 0 2", (last / "metacv.dat").read_text())
            self.assertTrue((first / "params/H-O.skf").exists())
            self.assertFalse((first / "dftb.out").exists())
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                prepare_us_prod(test_cfg, yaml_text, ROOT)

    def test_prepares_from_last_complete_trajectory_frame_without_restart(self):
        cfg = load_config(CONFIG)
        yaml_text = CONFIG.read_text()
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp)
            pull = replace(cfg.run.pull, system_dir=str(base))
            header = [
                line.replace("NSTEP=80000", "NSTEP=40000").replace(
                    "RESTART=TRUE", "RESTART=FALSE"
                )
                for line in cfg.run.dftb.header_lines
            ]
            test_cfg = replace(
                cfg,
                run=replace(
                    cfg.run,
                    pull=pull,
                    dftb=replace(cfg.run.dftb, header_lines=header),
                ),
                equil=replace(
                    cfg.equil,
                    pull=replace(cfg.equil.pull, system_dir=str(base)),
                ),
                equil_source="last_trajectory_frame",
            )
            root = base / "us-pull"
            symbols = ["C"] * 3 + ["O"] * 2 + ["H"] * 6
            xyz = (
                "11\nLattice=\"16 0 0 0 16 0 0 0 16\"\n"
                + "\n".join(f"{symbol} 0 0 0" for symbol in symbols)
                + "\n"
            )
            for index in range(19):
                stage = root / f"window-{index:03d}" / "equil"
                stage.mkdir(parents=True)
                (stage / "equil_spec.yaml").write_text(cfg.equil_yaml.read_text())
                (stage / "start.xyz").write_text(xyz)
                coordinates = "\n".join(
                    f"{symbol} {index + atom / 10:.1f} 0 0"
                    for atom, symbol in enumerate(symbols)
                )
                (stage / "traject").write_text(
                    "11\n"
                    " *** AT T= 12500.00 FSEC, THIS RUN'S STEP NO.= 12500\n"
                    f"{coordinates}\n"
                    "11\n"
                    " *** AT T= 12510.00 FSEC, THIS RUN'S STEP NO.= 12510\n"
                    "C 99 0 0\n"
                )

            outputs = prepare_us_prod(test_cfg, yaml_text, ROOT)
            first = outputs[0]
            self.assertFalse((first / "restart").exists())
            self.assertTrue((first / "equil_last_frame.xyz").is_file())
            self.assertIn("source_step=12500", (first / "equil_last_frame.xyz").read_text())
            self.assertIn("RESTART=FALSE", (first / "dftb.inp").read_text())
            source = __import__("json").loads(
                (first / "equil_source.json").read_text()
            )
            self.assertEqual(source["source_mode"], "last_trajectory_frame")
            self.assertEqual(source["selected_frame_step"], 12500)
            self.assertFalse(source["velocities_preserved"])

    def test_incomplete_equilibration_requires_explicit_override(self):
        cfg = load_config(CONFIG)
        yaml_text = CONFIG.read_text()
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp)
            pull = replace(cfg.run.pull, system_dir=str(base))
            test_cfg = replace(
                cfg,
                run=replace(cfg.run, pull=pull),
                equil=replace(cfg.equil, pull=replace(cfg.equil.pull, system_dir=str(base))),
                allow_incomplete_equilibration=True,
            )
            root = base / "us-pull"
            xyz = "11\nLattice=\"16 0 0 0 16 0 0 0 16\"\n" + "\n".join(
                ["C 0 0 0"] * 3 + ["O 0 0 0"] * 2 + ["H 0 0 0"] * 6
            ) + "\n"
            for index in range(19):
                stage = root / f"window-{index:03d}" / "equil"
                stage.mkdir(parents=True)
                (stage / "equil_spec.yaml").write_text(cfg.equil_yaml.read_text())
                (stage / "dftb.out").write_text(
                    " *** AT T= 39000.00 FSEC, THIS RUN'S STEP NO.= 39000\n"
                )
                (stage / "restart").write_bytes(f"restart-{index}".encode())
                (stage / "start.xyz").write_text(xyz)

            outputs = prepare_us_prod(test_cfg, yaml_text, ROOT)
            source = __import__("json").loads(
                (outputs[0] / "equil_source.json").read_text()
            )
            self.assertFalse(source["normal_termination"])
            self.assertTrue(source["incomplete_equilibration_override"])
            self.assertEqual(source["last_dftb_output_step"], 39000)

    def test_incomplete_override_can_be_limited_to_named_windows(self):
        cfg = load_config(CONFIG)
        yaml_text = CONFIG.read_text()
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp)
            pull = replace(cfg.run.pull, system_dir=str(base))
            test_cfg = replace(
                cfg,
                run=replace(cfg.run, pull=pull),
                equil=replace(
                    cfg.equil,
                    pull=replace(cfg.equil.pull, system_dir=str(base)),
                ),
                allow_incomplete_equilibration=True,
                incomplete_equilibration_windows=(0,),
            )
            root = base / "us-pull"
            xyz = "11\nLattice=\"16 0 0 0 16 0 0 0 16\"\n" + "\n".join(
                ["C 0 0 0"] * 3 + ["O 0 0 0"] * 2 + ["H 0 0 0"] * 6
            ) + "\n"
            for index in range(19):
                stage = root / f"window-{index:03d}" / "equil"
                stage.mkdir(parents=True)
                (stage / "equil_spec.yaml").write_text(cfg.equil_yaml.read_text())
                (stage / "dftb.out").write_text(
                    " *** AT T= 39000.00 FSEC, THIS RUN'S STEP NO.= 39000\n"
                )
                (stage / "restart").write_bytes(f"restart-{index}".encode())
                (stage / "start.xyz").write_text(xyz)

            with self.assertRaisesRegex(
                RuntimeError, "window 1 is not in incomplete_equilibration_windows"
            ):
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
