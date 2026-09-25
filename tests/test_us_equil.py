import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from qmmd.us_equil import (
    load_config,
    prepare_us_equil,
    render_metacv,
    submit_us_equil,
    validate_zero_height_metawall,
)
from qmmd.us_pull import window_centers


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/PRN-anti/us/equil.yaml"


class USEquilTests(unittest.TestCase):
    def _write_submit_tree(self, base: Path, yaml_text: str) -> None:
        root = base / "us-pull"
        for index in range(19):
            stage = root / f"window-{index:03d}" / "equil"
            stage.mkdir(parents=True)
            (stage / "equil_spec.yaml").write_text(yaml_text)
            for filename in ("dftb.inp", "metacv.dat", "run.sh", "slurm.sh"):
                (stage / filename).write_text("prepared\n")

    def test_config_and_wall_plan(self):
        cfg = load_config(CONFIG)
        self.assertEqual(len(window_centers(cfg.pull.windows)), 19)
        self.assertAlmostEqual(
            cfg.wall.coefficient_kcal_mol_deg2,
            50.0 * (3.141592653589793 / 180.0) ** 2,
        )
        self.assertEqual(
            render_metacv(cfg, 170.0),
            "BONDDIHEDRAL 0.1 5 3 4 11\n\n"
            "L 1 0.015230870989 170 2\n"
            "U 1 0.015230870989 170 2\n",
        )

    def test_metadynamics_hack_is_enforced(self):
        valid = load_config(CONFIG).dftb.header_lines
        validate_zero_height_metawall(valid)
        with self.assertRaisesRegex(ValueError, "METAHEIGHT=0"):
            validate_zero_height_metawall(
                [line.replace("METAHEIGHT=0.0", "METAHEIGHT=0.001") for line in valid]
            )
        with self.assertRaisesRegex(ValueError, "METAPRINTFES=FALSE"):
            validate_zero_height_metawall(
                [line.replace("METAPRINTFES=FALSE", "METAPRINTFES=TRUE") for line in valid]
            )
        with self.assertRaisesRegex(ValueError, "too small"):
            validate_zero_height_metawall(
                [line.replace("METAMAXGAUSS=400", "METAMAXGAUSS=399") for line in valid]
            )

    def test_non_mapping_yaml_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "equil.yaml"
            path.write_text("[]\n")
            with self.assertRaisesRegex(ValueError, "YAML mapping"):
                load_config(path)

    def test_preparation_layout_in_tmp_without_running_dftb(self):
        cfg = load_config(CONFIG)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp)
            pull_cfg = replace(cfg.pull, system_dir=str(base))
            test_cfg = replace(cfg, pull=pull_cfg)
            (base / "prep").mkdir()
            (base / "prep/solv.parm7").write_text("test topology\n")
            pull_root = base / "us-pull"
            pull_root.mkdir()
            (pull_root / "pull_spec.yaml").write_text(cfg.pull_yaml.read_text())
            for index in range(19):
                stage = pull_root / f"window-{index:03d}" / "pull"
                stage.mkdir(parents=True)
                (stage / "pull.rst7").write_text("restart\n")
                (stage / "pull.out").write_text("Final Performance Info:\n")

            def fake_converter(_topology, _restart, xyz_path, _module):
                symbols = ["C", "C", "C", "O", "O", "H", "H", "H", "H", "H", "H"]
                lines = [
                    "11",
                    'Lattice="16 0 0 0 16 0 0 0 16"',
                    *[
                        f"{symbol} {atom_index:.1f} 0.0 0.0"
                        for atom_index, symbol in enumerate(symbols)
                    ],
                ]
                xyz_path.write_text("\n".join(lines) + "\n")

            outputs = prepare_us_equil(
                test_cfg,
                CONFIG.read_text(),
                ROOT,
                converter=fake_converter,
            )
            self.assertEqual(len(outputs), 19)
            first = outputs[0]
            last = outputs[-1]
            self.assertEqual(first, pull_root / "window-000/equil")
            self.assertIn("METAHEIGHT=0.0", (first / "dftb.inp").read_text())
            self.assertIn("RANDOMSEED=0", CONFIG.read_text())
            self.assertNotIn("RANDOMSEED=0", (first / "dftb.inp").read_text())
            self.assertIn("L 1 0.015230870989 180 2", (first / "metacv.dat").read_text())
            self.assertIn("U 1 0.015230870989 0 2", (last / "metacv.dat").read_text())
            self.assertTrue((first / "params/H-O.skf").exists())
            self.assertFalse((first / "dftb.out").exists())
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                prepare_us_equil(
                    test_cfg,
                    CONFIG.read_text(),
                    ROOT,
                    converter=fake_converter,
                )

    def test_submit_complete_set_after_one_confirmation(self):
        cfg = load_config(CONFIG)
        yaml_text = CONFIG.read_text()
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp)
            test_cfg = replace(cfg, pull=replace(cfg.pull, system_dir=str(base)))
            self._write_submit_tree(base, yaml_text)
            prompts: list[str] = []
            submissions: list[Path] = []

            submitted = submit_us_equil(
                test_cfg,
                yaml_text,
                ROOT,
                confirm=lambda prompt: prompts.append(prompt) or "yes",
                submitter=submissions.append,
            )

            self.assertTrue(submitted)
            self.assertEqual(prompts, ["Proceed to submit 19 jobs? [y/N] "])
            self.assertEqual(len(submissions), 19)
            self.assertEqual(submissions[0], base / "us-pull/window-000/equil/slurm.sh")
            self.assertEqual(submissions[-1], base / "us-pull/window-018/equil/slurm.sh")

    def test_submit_preflight_is_all_or_nothing(self):
        cfg = load_config(CONFIG)
        yaml_text = CONFIG.read_text()
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp)
            test_cfg = replace(cfg, pull=replace(cfg.pull, system_dir=str(base)))
            self._write_submit_tree(base, yaml_text)
            (base / "us-pull/window-007/equil/equil_spec.yaml").write_text("changed: true\n")
            prompts: list[str] = []
            submissions: list[Path] = []

            submitted = submit_us_equil(
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
