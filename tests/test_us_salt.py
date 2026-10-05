import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from qmmd.us_salt import (
    WaterDistance,
    load_config,
    prepare_us_salt,
    select_common_water,
    summarize_water_distances,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/BV/us/salt-dih.yaml"


class USSaltTests(unittest.TestCase):
    def test_bv_config(self):
        cfg = load_config(CONFIG)
        self.assertEqual(cfg.counterion, "Cl-")
        self.assertEqual(cfg.delete_h, "H1")
        self.assertEqual(cfg.minimum_solute_distance_angstrom, 6.0)

    def test_common_water_maximizes_worst_window_distance(self):
        summaries = [
            WaterDistance(4, 10, 7.0, 9.0, 11.0),
            WaterDistance(5, 13, 8.0, 8.1, 8.2),
        ]
        self.assertEqual(select_common_water(summaries).residue, 5)

    def test_distance_parser_requires_every_frame(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "closest.dat"
            path.write_text(
                "# header\n"
                "1 3 7.0 10\n2 3 8.0 10\n"
                "1 4 9.0 13\n2 4 6.0 13\n"
                "1 5 99.0 16\n"
            )
            summaries = summarize_water_distances(path, 2)
            self.assertEqual([item.residue for item in summaries], [3, 4])
            self.assertEqual(select_common_water(summaries).residue, 3)

    def test_prepare_layout_with_one_identical_topology(self):
        cfg = load_config(CONFIG)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp)
            (base / "prep").mkdir()
            (base / "prep/solv.parm7").write_text("topology\n")
            pull_root = base / "us-single15"
            pull_root.mkdir()
            (pull_root / "pull_spec.yaml").write_text(cfg.pull_yaml.read_text())
            for index in range(19):
                stage = pull_root / f"window-{index:03d}" / "pull"
                stage.mkdir(parents=True)
                (stage / "pull.rst7").write_text("restart\n")
                (stage / "pull.out").write_text("Final Performance Info:\n")

            test_cfg = replace(cfg, pull=replace(cfg.pull, system_dir=str(base)))

            def fake_runner(input_path: Path, _log_path: Path, _module: str) -> None:
                text = input_path.read_text()
                if input_path.name == "select_common_water.in":
                    output = Path(re.search(r'closestout "([^"]+)"', text).group(1))
                    rows = ["# Frame Mol Dist FirstAtm"]
                    for frame in range(1, 20):
                        rows.append(f"{frame} 3 {7.0 + frame / 100:.3f} 10")
                        rows.append(f"{frame} 4 {8.0 + frame / 100:.3f} 13")
                    output.write_text("\n".join(rows) + "\n")
                    return
                parmwrite = re.search(r'parmwrite out "([^"]+)"', text)
                if parmwrite:
                    Path(parmwrite.group(1)).write_text("common topology\n")
                    return
                for output, kind in re.findall(r'trajout "([^"]+)" (restart|xyz)', text):
                    path = Path(output)
                    if kind == "restart":
                        path.write_text("restart\n")
                    else:
                        path.write_text('1\nLattice="10 0 0 0 10 0 0 0 10"\nH 0 0 0\n')

            common, windows, selected = prepare_us_salt(
                test_cfg, CONFIG.read_text(), ROOT, runner=fake_runner
            )
            self.assertEqual(selected.residue, 4)
            self.assertEqual(len(windows), 19)
            self.assertTrue((common / "water_distance_summary.csv").is_file())
            hashes = {(window / "ready.parm7").read_text() for window in windows}
            self.assertEqual(hashes, {"common topology\n"})
            self.assertTrue((windows[0] / "ready.xyz").is_file())
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                prepare_us_salt(test_cfg, CONFIG.read_text(), ROOT, runner=fake_runner)


if __name__ == "__main__":
    unittest.main()
