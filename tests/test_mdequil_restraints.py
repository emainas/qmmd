import tempfile
import unittest
from pathlib import Path

from qmmd.mdequil import MDStage, load_config, render_mdin, render_dihedral_restraints, write_mdin_files


class RestraintTests(unittest.TestCase):
    def test_old_input_unchanged(self):
        stage = MDStage('test', {'imin': 1})
        self.assertEqual(render_mdin(stage), 'test\n&cntrl\n  imin=1,\n/\n')

    def test_restraints_all_stages_preserve_heating(self):
        root = Path(__file__).resolve().parents[1]
        for state, target in [('anti', 180), ('syn', 0)]:
            cfg = load_config(root / f'configs/PRN-{state}/mdequil/mdequil.yaml')
            self.assertEqual(cfg.buffer, 6.5)
            self.assertEqual(cfg.md.equilibrate_npt.cntrl['nstlim'], 10000000)
            with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
                out = Path(tmp)
                write_mdin_files(cfg, out)
                for name in ['min', 'heat', 'equil-nvt', 'equil-npt']:
                    text = (out / f'{name}.in').read_text()
                    self.assertIn('cut=6.0,', text)
                    self.assertIn('nmropt=1,', text)
                    self.assertIn('DISANG=dihedral.rst', text)
                    self.assertEqual(text.count("type='END'"), 1)
                self.assertIn("type='TEMP0'", (out / 'heat.in').read_text())
                self.assertIn(f'r2={float(target)}', (out / 'dihedral.rst').read_text())

    def test_invalid_restraints(self):
        for atoms, target, strength in [([0, 3, 4, 11], 0, 50),
                                        ([5, 3, 4, 4], 0, 50),
                                        ([5, 3, 4, 11], float('nan'), 50),
                                        ([5, 3, 4, 11], 0, -1)]:
            with self.assertRaises(ValueError):
                render_dihedral_restraints([dict(atoms=atoms, target_deg=target, force_constant=strength)])
