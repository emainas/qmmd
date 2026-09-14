import tempfile
import unittest
from pathlib import Path
import yaml
from qmmd.lcod import load_config, render_metacv, run_lcod


class TestLcod(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'pyproject.toml').touch()
        self.path = self.root/'lcod.yaml'
        self.data = dict(system='HPD', buffer=5.5, bench_tag='N1T48C1',run_ids='1-2',
                         lcod=dict(gaussian_width=.1, atoms=[1,8,7,8]))
        self.write()
        for r in [1,2]:
            p=self.root/f'systems/HPD/solv_5.5/dftb/N1T48C1/run-{r}/equil'
            p.mkdir(parents=True)
            (p/'dftb.inp').write_text('8 0 1\n'+'N 0 0 0\n'*8)

    def write(self):
        self.path.write_text(yaml.safe_dump(self.data))

    def test_render_and_preserve(self):
        self.assertEqual(render_metacv(load_config(self.path), ['N']*8),
                         'BONDDISTANCEDIFFERENCE 0.1 1 8 7 8\n')
        run_lcod(self.path)
        out=self.root/'systems/HPD/solv_5.5/dftb/N1T48C1/run-1/meta-lcod/metacv.dat'
        out.write_text('preserve')
        run_lcod(self.path)
        self.assertEqual(out.read_text(),'preserve')

    def test_validation_before_writes(self):
        inp=self.root/'systems/HPD/solv_5.5/dftb/N1T48C1/run-2/equil/dftb.inp'
        inp.write_text('1 0 1\nN 0 0 0\n')
        with self.assertRaises(ValueError): run_lcod(self.path)
        self.assertFalse((inp.parents[1].parent/'run-1/meta-lcod').exists())

    def test_bad_parameters(self):
        for atoms in [[0,8,7,8], [1,8,7], [1,1,7,8], [1,8,8,1], [1,8,7,8.5]]:
            self.data['lcod']['atoms']=atoms;self.write()
            with self.assertRaises(ValueError): load_config(self.path)
        self.data['lcod']['atoms']=[1,8,7,8]
        self.data['lcod']['gaussian_width']=float('nan');self.write()
        with self.assertRaises(ValueError): load_config(self.path)

    def test_explicit_system_dir(self):
        self.data.pop('buffer');self.data['system_dir']='systems/custom';self.write()
        self.assertEqual(load_config(self.path).system_dir,Path('systems/custom'))

    def test_grid_output(self):
        self.data['lcod'].update(grid_min=-3.,grid_max=3.,grid_step=.01)
        self.write()
        expected='BONDDISTANCEDIFFERENCE 0.1 1 8 7 8 -3 3 0.01\n'
        self.assertEqual(render_metacv(load_config(self.path), ['N']*8),expected)
        run_lcod(self.path)
        out=self.root/'systems/HPD/solv_5.5/dftb/N1T48C1/run-1/meta-lcod'
        self.assertEqual((out/'metacv.dat').read_text(),expected)
        self.assertEqual((out/'spec.yaml').read_text(),self.path.read_text())

    def test_invalid_grids(self):
        for grid in [dict(grid_min=-3),dict(grid_min=None,grid_max=3,grid_step=.01),
                     dict(grid_min=3,grid_max=-3,grid_step=.01),
                     dict(grid_min=0,grid_max=0,grid_step=.01),
                     dict(grid_min=-3,grid_max=3,grid_step=0),
                     dict(grid_min=-3,grid_max=3,grid_step=-.1),
                     dict(grid_min=-3,grid_max=float('inf'),grid_step=.01),
                     dict(grid_min=-3,grid_max=3,grid_step=7)]:
            with self.subTest(grid=grid):
                self.data['lcod']=dict(gaussian_width=.1,atoms=[1,8,7,8],**grid)
                self.write()
                with self.assertRaises(ValueError): load_config(self.path)


if __name__=='__main__': unittest.main()
