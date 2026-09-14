"""Explicit geometry sources and append regression tests; no simulation jobs."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qmmd import dftb

ROOT = Path(__file__).resolve().parents[1]


class XYZSourceTests(unittest.TestCase):
    def setUp(self):
        self.config_path = ROOT / 'configs/PRN-anti/dftb/dftb.yaml'
        self.cfg = dftb.load_config(self.config_path)

    def test_source_and_legacy_paths(self):
        self.assertEqual(dftb.input_xyz_path(self.cfg, ROOT), ROOT / self.cfg.source_xyz)
        legacy = replace(self.cfg, source_xyz=None, source_sha256=None)
        self.assertEqual(dftb.input_xyz_path(legacy, ROOT), dftb.salt_dir(legacy, ROOT)/'ready.xyz')
        absolute = replace(self.cfg, source_xyz='/tmp/example.xyz')
        self.assertEqual(dftb.input_xyz_path(absolute, ROOT), Path('/tmp/example.xyz'))

    def test_box_formats(self):
        expected = [(1., 2., 3.), (4., 5., 6.), (7., 8., 10.)]
        for comment in ('Lattice="1 2 3 4 5 6 7 8 10"',
                        'Conf 1. Box X: 1 2 3 Y: 4 5 6 Z: 7 8 10'):
            self.assertEqual(dftb.parse_box_vectors_from_comment(comment), expected)
        with self.assertRaises(RuntimeError):
            dftb.parse_box_vectors_from_comment('Lattice="1 2 3"')

    def test_hash_rejection(self):
        with self.assertRaisesRegex(RuntimeError, 'source_sha256 mismatch'):
            dftb.validate_xyz_source(replace(self.cfg, source_sha256='0'*64), ROOT)

    def test_real_source_append_in_temporary_directory(self):
        cfg = replace(self.cfg, replicas=30, append=True)
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            bench = Path(temp)
            for run in range(1, 21):
                folder = bench/f'run-{run}'/'equil'
                folder.mkdir(parents=True)
                (folder/'sentinel').write_text('untouched')
            with patch.object(dftb, 'load_config', return_value=cfg), \
                 patch.object(dftb, 'out_dir', return_value=bench), \
                 patch.object(dftb, 'submit_slurm') as submit, patch('builtins.print'):
                dftb.run_dftb_prep(self.config_path)
                submit.assert_not_called()
            self.assertEqual(dftb.find_existing_run_indices(bench), list(range(1, 51)))
            for run in range(1, 21):
                self.assertEqual((bench/f'run-{run}/equil/sentinel').read_text(), 'untouched')
            for run in range(21, 51):
                folder = bench/f'run-{run}/equil'
                text = (folder/'dftb.inp').read_text()
                self.assertIn('14.40887381', text)
                self.assertTrue((folder/'slurm.sh').is_file())
                self.assertEqual((folder/'spec.yaml').read_text(), self.config_path.read_text())


if __name__ == '__main__':
    unittest.main()
