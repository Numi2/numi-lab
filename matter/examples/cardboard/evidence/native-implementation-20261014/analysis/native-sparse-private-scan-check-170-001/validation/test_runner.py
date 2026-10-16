#!/usr/bin/env python3
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native170_runner', PACKAGE / 'runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerContractTests(unittest.TestCase):
    def test_default_dispatch_is_preflight_only(self):
        with mock.patch.object(runner, 'preflight', return_value={'mode':'preflight'}) as pre, \
             mock.patch.object(runner, 'execute') as execute, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main([]), 0)
        pre.assert_called_once_with()
        execute.assert_not_called()

    def test_explicit_execute_is_the_only_native_dispatch_branch(self):
        with mock.patch.object(runner, 'preflight') as pre, \
             mock.patch.object(runner, 'execute', return_value={'exit_code':0}) as execute, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(['--execute']), 0)
        execute.assert_called_once_with()
        pre.assert_not_called()

    def test_exact_stopped_lldb_exception_is_preserved(self):
        row = '77441 76106 77441 T /Users/home/numi-cardboard-20261006/build-cardboard-world/matter/numi-matter-cardboard-paper-material-check'
        active, preserved = runner.classify_process_rows([row], lambda pid: 'lldb' if pid == 76106 else '')
        self.assertEqual(active, [])
        self.assertEqual(len(preserved), 1)

    def test_stopped_exception_does_not_match_changed_identity(self):
        base = '77441 76106 77441 T /Users/home/numi-cardboard-20261006/build-cardboard-world/matter/numi-matter-cardboard-paper-material-check'
        variants = [
            base.replace('77441 76106', '77442 76106'),
            base.replace('76106 77441', '76107 77441'),
            base.replace('77441 T', '77441 S'),
            base.replace('numi-matter-cardboard-paper-material-check', 'numi-matter-cardboard-probe'),
        ]
        for row in variants:
            active, preserved = runner.classify_process_rows([row], lambda _pid: 'lldb')
            self.assertEqual(preserved, [])
            self.assertEqual(len(active), 1)

    def test_unrelated_process_is_ignored_and_active_checker_blocks(self):
        rows = [
            '100 1 100 S /usr/bin/python3',
            '200 1 200 S /tmp/sparse-scan-checker',
        ]
        active, preserved = runner.classify_process_rows(rows, lambda _pid: 'init')
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]['pid'], 200)
        self.assertEqual(preserved, [])

    def test_reserve_writer_materializes_and_hashes_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reserve.bin'
            result = runner.write_reserve(path, byte_count=131072, chunk_size=16384)
            self.assertEqual(result['file_bytes'], 131072)
            self.assertGreaterEqual(result['allocated_bytes'], 131072)
            self.assertTrue(result['fsynced'])
            self.assertEqual(result['sha256'], runner.sha(path))

    def test_case_inventory_and_abi_footprint_are_explicit(self):
        inventory = json.loads((PACKAGE / 'inputs/gpu-case-inventory.json').read_text())
        self.assertEqual(inventory['native_case_count'], 49)
        self.assertEqual(inventory['total_environment_pair_slots'], 6366845)
        self.assertEqual(len(inventory['cases']), 49)
        sizes = dict(line.strip().split('=', 1) for line in
                     (PACKAGE / 'validation/abi-sizes.txt').read_text().splitlines())
        self.assertEqual(sizes['checker_one_gpu_run_buffer_bytes'], '25703664')
        self.assertEqual(sizes['runtime_candidate_sample_buffer_bytes_not_allocated_by_checker'], '10485760')


if __name__ == '__main__':
    unittest.main(verbosity=2)
