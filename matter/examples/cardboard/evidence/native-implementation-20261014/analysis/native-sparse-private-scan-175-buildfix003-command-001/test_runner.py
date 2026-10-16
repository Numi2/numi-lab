import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).with_name('runner.py')
SPEC = importlib.util.spec_from_file_location('scan175_runner', MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class OutputContractTests(unittest.TestCase):
    def test_success_marker(self):
        result = runner.parse_checker_output(
            'hierarchical_sparse_scan_native_checks=129981 kernels_executed=1 physical_transactions=0\n')
        self.assertEqual(result, {'passed': True, 'checks': 129981, 'reason': None})

    def test_missing_marker_fails_closed(self):
        result = runner.parse_checker_output('all tests passed\n')
        self.assertFalse(result['passed'])
        self.assertIsNone(result['checks'])

    def test_zero_assertions_fails_closed(self):
        result = runner.parse_checker_output(
            'hierarchical_sparse_scan_native_checks=0 kernels_executed=1 physical_transactions=0\n')
        self.assertFalse(result['passed'])

    def test_wrong_transaction_claim_fails_closed(self):
        result = runner.parse_checker_output(
            'hierarchical_sparse_scan_native_checks=100 kernels_executed=1 physical_transactions=1\n')
        self.assertFalse(result['passed'])

    def test_active_matter_process_blocks(self):
        original = runner._parse_process_rows
        runner._parse_process_rows = lambda: [
            {'pid': 41, 'ppid': 1, 'pgid': 41, 'state': 'S',
             'command': '/build/numi-matter-cardboard-coupon-resume-check'}]
        try:
            self.assertEqual(len(runner.active_workers()), 1)
        finally:
            runner._parse_process_rows = original

    def test_zombie_and_non_matter_processes_do_not_block(self):
        original = runner._parse_process_rows
        runner._parse_process_rows = lambda: [
            {'pid': 41, 'ppid': 1, 'pgid': 41, 'state': 'Z',
             'command': '/build/numi-matter-cardboard-probe'},
            {'pid': 42, 'ppid': 1, 'pgid': 42, 'state': 'S', 'command': '/usr/bin/python3'}]
        try:
            self.assertEqual(runner.active_workers(), [])
        finally:
            runner._parse_process_rows = original


if __name__ == '__main__':
    unittest.main()
