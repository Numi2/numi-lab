#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('native470_guard', PACKAGE / 'runner.py')
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class FakeProcess:
    def __init__(self):
        self.pid = 100
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = -15
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class GuardContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='native470-ctest-contract-')
        self.root = Path(self.temp.name)
        self.report = {
            'schema': 'numi.cardboard.ctest-registration-validation.v1',
            'source_build_sha256': 'build-hash',
            'source167_validation_report_sha256': 'report-hash',
            'tests': list(guard.EXPECTED_CTESTS),
            'passed': 4,
        }
        self.packaged_report = self.root / 'packaged-report.json'
        raw_report = json.dumps(self.report, indent=2, sort_keys=True) + '\n'
        self.packaged_report.write_text(raw_report)
        (self.root / 'report.json').write_text(raw_report)
        for name, value in {
            'CTestTestfile.cmake': 'add_test(coupon_resume)\n',
            'LastTest.log': '100% tests passed\n',
            'seal.py': '# test fixture\n',
        }.items():
            (self.root / name).write_text(value)
        self.manifest = {}
        for name in ('CTestTestfile.cmake', 'LastTest.log', 'report.json', 'seal.py'):
            data = (self.root / name).read_bytes()
            self.manifest[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, report=None, manifest=None):
        guard.validate_ctest_registration(report or self.report, manifest or self.manifest,
                                          'build-hash', 'report-hash', self.root,
                                          self.packaged_report)

    def test_ctest_exact_report_and_manifest_pass(self):
        self.validate()

    def test_ctest_missing_pass_count_is_rejected(self):
        report = dict(self.report)
        report['passed'] = None
        with self.assertRaises(ValueError):
            self.validate(report=report)

    def test_ctest_incomplete_inventory_is_rejected(self):
        report = dict(self.report)
        report['tests'] = report['tests'][:-1]
        with self.assertRaises(ValueError):
            self.validate(report=report)

    def test_ctest_manifest_file_hash_is_checked(self):
        manifest = dict(self.manifest)
        manifest['LastTest.log'] = dict(manifest['LastTest.log'], sha256='0' * 64)
        with self.assertRaises(ValueError):
            self.validate(manifest=manifest)

    def test_ctest_manifest_must_bind_packaged_report(self):
        manifest = dict(self.manifest)
        manifest['report.json'] = dict(manifest['report.json'], sha256='0' * 64)
        with self.assertRaises(ValueError):
            self.validate(manifest=manifest)

    def test_exception_cleanup_signals_only_verified_native_group_and_reaps_wrapper(self):
        process = FakeProcess()
        with patch.object(guard, '_verified_direct_native_groups', return_value=[222]), \
             patch.object(guard, '_verified_native_still_in_group', return_value=False), \
             patch.object(guard.os, 'killpg') as killpg:
            result = guard.terminate_owned_native(process, '/exact/native/probe')
        killpg.assert_called_once_with(222, guard.signal.SIGTERM)
        self.assertTrue(result['wrapper_reaped'])
        self.assertEqual(result['verified_native_groups'], [222])
        self.assertEqual(result['native_groups_remaining'], [])

    def test_exception_cleanup_without_verified_native_group_does_not_signal_group(self):
        process = FakeProcess()
        with patch.object(guard, '_verified_direct_native_groups', return_value=[]), \
             patch.object(guard, '_verified_native_still_in_group', return_value=False), \
             patch.object(guard.os, 'killpg') as killpg:
            result = guard.terminate_owned_native(process, '/exact/native/probe')
        killpg.assert_not_called()
        self.assertTrue(result['wrapper_reaped'])
        self.assertEqual(result['verified_native_groups'], [])

    def test_exception_cleanup_reaps_real_wrapper_and_exact_native_child(self):
        with tempfile.TemporaryDirectory(prefix='native470-owned-child-test-') as temporary:
            directory = Path(temporary)
            pid_path = directory / 'child.pid'
            wrapper_path = directory / 'wrapper.py'
            wrapper_path.write_text(
                'import subprocess,sys\n'
                'from pathlib import Path\n'
                f'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],start_new_session=True)\n'
                f'Path({str(pid_path)!r}).write_text(str(child.pid))\n'
                'child.wait()\n'
            )
            wrapper = subprocess.Popen([sys.executable, str(wrapper_path)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       start_new_session=True)
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not pid_path.exists():
                    if wrapper.poll() is not None:
                        self.fail('test wrapper exited before starting its native-child stand-in')
                    time.sleep(0.02)
                self.assertTrue(pid_path.exists(), 'child pid marker was not written')
                child_pid = int(pid_path.read_text())
                result = guard.terminate_owned_native(wrapper, sys.executable, term_grace=3)
                self.assertTrue(result['wrapper_reaped'])
                self.assertIn({'pgid': child_pid, 'signal': 'SIGTERM', 'identity_verified': True}, result['signals'])
                self.assertEqual(result['native_child_status'], 'verified_groups_signalled_and_absent')
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        os.kill(child_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.02)
                else:
                    self.fail('native-child stand-in remained after cleanup')
            finally:
                if wrapper.poll() is None:
                    wrapper.terminate()
                    try:
                        wrapper.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        wrapper.kill()
                        wrapper.wait(timeout=2)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_reserve_allocator_materializes_exact_128_mib(self):
        with tempfile.TemporaryDirectory(prefix='native470-reserve-test-') as temporary:
            reserve_path = Path(temporary) / 'reserve.bin'
            details = guard.write_reserve_file(reserve_path)
            self.assertEqual(details['requested_bytes'], 128 * 1024 * 1024)
            self.assertEqual(details['file_size_bytes'], 128 * 1024 * 1024)
            self.assertGreaterEqual(details['allocated_bytes'], 128 * 1024 * 1024)
            self.assertTrue(details['fsynced'])
            self.assertEqual(len(details['sha256']), 64)
            self.assertEqual(reserve_path.stat().st_size, 128 * 1024 * 1024)


if __name__ == '__main__':
    unittest.main(verbosity=2)
