#!/opt/homebrew/Cellar/python@3.13/3.13.14_1/bin/python3.13
import importlib.util, tempfile, unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]/'runner.py'
spec=importlib.util.spec_from_file_location('checker_runner',R)
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
class GuardTests(unittest.TestCase):
    def test_worker_names(self):
        self.assertTrue(runner.worker_relevant('sparse-scan-checker'))
        self.assertTrue(runner.worker_relevant('sparse-scan-che'))
        self.assertTrue(runner.worker_relevant('numi-matter-cardboard-probe'))
        self.assertFalse(runner.worker_relevant('clang++'))
    def test_guard_constants_are_exact(self):
        self.assertEqual(runner.MIN_FREE,2147483648)
        self.assertEqual(runner.RESERVE,134217728)
        self.assertEqual(runner.ABORT_FLOOR,536870912)
        self.assertEqual(runner.TIMEOUT,900)
    def test_reserve_allocator_checks_actual_write(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'reserve.bin'
            row=runner.write_reserve(p,4096)
            self.assertEqual(row['bytes'],4096)
            self.assertGreaterEqual(row['allocated_bytes'],4096)
            self.assertTrue(row['fsynced'])
    def test_lock_is_not_silently_ignored(self):
        self.assertEqual(runner.LOCK.name,'.single-worker.lock')
if __name__=='__main__': unittest.main(verbosity=2)
