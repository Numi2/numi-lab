import contextlib
import fcntl
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).parents[1] / 'validation/run_guarded_source469_v4.py'
spec = importlib.util.spec_from_file_location('guarded_v4', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class GuardedRunnerV4Tests(unittest.TestCase):
    def test_repeated_preflight_and_execute_get_fresh_namespaces(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            dirs = [runner.prepare_invocation('preflight', root) for _ in range(2)]
            dirs += [runner.prepare_invocation('execute', root) for _ in range(2)]
            self.assertEqual(len({p.name for p in dirs}), 4)
            self.assertEqual(len(list(root.iterdir())), 4)
            self.assertEqual([json.loads((p/'invocation.json').read_text())['mode'] for p in dirs],
                             ['preflight','preflight','execute','execute'])
            with self.assertRaises(FileExistsError):
                runner.new_run_namespace(root, mode='execute', token=dirs[0].name)

    def test_cpu_admission_receipts_use_per_invocation_namespace(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            original_pkg, original_command = runner.ARTIFACT_PKG, runner.command
            runner.ARTIFACT_PKG = root
            runner.command = lambda mode,outdir: [sys.executable, '-c',
                "import json; print(json.dumps({'schema':'test.cpu-admission','gpu_execution':False}))"]
            try:
                runs=[]
                for _ in range(2):
                    run=runner.prepare_invocation('preflight',root/'runs')
                    got=runner.run_cpu_admission(run,timeout_seconds=10)
                    self.assertEqual(got['schema'],'test.cpu-admission')
                    runs.append(run)
                self.assertNotEqual(runs[0],runs[1])
                for run in runs:
                    self.assertTrue((run/'cpu-admission-command.json').is_file())
                    self.assertTrue((run/'cpu-admission-receipt.json').is_file())
                    self.assertTrue((run/'cpu-admission.stdout.log').is_file())
                    self.assertTrue((run/'cpu-admission.stderr.log').is_file())
            finally:
                runner.ARTIFACT_PKG, runner.command = original_pkg, original_command

    def test_own_lease_is_not_reprobed_as_competing_lock(self):
        with tempfile.TemporaryDirectory() as td:
            lock=pathlib.Path(td)/'worker.lock'
            with runner.worker_lease(lock):
                # An empty external-worker snapshot is accepted while this process owns the lease.
                runner.reject_workers([])
                with self.assertRaises(runner.GuardError):
                    with runner.worker_lease(lock):
                        pass
            # The non-owning probe is free once the lease has been released.
            self.assertFalse(runner.lock_busy(lock))
            source=SCRIPT.read_text()
            body=source.split('def execute_native(',1)[1].split('\ndef main(',1)[0]
            self.assertNotIn('lock_busy(',body)

    def test_competing_lock_blocks_before_child_launch(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            lock=root/'worker.lock'
            lock.parent.mkdir(parents=True,exist_ok=True)
            with lock.open('a+') as held:
                fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(runner.GuardError):
                    with runner.worker_lease(lock):
                        self.fail('competing lock unexpectedly admitted')
            self.assertFalse(runner.lock_busy(lock))

    def test_large_child_output_is_file_backed_without_pipe_deadlock(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            out,err=root/'stdout.log',root/'stderr.log'
            code="import sys; sys.stdout.write('o'*2500000); sys.stderr.write('e'*2500000)"
            rc,wall,reason=runner.run_file_backed([sys.executable,'-c',code],root,out,err,20)
            self.assertEqual(rc,0)
            self.assertIsNone(reason)
            self.assertEqual(out.stat().st_size,2500000)
            self.assertEqual(err.stat().st_size,2500000)
            body=SCRIPT.read_text().split('def run_file_backed(',1)[1].split('\ndef run_cpu_admission(',1)[0]
            self.assertNotIn('PIPE',body)
            self.assertNotIn('capture_output',body)

    def test_repeated_native_execute_uses_fresh_receipt_and_output_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            old=(runner.PKG,runner.ARTIFACT_PKG,runner.LOCK,runner.command,runner.process_snapshot,runner.DISK_FLOOR)
            runner.PKG=root
            runner.ARTIFACT_PKG=root
            runner.LOCK=root/'source'/'worker.lock'
            runner.process_snapshot=lambda: []
            runner.DISK_FLOOR=0
            def fake_command(mode,outdir):
                return [sys.executable,'-c',
                    "import pathlib,sys; pathlib.Path(sys.argv[1]).mkdir(parents=True); print('fake-native')",
                    str(outdir)]
            runner.command=fake_command
            try:
                runs=[]
                for _ in range(2):
                    run=runner.prepare_invocation('execute',root/'runs')
                    receipt=runner.execute_native(run,10,[],10**12)
                    self.assertEqual(receipt['exit_code'],0)
                    self.assertTrue(receipt['output_exists'])
                    self.assertTrue((run/'native-execution-receipt.json').is_file())
                    self.assertTrue((run/'native.stdout.log').is_file())
                    runs.append(run)
                self.assertNotEqual(runs[0],runs[1])
                self.assertNotEqual(runs[0]/'native-output',runs[1]/'native-output')
            finally:
                runner.PKG,runner.ARTIFACT_PKG,runner.LOCK,runner.command,runner.process_snapshot,runner.DISK_FLOOR=old

if __name__=='__main__':
    unittest.main(verbosity=2)
