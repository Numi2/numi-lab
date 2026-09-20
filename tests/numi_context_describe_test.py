#!/usr/bin/env python3
"""CPU-only regression tests for bounded capability-description discovery."""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


DISPATCHER = Path(__file__).resolve().parents[1] / "tools" / "numi"
if len(sys.argv) > 2 and sys.argv[1] == "--dispatcher":
    DISPATCHER = Path(sys.argv[2]).resolve()
    del sys.argv[1:3]

CAPABILITY = r'''
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(os.environ["NUMI_DESCRIBE_FIXTURE"])
if sys.argv[1:] != ["--numi-describe"]:
    time.sleep(2.2)
    print("normal:" + repr(sys.argv[1:]))
    sys.exit(17)
mode = os.environ["NUMI_DESCRIBE_MODE"]
(root / "parent.pid").write_text(str(os.getpid()))
if mode in ("timeout", "child"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    subprocess.Popen([sys.executable, "-c", "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path(os.environ['NUMI_DESCRIBE_FIXTURE'], 'child.pid').write_text(str(os.getpid())); time.sleep(60)"])
    deadline = time.monotonic() + 1
    while not (root / "child.pid").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
if mode == "timeout":
    print("Partial output is not a valid description.", flush=True)
    time.sleep(60)
elif mode == "failure":
    print("Partial output is not a valid description.")
    print("description fixture failed", file=sys.stderr)
    sys.exit(23)
elif mode == "empty":
    pass
elif mode == "invalid":
    os.write(1, b"\xff\n")
elif mode == "oversized":
    print("x" * 65537)
else:
    print("  Exact description: μ, $value, 'quotes'.  ")
    print("Second line must not appear.")
'''


class ContextDescribeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="numi-context-describe-")
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace with spaces"
        commands = self.workspace / ".numi" / "commands"
        commands.mkdir(parents=True)
        self.capability = commands / "fixture"
        self.capability.write_text("#!" + sys.executable + "\n" + CAPABILITY)
        self.capability.chmod(0o755)
        self.healthy = commands / "healthy"
        self.healthy.write_text("#!/bin/sh\nprintf 'Following capability stays available.\\n'\n")
        self.healthy.chmod(0o755)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.env = dict(
            os.environ,
            NUMI_LAB_ROOT=str(self.runtime),
            NUMI_DESCRIBE_FIXTURE=str(self.root),
            NUMI_COMMAND_PATH="",
            XDG_CONFIG_HOME=str(self.root / "config"),
        )

    def tearDown(self):
        # Kill only fixture-owned processes if a dispatcher regression fails.
        for name in ("parent.pid", "child.pid"):
            path = self.root / name
            if path.exists():
                try:
                    os.kill(int(path.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.temporary.cleanup()

    def run_context(self, mode):
        self.env["NUMI_DESCRIBE_MODE"] = mode
        started = time.monotonic()
        result = subprocess.run(
            ["/bin/sh", str(DISPATCHER), "context", "--paths"],
            cwd=self.workspace, env=self.env, capture_output=True, text=True,
            timeout=5,
        )
        elapsed = time.monotonic() - started
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("source: " + str(self.capability.resolve()), result.stdout)
        self.assertIn("source: " + str(self.healthy.resolve()), result.stdout)
        self.assertIn("Active configuration overlays:", result.stdout)
        self.assertIn("Freedom model:", result.stdout)
        self.assertNotIn("Partial output is not a valid description.", result.stdout)
        self.assertNotIn("Second line must not appear.", result.stdout)
        if shutil.which("python3", path=self.env.get("PATH")):
            self.assertIn("Following capability stays available.", result.stdout)
        return result, elapsed

    def assert_fixture_processes_stopped(self):
        for name in ("parent.pid", "child.pid"):
            path = self.root / name
            self.assertTrue(path.exists(), name)
            pid = int(path.read_text())
            deadline = time.monotonic() + 0.5
            while True:
                result = subprocess.run(
                    ["/bin/ps", "-o", "stat=", "-p", str(pid)],
                    capture_output=True, text=True, timeout=1,
                )
                status = result.stdout.strip()
                if not status or status.startswith("Z"):
                    break
                if time.monotonic() >= deadline:
                    self.fail("fixture process %d remains alive (%s)" % (pid, status))
                time.sleep(0.02)

    def test_healthy_description_preserves_exact_first_line(self):
        result, _ = self.run_context("healthy")
        self.assertIn("  fixture    Exact description: μ, $value, 'quotes'.  \n", result.stdout)

    def test_failed_description_does_not_publish_partial_output(self):
        result, _ = self.run_context("failure")
        self.assertIn("unknown (description exited 23)", result.stdout)

    def test_empty_invalid_and_oversized_description_are_unknown(self):
        for mode in ("empty", "invalid", "oversized"):
            with self.subTest(mode=mode):
                result, _ = self.run_context(mode)
                self.assertIn("unknown (description unavailable:", result.stdout)

    def test_timeout_stops_parent_and_descendants_ignoring_term(self):
        result, elapsed = self.run_context("timeout")
        self.assertIn("unknown (description timed out after 2s)", result.stdout)
        self.assertLess(elapsed, 4)
        self.assert_fixture_processes_stopped()

    def test_normal_exit_stops_descendants_holding_stdout_open(self):
        result, elapsed = self.run_context("child")
        self.assertIn("Exact description:", result.stdout)
        self.assertLess(elapsed, 2)
        self.assert_fixture_processes_stopped()

    def test_unavailable_executable_is_unknown_and_continues(self):
        self.capability.write_text("#!/missing/numi-fixture-interpreter\n")
        result, _ = self.run_context("healthy")
        self.assertIn("unknown (description exited 1)", result.stdout)

    def test_missing_python_preserves_capabilities_and_paths(self):
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        for name in ("dirname", "basename", "awk", "sort", "sed", "tr"):
            executable = shutil.which(name)
            self.assertIsNotNone(executable)
            (fake_bin / name).symlink_to(executable)
        self.env["PATH"] = str(fake_bin)
        result, _ = self.run_context("healthy")
        self.assertEqual(result.stdout.count("unknown (Python 3 unavailable for bounded description)"), 2)
        self.assertFalse((self.root / "parent.pid").exists())

    def test_normal_dispatch_keeps_arguments_exit_status_and_longer_runtime(self):
        self.env["NUMI_DESCRIBE_MODE"] = "timeout"
        result = subprocess.run(
            ["/bin/sh", str(DISPATCHER), "fixture", "two words", "--literal=$value"],
            cwd=self.workspace, env=self.env, capture_output=True, text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(result.stdout, "normal:['two words', '--literal=$value']\n")


if __name__ == "__main__":
    unittest.main()
