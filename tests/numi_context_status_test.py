#!/usr/bin/env python3
"""CPU-only regression tests for bounded, read-only context status discovery."""

import json
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

FAKE_GIT = r'''
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

args = sys.argv[3:]
root = Path(os.environ["NUMI_STATUS_FIXTURE"])
with (root / "calls.jsonl").open("a") as stream:
    stream.write(json.dumps({"args": args, "optional_locks": os.environ.get("GIT_OPTIONAL_LOCKS")}) + "\n")
if args == ["rev-parse", "--is-inside-work-tree"]:
    print("true")
elif args == ["rev-parse", "--short", "HEAD"]:
    print("fixture123")
elif args == ["branch", "--show-current"]:
    print("fixture-branch")
elif args == ["status", "--porcelain=v1", "-z", "--untracked-files=normal"]:
    mode = os.environ["NUMI_STATUS_MODE"]
    (root / "parent.pid").write_text(str(os.getpid()))
    if mode in ("timeout", "child"):
        child = subprocess.Popen([sys.executable, "-c", "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path(os.environ['NUMI_STATUS_FIXTURE'], 'child.pid').write_text(str(os.getpid())); time.sleep(60)"])
        deadline = time.monotonic() + 1
        while not (root / "child.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
    if mode == "timeout":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.write(1, b"?? partial\0")
        time.sleep(60)
    elif mode == "dirty":
        os.write(1, b" M one\0R  new\nname\0old name\0?? folder/\0")
    elif mode == "error":
        os.write(1, b"?? partial\0")
        print("fixture status failed", file=sys.stderr)
        sys.exit(128)
    elif mode == "truncated":
        os.write(1, b" M partial")
    elif mode == "truncated-rename":
        os.write(1, b"R  new\0")
    elif mode == "invalid":
        os.write(1, b"XY invalid\0")
    elif mode not in ("clean", "child"):
        sys.exit(99)
else:
    print("unexpected git invocation: " + repr(args), file=sys.stderr)
    sys.exit(97)
'''


class ContextStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="numi-context-status-")
        self.root = Path(self.temporary.name)
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        git = self.fake_bin / "git"
        git.write_text("#!" + sys.executable + "\n" + FAKE_GIT)
        git.chmod(0o755)
        self.workspace = self.root / "workspace"
        commands = self.workspace / ".numi" / "commands"
        commands.mkdir(parents=True)
        self.capability = commands / "healthy"
        self.capability.write_text("#!/bin/sh\nprintf 'Fixture capability remains available.\\n'\n")
        self.capability.chmod(0o755)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.env = dict(
            os.environ,
            PATH=str(self.fake_bin) + os.pathsep + os.environ.get("PATH", ""),
            NUMI_LAB_ROOT=str(self.runtime),
            NUMI_STATUS_FIXTURE=str(self.root),
            NUMI_COMMAND_PATH="",
            XDG_CONFIG_HOME=str(self.root / "config"),
        )

    def tearDown(self):
        # Clean only fixture-owned PIDs if an assertion or dispatcher fails.
        for name in ("parent.pid", "child.pid"):
            path = self.root / name
            if path.exists():
                try:
                    os.kill(int(path.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.temporary.cleanup()

    def run_context(self, mode):
        self.env["NUMI_STATUS_MODE"] = mode
        start = time.monotonic()
        result = subprocess.run(
            ["/bin/sh", str(DISPATCHER), "context", "--paths"],
            cwd=self.workspace, env=self.env, capture_output=True, text=True,
            timeout=5,
        )
        elapsed = time.monotonic() - start
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Revision:     fixture123\n", result.stdout)
        self.assertIn("Branch:       fixture-branch\n", result.stdout)
        if shutil.which("python3", path=self.env["PATH"]):
            self.assertIn("Fixture capability remains available.", result.stdout)
        else:
            self.assertIn("unknown (Python 3 unavailable for bounded description)", result.stdout)
        self.assertIn("source: " + str(self.capability.resolve()), result.stdout)
        self.assertIn("Active configuration overlays:", result.stdout)
        self.assertIn("Freedom model:", result.stdout)
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        status_calls = [call for call in calls if call["args"][0] == "status"]
        if status_calls:
            self.assertEqual(len(status_calls), 1)
            self.assertEqual(status_calls[0]["optional_locks"], "0")
        return result, elapsed, status_calls

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

    def test_clean(self):
        result, _, calls = self.run_context("clean")
        self.assertTrue(calls)
        self.assertIn("Dirty paths:  0\n", result.stdout)

    def test_dirty_counts_rename_once_and_preserves_newline_path(self):
        result, _, _ = self.run_context("dirty")
        self.assertIn("Dirty paths:  3\n", result.stdout)

    def test_error_does_not_count_partial_output(self):
        result, _, _ = self.run_context("error")
        self.assertIn("Dirty paths:  unknown (git status exited 128)\n", result.stdout)
        self.assertIn("fixture status failed", result.stderr)

    def test_invalid_or_truncated_output_is_unknown(self):
        for mode in ("truncated", "truncated-rename", "invalid"):
            with self.subTest(mode=mode):
                (self.root / "calls.jsonl").unlink(missing_ok=True)
                result, _, _ = self.run_context(mode)
                self.assertIn("Dirty paths:  unknown (git status unavailable:", result.stdout)

    def test_timeout_is_bounded_and_reaps_ignoring_process_group(self):
        result, elapsed, _ = self.run_context("timeout")
        self.assertIn("Dirty paths:  unknown (git status timed out after 2s)\n", result.stdout)
        self.assertLess(elapsed, 4)
        self.assert_fixture_processes_stopped()

    def test_normal_exit_also_stops_remaining_helpers(self):
        result, elapsed, _ = self.run_context("child")
        self.assertIn("Dirty paths:  0\n", result.stdout)
        self.assertLess(elapsed, 2)
        self.assert_fixture_processes_stopped()

    def test_missing_python_remains_unknown_and_preserves_capabilities(self):
        for name in ("dirname", "basename", "awk", "sort", "sed", "tr"):
            executable = shutil.which(name)
            self.assertIsNotNone(executable)
            (self.fake_bin / name).symlink_to(executable)
        self.env["PATH"] = str(self.fake_bin)
        result, _, calls = self.run_context("clean")
        self.assertFalse(calls)
        self.assertIn("Dirty paths:  unknown (Python 3 unavailable for bounded Git status)\n", result.stdout)


if __name__ == "__main__":
    unittest.main()
