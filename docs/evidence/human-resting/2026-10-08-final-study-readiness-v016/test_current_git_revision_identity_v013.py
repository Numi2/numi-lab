#!/usr/bin/env python3
import hashlib
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("readiness_v012", ROOT / "prepare_final_plan.py")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def init_two_commit_repo(repo):
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "base.txt").write_text("base source\\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "base.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "base"], check=True,
                   capture_output=True)
    source = repo / "source.txt"
    source.write_text("committed source\\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "source"], check=True,
                   capture_output=True)
    return source


class CurrentGitRevisionIdentityTests(unittest.TestCase):
    def test_clean_commit_uses_empty_current_diff_and_keeps_commit_delta_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            init_two_commit_repo(repo)
            identity = PREPARE.current_git_identity(repo)
            self.assertEqual(identity["revision"], subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip())
            self.assertEqual(identity["status"], "")
            self.assertEqual(identity["diff_sha256"], hashlib.sha256(b"").hexdigest())
            commit_diff = subprocess.check_output(
                ["git", "-C", str(repo), "diff", "HEAD^", "HEAD", "--binary"])
            self.assertNotEqual(identity["diff_sha256"], hashlib.sha256(commit_diff).hexdigest())

    def test_dirty_build_delta_is_separate_from_canonical_diff_field(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            source = init_two_commit_repo(repo)
            source.write_text("build-time source delta\\n", encoding="utf-8")
            identity = PREPARE.current_git_identity(repo)
            current_diff = subprocess.check_output(
                ["git", "-C", str(repo), "diff", "HEAD", "--binary"])
            self.assertTrue(identity["status"])
            self.assertEqual(identity["diff_sha256"], hashlib.sha256(current_diff).hexdigest())
            self.assertNotEqual(identity["diff_sha256"], hashlib.sha256(b"").hexdigest())


if __name__ == "__main__":
    unittest.main()
