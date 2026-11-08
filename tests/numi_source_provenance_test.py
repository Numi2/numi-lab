import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "python/metalrobo/numi_source_provenance.py"
spec = importlib.util.spec_from_file_location("numi_source_provenance", MODULE)
provenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provenance)


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        (self.repo / "sub").mkdir()
        (self.repo / "CMakeLists.txt").write_text("source\n")
        (self.repo / "sub/tool").write_text("tool\n")
        (self.repo / "sub/tool").chmod(0o755)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
        self.revision = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        self.bundle = self.root / "bundle"
        shutil.copytree(self.repo, self.bundle / "source", ignore=shutil.ignore_patterns(".git"))
        self.manifest = {"schema": "numi.sparse-source-export.v1", "git_revision": self.revision, "files": {}}
        self.update_manifest()
        provenance.make_proof(self.repo, self.bundle)
        self.archive = self.root / "bundle.tar.gz"
        self.pack()
        self.prefix = self.root / "installed"
        for name in provenance.ARTIFACTS:
            path = self.prefix / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("binary:" + name).encode())
        (self.prefix / "share/numi/REVISION").write_text("unknown\n")

    def tearDown(self):
        self.tmp.cleanup()

    def update_manifest(self):
        entries = {}
        for path in (self.bundle / "source").rglob("*"):
            if path.is_file():
                data = path.read_bytes()
                entries[str(path.relative_to(self.bundle))] = {"kind": "file", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "mode": oct(path.stat().st_mode & 0o777)}
        self.manifest["files"] = entries
        self.manifest["file_count"] = len(entries)
        (self.bundle / "BUNDLE-MANIFEST.json").write_text(json.dumps(self.manifest))

    def pack(self):
        with tarfile.open(self.archive, "w:gz") as tf:
            for name in ("source", "BUNDLE-MANIFEST.json", "GIT-PROOF.json"):
                tf.add(self.bundle / name, arcname=name)
        self.digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()

    def test_verified_install_and_audit(self):
        provenance.stamp(self.archive, self.digest, self.revision, self.bundle / "source", self.prefix)
        self.assertEqual((self.prefix / "share/numi/REVISION").read_text().strip(), "verified-source:" + self.revision)
        provenance.audit(self.prefix, self.digest, self.revision)
        (self.prefix / provenance.ARTIFACTS[0]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            provenance.audit(self.prefix, self.digest, self.revision)

    def test_neuron_consumer_audits_real_receipt_and_artifacts(self):
        provenance.stamp(self.archive, self.digest, self.revision, self.bundle / "source", self.prefix)
        helper = self.prefix / "libexec/numi/numi_source_provenance.py"
        helper.parent.mkdir(parents=True)
        shutil.copy2(MODULE, helper)
        helper.chmod(0o755)
        probe = self.prefix / "bin/metalrobo_neuron_culture_probe"
        probe.write_text('#!/bin/sh\nprintf "%s\\n" "${NUMI_NEURON_METALROBO_REVISION-unset}"\n')
        probe.chmod(0o755)
        workspace = self.root / "workspace"
        workspace.mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("NUMI_", "NUMANX_"))}
        env.update(NUMI_LAB_ROOT=str(self.prefix), NUMI_WORKSPACE_ROOT=str(workspace),
                   NUMI_BUILD_DIR=str(self.root / "no-build"), HOME=str(self.root),
                   NUMI_SOURCE_EXPECTED_REVISION=self.revision, NUMI_SOURCE_EXPECTED_ARCHIVE_SHA256=self.digest)
        command = ["sh", str(MODULE.parents[2] / "numi/commands/neurons"), "qualify"]
        def call(): return subprocess.check_output(command, env=env, text=True).strip()
        self.assertEqual(call(), self.revision)
        env["NUMI_SOURCE_EXPECTED_ARCHIVE_SHA256"] = "0" * 64
        self.assertEqual(call(), "unknown")
        env["NUMI_SOURCE_EXPECTED_ARCHIVE_SHA256"] = self.digest
        (self.prefix / provenance.ARTIFACTS[0]).write_bytes(b"changed")
        self.assertEqual(call(), "unknown")

    def test_tampered_build_source_does_not_stamp(self):
        (self.bundle / "source/CMakeLists.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "build source content"):
            provenance.stamp(self.archive, self.digest, self.revision, self.bundle / "source", self.prefix)
        self.assertEqual((self.prefix / "share/numi/REVISION").read_text(), "unknown\n")
        self.assertFalse((self.prefix / "share/numi/SOURCE-PROVENANCE.json").exists())

    def test_manifest_and_blob_forgery_rejected(self):
        (self.bundle / "source/CMakeLists.txt").write_text("forged")
        self.update_manifest()
        self.pack()
        with self.assertRaisesRegex(ValueError, "Git blob or mode mismatch"):
            provenance.verify_archive(self.archive, self.digest, self.revision)

    def test_git_mode_mismatch_rejected(self):
        (self.bundle / "source/sub/tool").chmod(0o644)
        self.update_manifest()
        self.pack()
        with self.assertRaisesRegex(ValueError, "Git blob or mode mismatch"):
            provenance.verify_archive(self.archive, self.digest, self.revision)

    def test_extra_build_source_and_wrong_expected_identity_rejected(self):
        (self.bundle / "source/extra").write_text("extra")
        with self.assertRaisesRegex(ValueError, "file set differs"):
            provenance.verify_archive(self.archive, self.digest, self.revision, self.bundle / "source")
        with self.assertRaisesRegex(ValueError, "archive SHA-256"):
            provenance.verify_archive(self.archive, "0" * 64, self.revision)
        with self.assertRaisesRegex(ValueError, "revision or proof format"):
            provenance.verify_archive(self.archive, self.digest, "0" * 40)

    def test_corrupt_tree_and_missing_inclusion_rejected(self):
        path = self.bundle / "GIT-PROOF.json"
        proof = json.loads(path.read_text())
        key = next(iter(proof["trees"]))
        proof["trees"][key] = "YQ=="
        path.write_text(json.dumps(proof))
        self.pack()
        with self.assertRaisesRegex(ValueError, "tree proof hash mismatch"):
            provenance.verify_archive(self.archive, self.digest, self.revision)


if __name__ == "__main__":
    unittest.main()
