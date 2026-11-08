#!/usr/bin/env python3
"""CPU shell contract only; no neural simulation or Metal device is run."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'numi/commands/neurons'


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.runtime = self.base / 'runtime'
        self.workspace = self.base / 'workspace'
        self.workspace.mkdir()
        (self.runtime / 'bin').mkdir(parents=True)
        self.probe = self.runtime / 'bin/metalrobo_neuron_culture_probe'
        self.probe.write_text('''#!/bin/sh
printf 'revision=%s\\nshader=%s\\n' "${NUMI_NEURON_METALROBO_REVISION-unset}" "${NUMI_NEURON_METALLIB_SHA256-unset}"
''')
        self.probe.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('NUMI_', 'NUMANX_'))}
        self.env.update(NUMI_LAB_ROOT=str(self.runtime), NUMI_WORKSPACE_ROOT=str(self.workspace),
                        NUMI_BUILD_DIR=str(self.base / 'build'), HOME=str(self.base))

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, env=None):
        result = subprocess.run(['sh', str(SCRIPT), 'qualify'], env=env or self.env,
                                text=True, capture_output=True, check=True)
        return dict(line.split('=', 1) for line in result.stdout.splitlines())

    def shader(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    def test_installed_metadata_and_shader(self):
        meta = self.runtime / 'share/numi/REVISION'
        meta.parent.mkdir(parents=True)
        meta.write_text('revision-from-package\n')
        expected = self.shader(self.runtime / 'lib/metalrobo/NumiNeuron.metallib', b'installed')
        self.assertEqual(self.call(), {'revision': 'revision-from-package', 'shader': expected})

    def test_verified_bundle_requires_independent_identity_and_audit(self):
        meta = self.runtime / 'share/numi/REVISION'
        meta.parent.mkdir(parents=True)
        meta.write_text('verified-source:abc123\n')
        helper = self.runtime / 'libexec/numi/numi_source_provenance.py'
        helper.parent.mkdir(parents=True)
        helper.write_text("#!/bin/sh\n[ \"$1\" = audit ] && [ \"$5\" = abc123 ] && [ \"$7\" = expectedhash ]\n")
        helper.chmod(0o755)
        self.assertEqual(self.call()['revision'], 'unknown')
        expected = dict(self.env, NUMI_SOURCE_EXPECTED_REVISION='abc123', NUMI_SOURCE_EXPECTED_ARCHIVE_SHA256='expectedhash')
        self.assertEqual(self.call(expected)['revision'], 'abc123')
        expected['NUMI_SOURCE_EXPECTED_ARCHIVE_SHA256'] = 'wrong'
        self.assertEqual(self.call(expected)['revision'], 'unknown')
        helper.unlink()
        self.assertEqual(self.call(expected)['revision'], 'unknown')
        expected['NUMI_NEURON_METALROBO_REVISION'] = 'explicit'
        self.assertEqual(self.call(expected)['revision'], 'explicit')

    def test_installed_revision_wins_inside_git_tree(self):
        subprocess.run(['git', 'init', '-q', str(self.base)], check=True)
        subprocess.run(['git', '-C', str(self.base), '-c', 'user.name=Test',
                        '-c', 'user.email=test@example.invalid', 'commit', '--allow-empty',
                        '-qm', 'unrelated ancestor'], check=True)
        meta = self.runtime / 'share/numi/REVISION'
        meta.parent.mkdir(parents=True)
        meta.write_text('installed-revision\n')
        result = self.call()
        self.assertEqual(result['revision'], 'installed-revision')

    def test_source_tree_fallback(self):
        subprocess.run(['git', 'init', '-q', str(self.runtime)], check=True)
        subprocess.run(['git', '-C', str(self.runtime), '-c', 'user.name=Test',
                        '-c', 'user.email=test@example.invalid', 'commit', '--allow-empty',
                        '-qm', 'fixture'], check=True)
        revision = subprocess.check_output(['git', '-C', str(self.runtime), 'rev-parse', 'HEAD'], text=True).strip()
        build_probe = self.base / 'build/bin/metalrobo_neuron_culture_probe'
        build_probe.parent.mkdir(parents=True)
        build_probe.write_bytes(self.probe.read_bytes())
        build_probe.chmod(0o755)
        expected = self.shader(self.base / 'build/shaders/NumiNeuron.metallib', b'build')
        self.assertEqual(self.call(), {'revision': revision, 'shader': expected})

    def test_explicit_overrides_and_missing_shader(self):
        override = self.base / 'custom.metallib'
        shader_hash = self.shader(override, b'override')
        env = dict(self.env, NUMI_NEURON_METALROBO_REVISION='explicit', NUMI_NEURON_METALLIB=str(override))
        self.assertEqual(self.call(env), {'revision': 'explicit', 'shader': shader_hash})
        env['NUMI_NEURON_METALLIB_SHA256'] = 'supplied-hash'
        self.assertEqual(self.call(env), {'revision': 'explicit', 'shader': 'supplied-hash'})
        self.assertEqual(self.call(), {'revision': '', 'shader': 'unset'})

    def test_external_build_does_not_claim_installed_revision(self):
        meta = self.runtime / 'share/numi/REVISION'
        meta.parent.mkdir(parents=True)
        meta.write_text('installed-only\n')
        build_probe = self.base / 'build/bin/metalrobo_neuron_culture_probe'
        build_probe.parent.mkdir(parents=True)
        build_probe.write_bytes(self.probe.read_bytes())
        build_probe.chmod(0o755)
        self.assertEqual(self.call(), {'revision': '', 'shader': 'unset'})


if __name__ == '__main__':
    unittest.main()
