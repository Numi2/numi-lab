import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
PLUGIN = REPOSITORY / "plugins" / "numi-lab"
SKILL = PLUGIN / "skills" / "numi-lab" / "SKILL.md"
EVALS = PLUGIN / "skills" / "numi-lab" / "evals" / "evals.json"
CHECK_INSTALL = PLUGIN / "scripts" / "check_install.py"


class SkillContractTests(unittest.TestCase):
    def test_skill_and_manifest_contract(self):
        skill = SKILL.read_text()
        manifest = json.loads((PLUGIN / ".codex-plugin" / "plugin.json").read_text())
        self.assertEqual(manifest["name"], "numi-lab")
        self.assertEqual(manifest["skills"], "./skills/")
        self.assertEqual(len(manifest["interface"]["defaultPrompt"]), 3)
        self.assertIn("what is ready", manifest["interface"]["defaultPrompt"][0])
        self.assertIn("name: numi-lab", skill)
        self.assertIn("numi doctor", skill)
        self.assertIn("numi context", skill)
        self.assertIn("Do not activate it", skill)
        self.assertIn("Ask only when live discovery cannot resolve", skill)
        self.assertIn("For discovery commands", skill)
        self.assertIn("Before real hardware can move", skill)
        self.assertIn("give the next safe recovery command", skill)
        self.assertIn("numi solvers list", skill)
        self.assertIn("docs/NUMI_SOLVERS.md", skill)
        self.assertLessEqual(len(skill.splitlines()), 155)

    def test_representative_eval_classes(self):
        corpus = json.loads(EVALS.read_text())
        self.assertEqual(corpus["skill_name"], "numi-lab")
        cases = corpus["evals"]
        categories = {case["category"] for case in cases}
        self.assertTrue(
            {
                "direct-trigger",
                "indirect-trigger",
                "incomplete-input",
                "non-trigger",
                "edge-hardware",
                "edge-evidence",
                "edge-failure-recovery",
                "solver-configuration",
            }.issubset(categories)
        )
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertTrue(any(not case["should_activate"] for case in cases))
        self.assertTrue(any(case["should_activate"] for case in cases))
        for case in cases:
            self.assertTrue(case["prompt"].strip())
            self.assertTrue(case["expected_output"].strip())
            self.assertTrue(case["expectations"])


class InstallStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.source = root / "source"
        shutil.copytree(PLUGIN, self.source)
        self.version = json.loads(
            (self.source / ".codex-plugin" / "plugin.json").read_text()
        )["version"]
        self.cache_root = root / "cache"
        self.cache = self.cache_root / "numi-lab" / "numi-lab" / self.version
        shutil.copytree(self.source, self.cache)

    def tearDown(self):
        self.temporary.cleanup()

    def listing(self, *, version=None, enabled=True, source=None, installed=True):
        return {
            "installed": [
                {
                    "pluginId": "numi-lab@numi-lab",
                    "version": version or self.version,
                    "installed": installed,
                    "enabled": enabled,
                    "source": {"path": str(source or self.source)},
                }
            ]
        }

    def run_check(self, listing):
        return subprocess.run(
            [
                sys.executable,
                str(CHECK_INSTALL),
                "--source",
                str(self.source),
                "--cache-root",
                str(self.cache_root),
            ],
            input=json.dumps(listing),
            text=True,
            capture_output=True,
            check=False,
        )

    def test_current_exact_cache_passes(self):
        result = self.run_check(self.listing())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("installed, enabled, and current", result.stdout)

    def test_stale_version_fails(self):
        result = self.run_check(self.listing(version="0.1.0+codex.old"))
        self.assertEqual(result.returncode, 4)
        self.assertIn("is stale", result.stderr)

    def test_disabled_plugin_fails(self):
        result = self.run_check(self.listing(enabled=False))
        self.assertEqual(result.returncode, 3)
        self.assertIn("disabled", result.stderr)

    def test_not_installed_fails(self):
        result = self.run_check({"installed": []})
        self.assertEqual(result.returncode, 2)
        self.assertIn("not installed", result.stderr)

    def test_other_source_fails(self):
        result = self.run_check(self.listing(source=self.source.parent / "other"))
        self.assertEqual(result.returncode, 5)
        self.assertIn("source mismatch", result.stderr)

    def test_cache_content_drift_fails(self):
        with (self.cache / "skills" / "numi-lab" / "SKILL.md").open("a") as stream:
            stream.write("\ndrift\n")
        result = self.run_check(self.listing())
        self.assertEqual(result.returncode, 7)
        self.assertIn("differs", result.stderr)

    def test_missing_cache_fails(self):
        self.cache.rename(self.cache.parent / "removed-cache")
        result = self.run_check(self.listing())
        self.assertEqual(result.returncode, 6)
        self.assertIn("cache is missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
