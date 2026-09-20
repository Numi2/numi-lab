import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
TOOL = REPOSITORY / "python" / "metalrobo" / "solver_catalog.py"
CATALOG = REPOSITORY / "numi" / "solvers" / "catalog.json"


class SolverCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        self.config_home = root / "config"

    def tearDown(self):
        self.temporary.cleanup()

    def run_tool(self, *arguments, check=True, extra_environment=None):
        environment = os.environ.copy()
        environment["XDG_CONFIG_HOME"] = str(self.config_home)
        if extra_environment:
            environment.update(extra_environment)
        result = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--runtime-root",
                str(REPOSITORY),
                "--workspace",
                str(self.workspace),
                "--catalog",
                str(CATALOG),
                *arguments,
            ],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        if check and result.returncode != 0:
            self.fail(result.stderr)
        return result

    def test_bundled_catalog_covers_domains_and_distinct_quality_configs(self):
        family_result = self.run_tool("list", "--json")
        families = json.loads(family_result.stdout)["families"]
        self.assertEqual(len(families), 9)
        self.assertEqual(
            {family["id"] for family in families},
            {
                "articulated-dynamics",
                "throughput-pgs",
                "temporal-cone",
                "quality-newton",
                "reference-projected-gradient",
                "matter-newton-fgmres",
                "der-xpbd",
                "symplectic-euler",
                "implicit-midpoint",
            },
        )
        result = self.run_tool("list", "--implementations", "--json")
        solvers = json.loads(result.stdout)["solvers"]
        identifiers = {solver["id"] for solver in solvers}
        domains = {solver["domain"] for solver in solvers}
        self.assertGreaterEqual(len(solvers), 19)
        self.assertTrue(
            {"contact", "dynamics", "constraints", "integration", "continuum", "rod"}
            .issubset(domains)
        )
        self.assertIn("contact.quality-semismooth-newton-metal", identifiers)
        self.assertIn("contact.quality-newton-metal-world", identifiers)
        self.assertIn("constraints.unified-quality-newton-metal", identifiers)
        for solver in solvers:
            for implementation in solver["owner"]["implementation"].split(" and "):
                self.assertTrue(
                    (REPOSITORY / implementation).is_file(),
                    f"{solver['id']} has stale implementation path {implementation}",
                )

        task_solvers = json.loads(
            self.run_tool("list", "--target", "CompiledRun task rollout", "--json").stdout
        )["families"]
        self.assertEqual(
            {solver["id"] for solver in task_solvers},
            {"temporal-cone", "quality-newton"},
        )

    def test_configure_show_and_validate_fingerprinted_profile(self):
        self.run_tool(
            "configure",
            "temporal-cone",
            "--profile",
            "production-contact",
            "--set",
            "velocity_iterations=7",
            "--set",
            "warm_start=false",
        )
        profile_path = (
            self.workspace
            / ".numi"
            / "profiles"
            / "solvers"
            / "production-contact.json"
        )
        profile = json.loads(profile_path.read_text())
        self.assertEqual(profile["parameters"]["velocity_iterations"], 7)
        self.assertFalse(profile["parameters"]["warm_start"])
        self.assertEqual(profile["family_id"], "temporal-cone")
        self.assertEqual(profile["variant"], "metalworld")
        shown = json.loads(
            self.run_tool("show", "production-contact", "--json").stdout
        )
        self.assertEqual(shown["status"], "current")
        self.assertEqual(
            self.run_tool("validate", str(profile_path)).returncode,
            0,
        )
        refused = self.run_tool(
            "configure",
            "temporal-cone",
            "--profile",
            "production-contact",
            check=False,
        )
        self.assertEqual(refused.returncode, 2)
        self.assertIn("refusing to overwrite", refused.stderr)

    def test_family_defaults_variants_and_targets_resolve_explicitly(self):
        quality = json.loads(
            self.run_tool("inspect", "quality-newton", "--json").stdout
        )
        self.assertEqual(quality["default_variant"], "metalworld")
        self.assertEqual(len(quality["variants"]), 5)
        unified = json.loads(
            self.run_tool(
                "inspect",
                "quality-newton",
                "--variant",
                "unified-metal",
                "--json",
            ).stdout
        )
        self.assertEqual(
            unified["id"], "constraints.unified-quality-newton-metal"
        )

        self.run_tool(
            "configure",
            "quality-newton",
            "--profile",
            "quality-default",
        )
        default_profile = json.loads(
            (
                self.workspace
                / ".numi/profiles/solvers/quality-default.json"
            ).read_text()
        )
        self.assertEqual(
            default_profile["solver_id"], "contact.quality-newton-metal-world"
        )

        self.run_tool(
            "configure",
            "quality-newton",
            "--target",
            "MetalUnifiedQuality API",
            "--profile",
            "quality-unified",
        )
        target_profile = json.loads(
            (
                self.workspace
                / ".numi/profiles/solvers/quality-unified.json"
            ).read_text()
        )
        self.assertEqual(
            target_profile["solver_id"],
            "constraints.unified-quality-newton-metal",
        )

        self.run_tool(
            "configure",
            "symplectic-euler",
            "--variant",
            "free-body",
            "--profile",
            "free-body-integrator",
        )
        variant_profile = json.loads(
            (
                self.workspace
                / ".numi/profiles/solvers/free-body-integrator.json"
            ).read_text()
        )
        self.assertEqual(
            variant_profile["solver_id"],
            "integration.free-body-symplectic-euler",
        )

        typo = self.run_tool("inspect", "tempral-cone", check=False)
        self.assertEqual(typo.returncode, 2)
        self.assertIn("did you mean 'temporal-cone'", typo.stderr)

    def test_invalid_nonfinite_and_unknown_parameters_fail_closed(self):
        nonfinite = self.run_tool(
            "configure",
            "temporal-cone",
            "--profile",
            "bad",
            "--set",
            "velocity_iterations=NaN",
            check=False,
        )
        self.assertEqual(nonfinite.returncode, 2)
        self.assertIn("must have type integer", nonfinite.stderr)
        unknown = self.run_tool(
            "configure",
            "temporal-cone",
            "--profile",
            "bad",
            "--set",
            "secret_command=launch",
            check=False,
        )
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("has no parameter", unknown.stderr)

    def test_external_overlay_is_data_only_and_first_match_wins(self):
        descriptor = {
            "schema": "numi.solver.v1",
            "id": "contact.external-example",
            "family": {
                "id": "external-example",
                "name": "External example",
                "summary": "Test-only external family.",
                "variant": "owner-default",
                "default": True,
            },
            "name": "External example",
            "domain": "contact",
            "summary": "Test-only external descriptor.",
            "backend": "external",
            "precision": "declared-by-owner",
            "role": "experimental",
            "availability": "external",
            "owner": {
                "implementation": "/opt/example/solver",
                "configuration": "ExampleConfig",
                "documentation": "/opt/example/README.md",
            },
            "selection": {
                "kind": "external-adapter-required",
                "selector": "ExampleConfig.algorithm=external",
                "targets": ["Example API"],
            },
            "parameters": {
                "iterations": {"type": "integer", "default": 8, "minimum": 1}
            },
            "evidence_boundary": ["Descriptor validation does not execute code."],
        }
        source = Path(self.temporary.name) / "external.json"
        source.write_text(json.dumps(descriptor))
        self.run_tool("validate", str(source))
        self.run_tool("register", str(source), "--scope", "workspace")
        inspected = json.loads(
            self.run_tool("inspect", descriptor["id"], "--json").stdout
        )
        self.assertEqual(inspected["owner"]["implementation"], "/opt/example/solver")
        self.assertIn(str(self.workspace / ".numi" / "solvers"), inspected["_provenance"]["source"])

        conflicting = json.loads(json.dumps(descriptor))
        conflicting["id"] = "contact.external-quality-conflict"
        conflicting["family"] = {
            "id": "quality-newton",
            "name": "Quality Newton",
            "summary": "Certificate-bearing semismooth Newton quality solves across CPU and Metal targets.",
            "variant": "external-conflict",
            "default": True,
        }
        conflict_source = Path(self.temporary.name) / "conflict.json"
        conflict_source.write_text(json.dumps(conflicting))
        conflict = self.run_tool(
            "register", str(conflict_source), "--scope", "workspace", check=False
        )
        self.assertEqual(conflict.returncode, 2)
        self.assertIn("exactly one default variant", conflict.stderr)
        self.assertFalse(
            (
                self.workspace
                / ".numi/solvers/contact.external-quality-conflict.json"
            ).exists()
        )

    def test_descriptor_drift_and_incomplete_profiles_fail_closed(self):
        self.run_tool(
            "configure",
            "temporal-cone",
            "--profile",
            "drift-check",
        )
        profile_path = (
            self.workspace / ".numi" / "profiles" / "solvers" / "drift-check.json"
        )
        profile = json.loads(profile_path.read_text())
        profile["parameters"].pop("warm_start")
        profile_path.write_text(json.dumps(profile))
        result = self.run_tool("show", "drift-check", check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("parameter set differs", result.stderr)


if __name__ == "__main__":
    unittest.main()
